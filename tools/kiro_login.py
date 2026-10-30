#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kiro 桌面登录（复刻 Kiro IDE 的 portal OAuth 流程）。

Kiro IDE 的 GitHub / Google 登录走的是 app.kiro.dev 的 portal + PKCE：

1. IDE 在本地起一个回调服务器 http://127.0.0.1:<port>/oauth/callback
2. 打开 https://app.kiro.dev/signin?state=..&code_challenge=..&redirect_uri=..&redirect_from=KiroIDE
3. 回调带 ?code=..&state=..&login_option=github
4. POST https://prod.us-east-1.auth.desktop.kiro.dev/oauth/token
   {"code": .., "code_verifier": .., "redirect_uri": "<callback>?login_option=github"}
5. 写入 ~/.aws/sso/cache/kiro-auth-token.json

本脚本做同样的流程，方便在没有 IDE 界面、或 IDE 登录态失效时重新拿到凭据。
浏览器里的账号授权仍需本人完成。

用法：
    python3 tools/kiro_login.py [--output ~/.aws/sso/cache/kiro-auth-token.json]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional, Tuple

PORTAL_URL = os.getenv("KIRO_AUTH_PORTAL_URL", "https://app.kiro.dev")
AUTH_ENDPOINT = os.getenv(
    "KIRO_AUTH_ENDPOINT", "https://prod.us-east-1.auth.desktop.kiro.dev"
)
USER_AGENT = f"KiroIDE-{os.getenv('KIRO_IDE_VERSION', '1.4.62')}"
CALLBACK_PATH = "/oauth/callback"
CALLBACK_TIMEOUT_SECONDS = 600

DEFAULT_OUTPUT = Path.home() / ".aws" / "sso" / "cache" / "kiro-auth-token.json"
IDE_PROFILE = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Kiro"
    / "User"
    / "globalStorage"
    / "kiro.kiroagent"
    / "profile.json"
)


def b64url(data: bytes) -> str:
    """Base64url 编码，去掉 padding（PKCE 要求）。"""
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def load_ide_profile_arn() -> Optional[str]:
    """读取 Kiro IDE 已选中的 profileArn（登录后接口仍需要它）。"""
    try:
        data = json.loads(IDE_PROFILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    arn = data.get("arn")
    return arn if isinstance(arn, str) and arn else None


def http_json(
    url: str,
    payload: Optional[dict] = None,
    headers: Optional[Dict[str, str]] = None,
    method: Optional[str] = None,
    timeout: float = 30,
) -> Tuple[int, dict]:
    """发一个 JSON 请求，返回 (status, body)。"""
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=body,
        method=method or ("POST" if body is not None else "GET"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            **(headers or {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return response.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(raw) if raw else {}
        except ValueError:
            return error.code, {"message": raw}


class _CallbackState:
    """回调结果容器，供主线程等待。"""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.code: Optional[str] = None
        self.error: Optional[str] = None
        self.login_option: Optional[str] = None


class _CallbackHandler(BaseHTTPRequestHandler):
    """只处理 /oauth/callback，其他路径直接 404。"""

    server_version = "kiro-login"
    state: _CallbackState
    expected_state: str

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return

        query = urllib.parse.parse_qs(parsed.query)
        code = (query.get("code") or [None])[0]
        state = (query.get("state") or [None])[0]
        error = (query.get("error") or [None])[0]
        login_option = (query.get("login_option") or ["github"])[0]

        if error:
            self.state.error = error
        elif state != self.expected_state:
            self.state.error = f"state 不匹配（收到 {state!r}）"
        elif not code:
            self.state.error = "回调里没有 code"
        else:
            self.state.code = code
            self.state.login_option = login_option

        ok = self.state.error is None
        message = (
            "登录成功，可以关闭此页面回到终端。"
            if ok
            else f"登录失败：{self.state.error}"
        )
        html = (
            "<!doctype html><meta charset='utf-8'>"
            f"<title>Kiro 登录</title>"
            "<body style='font-family:-apple-system,sans-serif;padding:48px'>"
            f"<h2>{message}</h2></body>"
        ).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)
        self.state.event.set()

    def log_message(self, *args, **kwargs) -> None:  # noqa: D401 - 静音访问日志
        return


def start_callback_server(expected_state: str) -> Tuple[ThreadingHTTPServer, _CallbackState]:
    """在 127.0.0.1 上随机端口起回调服务器。"""
    state = _CallbackState()
    handler = type(
        "_Handler",
        (_CallbackHandler,),
        {"state": state, "expected_state": expected_state},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, state


def build_portal_url(state: str, challenge: str, redirect_uri: str) -> str:
    """构造 app.kiro.dev 的登录地址（与 Kiro IDE 一致）。"""
    params = urllib.parse.urlencode(
        {
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "redirect_uri": redirect_uri,
            "redirect_from": "KiroIDE",
        }
    )
    return f"{PORTAL_URL}/signin?{params}"


def exchange_code(
    code: str, verifier: str, redirect_uri: str, login_option: str
) -> dict:
    """用授权码换 accessToken / refreshToken。"""
    status, body = http_json(
        f"{AUTH_ENDPOINT}/oauth/token",
        payload={
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": f"{redirect_uri}?login_option={login_option}",
        },
    )
    if status != 200:
        raise RuntimeError(f"换取 token 失败: HTTP {status} {body}")
    return body


def write_credentials(path: Path, token: dict, profile_arn: Optional[str]) -> None:
    """按 Kiro IDE 的格式写入凭据文件（权限 0600）。"""
    expires_in = token.get("expiresIn")
    if expires_in is None:
        raise RuntimeError(f"token 响应缺少 expiresIn: {token}")
    expires_at = time.strftime(
        "%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + int(expires_in))
    ) + ".000Z"

    credentials = {
        "accessToken": token["accessToken"],
        "refreshToken": token["refreshToken"],
        "profileArn": token.get("profileArn") or profile_arn or "",
        "expiresAt": expires_at,
        "authMethod": "social",
        "provider": "Github",
    }
    if not credentials["profileArn"]:
        raise RuntimeError(
            "没有拿到 profileArn（token 响应和 Kiro IDE profile.json 都为空）"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(credentials, handle, indent=2)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def machine_fingerprint() -> str:
    """与本机网关一致的设备指纹（用于 IDE 形态的 User-Agent）。"""
    import getpass
    import socket

    raw = f"{socket.gethostname()}-{getpass.getuser()}-kiro-gateway"
    return hashlib.sha256(raw.encode()).hexdigest()


def list_models(access_token: str, profile_arn: str) -> list:
    """登录后立刻验证一次，顺便看看账号能用的模型。"""
    query = urllib.parse.urlencode({"origin": "AI_EDITOR", "profileArn": profile_arn})
    status, body = http_json(
        f"https://codewhisperer.us-east-1.amazonaws.com/ListAvailableModels?{query}",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": (
                "aws-sdk-js/1.0.27 ua/2.1 os/win32#10.0.19044 lang/js "
                "md/nodejs#22.21.1 api/codewhispererstreaming#1.0.27 m/E "
                f"KiroIDE-{os.getenv('KIRO_IDE_VERSION', '1.4.62')}-{machine_fingerprint()}"
            ),
        },
    )
    if status != 200:
        print(f"[!] 模型目录验证失败: HTTP {status} {body}")
        return []
    return [model.get("modelId") for model in body.get("models") or []]


def main() -> int:
    parser = argparse.ArgumentParser(description="重新登录 Kiro 并写入凭据文件")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"凭据输出路径（默认 {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="只打印登录地址，不自动打开浏览器"
    )
    args = parser.parse_args()

    state = str(uuid.uuid4())
    verifier = secrets.token_urlsafe(32)
    challenge = b64url(hashlib.sha256(verifier.encode()).digest())

    server, callback = start_callback_server(state)
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}{CALLBACK_PATH}"
    portal_url = build_portal_url(state, challenge, redirect_uri)

    print("=" * 72)
    print("在浏览器里完成 Kiro 登录（GitHub / Google），等待回调…")
    print(f"回调地址: {redirect_uri}")
    print("\n登录地址:\n" + portal_url + "\n")
    print("=" * 72)

    if not args.no_browser:
        try:
            import subprocess

            subprocess.run(["open", portal_url], check=False)
            print("已尝试用默认浏览器打开上面的地址。")
        except OSError as error:
            print(f"自动打开浏览器失败（{error}），请手动复制上面的地址。")

    if not callback.event.wait(CALLBACK_TIMEOUT_SECONDS):
        server.shutdown()
        print("[x] 等待回调超时（10 分钟），请重跑本脚本。")
        return 1
    server.shutdown()

    if callback.error:
        print(f"[x] {callback.error}")
        return 1

    print("[*] 收到授权码，正在换取 token…")
    token = exchange_code(
        callback.code or "", verifier, redirect_uri, callback.login_option or "github"
    )

    profile_arn = token.get("profileArn") or load_ide_profile_arn()
    write_credentials(args.output, token, profile_arn)
    print(f"[√] 凭据已写入: {args.output}")

    models = list_models(token["accessToken"], profile_arn or "")
    if models:
        claude = [model for model in models if model and model.startswith("claude")]
        print(f"[√] 账号可用模型 {len(models)} 个，其中 Claude {len(claude)} 个")
        print("    " + ", ".join(models))
    print("\n下一步: ./stop.sh && ./start.sh 重启网关。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
