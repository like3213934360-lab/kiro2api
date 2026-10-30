# Kiro Gateway（中文说明）

把 **Kiro（Amazon Q Developer）** 的接口转成 **OpenAI / Anthropic 兼容**接口，让 Claude Code、Codex CLI、CC Switch、Open WebUI 这类客户端直接连本机网关用。

- 本地地址：`http://127.0.0.1:8000`
- 管理后台：`http://127.0.0.1:8000/admin`
- 许可证：AGPL-3.0（个人自用、研究为主，请遵守 Kiro 服务条款）

> 这是 `jwadow/kiro-gateway` 的分支。上游的基础功能不再重复抄一遍，下面只讲**我们改了什么、怎么用、出问题怎么查**。

---

## 一、我们改了什么

### 1. 模型目录
- 模型列表改从**控制面主机**（`codewhisperer.<region>.amazonaws.com`）拉取，因为 `runtime.<region>.kiro.dev` 上的 `/ListAvailableModels` 已经 404。
- 拉取失败或返回空时**回落到内置目录**，`/v1/models` 不会变空；刷新失败时保留旧缓存。
- **别名翻译**：列表里显示的 `auto-kiro` 在请求时真的会转成 `auto`（以前只显示、请求会报错）。
- 支持 router 风格的 `-thinking` 后缀。

### 2. 思考级别（effort）
- 原生 effort：`low / medium / high / xhigh / max`。
- 三种入口都能用：模型名后缀（`claude-opus-5.5-high`）、`output_config.effort`、`thinking.budget_tokens`。
- 每个模型有默认级别，后台可以直接改，也可以写 `MODEL_EFFORT_DEFAULTS` 环境变量。

### 3. 客户端兼容
- **Claude Code**：接受 `messages` 里 `role: "system"`；接受 `web_search` 等服务端工具内容块；修掉一批 422。
- **Codex CLI**：原生 `/v1/responses`（流式、工具调用、reasoning）；忽略 `rs_` 开头的 reasoning item，`store:false` 多轮不报错。

### 4. 省钱
- **会话缓存亲和**：同一会话复用固定的 `conversationId`，命中 Kiro 服务端前缀缓存。实测同一段对话三跳的 credit：`0.594 → 0.536 → 0.072`（以前每跳都是 0.59）。
- 自动裁剪旧工具结果，首 token 超时按上下文自适应。
- 三个 API 都返回 `usage.credits_used`，日志里打印 `[Credits] <模型>: used=<n>`。

### 5. 多账号与额度
- 账号池 + 熔断 + 失败自动切换。
- `GET /v1/credits` 查额度；后台可开关 paid overage。

### 6. 管理后台 `/admin`
- 外部 API Key：单独的 `sk-gw-...` 密钥，可设 token 配额、额度阈值、过期时间。
- 请求日志：耗时、模型、token、credit，可展开看细节。
- 模型上下文长度、默认思考级别可改。
- 账号总览、缓存清理、登录状态记住。

### 7. 稳定性与运维
- **401 日志节流**：同一来源、同一路径 60 秒内只记一条，带客户端地址、失败原因（没带 key / key 不对）、key 指纹（`len=..., sha256=前8位`），**不打印 key**；重复次数合并进下一条。
- `start.sh` 改成轮询 `/health`（最多等 60 秒），不会再出现「服务其实起来了却报启动失败」。
- `stop.sh`、`install-autostart.sh`（macOS launchd 开机自启）。
- 调试日志按请求隔离，`DEBUG_MODE=errors` 只存失败的请求。

---

## 二、快速开始（macOS）

```bash
cp .env.example .env
# 至少改这两项：
#   PROXY_API_KEY=自己编一个密码
#   KIRO_CREDS_FILE=~/.aws/sso/cache/kiro-auth-token.json
./start.sh
```

启动成功会看到：

```
已启动，PID=xxxx
地址：http://127.0.0.1:8000   管理后台：http://127.0.0.1:8000/admin
```

手动启动（不用脚本）：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --host 127.0.0.1 --port 8000
```

Docker：`docker compose up -d`（配置见 `docker-compose.yml`）。

### 凭据失效后重新登录

```bash
.venv/bin/python tools/kiro_login.py
```

它会打开 Kiro 的登录页，浏览器里授权完成后自动写回 `~/.aws/sso/cache/kiro-auth-token.json`，再重启网关就行。不需要 IDE 界面；浏览器里如果已经有该账号的登录态，点一下授权即可，否则仍需输入该账号的密码。

---

## 三、配置（`.env`）

| 变量 | 必填 | 说明 |
|---|---|---|
| `PROXY_API_KEY` | 是 | 网关自己的密码，客户端用它当 api key |
| `KIRO_CREDS_FILE` | 凭据三选一 | Kiro IDE 的 `kiro-auth-token.json` 路径 |
| `REFRESH_TOKEN` | 凭据三选一 | 直接给 refresh token |
| `KIRO_CLI_DB_FILE` | 凭据三选一 | kiro-cli 的 `data.sqlite3` |
| `PROFILE_ARN` | 否 | 企业账号才需要，IDE 会自动带上 |
| `KIRO_REGION` | 否 | 默认 `us-east-1` |
| `SERVER_HOST` / `SERVER_PORT` | 否 | 默认 `0.0.0.0:8000`（脚本用 `127.0.0.1`） |
| `ADMIN_PASSWORD` | 否 | 设置后开启 `/admin` |
| `VPN_PROXY_URL` | 否 | HTTP/SOCKS5 代理，如 `http://127.0.0.1:7890` |
| `CONVERSATION_CACHE_AFFINITY` | 否 | 默认 `true`，关掉就恢复「每跳全新 conversationId」 |
| `DEBUG_MODE` | 否 | `off` / `errors` / `all`，日志落在 `debug_logs/` |
| `LOG_LEVEL` | 否 | 默认 `INFO` |

---

## 四、接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` `/health` | 健康检查 |
| GET | `/v1/models` | 模型列表 |
| POST | `/v1/chat/completions` | OpenAI Chat Completions |
| POST | `/v1/messages` | Anthropic Messages |
| POST | `/v1/responses` | OpenAI Responses（Codex CLI） |
| GET | `/v1/credits` | 查额度 |
| GET | `/admin` | 管理后台 |

```bash
# OpenAI 风格
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4.5","messages":[{"role":"user","content":"hi"}]}'

# Anthropic 风格
curl http://127.0.0.1:8000/v1/messages \
  -H "x-api-key: $PROXY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4.5","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
```

客户端填法：base_url 填 `http://127.0.0.1:8000`（Anthropic 客户端填 `http://127.0.0.1:8000` 或 `.../v1`，看客户端要求，**末尾不要加斜杠**），key 填 `PROXY_API_KEY` 或后台生成的 `sk-gw-...`。

---

## 五、常见问题

**1. 日志一直刷 `Access attempt with invalid API key`**
客户端没带 key 或 key 不对。现在每 60 秒每个来源只记一条，并带指纹，方便对照是哪个 key 发出去的。健康探针这类不带 key 的请求不会再把日志刷满。

**2. `/v1/models` 只有 9~10 个、没有 Claude**
网关只是把 Kiro 返回的目录原样转出来。同一个账号在 Kiro IDE 里有没有 Claude，决定了这里有没有。没有就是账号侧（订阅/灰度）的问题，不是网关过滤。（用 IDE 的 `User-Agent` 请求时才返回「账号有权限的那份目录」，不带 IDE 标识会返回全量目录，两者不要混淆。）

**3. 突然全部 401，/v1/credentials 刷新失败**
`refreshToken` 是**一次性轮换**的：谁刷新，旧的立刻作废。Kiro IDE、网关、账号管理 App 各存一份就会互相顶掉，最后服务端判定「token 被复用」，整条链全部吊销。
处理：重新登录一次，然后**立刻重启网关**；日常只让一个程序持有凭据（建议让 Kiro IDE 持有，网关只读同一个文件）。

**4. 重新登录了，网关还是不认**
网关只在**启动时**读一次 `kiro-auth-token.json`，必须重启：`./stop.sh && ./start.sh`。目前没有做「刷新失败自动重读文件」。

**5. 换端口**
`./start.sh` 固定 8000；要换端口用 `python main.py --port 9000`，同时客户端地址也要改。

**6. 日志在哪**
`server.log`（脚本启动时）或终端输出；调试明细在 `debug_logs/`。

---

## 六、目录结构（主要部分）

```
main.py                 # 入口，FastAPI 应用
kiro/
  routes_openai.py      # OpenAI 接口
  routes_anthropic.py   # Anthropic 接口
  routes_responses.py   # Responses API
  routes_admin.py       # 管理后台接口
  converters_*.py       # 各 API → Kiro 请求
  streaming_*.py        # Kiro SSE → 各 API SSE
  auth.py               # 凭据与刷新
  account_manager.py    # 多账号 / 熔断 / 模型目录
  model_resolver.py     # 模型名归一化、别名
  effort.py             # 思考级别
  auth_failures.py      # 401 日志节流
  static/admin.html     # 后台前端
tests/unit, tests/integration
start.sh / stop.sh / install-autostart.sh
```

---

## 七、说明

- 本项目只做协议转换，不提供账号、不绕过订阅；能用哪些模型由 Kiro 账号本身决定。
- 底层用的是未公开的 Kiro 接口，Kiro 侧随时可能改，遇到问题先看 `server.log`。
- 欢迎提 issue，但请先附上 `DEBUG_MODE=errors` 生成的 `debug_logs/`（含 token 的请求请不要直接贴出来）。
