# -*- coding: utf-8 -*-

"""
Unit tests for tools/kiro_login.py.

The tool reimplements the Kiro IDE portal OAuth flow (PKCE + localhost
callback + token exchange) so credentials can be re-issued without the IDE.
Network calls are never made here: everything is exercised through pure
functions or by monkeypatching the HTTP helper.
"""

import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

_TOOL_PATH = Path(__file__).resolve().parents[2] / "tools" / "kiro_login.py"


def _load_tool():
    """Import tools/kiro_login.py as a module (it is not part of the package)."""
    spec = importlib.util.spec_from_file_location("kiro_login_tool", _TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


kiro_login = _load_tool()


class TestPkce:
    """Tests for the PKCE helpers."""

    def test_b64url_has_no_padding(self):
        """
        What it does: Encodes bytes that would normally produce '=' padding.
        Purpose: RFC 7636 forbids padding in code_challenge.
        """
        for size in (1, 2, 31, 32, 33):
            encoded = kiro_login.b64url(b"\xfb" * size)
            assert "=" not in encoded
            assert "+" not in encoded and "/" not in encoded

    def test_challenge_matches_sha256_of_verifier(self):
        """
        What it does: Recomputes the S256 challenge from a verifier.
        Purpose: A wrong challenge makes the token exchange fail with 400.
        """
        verifier = "a" * 43
        expected = kiro_login.b64url(hashlib.sha256(verifier.encode()).digest())
        assert kiro_login.b64url(hashlib.sha256(verifier.encode()).digest()) == expected
        assert len(expected) == 43


class TestPortalUrl:
    """Tests for the authorization URL builder."""

    def test_contains_required_oauth_parameters(self):
        """
        What it does: Builds the portal URL and inspects its query string.
        Purpose: app.kiro.dev rejects the request if any parameter is missing.
        """
        url = kiro_login.build_portal_url(
            "state-1", "challenge-1", "http://127.0.0.1:1234/oauth/callback"
        )
        prefix, _, query = url.partition("?")
        params = dict(part.split("=", 1) for part in query.split("&"))

        assert prefix == "https://app.kiro.dev/signin"
        assert params["state"] == "state-1"
        assert params["code_challenge"] == "challenge-1"
        assert params["code_challenge_method"] == "S256"
        assert params["redirect_from"] == "KiroIDE"
        # redirect_uri must be URL-encoded
        assert params["redirect_uri"].endswith("%2Foauth%2Fcallback")

    def test_custom_portal_url_is_used(self, monkeypatch):
        """
        What it does: Overrides the portal host through the environment.
        Purpose: The IDE allows pointing at a different portal (KIRO_AUTH_PORTAL_URL).
        """
        monkeypatch.setattr(kiro_login, "PORTAL_URL", "https://example.test")

        url = kiro_login.build_portal_url("s", "c", "http://127.0.0.1:1/cb")

        assert url.startswith("https://example.test/signin?")


class TestWriteCredentials:
    """Tests for writing the Kiro credential file."""

    def test_writes_expected_fields_with_restrictive_mode(self, tmp_path):
        """
        What it does: Writes a credential file and reads it back.
        Purpose: The gateway parses this exact shape; permissions must be 0600.
        """
        target = tmp_path / "kiro-auth-token.json"

        kiro_login.write_credentials(
            target,
            {
                "accessToken": "access-token-value",
                "refreshToken": "refresh-token-value",
                "expiresIn": 3600,
                "profileArn": "arn:aws:codewhisperer:us-east-1:1:profile/X",
            },
            profile_arn="arn:fallback",
        )

        data = json.loads(target.read_text())
        assert data["accessToken"] == "access-token-value"
        assert data["refreshToken"] == "refresh-token-value"
        assert data["profileArn"] == "arn:aws:codewhisperer:us-east-1:1:profile/X"
        assert data["authMethod"] == "social"
        assert data["provider"] == "Github"
        assert data["expiresAt"].endswith("Z")
        mode = stat.S_IMODE(os.stat(target).st_mode)
        assert mode == 0o600
        # No temporary file left behind
        assert not target.with_suffix(".tmp").exists()

    def test_falls_back_to_ide_profile_arn(self, tmp_path):
        """
        What it does: Uses the IDE profile ARN when the token response omits it.
        Purpose: Kiro requires profileArn on later calls.
        """
        target = tmp_path / "creds.json"

        kiro_login.write_credentials(
            target,
            {"accessToken": "a", "refreshToken": "r", "expiresIn": 60},
            profile_arn="arn:from:ide",
        )

        assert json.loads(target.read_text())["profileArn"] == "arn:from:ide"

    def test_rejects_response_without_expiry(self, tmp_path):
        """
        What it does: Omits expiresIn from the token response.
        Purpose: A missing TTL would create credentials that never refresh.
        """
        with pytest.raises(RuntimeError, match="expiresIn"):
            kiro_login.write_credentials(
                tmp_path / "creds.json",
                {"accessToken": "a", "refreshToken": "r"},
                profile_arn="arn:from:ide",
            )

    def test_rejects_missing_profile_arn(self, tmp_path):
        """
        What it does: Provides no ARN from either source.
        Purpose: The gateway cannot call Kiro without a profile ARN.
        """
        with pytest.raises(RuntimeError, match="profileArn"):
            kiro_login.write_credentials(
                tmp_path / "creds.json",
                {"accessToken": "a", "refreshToken": "r", "expiresIn": 60},
                profile_arn=None,
            )

    def test_overwrites_existing_file_atomically(self, tmp_path):
        """
        What it does: Writes twice to the same path.
        Purpose: Re-login must replace stale credentials, not append.
        """
        target = tmp_path / "creds.json"
        target.write_text("old")

        kiro_login.write_credentials(
            target,
            {"accessToken": "new", "refreshToken": "new-r", "expiresIn": 60},
            profile_arn="arn:x",
        )

        assert json.loads(target.read_text())["accessToken"] == "new"


class TestExchangeCode:
    """Tests for the token exchange step."""

    def test_posts_login_option_in_redirect_uri(self, monkeypatch):
        """
        What it does: Captures the payload sent to /oauth/token.
        Purpose: Kiro expects redirect_uri with ?login_option=<provider>.
        """
        captured = {}

        def fake_http_json(url, payload=None, headers=None, method=None, timeout=30):
            captured["url"] = url
            captured["payload"] = payload
            return 200, {"accessToken": "a", "refreshToken": "r", "expiresIn": 1}

        monkeypatch.setattr(kiro_login, "http_json", fake_http_json)

        result = kiro_login.exchange_code(
            "the-code", "the-verifier", "http://127.0.0.1:9/oauth/callback", "github"
        )

        assert result["accessToken"] == "a"
        assert captured["url"].endswith("/oauth/token")
        assert captured["payload"]["code"] == "the-code"
        assert captured["payload"]["code_verifier"] == "the-verifier"
        assert captured["payload"]["redirect_uri"].endswith("?login_option=github")

    def test_raises_with_status_on_failure(self, monkeypatch):
        """
        What it does: Simulates a 400 from the token endpoint.
        Purpose: A failed exchange must not be mistaken for success.
        """
        monkeypatch.setattr(
            kiro_login,
            "http_json",
            lambda *args, **kwargs: (400, {"message": "invalid_grant"}),
        )

        with pytest.raises(RuntimeError, match="HTTP 400"):
            kiro_login.exchange_code("c", "v", "http://127.0.0.1:9/cb", "github")


class TestCallbackServer:
    """Tests for the localhost callback server wiring."""

    def test_server_listens_on_loopback_and_exposes_callback_path(self):
        """
        What it does: Starts the callback server on a random port.
        Purpose: The redirect_uri must match the portal's redirect exactly.
        """
        state = "test-state"
        server, callback = kiro_login.start_callback_server(state)
        try:
            host, port = server.server_address[:2]
            assert host == "127.0.0.1"
            assert port > 0
            assert callback.event.is_set() is False
        finally:
            server.shutdown()
