"""Tests for talk_box.server module."""

from __future__ import annotations

import pytest

from talk_box.server import ServerConfig, _create_app, serve


class TestServerConfig:
    def test_defaults(self):
        cfg = ServerConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8567
        assert cfg.cors_origins == ["*"]
        assert cfg.auth_token is None

    def test_custom_values(self):
        cfg = ServerConfig(host="0.0.0.0", port=9000, auth_token="secret")
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 9000
        assert cfg.auth_token == "secret"


class TestCreateApp:
    def test_creates_starlette_app(self):
        starlette = pytest.importorskip("starlette")  # noqa: F841

        class FakeBot:
            persona_name = "test"
            model = "echo:test"

            def chat(self, msg):
                return f"echo: {msg}"

        cfg = ServerConfig()
        app = _create_app(FakeBot(), cfg)
        assert app is not None

    def test_health_endpoint(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            persona_name = "tester"
            model = "echo:test"

            def chat(self, msg):
                return msg

        cfg = ServerConfig(title="Test API")
        app = _create_app(FakeBot(), cfg)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["title"] == "Test API"
        assert data["persona"] == "tester"

    def test_chat_endpoint(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            persona_name = "tester"
            model = "echo:test"

            def chat(self, msg):
                return f"reply: {msg}"

        cfg = ServerConfig()
        app = _create_app(FakeBot(), cfg)
        client = TestClient(app)

        resp = client.post("/chat", json={"message": "hello"})
        assert resp.status_code == 200
        assert resp.json()["response"] == "reply: hello"

    def test_chat_missing_message(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            def chat(self, msg):
                return msg

        cfg = ServerConfig()
        app = _create_app(FakeBot(), cfg)
        client = TestClient(app)

        resp = client.post("/chat", json={"text": "hi"})
        assert resp.status_code == 400
        assert "message" in resp.json()["error"]

    def test_auth_token_required(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            def chat(self, msg):
                return msg

        cfg = ServerConfig(auth_token="my-secret")
        app = _create_app(FakeBot(), cfg)
        client = TestClient(app)

        # No token
        resp = client.post("/chat", json={"message": "hi"})
        assert resp.status_code == 401

        # Wrong token
        resp = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

        # Correct token
        resp = client.post(
            "/chat",
            json={"message": "hi"},
            headers={"Authorization": "Bearer my-secret"},
        )
        assert resp.status_code == 200

    def test_health_no_auth_required(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            def chat(self, msg):
                return msg

        cfg = ServerConfig(auth_token="secret")
        app = _create_app(FakeBot(), cfg)
        client = TestClient(app)

        resp = client.get("/health")
        assert resp.status_code == 200

    def test_info_endpoint(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            persona_name = "helper"
            model = "test:model"
            guardrails = []

            def chat(self, msg):
                return msg

        cfg = ServerConfig()
        app = _create_app(FakeBot(), cfg)
        client = TestClient(app)

        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["persona"] == "helper"
        assert data["model"] == "test:model"


class TestServe:
    def test_serve_import_error(self, monkeypatch):
        """serve() raises ImportError if uvicorn not available."""
        import builtins

        real_import = builtins.__import__

        def _block_uvicorn(name, *args, **kwargs):
            if name == "uvicorn":
                raise ImportError("no uvicorn")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_uvicorn)

        class FakeBot:
            def chat(self, msg):
                return msg

        with pytest.raises(ImportError, match="uvicorn"):
            serve(FakeBot())
