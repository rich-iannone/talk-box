"""Tests for talk_box.playground module."""

from __future__ import annotations

import pytest

from talk_box.playground import _create_playground_app, playground


class TestCreatePlaygroundApp:
    def test_creates_app(self):
        starlette = pytest.importorskip("starlette")  # noqa: F841

        class FakeBot:
            persona_name = "tester"
            model = "echo:test"

            def chat(self, msg):
                return f"echo: {msg}"

        app = _create_playground_app(FakeBot(), "Test Playground")
        assert app is not None

    def test_index_returns_html(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            persona_name = "tester"
            model = "echo:test"

            def chat(self, msg):
                return msg

        app = _create_playground_app(FakeBot(), "Test Playground")
        client = TestClient(app)

        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Test Playground" in resp.text
        assert "tester" in resp.text

    def test_chat_returns_diagnostics(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            persona_name = "tester"
            model = "echo:test"
            guardrails = []
            tools = []
            system_prompt = "You are helpful."

            def chat(self, msg):
                return f"reply: {msg}"

        app = _create_playground_app(FakeBot(), "Test")
        client = TestClient(app)

        resp = client.post("/playground/chat", json={"message": "hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "reply: hi"
        assert "elapsed_ms" in data
        assert "diagnostics" in data
        diag = data["diagnostics"]
        assert "prompt" in diag
        assert "guards" in diag
        assert "tools" in diag
        assert "timing" in diag
        assert diag["prompt"]["persona"] == "tester"

    def test_chat_missing_message(self):
        starlette = pytest.importorskip("starlette")
        from starlette.testclient import TestClient

        class FakeBot:
            persona_name = "tester"
            model = "echo:test"

            def chat(self, msg):
                return msg

        app = _create_playground_app(FakeBot(), "Test")
        client = TestClient(app)

        resp = client.post("/playground/chat", json={})
        assert resp.status_code == 400


class TestPlayground:
    def test_playground_import_error(self, monkeypatch):
        """playground() raises ImportError if uvicorn not available."""
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
            playground(FakeBot())
