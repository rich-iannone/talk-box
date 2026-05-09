"""Lightweight REST API server for Talk Box chatbots.

Provides ``serve()`` to expose a chatbot over HTTP with health checks,
CORS support, and optional token-based authentication.

Requires ``starlette`` and ``uvicorn`` (install with ``pip install starlette uvicorn``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from talk_box.builder import ChatBot


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """Configuration for the Talk Box API server.

    Parameters
    ----------
    host
        Hostname to bind to.
    port
        Port to listen on.
    cors_origins
        Allowed CORS origins.  ``["*"]`` allows all.
    auth_token
        If set, requests must include ``Authorization: Bearer <token>``.
    title
        API title shown in docs/health endpoint.
    """

    host: str = "127.0.0.1"
    port: int = 8567
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    auth_token: str | None = None
    title: str = "Talk Box API"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _create_app(
    bot: "ChatBot",
    config: ServerConfig,
) -> Any:
    """Build a Starlette ASGI app for the given bot."""
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    # ------------------------------------------------------------------
    # Auth middleware (optional)
    # ------------------------------------------------------------------
    async def _check_auth(request: Request) -> JSONResponse | None:
        if config.auth_token is None:
            return None
        auth = request.headers.get("authorization", "")
        if auth != f"Bearer {config.auth_token}":
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return None

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    async def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "title": config.title,
                "persona": getattr(bot, "persona_name", None),
            }
        )

    async def chat(request: Request) -> JSONResponse:
        auth_err = await _check_auth(request)
        if auth_err is not None:
            return auth_err

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        message = body.get("message")
        if not message or not isinstance(message, str):
            return JSONResponse({"error": "'message' field required (string)"}, status_code=400)

        try:
            response = bot.chat(message)
            return JSONResponse({"response": response})
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def info(request: Request) -> JSONResponse:
        auth_err = await _check_auth(request)
        if auth_err is not None:
            return auth_err

        return JSONResponse(
            {
                "persona": getattr(bot, "persona_name", None),
                "model": getattr(bot, "model", None),
                "guardrails": [
                    g.name if hasattr(g, "name") else str(g) for g in getattr(bot, "guardrails", [])
                ],
            }
        )

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/chat", chat, methods=["POST"]),
        Route("/info", info, methods=["GET"]),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
    ]

    return Starlette(routes=routes, middleware=middleware)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serve(
    bot: "ChatBot",
    *,
    host: str = "127.0.0.1",
    port: int = 8567,
    cors_origins: list[str] | None = None,
    auth_token: str | None = None,
    title: str = "Talk Box API",
) -> None:
    """Serve a chatbot as a REST API.

    Starts a uvicorn server exposing the bot over HTTP.

    Endpoints
    ---------
    - ``GET /health`` — health check (no auth required)
    - ``POST /chat`` — send ``{"message": "..."}`` and get ``{"response": "..."}``
    - ``GET /info`` — bot metadata (persona, model, guardrails)

    Parameters
    ----------
    bot
        The ``ChatBot`` instance to serve.
    host
        Hostname to bind to (default ``127.0.0.1``).
    port
        Port number (default ``8567``).
    cors_origins
        Allowed CORS origins.  Defaults to ``["*"]``.
    auth_token
        If set, all endpoints (except ``/health``) require
        ``Authorization: Bearer <token>`` header.
    title
        API title shown in ``/health`` response.

    Examples
    --------
    ```python
    import talk_box as tb

    bot = tb.ChatBot("helper", model="ollama:llama3.3")
    tb.serve(bot, port=8080)
    ```
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required to serve. Install with: pip install uvicorn starlette"
        ) from exc

    config = ServerConfig(
        host=host,
        port=port,
        cors_origins=cors_origins or ["*"],
        auth_token=auth_token,
        title=title,
    )
    app = _create_app(bot, config)
    uvicorn.run(app, host=config.host, port=config.port)
