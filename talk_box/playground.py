"""Developer playground for interactive chatbot testing.

Provides ``playground()`` which launches a local dev-mode chat interface
with live diagnostics — prompt inspector, guard log, tool trace, and timing.

Requires ``starlette`` and ``uvicorn``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from talk_box.builder import ChatBot


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_PLAYGROUND_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }}
header {{ background: #16213e; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #0f3460; }}
header h1 {{ font-size: 18px; color: #e94560; }}
header .meta {{ font-size: 12px; color: #888; }}
.container {{ display: flex; flex: 1; overflow: hidden; }}
.chat-panel {{ flex: 2; display: flex; flex-direction: column; }}
.messages {{ flex: 1; overflow-y: auto; padding: 16px; }}
.msg {{ margin-bottom: 12px; padding: 10px 14px; border-radius: 8px; max-width: 80%; white-space: pre-wrap; word-wrap: break-word; }}
.msg.user {{ background: #0f3460; margin-left: auto; }}
.msg.bot {{ background: #16213e; border: 1px solid #0f3460; }}
.msg .timing {{ font-size: 11px; color: #666; margin-top: 4px; }}
.input-row {{ display: flex; padding: 12px; gap: 8px; border-top: 1px solid #0f3460; }}
.input-row input {{ flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #0f3460; background: #16213e; color: #e0e0e0; font-size: 14px; }}
.input-row button {{ padding: 10px 20px; border-radius: 6px; border: none; background: #e94560; color: white; cursor: pointer; font-size: 14px; }}
.input-row button:hover {{ background: #c73e54; }}
.diag-panel {{ flex: 1; border-left: 1px solid #0f3460; display: flex; flex-direction: column; min-width: 300px; }}
.diag-tabs {{ display: flex; background: #16213e; border-bottom: 1px solid #0f3460; }}
.diag-tabs button {{ flex: 1; padding: 8px; border: none; background: transparent; color: #888; cursor: pointer; font-size: 12px; }}
.diag-tabs button.active {{ color: #e94560; border-bottom: 2px solid #e94560; }}
.diag-content {{ flex: 1; overflow-y: auto; padding: 12px; font-family: "SF Mono", "Fira Code", monospace; font-size: 12px; white-space: pre-wrap; word-wrap: break-word; }}
</style>
</head>
<body>
<header>
  <h1>{title}</h1>
  <div class="meta">Persona: {persona} &middot; Model: {model}</div>
</header>
<div class="container">
  <div class="chat-panel">
    <div class="messages" id="messages"></div>
    <div class="input-row">
      <input type="text" id="input" placeholder="Type a message..." autocomplete="off">
      <button onclick="send()">Send</button>
    </div>
  </div>
  <div class="diag-panel">
    <div class="diag-tabs">
      <button class="active" onclick="showTab('prompt')">Prompt</button>
      <button onclick="showTab('guards')">Guards</button>
      <button onclick="showTab('tools')">Tools</button>
      <button onclick="showTab('timing')">Timing</button>
    </div>
    <div class="diag-content" id="diag">Diagnostics will appear here after sending a message.</div>
  </div>
</div>
<script>
const msgs = document.getElementById('messages');
const input = document.getElementById('input');
const diag = document.getElementById('diag');
let lastDiag = {{}};

input.addEventListener('keydown', e => {{ if (e.key === 'Enter') send(); }});

async function send() {{
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg(text, 'user');
  try {{
    const r = await fetch('/playground/chat', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{message: text}})
    }});
    const data = await r.json();
    addMsg(data.response || data.error, r.ok ? 'bot' : 'bot', data.elapsed_ms);
    lastDiag = data.diagnostics || {{}};
    showTab('prompt');
  }} catch(e) {{ addMsg('Error: ' + e.message, 'bot'); }}
}}

function addMsg(text, cls, ms) {{
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  if (ms !== undefined) {{
    const t = document.createElement('div');
    t.className = 'timing';
    t.textContent = ms.toFixed(0) + 'ms';
    d.appendChild(t);
  }}
  msgs.appendChild(d);
  msgs.scrollTop = msgs.scrollHeight;
}}

function showTab(name) {{
  document.querySelectorAll('.diag-tabs button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  diag.textContent = JSON.stringify(lastDiag[name] || lastDiag, null, 2);
}}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _create_playground_app(bot: "ChatBot", title: str) -> Any:
    """Build a Starlette app for the playground."""
    import time

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route

    persona = getattr(bot, "persona_name", "default")
    model = getattr(bot, "model", "unknown")

    async def index(request: Request) -> HTMLResponse:
        html = _PLAYGROUND_HTML.format(title=title, persona=persona, model=model)
        return HTMLResponse(html)

    async def chat_endpoint(request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "Invalid JSON"}, status_code=400)

        message = body.get("message")
        if not message or not isinstance(message, str):
            return JSONResponse({"error": "'message' field required"}, status_code=400)

        start = time.perf_counter()
        try:
            response = bot.chat(message)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Gather diagnostics
        diagnostics: dict[str, Any] = {
            "prompt": {
                "persona": persona,
                "model": model,
                "system_prompt_length": len(getattr(bot, "system_prompt", "") or ""),
                "message": message,
            },
            "guards": {
                "active": [
                    g.name if hasattr(g, "name") else str(g) for g in getattr(bot, "guardrails", [])
                ],
                "count": len(getattr(bot, "guardrails", [])),
            },
            "tools": {
                "available": [
                    t if isinstance(t, str) else getattr(t, "__name__", str(t))
                    for t in getattr(bot, "tools", [])
                ],
                "count": len(getattr(bot, "tools", [])),
            },
            "timing": {
                "elapsed_ms": round(elapsed_ms, 1),
                "response_length": len(response),
            },
        }

        return JSONResponse(
            {
                "response": response,
                "elapsed_ms": elapsed_ms,
                "diagnostics": diagnostics,
            }
        )

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/playground/chat", chat_endpoint, methods=["POST"]),
    ]

    return Starlette(routes=routes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def playground(
    bot: "ChatBot",
    *,
    host: str = "127.0.0.1",
    port: int = 8568,
    title: str = "Talk Box Playground",
) -> None:
    """Launch an interactive dev-mode playground for a chatbot.

    Opens a browser-based chat interface with live diagnostics including
    prompt inspection, guard status, tool traces, and response timing.

    Parameters
    ----------
    bot
        The ``ChatBot`` instance to test interactively.
    host
        Hostname to bind to.
    port
        Port number (default ``8568``).
    title
        Page title displayed in the browser.

    Examples
    --------
    ```python
    import talk_box as tb

    bot = tb.ChatBot("code_reviewer", model="ollama:llama3.3")
    tb.playground(bot)  # opens http://127.0.0.1:8568
    ```
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ImportError(
            "uvicorn is required for playground. Install with: pip install uvicorn starlette"
        ) from exc

    print(f"🎮 Talk Box Playground: http://{host}:{port}/")
    print("   Press Ctrl+C to stop.\n")

    app = _create_playground_app(bot, title)
    uvicorn.run(app, host=host, port=port, log_level="warning")
