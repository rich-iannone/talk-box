from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from talk_box.capture import CaptureEvent, ConversationCapture, EventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVENT_LABELS: dict[EventType, str] = {
    EventType.PROMPT: "Prompt",
    EventType.RESPONSE: "Response",
    EventType.TOOL_CALL: "Tool Call",
    EventType.TOOL_RESULT: "Tool Result",
    EventType.GUARD_CHECK: "Guard Check",
    EventType.PATHWAY_TRANSITION: "Pathway Transition",
    EventType.ERROR: "Error",
    EventType.METADATA: "Metadata",
}


def _format_timestamp(ts: float) -> str:
    """Format a Unix timestamp as an ISO 8601 string in UTC."""
    if ts <= 0:
        return ""
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _event_to_audit_dict(event: CaptureEvent) -> dict[str, Any]:
    """Convert a CaptureEvent to an audit-friendly dictionary."""
    d: dict[str, Any] = {
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "timestamp": _format_timestamp(event.timestamp),
        "content": event.content,
    }
    if event.model:
        d["model"] = event.model
    if event.role:
        d["role"] = event.role
    if event.duration_ms is not None:
        d["duration_ms"] = event.duration_ms
    if event.metadata:
        d["metadata"] = event.metadata
    return d


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def export_json(
    capture: ConversationCapture,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
    indent: int = 2,
) -> Path:
    """Export a conversation capture as an auditable JSON file.

    The output is a structured JSON document with session metadata, a
    timestamp, and a full event log with ISO 8601 timestamps.

    Parameters
    ----------
    capture
        The conversation capture to export.
    path
        File path to write the JSON export to.
    metadata
        Additional metadata to include in the export header (e.g.,
        `{"reviewer": "Jane Doe", "department": "compliance"}`).
    indent
        JSON indentation level (default `2`).

    Returns
    -------
    Path
        The path the file was written to.

    Examples
    --------
    ```python
    import talk_box as tb

    capture = tb.ConversationCapture(session_id="audit-001")
    capture.record_prompt("Summarize the Q4 report.")
    capture.record_response("Revenue increased 12% year-over-year.", model="openai:gpt-4o")

    tb.export_json(capture, "audit-001.json", metadata={"reviewer": "Jane Doe"})
    ```
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc: dict[str, Any] = {
        "export_format": "talk_box_compliance_v1",
        "exported_at": datetime.now(tz=timezone.utc).isoformat(),
        "session_id": capture.session_id,
        "session_start": _format_timestamp(capture.start_time),
        "session_metadata": capture.metadata,
    }

    if metadata:
        doc["export_metadata"] = metadata

    doc["event_count"] = len(capture)
    doc["events"] = [_event_to_audit_dict(e) for e in capture.events]

    path.write_text(json.dumps(doc, indent=indent, ensure_ascii=False))
    return path


# ---------------------------------------------------------------------------
# HTML export
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Conversation Transcript — {session_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 800px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a;
         line-height: 1.6; }}
  h1 {{ font-size: 1.4rem; border-bottom: 2px solid #333; padding-bottom: 0.5rem; }}
  .meta {{ background: #f5f5f5; padding: 1rem; border-radius: 6px;
           margin-bottom: 1.5rem; font-size: 0.9rem; }}
  .meta dt {{ font-weight: 600; display: inline; }}
  .meta dd {{ display: inline; margin: 0 1rem 0 0; }}
  .event {{ margin-bottom: 1rem; padding: 0.75rem 1rem; border-left: 4px solid #ccc;
            border-radius: 0 6px 6px 0; background: #fafafa; }}
  .event.prompt {{ border-left-color: #2563eb; }}
  .event.response {{ border-left-color: #16a34a; }}
  .event.tool_call {{ border-left-color: #9333ea; }}
  .event.tool_result {{ border-left-color: #7c3aed; }}
  .event.guard_check {{ border-left-color: #eab308; }}
  .event.pathway_transition {{ border-left-color: #06b6d4; }}
  .event.error {{ border-left-color: #dc2626; background: #fef2f2; }}
  .event.metadata {{ border-left-color: #6b7280; }}
  .event-header {{ font-size: 0.8rem; color: #666; margin-bottom: 0.25rem; }}
  .event-label {{ font-weight: 600; text-transform: uppercase; font-size: 0.75rem;
                  letter-spacing: 0.05em; }}
  .event-content {{ white-space: pre-wrap; word-break: break-word; }}
  .footer {{ margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd;
             font-size: 0.8rem; color: #888; }}
</style>
</head>
<body>
<h1>Conversation Transcript</h1>
<div class="meta">
<dl>
  <dt>Session ID:</dt><dd>{session_id}</dd>
  <dt>Started:</dt><dd>{session_start}</dd>
  <dt>Events:</dt><dd>{event_count}</dd>
{extra_meta}
</dl>
</div>
{events_html}
<div class="footer">
  Exported {exported_at} &middot; talk_box compliance export v1
</div>
</body>
</html>"""


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _event_to_html(event: CaptureEvent) -> str:
    """Render a single event as an HTML block."""
    label = _EVENT_LABELS.get(event.event_type, event.event_type.value)
    css_class = event.event_type.value

    parts: list[str] = []
    header_parts: list[str] = []
    if event.timestamp > 0:
        header_parts.append(_format_timestamp(event.timestamp))
    if event.role:
        header_parts.append(f"role={_escape_html(event.role)}")
    if event.model:
        header_parts.append(f"model={_escape_html(event.model)}")
    if event.duration_ms is not None:
        header_parts.append(f"{event.duration_ms:.0f}ms")

    header = " · ".join(header_parts)

    parts.append(f'<div class="event {css_class}">')
    parts.append(
        f'  <div class="event-header"><span class="event-label">{_escape_html(label)}</span>'
    )
    if header:
        parts.append(f"  — {header}")
    parts.append("  </div>")
    if event.content:
        parts.append(f'  <div class="event-content">{_escape_html(event.content)}</div>')
    parts.append("</div>")
    return "\n".join(parts)


def export_html(
    capture: ConversationCapture,
    path: str | Path,
    *,
    title: str = "",
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Export a conversation capture as a human-readable HTML transcript.

    The output is a self-contained HTML file with styled event blocks,
    suitable for review, printing, or archival.

    Parameters
    ----------
    capture
        The conversation capture to export.
    path
        File path to write the HTML export to.
    title
        Optional title override for the HTML page.
    metadata
        Additional metadata to display in the header (e.g.,
        `{"reviewer": "Jane Doe"}`).

    Returns
    -------
    Path
        The path the file was written to.

    Examples
    --------
    ```python
    import talk_box as tb

    capture = tb.ConversationCapture(session_id="audit-001")
    capture.record_prompt("Summarize the Q4 report.")
    capture.record_response("Revenue increased 12%.", model="openai:gpt-4o")

    tb.export_html(capture, "transcript.html", metadata={"reviewer": "Jane Doe"})
    ```
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build extra metadata lines
    extra_lines: list[str] = []
    all_meta = dict(capture.metadata)
    if metadata:
        all_meta.update(metadata)
    for key, value in all_meta.items():
        extra_lines.append(
            f"  <dt>{_escape_html(str(key))}:</dt><dd>{_escape_html(str(value))}</dd>"
        )
    extra_meta = "\n".join(extra_lines)

    # Build event HTML
    events_html = "\n".join(_event_to_html(e) for e in capture.events)

    session_id = _escape_html(capture.session_id)
    if title:
        session_id = _escape_html(title)

    html = _HTML_TEMPLATE.format(
        session_id=session_id,
        session_start=_format_timestamp(capture.start_time),
        event_count=len(capture),
        extra_meta=extra_meta,
        events_html=events_html,
        exported_at=datetime.now(tz=timezone.utc).isoformat(),
    )

    path.write_text(html)
    return path
