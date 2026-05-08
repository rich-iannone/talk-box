import json

from talk_box.capture import ConversationCapture
from talk_box.compliance import export_html, export_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_capture(
    session_id: str = "test-session",
    *,
    system_prompt: str = "",
    prompts_responses: list[tuple[str, str]] | None = None,
    model: str = "test-model",
):
    cap = ConversationCapture(session_id=session_id)
    if system_prompt:
        cap.record_prompt(system_prompt, role="system")
    for prompt, response in prompts_responses or []:
        cap.record_prompt(prompt)
        cap.record_response(response, model=model)
    return cap


# ---------------------------------------------------------------------------
# export_json — basic
# ---------------------------------------------------------------------------


class TestExportJsonBasic:
    def test_creates_file(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Hello", "Hi")])
        path = export_json(cap, tmp_path / "export.json")
        assert path.exists()

    def test_returns_path(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Hello", "Hi")])
        path = export_json(cap, tmp_path / "export.json")
        assert path == tmp_path / "export.json"

    def test_valid_json(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Hello", "Hi")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert isinstance(data, dict)

    def test_format_marker(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Hello", "Hi")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["export_format"] == "talk_box_compliance_v1"

    def test_session_id(self, tmp_path):
        cap = _make_capture("my-session", prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["session_id"] == "my-session"

    def test_exported_at(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert "exported_at" in data
        assert len(data["exported_at"]) > 0

    def test_session_start(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert "session_start" in data
        assert len(data["session_start"]) > 0

    def test_event_count(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q1", "A1"), ("Q2", "A2")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["event_count"] == 4  # 2 prompts + 2 responses

    def test_events_present(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert len(data["events"]) == 2
        assert data["events"][0]["event_type"] == "prompt"
        assert data["events"][0]["content"] == "Q"
        assert data["events"][1]["event_type"] == "response"
        assert data["events"][1]["content"] == "A"


# ---------------------------------------------------------------------------
# export_json — metadata
# ---------------------------------------------------------------------------


class TestExportJsonMetadata:
    def test_session_metadata(self, tmp_path):
        cap = ConversationCapture(session_id="s1", metadata={"model": "gpt-4o"})
        cap.record_prompt("Hello")
        cap.record_response("Hi")
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["session_metadata"] == {"model": "gpt-4o"}

    def test_export_metadata(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json", metadata={"reviewer": "Jane"})
        data = json.loads(path.read_text())
        assert data["export_metadata"]["reviewer"] == "Jane"

    def test_no_export_metadata(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert "export_metadata" not in data


# ---------------------------------------------------------------------------
# export_json — event fields
# ---------------------------------------------------------------------------


class TestExportJsonEvents:
    def test_event_id_present(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["events"][0]["event_id"] != ""

    def test_iso_timestamp(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        ts = data["events"][0]["timestamp"]
        assert "T" in ts  # ISO 8601 format

    def test_model_included(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")], model="gpt-4o")
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        response_event = data["events"][1]
        assert response_event["model"] == "gpt-4o"

    def test_model_omitted_when_empty(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        prompt_event = data["events"][0]
        assert "model" not in prompt_event

    def test_role_included(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["events"][0]["role"] == "user"
        assert data["events"][1]["role"] == "assistant"

    def test_duration_included(self, tmp_path):
        cap = ConversationCapture(session_id="s1")
        cap.record_prompt("Q")
        cap.record_response("A", model="m1", duration_ms=150.5)
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["events"][1]["duration_ms"] == 150.5

    def test_duration_omitted_when_none(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert "duration_ms" not in data["events"][0]

    def test_event_metadata_included(self, tmp_path):
        cap = ConversationCapture(session_id="s1")
        cap.record_tool_call("search", arguments={"q": "test"})
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["events"][0]["metadata"]["tool_name"] == "search"

    def test_event_metadata_omitted_when_empty(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert "metadata" not in data["events"][0]


# ---------------------------------------------------------------------------
# export_json — edge cases
# ---------------------------------------------------------------------------


class TestExportJsonEdgeCases:
    def test_creates_directories(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "sub" / "dir" / "export.json")
        assert path.exists()

    def test_empty_capture(self, tmp_path):
        cap = ConversationCapture(session_id="empty")
        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        assert data["event_count"] == 0
        assert data["events"] == []

    def test_all_event_types(self, tmp_path):
        cap = ConversationCapture(session_id="full")
        cap.record_prompt("You are helpful", role="system")
        cap.record_prompt("Hello")
        cap.record_response("Hi", model="m1")
        cap.record_tool_call("search", arguments={"q": "test"})
        cap.record_tool_result("search", "Found it", duration_ms=50)
        cap.record_guard_check("no_pii", True)
        cap.record_pathway_transition("a", "b", trigger="done")
        cap.record_error("oops", error_type="RuntimeError")
        cap.record(cap.events[0].event_type, "meta")  # metadata via generic

        path = export_json(cap, tmp_path / "export.json")
        data = json.loads(path.read_text())
        types = [e["event_type"] for e in data["events"]]
        assert "prompt" in types
        assert "response" in types
        assert "tool_call" in types
        assert "tool_result" in types
        assert "guard_check" in types
        assert "pathway_transition" in types
        assert "error" in types

    def test_custom_indent(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_json(cap, tmp_path / "export.json", indent=4)
        text = path.read_text()
        assert "    " in text  # 4-space indent


# ---------------------------------------------------------------------------
# export_html — basic
# ---------------------------------------------------------------------------


class TestExportHtmlBasic:
    def test_creates_file(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Hello", "Hi")])
        path = export_html(cap, tmp_path / "export.html")
        assert path.exists()

    def test_returns_path(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Hello", "Hi")])
        path = export_html(cap, tmp_path / "export.html")
        assert path == tmp_path / "export.html"

    def test_valid_html(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Hello", "Hi")])
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert text.startswith("<!DOCTYPE html>")
        assert "</html>" in text

    def test_session_id_in_html(self, tmp_path):
        cap = _make_capture("my-session", prompts_responses=[("Q", "A")])
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert "my-session" in text

    def test_content_in_html(self, tmp_path):
        cap = _make_capture(prompts_responses=[("What is Python?", "A programming language.")])
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert "What is Python?" in text
        assert "A programming language." in text

    def test_event_type_classes(self, tmp_path):
        cap = ConversationCapture(session_id="s1")
        cap.record_prompt("Q")
        cap.record_response("A", model="m1")
        cap.record_error("oops")
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert 'class="event prompt"' in text
        assert 'class="event response"' in text
        assert 'class="event error"' in text


# ---------------------------------------------------------------------------
# export_html — metadata and title
# ---------------------------------------------------------------------------


class TestExportHtmlMetadata:
    def test_custom_title(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_html(cap, tmp_path / "export.html", title="Audit Report")
        text = path.read_text()
        assert "Audit Report" in text

    def test_session_metadata_shown(self, tmp_path):
        cap = ConversationCapture(session_id="s1", metadata={"model": "gpt-4o"})
        cap.record_prompt("Q")
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert "gpt-4o" in text

    def test_export_metadata_shown(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_html(cap, tmp_path / "export.html", metadata={"reviewer": "Jane Doe"})
        text = path.read_text()
        assert "Jane Doe" in text


# ---------------------------------------------------------------------------
# export_html — HTML escaping
# ---------------------------------------------------------------------------


class TestExportHtmlEscaping:
    def test_html_in_content_escaped(self, tmp_path):
        cap = ConversationCapture(session_id="s1")
        cap.record_prompt("<script>alert('xss')</script>")
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert "<script>" not in text
        assert "&lt;script&gt;" in text

    def test_html_in_session_id_escaped(self, tmp_path):
        cap = ConversationCapture(session_id='<img src="x">')
        cap.record_prompt("Q")
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert '<img src="x">' not in text

    def test_html_in_metadata_escaped(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_html(cap, tmp_path / "export.html", metadata={"note": "<b>bold</b>"})
        text = path.read_text()
        assert "<b>bold</b>" not in text
        assert "&lt;b&gt;" in text


# ---------------------------------------------------------------------------
# export_html — edge cases
# ---------------------------------------------------------------------------


class TestExportHtmlEdgeCases:
    def test_creates_directories(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_html(cap, tmp_path / "sub" / "dir" / "export.html")
        assert path.exists()

    def test_empty_capture(self, tmp_path):
        cap = ConversationCapture(session_id="empty")
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert "<!DOCTYPE html>" in text
        assert "empty" in text

    def test_model_shown(self, tmp_path):
        cap = ConversationCapture(session_id="s1")
        cap.record_response("Answer", model="openai:gpt-4o", duration_ms=200)
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert "openai:gpt-4o" in text
        assert "200ms" in text

    def test_footer_present(self, tmp_path):
        cap = _make_capture(prompts_responses=[("Q", "A")])
        path = export_html(cap, tmp_path / "export.html")
        text = path.read_text()
        assert "compliance export v1" in text
