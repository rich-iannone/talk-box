import pytest
from datetime import datetime
import json

import talk_box as tb
from talk_box.tools import ToolContext, ToolResult


class TestBuiltinToolsExtended:
    """Extended tests for builtin tools."""

    def test_text_stats_comprehensive(self):
        """Test text_stats with comprehensive text."""
        from talk_box.builtin_tools import text_stats

        context = ToolContext()

        # Test with complex text
        text = """This is a sample text.

It has multiple paragraphs! And various punctuation?
Numbers like 123 and symbols like @#$%.

Final paragraph here."""

        result = text_stats(context, text)
        assert result.success

        # Data is returned as a dict with stats
        data = result.data
        assert isinstance(data, dict)
        assert "words" in data
        assert "characters" in data
        assert "lines" in data
        assert "sentences" in data
        assert "paragraphs" in data

        assert data["words"] > 15
        assert data["lines"] > 5
        assert data["sentences"] >= 2
        assert data["paragraphs"] >= 2

    def test_convert_case_comprehensive(self):
        """Test convert_case with different case types."""
        from talk_box.builtin_tools import convert_case

        context = ToolContext()

        # Test camel case conversion - returns string directly
        result = convert_case(context, "hello_world_example", "camel")
        assert result.success
        assert result.data == "helloWorldExample"

        # Test snake case
        result = convert_case(context, "Hello World Example", "snake")
        assert result.success
        assert result.data == "hello_world_example"

        # Test invalid case type
        result = convert_case(context, "test", "invalid")
        assert not result.success
        assert result.error is not None

    def test_calculate_comprehensive(self):
        """Test calculate tool with various expressions."""
        from talk_box.builtin_tools import calculate

        context = ToolContext()

        # Test basic operations - returns number directly
        result = calculate(context, "2 + 3 * 4")
        assert result.success
        assert result.data == 14

        # Test math functions
        result = calculate(context, "sqrt(16)")
        assert result.success
        assert result.data == 4.0

        # Test with dangerous expression
        result = calculate(context, "import os")
        assert not result.success
        assert "dangerous" in result.error.lower()

    def test_number_sequence_comprehensive(self):
        """Test number_sequence tool."""
        from talk_box.builtin_tools import number_sequence

        context = ToolContext()

        # Test simple sequence - returns list directly
        result = number_sequence(context, 1, 5, 1)
        assert result.success
        assert isinstance(result.data, list)
        assert result.data == [1, 2, 3, 4]  # stops before 5

        # Test zero step (should fail)
        result = number_sequence(context, 1, 5, 0)
        assert not result.success
        assert "zero" in result.error.lower()

    def test_current_time_comprehensive(self):
        """Test current_time tool."""
        from talk_box.builtin_tools import current_time

        context = ToolContext()

        # Test default format - returns dict with time info
        result = current_time(context)
        assert result.success
        assert isinstance(result.data, dict)

        expected_keys = ["iso", "unix", "readable", "date", "time", "weekday", "month", "year"]
        for key in expected_keys:
            assert key in result.data

    def test_parse_json_comprehensive(self):
        """Test parse_json tool."""
        from talk_box.builtin_tools import parse_json

        context = ToolContext()

        # Test valid JSON - returns parsed data directly
        json_str = '{"name": "Alice", "age": 30, "items": [1, 2, 3]}'
        result = parse_json(context, json_str)
        assert result.success
        assert isinstance(result.data, dict)
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 30
        assert result.data["items"] == [1, 2, 3]

        # Test invalid JSON
        result = parse_json(context, "not valid json {")
        assert not result.success
        assert "invalid" in result.error.lower()

    def test_to_json_comprehensive(self):
        """Test to_json tool."""
        from talk_box.builtin_tools import to_json

        context = ToolContext()

        # Test with dict - returns JSON string directly
        data = {"name": "Bob", "values": [1, 2, 3]}
        result = to_json(context, data)
        assert result.success
        assert isinstance(result.data, str)

        # Parse back to verify
        parsed = json.loads(result.data)
        assert parsed["name"] == "Bob"
        assert parsed["values"] == [1, 2, 3]

    def test_validate_email_comprehensive(self):
        """Test validate_email tool."""
        from talk_box.builtin_tools import validate_email

        context = ToolContext()

        # Test valid email - returns dict with validation info
        result = validate_email(context, "user@example.com")
        assert result.success
        assert isinstance(result.data, dict)
        assert result.data["valid"] is True
        assert result.data["email"] == "user@example.com"
        assert "domain" in result.data

        # Test invalid email
        result = validate_email(context, "not-an-email")
        assert result.success
        assert isinstance(result.data, dict)
        assert result.data["valid"] is False
        assert "error" in result.data

    def test_generate_uuid_comprehensive(self):
        """Test generate_uuid tool."""
        from talk_box.builtin_tools import generate_uuid

        context = ToolContext()

        # Test default UUID - returns string directly
        result = generate_uuid(context)
        assert result.success
        assert isinstance(result.data, str)
        assert len(result.data) == 36  # Standard UUID format

        # Test short format
        result = generate_uuid(context, format_type="short")
        assert result.success
        assert isinstance(result.data, str)
        assert len(result.data) == 32  # No dashes


class TestBuiltinToolsErrorHandling:
    """Test error handling in builtin tools."""

    def test_calculate_edge_cases(self):
        """Test calculate tool with edge cases."""
        from talk_box.builtin_tools import calculate

        context = ToolContext()

        # Test division by zero - should handle gracefully
        result = calculate(context, "1 / 0")
        assert not result.success
        assert result.error is not None


class TestBuiltinToolsIntegration:
    """Test integration aspects of builtin tools."""

    def test_builtin_tools_registration(self):
        """Test that builtin tools can be loaded."""
        from talk_box.builtin_tools import load_tool_box
        from talk_box.tools import get_global_registry, clear_global_registry

        clear_global_registry()
        initial_count = len(get_global_registry().get_all_tools())

        # Load builtin tools
        load_tool_box()

        registry = get_global_registry()
        final_count = len(registry.get_all_tools())

        # Should have more tools after loading
        assert final_count > initial_count

        # Check for some expected tools
        tool_names = [tool.name for tool in registry.get_all_tools()]
        expected_tools = ["text_stats", "convert_case", "calculate", "current_time"]

        for expected in expected_tools:
            assert expected in tool_names

    def test_tool_metadata_consistency(self):
        """Test that all builtin tools have consistent metadata."""
        from talk_box.builtin_tools import load_tool_box
        from talk_box.tools import get_global_registry, clear_global_registry

        # Clear registry and load builtin tools
        clear_global_registry()
        load_tool_box()

        registry = get_global_registry()

        # Verify all tools have required attributes
        for tool in registry.get_all_tools():
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert hasattr(tool, "func")
            assert tool.name is not None
            assert tool.description is not None
            assert len(tool.description) > 10  # Should have meaningful description


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

    def test_generate_uuid_tool(self):
        """Test generate_uuid builtin tool."""
        from talk_box.builtin_tools import generate_uuid

        context = ToolContext()

        # Test default UUID - returns string directly
        result = generate_uuid(context)
        assert result.success
        assert isinstance(result.data, str)
        assert len(result.data) == 36  # Standard UUID format

        # Test short format
        result = generate_uuid(context, format_type="short")
        assert result.success
        assert isinstance(result.data, str)
        assert len(result.data) == 32  # No dashes

    def test_parse_json_tool(self):
        """Test parse_json builtin tool."""
        from talk_box.builtin_tools import parse_json

        context = ToolContext()

        # Test valid JSON - returns parsed data directly
        json_str = '{"name": "Alice", "age": 30, "active": true}'
        result = parse_json(context, json_str)
        assert result.success
        assert isinstance(result.data, dict)
        assert result.data["name"] == "Alice"
        assert result.data["age"] == 30
        assert result.data["active"] is True

        # Test invalid JSON
        result = parse_json(context, "not valid json {")
        assert not result.success
        assert result.error is not None

    def test_to_json_tool(self):
        """Test to_json builtin tool."""
        from talk_box.builtin_tools import to_json

        context = ToolContext()

        # Test with dict - returns JSON string directly
        data = {"name": "Bob", "values": [1, 2, 3]}
        result = to_json(context, data)
        assert result.success
        assert isinstance(result.data, str)

        # Parse back to verify
        parsed = json.loads(result.data)
        assert parsed["name"] == "Bob"
        assert parsed["values"] == [1, 2, 3]

    def test_parse_url_tool(self):
        """Test parse_url builtin tool."""
        from talk_box.builtin_tools import parse_url

        context = ToolContext()

        # Test valid URL
        url = "https://example.com:8080/path/to/page?param=value#section"
        result = parse_url(context, url)
        assert result.success
        assert result.data["scheme"] == "https"
        assert result.data["netloc"] == "example.com:8080"
        assert result.data["path"] == "/path/to/page"
        assert result.data["query"] == "param=value"
        assert result.data["fragment"] == "section"

        # Test malformed URL
        result = parse_url(context, "not a url")
        assert result.success  # urlparse is very lenient

    def test_url_encode_decode_tool(self):
        """Test url_encode_decode builtin tool."""
        from talk_box.builtin_tools import url_encode_decode

        context = ToolContext()

        # Test encoding
        text = "hello world & special chars"
        result = url_encode_decode(context, text, "encode")
        assert result.success
        assert "hello%20world" in result.data["result"]

        # Test decoding
        encoded = "hello%20world%20%26%20special%20chars"
        result = url_encode_decode(context, encoded, "decode")
        assert result.success
        assert result.data["result"] == "hello world & special chars"

        # Test invalid operation
        result = url_encode_decode(context, "text", "invalid")
        assert not result.success
        assert result.error is not None

    def test_sort_list_tool(self):
        """Test sort_list builtin tool."""
        from talk_box.builtin_tools import sort_list

        context = ToolContext()

        # Test ascending sort
        items = [3, 1, 4, 1, 5, 9, 2, 6]
        result = sort_list(context, items, "asc")
        assert result.success
        assert result.data["sorted_list"] == [1, 1, 2, 3, 4, 5, 6, 9]

        # Test descending sort
        result = sort_list(context, items, "desc")
        assert result.success
        assert result.data["sorted_list"] == [9, 6, 5, 4, 3, 2, 1, 1]

        # Test with strings
        strings = ["zebra", "apple", "banana", "cherry"]
        result = sort_list(context, strings, "asc")
        assert result.success
        assert result.data["sorted_list"] == ["apple", "banana", "cherry", "zebra"]

    def test_number_sequence_tool(self):
        """Test number_sequence builtin tool."""
        from talk_box.builtin_tools import number_sequence

        context = ToolContext()

        # Test simple sequence - returns list directly
        result = number_sequence(context, 1, 5, 1)
        assert result.success
        assert isinstance(result.data, list)
        assert result.data == [1, 2, 3, 4]  # stops before 5

        # Test with step
        result = number_sequence(context, 0, 11, 2)
        assert result.success
        assert isinstance(result.data, list)
        assert result.data == [0, 2, 4, 6, 8, 10]

        # Test descending
        result = number_sequence(context, 10, 5, -1)
        assert result.success
        assert result.data["sequence"] == [10, 9, 8, 7, 6, 5]

        # Test invalid step (zero)
        result = number_sequence(context, 1, 5, 0)
        assert not result.success
        assert result.error is not None

    def test_date_diff_tool(self):
        """Test date_diff builtin tool."""
        from talk_box.builtin_tools import date_diff

        context = ToolContext()

        # Test date difference
        date1 = "2023-01-01"
        date2 = "2023-01-15"
        result = date_diff(context, date1, date2, "days")
        assert result.success
        assert result.data["difference"] == 14

        # Test different unit
        result = date_diff(context, date1, date2, "weeks")
        assert result.success
        assert result.data["difference"] == 2  # 14 days = 2 weeks

        # Test invalid date format
        result = date_diff(context, "invalid-date", date2, "days")
        assert not result.success
        assert result.error is not None

    def test_path_info_tool(self):
        """Test path_info builtin tool."""
        from talk_box.builtin_tools import path_info

        context = ToolContext()

        # Test with file path
        path = "/home/user/documents/file.txt"
        result = path_info(context, path)
        assert result.success
        assert result.data["dirname"] == "/home/user/documents"
        assert result.data["basename"] == "file.txt"
        assert result.data["extension"] == ".txt"
        assert result.data["stem"] == "file"

        # Test with directory path
        path = "/home/user/documents/"
        result = path_info(context, path)
        assert result.success
        assert result.data["basename"] == ""
        assert result.data["extension"] == ""

    def test_convert_case_edge_cases(self):
        """Test convert_case with edge cases."""
        from talk_box.builtin_tools import convert_case

        context = ToolContext()

        # Test camel case conversion
        result = convert_case(context, "hello_world_example", "camel")
        assert result.success
        assert result.data["converted_text"] == "helloWorldExample"

        # Test snake case conversion
        result = convert_case(context, "HelloWorldExample", "snake")
        assert result.success

        # Test invalid case type
        result = convert_case(context, "text", "invalid_case")
        assert not result.success
        assert result.error is not None

    def test_text_stats_comprehensive(self):
        """Test text_stats with more comprehensive text."""
        from talk_box.builtin_tools import text_stats

        context = ToolContext()

        # Test with complex text
        text = """This is a sample text.

        It has multiple paragraphs! And various punctuation?
        Numbers like 123 and symbols like @#$%.

        Final paragraph here."""

        result = text_stats(context, text)
        assert result.success

        data = result.data
        assert data["words"] > 15
        assert data["lines"] > 5
        assert data["sentences"] >= 3
        assert data["paragraphs"] >= 2
        assert data["characters"] > 100
        assert data["characters_no_spaces"] < data["characters"]


# ---------------------------------------------------------------------------
# Workspace file tools
# ---------------------------------------------------------------------------


class TestWorkspaceFileTools:
    def test_file_write_and_read(self, tmp_path, monkeypatch):
        """file_write creates a file, file_read reads it back."""
        from talk_box.builtin_tools import file_read, file_write

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        # Write a file
        wr = file_write(context, "test.txt", "hello world")
        assert wr.success
        assert "test.txt" in (wr.data or "")

        # Read it back
        rd = file_read(context, "test.txt")
        assert rd.success
        assert rd.data == "hello world"

    def test_file_write_creates_dirs(self, tmp_path, monkeypatch):
        """file_write creates parent directories."""
        from talk_box.builtin_tools import file_write

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        result = file_write(context, "sub/dir/output.txt", "nested content")
        assert result.success
        assert (tmp_path / "sub" / "dir" / "output.txt").read_text() == "nested content"

    def test_file_read_not_found(self, tmp_path, monkeypatch):
        """file_read returns error for missing file."""
        from talk_box.builtin_tools import file_read

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        result = file_read(context, "nope.txt")
        assert not result.success

    def test_file_edit(self, tmp_path, monkeypatch):
        """file_edit replaces text in a file."""
        from talk_box.builtin_tools import file_edit

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        (tmp_path / "config.py").write_text("debug = False\nverbose = True\n")
        result = file_edit(context, "config.py", "debug = False", "debug = True")
        assert result.success
        assert (tmp_path / "config.py").read_text() == "debug = True\nverbose = True\n"

    def test_list_files(self, tmp_path, monkeypatch):
        """list_files returns files in directory."""
        from talk_box.builtin_tools import list_files

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        (tmp_path / "a.py").write_text("# a")
        (tmp_path / "b.txt").write_text("b")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "c.py").write_text("# c")

        result = list_files(context)
        assert result.success
        assert "a.py" in result.data
        assert "b.txt" in result.data

    def test_list_files_with_pattern(self, tmp_path, monkeypatch):
        """list_files filters by glob pattern."""
        from talk_box.builtin_tools import list_files

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        (tmp_path / "a.py").write_text("# a")
        (tmp_path / "b.txt").write_text("b")

        result = list_files(context, pattern="*.py")
        assert result.success
        assert "a.py" in result.data
        assert "b.txt" not in result.data

    def test_file_search(self, tmp_path, monkeypatch):
        """file_search finds text in files."""
        from talk_box.builtin_tools import file_search

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        (tmp_path / "main.py").write_text("import pandas as pd\nprint('hello')\n")
        (tmp_path / "other.py").write_text("import numpy\n")

        result = file_search(context, "pandas")
        assert result.success
        assert "pandas" in result.data
        assert "main.py" in result.data

    def test_file_write_path_traversal_blocked(self, tmp_path, monkeypatch):
        """file_write rejects path traversal."""
        from talk_box.builtin_tools import file_write

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        context = ToolContext()

        result = file_write(context, "../../etc/passwd", "hacked")
        assert not result.success
        assert "traversal" in (result.error or "").lower()

    def test_get_all_includes_workspace_tools(self):
        """get_all_tool_box_tools includes the workspace tools."""
        from talk_box.builtin_tools import get_all_tool_box_tools

        tools = get_all_tool_box_tools()
        assert "file_read" in tools
        assert "file_write" in tools
        assert "file_edit" in tools
        assert "list_files" in tools
        assert "file_search" in tools


class TestFileApprovalCallback:
    def test_no_callback_auto_approves(self, tmp_path, monkeypatch):
        """Without a callback, file_write succeeds immediately."""
        from talk_box.builtin_tools import file_write, set_file_approval_callback

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        set_file_approval_callback(None)
        context = ToolContext()

        result = file_write(context, "test.txt", "hello")
        assert result.success

    def test_callback_approves(self, tmp_path, monkeypatch):
        """Callback returning True allows the write."""
        from talk_box.builtin_tools import file_write, set_file_approval_callback

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        approvals = []

        def approve_all(action, path, details):
            approvals.append((action, path))
            return True

        set_file_approval_callback(approve_all)
        context = ToolContext()

        result = file_write(context, "approved.txt", "content")
        assert result.success
        assert approvals == [("write", "approved.txt")]
        set_file_approval_callback(None)  # cleanup

    def test_callback_rejects_write(self, tmp_path, monkeypatch):
        """Callback returning False blocks the write."""
        from talk_box.builtin_tools import file_write, set_file_approval_callback

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        set_file_approval_callback(lambda action, path, details: False)
        context = ToolContext()

        result = file_write(context, "rejected.txt", "content")
        assert not result.success
        assert "rejected" in (result.error or "").lower()
        assert not (tmp_path / "rejected.txt").exists()
        set_file_approval_callback(None)

    def test_callback_rejects_edit(self, tmp_path, monkeypatch):
        """Callback returning False blocks file_edit."""
        from talk_box.builtin_tools import file_edit, set_file_approval_callback

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        (tmp_path / "config.py").write_text("debug = False\n")
        set_file_approval_callback(lambda action, path, details: False)
        context = ToolContext()

        result = file_edit(context, "config.py", "debug = False", "debug = True")
        assert not result.success
        # File should be unchanged
        assert (tmp_path / "config.py").read_text() == "debug = False\n"
        set_file_approval_callback(None)

    def test_callback_receives_edit_details(self, tmp_path, monkeypatch):
        """Edit callback receives old_text and new_text in details."""
        from talk_box.builtin_tools import file_edit, set_file_approval_callback

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        (tmp_path / "data.py").write_text("x = 1\n")
        captured = {}

        def capture(action, path, details):
            captured.update({"action": action, "path": path, "details": details})
            return True

        set_file_approval_callback(capture)
        context = ToolContext()
        file_edit(context, "data.py", "x = 1", "x = 2")

        assert captured["action"] == "edit"
        assert captured["details"]["old_text"] == "x = 1"
        assert captured["details"]["new_text"] == "x = 2"
        set_file_approval_callback(None)

    def test_get_file_approval_callback(self):
        """get_file_approval_callback returns current callback."""
        from talk_box.builtin_tools import (
            get_file_approval_callback,
            set_file_approval_callback,
        )

        assert get_file_approval_callback() is None
        cb = lambda a, p, d: True  # noqa: E731
        set_file_approval_callback(cb)
        assert get_file_approval_callback() is cb
        set_file_approval_callback(None)


class TestBatchApproval:
    def test_run_batch_approval_no_callback(self):
        """Without a batch callback, _batch_decisions stays empty."""
        import talk_box.builtin_tools as bt

        bt._batch_approval_callback = None
        bt._batch_decisions = {}
        bt.run_batch_approval([{"name": "file_write", "input": {"path": "a.txt", "content": "x"}}])
        assert bt._batch_decisions == {}

    def test_run_batch_approval_extracts_file_ops(self):
        """Batch callback receives file ops extracted from tool calls."""
        import talk_box.builtin_tools as bt

        captured = []

        def capture_batch(pending):
            captured.extend(pending)
            return {(a, p): True for a, p, _ in pending}

        bt.set_batch_approval_callback(capture_batch)
        tool_calls = [
            {"name": "file_write", "input": {"path": "a.txt", "content": "aaa"}},
            {"name": "file_read", "input": {"path": "b.txt"}},  # not a write
            {"name": "file_edit", "input": {"path": "c.py", "old_text": "x", "new_text": "y"}},
        ]
        bt.run_batch_approval(tool_calls)

        assert len(captured) == 2
        assert captured[0] == ("write", "a.txt", {"content": "aaa"})
        assert captured[1] == ("edit", "c.py", {"old_text": "x", "new_text": "y"})
        bt.set_batch_approval_callback(None)

    def test_batch_decisions_override_per_file_callback(self, tmp_path, monkeypatch):
        """When batch decisions exist, per-file callback is not called."""
        import talk_box.builtin_tools as bt

        monkeypatch.setattr("talk_box.builtin_tools._get_workspace_root", lambda: tmp_path)
        per_file_called = []
        bt.set_file_approval_callback(lambda a, p, d: per_file_called.append(1) or True)
        bt._batch_decisions = {("write", "approved.txt"): True, ("write", "rejected.txt"): False}

        context = ToolContext()
        r1 = bt.file_write(context, "approved.txt", "content")
        assert r1.success
        r2 = bt.file_write(context, "rejected.txt", "content")
        assert not r2.success

        # Per-file callback should NOT have been called
        assert per_file_called == []

        bt.set_file_approval_callback(None)
        bt._batch_decisions = {}

    def test_batch_decisions_skip_non_file_tools(self):
        """Non-file tool calls are ignored by run_batch_approval."""
        import talk_box.builtin_tools as bt

        bt.set_batch_approval_callback(lambda pending: {})
        bt.run_batch_approval(
            [
                {"name": "web_search", "input": {"query": "hello"}},
                {"name": "list_files", "input": {"pattern": "*"}},
            ]
        )
        assert bt._batch_decisions == {}
        bt.set_batch_approval_callback(None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
