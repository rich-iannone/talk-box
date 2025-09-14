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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
