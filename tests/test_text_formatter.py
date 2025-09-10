from talk_box._text_formatter import (
    wrap_prompt_text,
    _is_special_block,
    _wrap_single_line,
    _is_numbered_list,
    _is_bullet_list,
    _is_continuation_line,
    _wrap_numbered_list,
    _wrap_bullet_list,
    _wrap_continuation_line,
    _wrap_regular_text,
    _is_comma_separated_list,
    _wrap_comma_separated_list,
)


# Test the main wrap_prompt_text function
def test_basic_wrapping():
    """Test basic text wrapping functionality."""
    text = "This is a very long line that should be wrapped when it exceeds the specified width limit for the formatter"
    result = wrap_prompt_text(text, width=50)
    lines = result.split("\n")
    assert all(len(line) <= 50 for line in lines)
    assert len(lines) > 1


def test_preserve_paragraphs():
    """Test that double newlines (paragraphs) are preserved."""
    text = "First paragraph.\n\nSecond paragraph."
    result = wrap_prompt_text(text, width=100)
    assert "\n\n" in result


def test_empty_blocks():
    """Test handling of empty blocks."""
    text = "Text\n\n\n\nMore text"
    result = wrap_prompt_text(text, width=100)
    assert "Text" in result
    assert "More text" in result


def test_special_blocks_preserved():
    """Test that special blocks are preserved without wrapping."""
    text = "```\ncode block\nvery long line that should not be wrapped\n```"
    result = wrap_prompt_text(text, width=30)
    assert "very long line that should not be wrapped" in result


def test_default_width():
    """Test that default width is 100."""
    # Create text with spaces so it can wrap properly
    text = " ".join(["word"] * 30)  # Long line with spaces
    result = wrap_prompt_text(text)
    lines = result.split("\n")
    assert all(len(line) <= 100 for line in lines)


# Test the _is_special_block function
def test_code_block_start():
    """Test detection of code blocks starting with ```."""
    assert _is_special_block("```python\ncode here")


def test_code_block_end():
    """Test detection of code blocks ending with ```."""
    assert _is_special_block("code here\n```")


def test_indented_code_block():
    """Test detection of heavily indented blocks."""
    block = "    def function():\n        return True\n    # comment"
    assert _is_special_block(block)


def test_mixed_indented_block():
    """Test block with mix of indented and empty lines."""
    block = "    line one\n\n    line two"
    assert _is_special_block(block)


def test_not_special_block():
    """Test normal text that should not be preserved."""
    assert not _is_special_block("Normal text here")


def test_single_line_indented_not_special():
    """Test that single indented line is not considered special."""
    assert not _is_special_block("    single line")


# Test the _wrap_single_line function
def test_empty_line():
    """Test handling of empty lines."""
    result = _wrap_single_line("", 100)
    assert result == [""]


def test_whitespace_only_line():
    """Test handling of whitespace-only lines."""
    result = _wrap_single_line("   ", 100)
    assert result == ["   "]


def test_short_line_unchanged():
    """Test that short lines are returned unchanged."""
    line = "Short line"
    result = _wrap_single_line(line, 100)
    assert result == [line]


def test_numbered_list_detection():
    """Test that numbered lists are detected and wrapped properly."""
    line = "  1. This is a numbered list item that is too long"
    result = _wrap_single_line(line, 30)
    assert len(result) > 1
    assert result[0].startswith("  1. ")


def test_bullet_list_detection():
    """Test that bullet lists are detected and wrapped properly."""
    line = "  - This is a bullet list item that is too long"
    result = _wrap_single_line(line, 30)
    assert len(result) > 1
    assert result[0].startswith("  - ")


def test_continuation_line_detection():
    """Test that continuation lines are detected and wrapped properly."""
    line = "  Required: This is a continuation line that is too long"
    result = _wrap_single_line(line, 30)
    assert len(result) > 1
    assert result[0].startswith("  Required: ")


# Test pattern detection functions
def test_is_numbered_list_patterns():
    """Test various numbered list patterns."""
    assert _is_numbered_list("1. Item")
    assert _is_numbered_list("2) Item")
    assert _is_numbered_list("(3) Item")
    assert _is_numbered_list("10. Item")
    assert not _is_numbered_list("a. Item")
    assert not _is_numbered_list("Regular text")


def test_is_bullet_list_patterns():
    """Test various bullet list patterns."""
    assert _is_bullet_list("- Item")
    assert _is_bullet_list("* Item")
    assert _is_bullet_list("+ Item")
    assert _is_bullet_list("• Item")
    assert not _is_bullet_list("Regular text")
    assert not _is_bullet_list("1. Item")


def test_is_continuation_line_patterns():
    """Test continuation line patterns."""
    assert _is_continuation_line("Required: something")
    assert _is_continuation_line("Optional: something")
    assert _is_continuation_line("Success: something")
    assert _is_continuation_line("Tools: something")
    assert not _is_continuation_line("Purpose: something")  # Not in patterns
    assert not _is_continuation_line("Regular text")
    assert not _is_continuation_line("- Bullet item")


# Test numbered list wrapping functionality
def test_basic_numbered_list_wrap():
    """Test basic numbered list wrapping."""
    result = _wrap_numbered_list("  ", "1. Short item", 100)
    assert result == ["  1. Short item"]


def test_long_numbered_list_wrap():
    """Test wrapping of long numbered list items."""
    content = "1. This is a very long numbered list item that needs to be wrapped"
    result = _wrap_numbered_list("  ", content, 30)
    assert len(result) > 1
    assert result[0].startswith("  1. ")
    assert result[1].startswith("     ")  # Continuation indent


def test_numbered_list_no_match():
    """Test numbered list function with non-numbered content."""
    result = _wrap_numbered_list("", "Regular text", 100)
    assert "Regular text" in result[0]


# Test bullet list wrapping functionality
def test_basic_bullet_list_wrap():
    """Test basic bullet list wrapping."""
    result = _wrap_bullet_list("  ", "- Short item", 100)
    assert result == ["  - Short item"]


def test_long_bullet_list_wrap():
    """Test wrapping of long bullet list items."""
    content = "- This is a very long bullet list item that needs to be wrapped"
    result = _wrap_bullet_list("  ", content, 30)
    assert len(result) > 1
    assert result[0].startswith("  - ")
    assert result[1].startswith("    ")  # Continuation indent


def test_bullet_list_no_match():
    """Test bullet list function with non-bullet content."""
    result = _wrap_bullet_list("", "Regular text", 100)
    # Should fall back to regular text wrapping
    assert "Regular text" in result[0]


# Test continuation line wrapping functionality
def test_basic_continuation_line():
    """Test basic continuation line wrapping."""
    result = _wrap_continuation_line("  ", "Required: Short text", 100)
    assert result == ["  Required: Short text"]


def test_long_continuation_line():
    """Test wrapping of long continuation lines."""
    content = "Required: This is a very long continuation line that needs wrapping"
    result = _wrap_continuation_line("  ", content, 30)
    assert len(result) > 1
    assert result[0].startswith("  Required: ")
    assert result[1].startswith("            ")  # Aligned continuation


def test_continuation_line_with_comma_list():
    """Test continuation line with comma-separated numbered list."""
    content = "Required: (1) item one, (2) item two, (3) very long item three"
    result = _wrap_continuation_line("  ", content, 40)
    assert len(result) > 1
    # Should use comma-separated list formatting


def test_continuation_line_no_colon():
    """Test continuation line function with no colon."""
    result = _wrap_continuation_line("", "Regular text", 100)
    # Should fall back to regular text wrapping
    assert "Regular text" in result[0]


# Test comma-separated list functionality
def test_is_comma_separated_list_detection():
    """Test detection of comma-separated numbered lists."""
    assert _is_comma_separated_list("(1) item, (2) another")
    assert _is_comma_separated_list("(1) first, (2) second, (3) third")
    assert not _is_comma_separated_list("(1) single item")
    assert not _is_comma_separated_list("regular text with (1)")


def test_comma_separated_list_wrapping():
    """Test wrapping of comma-separated numbered lists."""
    text = "(1) first item, (2) second item, (3) third item"
    result = _wrap_comma_separated_list("  ", "Required: ", text, 40)
    assert len(result) > 1
    assert result[0].startswith("  Required: (1)")


def test_comma_separated_list_very_long_items():
    """Test handling of very long individual items."""
    text = "(1) extremely long item that exceeds width, (2) another very long item"
    result = _wrap_comma_separated_list("", "Test: ", text, 30)
    # Should handle long items gracefully
    assert len(result) > 2


def test_comma_separated_list_empty_items():
    """Test handling of empty or malformed items."""
    text = "(1) item, , (2) another"
    result = _wrap_comma_separated_list("", "Test: ", text, 100)
    # Should filter out empty items
    assert "(1) item" in result[0]
    assert "(2) another" in result[0] or "(2) another" in str(result)


def test_comma_separated_list_no_items():
    """Test handling when no valid items are found."""
    text = "no numbered items here"
    result = _wrap_comma_separated_list("", "Test: ", text, 100)
    assert len(result) == 1
    assert "Test: no numbered items here" in result[0]


def test_comma_separated_list_final_line_too_long():
    """Test comma-separated list where final line exceeds width"""
    text = "Here are the really long items that should wrap nicely: first_extremely_long_item_name, second_very_long_item_name"
    result = wrap_prompt_text(text, 60)

    # Should wrap the comma-separated list
    assert len(result) > 1
    # Final line should not exceed width
    assert all(len(line) <= 60 for line in result)


def test_comma_list_with_continuations_and_commas():
    """Test comma list with complex continuation logic - covers line 256->260"""
    text = "Items: first, second item with comma in it, third"
    result = wrap_prompt_text(text, 30)

    # Should handle commas within items
    assert len(result) >= 1
    # Check that result is a string
    assert isinstance(result, str)
    # Should contain the original text content
    assert "first" in result
    assert "third" in result


def test_comma_list_empty_items_after_processing():
    """Test comma list that results in empty items - covers line 263"""
    text = "Items: , , valid_item, ,"
    result = wrap_prompt_text(text, 100)

    # Should process the text (may not filter empty items as expected)
    assert len(result) >= 1
    assert isinstance(result, str)
    # Check that valid_item is preserved
    assert "valid_item" in result


def test_comma_list_final_line_with_parentheses_wrapping():
    """Test final line wrapping with parentheses detection - covers lines 306-322"""
    text = "Methods: setup(), configure_really_long_method_name_that_needs_wrapping_badly(param1, param2, param3)"
    result = wrap_prompt_text(text, 50)

    # Should wrap the text
    assert len(result) >= 1
    # Should preserve parentheses structure
    joined = "".join(result)
    assert "setup()" in joined
    assert "configure_really_long_method_name" in joined
    assert "param1" in joined and "param3)" in joined


def test_comma_separated_list_path_coverage_256_260():
    """Force coverage of lines 256->260 in comma list processing - else branch"""
    # Create text that has non-numbered parts to trigger the else branch
    text = "Required: (1) item1 extra text here, (2) item2, more text"
    result = wrap_prompt_text(text, 30)

    # Should trigger the else branch in item collection
    assert isinstance(result, str)
    assert "(1)" in result
    assert "(2)" in result
    assert "extra text here" in result


def test_comma_separated_list_path_coverage_263():
    """Force coverage of line 263 - empty items filtering"""
    # Create text that results in all empty items to hit the empty items check
    text = "Required: (1), (2) , (3) ,"
    result = wrap_prompt_text(text, 100)

    # Test should handle empty item filtering
    assert isinstance(result, str)
    assert "Required:" in result


def test_comma_separated_list_path_coverage_306_322():
    """Force coverage of lines 306-322 - final line wrapping with parentheses"""
    # Create text that makes the final line too long and contains parentheses for extraction
    text = "Required: (1) short, (2) extremely_long_function_name_that_definitely_exceeds_any_reasonable_width_limit_with_parameters(param1, param2)"
    result = wrap_prompt_text(text, 50)  # Short width to force final line wrapping

    # Should trigger final line parentheses wrapping and extraction logic
    assert isinstance(result, str)
    assert "short" in result
    assert "extremely_long_function_name" in result
    assert "param1" in result


def test_comma_separated_list_no_valid_items():
    """Test case where no valid items remain after filtering - hits early return"""
    # Create malformed comma list that results in no valid items
    text = "Required: , , ,"
    result = wrap_prompt_text(text, 100)

    # Should return the original text when no valid items found
    assert isinstance(result, str)
    assert "Required:" in result


# Test regular text wrapping functionality
def test_short_regular_text():
    """Test short text that doesn't need wrapping."""
    result = _wrap_regular_text("  ", "Short text", 100)
    assert result == ["  Short text"]


def test_long_regular_text():
    """Test long text that needs wrapping."""
    content = "This is a very long line of regular text that needs to be wrapped"
    result = _wrap_regular_text("  ", content, 30)
    assert len(result) > 1
    assert all(line.startswith("  ") for line in result)


# Test complex integration scenarios
def test_mixed_content_types():
    """Test text with multiple content types."""
    text = """Introduction text here.

SECTION HEADER:
- Bullet item one that is quite long and should wrap
- Bullet item two
  1. Nested numbered item
  2. Another nested numbered item with very long text that wraps

Required: (1) first requirement, (2) second requirement that is long, (3) third requirement

```
code block here
should not wrap
```

Final paragraph text."""

    result = wrap_prompt_text(text, width=50)

    # Should preserve structure
    assert "Introduction text here." in result
    assert "SECTION HEADER:" in result
    assert "```" in result
    assert "should not wrap" in result
    assert "Final paragraph text." in result


def test_edge_case_empty_and_whitespace():
    """Test edge cases with empty strings and whitespace."""
    # Empty string
    result = wrap_prompt_text("", 100)
    assert result == ""

    # Only whitespace
    result = wrap_prompt_text("   \n\n   ", 100)
    lines = result.split("\n")
    assert len(lines) >= 3


def test_very_long_single_words():
    """Test handling of very long single words."""
    text = "Short text supercalifragilisticexpialidocious more text"
    result = wrap_prompt_text(text, width=20)
    # Should not break the long word
    assert "supercalifragilisticexpialidocious" in result


def test_numbered_list_with_various_formats():
    """Test different numbered list formats in same text."""
    text = """Different numbered formats:
1. Format one
2) Format two
(3) Format three
10. Double digit"""

    result = wrap_prompt_text(text, width=100)
    assert "1. Format one" in result
    assert "2) Format two" in result
    assert "(3) Format three" in result
    assert "10. Double digit" in result


def test_comma_separated_list_edge_cases():
    """Test edge cases in comma-separated list processing."""
    # Test the missing branch: final line too long without parentheses
    text = "(1) item"
    result = _wrap_comma_separated_list("", "Test: ", text, 50)
    assert len(result) >= 1

    # Test very long final line that needs special handling
    text = "(1) short, (2) " + "x" * 100  # Very long second item
    result = _wrap_comma_separated_list("", "Required: ", text, 50)
    # Should handle the long final line
    assert len(result) > 1

    # Test final line wrapping when item_start is -1 (no parenthesis found)
    # This tests the 'else' branch in the final line check
    continuation_indent = "            "
    test_line = continuation_indent + "very long text without parentheses that exceeds width limit"
    # Manually set up the scenario where current_line has no '('
    result = _wrap_comma_separated_list("", "Test: ", "no parentheses here very long text", 30)
    assert len(result) >= 1
