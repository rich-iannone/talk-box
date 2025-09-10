import re
import textwrap
from typing import List


def wrap_prompt_text(text: str, width: int = 100) -> str:
    """
    Wrap prompt text with intelligent handling of structure, indentation, and lists.

    Features:
    - Preserves paragraph structure (double newlines)
    - Maintains indentation and alignment
    - Aligns numbered lists vertically: (1), (2), (3) line up
    - Handles bullet points and nested structures
    - Preserves code blocks and special formatting

    Parameters
    ----------
    text
        The text to wrap.
    width
        Maximum line width in characters.

    Returns
    -------
    str
        Formatted text with intelligent line wrapping
    """
    # Split text into blocks separated by double newlines (paragraphs)
    blocks = text.split("\n\n")
    wrapped_blocks = []

    for block in blocks:
        if not block.strip():
            wrapped_blocks.append("")
            continue

        # Check if this is a code block or special formatting to preserve
        if _is_special_block(block):
            wrapped_blocks.append(block)
            continue

        # Process each line in the block
        lines = block.split("\n")
        wrapped_lines = []

        for line in lines:
            wrapped_lines.extend(_wrap_single_line(line, width))

        wrapped_blocks.append("\n".join(wrapped_lines))

    return "\n\n".join(wrapped_blocks)


def _is_special_block(block: str) -> bool:
    """Check if a block should be preserved without wrapping."""
    # Preserve code blocks (marked with ``` or indented blocks)
    if block.startswith("```") or block.endswith("```"):
        return True

    # Preserve heavily indented blocks (likely code or formatted data)
    lines = block.split("\n")
    if len(lines) > 1 and all(line.startswith("    ") or not line.strip() for line in lines):
        return True

    return False


def _wrap_single_line(line: str, width: int) -> List[str]:
    """
    Wrap a single line with intelligent handling of lists and indentation.

    Returns a list of wrapped lines.
    """
    if not line.strip():
        return [line]

    # If line is already short enough, return as-is
    if len(line) <= width:
        return [line]

    # Detect leading whitespace (indentation)
    leading_match = re.match(r"^(\s*)", line)
    leading_indent = leading_match.group(1) if leading_match else ""
    content = line[len(leading_indent) :]

    # Handle different list patterns
    if _is_numbered_list(content):
        return _wrap_numbered_list(leading_indent, content, width)
    elif _is_bullet_list(content):
        return _wrap_bullet_list(leading_indent, content, width)
    elif _is_continuation_line(content):
        return _wrap_continuation_line(leading_indent, content, width)
    else:
        return _wrap_regular_text(leading_indent, content, width)


def _is_numbered_list(content: str) -> bool:
    """Check if content starts with a numbered list pattern."""
    patterns = [
        r"^\(\d+\)\s+",  # (1) format
        r"^\d+\.\s+",  # 1. format
        r"^\d+\)\s+",  # 1) format
    ]
    return any(re.match(pattern, content) for pattern in patterns)


def _is_bullet_list(content: str) -> bool:
    """Check if content starts with a bullet point."""
    return re.match(r"^[-*+•]\s+", content) is not None


def _is_continuation_line(content: str) -> bool:
    """Check if this looks like a continuation of a list item."""
    # Lines that start with common continuation patterns
    patterns = [
        r"^Required:",
        r"^Optional:",
        r"^Tools:",
        r"^Success:",
    ]
    return any(re.match(pattern, content) for pattern in patterns)


def _wrap_numbered_list(leading_indent: str, content: str, width: int) -> List[str]:
    """Wrap a numbered list with proper alignment."""
    # Extract the number/marker and the text
    for pattern in [r"^(\(\d+\)\s+)", r"^(\d+\.\s+)", r"^(\d+\)\s+)"]:
        match = re.match(pattern, content)
        if match:
            marker = match.group(1)
            text = content[len(marker) :]

            # Calculate available width for text
            available_width = width - len(leading_indent) - len(marker)

            if len(text) <= available_width:
                return [leading_indent + marker + text]

            # Wrap the text
            wrapped_text = textwrap.fill(
                text, width=available_width, break_long_words=False, break_on_hyphens=False
            )

            lines = wrapped_text.split("\n")
            result = [leading_indent + marker + lines[0]]

            # Align continuation lines with the start of the text (after marker)
            continuation_indent = leading_indent + " " * len(marker)
            for line in lines[1:]:
                result.append(continuation_indent + line)

            return result

    # Fallback if no pattern matched
    return _wrap_regular_text(leading_indent, content, width)


def _wrap_bullet_list(leading_indent: str, content: str, width: int) -> List[str]:
    """Wrap a bullet point with proper alignment."""
    match = re.match(r"^([-*+•]\s+)", content)
    if not match:
        return _wrap_regular_text(leading_indent, content, width)

    bullet = match.group(1)
    text = content[len(bullet) :]

    # Calculate available width
    available_width = width - len(leading_indent) - len(bullet)

    if len(text) <= available_width:
        return [leading_indent + bullet + text]

    # Wrap the text
    wrapped_text = textwrap.fill(
        text, width=available_width, break_long_words=False, break_on_hyphens=False
    )

    lines = wrapped_text.split("\n")
    result = [leading_indent + bullet + lines[0]]

    # Align continuation lines with the start of the text (after bullet)
    continuation_indent = leading_indent + " " * len(bullet)
    for line in lines[1:]:
        result.append(continuation_indent + line)

    return result


def _wrap_continuation_line(leading_indent: str, content: str, width: int) -> List[str]:
    """Wrap continuation lines (like 'Required:', 'Optional:') with proper alignment."""
    # Find the colon and align subsequent lines
    colon_match = re.match(r"^([^:]+:\s*)", content)
    if colon_match:
        prefix = colon_match.group(1)
        text = content[len(prefix) :]

        available_width = width - len(leading_indent) - len(prefix)

        if len(text) <= available_width:
            return [leading_indent + prefix + text]

        # Special handling for comma-separated lists
        if _is_comma_separated_list(text):
            return _wrap_comma_separated_list(leading_indent, prefix, text, width)

        # Regular text wrapping
        wrapped_text = textwrap.fill(
            text, width=available_width, break_long_words=False, break_on_hyphens=False
        )

        lines = wrapped_text.split("\n")
        result = [leading_indent + prefix + lines[0]]

        # Continuation lines align with start of text after prefix
        continuation_indent = leading_indent + " " * len(prefix)
        for line in lines[1:]:
            result.append(continuation_indent + line)

        return result

    return _wrap_regular_text(leading_indent, content, width)


def _is_comma_separated_list(text: str) -> bool:
    """Check if text appears to be a comma-separated list."""
    # Look for numbered items in parentheses with commas
    return re.search(r"\(\d+\)[^,]*,", text) is not None


def _wrap_comma_separated_list(
    leading_indent: str, prefix: str, text: str, width: int
) -> List[str]:
    """
    Handle comma-separated numbered lists with special alignment.

    Example:
    Required: (1) detailed problem description, (2) error messages or codes, (3) when problem started,
              (4) recent changes or updates
    """
    # Split on numbered items but keep the numbers
    parts = re.split(r"(\(\d+\))", text)

    # Remove empty parts and reconstruct items
    items = []
    current_item = ""

    for i, part in enumerate(parts):
        if re.match(r"\(\d+\)", part):
            if current_item.strip():
                items.append(current_item.strip().rstrip(",").strip())
            current_item = part
        else:
            current_item += part

    if current_item.strip():
        items.append(current_item.strip().rstrip(",").strip())

    # Remove empty items
    items = [item for item in items if item.strip() and not item.strip() == ","]

    if not items:
        return [leading_indent + prefix + text]

    # Calculate alignment - continuation lines should align with first number
    first_item = items[0]
    continuation_indent = leading_indent + " " * len(prefix)

    lines = []
    current_line = leading_indent + prefix + first_item

    for item in items[1:]:
        # Add comma and space
        test_line = current_line + ", " + item

        if len(test_line) <= width:
            current_line = test_line
        else:
            # Current line is full, start a new one
            lines.append(current_line + ",")

            # Check if the item itself is too long for the continuation line
            item_line = continuation_indent + item
            if len(item_line) <= width:
                current_line = item_line
            else:
                # Need to wrap this individual item
                item_available_width = width - len(continuation_indent)
                wrapped_item = textwrap.fill(
                    item, width=item_available_width, break_long_words=False, break_on_hyphens=False
                )
                wrapped_lines = wrapped_item.split("\n")
                current_line = continuation_indent + wrapped_lines[0]

                # Add any additional wrapped lines with extra indentation
                extra_indent = continuation_indent + "    "  # Extra 4 spaces
                for wrapped_line in wrapped_lines[1:]:
                    lines.append(extra_indent + wrapped_line)

    # Check if final line is too long
    if len(current_line) > width:
        # Need to wrap the final line too
        # Extract the item text from current_line
        item_start = current_line.find("(")
        if item_start != -1:
            item_text = current_line[item_start:]
            line_prefix = current_line[:item_start]

            item_available_width = width - len(line_prefix)
            wrapped_item = textwrap.fill(
                item_text,
                width=item_available_width,
                break_long_words=False,
                break_on_hyphens=False,
            )
            wrapped_lines = wrapped_item.split("\n")
            lines.append(line_prefix + wrapped_lines[0])

            # Add continuation lines with extra indent
            extra_indent = leading_indent + " " * len(prefix) + "    "
            for wrapped_line in wrapped_lines[1:]:
                lines.append(extra_indent + wrapped_line)
        else:
            lines.append(current_line)
    else:
        lines.append(current_line)

    return lines


def _wrap_regular_text(leading_indent: str, content: str, width: int) -> List[str]:
    """Wrap regular text content."""
    available_width = width - len(leading_indent)

    if len(content) <= available_width:
        return [leading_indent + content]

    wrapped_text = textwrap.fill(
        content, width=available_width, break_long_words=False, break_on_hyphens=False
    )

    lines = wrapped_text.split("\n")
    return [leading_indent + line for line in lines]
