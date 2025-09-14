import json
import math
import os
import re
import urllib.parse
from datetime import datetime
from typing import Any, List, Optional

from .tools import ToolCategory, ToolContext, ToolResult, tool

# ============================================================================
# TOOL BOX: TEXT AND STRING TOOLS
# ============================================================================


@tool(
    description="Count words, characters, and lines in text",
    category=ToolCategory.DATA,
    examples=[
        "text_stats('Hello world!') -> {'words': 2, 'chars': 12, 'lines': 1}",
        "text_stats('Line 1\\nLine 2') -> {'words': 4, 'chars': 13, 'lines': 2}",
    ],
    tags=["tool_box", "text", "analysis", "statistics"],
)
def text_stats(context: ToolContext, text: str) -> ToolResult:
    """Analyze text and return statistics about words, characters, and lines."""
    lines = text.split("\n")
    words = text.split()

    stats = {
        "characters": len(text),
        "characters_no_spaces": len(text.replace(" ", "")),
        "words": len(words),
        "lines": len(lines),
        "sentences": len(re.findall(r"[.!?]+", text)),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
    }

    return ToolResult(data=stats, metadata={"text_length": len(text)}, display_format="json")


@tool(
    description="Convert text between different cases (upper, lower, title, camel)",
    category=ToolCategory.DATA,
    examples=[
        "convert_case('hello world', 'upper') -> 'HELLO WORLD'",
        "convert_case('hello_world', 'camel') -> 'helloWorld'",
    ],
    tags=["tool_box", "text", "formatting", "case"],
)
def convert_case(context: ToolContext, text: str, case_type: str) -> ToolResult:
    """Convert text to different case formats."""
    case_type = case_type.lower()

    if case_type == "upper":
        result = text.upper()
    elif case_type == "lower":
        result = text.lower()
    elif case_type == "title":
        result = text.title()
    elif case_type == "camel":
        words = re.sub(r"[_\-\s]+", " ", text).split()
        result = words[0].lower() + "".join(word.capitalize() for word in words[1:])
    elif case_type == "pascal":
        words = re.sub(r"[_\-\s]+", " ", text).split()
        result = "".join(word.capitalize() for word in words)
    elif case_type == "snake":
        result = re.sub(r"[^\w\s]", "", text)
        result = re.sub(r"\s+", "_", result).lower()
    elif case_type == "kebab":
        result = re.sub(r"[^\w\s]", "", text)
        result = re.sub(r"\s+", "-", result).lower()
    else:
        return ToolResult(
            data=None,
            success=False,
            error=f"Unknown case type: {case_type}. Supported: upper, lower, title, camel, pascal, snake, kebab",
        )

    return ToolResult(data=result, metadata={"original": text, "case_type": case_type})


# ============================================================================
# TOOL BOX: MATH AND CALCULATION TOOLS
# ============================================================================


@tool(
    description="Perform basic mathematical calculations",
    category=ToolCategory.DATA,
    examples=["calculate('2 + 3 * 4') -> 14", "calculate('sqrt(16)') -> 4.0"],
    tags=["tool_box", "math", "calculation", "arithmetic"],
)
def calculate(context: ToolContext, expression: str) -> ToolResult:
    """Safely evaluate mathematical expressions."""
    # Define safe functions that can be used
    safe_functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "sqrt": math.sqrt,
        "pow": pow,
        "log": math.log,
        "log10": math.log10,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pi": math.pi,
        "e": math.e,
    }

    # Remove potentially dangerous functions/keywords
    dangerous = ["import", "exec", "eval", "__", "open", "file"]
    if any(danger in expression.lower() for danger in dangerous):
        return ToolResult(
            data=None, success=False, error="Expression contains potentially dangerous operations"
        )

    try:
        # Use eval with restricted globals
        result = eval(expression, {"__builtins__": {}}, safe_functions)
        return ToolResult(data=result, metadata={"expression": expression})
    except Exception as e:
        return ToolResult(data=None, success=False, error=f"Calculation error: {str(e)}")


@tool(
    description="Generate a sequence of numbers with specified start, stop, and step",
    category=ToolCategory.DATA,
    examples=[
        "number_sequence(1, 10, 2) -> [1, 3, 5, 7, 9]",
        "number_sequence(0, 1, 0.25) -> [0, 0.25, 0.5, 0.75]",
    ],
    tags=["tool_box", "math", "sequence", "range"],
)
def number_sequence(
    context: ToolContext, start: float, stop: float, step: float = 1.0
) -> ToolResult:
    """Generate a sequence of numbers."""
    if step == 0:
        return ToolResult(data=None, success=False, error="Step cannot be zero")

    sequence = []
    current = start

    while (step > 0 and current < stop) or (step < 0 and current > stop):
        sequence.append(current)
        current += step

        # Prevent infinite loops
        if len(sequence) > 10000:
            return ToolResult(data=None, success=False, error="Sequence too long (>10000 elements)")

    return ToolResult(
        data=sequence,
        metadata={"start": start, "stop": stop, "step": step, "length": len(sequence)},
    )


# ============================================================================
# TOOL BOX: DATE AND TIME TOOLS
# ============================================================================


@tool(
    description="Get current date and time in various formats",
    category=ToolCategory.SYSTEM,
    examples=[
        "current_time() -> {'iso': '2023-10-01T12:00:00Z', 'unix': 1696176000}",
        "current_time('US/Eastern') -> timezone-aware datetime",
    ],
    tags=["tool_box", "time", "date", "timestamp"],
)
def current_time(context: ToolContext, timezone: Optional[str] = None) -> ToolResult:
    """Get current date and time information."""
    now = datetime.utcnow()

    result = {
        "iso": now.isoformat() + "Z",
        "unix": int(now.timestamp()),
        "readable": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "month": now.strftime("%B"),
        "year": now.year,
    }

    if timezone:
        result["requested_timezone"] = timezone
        result["note"] = "Timezone conversion requires pytz library"

    return ToolResult(data=result, metadata={"timezone": timezone or "UTC"})


@tool(
    description="Calculate the difference between two dates",
    category=ToolCategory.DATA,
    examples=[
        "date_diff('2023-01-01', '2023-12-31') -> {'days': 364, 'weeks': 52}",
        "date_diff('2023-01-01 10:00', '2023-01-01 15:30') -> {'hours': 5.5}",
    ],
    tags=["tool_box", "time", "date", "calculation", "difference"],
)
def date_diff(context: ToolContext, start_date: str, end_date: str) -> ToolResult:
    """Calculate the difference between two dates."""
    try:
        # Try to parse dates in common formats
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
        ]

        start_dt = None
        end_dt = None

        for fmt in formats:
            try:
                start_dt = datetime.strptime(start_date, fmt)
                break
            except ValueError:
                continue

        for fmt in formats:
            try:
                end_dt = datetime.strptime(end_date, fmt)
                break
            except ValueError:
                continue

        if not start_dt or not end_dt:
            return ToolResult(
                data=None,
                success=False,
                error="Could not parse date format. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS",
            )

        diff = end_dt - start_dt
        total_seconds = abs(diff.total_seconds())

        result = {
            "days": diff.days,
            "total_seconds": total_seconds,
            "hours": total_seconds / 3600,
            "minutes": total_seconds / 60,
            "weeks": diff.days / 7,
            "is_future": diff.total_seconds() > 0,
            "absolute_days": abs(diff.days),
        }

        return ToolResult(
            data=result,
            metadata={
                "start_date": start_date,
                "end_date": end_date,
                "parsed_start": start_dt.isoformat(),
                "parsed_end": end_dt.isoformat(),
            },
        )

    except Exception as e:
        return ToolResult(data=None, success=False, error=f"Date calculation error: {str(e)}")


# ============================================================================
# TOOL BOX: DATA MANIPULATION TOOLS
# ============================================================================


@tool(
    description="Parse JSON string and return structured data",
    category=ToolCategory.DATA,
    examples=[
        'parse_json(\'{"name": "John", "age": 30}\') -> {"name": "John", "age": 30}',
        "parse_json('[1, 2, 3]') -> [1, 2, 3]",
    ],
    tags=["tool_box", "json", "parsing", "data"],
)
def parse_json(context: ToolContext, json_string: str) -> ToolResult:
    """Parse a JSON string and return the structured data."""
    try:
        data = json.loads(json_string)
        return ToolResult(
            data=data,
            metadata={
                "original_string": json_string[:100] + "..."
                if len(json_string) > 100
                else json_string,
                "data_type": type(data).__name__,
            },
        )
    except json.JSONDecodeError as e:
        return ToolResult(data=None, success=False, error=f"Invalid JSON: {str(e)}")


@tool(
    description="Convert data to JSON string with formatting options",
    category=ToolCategory.DATA,
    examples=[
        'to_json({"name": "John"}) -> \'{"name": "John"}\'',
        "to_json([1, 2, 3], indent=2) -> formatted JSON",
    ],
    tags=["tool_box", "json", "serialization", "formatting"],
)
def to_json(
    context: ToolContext, data: Any, indent: Optional[int] = None, sort_keys: bool = False
) -> ToolResult:
    """Convert data to JSON string with formatting options."""
    try:
        json_string = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
        return ToolResult(
            data=json_string,
            metadata={
                "indent": indent,
                "sort_keys": sort_keys,
                "data_type": type(data).__name__,
                "length": len(json_string),
            },
        )
    except (TypeError, ValueError) as e:
        return ToolResult(data=None, success=False, error=f"Serialization error: {str(e)}")


@tool(
    description="Sort a list of items with various options",
    category=ToolCategory.DATA,
    examples=[
        "sort_list([3, 1, 4, 1, 5]) -> [1, 1, 3, 4, 5]",
        "sort_list(['apple', 'Banana'], reverse=True) -> ['Banana', 'apple']",
    ],
    tags=["tool_box", "sorting", "list", "data"],
)
def sort_list(
    context: ToolContext, items: List[Any], reverse: bool = False, case_sensitive: bool = True
) -> ToolResult:
    """Sort a list of items with various options."""
    try:
        if not isinstance(items, list):
            return ToolResult(data=None, success=False, error="Input must be a list")

        # Handle string sorting with case sensitivity
        if items and isinstance(items[0], str) and not case_sensitive:
            sorted_items = sorted(items, key=str.lower, reverse=reverse)
        else:
            sorted_items = sorted(items, reverse=reverse)

        return ToolResult(
            data=sorted_items,
            metadata={
                "original_length": len(items),
                "reverse": reverse,
                "case_sensitive": case_sensitive,
                "data_type": type(items[0]).__name__ if items else "empty",
            },
        )

    except TypeError as e:
        return ToolResult(data=None, success=False, error=f"Cannot sort items: {str(e)}")


# ============================================================================
# TOOL BOX: URL AND WEB TOOLS
# ============================================================================


@tool(
    description="Parse and analyze URLs to extract components",
    category=ToolCategory.WEB,
    examples=[
        "parse_url('https://example.com/path?q=value') -> {'scheme': 'https', 'domain': 'example.com'}",
        "parse_url('ftp://user@server.com:21/file.txt') -> detailed URL components",
    ],
    tags=["tool_box", "url", "parsing", "web"],
)
def parse_url(context: ToolContext, url: str) -> ToolResult:
    """Parse a URL and extract its components."""
    try:
        parsed = urllib.parse.urlparse(url)
        query_params = dict(urllib.parse.parse_qsl(parsed.query))

        result = {
            "scheme": parsed.scheme,
            "domain": parsed.netloc.split("@")[-1].split(":")[0]
            if "@" in parsed.netloc
            else parsed.netloc.split(":")[0],
            "netloc": parsed.netloc,
            "path": parsed.path,
            "query": parsed.query,
            "fragment": parsed.fragment,
            "query_params": query_params,
            "port": parsed.port,
            "username": parsed.username,
            "password": "***" if parsed.password else None,
        }

        return ToolResult(
            data=result,
            metadata={
                "original_url": url,
                "is_absolute": bool(parsed.scheme),
                "has_query": bool(parsed.query),
                "param_count": len(query_params),
            },
        )

    except Exception as e:
        return ToolResult(data=None, success=False, error=f"URL parsing error: {str(e)}")


@tool(
    description="Encode or decode URL components",
    category=ToolCategory.WEB,
    examples=[
        "url_encode('hello world!') -> 'hello%20world%21'",
        "url_decode('hello%20world%21') -> 'hello world!'",
    ],
    tags=["tool_box", "url", "encoding", "web"],
)
def url_encode_decode(context: ToolContext, text: str, operation: str = "encode") -> ToolResult:
    """Encode or decode URL components."""
    operation = operation.lower()

    try:
        if operation == "encode":
            result = urllib.parse.quote(text)
        elif operation == "decode":
            result = urllib.parse.unquote(text)
        else:
            return ToolResult(
                data=None, success=False, error="Operation must be 'encode' or 'decode'"
            )

        return ToolResult(
            data=result,
            metadata={
                "original": text,
                "operation": operation,
                "length_change": len(result) - len(text),
            },
        )

    except Exception as e:
        return ToolResult(data=None, success=False, error=f"URL {operation} error: {str(e)}")


# ============================================================================
# TOOL BOX: FILE AND SYSTEM TOOLS
# ============================================================================


@tool(
    description="Get information about file paths and extensions",
    category=ToolCategory.FILE,
    examples=[
        "path_info('/home/user/document.pdf') -> {'name': 'document.pdf', 'ext': '.pdf'}",
        "path_info('../../data/file.txt') -> path components and info",
    ],
    tags=["tool_box", "file", "path", "system"],
)
def path_info(context: ToolContext, file_path: str) -> ToolResult:
    """Get information about a file path."""
    try:
        abs_path = os.path.abspath(file_path)

        result = {
            "original_path": file_path,
            "absolute_path": abs_path,
            "directory": os.path.dirname(abs_path),
            "filename": os.path.basename(abs_path),
            "name_without_ext": os.path.splitext(os.path.basename(abs_path))[0],
            "extension": os.path.splitext(abs_path)[1],
            "is_absolute": os.path.isabs(file_path),
            "exists": os.path.exists(abs_path),
            "is_file": os.path.isfile(abs_path),
            "is_directory": os.path.isdir(abs_path),
        }

        # Add file stats if file exists
        if os.path.exists(abs_path):
            stat = os.stat(abs_path)
            result.update(
                {
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                }
            )

        return ToolResult(
            data=result,
            metadata={
                "path_length": len(file_path),
                "depth": len([p for p in abs_path.split(os.sep) if p]),
            },
        )

    except Exception as e:
        return ToolResult(data=None, success=False, error=f"Path analysis error: {str(e)}")


# ============================================================================
# TOOL BOX: UTILITY TOOLS
# ============================================================================


@tool(
    description="Generate a UUID (Universally Unique Identifier)",
    category=ToolCategory.SYSTEM,
    examples=[
        "generate_uuid() -> 'f47ac10b-58cc-4372-a567-0e02b2c3d479'",
        "generate_uuid('short') -> shortened UUID format",
    ],
    tags=["tool_box", "uuid", "identifier", "random"],
)
def generate_uuid(context: ToolContext, format_type: str = "standard") -> ToolResult:
    """Generate a UUID in various formats."""
    import uuid

    new_uuid = uuid.uuid4()

    if format_type == "standard":
        result = str(new_uuid)
    elif format_type == "short":
        result = str(new_uuid).replace("-", "")
    elif format_type == "upper":
        result = str(new_uuid).upper()
    elif format_type == "short_upper":
        result = str(new_uuid).replace("-", "").upper()
    else:
        return ToolResult(
            data=None, success=False, error="Format must be: standard, short, upper, or short_upper"
        )

    return ToolResult(
        data=result,
        metadata={
            "format": format_type,
            "version": new_uuid.version,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@tool(
    description="Validate and format email addresses",
    category=ToolCategory.DATA,
    examples=[
        "validate_email('user@example.com') -> {'valid': True, 'domain': 'example.com'}",
        "validate_email('invalid-email') -> {'valid': False, 'error': 'Invalid format'}",
    ],
    tags=["tool_box", "email", "validation", "formatting"],
)
def validate_email(context: ToolContext, email: str) -> ToolResult:
    """Validate and analyze an email address."""
    # Basic email regex (simplified for demo)
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    is_valid = re.match(email_pattern, email) is not None

    result = {"email": email, "valid": is_valid}

    if is_valid:
        parts = email.split("@")
        result.update(
            {
                "local_part": parts[0],
                "domain": parts[1],
                "tld": parts[1].split(".")[-1],
                "formatted": email.lower().strip(),
            }
        )
    else:
        result["error"] = "Invalid email format"

    return ToolResult(data=result, metadata={"original": email, "length": len(email)})


# ============================================================================
# TOOL BOX: REGISTRY AND EXPORT FUNCTIONS
# ============================================================================


def get_all_tool_box_tools() -> List[str]:
    """Get names of all Tool Box built-in tools."""
    return [
        "text_stats",
        "convert_case",
        "calculate",
        "number_sequence",
        "current_time",
        "date_diff",
        "parse_json",
        "to_json",
        "sort_list",
        "parse_url",
        "url_encode_decode",
        "path_info",
        "generate_uuid",
        "validate_email",
    ]


def get_builtin_tool(name: str):
    """Get a specific built-in tool by name.

    This allows cherry-picking individual tools from the Tool Box rather than
    loading all tools with load_tool_box(). The tool will be automatically
    loaded into the registry if it's not already there.

    Parameters
    ----------
    name : str
        Name of the tool to retrieve

    Returns
    -------
    TalkBoxTool
        The requested tool

    Examples
    --------
    >>> calculate_tool = get_builtin_tool("calculate")
    >>> email_tool = get_builtin_tool("validate_email")
    """
    from .tools import get_global_registry

    # Available Tool Box tools (all tools defined in this module)
    available_tools = {
        "text_stats",
        "convert_case",
        "calculate",
        "number_sequence",
        "current_time",
        "date_diff",
        "parse_json",
        "to_json",
        "sort_list",
        "parse_url",
        "url_encode_decode",
        "path_info",
        "generate_uuid",
        "validate_email",
    }

    if name not in available_tools:
        available = ", ".join(sorted(available_tools))
        raise ValueError(f"Tool '{name}' not found. Available Tool Box tools: {available}")

    # Check if tool is already in registry
    registry = get_global_registry()
    tool = registry.get_tool(name)

    if tool:
        return tool

    # If tool not in registry, load the full Tool Box to register all tools
    # This is necessary because @tool decorators register tools lazily
    load_tool_box()

    # Now try again
    tool = registry.get_tool(name)
    if not tool:
        raise RuntimeError(
            f"Failed to load Tool Box tool '{name}' even after load_tool_box(). This is a bug."
        )

    return tool


def load_selected_tools(tool_names: List[str]) -> None:
    """Load only specific tools from the Tool Box.

    This provides a middle ground between loading all tools and cherry-picking
    individual ones. Useful for loading several related tools at once.

    Parameters
    ----------
    tool_names : List[str]
        Names of the tools to load

    Examples
    --------
    >>> # Load just text and math tools
    >>> load_selected_tools(["text_stats", "calculate", "convert_case"])
    >>>
    >>> # Load web-related tools
    >>> load_selected_tools(["parse_url", "url_encode_decode"])
    """
    # Simply get each tool - this will automatically handle validation and registration
    for name in tool_names:
        get_builtin_tool(name)


def load_tool_box():
    """Load all Tool Box tools into the global registry."""
    import inspect
    import sys

    from .tools import get_global_registry

    registry = get_global_registry()

    # Check if builtin tools are already registered
    expected_tools = get_all_tool_box_tools()
    missing_tools = [name for name in expected_tools if not registry.get_tool(name)]

    if not missing_tools:
        # All tools already registered
        tool_box_count = len(
            [
                tool
                for tool in registry.get_all_tools()
                if any(
                    tag in tool.tags
                    for tag in ["tool_box", "text", "math", "time", "data", "web", "file"]
                )
                or tool.name in expected_tools
            ]
        )
        return tool_box_count

    # Some tools are missing, need to re-register them
    # This happens when the registry was cleared (e.g., in tests)
    current_module = sys.modules[__name__]

    for name, obj in inspect.getmembers(current_module, inspect.isfunction):
        # Check if this function should be a builtin tool
        if hasattr(obj, "_talk_box_tool") and name in expected_tools:
            # This is a tool function that was decorated but not in registry
            # Re-register it using the existing tool object
            if not registry.get_tool(name):
                tool_obj = getattr(obj, "_talk_box_tool")
                registry.register(tool_obj)

    # Count registered tool box tools
    tool_box_count = len(
        [
            tool
            for tool in registry.get_all_tools()
            if any(
                tag in tool.tags
                for tag in ["tool_box", "text", "math", "time", "data", "web", "file"]
            )
            or tool.name in expected_tools
        ]
    )

    return tool_box_count
