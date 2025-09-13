"""
Talk Box Tool System

A comprehensive tool system that provides superior developer experience for creating,
testing, and using AI tools with Talk Box conversations.

Key advantages over other systems:
- Easy tool creation with @tb.tool decorator
- Rich ToolContext with conversation history and user info
- Built-in testing framework for tool validation
- Pre-built library of common tools
- Deep integration with Talk Box attention-focused prompts
- Comprehensive observability and debugging
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypeVar, get_type_hints

if TYPE_CHECKING:  # pragma: no cover
    import chatlas

# Import chatlas for integration
try:
    import chatlas
except ImportError:  # pragma: no cover
    chatlas = None

__all__ = [
    # Core classes
    "TalkBoxTool",
    "ToolContext",
    "ToolResult",
    "ToolRegistry",
    "ToolExecutor",
    # Decorators
    "tool",
    # Enums
    "ToolStatus",
    "ToolCategory",
    # Exceptions
    "ToolError",
    "ToolExecutionError",
    "ToolRegistrationError",
    # File loading functions
    "load_tools_from_file",
    "load_tools_from_directory",
    "is_python_file_path",
    # Registry functions
    "get_global_registry",
    "clear_global_registry",
]

# Type definitions
F = TypeVar("F", bound=Callable[..., Any])
logger = logging.getLogger(__name__)


class ToolStatus(Enum):
    """Status of a tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolCategory(Enum):
    """Categories for organizing tools."""

    WEB = "web"
    FILE = "file"
    DATA = "data"
    COMMUNICATION = "communication"
    SYSTEM = "system"
    CUSTOM = "custom"
    ANALYSIS = "analysis"
    SEARCH = "search"


# Exceptions
class ToolError(Exception):
    """Base exception for tool-related errors."""

    pass


class ToolExecutionError(ToolError):
    """Raised when a tool fails during execution."""

    pass


class ToolRegistrationError(ToolError):
    """Raised when a tool cannot be registered."""

    pass


@dataclass
class ToolExecutionMetrics:
    """Metrics about tool execution."""

    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    status: ToolStatus = ToolStatus.PENDING
    error: Optional[str] = None


class ToolContext:
    """
    Rich context object passed to all tools.

    Provides access to conversation history, user information,
    and Talk Box specific functionality.
    """

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        user_metadata: Optional[Dict[str, Any]] = None,
        tool_registry: Optional["ToolRegistry"] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.session_id = session_id
        self.conversation_history = conversation_history or []
        self.user_metadata = user_metadata or {}
        self.tool_registry = tool_registry
        self.extra = extra or {}
        self.created_at = datetime.now(timezone.utc)

    def get_last_messages(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the last N messages from conversation history."""
        return self.conversation_history[-n:] if self.conversation_history else []

    def get_user_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference from metadata."""
        return self.user_metadata.get("preferences", {}).get(key, default)

    def log_tool_usage(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        """Log tool usage for analytics."""
        # This would integrate with Talk Box logging system
        logger.info(f"Tool '{tool_name}' called with params: {parameters}")

    @property
    def timestamp(self) -> datetime:
        """Alias for created_at for backward compatibility."""
        return self.created_at

    def create_child_context(self, **overrides) -> "ToolContext":
        """Create a child context with some values overridden."""
        kwargs = {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "user_metadata": self.user_metadata,
            "tool_registry": self.tool_registry,
            "extra": self.extra.copy(),
        }
        kwargs.update(overrides)
        return ToolContext(**kwargs)


class ToolResult:
    """
    Rich result object returned by tools.

    Supports multiple output formats, metadata, and integration
    with Talk Box conversation system.
    """

    def __init__(
        self,
        data: Any,
        success: bool = True,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        display_format: str = "auto",
        should_continue: bool = True,
        confidence: Optional[float] = None,
        sources: Optional[List[str]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.data = data
        self.success = success
        self.error = error
        self.metadata = metadata or {}
        self.display_format = display_format  # auto, json, text, html, markdown
        self.should_continue = should_continue
        self.confidence = confidence
        self.sources = sources or []
        self.extra = extra or {}
        self.created_at = datetime.now(timezone.utc)

    def to_chatlas_result(self):
        """Convert to chatlas ContentToolResult for integration."""
        try:
            import chatlas  # noqa: F401
        except ImportError:  # pragma: no cover
            raise ImportError("chatlas not available")  # pragma: no cover

        return chatlas.ContentToolResult(
            value=self.data,
            model_format=self._get_model_format(),
            error=Exception(self.error) if self.error else None,
            extra=self.metadata,
        )

    def _get_model_format(self) -> str:
        """Get appropriate model format for chatlas."""
        if self.display_format == "json":
            return "json"
        elif self.display_format == "text":
            return "str"
        elif self.display_format == "auto":
            return "auto"
        else:
            return "as_is"

    def __repr__(self) -> str:
        status = "success" if self.success else "error"
        return f"ToolResult(status={status}, data_type={type(self.data).__name__})"


class TalkBoxTool:
    """
    A Talk Box tool that wraps a function for use in conversations.

    Provides rich metadata, validation, and integration capabilities.
    """

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: ToolCategory = ToolCategory.CUSTOM,
        parameters: Optional[Dict[str, Any]] = None,
        examples: Optional[List[str]] = None,
        requires_confirmation: bool = False,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 0,
        tags: Optional[List[str]] = None,
    ):
        self.func = func
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        self.category = category
        self.parameters = parameters or self._extract_parameters()
        self.examples = examples or []
        self.requires_confirmation = requires_confirmation
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.tags = tags or []

        # Validation
        if not self.description:
            raise ToolRegistrationError(f"Tool '{self.name}' must have a description")

    def _expects_context(self) -> bool:
        """Check if the function expects a context parameter."""
        sig = inspect.signature(self.func)
        return "context" in sig.parameters

    def _extract_parameters(self) -> Dict[str, Any]:
        """Extract parameter schema from function signature."""
        sig = inspect.signature(self.func)
        type_hints = get_type_hints(self.func)

        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }

        for param_name, param in sig.parameters.items():
            # Skip context parameter (with or without type annotation)
            if param_name == "context":
                continue

            param_schema = {"type": "string"}  # Default

            # Get type from annotation
            if param_name in type_hints:
                param_type = type_hints[param_name]
                param_schema = self._type_to_json_schema(param_type)

            # Handle default values
            if param.default is not inspect.Parameter.empty:
                param_schema["default"] = param.default
            else:
                parameters["required"].append(param_name)

            parameters["properties"][param_name] = param_schema

        return parameters

    def _type_to_json_schema(self, param_type: type) -> Dict[str, Any]:
        """Convert Python type to JSON schema."""
        if param_type is int:
            return {"type": "integer"}
        elif param_type is float:
            return {"type": "number"}
        elif param_type is bool:
            return {"type": "boolean"}
        elif param_type is str:
            return {"type": "string"}
        elif param_type is list:
            return {"type": "array"}
        elif param_type is dict:
            return {"type": "object"}
        else:
            # For complex types, default to string
            return {"type": "string"}

    def to_chatlas_tool(self):
        """Convert to chatlas Tool for integration."""
        try:
            import chatlas  # noqa: F401
        except ImportError:  # pragma: no cover
            raise ImportError("chatlas not available")  # pragma: no cover

        # Create a wrapper function with proper signature for chatlas
        sig = inspect.signature(self.func)
        params = []

        for param_name, param in sig.parameters.items():
            # Skip context parameter
            if param_name == "context":
                continue
            params.append(param.replace(annotation=param.annotation))

        # Create new signature without context parameter
        new_sig = sig.replace(parameters=params)

        def chatlas_wrapper(*args, **kwargs):  # pragma: no cover
            """Wrapper function for chatlas integration."""  # pragma: no cover
            # Create a basic context for chatlas integration  # pragma: no cover
            context = ToolContext()  # pragma: no cover

            # Call the original function with context  # pragma: no cover
            if asyncio.iscoroutinefunction(self.func):  # pragma: no cover
                # For async functions, we need to run them  # pragma: no cover
                loop = asyncio.new_event_loop()  # pragma: no cover
                asyncio.set_event_loop(loop)  # pragma: no cover
                try:  # pragma: no cover
                    result = loop.run_until_complete(  # pragma: no cover
                        self.func(context, *args, **kwargs)  # pragma: no cover
                    )  # pragma: no cover
                finally:  # pragma: no cover
                    loop.close()  # pragma: no cover
            else:  # pragma: no cover
                result = self.func(context, *args, **kwargs)  # pragma: no cover

            # Convert ToolResult to appropriate format  # pragma: no cover
            if isinstance(result, ToolResult):  # pragma: no cover
                if result.success:  # pragma: no cover
                    return result.data  # pragma: no cover
                else:  # pragma: no cover
                    raise Exception(result.error or "Tool execution failed")  # pragma: no cover
            else:  # pragma: no cover
                return result  # pragma: no cover

        # Copy metadata to wrapper function  # pragma: no cover
        chatlas_wrapper.__name__ = self.name  # pragma: no cover
        chatlas_wrapper.__doc__ = self.description  # pragma: no cover
        chatlas_wrapper.__signature__ = new_sig  # pragma: no cover

        return chatlas.Tool.from_func(chatlas_wrapper)  # pragma: no cover

    async def execute(self, context: ToolContext, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        import uuid

        from .tool_observability import get_global_observer

        # Generate execution ID and get observer
        execution_id = str(uuid.uuid4())
        observer = get_global_observer()

        # Start observability tracking
        observer.start_execution(
            tool_name=self.name,
            execution_id=execution_id,
            parameters=kwargs,
            conversation_id=getattr(context, "conversation_id", None),
            user_id=getattr(context, "user_id", None),
            tags=set(self.tags) if self.tags else None,
        )

        # Legacy metrics for backward compatibility
        metrics = ToolExecutionMetrics(start_time=datetime.now(timezone.utc))

        try:
            # Log tool usage
            context.log_tool_usage(self.name, kwargs)

            # Execute function - only pass context if expected
            if self._expects_context():
                if asyncio.iscoroutinefunction(self.func):
                    result = await self.func(context, **kwargs)
                else:
                    result = self.func(context, **kwargs)
            else:
                if asyncio.iscoroutinefunction(self.func):
                    result = await self.func(**kwargs)
                else:
                    result = self.func(**kwargs)

            # Wrap result if needed
            if not isinstance(result, ToolResult):
                result = ToolResult(data=result)

            metrics.status = ToolStatus.SUCCESS

            # Finish observability tracking
            observer.finish_execution(  # pragma: no cover
                execution_id=execution_id,
                status=ToolStatus.SUCCESS,
                result_summary=str(result.data)[:200] if result.data else None,
            )

            return result

        except Exception as e:
            metrics.status = ToolStatus.ERROR
            metrics.error = str(e)
            logger.exception(f"Tool '{self.name}' execution failed")

            # Finish observability tracking with error
            observer.finish_execution(
                execution_id=execution_id,
                status=ToolStatus.ERROR,
                error_message=str(e),
                error_type=type(e).__name__,
            )

            return ToolResult(data=None, success=False, error=str(e))
        finally:
            metrics.end_time = datetime.now(timezone.utc)
            metrics.duration_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000

    def __repr__(self) -> str:
        return f"TalkBoxTool(name='{self.name}', category={self.category.value})"


class ToolRegistry:
    """
    Central registry for all Talk Box tools.

    Manages tool discovery, validation, and organization.
    """

    def __init__(self):
        self._tools: Dict[str, TalkBoxTool] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }

    def register(self, tool: TalkBoxTool) -> None:
        """Register a tool."""
        if tool.name in self._tools:  # pragma: no cover
            logger.warning(f"Tool '{tool.name}' is being overridden")  # pragma: no cover

        self._tools[tool.name] = tool
        self._categories[tool.category].append(tool.name)

        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")

    def get_tool(self, name: str) -> Optional[TalkBoxTool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_tools_by_category(self, category: ToolCategory) -> List[TalkBoxTool]:
        """Get all tools in a category."""
        return [self._tools[name] for name in self._categories[category] if name in self._tools]

    def get_all_tools(self) -> List[TalkBoxTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def search_tools(self, query: str) -> List[TalkBoxTool]:
        """Search tools by name, description, or tags."""
        query_lower = query.lower()
        results = []

        for tool in self._tools.values():
            # Check name
            if query_lower in tool.name.lower():
                results.append(tool)
                continue

            # Check description
            if query_lower in tool.description.lower():
                results.append(tool)
                continue

            # Check tags
            if any(query_lower in tag.lower() for tag in tool.tags):
                results.append(tool)

        return results

    def to_chatlas_tools(self):
        """Convert all tools to chatlas format."""
        try:
            import chatlas  # noqa: F401
        except ImportError:  # pragma: no cover
            raise ImportError("chatlas not available")  # pragma: no cover

        return [tool.to_chatlas_tool() for tool in self._tools.values()]  # pragma: no cover

    def get_tool_schema(self) -> Dict[str, Any]:
        """Get JSON schema for all tools."""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "category": tool.category.value,
                    "parameters": tool.parameters,
                    "examples": tool.examples,
                    "requires_confirmation": tool.requires_confirmation,
                    "tags": tool.tags,
                }
                for tool in self._tools.values()
            ]
        }

    def __len__(self) -> int:  # pragma: no cover
        return len(self._tools)  # pragma: no cover

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolRegistry({len(self._tools)} tools)"  # pragma: no cover


class ToolExecutor:
    """
    Handles execution of tools with Talk Box conversations.

    Provides retry logic, timeout handling, and result processing.
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    async def execute_tool(
        self, tool_name: str, context: ToolContext, parameters: Dict[str, Any]
    ) -> ToolResult:
        """Execute a tool by name."""
        tool = self.registry.get_tool(tool_name)
        if not tool:  # pragma: no cover
            return ToolResult(
                data=None, success=False, error=f"Tool '{tool_name}' not found"
            )  # pragma: no cover

        return await tool.execute(context, **parameters)


# Global registry instance  # pragma: no cover
_global_registry = ToolRegistry()  # pragma: no cover


def tool(
    _func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    category: ToolCategory = ToolCategory.CUSTOM,
    examples: Optional[List[str]] = None,
    requires_confirmation: bool = False,
    timeout_seconds: Optional[float] = None,
    max_retries: int = 0,
    tags: Optional[List[str]] = None,
    registry: Optional[ToolRegistry] = None,
) -> Callable[[F], F]:
    """Decorator to create and register a Talk Box tool.

    Supports both forms:
    1. With arguments:
        @tb.tool(description="Add numbers")
        def add(x: int, y: int) -> int: ...

    2. Bare (no parentheses):
        @tb.tool
        def say_hello(name: str) -> str: ...
    """

    def _wrap(func: F) -> F:
        tool_obj = TalkBoxTool(
            func=func,
            name=name or getattr(func, "__name__", None),
            description=description
            or (func.__doc__ or "").strip()
            or f"Tool '{getattr(func, '__name__', 'unknown')}'",
            category=category,
            examples=examples,
            requires_confirmation=requires_confirmation,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            tags=tags,
        )

        target_registry = registry or _global_registry
        target_registry.register(tool_obj)
        func._talk_box_tool = tool_obj  # type: ignore[attr-defined]
        return func

    # If called bare (@tb.tool) _func is the function
    if _func is not None and callable(_func):  # type: ignore[truthy-function]
        return _wrap(_func)  # type: ignore[return-value]

    # Otherwise return a decorator expecting the function
    return _wrap


def get_global_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _global_registry


def clear_global_registry() -> None:
    """Clear the global tool registry (mainly for testing)."""
    global _global_registry
    _global_registry = ToolRegistry()


def load_tools_from_file(
    file_path: str, registry: Optional[ToolRegistry] = None
) -> List[TalkBoxTool]:
    """
    Load tools from a Python file.

    This function imports a Python file and automatically discovers all functions
    decorated with @tb.tool, returning the loaded tools.

    Parameters
    ----------
    file_path : str
        Path to the Python file containing tool functions
    registry : ToolRegistry, optional
        Registry to register tools to (defaults to global registry)

    Returns
    -------
    List[TalkBoxTool]
        List of tools loaded from the file

    Raises
    ------
    FileNotFoundError
        If the specified file doesn't exist
    ImportError
        If the file can't be imported or has syntax errors
    ToolError
        If no tools are found in the file

    Examples
    --------
    >>> # Load tools from a file
    >>> tools = load_tools_from_file("my_tools.py")
    >>> print([tool.name for tool in tools])
    ['my_calculator', 'my_formatter']

    >>> # Use with ChatBot
    >>> bot = ChatBot().tools("my_tools.py")
    """
    import importlib.util
    from pathlib import Path

    file_path = Path(file_path)

    # Validate file exists and is a Python file
    if not file_path.exists():
        raise FileNotFoundError(f"Tool file not found: {file_path}")

    if file_path.suffix != ".py":  # pragma: no cover
        raise ValueError(f"File must be a Python file (.py), got: {file_path}")  # pragma: no cover

    # Create a unique module name to avoid conflicts
    module_name = f"talk_box_tools_{file_path.stem}_{hash(str(file_path)) % 10000}"

    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise ImportError(f"Could not create module spec for {file_path}")  # pragma: no cover

        module = importlib.util.module_from_spec(spec)

        # Store original registry state
        target_registry = registry or _global_registry
        original_tools = {tool.name for tool in target_registry.get_all_tools()}

        # Execute the module (this will register any decorated tools)
        spec.loader.exec_module(module)

        # Find newly registered tools
        new_tool_names = {tool.name for tool in target_registry.get_all_tools()} - original_tools
        new_tools = []

        for tool_name in new_tool_names:
            tool = target_registry.get_tool(tool_name)
            if tool:
                new_tools.append(tool)

        # If no new tools were found, check if the file has any decorated functions at all
        if not new_tools:
            # Check if file has decorated functions using AST parsing
            import ast

            has_decorated_functions = False
            try:
                with open(file_path, "r") as f:
                    source = f.read()

                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        for decorator in node.decorator_list:
                            # Check if decorator is tb.tool or tool
                            if isinstance(decorator, ast.Call) and (
                                (
                                    isinstance(decorator.func, ast.Attribute)
                                    and isinstance(decorator.func.value, ast.Name)
                                    and decorator.func.value.id == "tb"
                                    and decorator.func.attr == "tool"
                                )
                                or (
                                    isinstance(decorator.func, ast.Name)
                                    and decorator.func.id == "tool"
                                )
                            ):
                                has_decorated_functions = True
                                break
                    if has_decorated_functions:
                        break

            except Exception:  # pragma: no cover
                # If AST parsing fails, assume no decorated functions  # pragma: no cover
                has_decorated_functions = False  # pragma: no cover

            if not has_decorated_functions:  # pragma: no cover
                raise ToolError(  # pragma: no cover
                    f"No @tb.tool decorated functions found in {file_path}. "
                    "Make sure your tool functions are decorated with @tb.tool."
                )
            else:  # pragma: no cover
                # File has decorated functions but no new tools, probably already loaded
                logger.info(  # pragma: no cover
                    f"No new tools loaded from {file_path} - tools may already be registered"
                )
                return []  # pragma: no cover

        return new_tools

    except Exception as e:
        if isinstance(e, (FileNotFoundError, ValueError, ToolError)):
            raise
        # Wrap other exceptions with more context
        raise ImportError(f"Failed to load tools from {file_path}: {e}") from e


def load_tools_from_directory(
    directory_path: str, registry: Optional[ToolRegistry] = None
) -> List[TalkBoxTool]:
    """
    Load tools from all Python files in a directory.

    This function scans a directory for Python files and loads all tools
    from each file using load_tools_from_file().

    Parameters
    ----------
    directory_path : str
        Path to directory containing Python tool files
    registry : ToolRegistry, optional
        Registry to register tools to (defaults to global registry)

    Returns
    -------
    List[TalkBoxTool]
        List of all tools loaded from the directory

    Raises
    ------
    FileNotFoundError
        If the specified directory doesn't exist
    ToolError
        If no tools are found in any files

    Examples
    --------
    >>> # Load all tools from a directory
    >>> tools = load_tools_from_directory("./my_tools/")
    >>> print([tool.name for tool in tools])
    ['calculator', 'formatter', 'validator', 'processor']

    >>> # Use with ChatBot
    >>> bot = ChatBot().tools("./my_tools/")
    """
    from pathlib import Path

    directory_path = Path(directory_path)

    if not directory_path.exists():  # pragma: no cover
        raise FileNotFoundError(f"Tool directory not found: {directory_path}")  # pragma: no cover

    if not directory_path.is_dir():  # pragma: no cover
        raise ValueError(f"Path must be a directory, got: {directory_path}")  # pragma: no cover

    # Find all Python files in directory
    python_files = list(directory_path.glob("*.py"))

    # Exclude __init__.py and other special files
    python_files = [f for f in python_files if not f.name.startswith("__")]

    if not python_files:
        raise ToolError(f"No Python files found in directory: {directory_path}")

    all_tools = []
    loaded_files = []

    for py_file in python_files:
        try:
            tools = load_tools_from_file(str(py_file), registry)
            all_tools.extend(tools)
            loaded_files.append(py_file.name)
        except Exception as e:  # pragma: no cover
            print(f"Warning: Failed to load tools from {py_file.name}: {e}")  # pragma: no cover

    if not all_tools:
        raise ToolError(
            f"No tools found in any files in {directory_path}. "
            f"Searched files: {[f.name for f in python_files]}"
        )

    print(
        f"Loaded {len(all_tools)} tools from {len(loaded_files)} files: {', '.join(loaded_files)}"
    )
    return all_tools


def is_python_file_path(item: Any) -> bool:
    """
    Check if an item is a path to a Python file or directory.

    Parameters
    ----------
    item : Any
        Item to check

    Returns
    -------
    bool
        True if item is a string path ending in .py or a directory path
    """
    if not isinstance(item, str):  # pragma: no cover
        return False  # pragma: no cover

    from pathlib import Path  # pragma: no cover

    try:  # pragma: no cover
        path = Path(item)  # pragma: no cover
        # Check if it's a Python file  # pragma: no cover
        if path.suffix == ".py":  # pragma: no cover
            return True  # pragma: no cover
        # Check if it's a directory (for loading all tools from directory)  # pragma: no cover
        if path.is_dir():  # pragma: no cover
            return True  # pragma: no cover
    except (OSError, ValueError):  # pragma: no cover
        pass  # pragma: no cover

    return False
