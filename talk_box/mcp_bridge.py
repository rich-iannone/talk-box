"""MCP bridge: expose Talk Box tools as MCP servers, consume MCP tools as Talk Box tools."""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Lazy MCP imports — mcp is optional
# ---------------------------------------------------------------------------


def _require_mcp():
    """Import and return the mcp package, raising a clear error if missing."""
    try:
        import mcp  # noqa: F401
        import mcp.server  # noqa: F401
        import mcp.types  # noqa: F401

        return mcp
    except ImportError:
        raise ImportError(
            "The 'mcp' package is required for MCP bridge features. "
            "Install it with: pip install mcp"
        )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class MCPToolInfo:
    """Description of an MCP tool discovered from a remote server.

    Parameters
    ----------
    name
        The tool name as advertised by the MCP server.
    description
        Human-readable description of what the tool does.
    input_schema
        JSON Schema dict describing the tool's parameters.
    server_url
        The URL or transport identifier of the MCP server.

    Examples
    --------
    ```python
    import talk_box as tb

    info = tb.MCPToolInfo(
        name="search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        server_url="http://localhost:8000/mcp",
    )
    info.name  # "search"
    ```
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    server_url: str = ""


@dataclass
class MCPBridgeServer:
    """An MCP server that exposes Talk Box tools over the MCP protocol.

    Wraps a collection of ``TalkBoxTool`` instances and serves them
    via MCP's ``FastMCP`` server. Supports stdio, SSE, and
    streamable-HTTP transports.

    Parameters
    ----------
    name
        Server name shown to MCP clients.
    tools
        List of ``TalkBoxTool`` instances to expose.
    instructions
        Optional instructions string for the MCP server.

    Examples
    --------
    Create a server from a tool registry:

    ```python
    import talk_box as tb
    from talk_box.tools import get_global_registry

    registry = get_global_registry()
    server = tb.MCPBridgeServer(
        name="talk-box-tools",
        tools=registry.get_all_tools(),
    )
    server.tool_names()  # ["text_stats", "convert_case", ...]
    ```
    """

    name: str
    tools: list[Any] = field(default_factory=list)
    instructions: str = ""

    _server: Any = field(default=None, init=False, repr=False)

    def tool_names(self) -> list[str]:
        """Return the names of all tools registered with this server.

        Returns
        -------
        list[str]
            Sorted list of tool names.
        """
        return sorted(t.name for t in self.tools)

    def build(self) -> Any:
        """Build and return a ``FastMCP`` server instance.

        Each Talk Box tool is registered as an MCP tool. The tool's
        parameters (excluding the ``context`` parameter) are preserved
        so MCP clients see the correct input schema.

        Returns
        -------
        FastMCP
            A configured ``FastMCP`` server ready to be started with
            ``run()``.

        Examples
        --------
        ```python
        server = tb.MCPBridgeServer(name="my-tools", tools=[my_tool])
        mcp_server = server.build()
        # mcp_server.run()  # starts the server
        ```
        """
        _require_mcp()
        from mcp.server import FastMCP

        mcp_server = FastMCP(
            name=self.name,
            instructions=self.instructions or None,
        )

        for tb_tool in self.tools:
            _register_tool_on_server(mcp_server, tb_tool)

        self._server = mcp_server
        return mcp_server

    def run(self, transport: str = "stdio") -> None:
        """Build the server (if needed) and run it.

        Parameters
        ----------
        transport
            Transport mode: ``"stdio"``, ``"sse"``, or ``"streamable-http"``.
        """
        if self._server is None:
            self.build()
        self._server.run(transport=transport)  # pragma: no cover


# ---------------------------------------------------------------------------
# Expose helpers
# ---------------------------------------------------------------------------


def _register_tool_on_server(mcp_server: Any, tb_tool: Any) -> None:
    """Register a single TalkBoxTool on a FastMCP server."""
    from talk_box.tools import ToolContext, ToolResult

    func = tb_tool.func
    sig = inspect.signature(func)

    # Build new params without 'context'
    new_params = [p for name, p in sig.parameters.items() if name != "context"]
    new_sig = sig.replace(parameters=new_params)

    if asyncio.iscoroutinefunction(func):

        async def wrapper(*args: Any, **kwargs: Any) -> str:
            ctx = ToolContext()
            result = await func(ctx, *args, **kwargs)
            if isinstance(result, ToolResult):
                return json.dumps(result.data) if result.success else (result.error or "error")
            return str(result)
    else:

        def wrapper(*args: Any, **kwargs: Any) -> str:
            ctx = ToolContext()
            result = func(ctx, *args, **kwargs)
            if isinstance(result, ToolResult):
                return json.dumps(result.data) if result.success else (result.error or "error")
            return str(result)

    wrapper.__name__ = tb_tool.name
    wrapper.__doc__ = tb_tool.description
    wrapper.__signature__ = new_sig  # type: ignore[attr-defined]

    mcp_server.tool(
        name=tb_tool.name,
        description=tb_tool.description,
    )(wrapper)


def tools_to_mcp_server(
    tools: list[Any],
    *,
    name: str = "talk-box",
    instructions: str = "",
) -> MCPBridgeServer:
    """Create an ``MCPBridgeServer`` from a list of Talk Box tools.

    This is a convenience factory that builds and returns the server in
    one call.

    Parameters
    ----------
    tools
        Talk Box tool instances (``TalkBoxTool``).
    name
        Server name.
    instructions
        Optional server instructions.

    Returns
    -------
    MCPBridgeServer
        A server ready to ``build()`` and ``run()``.

    Examples
    --------
    ```python
    import talk_box as tb
    from talk_box.tools import get_global_registry

    server = tb.tools_to_mcp_server(
        get_global_registry().get_all_tools(),
        name="my-tools",
    )
    mcp = server.build()
    ```
    """
    return MCPBridgeServer(name=name, tools=tools, instructions=instructions)


# ---------------------------------------------------------------------------
# Consume helpers
# ---------------------------------------------------------------------------


def mcp_tool_to_talk_box(info: MCPToolInfo, *, call_fn: Callable[..., Any] | None = None) -> Any:
    """Convert an ``MCPToolInfo`` into a ``TalkBoxTool``.

    This wraps a remote MCP tool so it can be used inside Talk Box
    conversations just like a native tool.

    Parameters
    ----------
    info
        The MCP tool descriptor (from ``list_mcp_tools``).
    call_fn
        An optional callable ``(name, arguments) -> result`` used to
        invoke the remote tool.  When omitted a stub that returns the
        arguments as a JSON string is created (useful for testing).

    Returns
    -------
    TalkBoxTool
        A Talk Box tool that delegates to the MCP server.

    Examples
    --------
    ```python
    import talk_box as tb

    info = tb.MCPToolInfo(
        name="lookup",
        description="Look up a term",
        input_schema={
            "type": "object",
            "properties": {"term": {"type": "string"}},
            "required": ["term"],
        },
    )
    tool = tb.mcp_tool_to_talk_box(info)
    tool.name  # "lookup"
    ```
    """
    from talk_box.tools import TalkBoxTool, ToolContext, ToolResult

    properties = info.input_schema.get("properties", {})
    required = set(info.input_schema.get("required", []))

    # Build parameter list for the wrapper function
    params: list[inspect.Parameter] = [
        inspect.Parameter("context", inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]
    for pname, pschema in properties.items():
        default = pschema.get("default", inspect.Parameter.empty)
        if pname not in required and default is inspect.Parameter.empty:
            default = None
        params.append(
            inspect.Parameter(
                pname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
            )
        )

    actual_call = call_fn

    def _tool_func(context: ToolContext, **kwargs: Any) -> ToolResult:
        if actual_call is not None:
            result = actual_call(info.name, kwargs)
            return ToolResult(data=result)
        return ToolResult(data=kwargs)

    _tool_func.__name__ = info.name
    _tool_func.__doc__ = info.description
    _tool_func.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]

    # Ensure parameters is truthy so TalkBoxTool doesn't try to introspect
    params_schema = info.input_schema or {
        "type": "object",
        "properties": {},
        "required": [],
    }

    return TalkBoxTool(
        func=_tool_func,
        name=info.name,
        description=info.description,
        parameters=params_schema,
    )


def list_mcp_tools(tools_data: list[dict[str, Any]], *, server_url: str = "") -> list[MCPToolInfo]:
    """Convert raw MCP tool dicts into ``MCPToolInfo`` objects.

    This is a synchronous helper for when you already have the tool list
    (e.g. from a cached ``list_tools()`` response).  For live discovery,
    use ``discover_mcp_tools()``.

    Parameters
    ----------
    tools_data
        A list of dicts, each with ``name``, ``description``, and
        ``inputSchema`` keys (matching ``mcp.types.Tool``).
    server_url
        Optional server URL to attach to each tool info.

    Returns
    -------
    list[MCPToolInfo]
        Parsed tool descriptors.

    Examples
    --------
    ```python
    import talk_box as tb

    raw = [
        {
            "name": "greet",
            "description": "Say hello",
            "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}},
        }
    ]
    tools = tb.list_mcp_tools(raw)
    tools[0].name  # "greet"
    ```
    """
    result = []
    for entry in tools_data:
        result.append(
            MCPToolInfo(
                name=entry.get("name", ""),
                description=entry.get("description", ""),
                input_schema=entry.get("inputSchema", {}),
                server_url=server_url,
            )
        )
    return result


async def discover_mcp_tools(
    session: Any,
    *,
    server_url: str = "",
) -> list[MCPToolInfo]:
    """Discover tools from a live MCP client session.

    Calls ``session.list_tools()`` and converts the results into
    ``MCPToolInfo`` objects.

    Parameters
    ----------
    session
        An ``mcp.client.session.ClientSession`` instance.
    server_url
        Optional URL to attach to each tool info.

    Returns
    -------
    list[MCPToolInfo]
        Discovered tool descriptors.
    """
    result = await session.list_tools()
    infos = []
    for tool in result.tools:
        infos.append(
            MCPToolInfo(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema if tool.inputSchema else {},
                server_url=server_url,
            )
        )
    return infos
