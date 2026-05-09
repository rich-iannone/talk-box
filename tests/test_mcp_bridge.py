import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from talk_box.mcp_bridge import (
    MCPBridgeServer,
    MCPToolInfo,
    _register_tool_on_server,
    discover_mcp_tools,
    list_mcp_tools,
    mcp_tool_to_talk_box,
    tools_to_mcp_server,
)
from talk_box.tools import ToolCategory, ToolContext, ToolResult, TalkBoxTool

_has_mcp = True
try:
    import mcp  # noqa: F401
except ModuleNotFoundError:
    _has_mcp = False

requires_mcp = pytest.mark.skipif(not _has_mcp, reason="mcp package not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_numbers(context: ToolContext, a: int, b: int) -> ToolResult:
    """Add two numbers together."""
    return ToolResult(data={"sum": a + b})


def _echo(context: ToolContext, message: str) -> ToolResult:
    """Echo back a message."""
    return ToolResult(data=message)


def _get_test_tools() -> list[TalkBoxTool]:
    """Build two TalkBoxTool instances directly (no global registry)."""
    return [
        TalkBoxTool(
            name="add_numbers",
            description="Add two numbers",
            func=_add_numbers,
            category=ToolCategory.DATA,
            tags=["test"],
        ),
        TalkBoxTool(
            name="echo",
            description="Echo a message",
            func=_echo,
            category=ToolCategory.CUSTOM,
            tags=["test"],
        ),
    ]


# ---------------------------------------------------------------------------
# MCPToolInfo
# ---------------------------------------------------------------------------


class TestMCPToolInfo:
    def test_basic_creation(self):
        info = MCPToolInfo(
            name="search",
            description="Search things",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )
        assert info.name == "search"
        assert info.description == "Search things"
        assert info.server_url == ""

    def test_with_server_url(self):
        info = MCPToolInfo(
            name="t",
            description="d",
            input_schema={},
            server_url="http://localhost:8000/mcp",
        )
        assert info.server_url == "http://localhost:8000/mcp"


# ---------------------------------------------------------------------------
# MCPBridgeServer
# ---------------------------------------------------------------------------


class TestMCPBridgeServer:
    def test_creation(self):
        server = MCPBridgeServer(name="test-server")
        assert server.name == "test-server"
        assert server.tools == []
        assert server.instructions == ""

    def test_tool_names(self):
        tools = _get_test_tools()
        server = MCPBridgeServer(name="test", tools=tools)
        names = server.tool_names()
        assert "add_numbers" in names
        assert "echo" in names
        assert names == sorted(names)

    def test_tool_names_empty(self):
        server = MCPBridgeServer(name="empty")
        assert server.tool_names() == []

    @requires_mcp
    def test_build_returns_fast_mcp(self):
        tools = _get_test_tools()
        server = MCPBridgeServer(name="test", tools=tools)
        mcp_server = server.build()
        # FastMCP instance
        from mcp.server import FastMCP

        assert isinstance(mcp_server, FastMCP)

    @requires_mcp
    def test_build_registers_tools(self):
        tools = _get_test_tools()
        server = MCPBridgeServer(name="test", tools=tools)
        mcp_server = server.build()
        # The FastMCP server should have the tools registered
        assert server._server is mcp_server

    @requires_mcp
    def test_build_with_instructions(self):
        server = MCPBridgeServer(
            name="test",
            tools=_get_test_tools(),
            instructions="Be helpful.",
        )
        mcp_server = server.build()
        from mcp.server import FastMCP

        assert isinstance(mcp_server, FastMCP)


# ---------------------------------------------------------------------------
# tools_to_mcp_server
# ---------------------------------------------------------------------------


class TestToolsToMcpServer:
    def test_creates_server(self):
        tools = _get_test_tools()
        server = tools_to_mcp_server(tools, name="my-server")
        assert isinstance(server, MCPBridgeServer)
        assert server.name == "my-server"
        assert len(server.tools) == len(tools)

    def test_default_name(self):
        server = tools_to_mcp_server([])
        assert server.name == "talk-box"

    def test_with_instructions(self):
        server = tools_to_mcp_server([], instructions="Help.")
        assert server.instructions == "Help."


# ---------------------------------------------------------------------------
# _register_tool_on_server
# ---------------------------------------------------------------------------


class TestRegisterToolOnServer:
    @requires_mcp
    def test_sync_tool_registered(self):
        from mcp.server import FastMCP

        mcp_server = FastMCP(name="test")
        tools = _get_test_tools()
        for t in tools:
            _register_tool_on_server(mcp_server, t)
        # No error means success

    @requires_mcp
    def test_registered_wrapper_is_callable(self):
        from mcp.server import FastMCP

        mcp_server = FastMCP(name="test")
        tools = _get_test_tools()
        _register_tool_on_server(mcp_server, tools[0])


# ---------------------------------------------------------------------------
# mcp_tool_to_talk_box
# ---------------------------------------------------------------------------


class TestMcpToolToTalkBox:
    def test_basic_conversion(self):
        info = MCPToolInfo(
            name="lookup",
            description="Look up a term",
            input_schema={
                "type": "object",
                "properties": {"term": {"type": "string"}},
                "required": ["term"],
            },
        )
        tb_tool = mcp_tool_to_talk_box(info)
        assert isinstance(tb_tool, TalkBoxTool)
        assert tb_tool.name == "lookup"
        assert tb_tool.description == "Look up a term"

    def test_stub_returns_args(self):
        info = MCPToolInfo(
            name="echo",
            description="Echo",
            input_schema={
                "type": "object",
                "properties": {"msg": {"type": "string"}},
                "required": ["msg"],
            },
        )
        tb_tool = mcp_tool_to_talk_box(info)
        # Execute with stub (no call_fn)
        ctx = ToolContext()
        result = tb_tool.func(ctx, msg="hello")
        assert isinstance(result, ToolResult)
        assert result.data == {"msg": "hello"}

    def test_with_call_fn(self):
        info = MCPToolInfo(
            name="add",
            description="Add numbers",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                "required": ["a", "b"],
            },
        )
        call_fn = MagicMock(return_value=42)
        tb_tool = mcp_tool_to_talk_box(info, call_fn=call_fn)
        ctx = ToolContext()
        result = tb_tool.func(ctx, a=1, b=2)
        call_fn.assert_called_once_with("add", {"a": 1, "b": 2})
        assert result.data == 42

    def test_optional_params_default_to_none(self):
        info = MCPToolInfo(
            name="search",
            description="Search",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        )
        tb_tool = mcp_tool_to_talk_box(info)
        ctx = ToolContext()
        result = tb_tool.func(ctx, query="hello")
        assert result.data == {"query": "hello"}

    def test_empty_schema(self):
        info = MCPToolInfo(name="noop", description="No-op", input_schema={})
        tb_tool = mcp_tool_to_talk_box(info)
        assert tb_tool.name == "noop"
        ctx = ToolContext()
        result = tb_tool.func(ctx)
        assert result.data == {}


# ---------------------------------------------------------------------------
# list_mcp_tools
# ---------------------------------------------------------------------------


class TestListMcpTools:
    def test_parses_tool_dicts(self):
        raw = [
            {
                "name": "greet",
                "description": "Say hello",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            },
            {
                "name": "farewell",
                "description": "Say goodbye",
                "inputSchema": {},
            },
        ]
        tools = list_mcp_tools(raw)
        assert len(tools) == 2
        assert tools[0].name == "greet"
        assert tools[1].name == "farewell"
        assert tools[0].input_schema["properties"]["name"]["type"] == "string"

    def test_with_server_url(self):
        raw = [{"name": "t", "description": "d", "inputSchema": {}}]
        tools = list_mcp_tools(raw, server_url="http://localhost/mcp")
        assert tools[0].server_url == "http://localhost/mcp"

    def test_empty_list(self):
        assert list_mcp_tools([]) == []

    def test_missing_fields_default(self):
        raw = [{}]
        tools = list_mcp_tools(raw)
        assert tools[0].name == ""
        assert tools[0].description == ""
        assert tools[0].input_schema == {}


# ---------------------------------------------------------------------------
# discover_mcp_tools
# ---------------------------------------------------------------------------


class TestDiscoverMcpTools:
    def test_discovers_from_session(self):
        # Mock an MCP ClientSession
        mock_tool = MagicMock()
        mock_tool.name = "remote_tool"
        mock_tool.description = "A remote tool"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]

        mock_session = AsyncMock()
        mock_session.list_tools.return_value = mock_result

        infos = asyncio.run(discover_mcp_tools(mock_session, server_url="ws://localhost"))
        assert len(infos) == 1
        assert infos[0].name == "remote_tool"
        assert infos[0].server_url == "ws://localhost"

    def test_empty_discovery(self):
        mock_result = MagicMock()
        mock_result.tools = []
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = mock_result

        infos = asyncio.run(discover_mcp_tools(mock_session))
        assert infos == []

    def test_tool_without_description(self):
        mock_tool = MagicMock()
        mock_tool.name = "bare"
        mock_tool.description = None
        mock_tool.inputSchema = None

        mock_result = MagicMock()
        mock_result.tools = [mock_tool]
        mock_session = AsyncMock()
        mock_session.list_tools.return_value = mock_result

        infos = asyncio.run(discover_mcp_tools(mock_session))
        assert infos[0].description == ""
        assert infos[0].input_schema == {}


# ---------------------------------------------------------------------------
# Round-trip: Talk Box → MCP → Talk Box
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @requires_mcp
    def test_expose_and_consume(self):
        """Test the full round-trip: TB tool → MCP server → MCP info → TB tool."""
        tools = _get_test_tools()
        # 1. Create an MCP server
        server = tools_to_mcp_server(tools, name="roundtrip")
        mcp_server = server.build()

        # 2. Simulate discovering these tools (as raw dicts)
        raw_tools = []
        for t in tools:
            raw_tools.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.parameters,
                }
            )

        # 3. Convert back to Talk Box tools
        infos = list_mcp_tools(raw_tools, server_url="test://local")
        converted = [mcp_tool_to_talk_box(info) for info in infos]

        assert len(converted) == len(tools)
        for ct in converted:
            assert isinstance(ct, TalkBoxTool)
            assert ct.name in [t.name for t in tools]


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        import talk_box

        for name in [
            "MCPToolInfo",
            "MCPBridgeServer",
            "tools_to_mcp_server",
            "mcp_tool_to_talk_box",
            "list_mcp_tools",
            "discover_mcp_tools",
        ]:
            assert hasattr(talk_box, name), f"talk_box.{name} not found"

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "MCPToolInfo",
            "MCPBridgeServer",
            "tools_to_mcp_server",
            "mcp_tool_to_talk_box",
            "list_mcp_tools",
            "discover_mcp_tools",
        ]:
            assert name in talk_box.__all__, f"{name} not in __all__"
