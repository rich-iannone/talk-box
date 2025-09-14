import os
import pytest
import tempfile
from pathlib import Path

import talk_box as tb
from talk_box.tools import (
    load_tools_from_file,
    load_tools_from_directory,
    is_python_file_path,
    clear_global_registry,
    ToolError,
)


@pytest.fixture
def sample_tools_file(tmp_path: Path) -> Path:
    """Create a temporary sample_tools.py file with three decorated tools."""
    content = """
import talk_box as tb

@tb.tool(name="greet_customer", description="Greet a customer by name")
def greet_customer(context: tb.ToolContext, customer_name: str, time_of_day: str = "day") -> tb.ToolResult:
    greeting = f"Good {time_of_day}, {customer_name}!"
    return tb.ToolResult(data={"greeting": greeting})

@tb.tool(name="format_phone_number", description="Format a phone number")
def format_phone_number(context: tb.ToolContext, phone: str) -> tb.ToolResult:
    cleaned = ''.join(ch for ch in phone if ch.isdigit())
    return tb.ToolResult(data={"original": phone, "cleaned": cleaned})

@tb.tool(name="calculate_discount", description="Calculate discount amount")
def calculate_discount(context: tb.ToolContext, price: float, percent: float) -> tb.ToolResult:
    amount = price * (percent / 100.0)
    return tb.ToolResult(data={"discount": round(amount, 2)})
"""
    file_path = tmp_path / "sample_tools.py"
    file_path.write_text(content)
    return file_path


@pytest.fixture
def business_tools_dir(tmp_path: Path) -> Path:
    """Create a temporary directory with inventory and customer tool files."""
    dir_path = tmp_path / "business_tools"
    dir_path.mkdir()

    inventory_content = """
import talk_box as tb

@tb.tool(name="check_stock_level", description="Check stock level for a product")
def check_stock_level(context: tb.ToolContext, product_id: str) -> tb.ToolResult:
    return tb.ToolResult(data={"product_id": product_id, "available_stock": 100})

@tb.tool(name="reserve_inventory", description="Reserve inventory units")
def reserve_inventory(context: tb.ToolContext, product_id: str, quantity: int = 1) -> tb.ToolResult:
    return tb.ToolResult(data={"product_id": product_id, "reserved": quantity})

@tb.tool(name="update_stock_level", description="Update stock level for a product")
def update_stock_level(context: tb.ToolContext, product_id: str, new_level: int) -> tb.ToolResult:
    return tb.ToolResult(data={"product_id": product_id, "new_level": new_level})
"""

    customer_content = """
import talk_box as tb

@tb.tool(name="lookup_customer", description="Lookup a customer record")
def lookup_customer(context: tb.ToolContext, customer_id: str = "CUST1") -> tb.ToolResult:
    return tb.ToolResult(data={"customer_id": customer_id, "status": "active"})

@tb.tool(name="create_customer_note", description="Create a note for a customer")
def create_customer_note(context: tb.ToolContext, customer_id: str, note: str = "") -> tb.ToolResult:
    return tb.ToolResult(data={"customer_id": customer_id, "note": note or "Added."})

@tb.tool(name="calculate_customer_lifetime_value", description="Calculate CLV")
def calculate_customer_lifetime_value(context: tb.ToolContext, customer_id: str) -> tb.ToolResult:
    return tb.ToolResult(data={"customer_id": customer_id, "clv": 1234.56})
"""

    (dir_path / "inventory.py").write_text(inventory_content)
    (dir_path / "customer.py").write_text(customer_content)
    return dir_path


class TestToolFileLoading:
    """Test loading tools from dynamically created Python files."""

    def setup_method(self):
        clear_global_registry()

    def test_is_python_file_path(self, business_tools_dir):
        # Valid Python file paths
        assert is_python_file_path("tools.py")
        assert is_python_file_path("./my_tools.py")
        assert is_python_file_path("/path/to/tools.py")

        # Directory path we created
        assert is_python_file_path(str(business_tools_dir))

        # Invalid paths
        assert not is_python_file_path("tools.txt")
        assert not is_python_file_path("not_python")
        assert not is_python_file_path(123)
        assert not is_python_file_path(None)
        assert not is_python_file_path([])

    def test_load_tools_from_file(self, sample_tools_file):
        tools = load_tools_from_file(str(sample_tools_file))
        assert len(tools) == 3
        tool_names = {t.name for t in tools}
        assert {"greet_customer", "format_phone_number", "calculate_discount"} <= tool_names

    def test_load_tools_from_directory(self, business_tools_dir):
        tools = load_tools_from_directory(str(business_tools_dir))
        assert len(tools) >= 6
        tool_names = {t.name for t in tools}
        expected = {
            "check_stock_level",
            "reserve_inventory",
            "update_stock_level",
            "lookup_customer",
            "create_customer_note",
            "calculate_customer_lifetime_value",
        }
        assert expected <= tool_names

    def test_load_tools_from_nonexistent_file(self):
        """Test error handling for non-existent files."""
        with pytest.raises(FileNotFoundError):
            load_tools_from_file("nonexistent_file.py")

    def test_load_tools_from_invalid_file(self):
        """Test error handling for non-Python files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is not a Python file")
            temp_file.flush()

            try:
                with pytest.raises(ValueError):
                    load_tools_from_file(temp_file.name)
            finally:
                os.unlink(temp_file.name)

    def test_load_tools_from_file_no_tools(self):
        """Test error when Python file contains no decorated tools."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write("""
# This Python file has no @tool decorated functions
def regular_function():
    return "not a tool"
            """)
            temp_file.flush()

            try:
                with pytest.raises(ToolError):
                    load_tools_from_file(temp_file.name)
            finally:
                os.unlink(temp_file.name)

    def test_load_tools_from_file_with_syntax_error(self):
        """Test error handling for files with syntax errors."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as temp_file:
            temp_file.write("""
# This file has a syntax error
def broken_function(
    return "syntax error"
            """)
            temp_file.flush()

            try:
                with pytest.raises(ImportError):
                    load_tools_from_file(temp_file.name)
            finally:
                os.unlink(temp_file.name)


class TestChatBotFileIntegration:
    """Test ChatBot integration with dynamically created tool files."""

    def setup_method(self):
        clear_global_registry()

    def test_chatbot_tools_single_file(self, sample_tools_file):
        bot = tb.ChatBot().tools(str(sample_tools_file))
        tools = bot.get_config().get("tools", [])
        for name in ["greet_customer", "format_phone_number", "calculate_discount"]:
            assert name in tools

    def test_chatbot_tools_directory(self, business_tools_dir):
        bot = tb.ChatBot().tools(str(business_tools_dir))
        tools = bot.get_config().get("tools", [])
        for name in [
            "check_stock_level",
            "reserve_inventory",
            "update_stock_level",
            "lookup_customer",
            "create_customer_note",
            "calculate_customer_lifetime_value",
        ]:
            assert name in tools

    def test_chatbot_tools_mixed_list(self, sample_tools_file):
        bot = tb.ChatBot().tools(["calculate", str(sample_tools_file), "validate_email"])
        tools = bot.get_config().get("tools", [])
        assert {
            "calculate",
            "validate_email",
            "greet_customer",
            "format_phone_number",
            "calculate_discount",
        } <= set(tools)

    def test_chatbot_tools_invalid_file_path(self):
        """Test ChatBot.tools() with invalid file path (should warn but not crash)."""
        bot = tb.ChatBot().tools("nonexistent_file.py")

        # Should not crash, but tools list should be empty
        config = bot.get_config()
        tools = config.get("tools", [])
        assert len(tools) == 0

    def test_chatbot_tools_file_path_in_list(self, sample_tools_file):
        bot = tb.ChatBot().tools(
            [
                "calculate",
                str(sample_tools_file),
                "current_time",
            ]
        )
        tools = bot.get_config().get("tools", [])
        assert {"calculate", "current_time", "greet_customer", "format_phone_number"} <= set(tools)

    def test_chatbot_tools_chaining_with_files(self, sample_tools_file, business_tools_dir):
        bot = (
            tb.ChatBot()
            .tools(str(sample_tools_file))
            .tools(["calculate", "validate_email"])  # Built-ins
            .tools(str(business_tools_dir))
        )
        tools = set(bot.get_config().get("tools", []))
        assert {"greet_customer", "calculate", "validate_email", "check_stock_level"} <= tools


class TestToolFileExecutionIntegration:
    """Test that dynamically loaded tools actually work."""

    def setup_method(self):
        clear_global_registry()

    @pytest.mark.asyncio
    async def test_execute_loaded_file_tools(self, sample_tools_file):
        tools = load_tools_from_file(str(sample_tools_file))
        greet_tool = next(t for t in tools if t.name == "greet_customer")
        from talk_box import ToolContext

        context = ToolContext()
        result = await greet_tool.execute(context, customer_name="Alice", time_of_day="morning")
        assert result.success
        assert "Alice" in result.data["greeting"]
        assert "Good morning" in result.data["greeting"]

    @pytest.mark.asyncio
    async def test_directory_tools_execution(self, business_tools_dir):
        tools = load_tools_from_directory(str(business_tools_dir))
        stock_tool = next(t for t in tools if t.name == "check_stock_level")
        from talk_box import ToolContext

        context = ToolContext()
        result = await stock_tool.execute(context, product_id="PROD001")
        assert result.success
        assert result.data["product_id"] == "PROD001"
        assert "available_stock" in result.data
