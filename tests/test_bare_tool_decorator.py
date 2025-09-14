import re
import uuid
import talk_box as tb


def test_bare_tool_decorator_registration_and_display():
    unique_suffix = uuid.uuid4().hex[:8]
    tool_name = f"say_hello_{unique_suffix}"

    @tb.tool
    def say_hello(name: str) -> str:  # type: ignore
        """Simple tool that says hello."""
        return f"Hello, {name}!"

    # Rename after decoration if needed (bare decorator uses original __name__)
    # For isolation we rely on unique function name via suffix variable above.
    say_hello.__name__ = (
        tool_name  # Adjust __name__ for readability (registry keeps original at decoration time)
    )

    bot = tb.ChatBot().tools([say_hello]).model("gpt-4")
    registry = tb.get_global_registry()

    # Registry stores the original name at decoration time (before reassignment); capture both possibilities
    registered = any(t.name.startswith("say_hello") for t in registry.get_all_tools())
    assert registered, "Bare-decorated tool not registered"

    html = bot._repr_html_()
    assert "🛠️ Tools" in html, "Tools section missing in HTML repr"
    assert re.search(r"say_hello", html), "Tool name not displayed in HTML"
    assert re.search(r"Tools \(1 enabled\)", html), "Incorrect tool count displayed"
