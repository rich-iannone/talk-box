try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version

try:  # pragma: no cover
    __version__ = version("talk-box")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__author__ = "Richard Iannone"
__email__ = "riannone@me.com"

# Core imports for easy access
from talk_box.attachments import AttachmentMetadata, Attachments
from talk_box.builder import BuilderTypes, ChatBot
from talk_box.builtin_tools import get_builtin_tool, load_selected_tools, load_tool_box
from talk_box.conversation import (
    Conversation,
    Message,
    ToolEnabledConversation,
    create_tool_conversation,
)
from talk_box.pathways import Pathways
from talk_box.presets import Preset, PresetManager, PresetNames
from talk_box.prompt_builder import (
    Priority,
    PromptBuilder,
    PromptSection,
    VocabularyTerm,
    architectural_analysis_prompt,
    code_review_prompt,
    debugging_prompt,
)
from talk_box.testing import (
    PathwayTestResults,
    TestResults,
    autotest_avoid_topics,
    autotest_pathways,
)
from talk_box.tool_debugging import (
    ToolDebugger,
    configure_debug_mode,
    debug_dashboard,
    debug_errors,
    debug_tool,
    export_debug_report,
    live_monitor,
)
from talk_box.tool_observability import (
    ObservabilityLevel,
    ToolObserver,
    configure_observability,
    get_global_observer,
)
from talk_box.tools import (
    ToolCategory,
    ToolContext,
    ToolResult,
    get_global_registry,
    tool,
)

# Make key classes available at package level
__all__ = [
    # Core classes
    "ChatBot",
    "Conversation",
    "Message",
    "ToolEnabledConversation",
    "create_tool_conversation",
    # File attachments
    "Attachments",
    "AttachmentMetadata",
    # Conversational pathways
    "Pathways",
    # Prompt engineering
    "PromptBuilder",
    "Priority",
    "PromptSection",
    "VocabularyTerm",
    "architectural_analysis_prompt",
    "code_review_prompt",
    "debugging_prompt",
    # Preset management
    "Preset",
    "PresetManager",
    "PresetNames",
    # Builder types
    "BuilderTypes",
    # Testing functions
    "autotest_avoid_topics",
    "autotest_pathways",
    # Testing classes
    "TestResults",
    "PathwayTestResults",
    # Tool Box
    "tool",
    "ToolCategory",
    "ToolContext",
    "ToolResult",
    "get_global_registry",
    "load_tool_box",
    "get_builtin_tool",
    "load_selected_tools",
    # Tool Observability
    "ObservabilityLevel",
    "ToolObserver",
    "configure_observability",
    "get_global_observer",
    # Tool Debugging
    "ToolDebugger",
    "configure_debug_mode",
    "debug_dashboard",
    "debug_errors",
    "debug_tool",
    "export_debug_report",
    "live_monitor",
]
