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
from talk_box.conversation import Conversation, Message
from talk_box.pathways import Pathways
from talk_box.presets import Preset, PresetManager, PresetNames
from talk_box.prompt_builder import (
    Priority,
    PromptBuilder,
    PromptSection,
    architectural_analysis_prompt,
    code_review_prompt,
    debugging_prompt,
)

# Testing functions for easy access
from talk_box.testing import (
    PathwayTestResults,
    TestResults,
    autotest_avoid_topics,
    autotest_pathways,
)

# Make key classes available at package level
__all__ = [
    # Core classes
    "ChatBot",
    "Conversation",
    "Message",
    # File attachments
    "Attachments",
    "AttachmentMetadata",
    # Conversational pathways
    "Pathways",
    # Prompt engineering
    "PromptBuilder",
    "Priority",
    "PromptSection",
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
]
