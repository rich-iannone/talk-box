"""
Talk Box - The best way to generate, test, and deploy LLM chatbots.

A Python framework designed for simplicity and extensibility, enabling developers
to create, test, and deploy LLM chatbots with a chainable API, built-in tools,
behavior presets, and comprehensive testing capabilities.
"""

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
    TestResults,
    autotest_avoid_topics,
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
    # Testing classes
    "TestResults",
]
