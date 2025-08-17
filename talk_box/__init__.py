"""
Talk Box - The best way to generate, test, and deploy LLM chatbots.

A Python framework designed for simplicity and extensibility, enabling developers
to create, test, and deploy LLM chatbots with a chainable API, built-in tools,
behavior presets, and comprehensive testing capabilities.
"""

__version__ = "0.1.0"
__author__ = "Richard Iannone"
__email__ = "riannone@me.com"

# Core imports for easy access
from talk_box.builder import ChatBot
from talk_box.conversation import Conversation, Message
from talk_box.presets import Preset, PresetManager

# Make key classes available at package level
__all__ = [
    "ChatBot",
    "Conversation",
    "Message",
    "Preset",
    "PresetManager",
    "__version__",
]
