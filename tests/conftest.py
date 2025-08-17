"""Test configuration and fixtures for Talk Box tests."""

import pytest

from talk_box import ChatBot, PresetManager


@pytest.fixture
def sample_chatbot():
    """Create a sample ChatBot instance for testing."""
    return ChatBot().model("gpt-3.5-turbo").temperature(0.5)


@pytest.fixture
def preset_manager():
    """Create a PresetManager instance for testing."""
    return PresetManager()
