"""Tests for the ChatBot builder functionality."""

import pytest

from talk_box import ChatBot


class TestChatBot:
    """Test cases for the ChatBot class."""

    def test_chatbot_creation(self):
        """Test basic ChatBot creation."""
        bot = ChatBot()
        assert bot is not None
        config = bot.get_config()
        assert config["model"] == "gpt-3.5-turbo"
        assert config["temperature"] == 0.7

    def test_method_chaining(self):
        """Test that method chaining works properly."""
        bot = (
            ChatBot()
            .model("gpt-4-turbo")
            .temperature(0.2)
            .max_tokens(500)
            .verbose(True)
        )

        config = bot.get_config()
        assert config["model"] == "gpt-4-turbo"
        assert config["temperature"] == 0.2
        assert config["max_tokens"] == 500
        assert config["verbose"] is True

    def test_temperature_validation(self):
        """Test temperature validation."""
        bot = ChatBot()

        # Valid temperatures should work
        bot.temperature(0.0)
        bot.temperature(1.0)
        bot.temperature(2.0)

        # Invalid temperatures should raise ValueError
        with pytest.raises(ValueError):
            bot.temperature(-0.1)

        with pytest.raises(ValueError):
            bot.temperature(2.1)

    def test_max_tokens_validation(self):
        """Test max_tokens validation."""
        bot = ChatBot()

        # Valid token counts should work
        bot.max_tokens(1)
        bot.max_tokens(1000)

        # Invalid token counts should raise ValueError
        with pytest.raises(ValueError):
            bot.max_tokens(0)

        with pytest.raises(ValueError):
            bot.max_tokens(-1)

    def test_tools_configuration(self):
        """Test tools configuration."""
        bot = ChatBot()
        tools = ["web_search", "calculator"]
        bot.tools(tools)

        config = bot.get_config()
        assert config["tools"] == tools

        # Ensure it's a copy, not the same list
        tools.append("new_tool")
        assert "new_tool" not in config["tools"]

    def test_avoid_configuration(self):
        """Test avoid configuration."""
        bot = ChatBot()
        avoid_list = ["politics", "medical_advice"]
        bot.avoid(avoid_list)

        config = bot.get_config()
        assert config["avoid"] == avoid_list

        # Ensure it's a copy, not the same list
        avoid_list.append("financial_advice")
        assert "financial_advice" not in config["avoid"]

    def test_persona_configuration(self):
        """Test persona configuration."""
        bot = ChatBot()
        persona = "Senior Data Scientist"
        bot.persona(persona)

        config = bot.get_config()
        assert config["persona"] == persona

    def test_preset_configuration(self):
        """Test preset configuration."""
        bot = ChatBot()
        preset_name = "technical_advisor"
        bot.preset(preset_name)

        config = bot.get_config()
        assert config["preset"] == preset_name

    def test_chat_basic_functionality(self):
        """Test basic chat functionality (echo for now)."""
        bot = ChatBot()
        conversation = bot.chat("Hello, world!")

        assert conversation is not None
        assert hasattr(conversation, "get_last_message")

        # Get the last message (assistant response)
        last_message = conversation.get_last_message()
        assert last_message is not None
        assert "Hello, world!" in last_message.content
        assert last_message.role == "assistant"
