import pytest
import socket
from unittest.mock import patch, MagicMock
import importlib.util

from talk_box import ChatBot
from talk_box.builder import find_available_port


# Core ChatBot functionality tests
def test_chatbot_creation():
    """Test basic ChatBot creation."""
    bot = ChatBot()
    assert bot is not None
    config = bot.get_config()
    assert config["model"] == "gpt-3.5-turbo"
    assert config["temperature"] == 0.7


def test_method_chaining():
    """Test that method chaining works properly."""
    bot = ChatBot().model("gpt-4-turbo").temperature(0.2).max_tokens(500).verbose(True)

    config = bot.get_config()
    assert config["model"] == "gpt-4-turbo"
    assert config["temperature"] == 0.2
    assert config["max_tokens"] == 500
    assert config["verbose"] is True


def test_temperature_validation():
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


def test_max_tokens_validation():
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


def test_tools_configuration():
    """Test tools configuration."""
    bot = ChatBot()
    tools = ["web_search", "calculator"]
    bot.tools(tools)

    config = bot.get_config()
    assert config["tools"] == tools

    # Ensure it's a copy, not the same list
    tools.append("new_tool")
    assert "new_tool" not in config["tools"]


def test_avoid_configuration():
    """Test avoid configuration."""
    bot = ChatBot()
    avoid_list = ["politics", "medical_advice"]
    bot.avoid(avoid_list)

    config = bot.get_config()
    assert config["avoid"] == avoid_list

    # Ensure it's a copy, not the same list
    avoid_list.append("financial_advice")
    assert "financial_advice" not in config["avoid"]


def test_persona_configuration():
    """Test persona configuration."""
    bot = ChatBot()
    persona = "Senior Data Scientist"
    bot.persona(persona)

    config = bot.get_config()
    assert config["persona"] == persona


def test_preset_configuration():
    """Test preset configuration."""
    bot = ChatBot()
    preset_name = "technical_advisor"
    bot.preset(preset_name)

    config = bot.get_config()
    assert config["preset"] == preset_name


def test_chat_basic_functionality():
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


# Utility function tests
def test_find_available_port_success():
    """Test finding an available port successfully."""
    # Mock socket to simulate finding a port on the second attempt
    with patch("socket.socket") as mock_socket:
        mock_socket_instance = MagicMock()
        mock_socket.return_value.__enter__.return_value = mock_socket_instance

        # First port (9000) fails, second port (9001) succeeds
        mock_socket_instance.bind.side_effect = [OSError("Port in use"), None]

        port = find_available_port(start_port=9000, max_attempts=10)
        assert isinstance(port, int)
        assert port == 9001


def test_find_available_port_no_ports_available():
    """Test behavior when no ports are available."""
    # Mock socket to always raise OSError
    with patch("socket.socket") as mock_socket:
        mock_socket.return_value.__enter__.return_value.bind.side_effect = OSError("Port in use")

        with pytest.raises(RuntimeError, match="No available ports found"):
            find_available_port(start_port=8000, max_attempts=5)


# LLM integration tests
def test_check_llm_status_disabled_for_testing():
    """Test LLM status check when disabled for testing."""
    bot = ChatBot()
    # Force disable LLM for testing
    bot._llm_enabled = False
    bot._llm_status = "disabled_for_testing"

    status = bot.check_llm_status()
    assert status["enabled"] is False
    assert status["status"] == "disabled_for_testing"
    assert "help" in status
    assert "disabled during testing" in status["help"]["issue"]


def test_check_llm_status_disabled_other_reason():
    """Test LLM status check when disabled for other reasons."""
    bot = ChatBot()
    # Force disable LLM for other reason
    bot._llm_enabled = False
    bot._llm_status = "some_other_error"

    status = bot.check_llm_status()
    assert status["enabled"] is False
    assert status["status"] == "some_other_error"
    assert "help" in status
    assert "LLM integration issue" in status["help"]["issue"]


def test_check_llm_status_enabled():
    """Test LLM status check when enabled."""
    bot = ChatBot()
    bot._llm_enabled = True
    bot._llm_status = "enabled"

    status = bot.check_llm_status()
    assert status["enabled"] is True
    assert status["status"] == "enabled"
    assert "help" in status


def test_quick_start_with_llm_disabled_for_testing():
    """Test quick start guide when LLM is disabled for testing."""
    bot = ChatBot()
    bot._llm_enabled = False
    bot._llm_status = "disabled_for_testing"

    guide = bot.quick_start()
    assert "🟡 Test mode" in guide
    assert "Disabled during testing" in guide


def test_quick_start_with_llm_enabled():
    """Test quick start guide when LLM is enabled."""
    bot = ChatBot()
    bot._llm_enabled = True

    guide = bot.quick_start()
    assert "🟢 Ready for real AI chat!" in guide
    assert "Disabled during testing" not in guide


def test_install_chatlas_help():
    """Test chatlas installation help."""
    bot = ChatBot()
    help_text = bot.install_chatlas_help()
    assert isinstance(help_text, str)
    assert len(help_text) > 0


@patch("sys.modules")
def test_auto_enable_llm_during_tests(mock_modules):
    """Test that LLM is disabled during tests."""

    # Simulate pytest being in sys.modules
    def contains_check(modules_self, item):
        return item == "pytest"

    mock_modules.__contains__ = contains_check

    bot = ChatBot()
    # Reset and call manually with mocked sys.modules
    bot._auto_enable_llm()

    assert bot._llm_enabled is False
    assert bot._llm_status == "disabled_for_testing"


@patch("sys.modules")
def test_auto_enable_llm_outside_tests(mock_modules):
    """Test that LLM is enabled outside of tests."""

    # Simulate pytest NOT being in sys.modules
    def contains_check(modules_self, item):
        return item != "pytest"  # Not in test mode

    mock_modules.__contains__ = contains_check

    bot = ChatBot()
    # Reset state and test manually
    bot._llm_enabled = False
    bot._llm_status = "unknown"
    bot._auto_enable_llm()

    assert bot._llm_enabled is True
    assert bot._llm_status == "enabled"


# LLM mode tests
def test_enable_llm_mode_when_already_enabled():
    """Test enable_llm_mode method when LLM is already enabled."""
    bot = ChatBot()
    bot._llm_enabled = True

    result = bot.enable_llm_mode()
    assert result is bot  # Should return self for chaining
    assert bot._llm_enabled is True


def test_enable_llm_mode_when_disabled():
    """Test enable_llm_mode method when LLM is disabled."""
    bot = ChatBot()
    bot._llm_enabled = False

    # Mock the _auto_enable_llm method
    with patch.object(bot, "_auto_enable_llm") as mock_auto_enable:
        result = bot.enable_llm_mode()

        mock_auto_enable.assert_called_once()
        assert result is bot  # Should return self for chaining


@patch("talk_box._utils_chatlas.ChatlasAdapter")
def test_chat_with_llm(mock_chatlas_adapter_class):
    """Test _chat_with_llm method."""
    # Set up mock
    mock_adapter = MagicMock()
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Test response from LLM"

    mock_chatlas_adapter_class.return_value = mock_adapter
    mock_adapter.create_chat_session.return_value = mock_session
    mock_adapter.chat_with_session.return_value = mock_response

    bot = ChatBot()
    bot._config["provider"] = "openai"
    bot._config["model"] = "gpt-4"

    result = bot._chat_with_llm("Hello")

    assert result == "Test response from LLM"
    mock_chatlas_adapter_class.assert_called_once_with(provider="openai", model="gpt-4")
    mock_adapter.create_chat_session.assert_called_once_with(bot._config)
    mock_adapter.chat_with_session.assert_called_once_with(mock_session, "Hello")


# Show method tests
@patch("builtins.print")
def test_show_help(mock_print):
    """Test show help mode."""
    bot = ChatBot()
    bot.show("help")
    mock_print.assert_called()
    # Check that quick_start content was printed
    call_args = str(mock_print.call_args_list)
    assert "Talk Box ChatBot Quick Start" in call_args


@patch("builtins.print")
def test_show_status(mock_print):
    """Test show status mode."""
    bot = ChatBot()
    bot.show("status")
    mock_print.assert_called()
    # Check that status information was printed
    call_args = str(mock_print.call_args_list)
    assert "LLM Integration Status" in call_args


@patch("builtins.print")
def test_show_install(mock_print):
    """Test show install mode."""
    bot = ChatBot()
    bot.show("install")
    mock_print.assert_called()
    # The install_chatlas_help method should be called


@patch("builtins.print")
def test_show_config(mock_print):
    """Test show config mode."""
    bot = ChatBot().model("gpt-4").temperature(0.5)
    bot.show("config")
    mock_print.assert_called()
    call_args = str(mock_print.call_args_list)
    assert "Configuration Summary" in call_args


@patch("builtins.print")
def test_show_prompt(mock_print):
    """Test show prompt mode."""
    bot = ChatBot()
    bot.show("prompt")
    mock_print.assert_called()
    call_args = str(mock_print.call_args_list)
    assert "System Prompt Analysis" in call_args


@patch("builtins.print")
def test_show_browser(mock_print):
    """Test show browser mode."""
    bot = ChatBot()
    bot.show("browser")
    mock_print.assert_called()
    call_args = str(mock_print.call_args_list)
    assert "Launching Browser Chat Interface" in call_args


@patch("builtins.print")
def test_show_basic_alias(mock_print):
    """Test show basic mode (alias for browser)."""
    bot = ChatBot()
    bot.show("basic")
    mock_print.assert_called()
    call_args = str(mock_print.call_args_list)
    assert "Launching Browser Chat Interface" in call_args


@patch("builtins.print")
def test_show_console(mock_print):
    """Test show console mode."""
    bot = ChatBot()
    bot.show("console")
    mock_print.assert_called()
    call_args = str(mock_print.call_args_list)
    assert "Launching Console Chat Interface" in call_args


@patch("builtins.print")
def test_show_invalid_mode(mock_print):
    """Test show with invalid mode."""
    bot = ChatBot()
    bot.show("invalid")
    mock_print.assert_called()
    call_args = str(mock_print.call_args_list)
    assert "Invalid mode" in call_args


# Simple console chat tests
@patch("builtins.input", side_effect=["Hello", "exit"])
@patch("builtins.print")
def test_simple_console_chat_normal_exit(mock_print, mock_input):
    """Test simple console chat with normal exit."""
    bot = ChatBot()
    bot._simple_console_chat()
    mock_print.assert_called()
    # Should have printed goodbye message
    call_args = str(mock_print.call_args_list)
    assert "Goodbye" in call_args


@patch("builtins.input", side_effect=["test", KeyboardInterrupt()])
@patch("builtins.print")
def test_simple_console_chat_keyboard_interrupt(mock_print, mock_input):
    """Test simple console chat with keyboard interrupt."""
    bot = ChatBot()
    bot._simple_console_chat()
    mock_print.assert_called()
    # Should handle KeyboardInterrupt gracefully
    call_args = str(mock_print.call_args_list)
    assert "Chat ended" in call_args


@patch("builtins.input", side_effect=["", "quit"])
@patch("builtins.print")
def test_simple_console_chat_empty_input(mock_print, mock_input):
    """Test simple console chat with empty input."""
    bot = ChatBot()
    bot._simple_console_chat()
    mock_print.assert_called()
    # Should continue after empty input


# Preset manager tests
@patch("talk_box.presets.PresetManager")
def test_preset_manager_property_success(mock_preset_manager_class):
    """Test successful preset manager creation via property."""
    mock_manager = MagicMock()
    mock_preset_manager_class.return_value = mock_manager

    bot = ChatBot()
    bot._preset_manager = None  # Reset to test lazy loading

    manager = bot.preset_manager

    assert manager == mock_manager
    mock_preset_manager_class.assert_called_once()


@patch("talk_box.presets.PresetManager", side_effect=ImportError("Module not found"))
def test_preset_manager_property_import_error(mock_preset_manager_class):
    """Test preset manager property creation with ImportError."""
    bot = ChatBot()
    bot._preset_manager = None  # Reset to test lazy loading

    manager = bot.preset_manager

    assert manager is None


def test_preset_manager_property_already_set():
    """Test preset manager property when already set."""
    bot = ChatBot()
    existing_manager = MagicMock()
    bot._preset_manager = existing_manager

    manager = bot.preset_manager

    assert manager is existing_manager


# Advanced configuration tests
def test_provider_model_parsing():
    """Test provider:model string parsing."""
    bot = ChatBot()

    # Test with provider:model format
    bot.provider_model("anthropic:claude-3-sonnet")
    config = bot.get_config()
    assert config["provider"] == "anthropic"
    assert config["model"] == "claude-3-sonnet"

    # Test without provider (should default to openai)
    bot.provider_model("gpt-4")
    config = bot.get_config()
    assert config["provider"] == "openai"
    assert config["model"] == "gpt-4"


def test_provider_model_validation():
    """Test provider_model validation."""
    bot = ChatBot()

    # Empty string should raise ValueError
    with pytest.raises(ValueError, match="provider_model must be a non-empty string"):
        bot.provider_model("")

    # None should raise ValueError
    with pytest.raises(ValueError, match="provider_model must be a non-empty string"):
        bot.provider_model(None)

    # Invalid format (empty provider or model) should raise ValueError
    with pytest.raises(ValueError, match="Both provider and model must be non-empty"):
        bot.provider_model(":")

    with pytest.raises(ValueError, match="Both provider and model must be non-empty"):
        bot.provider_model("provider:")

    with pytest.raises(ValueError, match="Both provider and model must be non-empty"):
        bot.provider_model(":model")


def test_get_system_prompt_with_preset():
    """Test system prompt generation with preset."""
    bot = ChatBot()
    # Mock a preset
    mock_preset = MagicMock()
    mock_preset.to_dict.return_value = {"system_prompt": "You are a technical advisor."}
    bot._current_preset = mock_preset
    bot._config["preset"] = "technical_advisor"

    prompt = bot.get_system_prompt()
    assert "You are a technical advisor." in prompt


def test_get_system_prompt_with_custom_prompt():
    """Test system prompt generation with custom prompt."""
    bot = ChatBot()
    bot.system_prompt("You are a helpful assistant specialized in data science.")

    prompt = bot.get_system_prompt()
    assert "You are a helpful assistant specialized in data science." in prompt


def test_get_system_prompt_with_persona():
    """Test system prompt generation with persona."""
    bot = ChatBot()
    bot.persona("Senior Data Scientist")

    prompt = bot.get_system_prompt()
    assert "You are: Senior Data Scientist" in prompt


def test_get_system_prompt_with_avoid_constraints():
    """Test system prompt generation with avoid constraints."""
    bot = ChatBot()
    bot.avoid(["politics", "medical advice"])

    prompt = bot.get_system_prompt()
    assert (
        "You MUST NOT provide any information, advice, or discussion about politics, or medical advice"
        in prompt
    )


def test_get_system_prompt_combination():
    """Test system prompt generation with multiple elements."""
    bot = ChatBot()
    bot.system_prompt("You are a helpful assistant.")
    bot.persona("Senior Engineer")
    bot.avoid(["inappropriate content"])

    prompt = bot.get_system_prompt()
    assert "You are a helpful assistant." in prompt
    assert "Additional persona: Senior Engineer" in prompt
    assert (
        "You MUST NOT provide any information, advice, or discussion about inappropriate content"
        in prompt
    )


def test_get_system_prompt_default_fallback():
    """Test system prompt generation with default fallback."""
    bot = ChatBot()
    # Clear all prompt configurations
    bot._config["preset"] = None
    bot._config["system_prompt"] = None
    bot._config["persona"] = None
    bot._config["avoid"] = []
    bot._current_preset = None

    prompt = bot.get_system_prompt()
    assert prompt == "You are a helpful AI assistant."


def test_get_config_summary():
    """Test configuration summary generation."""
    bot = ChatBot().model("gpt-4").temperature(0.3).persona("Data Scientist")
    summary = bot.get_config_summary()

    assert isinstance(summary, dict)
    assert summary["model"] == "gpt-4"
    assert summary["temperature"] == 0.3
    assert summary["persona"] == "Data Scientist"


def test_preset_with_preset_manager():
    """Test preset method with working preset manager."""
    bot = ChatBot()

    # Mock preset manager and preset
    mock_manager = MagicMock()
    mock_preset = MagicMock()
    mock_preset.to_dict.return_value = {"system_prompt": "Test preset"}
    mock_manager.get_preset.return_value = mock_preset

    bot._preset_manager = mock_manager
    bot.preset("technical_advisor")

    assert bot._config["preset"] == "technical_advisor"
    assert bot._current_preset == mock_preset


def test_preset_with_preset_manager_exception():
    """Test preset method when preset manager raises exception."""
    bot = ChatBot()

    # Mock preset manager that raises exception
    mock_manager = MagicMock()
    mock_manager.get_preset.side_effect = Exception("Preset not found")

    bot._preset_manager = mock_manager
    # Should not raise exception, just continue
    bot.preset("nonexistent_preset")

    assert bot._config["preset"] == "nonexistent_preset"
