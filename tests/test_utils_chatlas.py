import os
import pytest
from unittest.mock import patch, MagicMock

from talk_box._utils_chatlas import ChatlasAdapter
from talk_box.builder import ChatResponse


def test_chatlas_adapter_initialization_default():
    """Test ChatlasAdapter initialization with default values."""
    adapter = ChatlasAdapter()
    assert adapter.provider == "openai"
    assert adapter.default_model == "gpt-3.5-turbo"
    assert adapter.preset_manager is not None


def test_chatlas_adapter_initialization_with_params():
    """Test ChatlasAdapter initialization with custom parameters."""
    adapter = ChatlasAdapter(provider="anthropic", model="claude-3-sonnet")
    assert adapter.provider == "anthropic"
    assert adapter.default_model == "claude-3-sonnet"


@patch.dict(os.environ, {"CHATLAS_CHAT_PROVIDER": "google", "CHATLAS_CHAT_MODEL": "gemini-pro"})
def test_chatlas_adapter_initialization_from_env():
    """Test ChatlasAdapter initialization using environment variables."""
    adapter = ChatlasAdapter()
    assert adapter.provider == "google"
    assert adapter.default_model == "gemini-pro"


def test_chatlas_adapter_initialization_mixed():
    """Test ChatlasAdapter initialization with mixed sources."""
    with patch.dict(os.environ, {"CHATLAS_CHAT_PROVIDER": "google"}):
        adapter = ChatlasAdapter(model="custom-model")
        assert adapter.provider == "google"  # from env
        assert adapter.default_model == "custom-model"  # from param


@patch("talk_box._utils_chatlas.ChatOpenAI")
def test_create_chat_instance_openai(mock_chat_openai):
    """Test creating OpenAI chat instance."""
    mock_chat = MagicMock()
    mock_chat_openai.return_value = mock_chat

    adapter = ChatlasAdapter(provider="openai")
    result = adapter._create_chat_instance("gpt-4")

    mock_chat_openai.assert_called_once_with(model="gpt-4")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatAnthropic")
def test_create_chat_instance_anthropic(mock_chat_anthropic):
    """Test creating Anthropic chat instance."""
    mock_chat = MagicMock()
    mock_chat_anthropic.return_value = mock_chat

    adapter = ChatlasAdapter(provider="anthropic")
    result = adapter._create_chat_instance("claude-3-sonnet")

    mock_chat_anthropic.assert_called_once_with(model="claude-3-sonnet")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatGoogle")
def test_create_chat_instance_google(mock_chat_google):
    """Test creating Google chat instance."""
    mock_chat = MagicMock()
    mock_chat_google.return_value = mock_chat

    adapter = ChatlasAdapter(provider="google")
    result = adapter._create_chat_instance("gemini-pro")

    mock_chat_google.assert_called_once_with(model="gemini-pro")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatOllama")
def test_create_chat_instance_ollama(mock_chat_ollama):
    """Test creating Ollama chat instance."""
    mock_chat = MagicMock()
    mock_chat_ollama.return_value = mock_chat

    adapter = ChatlasAdapter(provider="ollama")
    result = adapter._create_chat_instance("llama2")

    mock_chat_ollama.assert_called_once_with(model="llama2")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatOpenRouter")
def test_create_chat_instance_openrouter(mock_chat_openrouter):
    """Test creating OpenRouter chat instance."""
    mock_chat = MagicMock()
    mock_chat_openrouter.return_value = mock_chat

    adapter = ChatlasAdapter(provider="openrouter")
    result = adapter._create_chat_instance("mistral-7b")

    mock_chat_openrouter.assert_called_once_with(model="mistral-7b")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatDeepSeek")
def test_create_chat_instance_deepseek(mock_chat_deepseek):
    """Test creating DeepSeek chat instance."""
    mock_chat = MagicMock()
    mock_chat_deepseek.return_value = mock_chat

    adapter = ChatlasAdapter(provider="deepseek")
    result = adapter._create_chat_instance("deepseek-coder")

    mock_chat_deepseek.assert_called_once_with(model="deepseek-coder")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatHuggingFace")
def test_create_chat_instance_huggingface(mock_chat_hf):
    """Test creating HuggingFace chat instance."""
    mock_chat = MagicMock()
    mock_chat_hf.return_value = mock_chat

    adapter = ChatlasAdapter(provider="huggingface")
    result = adapter._create_chat_instance("microsoft/DialoGPT-medium")

    mock_chat_hf.assert_called_once_with(model="microsoft/DialoGPT-medium")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatMistral")
def test_create_chat_instance_mistral(mock_chat_mistral):
    """Test creating Mistral chat instance."""
    mock_chat = MagicMock()
    mock_chat_mistral.return_value = mock_chat

    adapter = ChatlasAdapter(provider="mistral")
    result = adapter._create_chat_instance("mistral-medium")

    mock_chat_mistral.assert_called_once_with(model="mistral-medium")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatGroq")
def test_create_chat_instance_groq(mock_chat_groq):
    """Test creating Groq chat instance."""
    mock_chat = MagicMock()
    mock_chat_groq.return_value = mock_chat

    adapter = ChatlasAdapter(provider="groq")
    result = adapter._create_chat_instance("mixtral-8x7b")

    mock_chat_groq.assert_called_once_with(model="mixtral-8x7b")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatPerplexity")
def test_create_chat_instance_perplexity(mock_chat_perplexity):
    """Test creating Perplexity chat instance."""
    mock_chat = MagicMock()
    mock_chat_perplexity.return_value = mock_chat

    adapter = ChatlasAdapter(provider="perplexity")
    result = adapter._create_chat_instance("pplx-70b-online")

    mock_chat_perplexity.assert_called_once_with(model="pplx-70b-online")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatCloudflare")
def test_create_chat_instance_cloudflare(mock_chat_cloudflare):
    """Test creating Cloudflare chat instance."""
    mock_chat = MagicMock()
    mock_chat_cloudflare.return_value = mock_chat

    adapter = ChatlasAdapter(provider="cloudflare")
    result = adapter._create_chat_instance("@cf/meta/llama-2-7b-chat-int8")

    mock_chat_cloudflare.assert_called_once_with(model="@cf/meta/llama-2-7b-chat-int8")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatOpenAI")
def test_create_chat_instance_unknown_provider(mock_chat_openai):
    """Test creating chat instance with unknown provider defaults to OpenAI."""
    mock_chat = MagicMock()
    mock_chat_openai.return_value = mock_chat

    adapter = ChatlasAdapter(provider="unknown_provider")
    result = adapter._create_chat_instance("some-model")

    mock_chat_openai.assert_called_once_with(model="some-model")
    assert result == mock_chat


@patch("talk_box._utils_chatlas.ChatOpenAI")
def test_create_chat_instance_exception(mock_chat_openai):
    """Test exception handling in _create_chat_instance."""
    mock_chat_openai.side_effect = Exception("API Error")

    adapter = ChatlasAdapter(provider="openai")

    with pytest.raises(ValueError, match="Failed to create chat session"):
        adapter._create_chat_instance("gpt-4")


@patch("talk_box._utils_chatlas.ChatlasAdapter._create_chat_instance")
@patch.dict(os.environ, {}, clear=True)
def test_create_chat_session_basic(mock_create_chat):
    """Test basic chat session creation."""
    mock_chat = MagicMock()
    mock_create_chat.return_value = mock_chat

    adapter = ChatlasAdapter()
    config = {"model": "gpt-4", "provider": "openai"}

    result = adapter.create_chat_session(config)

    mock_create_chat.assert_called_once_with(model="gpt-4")
    assert result == mock_chat
    assert os.environ.get("CHATLAS_CHAT_PROVIDER") == "openai"


@patch("talk_box._utils_chatlas.ChatlasAdapter._create_chat_instance")
def test_create_chat_session_with_custom_system_prompt(mock_create_chat):
    """Test chat session creation with custom system prompt."""
    mock_chat = MagicMock()
    mock_create_chat.return_value = mock_chat

    adapter = ChatlasAdapter()
    config = {"model": "gpt-4", "system_prompt": "You are a helpful coding assistant."}

    result = adapter.create_chat_session(config)

    assert result.system_prompt == "You are a helpful coding assistant."


@patch("talk_box._utils_chatlas.ChatlasAdapter._create_chat_instance")
def test_create_chat_session_with_preset(mock_create_chat):
    """Test chat session creation with preset."""
    mock_chat = MagicMock()
    mock_create_chat.return_value = mock_chat

    # Mock preset manager
    mock_preset = MagicMock()
    mock_preset.system_prompt = "You are a technical advisor."

    adapter = ChatlasAdapter()
    adapter.preset_manager.get_preset = MagicMock(return_value=mock_preset)

    config = {"model": "gpt-4", "preset": "technical_advisor"}

    result = adapter.create_chat_session(config)

    adapter.preset_manager.get_preset.assert_called_once_with("technical_advisor")
    assert result.system_prompt == "You are a technical advisor."


@patch("talk_box._utils_chatlas.ChatlasAdapter._create_chat_instance")
def test_create_chat_session_with_persona(mock_create_chat):
    """Test chat session creation with persona."""
    mock_chat = MagicMock()
    mock_create_chat.return_value = mock_chat

    adapter = ChatlasAdapter()
    config = {"model": "gpt-4", "persona": "Senior Data Scientist"}

    result = adapter.create_chat_session(config)

    assert result.system_prompt == "You are Senior Data Scientist."


@patch("talk_box._utils_chatlas.ChatlasAdapter._create_chat_instance")
def test_create_chat_session_with_avoid_list(mock_create_chat):
    """Test chat session creation with avoid constraints."""
    mock_chat = MagicMock()
    mock_create_chat.return_value = mock_chat

    adapter = ChatlasAdapter()
    config = {"model": "gpt-4", "avoid": ["politics", "medical advice"]}

    result = adapter.create_chat_session(config)

    expected_prompt = "Important: Avoid discussing or providing advice on: politics, medical advice"
    assert result.system_prompt == expected_prompt


@patch("talk_box._utils_chatlas.ChatlasAdapter._create_chat_instance")
def test_create_chat_session_priority_order(mock_create_chat):
    """Test system prompt priority: custom > preset > persona."""
    mock_chat = MagicMock()
    mock_create_chat.return_value = mock_chat

    adapter = ChatlasAdapter()
    config = {
        "model": "gpt-4",
        "system_prompt": "Custom prompt",
        "preset": "technical_advisor",
        "persona": "Senior Engineer",
    }

    result = adapter.create_chat_session(config)

    # Custom prompt should take priority
    assert result.system_prompt == "Custom prompt"


@patch("talk_box._utils_chatlas.ChatlasAdapter._create_chat_instance")
def test_create_chat_session_complex_combination(mock_create_chat):
    """Test chat session creation with multiple elements."""
    mock_chat = MagicMock()
    mock_create_chat.return_value = mock_chat

    adapter = ChatlasAdapter()
    config = {"model": "gpt-4", "persona": "Senior Engineer", "avoid": ["inappropriate content"]}

    result = adapter.create_chat_session(config)

    expected_prompt = "You are Senior Engineer. Important: Avoid discussing or providing advice on: inappropriate content"
    assert result.system_prompt == expected_prompt


def test_chat_with_session_success():
    """Test successful chat interaction."""
    # Mock chatlas session
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = MagicMock(return_value="Hello! How can I help you?")
    mock_session.chat.return_value = mock_response
    mock_session._model = "gpt-4"

    adapter = ChatlasAdapter(provider="openai")
    result = adapter.chat_with_session(mock_session, "Hello")

    mock_session.chat.assert_called_once_with("Hello")
    assert isinstance(result, ChatResponse)
    assert result.content == "Hello! How can I help you?"
    assert result.metadata["provider"] == "openai"
    assert result.metadata["model"] == "gpt-4"
    assert result.metadata["success"] is True
    assert result.metadata["message_length"] == 5
    assert result.metadata["response_length"] == 26


def test_chat_with_session_no_model_info():
    """Test chat interaction when model info is not available."""
    # Mock chatlas session without _model attribute
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.__str__ = MagicMock(return_value="Hello!")
    mock_session.chat.return_value = mock_response
    del mock_session._model  # Remove _model attribute

    adapter = ChatlasAdapter(provider="anthropic")
    result = adapter.chat_with_session(mock_session, "Hi")

    assert result.metadata["model"] == "unknown"


def test_chat_with_session_exception():
    """Test chat interaction with exception."""
    # Mock chatlas session that raises exception
    mock_session = MagicMock()
    mock_session.chat.side_effect = Exception("Network error")

    adapter = ChatlasAdapter(provider="openai")
    result = adapter.chat_with_session(mock_session, "Hello")

    assert isinstance(result, ChatResponse)
    assert "Error communicating with LLM" in result.content
    assert result.metadata["provider"] == "openai"
    assert result.metadata["success"] is False
    assert "Network error" in result.metadata["error"]
