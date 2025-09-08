import os
from typing import Any, Optional

import chatlas
from chatlas import (
    ChatAnthropic,
    ChatCloudflare,
    ChatDeepSeek,
    ChatGoogle,
    ChatGroq,
    ChatHuggingFace,
    ChatMistral,
    ChatOllama,
    ChatOpenAI,
    ChatOpenRouter,
    ChatPerplexity,
)

from talk_box.builder import ChatResponse
from talk_box.presets import PresetManager


class ChatlasAdapter:
    """
    Adapter to integrate Talk Box with chatlas for real LLM interactions.

    Supports multiple providers through several chatlas classes and models.
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the chatlas adapter.

        Args:
            provider: LLM provider (openai, anthropic, etc.). If None, uses CHATLAS_CHAT_PROVIDER env var.
            model: Model name. If None, uses CHATLAS_CHAT_MODEL env var.
        """
        self.provider = provider or os.getenv("CHATLAS_CHAT_PROVIDER", "openai")
        self.default_model = model or os.getenv("CHATLAS_CHAT_MODEL", "gpt-3.5-turbo")
        self.preset_manager = PresetManager()

    def _create_chat_instance(self, model: str) -> chatlas.Chat:
        """Create a chatlas Chat instance using the appropriate provider class."""
        try:
            provider = self.provider.lower()

            # Map provider names to their corresponding chatlas classes
            if provider == "openai":
                chat = ChatOpenAI(model=model)
            elif provider == "anthropic":
                chat = ChatAnthropic(model=model)
            elif provider == "google":
                chat = ChatGoogle(model=model)
            elif provider == "ollama":
                chat = ChatOllama(model=model)
            elif provider == "openrouter":
                chat = ChatOpenRouter(model=model)
            elif provider == "deepseek":
                chat = ChatDeepSeek(model=model)
            elif provider == "huggingface":
                chat = ChatHuggingFace(model=model)
            elif provider == "mistral":
                chat = ChatMistral(model=model)
            elif provider == "groq":
                chat = ChatGroq(model=model)
            elif provider == "perplexity":
                chat = ChatPerplexity(model=model)
            elif provider == "cloudflare":
                chat = ChatCloudflare(model=model)
            else:
                # Default to OpenAI for unknown providers
                chat = ChatOpenAI(model=model)

            return chat
        except Exception as e:
            raise ValueError(
                f"Failed to create chat session with provider '{self.provider}' and model '{model}': {e}"
            ) from e

    def create_chat_session(self, config: dict[str, Any]) -> chatlas.Chat:
        """
        Create a chatlas Chat session from Talk Box configuration.

        Args:
            config: Talk Box configuration dictionary

        Returns:
            Configured chatlas.Chat instance
        """
        # Extract chatlas-compatible parameters
        model = config.get("model", self.default_model)
        provider = config.get("provider", self.provider)

        # Set provider in environment for chatlas to use
        if provider:
            os.environ["CHATLAS_CHAT_PROVIDER"] = provider

        # Create the chat instance with the specified model
        chat = self._create_chat_instance(model=model)

        # Build system prompt from config elements
        system_messages = []

        # Priority order: custom system_prompt > preset > persona
        # 1. Custom system prompt (highest priority)
        custom_system_prompt = config.get("system_prompt")
        if custom_system_prompt:
            system_messages.append(custom_system_prompt)
        else:
            # 2. Preset system prompt (if no custom prompt)
            preset_name = config.get("preset")
            if preset_name:
                preset = self.preset_manager.get_preset(preset_name)
                if preset and preset.system_prompt:
                    system_messages.append(preset.system_prompt)

            # 3. Persona (if no custom prompt)
            persona = config.get("persona")
            if persona:
                system_messages.append(f"You are {persona}.")

        # Always apply constraints from avoid list
        avoid_list = config.get("avoid", [])
        if avoid_list:
            constraints = ", ".join(avoid_list)
            system_messages.append(
                f"Important: Avoid discussing or providing advice on: {constraints}"
            )

        # Set system prompt if we have any messages
        if system_messages:
            combined_prompt = " ".join(system_messages)
            chat.system_prompt = combined_prompt

        return chat

    def chat_with_session(self, chat_session: chatlas.Chat, message) -> ChatResponse:
        """
        Send a message to a chatlas session and get response.

        Args:
            chat_session: Active chatlas.Chat session
            message: User message to send (str or list of content objects)

        Returns:
            ChatResponse with the LLM's response
        """
        try:
            # Use chatlas to get the response
            # If message is a list of content objects, unpack it as individual arguments
            if isinstance(message, list):
                response = chat_session.chat(*message)
            else:
                response = chat_session.chat(message)

            # Extract response content (chatlas returns a Turn object)
            content = str(response)

            # Get model info if available
            model_info = getattr(chat_session, "_model", "unknown")

            # Calculate message length for metadata
            if isinstance(message, str):
                message_length = len(message)
            else:
                # For content lists, estimate total length
                message_length = sum(len(str(item)) for item in message)

            return ChatResponse(
                content=content,
                metadata={
                    "provider": self.provider,
                    "model": model_info,
                    "success": True,
                    "message_length": message_length,
                    "response_length": len(content),
                },
            )

        except Exception as e:
            return ChatResponse(
                content=f"Error communicating with LLM: {e!s}",
                metadata={"provider": self.provider, "error": str(e), "success": False},
            )
