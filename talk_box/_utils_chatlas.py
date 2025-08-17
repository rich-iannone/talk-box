"""
Chatlas adapter for Talk Box framework.

This module provides seamless integration with the chatlas library to enable
real LLM interactions using various providers (OpenAI, Anthropic, etc.).
"""

import os
from typing import Any, Optional

import chatlas
from chatlas import ChatAuto

from talk_box.builder import ChatBot, ChatResponse
from talk_box.presets import PresetManager


class ChatlasAdapter:
    """
    Adapter to integrate Talk Box with chatlas for real LLM interactions.

    Supports multiple providers through chatlas.ChatAuto which automatically
    selects the appropriate provider based on environment variables.
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
        """Create a chatlas Chat instance using ChatAuto for automatic provider selection."""
        try:
            # Set provider in environment if not already set and we have one
            if self.provider and not os.getenv("CHATLAS_CHAT_PROVIDER"):
                os.environ["CHATLAS_CHAT_PROVIDER"] = self.provider

            # Use ChatAuto for automatic provider detection - only pass model
            chat = ChatAuto(model=model)
            return chat
        except Exception as e:
            raise ValueError(f"Failed to create chat session with model '{model}': {e}") from e

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

        # Create the chat instance (just with model)
        chat = self._create_chat_instance(model=model)

        # Build system prompt from preset and persona
        system_messages = []

        # Apply preset if specified
        preset_name = config.get("preset")
        if preset_name:
            preset = self.preset_manager.get_preset(preset_name)
            if preset and preset.system_prompt:
                system_messages.append(preset.system_prompt)

        # Apply persona if specified
        persona = config.get("persona")
        if persona:
            system_messages.append(f"You are {persona}.")

        # Apply constraints from avoid list
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

    def chat_with_session(
        self, chat_session: chatlas.Chat, message: str
    ) -> ChatResponse:
        """
        Send a message to a chatlas session and get response.

        Args:
            chat_session: Active chatlas.Chat session
            message: User message to send

        Returns:
            ChatResponse with the LLM's response
        """
        try:
            # Use chatlas to get the response
            response = chat_session.chat(message)

            # Extract response content (chatlas returns a Turn object)
            content = str(response)

            # Get model info if available
            model_info = getattr(chat_session, "_model", "unknown")

            return ChatResponse(
                content=content,
                metadata={
                    "provider": self.provider,
                    "model": model_info,
                    "success": True,
                    "message_length": len(message),
                    "response_length": len(content),
                },
            )

        except Exception as e:
            return ChatResponse(
                content=f"Error communicating with LLM: {e!s}",
                metadata={"provider": self.provider, "error": str(e), "success": False},
            )


def enhance_chatbot_with_chatlas():
    """
    Enhance the ChatBot class with real LLM integration via chatlas.

    This function monkey-patches the ChatBot class to add LLM functionality.
    Call this once at startup to enable real chat capabilities.
    """

    def create_chat_session(self) -> chatlas.Chat:
        """Create a chatlas session from the current configuration."""
        adapter = ChatlasAdapter()
        return adapter.create_chat_session(self._config)

    def chat_with_llm(self, message: str) -> ChatResponse:
        """
        Send a message to a real LLM via chatlas.

        This replaces the default echo behavior with actual LLM interaction.
        """
        adapter = ChatlasAdapter()
        chat_session = adapter.create_chat_session(self._config)
        return adapter.chat_with_session(chat_session, message)

    def enable_llm_mode(self):
        """Enable LLM mode by replacing the chat method with chat_with_llm."""
        # Replace the default chat method with LLM-powered version
        self.chat = lambda message: self.chat_with_llm(message)
        return self

    # Add methods to ChatBot class
    ChatBot.create_chat_session = create_chat_session
    ChatBot.chat_with_llm = chat_with_llm
    ChatBot.enable_llm_mode = enable_llm_mode

    return ChatBot
