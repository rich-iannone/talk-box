import os
from typing import Any, Generator, Optional

import chatlas
from chatlas import (
    ChatAnthropic,
    ChatCloudflare,
    ChatDeepSeek,
    ChatGithub,
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
            elif provider == "github":
                chat = ChatGithub(model=model)
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

        # Add tool-awareness instruction when tools are configured
        tool_names = config.get("tools", [])
        if config.get("tool_box_enabled", False) or tool_names:
            tool_list = ", ".join(tool_names) if tool_names else "various tools"
            system_messages.append(
                f"You have tools available that you can and should use when appropriate: {tool_list}. "
                "When the user asks you to create, read, edit, or search files, use the file tools "
                "directly rather than showing the content and asking them to save it manually."
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
            # Disable echo/streaming to avoid Rich live display conflicts
            if isinstance(message, list):
                response = chat_session.chat(*message, echo="none", stream=False)
            else:
                response = chat_session.chat(message, echo="none", stream=False)

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

    def stream_with_session(
        self, chat_session: chatlas.Chat, message
    ) -> Generator[str, None, None]:
        """
        Stream a response from a chatlas session, yielding text chunks.

        Args:
            chat_session: Active chatlas.Chat session
            message: User message to send (str or list of content objects)

        Yields:
            str: Text chunks as they arrive from the LLM.
        """
        if isinstance(message, list):
            yield from chat_session.stream(*message, echo="none")
        else:
            yield from chat_session.stream(message, echo="none")

    def stream_with_thinking(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        thinking_budget: int = 2048,
        chat_session: Optional[chatlas.Chat] = None,
    ) -> Generator[tuple[str, str], None, None]:
        """
        Stream a response with thinking support (Anthropic only).

        Yields (phase, text) tuples where phase is 'thinking' or 'text'.
        Falls back to chatlas stream() for non-Anthropic providers.

        Args:
            message: User message to send.
            system_prompt: Optional system prompt.
            thinking_budget: Token budget for thinking (min 1024).
            chat_session: Optional pre-configured chatlas.Chat session with tools registered.

        Yields:
            tuple[str, str]: (phase, chunk) pairs.
        """
        if self.provider.lower() == "anthropic":
            try:
                import anthropic

                client = anthropic.Anthropic()

                # Build tool definitions from the chat session if available
                tools_param: list[dict[str, Any]] = []
                if chat_session is not None:
                    try:
                        for _name, tool_def in chat_session._tools.items():
                            schema = tool_def.schema
                            tools_param.append(
                                {
                                    "name": schema["function"]["name"],
                                    "description": schema["function"].get("description", ""),
                                    "input_schema": schema["function"].get(
                                        "parameters", {"type": "object", "properties": {}}
                                    ),
                                }
                            )
                    except Exception:
                        pass

                kwargs: dict[str, Any] = {
                    "model": self.default_model,
                    "max_tokens": max(4096, thinking_budget + 2048),
                    "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                    "messages": [{"role": "user", "content": message}],
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                if tools_param:
                    kwargs["tools"] = tools_param

                # Handle tool use loop
                while True:
                    with client.messages.stream(**kwargs) as stream:
                        tool_use_blocks: list[dict[str, Any]] = []
                        assistant_content: list[dict[str, Any]] = []
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_input_json = ""

                        for event in stream:
                            etype = getattr(event, "type", "")
                            if etype == "thinking":
                                text = getattr(event, "thinking", "")
                                assistant_content.append({"type": "thinking", "thinking": text})
                                yield ("thinking", text)
                            elif etype == "text":
                                text = getattr(event, "text", "")
                                yield ("text", text)
                            elif etype == "content_block_start":
                                block = getattr(event, "content_block", None)
                                if block and getattr(block, "type", "") == "tool_use":
                                    current_tool_id = block.id
                                    current_tool_name = block.name
                                    current_tool_input_json = ""
                            elif etype == "content_block_delta":
                                delta = getattr(event, "delta", None)
                                if delta and getattr(delta, "type", "") == "input_json_delta":
                                    current_tool_input_json += getattr(delta, "partial_json", "")
                            elif etype == "content_block_stop":
                                if current_tool_id and current_tool_name:
                                    import json as _json

                                    try:
                                        tool_input = (
                                            _json.loads(current_tool_input_json)
                                            if current_tool_input_json
                                            else {}
                                        )
                                    except _json.JSONDecodeError:
                                        tool_input = {}
                                    tool_use_blocks.append(
                                        {
                                            "id": current_tool_id,
                                            "name": current_tool_name,
                                            "input": tool_input,
                                        }
                                    )
                                    assistant_content.append(
                                        {
                                            "type": "tool_use",
                                            "id": current_tool_id,
                                            "name": current_tool_name,
                                            "input": tool_input,
                                        }
                                    )
                                    current_tool_id = None
                                    current_tool_name = None
                                    current_tool_input_json = ""

                    # If no tool calls, we're done
                    if not tool_use_blocks:
                        return

                    # Execute tool calls and build tool results
                    tool_results: list[dict[str, Any]] = []
                    for tool_call in tool_use_blocks:
                        tool_name = tool_call["name"]
                        tool_input = tool_call["input"]
                        try:
                            tool_def = chat_session._tools.get(tool_name)
                            if tool_def:
                                result = tool_def.func(**tool_input)
                                result_str = str(result) if result is not None else "Done."
                            else:
                                result_str = f"Error: Unknown tool '{tool_name}'"
                        except Exception as e:
                            result_str = f"Error executing {tool_name}: {e}"
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call["id"],
                                "content": result_str,
                            }
                        )
                        yield ("text", f"\n\n*Used tool `{tool_name}`*\n\n")

                    # Continue conversation with tool results
                    kwargs["messages"] = kwargs["messages"] + [
                        {"role": "assistant", "content": assistant_content},
                        {"role": "user", "content": tool_results},
                    ]
                    # Reset for next iteration
                    tool_use_blocks = []
                    assistant_content = []

                return
            except Exception:
                pass

        # Fallback: no thinking support, just stream text via session or new instance
        session = chat_session or self._create_chat_instance(model=self.default_model)
        for chunk in session.stream(message, echo="none"):
            if isinstance(chunk, str):
                yield ("text", chunk)
