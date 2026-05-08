from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from talk_box.builder import ChatBot
from talk_box.capture import ConversationCapture
from talk_box.conversation import Conversation
from talk_box.memory import MemoryStore, MemoryTier
from talk_box.personas._loader import PersonaDefinition
from talk_box.retention import RetentionPolicy, apply_retention

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


@dataclass
class Agent:
    """A self-contained AI agent bundling persona, memory, and conversation history.

    An `Agent` wraps a `ChatBot`, a `PersonaDefinition`, a `MemoryStore`, and a
    `ConversationCapture` into a single entity that can participate in multi-agent workflows.

    The agent configures its `ChatBot` automatically from the persona (system prompt, temperature,
    tools, guardrails) and applies retention policies after each response when a `RetentionPolicy`
    is attached.

    Parameters
    ----------
    name
        Unique identifier for this agent (e.g., `"triage_bot"`).
    persona
        The persona definition that drives the agent's behavior.
    memory
        The agent's memory store. Defaults to an in-memory store.
    chatbot
        A pre-configured `ChatBot`. If `None`, one is built from the persona automatically.
    capture
        Conversation capture for recording events. Created automatically
        if not provided.
    instructions
        Additional instructions appended to the persona's system prompt.
    metadata
        Arbitrary metadata for the agent (e.g., team, version, owner).

    Examples
    --------
    Create an agent from a built-in persona:

    ```python
    import talk_box as tb

    agent = tb.Agent.from_persona("code_reviewer")
    agent.name          # "code_reviewer"
    agent.persona.display_name  # "Code Reviewer"
    ```

    Create a custom agent with memory and retention:

    ```python
    import talk_box as tb

    policy = tb.RetentionPolicy(
        remember_tags=["identity", "preference"],
        forget_tags=["scratch"],
    )
    agent = tb.Agent(
        name="support_agent",
        persona=tb.create_persona(
            "support",
            persona_role="customer support specialist",
            retention=policy,
        ),
        memory=tb.MemoryStore(long_term_path="support.db"),
    )
    ```

    Respond to a user message:

    ```python
    conversation = agent.respond("How do I reset my password?")
    ```
    """

    name: str
    persona: PersonaDefinition
    memory: MemoryStore = field(default_factory=lambda: MemoryStore(long_term_path=":memory:"))
    chatbot: ChatBot = field(default=None)  # type: ignore[assignment]
    capture: ConversationCapture = field(default_factory=ConversationCapture)
    instructions: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal state
    _conversation: Conversation | None = field(default=None, init=False, repr=False)
    _configured: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.chatbot is None:
            self.chatbot = ChatBot(name=self.name)
        if not self._configured:
            self._apply_persona()
            self._configured = True

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_persona(
        cls,
        persona_name: str,
        *,
        name: str | None = None,
        memory: MemoryStore | None = None,
        instructions: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Agent:
        """Create an agent from a registered persona name.

        Parameters
        ----------
        persona_name
            Name of a registered persona (e.g., `"code_reviewer"`).
        name
            Agent name. Defaults to the persona name.
        memory
            Memory store. Defaults to an in-memory store.
        instructions
            Additional instructions for the agent.
        metadata
            Arbitrary metadata.

        Returns
        -------
        Agent
            A fully configured agent.

        Raises
        ------
        KeyError
            If the persona name is not found.

        Examples
        --------
        ```python
        import talk_box as tb

        agent = tb.Agent.from_persona("data_analyst")
        agent.name  # "data_analyst"
        ```
        """
        from talk_box.personas import get_persona

        persona = get_persona(persona_name)
        return cls(
            name=name or persona_name,
            persona=persona,
            memory=memory or MemoryStore(long_term_path=":memory:"),
            instructions=instructions,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def respond(
        self,
        message: str,
        *,
        conversation: Conversation | None = None,
    ) -> Conversation:
        """Send a message and get a response from this agent.

        Uses the agent's configured `ChatBot` to generate a response. If no conversation is
        provided, continues the agent's internal conversation (or starts a new one).

        The agent records prompts and responses in its `capture` and applies its retention policy
        (if the persona defines one) after each response.

        Parameters
        ----------
        message
            The user message to respond to.
        conversation
            An existing conversation to continue. If `None`, uses the agent's internal conversation
            state.

        Returns
        -------
        Conversation
            The updated conversation with the response appended.

        Examples
        --------
        ```python
        import talk_box as tb

        agent = tb.Agent.from_persona("python_mentor")
        convo = agent.respond("What are list comprehensions?")
        convo = agent.respond("Show me an example", conversation=convo)
        ```
        """
        if conversation is not None:
            self._conversation = conversation

        # Record the prompt
        self.capture.record_prompt(message)

        # Chat
        result = self.chatbot.chat(message, conversation=self._conversation)
        self._conversation = result

        # Record the response
        last = result.get_last_message()
        if last is not None:
            self.capture.record_response(
                str(last.content),
                model=self._model_name(),
            )

        # Apply retention policy if persona defines one
        if self.persona.retention is not None:
            apply_retention(self.memory, self.persona.retention)

        return result

    def remember(
        self,
        key: str,
        value: Any,
        *,
        tier: MemoryTier = MemoryTier.WORKING,
        tags: tuple[str, ...] = (),
    ) -> None:
        """Store a value in the agent's memory.

        Convenience wrapper around `self.memory.remember()`.

        Parameters
        ----------
        key
            The key to store under.
        value
            The value to store.
        tier
            Which memory tier to use (default `WORKING`).
        tags
            Optional tags for categorization.
        """
        self.memory.remember(key, value, tier=tier, tags=tags)

    def recall(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the agent's memory.

        Convenience wrapper around `self.memory.recall()`.

        Parameters
        ----------
        key
            The key to look up.
        default
            Value to return if the key is not found.

        Returns
        -------
        Any
            The stored value, or *default*.
        """
        return self.memory.recall(key, default=default)

    @property
    def retention(self) -> RetentionPolicy | None:
        """The agent's retention policy (from its persona), if any."""
        return self.persona.retention

    @property
    def conversation(self) -> Conversation | None:
        """The agent's current conversation, if any."""
        return self._conversation

    def reset_conversation(self) -> None:
        """Clear the agent's conversation state and start fresh."""
        self._conversation = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_persona(self) -> None:
        """Configure the ChatBot from the persona definition."""
        persona = self.persona

        # Build the system prompt from the persona
        builder = persona.build_prompt_builder()
        if self.instructions:
            builder.constraint(self.instructions)
        self.chatbot.system_prompt(builder)

        # Set avoid topics
        if persona.avoid_topics:
            existing = self.chatbot._config.get("avoid_topics") or []
            self.chatbot._config["avoid_topics"] = list(set(existing + persona.avoid_topics))

        # Set temperature and max_tokens if persona defines them
        if persona.temperature is not None:
            self.chatbot._config["temperature"] = persona.temperature
        if persona.max_tokens is not None:
            self.chatbot._config["max_tokens"] = persona.max_tokens

        # Enable persona tools
        if persona.tools:
            existing_tools = self.chatbot._config.get("tools") or []
            self.chatbot._config["tools"] = list(set(existing_tools + persona.tools))

        # Apply default guardrails
        if persona.default_guards:
            from talk_box.guardrails import resolve_guards

            for guard in resolve_guards(persona.default_guards):
                self.chatbot.guardrail(guard)

        # Store metadata for introspection
        self.chatbot._config["persona_pack"] = persona.name
        self.chatbot._config["persona_definition"] = persona

    def _model_name(self) -> str:
        """Get the current model name from the ChatBot config."""
        return str(self.chatbot._config.get("model", "unknown"))
