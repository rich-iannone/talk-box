"""Conversation capture: record prompts, responses, tool calls, and events."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class EventType(Enum):
    """Type of event recorded during a conversation.

    Attributes
    ----------
    PROMPT
        A user prompt or system instruction sent to the model.
    RESPONSE
        A model response.
    TOOL_CALL
        A tool/function call initiated by the model.
    TOOL_RESULT
        The result returned from a tool/function call.
    GUARD_CHECK
        A guardrail check event (pass or fail).
    PATHWAY_TRANSITION
        A state transition in a conversational pathway.
    ERROR
        An error that occurred during processing.
    METADATA
        A metadata-only event (e.g., session start, configuration).
    """

    PROMPT = "prompt"
    RESPONSE = "response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GUARD_CHECK = "guard_check"
    PATHWAY_TRANSITION = "pathway_transition"
    ERROR = "error"
    METADATA = "metadata"


@dataclass(frozen=True)
class CaptureEvent:
    """A single recorded event in a conversation.

    Parameters
    ----------
    event_type
        The type of event.
    content
        The primary content (e.g., prompt text, response text, error message).
    timestamp
        Unix timestamp when the event occurred.
    event_id
        Unique identifier for this event.
    model
        Model identifier, if relevant.
    role
        Message role (``"user"``, ``"assistant"``, ``"system"``), if relevant.
    duration_ms
        Duration of the event in milliseconds (e.g., response latency).
    metadata
        Additional structured data about the event.
    """

    event_type: EventType
    content: str = ""
    timestamp: float = 0.0
    event_id: str = ""
    model: str = ""
    role: str = ""
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this event to a dictionary."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureEvent:
        """Deserialize an event from a dictionary.

        Parameters
        ----------
        data
            Dictionary with event fields.

        Returns
        -------
        CaptureEvent
            The deserialized event.
        """
        data = dict(data)  # Don't mutate the input
        data["event_type"] = EventType(data["event_type"])
        return cls(**data)


# ---------------------------------------------------------------------------
# ConversationCapture
# ---------------------------------------------------------------------------


class ConversationCapture:
    """Records a sequence of events during a conversation.

    Provides methods to record different event types, query recorded events,
    and serialize the capture to/from JSON.

    Parameters
    ----------
    session_id
        Unique identifier for this capture session. Auto-generated if not provided.
    metadata
        Session-level metadata (e.g., model name, persona, configuration).

    Examples
    --------
    ```python
    import talk_box as tb

    capture = tb.ConversationCapture()

    # Record events
    capture.record_prompt("What is Python?")
    capture.record_response("Python is a programming language.", model="openai:gpt-4o")

    # Query events
    capture.events                  # All events
    capture.prompts()               # Only prompts
    capture.responses()             # Only responses
    len(capture)                    # Number of events

    # Serialize
    capture.to_json("session.json")
    loaded = tb.ConversationCapture.from_json("session.json")
    ```
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._session_id = session_id or str(uuid4())
        self._metadata = metadata or {}
        self._events: list[CaptureEvent] = []
        self._start_time = time.time()

    @property
    def session_id(self) -> str:
        """Unique session identifier."""
        return self._session_id

    @property
    def metadata(self) -> dict[str, Any]:
        """Session-level metadata."""
        return dict(self._metadata)

    @property
    def events(self) -> list[CaptureEvent]:
        """All recorded events in order."""
        return list(self._events)

    @property
    def start_time(self) -> float:
        """Unix timestamp when the capture session started."""
        return self._start_time

    @property
    def duration_ms(self) -> float:
        """Total session duration in milliseconds."""
        return (time.time() - self._start_time) * 1000

    # -- Recording methods ---------------------------------------------------

    def record(
        self,
        event_type: EventType,
        content: str = "",
        *,
        model: str = "",
        role: str = "",
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record a generic event.

        Parameters
        ----------
        event_type
            The type of event.
        content
            The primary content.
        model
            Model identifier, if relevant.
        role
            Message role, if relevant.
        duration_ms
            Event duration in milliseconds.
        metadata
            Additional structured data.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        event = CaptureEvent(
            event_type=event_type,
            content=content,
            timestamp=time.time(),
            event_id=str(uuid4()),
            model=model,
            role=role,
            duration_ms=duration_ms,
            metadata=dict(metadata) if metadata else {},
        )
        self._events.append(event)
        return event

    def record_prompt(
        self,
        content: str,
        *,
        role: str = "user",
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record a prompt event.

        Parameters
        ----------
        content
            The prompt text.
        role
            The sender role (default ``"user"``).
        metadata
            Additional metadata.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        return self.record(EventType.PROMPT, content, role=role, metadata=metadata)

    def record_response(
        self,
        content: str,
        *,
        model: str = "",
        duration_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record a model response event.

        Parameters
        ----------
        content
            The response text.
        model
            Model that generated the response.
        duration_ms
            Response latency in milliseconds.
        metadata
            Additional metadata.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        return self.record(
            EventType.RESPONSE,
            content,
            model=model,
            role="assistant",
            duration_ms=duration_ms,
            metadata=metadata,
        )

    def record_tool_call(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any] | None = None,
        model: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record a tool/function call event.

        Parameters
        ----------
        tool_name
            Name of the tool being called.
        arguments
            Arguments passed to the tool.
        model
            Model that initiated the call.
        metadata
            Additional metadata.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        meta = dict(metadata or {})
        meta["tool_name"] = tool_name
        if arguments is not None:
            meta["arguments"] = arguments
        return self.record(EventType.TOOL_CALL, tool_name, model=model, metadata=meta)

    def record_tool_result(
        self,
        tool_name: str,
        result: str,
        *,
        duration_ms: float | None = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record a tool result event.

        Parameters
        ----------
        tool_name
            Name of the tool that produced the result.
        result
            The tool's output.
        duration_ms
            Tool execution duration in milliseconds.
        success
            Whether the tool call succeeded.
        metadata
            Additional metadata.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        meta = dict(metadata or {})
        meta["tool_name"] = tool_name
        meta["success"] = success
        return self.record(EventType.TOOL_RESULT, result, duration_ms=duration_ms, metadata=meta)

    def record_guard_check(
        self,
        guard_name: str,
        passed: bool,
        *,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record a guardrail check event.

        Parameters
        ----------
        guard_name
            Name of the guard that was checked.
        passed
            Whether the check passed.
        message
            Optional message from the guard.
        metadata
            Additional metadata.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        meta = dict(metadata or {})
        meta["guard_name"] = guard_name
        meta["passed"] = passed
        content = f"{guard_name}: {'passed' if passed else 'failed'}"
        if message:
            content += f" - {message}"
        return self.record(EventType.GUARD_CHECK, content, metadata=meta)

    def record_pathway_transition(
        self,
        from_state: str,
        to_state: str,
        *,
        trigger: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record a pathway state transition event.

        Parameters
        ----------
        from_state
            The state being left.
        to_state
            The state being entered.
        trigger
            What triggered the transition.
        metadata
            Additional metadata.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        meta = dict(metadata or {})
        meta["from_state"] = from_state
        meta["to_state"] = to_state
        if trigger:
            meta["trigger"] = trigger
        content = f"{from_state} -> {to_state}"
        if trigger:
            content += f" ({trigger})"
        return self.record(EventType.PATHWAY_TRANSITION, content, metadata=meta)

    def record_error(
        self,
        error: str,
        *,
        error_type: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> CaptureEvent:
        """Record an error event.

        Parameters
        ----------
        error
            Error message.
        error_type
            Type/class of the error.
        metadata
            Additional metadata.

        Returns
        -------
        CaptureEvent
            The recorded event.
        """
        meta = dict(metadata or {})
        if error_type:
            meta["error_type"] = error_type
        return self.record(EventType.ERROR, error, metadata=meta)

    # -- Query methods -------------------------------------------------------

    def filter(self, event_type: EventType) -> list[CaptureEvent]:
        """Get all events of a specific type.

        Parameters
        ----------
        event_type
            The event type to filter by.

        Returns
        -------
        list[CaptureEvent]
            Events matching the type, in order.
        """
        return [e for e in self._events if e.event_type == event_type]

    def prompts(self) -> list[CaptureEvent]:
        """Get all prompt events."""
        return self.filter(EventType.PROMPT)

    def responses(self) -> list[CaptureEvent]:
        """Get all response events."""
        return self.filter(EventType.RESPONSE)

    def tool_calls(self) -> list[CaptureEvent]:
        """Get all tool call events."""
        return self.filter(EventType.TOOL_CALL)

    def tool_results(self) -> list[CaptureEvent]:
        """Get all tool result events."""
        return self.filter(EventType.TOOL_RESULT)

    def errors(self) -> list[CaptureEvent]:
        """Get all error events."""
        return self.filter(EventType.ERROR)

    def turns(self) -> list[tuple[CaptureEvent, CaptureEvent | None]]:
        """Get prompt-response pairs as turns.

        Returns
        -------
        list[tuple[CaptureEvent, CaptureEvent | None]]
            Each tuple is ``(prompt, response)`` where response may be ``None``
            if the prompt didn't receive a response.
        """
        result: list[tuple[CaptureEvent, CaptureEvent | None]] = []
        prompts = self.prompts()
        resps = self.responses()

        for i, prompt in enumerate(prompts):
            response = resps[i] if i < len(resps) else None
            result.append((prompt, response))

        return result

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire capture to a dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the capture.
        """
        return {
            "session_id": self._session_id,
            "start_time": self._start_time,
            "metadata": self._metadata,
            "events": [e.to_dict() for e in self._events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationCapture:
        """Deserialize a capture from a dictionary.

        Parameters
        ----------
        data
            Dictionary with capture fields.

        Returns
        -------
        ConversationCapture
            The deserialized capture.
        """
        capture = cls(
            session_id=data["session_id"],
            metadata=data.get("metadata", {}),
        )
        capture._start_time = data.get("start_time", capture._start_time)
        capture._events = [CaptureEvent.from_dict(e) for e in data.get("events", [])]
        return capture

    def to_json(self, path: str | Path) -> None:
        """Save the capture to a JSON file.

        Parameters
        ----------
        path
            File path to write to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_json(cls, path: str | Path) -> ConversationCapture:
        """Load a capture from a JSON file.

        Parameters
        ----------
        path
            File path to read from.

        Returns
        -------
        ConversationCapture
            The loaded capture.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)

    def __len__(self) -> int:
        return len(self._events)
