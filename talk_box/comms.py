from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MessageType(Enum):
    """Type of inter-agent message.

    Attributes
    ----------
    REQUEST
        A request from one agent to another, expecting a response.
    RESPONSE
        A reply to a previous request (linked by `correlation_id`).
    BROADCAST
        A message sent to all agents in a mailbox.
    NOTIFY
        A one-way notification that does not expect a reply.
    """

    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFY = "notify"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentMessage:
    """A structured message passed between agents.

    Parameters
    ----------
    sender
        Name of the agent that sent the message.
    recipient
        Name of the intended recipient agent, or `"*"` for broadcasts.
    content
        The message body (text, instructions, data, etc.).
    message_type
        The type of message (request, response, broadcast, notify).
    metadata
        Arbitrary metadata attached to the message.
    correlation_id
        Links a response to its originating request. Auto-generated for requests; set to the
        request's `message_id` for responses.
    message_id
        Unique identifier for this message. Auto-generated.
    timestamp
        Unix timestamp when the message was created.

    Examples
    --------
    ```python
    import talk_box as tb

    msg = tb.AgentMessage(
        sender="analyst",
        recipient="reviewer",
        content="Please review this SQL query",
        message_type=tb.MessageType.REQUEST,
    )
    msg.sender       # "analyst"
    msg.message_type  # MessageType.REQUEST
    msg.message_id    # auto-generated UUID
    ```
    """

    sender: str
    recipient: str
    content: str
    message_type: MessageType = MessageType.NOTIFY
    metadata: dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Mailbox
# ---------------------------------------------------------------------------


class Mailbox:
    """Thread-safe message store for inter-agent communication.

    A `Mailbox` acts as a shared postal system. Agents send messages to named recipients and each
    agent can read its own inbox. Messages are stored centrally and filtered per-agent on read.

    All operations are thread-safe.

    Examples
    --------
    ```python
    import talk_box as tb

    mailbox = tb.Mailbox()

    # Send a request
    msg = tb.send(mailbox, sender="analyst", recipient="reviewer",
                  content="Review this query", message_type=tb.MessageType.REQUEST)

    # Recipient reads their inbox
    inbox = mailbox.inbox("reviewer")
    inbox[0].content  # "Review this query"

    # Reply
    tb.reply(mailbox, inbox[0], sender="reviewer", content="Looks good")

    # Broadcast to everyone
    tb.broadcast(mailbox, sender="manager", content="Meeting at 3pm")
    mailbox.inbox("analyst")   # includes the broadcast
    mailbox.inbox("reviewer")  # includes the broadcast too
    ```
    """

    def __init__(self) -> None:
        self._messages: list[AgentMessage] = []
        self._lock = threading.Lock()

    def deliver(self, message: AgentMessage) -> AgentMessage:
        """Add a message to the mailbox.

        Parameters
        ----------
        message
            The message to deliver.

        Returns
        -------
        AgentMessage
            The same message, for chaining.
        """
        with self._lock:
            self._messages.append(message)
        return message

    def inbox(self, agent: str) -> list[AgentMessage]:
        """Return all messages addressed to an agent (including broadcasts).

        Parameters
        ----------
        agent
            The agent name to get messages for.

        Returns
        -------
        list[AgentMessage]
            Messages where `recipient` matches *agent* or is `"*"` (broadcast).
        """
        with self._lock:
            return [m for m in self._messages if m.recipient == agent or m.recipient == "*"]

    def outbox(self, agent: str) -> list[AgentMessage]:
        """Return all messages sent by an agent.

        Parameters
        ----------
        agent
            The agent name to get sent messages for.

        Returns
        -------
        list[AgentMessage]
            Messages where `sender` matches `agent`.
        """
        with self._lock:
            return [m for m in self._messages if m.sender == agent]

    def thread(self, correlation_id: str) -> list[AgentMessage]:
        """Return all messages in a request/response thread.

        Parameters
        ----------
        correlation_id
            The correlation ID linking a request to its responses.

        Returns
        -------
        list[AgentMessage]
            Messages with a matching `correlation_id` or `message_id`,
            in chronological order.
        """
        with self._lock:
            return [
                m
                for m in self._messages
                if m.correlation_id == correlation_id or m.message_id == correlation_id
            ]

    def by_type(self, message_type: MessageType) -> list[AgentMessage]:
        """Return all messages of a specific type.

        Parameters
        ----------
        message_type
            The message type to filter by.

        Returns
        -------
        list[AgentMessage]
            All messages with the given type.
        """
        with self._lock:
            return [m for m in self._messages if m.message_type == message_type]

    @property
    def all(self) -> list[AgentMessage]:
        """All messages in the mailbox, in delivery order."""
        with self._lock:
            return list(self._messages)

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def send(
    mailbox: Mailbox,
    *,
    sender: str,
    recipient: str,
    content: str,
    message_type: MessageType = MessageType.NOTIFY,
    metadata: dict[str, Any] | None = None,
    correlation_id: str = "",
) -> AgentMessage:
    """Send a message from one agent to another.

    Parameters
    ----------
    mailbox
        The mailbox to deliver to.
    sender
        Name of the sending agent.
    recipient
        Name of the receiving agent.
    content
        The message body.
    message_type
        Type of message. Defaults to `NOTIFY`.
    metadata
        Optional metadata.
    correlation_id
        Links this message to a request/response thread.

    Returns
    -------
    AgentMessage
        The delivered message.

    Examples
    --------
    ```python
    import talk_box as tb

    mailbox = tb.Mailbox()
    msg = tb.send(mailbox, sender="a", recipient="b", content="Hello")
    mailbox.inbox("b")  # [msg]
    ```
    """
    msg = AgentMessage(
        sender=sender,
        recipient=recipient,
        content=content,
        message_type=message_type,
        metadata=metadata or {},
        correlation_id=correlation_id,
    )
    return mailbox.deliver(msg)


def broadcast(
    mailbox: Mailbox,
    *,
    sender: str,
    content: str,
    message_type: MessageType = MessageType.BROADCAST,
    metadata: dict[str, Any] | None = None,
) -> AgentMessage:
    """Broadcast a message to all agents.

    Parameters
    ----------
    mailbox
        The mailbox to deliver to.
    sender
        Name of the sending agent.
    content
        The message body.
    message_type
        Type of message. Defaults to `BROADCAST`.
    metadata
        Optional metadata.

    Returns
    -------
    AgentMessage
        The delivered broadcast message (`recipient="*"`).

    Examples
    --------
    ```python
    import talk_box as tb

    mailbox = tb.Mailbox()
    tb.broadcast(mailbox, sender="manager", content="Stand-up in 5 min")
    mailbox.inbox("dev1")  # includes the broadcast
    mailbox.inbox("dev2")  # includes the broadcast too
    ```
    """
    msg = AgentMessage(
        sender=sender,
        recipient="*",
        content=content,
        message_type=message_type,
        metadata=metadata or {},
    )
    return mailbox.deliver(msg)


def reply(
    mailbox: Mailbox,
    original: AgentMessage,
    *,
    sender: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> AgentMessage:
    """Reply to a message, linking via `correlation_id`.

    The reply is addressed to the original sender and its `correlation_id` is set to the original
    message's `message_id` so both messages belong to the same thread.

    Parameters
    ----------
    mailbox
        The mailbox to deliver to.
    original
        The message being replied to.
    sender
        Name of the agent sending the reply.
    content
        The reply body.
    metadata
        Optional metadata.

    Returns
    -------
    AgentMessage
        The delivered reply message.

    Examples
    --------
    ```python
    import talk_box as tb

    mailbox = tb.Mailbox()
    req = tb.send(mailbox, sender="a", recipient="b",
                  content="Review this", message_type=tb.MessageType.REQUEST)
    resp = tb.reply(mailbox, req, sender="b", content="Looks good")
    resp.correlation_id == req.message_id  # True
    resp.message_type  # MessageType.RESPONSE
    ```
    """
    msg = AgentMessage(
        sender=sender,
        recipient=original.sender,
        content=content,
        message_type=MessageType.RESPONSE,
        metadata=metadata or {},
        correlation_id=original.message_id,
    )
    return mailbox.deliver(msg)
