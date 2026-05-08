from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from talk_box.agent import Agent
from talk_box.conversation import Conversation
from talk_box.personas._loader import PersonaDefinition
from talk_box.shared_state import SharedState

# ---------------------------------------------------------------------------
# Metadata keys for parent-child tracking
# ---------------------------------------------------------------------------

_PARENT_KEY = "_parent"
_CHILDREN_KEY = "_children"
_SHARED_STATE_KEY = "_shared_state"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubagentResult:
    """Result of delegating a task to a subagent.

    Parameters
    ----------
    response
        The subagent's text response.
    agent
        Name of the subagent that produced the result.
    parent
        Name of the parent agent that delegated the task.
    conversation
        The full conversation after the subagent responded.
    duration
        Wall-clock time in seconds the delegation took.

    Examples
    --------
    ```python
    import talk_box as tb

    parent = tb.Agent.from_persona("project_manager")
    state = tb.SharedState()
    reviewer = tb.spawn(parent, "reviewer", persona="code_reviewer", shared_state=state)

    result = tb.delegate(parent, reviewer, "Review this function for bugs")
    result.response   # The reviewer's analysis
    result.agent      # "reviewer"
    result.parent     # "project_manager"
    result.duration   # Time in seconds
    ```
    """

    response: str
    agent: str
    parent: str
    conversation: Conversation
    duration: float


# ---------------------------------------------------------------------------
# spawn
# ---------------------------------------------------------------------------


def spawn(
    parent: Agent,
    name: str,
    *,
    persona: str | PersonaDefinition | None = None,
    instructions: str = "",
    shared_state: SharedState | None = None,
    metadata: dict[str, Any] | None = None,
) -> Agent:
    """Create a subagent linked to a parent agent.

    The subagent gets its own memory and conversation history but can share
    context with the parent through an optional `SharedState`.

    Parameters
    ----------
    parent
        The parent agent spawning the subagent.
    name
        Unique name for the subagent.
    persona
        Persona for the subagent.  Can be a registered persona name (str) or a
        `PersonaDefinition`.  If `None`, the parent's persona is inherited.
    instructions
        Additional instructions for the subagent.
    shared_state
        A `SharedState` instance for sharing context between parent and child.
        Stored in both agents' metadata for auto-discovery by `delegate()`.
    metadata
        Arbitrary metadata for the subagent.

    Returns
    -------
    Agent
        A new agent linked to the parent.

    Raises
    ------
    TypeError
        If *persona* is not a `str`, `PersonaDefinition`, or `None`.

    Examples
    --------
    Spawn a subagent from a registered persona:

    ```python
    import talk_box as tb

    parent = tb.Agent.from_persona("project_manager")
    state = tb.SharedState()
    reviewer = tb.spawn(parent, "reviewer", persona="code_reviewer", shared_state=state)
    tb.children(parent)  # ["reviewer"]
    tb.parent_name(reviewer)  # "project_manager"
    ```

    Spawn a subagent inheriting the parent's persona:

    ```python
    helper = tb.spawn(parent, "helper")
    helper.persona == parent.persona  # True
    ```
    """
    child_meta = dict(metadata or {})
    child_meta[_PARENT_KEY] = parent.name

    # Resolve persona
    if isinstance(persona, str):
        child = Agent.from_persona(
            persona,
            name=name,
            instructions=instructions,
            metadata=child_meta,
        )
    elif isinstance(persona, PersonaDefinition):
        child = Agent(
            name=name,
            persona=persona,
            instructions=instructions,
            metadata=child_meta,
        )
    elif persona is None:
        # Inherit parent's persona
        child = Agent(
            name=name,
            persona=parent.persona,
            instructions=instructions,
            metadata=child_meta,
        )
    else:
        raise TypeError(
            f"'persona' must be a str, PersonaDefinition, or None, got {type(persona).__name__}"
        )

    # Track parent → child relationship
    parent.metadata.setdefault(_CHILDREN_KEY, []).append(name)

    # Link shared state on both agents
    if shared_state is not None:
        parent.metadata[_SHARED_STATE_KEY] = shared_state
        child.metadata[_SHARED_STATE_KEY] = shared_state

    return child


# ---------------------------------------------------------------------------
# delegate
# ---------------------------------------------------------------------------


def delegate(
    parent: Agent,
    subagent: Agent,
    task: str,
    *,
    shared_state: SharedState | None = None,
) -> SubagentResult:
    """Delegate a task to a subagent and collect the result.

    Sends a message to the subagent, times the execution, and returns a
    structured `SubagentResult`.  If a `SharedState` is provided (or was
    linked during `spawn()`), the task and result are recorded in the
    subagent's namespace.

    Parameters
    ----------
    parent
        The parent agent delegating the task.
    subagent
        The subagent to delegate to.
    task
        The task description or message to send.
    shared_state
        A `SharedState` for recording task context.  If `None`, the shared
        state from `spawn()` is used when available.

    Returns
    -------
    SubagentResult
        The structured result of the delegation.

    Examples
    --------
    ```python
    import talk_box as tb

    parent = tb.Agent.from_persona("project_manager")
    state = tb.SharedState()
    reviewer = tb.spawn(parent, "reviewer", persona="code_reviewer", shared_state=state)

    result = tb.delegate(parent, reviewer, "Review this function for bugs")
    result.response  # The reviewer's analysis
    result.duration  # How long it took

    state.get("delegated_task", namespace="reviewer")   # The task that was sent
    state.get("delegated_result", namespace="reviewer") # The response
    ```
    """
    # Resolve shared state: explicit > parent metadata > subagent metadata
    ss = shared_state
    if ss is None:
        ss = parent.metadata.get(_SHARED_STATE_KEY) or subagent.metadata.get(_SHARED_STATE_KEY)

    # Record task in shared state
    if ss is not None:
        ss.set("delegated_task", task, namespace=subagent.name, agent=parent.name)

    start = time.monotonic()
    conversation = subagent.respond(task)
    duration = time.monotonic() - start

    # Extract response text
    last = conversation.get_last_message()
    response_text = str(last.content) if last is not None else ""

    # Record result in shared state
    if ss is not None:
        ss.set(
            "delegated_result",
            response_text,
            namespace=subagent.name,
            agent=subagent.name,
        )

    return SubagentResult(
        response=response_text,
        agent=subagent.name,
        parent=parent.name,
        conversation=conversation,
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------


def children(agent: Agent) -> list[str]:
    """Return names of subagents spawned by this agent.

    Parameters
    ----------
    agent
        The agent to inspect.

    Returns
    -------
    list[str]
        Names of child agents, in spawn order.

    Examples
    --------
    ```python
    import talk_box as tb

    parent = tb.Agent.from_persona("project_manager")
    tb.spawn(parent, "a")
    tb.spawn(parent, "b")
    tb.children(parent)  # ["a", "b"]
    ```
    """
    return list(agent.metadata.get(_CHILDREN_KEY, []))


def parent_name(agent: Agent) -> str | None:
    """Return the name of the agent's parent, if any.

    Parameters
    ----------
    agent
        The agent to inspect.

    Returns
    -------
    str or None
        The parent agent name, or ``None`` if the agent was not spawned.

    Examples
    --------
    ```python
    import talk_box as tb

    parent = tb.Agent.from_persona("project_manager")
    child = tb.spawn(parent, "child")
    tb.parent_name(child)   # "project_manager"
    tb.parent_name(parent)  # None
    ```
    """
    return agent.metadata.get(_PARENT_KEY)
