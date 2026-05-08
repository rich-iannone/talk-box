"""Replay mode: re-run captured conversations against different models or prompts."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from talk_box.capture import ConversationCapture

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# A responder takes (model_key, prompt) and returns response text.
Responder = Callable[[str, str], str]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayTurn:
    """A single turn comparing original and replayed responses.

    Parameters
    ----------
    turn_number
        The turn number (1-based).
    prompt
        The prompt text that was sent.
    original_response
        The response from the original capture (empty if missing).
    replayed_response
        The response from the replay responder.
    original_model
        Model used in the original capture.
    replay_model
        Model used in the replay.
    duration_ms
        Time taken for the replay response in milliseconds.
    """

    turn_number: int
    prompt: str
    original_response: str
    replayed_response: str
    original_model: str = ""
    replay_model: str = ""
    duration_ms: float | None = None


@dataclass(frozen=True)
class ReplayResult:
    """Result of replaying a captured conversation.

    Parameters
    ----------
    original_session_id
        Session ID of the original capture.
    replay_session_id
        Session ID of the replay capture.
    replay_model
        Model used for the replay.
    turns
        List of turn-by-turn comparisons.
    capture
        The full ``ConversationCapture`` from the replay session.
    system_prompt
        System prompt used during replay, if any.
    metadata
        Additional metadata about the replay.
    """

    original_session_id: str
    replay_session_id: str
    replay_model: str
    turns: list[ReplayTurn]
    capture: ConversationCapture
    system_prompt: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        """Number of turns in the replay."""
        return len(self.turns)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def replay(
    source: ConversationCapture,
    responder: Responder,
    *,
    model: str = "",
    system_prompt: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReplayResult:
    """Replay a captured conversation against a different model or prompt.

    Extracts user prompts from the source capture and sends each one to the
    ``responder`` callable. A new ``ConversationCapture`` is recorded during
    replay, enabling further analysis, export, or nested replays.

    Parameters
    ----------
    source
        The original ``ConversationCapture`` to replay.
    responder
        A callable ``(model_key, prompt) -> response_text`` that generates
        responses. This follows the same pattern as ``cascade()``.
    model
        Model identifier for the replay (passed to the responder and recorded
        in the replay capture).
    system_prompt
        Optional system prompt to inject before the first user prompt. If
        ``None``, the original system prompt (if any) is reused.
    metadata
        Additional metadata to attach to the replay capture.

    Returns
    -------
    ReplayResult
        Turn-by-turn comparison of original and replayed responses, plus the
        full replay capture.

    Raises
    ------
    ValueError
        If the source capture contains no user prompts.

    Examples
    --------
    ```python
    import talk_box as tb

    # Load a previously captured conversation
    original = tb.ConversationCapture.from_json("session.json")

    # Define a responder (could wrap any LLM call)
    def my_responder(model_key: str, prompt: str) -> str:
        return f"Response from {model_key}: {prompt[:50]}..."

    # Replay against a different model
    result = tb.replay(original, my_responder, model="openai:gpt-4o-mini")

    # Inspect turn-by-turn comparisons
    for turn in result.turns:
        print(f"Turn {turn.turn_number}:")
        print(f"  Prompt: {turn.prompt}")
        print(f"  Original: {turn.original_response}")
        print(f"  Replayed: {turn.replayed_response}")

    # Save the replay capture for later analysis
    result.capture.to_json("replay_session.json")
    ```
    """
    # Extract user prompts (not system prompts) from the original capture
    user_prompts = [
        e for e in source.prompts() if e.role != "system"
    ]

    if not user_prompts:
        raise ValueError("Source capture contains no user prompts to replay.")

    # Build original responses lookup (index-matched to user prompts)
    original_responses = source.responses()

    # Detect original system prompt
    original_system_prompts = [
        e for e in source.prompts() if e.role == "system"
    ]
    effective_system_prompt = (
        system_prompt
        if system_prompt is not None
        else (original_system_prompts[0].content if original_system_prompts else "")
    )

    # Build replay metadata
    replay_meta = dict(metadata or {})
    replay_meta["replay_of"] = source.session_id
    replay_meta["replay_model"] = model
    if effective_system_prompt:
        replay_meta["system_prompt"] = effective_system_prompt

    # Create the replay capture
    capture = ConversationCapture(metadata=replay_meta)

    # Record system prompt if present
    if effective_system_prompt:
        capture.record_prompt(effective_system_prompt, role="system")

    # Replay each user prompt
    replay_turns: list[ReplayTurn] = []

    for i, prompt_event in enumerate(user_prompts):
        # Record the prompt
        capture.record_prompt(prompt_event.content)

        # Get original response for comparison
        original_response = (
            original_responses[i].content if i < len(original_responses) else ""
        )
        original_model = (
            original_responses[i].model if i < len(original_responses) else ""
        )

        # Call the responder
        start = time.time()
        try:
            replayed_text = responder(model, prompt_event.content)
        except Exception as exc:
            # Record the error and continue
            capture.record_error(str(exc), error_type=type(exc).__name__)
            replayed_text = ""

        elapsed_ms = (time.time() - start) * 1000

        # Record the replay response
        if replayed_text:
            capture.record_response(replayed_text, model=model, duration_ms=elapsed_ms)

        # Build turn comparison
        turn = ReplayTurn(
            turn_number=i + 1,
            prompt=prompt_event.content,
            original_response=original_response,
            replayed_response=replayed_text,
            original_model=original_model,
            replay_model=model,
            duration_ms=elapsed_ms,
        )
        replay_turns.append(turn)

    return ReplayResult(
        original_session_id=source.session_id,
        replay_session_id=capture.session_id,
        replay_model=model,
        turns=replay_turns,
        capture=capture,
        system_prompt=effective_system_prompt,
        metadata=replay_meta,
    )
