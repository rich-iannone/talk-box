"""Token usage and cost tracking for chat sessions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionUsage:
    """Accumulated token usage and cost for a chat session.

    Tracks input tokens, output tokens, and estimated cost across
    multiple turns in a conversation.

    Examples
    --------
    ```python
    usage = SessionUsage()
    usage.add_turn(input_tokens=120, output_tokens=350, cost=0.002)
    usage.add_turn(input_tokens=500, output_tokens=800, cost=0.008)
    print(usage.total_tokens)   # 1770
    print(usage.total_cost)     # 0.01
    print(usage.turns)          # 2
    ```
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_cost: float = 0.0
    turns: int = 0
    _turn_details: list[dict] = field(default_factory=list, repr=False)

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across all turns."""
        return self.input_tokens + self.output_tokens

    def add_turn(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record a single turn's usage."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_cost += cost
        self.turns += 1
        self._turn_details.append(
            {
                "turn": self.turns,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }
        )

    def to_dict(self) -> dict:
        """Serialize usage to a dictionary."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "turns": self.turns,
        }

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.input_tokens = 0
        self.output_tokens = 0
        self.total_cost = 0.0
        self.turns = 0
        self._turn_details.clear()

    def update_from_chat_session(self, chat_session) -> None:
        """Extract token usage from a chatlas Chat session.

        Reads the latest turn's tokens and cost from the chatlas session
        and records them.

        Parameters
        ----------
        chat_session
            A ``chatlas.Chat`` instance with ``get_tokens()`` and ``get_cost()``
            methods.
        """
        try:
            tokens_list = chat_session.get_tokens()
            if tokens_list:
                last = tokens_list[-1]
                inp = last.get("input", 0) or 0
                out = last.get("output", 0) or 0
            else:
                inp = out = 0
        except Exception:
            inp = out = 0

        try:
            cost = chat_session.get_cost(options="last")
        except Exception:
            cost = 0.0

        self.add_turn(input_tokens=inp, output_tokens=out, cost=cost)
