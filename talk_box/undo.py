"""Undo buffer for reversing file changes made by tool execution.

Stores snapshots of file contents before each write/edit so that changes
can be rolled back within the session.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class UndoEntry:
    """A single reversible file change.

    Parameters
    ----------
    action
        ``"write"`` or ``"edit"``.
    path
        Relative file path (as used by the tool).
    previous_content
        The file content before the change, or ``None`` if the file
        did not exist (i.e. a new-file write).
    resolved_path
        Absolute path on disk (used for the actual rollback).
    timestamp
        When the change was recorded.
    description
        Human-readable summary (e.g. ``"Wrote config.py"``).
    """

    action: str
    path: str
    previous_content: str | None
    resolved_path: str
    timestamp: datetime = field(default_factory=datetime.now)
    description: str = ""


class UndoBuffer:
    """Session-scoped buffer of reversible file changes.

    Entries are appended by the tool layer before each disk write.
    The TUI exposes ``/undo`` to pop and restore the most recent entry.
    """

    def __init__(self, *, max_entries: int = 200) -> None:
        self._entries: list[UndoEntry] = []
        self._max_entries = max_entries

    # -- public API --------------------------------------------------------

    def record(
        self,
        *,
        action: str,
        path: str,
        previous_content: str | None,
        resolved_path: str,
        description: str = "",
    ) -> None:
        """Record a file snapshot before a change is applied."""
        entry = UndoEntry(
            action=action,
            path=path,
            previous_content=previous_content,
            resolved_path=resolved_path,
            description=description or f"{action.capitalize()} {path}",
        )
        self._entries.append(entry)
        # Evict oldest if over limit
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

    def undo_last(self) -> UndoEntry | None:
        """Pop and revert the most recent change. Returns the entry, or None."""
        if not self._entries:
            return None
        entry = self._entries.pop()
        self._apply_rollback(entry)
        return entry

    def undo_all(self) -> list[UndoEntry]:
        """Revert all changes in reverse order. Returns the reverted entries."""
        reverted: list[UndoEntry] = []
        while self._entries:
            entry = self._entries.pop()
            self._apply_rollback(entry)
            reverted.append(entry)
        return reverted

    @property
    def entries(self) -> list[UndoEntry]:
        """Read-only copy of the undo stack (oldest first)."""
        return list(self._entries)

    def clear(self) -> None:
        """Discard all recorded entries without reverting."""
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _apply_rollback(entry: UndoEntry) -> None:
        """Restore a file to its pre-change state."""
        if entry.previous_content is None:
            # File was newly created — delete it
            try:
                os.remove(entry.resolved_path)
            except FileNotFoundError:
                pass
        else:
            # Restore original content
            with open(entry.resolved_path, "w") as f:
                f.write(entry.previous_content)
