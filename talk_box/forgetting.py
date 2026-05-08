"""Forgetting policies: automatic memory lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass

from talk_box.memory import MemoryEntry, MemoryStore, MemoryTier

# ---------------------------------------------------------------------------
# Policy results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    """Result of applying a forgetting policy to a memory store.

    Parameters
    ----------
    removed
        Keys that were removed.
    retained
        Keys that were kept.
    policy
        Name of the policy that produced this result.
    """

    removed: list[str]
    retained: list[str]
    policy: str

    @property
    def removed_count(self) -> int:
        """Number of entries removed."""
        return len(self.removed)

    @property
    def retained_count(self) -> int:
        """Number of entries retained."""
        return len(self.retained)


# ---------------------------------------------------------------------------
# forget_after_resolution
# ---------------------------------------------------------------------------


def forget_after_resolution(
    store: MemoryStore,
    resolved_keys: list[str],
    *,
    tier: MemoryTier | None = None,
) -> PolicyResult:
    """Remove memories that have been resolved or acted upon.

    This policy removes specific keys that are no longer needed — for example,
    a question that has been answered, a task that has been completed, or a
    pending action that has been fulfilled.

    Parameters
    ----------
    store
        The memory store to apply the policy to.
    resolved_keys
        List of keys to remove because they have been resolved.
    tier
        Specific tier to remove from. ``None`` removes from all tiers.

    Returns
    -------
    PolicyResult
        Summary of what was removed and retained.

    Examples
    --------
    ```python
    import talk_box as tb

    store = tb.MemoryStore()
    store.remember("pending_question", "What is the deadline?", tier=tb.MemoryTier.SHORT_TERM)
    store.remember("user_name", "Alice", tier=tb.MemoryTier.SHORT_TERM)

    # After the question is answered, forget it
    result = tb.forget_after_resolution(store, ["pending_question"])
    result.removed   # ["pending_question"]
    result.retained  # ["user_name"]
    ```
    """
    removed: list[str] = []
    for key in resolved_keys:
        if store.forget(key, tier=tier):
            removed.append(key)

    # Collect remaining keys across requested tiers
    remaining = _collect_keys(store, tier=tier)

    return PolicyResult(
        removed=removed,
        retained=remaining,
        policy="forget_after_resolution",
    )


# ---------------------------------------------------------------------------
# compress_after_n_turns
# ---------------------------------------------------------------------------


def compress_after_n_turns(
    store: MemoryStore,
    turn_count: int,
    *,
    max_entries: int,
    tier: MemoryTier | None = None,
    preserve_tags: list[str] | None = None,
) -> PolicyResult:
    """Remove oldest memories when the conversation exceeds a turn threshold.

    After ``turn_count`` turns, if there are more than ``max_entries`` memories
    in the specified tier(s), the oldest entries are removed until the count is
    at or below ``max_entries``. Entries with any of the ``preserve_tags`` are
    never removed.

    Parameters
    ----------
    store
        The memory store to apply the policy to.
    turn_count
        The current conversation turn number. The policy only activates when
        this exceeds ``max_entries`` (i.e., the conversation is long enough to
        warrant cleanup).
    max_entries
        Maximum number of entries to retain after compression.
    tier
        Specific tier to compress. ``None`` compresses working and short-term
        tiers (long-term is excluded by default since it's meant to persist).
    preserve_tags
        Entries with any of these tags are never removed.

    Returns
    -------
    PolicyResult
        Summary of what was removed and retained.

    Examples
    --------
    ```python
    import talk_box as tb

    store = tb.MemoryStore()
    for i in range(20):
        store.remember(f"turn_{i}", f"data {i}", tier=tb.MemoryTier.WORKING)

    # After 10 turns, compress to keep only 5 entries
    result = tb.compress_after_n_turns(store, turn_count=10, max_entries=5)
    result.removed_count  # 15
    result.retained_count  # 5
    ```
    """
    if turn_count <= max_entries:
        remaining = _collect_keys(store, tier=tier)
        return PolicyResult(removed=[], retained=remaining, policy="compress_after_n_turns")

    # Determine which tiers to compress
    tiers_to_check = _resolve_tiers(tier, include_long_term=False)

    # Gather all entries from the targeted tiers
    entries: list[MemoryEntry] = []
    for t in tiers_to_check:
        entries.extend(store.search(tier=t))

    if len(entries) <= max_entries:
        remaining = _collect_keys(store, tier=tier)
        return PolicyResult(removed=[], retained=remaining, policy="compress_after_n_turns")

    # Split into preserved and candidates
    preserve_set = set(preserve_tags or [])
    preserved: list[MemoryEntry] = []
    candidates: list[MemoryEntry] = []

    for entry in entries:
        if preserve_set and preserve_set.intersection(set(entry.tags)):
            preserved.append(entry)
        else:
            candidates.append(entry)

    # Sort candidates by timestamp (oldest first)
    candidates.sort(key=lambda e: e.timestamp)

    # How many candidates can we keep?
    slots_for_candidates = max(0, max_entries - len(preserved))
    to_remove = (
        candidates[: len(candidates) - slots_for_candidates]
        if len(candidates) > slots_for_candidates
        else []
    )

    removed: list[str] = []
    for entry in to_remove:
        if store.forget(entry.key, tier=entry.tier):
            removed.append(entry.key)

    remaining = _collect_keys(store, tier=tier)

    return PolicyResult(
        removed=removed,
        retained=remaining,
        policy="compress_after_n_turns",
    )


# ---------------------------------------------------------------------------
# retain_only
# ---------------------------------------------------------------------------


def retain_only(
    store: MemoryStore,
    keys: list[str] | None = None,
    *,
    tags: list[str] | None = None,
    tier: MemoryTier | None = None,
) -> PolicyResult:
    """Keep only specified memories, removing everything else.

    Retains entries that match any of the provided keys OR have any of the
    provided tags. Everything else is removed from the specified tier(s).

    At least one of ``keys`` or ``tags`` must be provided.

    Parameters
    ----------
    store
        The memory store to apply the policy to.
    keys
        Keys to retain.
    tags
        Entries with any of these tags are retained.
    tier
        Specific tier to apply to. ``None`` applies to all tiers.

    Returns
    -------
    PolicyResult
        Summary of what was removed and retained.

    Raises
    ------
    ValueError
        If neither ``keys`` nor ``tags`` is provided.

    Examples
    --------
    ```python
    import talk_box as tb

    store = tb.MemoryStore()
    store.remember("user_name", "Alice", tier=tb.MemoryTier.WORKING, tags=("identity",))
    store.remember("temp_calc", "42", tier=tb.MemoryTier.WORKING)
    store.remember("scratch", "...", tier=tb.MemoryTier.WORKING)

    # Keep only identity-tagged entries
    result = tb.retain_only(store, tags=["identity"])
    result.retained  # ["user_name"]
    result.removed   # ["temp_calc", "scratch"]
    ```
    """
    if not keys and not tags:
        raise ValueError("At least one of 'keys' or 'tags' must be provided.")

    keep_keys = set(keys or [])
    keep_tags = set(tags or [])

    # Determine which tiers to check
    tiers_to_check = _resolve_tiers(tier, include_long_term=True)

    # Gather all entries
    entries: list[MemoryEntry] = []
    for t in tiers_to_check:
        entries.extend(store.search(tier=t))

    removed: list[str] = []
    retained: list[str] = []

    for entry in entries:
        should_keep = entry.key in keep_keys or (
            keep_tags and keep_tags.intersection(set(entry.tags))
        )
        if should_keep:
            retained.append(entry.key)
        else:
            if store.forget(entry.key, tier=entry.tier):
                removed.append(entry.key)

    return PolicyResult(
        removed=removed,
        retained=retained,
        policy="retain_only",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_tiers(tier: MemoryTier | None, *, include_long_term: bool = True) -> list[MemoryTier]:
    """Resolve which tiers to operate on."""
    if tier is not None:
        return [tier]
    if include_long_term:
        return [MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]
    return [MemoryTier.WORKING, MemoryTier.SHORT_TERM]


def _collect_keys(store: MemoryStore, *, tier: MemoryTier | None = None) -> list[str]:
    """Collect all keys from the specified tier(s)."""
    keys: list[str] = []
    tiers = _resolve_tiers(tier, include_long_term=True)
    for t in tiers:
        if t == MemoryTier.WORKING:
            keys.extend(store.working.keys())
        elif t == MemoryTier.SHORT_TERM:
            keys.extend(store.short_term.keys())
        elif t == MemoryTier.LONG_TERM:
            keys.extend(store.long_term.keys())
    return keys
