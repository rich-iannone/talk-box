from __future__ import annotations

from dataclasses import dataclass, field

from talk_box.forgetting import PolicyResult, retain_only
from talk_box.memory import MemoryStore, MemoryTier

__all__ = [
    "RetentionPolicy",
    "apply_retention",
]

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetentionPolicy:
    """Declarative rules for what a persona remembers and forgets.

    A retention policy defines which memory tags and keys are important to a
    persona, and which should be discarded. It can be attached to a
    `PersonaDefinition` and applied to a `MemoryStore` via
    `apply_retention()`.

    Parameters
    ----------
    remember_tags
        Tags that should always be retained (e.g., `["identity", "preference"]`).
    remember_keys
        Specific keys that should always be retained.
    forget_tags
        Tags that should be removed when the policy is applied
        (e.g., `["scratch", "temporary"]`).
    forget_keys
        Specific keys that should be removed when the policy is applied.
    tier
        Which memory tier to apply the policy to. `None` applies to all tiers.

    Examples
    --------
    ```python
    import talk_box as tb

    # A support persona that remembers user info but forgets scratch data
    policy = tb.RetentionPolicy(
        remember_tags=["identity", "preference", "issue"],
        forget_tags=["scratch", "debug"],
    )
    ```
    """

    remember_tags: list[str] = field(default_factory=list)
    remember_keys: list[str] = field(default_factory=list)
    forget_tags: list[str] = field(default_factory=list)
    forget_keys: list[str] = field(default_factory=list)
    tier: MemoryTier | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_retention(
    store: MemoryStore,
    policy: RetentionPolicy,
) -> PolicyResult:
    """Apply a retention policy to a memory store.

    Processes the policy in two steps:

    1. **Forget**: Remove entries matching `forget_keys` and `forget_tags`.
    2. **Retain**: If `remember_tags` or `remember_keys` are specified,
       keep only entries matching those criteria and remove everything else.

    If neither remember nor forget rules are specified, no changes are made.

    Parameters
    ----------
    store
        The memory store to apply the policy to.
    policy
        The retention policy defining what to remember and forget.

    Returns
    -------
    PolicyResult
        Summary of what was removed and retained.

    Examples
    --------
    ```python
    import talk_box as tb

    store = tb.MemoryStore()
    store.remember("user_name", "Alice", tier=tb.MemoryTier.WORKING, tags=("identity",))
    store.remember("scratch", "temp data", tier=tb.MemoryTier.WORKING, tags=("scratch",))
    store.remember("pref", "dark mode", tier=tb.MemoryTier.WORKING, tags=("preference",))

    policy = tb.RetentionPolicy(
        remember_tags=["identity", "preference"],
        forget_tags=["scratch"],
    )
    result = tb.apply_retention(store, policy)
    result.removed   # ["scratch"]
    result.retained  # ["user_name", "pref"]
    ```
    """
    has_forget = bool(policy.forget_keys or policy.forget_tags)
    has_remember = bool(policy.remember_keys or policy.remember_tags)

    if not has_forget and not has_remember:
        remaining = _collect_keys(store, tier=policy.tier)
        return PolicyResult(removed=[], retained=remaining, policy="apply_retention")

    removed: list[str] = []

    # Step 1: Explicit forget rules — remove matching entries
    if has_forget:
        tiers = _resolve_tiers(policy.tier)
        for t in tiers:
            entries = store.search(tier=t)
            forget_tag_set = set(policy.forget_tags)
            for entry in entries:
                should_forget = entry.key in policy.forget_keys or (
                    forget_tag_set and forget_tag_set.intersection(set(entry.tags))
                )
                if should_forget:
                    if store.forget(entry.key, tier=t):
                        removed.append(entry.key)

    # Step 2: Retain rules — keep only matching entries
    if has_remember:
        result = retain_only(
            store,
            keys=policy.remember_keys or None,
            tags=policy.remember_tags or None,
            tier=policy.tier,
        )
        # Merge any additional removals from retain_only
        for key in result.removed:
            if key not in removed:
                removed.append(key)

    remaining = _collect_keys(store, tier=policy.tier)

    return PolicyResult(
        removed=removed,
        retained=remaining,
        policy="apply_retention",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_tiers(tier: MemoryTier | None) -> list[MemoryTier]:
    """Resolve which tiers to operate on."""
    if tier is not None:
        return [tier]
    return [MemoryTier.WORKING, MemoryTier.SHORT_TERM, MemoryTier.LONG_TERM]


def _collect_keys(store: MemoryStore, *, tier: MemoryTier | None = None) -> list[str]:
    """Collect all keys from the specified tier(s)."""
    keys: list[str] = []
    for t in _resolve_tiers(tier):
        if t == MemoryTier.WORKING:
            keys.extend(store.working.keys())
        elif t == MemoryTier.SHORT_TERM:
            keys.extend(store.short_term.keys())
        elif t == MemoryTier.LONG_TERM:
            keys.extend(store.long_term.keys())
    return keys
