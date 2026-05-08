from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

_GLOBAL_NS = "__global__"


@dataclass(frozen=True)
class StateChange:
    """Record of a single state mutation.

    Parameters
    ----------
    namespace
        The namespace where the change occurred.
    key
        The key that was changed.
    old_value
        The previous value (`None` if the key was new).
    new_value
        The new value (`None` if the key was deleted).
    timestamp
        Unix timestamp of the change.
    agent
        Name of the agent that made the change, if known.
    """

    namespace: str
    key: str
    old_value: Any
    new_value: Any
    timestamp: float
    agent: str = ""


# ---------------------------------------------------------------------------
# SharedState
# ---------------------------------------------------------------------------


class SharedState:
    """Thread-safe key-value store for sharing context between agents.

    `SharedState` provides a central place for multiple agents to read and write shared context
    during multi-agent workflows. Each agent can have its own namespace for private state, and all
    agents share a global namespace for cross-agent communication.

    All operations are thread-safe and mutations are tracked in a change history for auditability.

    Parameters
    ----------
    track_history
        Whether to record every mutation in the change history. Defaults to `True`.

    Examples
    --------
    Basic shared state between agents:

    ```python
    import talk_box as tb

    state = tb.SharedState()

    # Global namespace (accessible to all agents)
    state.set("customer_id", "C-1234")
    state.get("customer_id")  # "C-1234"

    # Per-agent namespaces
    state.set("diagnosis", "network issue", namespace="tech_agent")
    state.get("diagnosis", namespace="tech_agent")  # "network issue"
    state.get("diagnosis")  # None (not in global namespace)
    ```

    Inspect all shared context:

    ```python
    state.snapshot()
    # {"__global__": {"customer_id": "C-1234"},
    #  "tech_agent": {"diagnosis": "network issue"}}
    ```
    """

    def __init__(self, *, track_history: bool = True) -> None:
        self._store: dict[str, dict[str, Any]] = {_GLOBAL_NS: {}}
        self._history: list[StateChange] = []
        self._track_history = track_history
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def set(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = _GLOBAL_NS,
        agent: str = "",
    ) -> None:
        """Store a value in the shared state.

        Parameters
        ----------
        key
            The key to store under.
        value
            The value to store.
        namespace
            Namespace to store in. Defaults to the global namespace. Use an agent name for per-agent
            private state.
        agent
            Name of the agent making the change (for history tracking).
        """
        with self._lock:
            ns = self._store.setdefault(namespace, {})
            old_value = ns.get(key)
            ns[key] = value
            if self._track_history:
                self._history.append(
                    StateChange(
                        namespace=namespace,
                        key=key,
                        old_value=old_value,
                        new_value=value,
                        timestamp=time.time(),
                        agent=agent,
                    )
                )

    def get(
        self,
        key: str,
        default: Any = None,
        *,
        namespace: str = _GLOBAL_NS,
    ) -> Any:
        """Retrieve a value from the shared state.

        Parameters
        ----------
        key
            The key to look up.
        default
            Value to return if the key is not found.
        namespace
            Namespace to look in. Defaults to the global namespace.

        Returns
        -------
        Any
            The stored value, or *default*.
        """
        with self._lock:
            ns = self._store.get(namespace, {})
            return ns.get(key, default)

    def delete(
        self,
        key: str,
        *,
        namespace: str = _GLOBAL_NS,
        agent: str = "",
    ) -> bool:
        """Remove a key from the shared state.

        Parameters
        ----------
        key
            The key to remove.
        namespace
            Namespace to remove from. Defaults to the global namespace.
        agent
            Name of the agent making the change (for history tracking).

        Returns
        -------
        bool
            `True` if the key existed and was removed.
        """
        with self._lock:
            ns = self._store.get(namespace, {})
            if key not in ns:
                return False
            old_value = ns.pop(key)
            if self._track_history:
                self._history.append(
                    StateChange(
                        namespace=namespace,
                        key=key,
                        old_value=old_value,
                        new_value=None,
                        timestamp=time.time(),
                        agent=agent,
                    )
                )
            return True

    def has(self, key: str, *, namespace: str = _GLOBAL_NS) -> bool:
        """Check if a key exists in the shared state.

        Parameters
        ----------
        key
            The key to check.
        namespace
            Namespace to check in. Defaults to the global namespace.

        Returns
        -------
        bool
            `True` if the key exists.
        """
        with self._lock:
            return key in self._store.get(namespace, {})

    def keys(self, *, namespace: str = _GLOBAL_NS) -> list[str]:
        """List all keys in a namespace.

        Parameters
        ----------
        namespace
            Namespace to list keys from. Defaults to the global namespace.

        Returns
        -------
        list[str]
            All keys in the namespace.
        """
        with self._lock:
            return list(self._store.get(namespace, {}).keys())

    def namespaces(self) -> list[str]:
        """List all namespaces that have been written to.

        Returns
        -------
        list[str]
            All namespace names, including the global namespace.
        """
        with self._lock:
            return list(self._store.keys())

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Return a shallow copy of the entire state across all namespaces.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of namespace to key-value pairs.

        Examples
        --------
        ```python
        import talk_box as tb

        state = tb.SharedState()
        state.set("issue_id", "T-42")
        state.set("notes", "rebooted", namespace="tech")
        state.snapshot()
        # {"__global__": {"issue_id": "T-42"}, "tech": {"notes": "rebooted"}}
        ```
        """
        with self._lock:
            return {ns: dict(kv) for ns, kv in self._store.items()}

    def clear(self, *, namespace: str | None = None) -> None:
        """Clear state from a namespace or all namespaces.

        Parameters
        ----------
        namespace
            Namespace to clear. If `None`, clears everything.
        """
        with self._lock:
            if namespace is None:
                self._store.clear()
                self._store[_GLOBAL_NS] = {}
            elif namespace in self._store:
                self._store[namespace].clear()

    @property
    def history(self) -> list[StateChange]:
        """Return the list of all recorded state changes.

        Returns
        -------
        list[StateChange]
            Chronological list of mutations.
        """
        with self._lock:
            return list(self._history)

    def __len__(self) -> int:
        """Total number of keys across all namespaces."""
        with self._lock:
            return sum(len(kv) for kv in self._store.values())
