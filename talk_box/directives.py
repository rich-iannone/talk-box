"""User directives: parse @context, @relates-to, @confidential, @expires from document text."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Directive types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextDirective:
    """An ``@context`` directive assigning explicit topic context.

    Parameters
    ----------
    value
        The context string (e.g. ``"Acme Corp partnership"``).
    line
        The 1-based line number where the directive appears.

    Examples
    --------
    ```python
    import talk_box as tb

    directives = tb.parse_directives("@context: Acme Corp partnership")
    directives.contexts[0].value  # "Acme Corp partnership"
    ```
    """

    value: str
    line: int = 0


@dataclass(frozen=True)
class RelatesToDirective:
    """An ``@relates-to`` directive creating a manual relationship edge.

    Parameters
    ----------
    target
        The target entity or document name.
    line
        The 1-based line number where the directive appears.

    Examples
    --------
    ```python
    import talk_box as tb

    directives = tb.parse_directives("@relates-to: Project Alpha")
    directives.relates_to[0].target  # "Project Alpha"
    ```
    """

    target: str
    line: int = 0


@dataclass(frozen=True)
class ConfidentialDirective:
    """An ``@confidential`` directive marking content as access-controlled.

    Parameters
    ----------
    line
        The 1-based line number where the directive appears.

    Examples
    --------
    ```python
    import talk_box as tb

    directives = tb.parse_directives("@confidential")
    directives.is_confidential  # True
    ```
    """

    line: int = 0


@dataclass(frozen=True)
class ExpiresDirective:
    """An ``@expires`` directive setting a temporal relevance cutoff.

    Parameters
    ----------
    date_str
        The expiration date as a string (e.g. ``"2026-12-31"``).
    line
        The 1-based line number where the directive appears.

    Examples
    --------
    ```python
    import talk_box as tb

    directives = tb.parse_directives("@expires: 2026-12-31")
    directives.expires[0].date_str        # "2026-12-31"
    directives.expires[0].is_expired()    # depends on current date
    ```
    """

    date_str: str
    line: int = 0

    def is_expired(self, *, now: float | None = None) -> bool:
        """Check whether this directive's date has passed.

        Parameters
        ----------
        now
            Unix timestamp to compare against.  Defaults to
            the current time.

        Returns
        -------
        bool
            ``True`` if the expiration date has passed.
        """
        from datetime import datetime

        now_ts = now if now is not None else time.time()
        try:
            expires_ts = datetime.strptime(self.date_str, "%Y-%m-%d").timestamp()
        except ValueError:
            return False
        return now_ts > expires_ts


# ---------------------------------------------------------------------------
# Parsed directives container
# ---------------------------------------------------------------------------


Directive = ContextDirective | RelatesToDirective | ConfidentialDirective | ExpiresDirective


@dataclass
class ParsedDirectives:
    """All directives parsed from a document.

    Parameters
    ----------
    contexts
        List of ``@context`` directives.
    relates_to
        List of ``@relates-to`` directives.
    confidentials
        List of ``@confidential`` directives.
    expires
        List of ``@expires`` directives.

    Examples
    --------
    ```python
    import talk_box as tb

    d = tb.parse_directives('''
    @context: Q3 Planning
    @relates-to: Project Alpha
    @confidential
    @expires: 2026-12-31
    ''')
    d.is_confidential        # True
    d.context_values         # ["Q3 Planning"]
    d.relates_to_targets     # ["Project Alpha"]
    d.all_directives         # list of 4 Directive objects
    ```
    """

    contexts: list[ContextDirective] = field(default_factory=list)
    relates_to: list[RelatesToDirective] = field(default_factory=list)
    confidentials: list[ConfidentialDirective] = field(default_factory=list)
    expires: list[ExpiresDirective] = field(default_factory=list)

    @property
    def is_confidential(self) -> bool:
        """Whether any ``@confidential`` directive was found."""
        return len(self.confidentials) > 0

    @property
    def context_values(self) -> list[str]:
        """Values from all ``@context`` directives."""
        return [c.value for c in self.contexts]

    @property
    def relates_to_targets(self) -> list[str]:
        """Targets from all ``@relates-to`` directives."""
        return [r.target for r in self.relates_to]

    @property
    def is_expired(self) -> bool:
        """Whether any ``@expires`` directive has passed."""
        return any(e.is_expired() for e in self.expires)

    @property
    def directive_count(self) -> int:
        """Total number of directives found."""
        return (
            len(self.contexts) + len(self.relates_to) + len(self.confidentials) + len(self.expires)
        )

    @property
    def all_directives(self) -> list[Directive]:
        """All parsed directives in source order."""
        items: list[Directive] = []
        items.extend(self.contexts)
        items.extend(self.relates_to)
        items.extend(self.confidentials)
        items.extend(self.expires)
        items.sort(key=lambda d: d.line)
        return items

    def to_metadata(self) -> dict[str, Any]:
        """Convert directives to a metadata dict for graph storage.

        Returns
        -------
        dict[str, Any]
            Metadata suitable for storing on a knowledge graph node.

        Examples
        --------
        ```python
        meta = directives.to_metadata()
        meta["_confidential"]  # True
        meta["_contexts"]      # ["Q3 Planning"]
        ```
        """
        meta: dict[str, Any] = {}
        if self.contexts:
            meta["_contexts"] = self.context_values
        if self.relates_to:
            meta["_relates_to"] = self.relates_to_targets
        if self.confidentials:
            meta["_confidential"] = True
        if self.expires:
            meta["_expires"] = [e.date_str for e in self.expires]
        return meta

    def __repr__(self) -> str:
        return (
            f"ParsedDirectives(contexts={len(self.contexts)}, "
            f"relates_to={len(self.relates_to)}, "
            f"confidential={self.is_confidential}, "
            f"expires={len(self.expires)})"
        )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

# Patterns for each directive type
_CONTEXT_RE = re.compile(r"^@context:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_RELATES_TO_RE = re.compile(r"^@relates-to:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_CONFIDENTIAL_RE = re.compile(r"^@confidential\s*$", re.IGNORECASE | re.MULTILINE)
_EXPIRES_RE = re.compile(r"^@expires:\s*(\S+)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_directives(text: str) -> ParsedDirectives:
    """Parse user directives from document text.

    Scans for ``@context``, ``@relates-to``, ``@confidential``, and
    ``@expires`` directives.  Each directive must appear at the start
    of a line (ignoring leading whitespace).

    Parameters
    ----------
    text
        The document text to scan.

    Returns
    -------
    ParsedDirectives
        Container with all parsed directives.

    Examples
    --------
    ```python
    import talk_box as tb

    d = tb.parse_directives(\"\"\"
    Meeting notes from Monday.
    @context: Q3 Planning
    @relates-to: Project Alpha
    @confidential
    @expires: 2026-12-31
    \"\"\")
    d.directive_count   # 4
    d.is_confidential   # True
    ```
    """
    # Strip leading whitespace from each line for matching
    # but preserve line numbers
    lines = text.split("\n")
    stripped_text = "\n".join(line.strip() for line in lines)

    result = ParsedDirectives()

    for match in _CONTEXT_RE.finditer(stripped_text):
        line_num = stripped_text[: match.start()].count("\n") + 1
        result.contexts.append(ContextDirective(value=match.group(1).strip(), line=line_num))

    for match in _RELATES_TO_RE.finditer(stripped_text):
        line_num = stripped_text[: match.start()].count("\n") + 1
        result.relates_to.append(RelatesToDirective(target=match.group(1).strip(), line=line_num))

    for match in _CONFIDENTIAL_RE.finditer(stripped_text):
        line_num = stripped_text[: match.start()].count("\n") + 1
        result.confidentials.append(ConfidentialDirective(line=line_num))

    for match in _EXPIRES_RE.finditer(stripped_text):
        line_num = stripped_text[: match.start()].count("\n") + 1
        result.expires.append(ExpiresDirective(date_str=match.group(1).strip(), line=line_num))

    return result


# ---------------------------------------------------------------------------
# strip_directives
# ---------------------------------------------------------------------------


def strip_directives(text: str) -> str:
    """Remove directive lines from document text.

    Returns the text with all directive lines removed, useful for
    getting the "clean" content without directives.

    Parameters
    ----------
    text
        The document text.

    Returns
    -------
    str
        Text with directive lines removed.

    Examples
    --------
    ```python
    import talk_box as tb

    clean = tb.strip_directives("Hello\\n@confidential\\nWorld")
    clean  # "Hello\\nWorld"
    ```
    """
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _CONTEXT_RE.match(stripped):
            continue
        if _RELATES_TO_RE.match(stripped):
            continue
        if _CONFIDENTIAL_RE.match(stripped):
            continue
        if _EXPIRES_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# apply_directives: integrate with knowledge graph
# ---------------------------------------------------------------------------


def apply_directives(
    kg: Any,
    node_id: str,
    directives: ParsedDirectives,
) -> ApplyResult:
    """Apply parsed directives to a node in the knowledge graph.

    Updates the node's metadata with directive information and creates
    ``relates_to`` edges for ``@relates-to`` directives.

    Parameters
    ----------
    kg
        A :class:`~talk_box.knowledge_graph.KnowledgeGraph` instance.
    node_id
        The ID of the node to update.
    directives
        Parsed directives to apply.

    Returns
    -------
    ApplyResult
        Summary of what was applied.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    # ... add a document node ...
    directives = tb.parse_directives(node.content)
    result = tb.apply_directives(kg, node.id, directives)
    result.metadata_updated  # True
    ```
    """
    from talk_box.knowledge_graph import Edge, Node

    node = kg.get_node(node_id)
    if node is None:
        return ApplyResult()

    # Merge directive metadata into node metadata
    meta = dict(node.metadata)
    directive_meta = directives.to_metadata()
    meta.update(directive_meta)
    meta["_has_directives"] = directives.directive_count > 0

    updated_node = Node(
        id=node.id,
        node_type=node.node_type,
        name=node.name,
        content=node.content,
        metadata=meta,
        embedding=node.embedding,
        created_at=node.created_at,
    )
    kg.add_node(updated_node)
    metadata_updated = True

    # Create edges for @relates-to directives
    edges_created = 0
    for rt in directives.relates_to:
        # Search for a node matching the target name
        matches = kg.search(rt.target, limit=1)
        if matches:
            target_node = matches[0]
            if target_node.id != node_id:
                edge = Edge(
                    source=node_id,
                    target=target_node.id,
                    relation="relates_to",
                    metadata={"directive": True},
                )
                kg.add_edge(edge)
                edges_created += 1

    return ApplyResult(
        metadata_updated=metadata_updated,
        edges_created=edges_created,
        directives_applied=directives.directive_count,
    )


# ---------------------------------------------------------------------------
# ApplyResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplyResult:
    """Result of applying directives to a knowledge graph node.

    Parameters
    ----------
    metadata_updated
        Whether node metadata was updated.
    edges_created
        Number of ``relates_to`` edges created.
    directives_applied
        Total number of directives that were applied.

    Examples
    --------
    ```python
    result = tb.apply_directives(kg, "doc-1", directives)
    result.edges_created  # 1
    ```
    """

    metadata_updated: bool = False
    edges_created: int = 0
    directives_applied: int = 0

    def __repr__(self) -> str:
        return (
            f"ApplyResult(metadata_updated={self.metadata_updated}, "
            f"edges={self.edges_created}, "
            f"directives={self.directives_applied})"
        )
