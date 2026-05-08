"""Knowledge graph connectors: ingest documents from files, directories, and Apple Notes."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

# ---------------------------------------------------------------------------
# Document: the unit of ingestion
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A document to ingest into the knowledge graph.

    Parameters
    ----------
    title
        Human-readable title or filename.
    content
        Full text content of the document.
    source
        Origin identifier (file path, URL, connector name).
    metadata
        Arbitrary key-value metadata (folder, tags, dates, etc.).

    Examples
    --------
    ```python
    import talk_box as tb

    doc = tb.Document(
        title="README.md",
        content="# My Project\\nThis is a project.",
        source="/path/to/README.md",
    )
    doc.title  # "README.md"
    ```
    """

    title: str
    content: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Connector protocol
# ---------------------------------------------------------------------------


class Connector(ABC):
    """Base class for knowledge graph connectors.

    Subclasses must implement :meth:`scan` which yields :class:`Document`
    instances.  The connector tracks which documents have been seen so
    that :meth:`sync` only re-ingests changed content.

    Parameters
    ----------
    name
        A short name identifying this connector instance.

    Examples
    --------
    ```python
    import talk_box as tb

    class MyConnector(tb.Connector):
        def __init__(self):
            super().__init__(name="my-source")

        def scan(self):
            yield tb.Document(title="hello", content="world")
    ```
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def scan(self) -> Iterator[Document]:
        """Yield documents from this source.

        Each call should yield all currently available documents.
        The sync engine uses content hashes to detect changes.
        """
        ...  # pragma: no cover

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# MarkdownDir connector
# ---------------------------------------------------------------------------


class MarkdownDir(Connector):
    """Ingest Markdown files from a directory.

    Recursively scans a directory for ``.md`` and ``.markdown`` files
    and yields each as a :class:`Document`.

    Parameters
    ----------
    path
        Path to the directory to scan.
    name
        Connector name (defaults to ``"markdown-dir"``).

    Examples
    --------
    ```python
    import talk_box as tb

    connector = tb.MarkdownDir("~/Documents/notes/")
    for doc in connector.scan():
        print(doc.title)
    ```
    """

    _EXTENSIONS = {".md", ".markdown"}

    def __init__(self, path: str | Path, *, name: str = "markdown-dir") -> None:
        super().__init__(name=name)
        self.path = Path(path).expanduser().resolve()

    def scan(self) -> Iterator[Document]:
        """Yield Markdown documents from the directory tree."""
        if not self.path.is_dir():
            return

        for file_path in sorted(self.path.rglob("*")):
            if file_path.suffix.lower() not in self._EXTENSIONS:
                continue
            if not file_path.is_file():
                continue

            content = file_path.read_text(encoding="utf-8", errors="replace")
            rel_path = file_path.relative_to(self.path)

            yield Document(
                title=file_path.stem,
                content=content,
                source=str(file_path),
                metadata={
                    "file_path": str(file_path),
                    "relative_path": str(rel_path),
                    "extension": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                    "modified_at": file_path.stat().st_mtime,
                },
            )

    def __repr__(self) -> str:
        return f"MarkdownDir(path={str(self.path)!r})"


# ---------------------------------------------------------------------------
# DirectoryConnector
# ---------------------------------------------------------------------------


class DirectoryConnector(Connector):
    """Ingest text-based files from a directory by extension.

    Scans a directory recursively and yields files matching the given
    extensions as :class:`Document` objects.  Binary files (e.g. PDF)
    are skipped with a note in metadata — full PDF extraction is left
    to enrichment plugins.

    Parameters
    ----------
    path
        Path to the directory to scan.
    extensions
        File extensions to include (e.g. ``[".txt", ".md", ".py"]``).
        Defaults to ``[".txt", ".md"]``.
    name
        Connector name (defaults to ``"directory"``).

    Examples
    --------
    ```python
    import talk_box as tb

    connector = tb.DirectoryConnector(
        "~/Documents/work/",
        extensions=[".txt", ".md", ".py"],
    )
    for doc in connector.scan():
        print(doc.title, len(doc.content))
    ```
    """

    _DEFAULT_EXTENSIONS = [".txt", ".md"]

    def __init__(
        self,
        path: str | Path,
        *,
        extensions: list[str] | None = None,
        name: str = "directory",
    ) -> None:
        super().__init__(name=name)
        self.path = Path(path).expanduser().resolve()
        self.extensions = {
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in (extensions or self._DEFAULT_EXTENSIONS)
        }

    def scan(self) -> Iterator[Document]:
        """Yield documents from the directory tree."""
        if not self.path.is_dir():
            return

        for file_path in sorted(self.path.rglob("*")):
            if file_path.suffix.lower() not in self.extensions:
                continue
            if not file_path.is_file():
                continue

            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue

            rel_path = file_path.relative_to(self.path)

            yield Document(
                title=file_path.stem,
                content=content,
                source=str(file_path),
                metadata={
                    "file_path": str(file_path),
                    "relative_path": str(rel_path),
                    "extension": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                    "modified_at": file_path.stat().st_mtime,
                },
            )

    def __repr__(self) -> str:
        exts = sorted(self.extensions)
        return f"DirectoryConnector(path={str(self.path)!r}, extensions={exts})"


# ---------------------------------------------------------------------------
# AppleNotes connector
# ---------------------------------------------------------------------------

# Default path on macOS
_APPLE_NOTES_DB = "~/Library/Group Containers/group.com.apple.notes/NoteStore.sqlite"

# Query to read notes.  The schema uses ZICCLOUDSYNCINGOBJECT for note
# records.  We join to ZICNOTEDATA (or ZICCLOUDSYNCINGOBJECT's own
# ZCONTENT) to get the body.  The body is stored as gzipped protobuf
# since macOS Catalina, but some older or simpler notes still have
# plaintext.  We read the ZPLAINTEXT / ZTITLE1 columns when available.
_APPLE_NOTES_SQL = """
SELECT
    n.Z_PK,
    n.ZTITLE1 AS title,
    nd.ZPLAINTEXT AS body,
    n.ZCREATIONDATE1 AS created,
    n.ZMODIFICATIONDATE1 AS modified,
    f.ZTITLE2 AS folder
FROM ZICCLOUDSYNCINGOBJECT n
LEFT JOIN ZICNOTEDATA nd ON nd.ZNOTE = n.Z_PK
LEFT JOIN ZICCLOUDSYNCINGOBJECT f ON f.Z_PK = n.ZFOLDER
WHERE n.ZTITLE1 IS NOT NULL
  AND n.ZMARKEDFORDELETION != 1
"""


class AppleNotes(Connector):
    """Ingest notes from macOS Apple Notes (read-only).

    Reads from the NoteStore.sqlite database.  Notes are **opt-in**:
    only notes in a specific folder or containing a tag are ingested.

    Parameters
    ----------
    folder
        Only ingest notes from this Apple Notes folder name
        (e.g. ``"Talk Box"``).  Case-insensitive.
    tag
        Only ingest notes containing this tag in the body
        (e.g. ``"#tb-knowledge"``).  Case-insensitive.
    db_path
        Override the database path (useful for testing).
    name
        Connector name (defaults to ``"apple-notes"``).

    Notes
    -----
    Apple Notes' SQLite schema is undocumented and may change between
    macOS versions.  This connector is best-effort and read-only.

    At least one of ``folder`` or ``tag`` must be provided to prevent
    accidental ingestion of all notes.

    Examples
    --------
    ```python
    import talk_box as tb

    connector = tb.AppleNotes(folder="Talk Box")
    for doc in connector.scan():
        print(doc.title)
    ```
    """

    def __init__(
        self,
        *,
        folder: str | None = None,
        tag: str | None = None,
        db_path: str | Path | None = None,
        name: str = "apple-notes",
    ) -> None:
        super().__init__(name=name)
        if folder is None and tag is None:
            raise ValueError("At least one of 'folder' or 'tag' must be provided")
        self.folder = folder
        self.tag = tag
        self.db_path = Path(db_path or _APPLE_NOTES_DB).expanduser().resolve()

    def scan(self) -> Iterator[Document]:
        """Yield documents from Apple Notes matching the folder/tag filter."""
        if not self.db_path.exists():
            return

        # Open read-only to avoid any risk of corruption
        uri = f"file:{self.db_path}?mode=ro"
        try:
            conn = sqlite3.connect(uri, uri=True)
        except sqlite3.OperationalError:
            return

        try:
            rows = conn.execute(_APPLE_NOTES_SQL).fetchall()
        except sqlite3.OperationalError:
            conn.close()
            return

        for row in rows:
            pk, title, body, created, modified, folder_name = row

            if not title or not body:
                continue

            # Apply folder filter
            if self.folder is not None:
                if folder_name is None or folder_name.lower() != self.folder.lower():
                    continue

            # Apply tag filter
            if self.tag is not None:
                if self.tag.lower() not in body.lower():
                    continue

            # Apple Notes stores dates as CoreData timestamps
            # (seconds since 2001-01-01).  Convert to Unix epoch.
            _CD_EPOCH_OFFSET = 978307200.0
            created_unix = (created + _CD_EPOCH_OFFSET) if created else time.time()
            modified_unix = (modified + _CD_EPOCH_OFFSET) if modified else time.time()

            yield Document(
                title=title,
                content=body,
                source=f"apple-notes://{pk}",
                metadata={
                    "apple_notes_pk": pk,
                    "folder": folder_name or "",
                    "created_at": created_unix,
                    "modified_at": modified_unix,
                },
            )

        conn.close()

    def __repr__(self) -> str:
        parts = ["AppleNotes("]
        if self.folder:
            parts.append(f"folder={self.folder!r}")
        if self.tag:
            if self.folder:
                parts.append(", ")
            parts.append(f"tag={self.tag!r}")
        parts.append(")")
        return "".join(parts)


# ---------------------------------------------------------------------------
# connector decorator for custom connectors
# ---------------------------------------------------------------------------


def connector(fn: Callable[[], Iterator[Document]]) -> Connector:
    """Create a connector from a generator function.

    The decorated function should yield :class:`Document` instances.

    Parameters
    ----------
    fn
        A callable that returns an iterator of documents.

    Returns
    -------
    Connector
        A connector wrapping the function.

    Examples
    --------
    ```python
    import talk_box as tb

    @tb.connector
    def my_source():
        yield tb.Document(title="hello", content="world")

    for doc in my_source.scan():
        print(doc.title)
    ```
    """

    class _FnConnector(Connector):
        def __init__(self) -> None:
            super().__init__(name=fn.__name__)
            self._fn = fn

        def scan(self) -> Iterator[Document]:
            return self._fn()

        def __repr__(self) -> str:
            return f"connector({self._fn.__name__})"

    return _FnConnector()


# ---------------------------------------------------------------------------
# Content hashing for incremental sync
# ---------------------------------------------------------------------------


def _content_hash(content: str) -> str:
    """SHA-256 hex digest of content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


def sync(
    kg: Any,
    *connectors: Connector,
    node_prefix: str = "doc",
) -> SyncResult:
    """Sync documents from connectors into a knowledge graph.

    Performs incremental sync: only adds or updates documents whose
    content has changed since the last sync.  Optionally removes
    nodes whose source documents no longer exist.

    Parameters
    ----------
    kg
        A :class:`~talk_box.knowledge_graph.KnowledgeGraph` instance.
    *connectors
        One or more :class:`Connector` instances to sync from.
    node_prefix
        Prefix for generated node IDs (default ``"doc"``).

    Returns
    -------
    SyncResult
        Summary of what was added, updated, and unchanged.

    Examples
    --------
    ```python
    import talk_box as tb

    kg = tb.KnowledgeGraph(":memory:")
    md = tb.MarkdownDir("~/notes/")
    result = tb.sync(kg, md)
    result.added    # 5
    result.updated  # 2
    ```
    """
    from talk_box.knowledge_graph import Node, NodeType

    added = 0
    updated = 0
    unchanged = 0

    for conn in connectors:
        for doc in conn.scan():
            # Deterministic node ID from connector + source
            node_id = _make_node_id(node_prefix, conn.name, doc)

            # Check if node already exists with same content
            existing = kg.get_node(node_id)
            new_hash = _content_hash(doc.content)

            if existing is not None:
                old_hash = existing.metadata.get("_content_hash", "")
                if old_hash == new_hash:
                    unchanged += 1
                    continue
                updated += 1
            else:
                added += 1

            # Build metadata, preserving doc metadata and adding sync info
            meta = dict(doc.metadata)
            meta["_content_hash"] = new_hash
            meta["_connector"] = conn.name
            meta["_source"] = doc.source
            meta["_synced_at"] = time.time()

            node = Node(
                id=node_id,
                node_type=NodeType.DOCUMENT,
                name=doc.title,
                content=doc.content,
                metadata=meta,
            )
            kg.add_node(node)

    return SyncResult(added=added, updated=updated, unchanged=unchanged)


def _make_node_id(prefix: str, connector_name: str, doc: Document) -> str:
    """Generate a deterministic node ID for a document."""
    # Use source if available, otherwise hash the title
    key = doc.source or doc.title
    digest = hashlib.sha256(f"{connector_name}:{key}".encode()).hexdigest()[:16]
    return f"{prefix}-{digest}"


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncResult:
    """Result of a connector sync operation.

    Parameters
    ----------
    added
        Number of new documents added.
    updated
        Number of existing documents updated (content changed).
    unchanged
        Number of documents skipped (content unchanged).

    Examples
    --------
    ```python
    result.added      # 5
    result.updated    # 2
    result.unchanged  # 10
    result.total      # 17
    ```
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0

    @property
    def total(self) -> int:
        """Total documents processed."""
        return self.added + self.updated + self.unchanged

    def __repr__(self) -> str:
        return f"SyncResult(added={self.added}, updated={self.updated}, unchanged={self.unchanged})"
