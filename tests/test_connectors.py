"""Tests for talk_box.connectors."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from talk_box.connectors import (
    AppleNotes,
    Connector,
    DirectoryConnector,
    Document,
    MarkdownDir,
    SyncResult,
    _content_hash,
    _make_node_id,
    connector,
    sync,
)
from talk_box.knowledge_graph import KnowledgeGraph, NodeType


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_basic_creation(self):
        doc = Document(title="hello", content="world")
        assert doc.title == "hello"
        assert doc.content == "world"
        assert doc.source == ""
        assert doc.metadata == {}

    def test_with_metadata(self):
        doc = Document(
            title="test",
            content="body",
            source="/path/to/file",
            metadata={"key": "value"},
        )
        assert doc.source == "/path/to/file"
        assert doc.metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# MarkdownDir
# ---------------------------------------------------------------------------


class TestMarkdownDir:
    def test_creation(self, tmp_path: Path):
        md = MarkdownDir(tmp_path)
        assert md.name == "markdown-dir"
        assert md.path == tmp_path

    def test_custom_name(self, tmp_path: Path):
        md = MarkdownDir(tmp_path, name="my-notes")
        assert md.name == "my-notes"

    def test_scan_empty_dir(self, tmp_path: Path):
        md = MarkdownDir(tmp_path)
        docs = list(md.scan())
        assert docs == []

    def test_scan_nonexistent_dir(self):
        md = MarkdownDir("/nonexistent/path/xyz")
        docs = list(md.scan())
        assert docs == []

    def test_scan_md_files(self, tmp_path: Path):
        (tmp_path / "readme.md").write_text("# Hello", encoding="utf-8")
        (tmp_path / "notes.markdown").write_text("Some notes", encoding="utf-8")
        (tmp_path / "ignore.txt").write_text("Not a markdown file", encoding="utf-8")

        md = MarkdownDir(tmp_path)
        docs = list(md.scan())
        assert len(docs) == 2
        titles = {d.title for d in docs}
        assert titles == {"readme", "notes"}

    def test_scan_recursive(self, tmp_path: Path):
        subdir = tmp_path / "sub" / "deep"
        subdir.mkdir(parents=True)
        (tmp_path / "top.md").write_text("top level", encoding="utf-8")
        (subdir / "nested.md").write_text("nested file", encoding="utf-8")

        md = MarkdownDir(tmp_path)
        docs = list(md.scan())
        assert len(docs) == 2

    def test_scan_metadata(self, tmp_path: Path):
        f = tmp_path / "test.md"
        f.write_text("content", encoding="utf-8")

        md = MarkdownDir(tmp_path)
        docs = list(md.scan())
        assert len(docs) == 1

        doc = docs[0]
        assert doc.title == "test"
        assert doc.content == "content"
        assert doc.source == str(f)
        assert doc.metadata["file_path"] == str(f)
        assert doc.metadata["relative_path"] == "test.md"
        assert doc.metadata["extension"] == ".md"
        assert doc.metadata["size_bytes"] > 0
        assert "modified_at" in doc.metadata

    def test_case_insensitive_extensions(self, tmp_path: Path):
        (tmp_path / "upper.MD").write_text("caps", encoding="utf-8")
        md = MarkdownDir(tmp_path)
        docs = list(md.scan())
        assert len(docs) == 1

    def test_repr(self, tmp_path: Path):
        md = MarkdownDir(tmp_path)
        r = repr(md)
        assert "MarkdownDir" in r
        assert str(tmp_path) in r


# ---------------------------------------------------------------------------
# DirectoryConnector
# ---------------------------------------------------------------------------


class TestDirectoryConnector:
    def test_creation(self, tmp_path: Path):
        dc = DirectoryConnector(tmp_path)
        assert dc.name == "directory"
        assert dc.path == tmp_path
        assert ".txt" in dc.extensions
        assert ".md" in dc.extensions

    def test_custom_extensions(self, tmp_path: Path):
        dc = DirectoryConnector(tmp_path, extensions=[".py", ".rs", "toml"])
        assert ".py" in dc.extensions
        assert ".rs" in dc.extensions
        assert ".toml" in dc.extensions

    def test_scan_filters_by_extension(self, tmp_path: Path):
        (tmp_path / "a.txt").write_text("text file", encoding="utf-8")
        (tmp_path / "b.py").write_text("python file", encoding="utf-8")
        (tmp_path / "c.md").write_text("markdown", encoding="utf-8")

        dc = DirectoryConnector(tmp_path, extensions=[".txt"])
        docs = list(dc.scan())
        assert len(docs) == 1
        assert docs[0].title == "a"

    def test_scan_empty_dir(self, tmp_path: Path):
        dc = DirectoryConnector(tmp_path)
        docs = list(dc.scan())
        assert docs == []

    def test_scan_nonexistent(self):
        dc = DirectoryConnector("/nonexistent/xyz")
        docs = list(dc.scan())
        assert docs == []

    def test_scan_metadata(self, tmp_path: Path):
        f = tmp_path / "data.txt"
        f.write_text("hello", encoding="utf-8")

        dc = DirectoryConnector(tmp_path, extensions=[".txt"])
        docs = list(dc.scan())
        assert len(docs) == 1

        doc = docs[0]
        assert doc.metadata["file_path"] == str(f)
        assert doc.metadata["extension"] == ".txt"

    def test_scan_recursive(self, tmp_path: Path):
        sub = tmp_path / "child"
        sub.mkdir()
        (tmp_path / "a.txt").write_text("1", encoding="utf-8")
        (sub / "b.txt").write_text("2", encoding="utf-8")

        dc = DirectoryConnector(tmp_path, extensions=[".txt"])
        docs = list(dc.scan())
        assert len(docs) == 2

    def test_repr(self, tmp_path: Path):
        dc = DirectoryConnector(tmp_path, extensions=[".py", ".txt"])
        r = repr(dc)
        assert "DirectoryConnector" in r
        assert str(tmp_path) in r


# ---------------------------------------------------------------------------
# AppleNotes
# ---------------------------------------------------------------------------


def _create_mock_notes_db(db_path: Path) -> None:
    """Create a minimal mock Apple Notes database."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE ZICCLOUDSYNCINGOBJECT (
            Z_PK INTEGER PRIMARY KEY,
            ZTITLE1 TEXT,
            ZCREATIONDATE1 REAL,
            ZMODIFICATIONDATE1 REAL,
            ZFOLDER INTEGER,
            ZTITLE2 TEXT,
            ZMARKEDFORDELETION INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE ZICNOTEDATA (
            Z_PK INTEGER PRIMARY KEY,
            ZNOTE INTEGER,
            ZPLAINTEXT TEXT
        )
    """)

    # Insert a folder
    conn.execute("""
        INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, ZTITLE2, ZMARKEDFORDELETION)
        VALUES (100, 'Talk Box', 0)
    """)

    # Insert notes
    # CoreData epoch offset: 978307200 seconds
    cd_time = time.time() - 978307200.0

    # Note in "Talk Box" folder
    conn.execute(
        """
        INSERT INTO ZICCLOUDSYNCINGOBJECT
        (Z_PK, ZTITLE1, ZCREATIONDATE1, ZMODIFICATIONDATE1, ZFOLDER, ZMARKEDFORDELETION)
        VALUES (1, 'My Note', ?, ?, 100, 0)
    """,
        (cd_time, cd_time),
    )
    conn.execute("""
        INSERT INTO ZICNOTEDATA (Z_PK, ZNOTE, ZPLAINTEXT)
        VALUES (1, 1, 'This is my note content with #tb-knowledge tag')
    """)

    # Note in different folder
    conn.execute("""
        INSERT INTO ZICCLOUDSYNCINGOBJECT (Z_PK, ZTITLE2, ZMARKEDFORDELETION)
        VALUES (200, 'Other Folder', 0)
    """)
    conn.execute(
        """
        INSERT INTO ZICCLOUDSYNCINGOBJECT
        (Z_PK, ZTITLE1, ZCREATIONDATE1, ZMODIFICATIONDATE1, ZFOLDER, ZMARKEDFORDELETION)
        VALUES (2, 'Other Note', ?, ?, 200, 0)
    """,
        (cd_time, cd_time),
    )
    conn.execute("""
        INSERT INTO ZICNOTEDATA (Z_PK, ZNOTE, ZPLAINTEXT)
        VALUES (2, 2, 'A note in another folder')
    """)

    # Deleted note (should be excluded)
    conn.execute(
        """
        INSERT INTO ZICCLOUDSYNCINGOBJECT
        (Z_PK, ZTITLE1, ZCREATIONDATE1, ZMODIFICATIONDATE1, ZFOLDER, ZMARKEDFORDELETION)
        VALUES (3, 'Deleted Note', ?, ?, 100, 1)
    """,
        (cd_time, cd_time),
    )
    conn.execute("""
        INSERT INTO ZICNOTEDATA (Z_PK, ZNOTE, ZPLAINTEXT)
        VALUES (3, 3, 'This note was deleted')
    """)

    conn.commit()
    conn.close()


class TestAppleNotes:
    def test_requires_folder_or_tag(self):
        with pytest.raises(ValueError, match="folder.*tag"):
            AppleNotes()

    def test_folder_filter(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        _create_mock_notes_db(db)

        an = AppleNotes(folder="Talk Box", db_path=db)
        docs = list(an.scan())
        assert len(docs) == 1
        assert docs[0].title == "My Note"

    def test_folder_case_insensitive(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        _create_mock_notes_db(db)

        an = AppleNotes(folder="talk box", db_path=db)
        docs = list(an.scan())
        assert len(docs) == 1

    def test_tag_filter(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        _create_mock_notes_db(db)

        an = AppleNotes(tag="#tb-knowledge", db_path=db)
        docs = list(an.scan())
        assert len(docs) == 1
        assert docs[0].title == "My Note"

    def test_tag_case_insensitive(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        _create_mock_notes_db(db)

        an = AppleNotes(tag="#TB-KNOWLEDGE", db_path=db)
        docs = list(an.scan())
        assert len(docs) == 1

    def test_excludes_deleted(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        _create_mock_notes_db(db)

        # Use tag that would match the deleted note too
        an = AppleNotes(folder="Talk Box", db_path=db)
        docs = list(an.scan())
        titles = {d.title for d in docs}
        assert "Deleted Note" not in titles

    def test_nonexistent_db(self):
        an = AppleNotes(folder="Test", db_path="/nonexistent/NoteStore.sqlite")
        docs = list(an.scan())
        assert docs == []

    def test_document_metadata(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        _create_mock_notes_db(db)

        an = AppleNotes(folder="Talk Box", db_path=db)
        docs = list(an.scan())
        assert len(docs) == 1

        doc = docs[0]
        assert doc.source.startswith("apple-notes://")
        assert doc.metadata["apple_notes_pk"] == 1
        assert doc.metadata["folder"] == "Talk Box"
        assert "created_at" in doc.metadata
        assert "modified_at" in doc.metadata

    def test_repr_folder(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        an = AppleNotes(folder="Test", db_path=db)
        assert "AppleNotes" in repr(an)
        assert "Test" in repr(an)

    def test_repr_tag(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        an = AppleNotes(tag="#tb", db_path=db)
        assert "#tb" in repr(an)

    def test_custom_name(self, tmp_path: Path):
        db = tmp_path / "NoteStore.sqlite"
        an = AppleNotes(folder="X", db_path=db, name="my-notes")
        assert an.name == "my-notes"


# ---------------------------------------------------------------------------
# connector decorator
# ---------------------------------------------------------------------------


class TestConnectorDecorator:
    def test_creates_connector(self):
        @connector
        def my_source():
            yield Document(title="a", content="b")

        assert isinstance(my_source, Connector)
        assert my_source.name == "my_source"

    def test_scan_yields_documents(self):
        @connector
        def docs():
            yield Document(title="x", content="y")
            yield Document(title="z", content="w")

        results = list(docs.scan())
        assert len(results) == 2
        assert results[0].title == "x"

    def test_repr(self):
        @connector
        def my_fn():
            yield Document(title="a", content="b")

        assert "my_fn" in repr(my_fn)


# ---------------------------------------------------------------------------
# Content hashing
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        h1 = _content_hash("hello")
        h2 = _content_hash("hello")
        assert h1 == h2

    def test_different_content(self):
        h1 = _content_hash("hello")
        h2 = _content_hash("world")
        assert h1 != h2

    def test_returns_hex_string(self):
        h = _content_hash("test")
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# _make_node_id
# ---------------------------------------------------------------------------


class TestMakeNodeId:
    def test_deterministic(self):
        doc = Document(title="t", content="c", source="/path")
        id1 = _make_node_id("doc", "conn", doc)
        id2 = _make_node_id("doc", "conn", doc)
        assert id1 == id2

    def test_uses_prefix(self):
        doc = Document(title="t", content="c", source="/path")
        nid = _make_node_id("myprefix", "conn", doc)
        assert nid.startswith("myprefix-")

    def test_different_sources_different_ids(self):
        d1 = Document(title="t", content="c", source="/a")
        d2 = Document(title="t", content="c", source="/b")
        assert _make_node_id("doc", "conn", d1) != _make_node_id("doc", "conn", d2)

    def test_uses_title_when_no_source(self):
        d1 = Document(title="title1", content="c")
        d2 = Document(title="title2", content="c")
        assert _make_node_id("doc", "conn", d1) != _make_node_id("doc", "conn", d2)


# ---------------------------------------------------------------------------
# SyncResult
# ---------------------------------------------------------------------------


class TestSyncResult:
    def test_creation(self):
        r = SyncResult(added=5, updated=2, unchanged=10)
        assert r.added == 5
        assert r.updated == 2
        assert r.unchanged == 10

    def test_total(self):
        r = SyncResult(added=3, updated=1, unchanged=6)
        assert r.total == 10

    def test_defaults(self):
        r = SyncResult()
        assert r.added == 0
        assert r.total == 0

    def test_repr(self):
        r = SyncResult(added=1, updated=2, unchanged=3)
        assert "added=1" in repr(r)
        assert "updated=2" in repr(r)

    def test_frozen(self):
        r = SyncResult(added=1)
        with pytest.raises(AttributeError):
            r.added = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# sync()
# ---------------------------------------------------------------------------


class TestSync:
    def test_sync_adds_documents(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("Alpha content", encoding="utf-8")
        (tmp_path / "b.md").write_text("Beta content", encoding="utf-8")

        kg = KnowledgeGraph(":memory:")
        md = MarkdownDir(tmp_path)
        result = sync(kg, md)

        assert result.added == 2
        assert result.updated == 0
        assert result.unchanged == 0
        assert kg.node_count() == 2

    def test_sync_incremental(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("Original", encoding="utf-8")

        kg = KnowledgeGraph(":memory:")
        md = MarkdownDir(tmp_path)

        # First sync
        r1 = sync(kg, md)
        assert r1.added == 1

        # Second sync — no changes
        r2 = sync(kg, md)
        assert r2.added == 0
        assert r2.unchanged == 1

    def test_sync_detects_updates(self, tmp_path: Path):
        f = tmp_path / "a.md"
        f.write_text("Version 1", encoding="utf-8")

        kg = KnowledgeGraph(":memory:")
        md = MarkdownDir(tmp_path)

        r1 = sync(kg, md)
        assert r1.added == 1

        # Modify content
        f.write_text("Version 2", encoding="utf-8")
        r2 = sync(kg, md)
        assert r2.updated == 1
        assert r2.added == 0

        # Verify content updated
        nodes = kg.list_nodes()
        assert len(nodes) == 1
        assert nodes[0].content == "Version 2"

    def test_sync_multiple_connectors(self, tmp_path: Path):
        dir1 = tmp_path / "notes"
        dir2 = tmp_path / "docs"
        dir1.mkdir()
        dir2.mkdir()

        (dir1 / "note.md").write_text("A note", encoding="utf-8")
        (dir2 / "doc.txt").write_text("A doc", encoding="utf-8")

        kg = KnowledgeGraph(":memory:")
        md = MarkdownDir(dir1)
        dc = DirectoryConnector(dir2, extensions=[".txt"])

        result = sync(kg, md, dc)
        assert result.added == 2
        assert kg.node_count() == 2

    def test_sync_sets_metadata(self, tmp_path: Path):
        (tmp_path / "test.md").write_text("Content here", encoding="utf-8")

        kg = KnowledgeGraph(":memory:")
        md = MarkdownDir(tmp_path)
        sync(kg, md)

        nodes = kg.list_nodes()
        assert len(nodes) == 1
        meta = nodes[0].metadata
        assert "_content_hash" in meta
        assert meta["_connector"] == "markdown-dir"
        assert "_synced_at" in meta

    def test_sync_node_type_is_document(self, tmp_path: Path):
        (tmp_path / "x.md").write_text("hi", encoding="utf-8")

        kg = KnowledgeGraph(":memory:")
        sync(kg, MarkdownDir(tmp_path))

        nodes = kg.list_nodes()
        assert nodes[0].node_type == NodeType.DOCUMENT

    def test_sync_with_custom_connector(self):
        @connector
        def my_src():
            yield Document(title="custom", content="custom content", source="custom://1")

        kg = KnowledgeGraph(":memory:")
        result = sync(kg, my_src)
        assert result.added == 1
        assert kg.node_count() == 1
        assert kg.list_nodes()[0].name == "custom"

    def test_sync_custom_prefix(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("hi", encoding="utf-8")

        kg = KnowledgeGraph(":memory:")
        sync(kg, MarkdownDir(tmp_path), node_prefix="note")

        nodes = kg.list_nodes()
        assert nodes[0].id.startswith("note-")

    def test_sync_empty_connector(self):
        @connector
        def empty():
            return iter([])

        kg = KnowledgeGraph(":memory:")
        result = sync(kg, empty)
        assert result.total == 0
        assert kg.node_count() == 0


# ---------------------------------------------------------------------------
# Connector base class
# ---------------------------------------------------------------------------


class TestConnectorBase:
    def test_repr(self):
        @connector
        def source():
            yield Document(title="a", content="b")

        # The Connector base repr is covered by subclasses
        assert source.name == "source"

    def test_is_abstract(self):
        # Cannot instantiate directly
        with pytest.raises(TypeError):
            Connector(name="test")  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        from talk_box import (
            AppleNotes,
            Connector,
            DirectoryConnector,
            Document,
            MarkdownDir,
            SyncResult,
            connector,
            sync,
        )

        assert AppleNotes is not None
        assert Connector is not None
        assert DirectoryConnector is not None
        assert Document is not None
        assert MarkdownDir is not None
        assert SyncResult is not None
        assert connector is not None
        assert sync is not None

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "AppleNotes",
            "Connector",
            "DirectoryConnector",
            "Document",
            "MarkdownDir",
            "SyncResult",
            "connector",
            "sync",
        ]:
            assert name in talk_box.__all__, f"{name} missing from __all__"
