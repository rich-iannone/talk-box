"""Tests for the undo buffer (talk_box.undo)."""

from __future__ import annotations

import os
import tempfile

import pytest

from talk_box.undo import UndoBuffer, UndoEntry


class TestUndoEntry:
    """UndoEntry dataclass basics."""

    def test_frozen(self):
        entry = UndoEntry(
            action="write",
            path="foo.txt",
            previous_content="old",
            resolved_path="/tmp/foo.txt",
        )
        with pytest.raises(AttributeError):
            entry.action = "edit"  # type: ignore[misc]

    def test_defaults(self):
        entry = UndoEntry(
            action="edit",
            path="bar.py",
            previous_content="x = 1",
            resolved_path="/tmp/bar.py",
        )
        assert entry.description == ""
        assert entry.timestamp is not None


class TestUndoBufferRecord:
    """Recording entries into the buffer."""

    def test_record_single(self):
        buf = UndoBuffer()
        buf.record(
            action="write",
            path="a.txt",
            previous_content=None,
            resolved_path="/tmp/a.txt",
        )
        assert len(buf) == 1
        assert buf.entries[0].action == "write"
        assert buf.entries[0].previous_content is None

    def test_record_generates_description(self):
        buf = UndoBuffer()
        buf.record(
            action="edit",
            path="main.py",
            previous_content="old",
            resolved_path="/tmp/main.py",
        )
        assert buf.entries[0].description == "Edit main.py"

    def test_record_custom_description(self):
        buf = UndoBuffer()
        buf.record(
            action="write",
            path="out.txt",
            previous_content=None,
            resolved_path="/tmp/out.txt",
            description="Created output",
        )
        assert buf.entries[0].description == "Created output"

    def test_max_entries_eviction(self):
        buf = UndoBuffer(max_entries=5)
        for i in range(10):
            buf.record(
                action="write",
                path=f"f{i}.txt",
                previous_content=None,
                resolved_path=f"/tmp/f{i}.txt",
            )
        assert len(buf) == 5
        # Oldest entries (0-4) should be evicted
        assert buf.entries[0].path == "f5.txt"
        assert buf.entries[-1].path == "f9.txt"


class TestUndoBufferUndoLast:
    """Undo last change."""

    def test_undo_last_restores_content(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("original")

        buf = UndoBuffer()
        buf.record(
            action="write",
            path="test.txt",
            previous_content="original",
            resolved_path=str(filepath),
        )
        # Simulate the write
        filepath.write_text("modified")
        assert filepath.read_text() == "modified"

        entry = buf.undo_last()
        assert entry is not None
        assert entry.path == "test.txt"
        assert filepath.read_text() == "original"
        assert len(buf) == 0

    def test_undo_last_deletes_new_file(self, tmp_path):
        filepath = tmp_path / "new.txt"
        filepath.write_text("created")

        buf = UndoBuffer()
        buf.record(
            action="write",
            path="new.txt",
            previous_content=None,
            resolved_path=str(filepath),
        )

        entry = buf.undo_last()
        assert entry is not None
        assert not filepath.exists()

    def test_undo_last_empty_buffer(self):
        buf = UndoBuffer()
        assert buf.undo_last() is None

    def test_undo_last_delete_missing_file_no_error(self, tmp_path):
        """Undoing a new-file-write when the file is already gone should not raise."""
        filepath = tmp_path / "gone.txt"

        buf = UndoBuffer()
        buf.record(
            action="write",
            path="gone.txt",
            previous_content=None,
            resolved_path=str(filepath),
        )
        # File was never actually created or already removed
        entry = buf.undo_last()
        assert entry is not None
        assert not filepath.exists()


class TestUndoBufferUndoAll:
    """Undo all changes in reverse order."""

    def test_undo_all_reverses_stack(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a-original")
        f2.write_text("b-original")

        buf = UndoBuffer()
        buf.record(
            action="edit",
            path="a.txt",
            previous_content="a-original",
            resolved_path=str(f1),
        )
        f1.write_text("a-modified")

        buf.record(
            action="edit",
            path="b.txt",
            previous_content="b-original",
            resolved_path=str(f2),
        )
        f2.write_text("b-modified")

        reverted = buf.undo_all()
        assert len(reverted) == 2
        # Most recent first
        assert reverted[0].path == "b.txt"
        assert reverted[1].path == "a.txt"
        assert f1.read_text() == "a-original"
        assert f2.read_text() == "b-original"
        assert len(buf) == 0

    def test_undo_all_empty(self):
        buf = UndoBuffer()
        assert buf.undo_all() == []


class TestUndoBufferClear:
    """Clear without reverting."""

    def test_clear(self, tmp_path):
        filepath = tmp_path / "keep.txt"
        filepath.write_text("modified")

        buf = UndoBuffer()
        buf.record(
            action="write",
            path="keep.txt",
            previous_content="original",
            resolved_path=str(filepath),
        )
        buf.clear()
        assert len(buf) == 0
        # File should NOT be restored
        assert filepath.read_text() == "modified"


class TestUndoBufferEntries:
    """entries property returns a copy."""

    def test_entries_is_copy(self):
        buf = UndoBuffer()
        buf.record(
            action="write",
            path="x.txt",
            previous_content=None,
            resolved_path="/tmp/x.txt",
        )
        entries = buf.entries
        entries.clear()
        assert len(buf) == 1


class TestSnapshotForUndo:
    """Integration: _snapshot_for_undo in builtin_tools captures content."""

    def test_snapshot_captures_existing_file(self, tmp_path, monkeypatch):
        from talk_box import builtin_tools

        filepath = tmp_path / "existing.py"
        filepath.write_text("x = 1\n")

        monkeypatch.setattr(builtin_tools, "_get_workspace_root", lambda: tmp_path)

        buf = UndoBuffer()
        builtin_tools.set_undo_buffer(buf)
        try:
            builtin_tools._snapshot_for_undo("edit", "existing.py")
            assert len(buf) == 1
            assert buf.entries[0].previous_content == "x = 1\n"
            assert buf.entries[0].action == "edit"
        finally:
            builtin_tools.set_undo_buffer(None)

    def test_snapshot_records_none_for_new_file(self, tmp_path, monkeypatch):
        from talk_box import builtin_tools

        monkeypatch.setattr(builtin_tools, "_get_workspace_root", lambda: tmp_path)

        buf = UndoBuffer()
        builtin_tools.set_undo_buffer(buf)
        try:
            builtin_tools._snapshot_for_undo("write", "new_file.txt")
            assert len(buf) == 1
            assert buf.entries[0].previous_content is None
        finally:
            builtin_tools.set_undo_buffer(None)

    def test_no_snapshot_when_buffer_not_set(self, tmp_path, monkeypatch):
        from talk_box import builtin_tools

        monkeypatch.setattr(builtin_tools, "_get_workspace_root", lambda: tmp_path)
        builtin_tools.set_undo_buffer(None)
        # Should not raise
        builtin_tools._snapshot_for_undo("write", "anything.txt")


class TestUndoSlashCommand:
    """Tests for /undo command handling in ChatScreen."""

    def _make_screen(self):
        from unittest.mock import MagicMock

        from talk_box.tui.screens import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        screen._append_system_message = MagicMock()
        return screen

    def test_handle_undo_empty(self):
        screen = self._make_screen()
        screen._undo_buffer = None
        screen._handle_undo("")
        screen._append_system_message.assert_called_once()
        assert "Nothing to undo" in screen._append_system_message.call_args[0][0]

    def test_handle_undo_list_empty(self):
        screen = self._make_screen()
        screen._undo_buffer = UndoBuffer()
        screen._handle_undo("list")
        assert "No file changes" in screen._append_system_message.call_args[0][0]

    def test_handle_undo_list_with_entries(self, tmp_path):
        screen = self._make_screen()
        buf = UndoBuffer()
        buf.record(
            action="write",
            path="a.txt",
            previous_content=None,
            resolved_path=str(tmp_path / "a.txt"),
        )
        buf.record(
            action="edit",
            path="b.py",
            previous_content="old",
            resolved_path=str(tmp_path / "b.py"),
        )
        screen._undo_buffer = buf
        screen._handle_undo("list")
        msg = screen._append_system_message.call_args[0][0]
        assert "Undo Buffer" in msg
        assert "a.txt" in msg
        assert "b.py" in msg
        assert "2 changes" in msg

    def test_handle_undo_last(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("modified")

        screen = self._make_screen()
        buf = UndoBuffer()
        buf.record(
            action="edit",
            path="test.txt",
            previous_content="original",
            resolved_path=str(filepath),
        )
        screen._undo_buffer = buf
        screen._handle_undo("")
        msg = screen._append_system_message.call_args[0][0]
        assert "test.txt" in msg
        assert "restored" in msg
        assert filepath.read_text() == "original"

    def test_handle_undo_all(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("a-new")
        f2.write_text("b-new")

        screen = self._make_screen()
        buf = UndoBuffer()
        buf.record(action="write", path="a.txt", previous_content="a-old", resolved_path=str(f1))
        buf.record(action="write", path="b.txt", previous_content="b-old", resolved_path=str(f2))
        screen._undo_buffer = buf
        screen._handle_undo("all")
        msg = screen._append_system_message.call_args[0][0]
        assert "Reverted 2" in msg
        assert f1.read_text() == "a-old"
        assert f2.read_text() == "b-old"

    def test_handle_undo_bad_arg(self):
        screen = self._make_screen()
        buf = UndoBuffer()
        buf.record(action="write", path="x.txt", previous_content=None, resolved_path="/tmp/x.txt")
        screen._undo_buffer = buf
        screen._handle_undo("bogus")
        msg = screen._append_system_message.call_args[0][0]
        assert "Usage" in msg

    def test_clear_resets_undo_buffer(self):
        from unittest.mock import MagicMock

        from talk_box.tui.screens import ChatScreen

        screen = ChatScreen.__new__(ChatScreen)
        buf = UndoBuffer()
        buf.record(action="write", path="x.txt", previous_content=None, resolved_path="/tmp/x.txt")
        screen._undo_buffer = buf
        screen._conversation = None
        screen._message_count = 0
        screen._prompt_history = []
        screen._history_index = -1
        screen._capture = None
        screen._bot = None
        screen._active_model = None
        screen._active_persona = None
        screen._active_guards = []
        screen._active_traits = []
        screen._enter_sends = True
        screen._output_format = None
        screen._require_approvals = True
        screen._rebuild_bot = MagicMock()
        screen.run_worker = MagicMock()

        screen._do_clear()
        assert len(buf) == 0
