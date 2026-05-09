"""Tests for talk_box.workspace_tools module."""

from __future__ import annotations

import pytest

from talk_box.workspace_tools import ToolOutput, WorkspaceAgent


class TestWorkspaceAgent:
    """Tests for WorkspaceAgent file operations."""

    def test_file_read(self, tmp_path):
        (tmp_path / "hello.py").write_text("line1\nline2\nline3\n")
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_read("hello.py")
        assert result.success
        assert "line1" in result.output

    def test_file_read_with_range(self, tmp_path):
        (tmp_path / "data.txt").write_text("a\nb\nc\nd\ne\n")
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_read("data.txt", start_line=2, end_line=4)
        assert result.success
        assert "b" in result.output
        assert "d" in result.output
        assert "a" not in result.output

    def test_file_read_not_found(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_read("nope.txt")
        assert not result.success
        assert "not found" in result.output.lower()

    def test_file_read_path_traversal(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_read("../../etc/passwd")
        assert not result.success
        assert "traversal" in result.output.lower()

    def test_file_write(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_write("new.txt", "hello world")
        assert result.success
        assert (tmp_path / "new.txt").read_text() == "hello world"

    def test_file_write_creates_dirs(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_write("sub/dir/file.txt", "content")
        assert result.success
        assert (tmp_path / "sub" / "dir" / "file.txt").read_text() == "content"

    def test_file_write_size_limit(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path, max_file_size=100)
        result = agent.file_write("big.txt", "x" * 200)
        assert not result.success
        assert "max size" in result.output.lower()

    def test_file_write_path_traversal(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_write("../../evil.txt", "hack")
        assert not result.success

    def test_file_edit(self, tmp_path):
        (tmp_path / "code.py").write_text("def foo():\n    pass\n")
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_edit("code.py", "pass", "return 42")
        assert result.success
        assert "return 42" in (tmp_path / "code.py").read_text()

    def test_file_edit_not_found(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_edit("nope.py", "x", "y")
        assert not result.success

    def test_file_edit_old_text_missing(self, tmp_path):
        (tmp_path / "code.py").write_text("hello\n")
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_edit("code.py", "MISSING", "replacement")
        assert not result.success
        assert "not found" in result.output.lower()

    def test_file_search(self, tmp_path):
        (tmp_path / "a.py").write_text("def hello():\n    pass\n")
        (tmp_path / "b.py").write_text("def world():\n    pass\n")
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_search("hello")
        assert result.success
        assert "a.py" in result.output

    def test_file_search_no_match(self, tmp_path):
        (tmp_path / "a.py").write_text("nothing here\n")
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.file_search("ZZZZZ")
        assert result.success
        assert "no matches" in result.output.lower()

    def test_list_files(self, tmp_path):
        (tmp_path / "foo.py").write_text("")
        (tmp_path / "bar.txt").write_text("")
        (tmp_path / "sub").mkdir()
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.list_files()
        assert result.success
        assert "foo.py" in result.output
        assert "sub/" in result.output

    def test_list_files_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("")
        (tmp_path / "b.txt").write_text("")
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.list_files(".", pattern="*.py")
        assert result.success
        assert "a.py" in result.output
        assert "b.txt" not in result.output

    def test_list_files_not_dir(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.list_files("nonexistent")
        assert not result.success

    def test_shell_exec(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path, trusted_commands=["echo"])
        result = agent.shell_exec("echo hello")
        assert result.success
        assert "hello" in result.output

    def test_shell_exec_untrusted(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path, trusted_commands=["echo"])
        result = agent.shell_exec("rm -rf /")
        assert not result.success
        assert "not in trusted" in result.output.lower()

    def test_shell_exec_empty(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        result = agent.shell_exec("")
        assert not result.success

    def test_shell_exec_timeout(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path, trusted_commands=[])
        result = agent.shell_exec("sleep 10", timeout=1)
        assert not result.success
        assert "timed out" in result.output.lower()

    def test_change_log(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        agent.file_write("a.txt", "hi")
        agent.file_edit("a.txt", "hi", "bye")
        assert len(agent.changes) == 2
        assert agent.changes[0]["action"] == "write"
        assert agent.changes[1]["action"] == "edit"

    def test_reset_clears_log(self, tmp_path):
        agent = WorkspaceAgent(root=tmp_path)
        agent.file_write("a.txt", "hi")
        assert len(agent.changes) == 1
        agent.reset()
        assert len(agent.changes) == 0
