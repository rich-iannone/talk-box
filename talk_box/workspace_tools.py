"""Workspace agent tools for file operations and shell execution.

These tools provide the agentic capabilities for the Workspace screen,
allowing the agent to read, write, edit, and search files, and execute
shell commands within the project directory.

All operations are sandboxed to a root directory (typically the cwd).
Path traversal outside the root is rejected.
"""

from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ToolOutput:
    """Result from a workspace tool execution."""

    success: bool
    output: str
    path: str | None = None


@dataclass
class WorkspaceAgent:
    """Agentic file and shell operations sandboxed to a project root.

    Parameters
    ----------
    root
        The project root directory. All paths are resolved relative to this.
    trusted_commands
        Shell commands allowed for ``shell_exec``. If empty, all commands are allowed.
    max_file_size
        Maximum file size in bytes for read/write operations.
    """

    root: Path
    trusted_commands: list[str] = field(
        default_factory=lambda: ["python", "uv", "pytest", "grep", "find", "cat", "ls", "wc"]
    )
    max_file_size: int = 500_000
    _change_log: list[dict] = field(default_factory=list, repr=False)

    def _resolve(self, path: str) -> Path | None:
        """Resolve a path relative to root, rejecting traversals."""
        try:
            resolved = (self.root / path).resolve()
        except (ValueError, OSError):
            return None
        # Ensure the resolved path is inside root
        try:
            resolved.relative_to(self.root.resolve())
        except ValueError:
            return None
        return resolved

    # ------------------------------------------------------------------
    # file_read
    # ------------------------------------------------------------------

    def file_read(
        self, path: str, *, start_line: int = 1, end_line: int | None = None
    ) -> ToolOutput:
        """Read a file's contents.

        Parameters
        ----------
        path
            Path relative to the project root.
        start_line
            1-based start line (inclusive).
        end_line
            1-based end line (inclusive). ``None`` reads to EOF.
        """
        resolved = self._resolve(path)
        if resolved is None:
            return ToolOutput(success=False, output=f"Path traversal rejected: {path}")
        if not resolved.is_file():
            return ToolOutput(success=False, output=f"File not found: {path}", path=path)

        try:
            text = resolved.read_text(errors="replace")
        except OSError as exc:
            return ToolOutput(success=False, output=f"Cannot read: {exc}", path=path)

        lines = text.splitlines(keepends=True)
        start_idx = max(0, start_line - 1)
        end_idx = end_line if end_line is not None else len(lines)
        selected = lines[start_idx:end_idx]
        content = "".join(selected)

        return ToolOutput(
            success=True,
            output=content,
            path=path,
        )

    # ------------------------------------------------------------------
    # file_write
    # ------------------------------------------------------------------

    def file_write(self, path: str, content: str, *, create_dirs: bool = True) -> ToolOutput:
        """Write content to a file (creates or overwrites).

        Parameters
        ----------
        path
            Path relative to the project root.
        content
            Full file content.
        create_dirs
            Create parent directories if they don't exist.
        """
        resolved = self._resolve(path)
        if resolved is None:
            return ToolOutput(success=False, output=f"Path traversal rejected: {path}")

        if len(content.encode()) > self.max_file_size:
            return ToolOutput(
                success=False,
                output=f"Content exceeds max size ({self.max_file_size} bytes)",
                path=path,
            )

        try:
            if create_dirs:
                resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content)
        except OSError as exc:
            return ToolOutput(success=False, output=f"Cannot write: {exc}", path=path)

        self._change_log.append({"action": "write", "path": path, "size": len(content)})
        return ToolOutput(success=True, output=f"Wrote {len(content)} chars to {path}", path=path)

    # ------------------------------------------------------------------
    # file_edit
    # ------------------------------------------------------------------

    def file_edit(self, path: str, old_text: str, new_text: str) -> ToolOutput:
        """Replace the first occurrence of *old_text* with *new_text* in a file.

        Parameters
        ----------
        path
            Path relative to the project root.
        old_text
            Exact text to find (must appear at least once).
        new_text
            Replacement text.
        """
        resolved = self._resolve(path)
        if resolved is None:
            return ToolOutput(success=False, output=f"Path traversal rejected: {path}")
        if not resolved.is_file():
            return ToolOutput(success=False, output=f"File not found: {path}", path=path)

        try:
            content = resolved.read_text(errors="replace")
        except OSError as exc:
            return ToolOutput(success=False, output=f"Cannot read: {exc}", path=path)

        if old_text not in content:
            return ToolOutput(
                success=False,
                output="Old text not found in file",
                path=path,
            )

        updated = content.replace(old_text, new_text, 1)

        try:
            resolved.write_text(updated)
        except OSError as exc:
            return ToolOutput(success=False, output=f"Cannot write: {exc}", path=path)

        self._change_log.append({"action": "edit", "path": path})
        return ToolOutput(success=True, output=f"Replaced text in {path}", path=path)

    # ------------------------------------------------------------------
    # file_search
    # ------------------------------------------------------------------

    def file_search(
        self,
        pattern: str,
        *,
        glob: str = "**/*",
        max_results: int = 50,
    ) -> ToolOutput:
        """Search for a text pattern across project files.

        Parameters
        ----------
        pattern
            Text or substring to search for (case-insensitive).
        glob
            Glob pattern for which files to search. Default searches all files.
        max_results
            Maximum number of matching lines to return.
        """
        results: list[str] = []
        root = self.root.resolve()

        for filepath in sorted(root.glob(glob)):
            if not filepath.is_file():
                continue
            # Skip hidden dirs and common large dirs
            rel = str(filepath.relative_to(root))
            skip_dirs = {".git", ".venv", "node_modules", "__pycache__", ".tox", "htmlcov"}
            if any(part in skip_dirs for part in Path(rel).parts):
                continue

            try:
                text = filepath.read_text(errors="replace")
            except OSError:
                continue

            if "\x00" in text[:512]:
                continue  # skip binary

            for i, line in enumerate(text.splitlines(), 1):
                if pattern.lower() in line.lower():
                    results.append(f"{rel}:{i}: {line.rstrip()}")
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break

        if not results:
            return ToolOutput(success=True, output=f"No matches for '{pattern}'")

        return ToolOutput(
            success=True,
            output=f"{len(results)} match(es):\n" + "\n".join(results),
        )

    # ------------------------------------------------------------------
    # list_files
    # ------------------------------------------------------------------

    def list_files(self, path: str = ".", *, pattern: str = "*") -> ToolOutput:
        """List files in a directory.

        Parameters
        ----------
        path
            Directory path relative to root.
        pattern
            Filename glob pattern (e.g. ``*.py``).
        """
        resolved = self._resolve(path)
        if resolved is None:
            return ToolOutput(success=False, output=f"Path traversal rejected: {path}")
        if not resolved.is_dir():
            return ToolOutput(success=False, output=f"Not a directory: {path}")

        entries: list[str] = []
        for child in sorted(resolved.iterdir()):
            if not fnmatch.fnmatch(child.name, pattern):
                continue
            suffix = "/" if child.is_dir() else ""
            rel = str(child.relative_to(self.root.resolve()))
            entries.append(f"{rel}{suffix}")

        return ToolOutput(
            success=True,
            output="\n".join(entries) if entries else "(empty directory)",
        )

    # ------------------------------------------------------------------
    # shell_exec
    # ------------------------------------------------------------------

    def shell_exec(self, command: str, *, timeout: int = 30) -> ToolOutput:
        """Execute a shell command in the project root.

        Parameters
        ----------
        command
            The command string to execute.
        timeout
            Maximum seconds to wait for the command.

        Only the first word of the command is checked against
        ``trusted_commands``. If the trusted list is non-empty
        and the command's program is not in it, execution is rejected.
        """
        parts = command.split()
        if not parts:
            return ToolOutput(success=False, output="Empty command")

        program = parts[0]
        if self.trusted_commands and program not in self.trusted_commands:
            return ToolOutput(
                success=False,
                output=f"Command '{program}' not in trusted list: {self.trusted_commands}",
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return ToolOutput(success=False, output=f"Command timed out after {timeout}s")
        except OSError as exc:
            return ToolOutput(success=False, output=f"Execution error: {exc}")

        output_parts = []
        if result.stdout.strip():
            output_parts.append(result.stdout.strip())
        if result.stderr.strip():
            output_parts.append(f"STDERR: {result.stderr.strip()}")

        combined = "\n".join(output_parts) if output_parts else "(no output)"

        # Truncate very long output
        if len(combined) > self.max_file_size:
            combined = combined[: self.max_file_size] + "\n… (truncated)"

        self._change_log.append(
            {"action": "shell", "command": command, "exit_code": result.returncode}
        )

        return ToolOutput(
            success=result.returncode == 0,
            output=combined,
        )

    # ------------------------------------------------------------------
    # change_log
    # ------------------------------------------------------------------

    @property
    def changes(self) -> list[dict]:
        """Return the list of changes made during this session."""
        return list(self._change_log)

    def reset(self) -> None:
        """Clear the change log."""
        self._change_log.clear()
