"""Talk Box TUI application shell.

Provides the main Textual application with header navigation, footer keybindings,
and command-mode screen routing. Designed to work reliably inside VS Code's
integrated terminal (no Ctrl+ shortcuts).
"""

from __future__ import annotations

import os
from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container
from textual.screen import Screen

from talk_box.tui.screens import (
    ChatScreen,
    CommandListScreen,
    EvalScreen,
    GuardrailScreen,
    HomeScreen,
    KnowledgeScreen,
    MemoryScreen,
    ModelScreen,
    PathwayScreen,
    PersonaScreen,
    ProfileScreen,
    SessionsScreen,
    SkillScreen,
    TraitScreen,
    WelcomeScreen,
    WorkspaceScreen,
)

# ---------------------------------------------------------------------------
# Screen registry — maps command-mode keys to screen IDs and labels
# ---------------------------------------------------------------------------

SCREEN_NAV: list[tuple[str, str, str, type[Screen]]] = [
    # (key, screen_id, label, screen_class)
    ("h", "home", "Home", HomeScreen),
    ("c", "chat", "Chat", ChatScreen),
    ("w", "workspace", "Workspace", WorkspaceScreen),
    ("s", "sessions", "Sessions", SessionsScreen),
    ("p", "personas", "Personas", PersonaScreen),
    ("m", "models", "Models", ModelScreen),
    ("r", "profiles", "Profiles", ProfileScreen),
    ("t", "traits", "Traits", TraitScreen),
    ("g", "guardrails", "Guardrails", GuardrailScreen),
    ("y", "pathways", "Pathways", PathwayScreen),
    ("i", "skills", "Skills", SkillScreen),
    ("k", "knowledge", "Knowledge", KnowledgeScreen),
    ("e", "eval", "Eval", EvalScreen),
    ("x", "memory", "Memory", MemoryScreen),
]

# Primary tabs shown in the header (the rest go in "More" overflow)
PRIMARY_TABS = {"home", "chat", "workspace", "personas", "models"}


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

# Detect terminal host for adaptive hints
_HOST = os.environ.get("TERM_PROGRAM", "")
_IN_VSCODE = _HOST == "vscode"


class TalkBoxApp(App):
    """Talk Box terminal application.

    A full TUI for building, configuring, and chatting with AI assistants.
    Uses command-mode navigation (Escape → key) to avoid Ctrl+ conflicts
    with terminal hosts like VS Code.
    """

    TITLE = "Talk Box"
    SUB_TITLE = ""
    CSS_PATH = "theme.tcss"

    COMMANDS: ClassVar[set] = set()
    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("colon", "open_command_list", "Commands (:)", show=True),
        Binding("escape", "open_command_list", "Commands", show=False),
        Binding("f1", "open_command_list", "Commands", show=False),
    ]

    # -- State ----------------------------------------------------------------

    command_mode: bool = False
    """Whether the command list is currently open."""

    _current_screen_id: str = "home"
    """Track the active screen ID for tab highlighting."""

    _shown_vscode_hint: bool = False
    """Track whether the VS Code first-run hint has been shown."""

    _simple_mode: bool = False
    """When True, only the chat screen is available (no tabs, no command list)."""

    # -- Compose --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Container(id="screen-container")

    # -- Lifecycle ------------------------------------------------------------

    def on_mount(self) -> None:
        """Initialize screens and show the home screen."""
        # Load .env file so API keys are available
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        # Detect simple mode from config
        try:
            from talk_box.config import TUIMode, load_config

            cfg = load_config()
            if cfg.mode == TUIMode.SIMPLE:
                self._simple_mode = True
        except Exception:
            pass

        if self._simple_mode:
            # Simple mode: only install the chat screen
            self.install_screen(ChatScreen, name="chat")
            self.push_screen("chat")
            self._current_screen_id = "chat"
            self.sub_title = "Chat"
            return

        # Install all screens
        for _key, screen_id, _label, screen_cls in SCREEN_NAV:
            self.install_screen(screen_cls, name=screen_id)

        # Also install the welcome screen (not in nav — it's a one-time flow)
        self.install_screen(WelcomeScreen, name="welcome")

        # Check if first run (no config exists)
        if not self._has_global_config():
            self.push_screen("welcome")
        else:
            self.push_screen("home")
            self._current_screen_id = "home"

        # VS Code first-run hint
        if _IN_VSCODE and not self._shown_vscode_hint:
            self._shown_vscode_hint = True
            self.notify(
                "Press [b]:[/b] (colon) to toggle command mode and see shortcuts.",
                title="Running inside VS Code",
                timeout=8,
            )

        self._update_sub_title()

    def _has_global_config(self) -> bool:
        """Check whether a global config file exists."""
        config_path = os.path.expanduser("~/.config/talk-box/config.yml")
        return os.path.exists(config_path)

    # -- Command list ---------------------------------------------------------

    def action_open_command_list(self) -> None:
        """Show the command list modal."""
        if self._simple_mode:
            return  # no navigation in simple mode
        if self.command_mode:
            return  # already open
        self.command_mode = True
        self.sub_title = "COMMAND MODE"
        self.push_screen(CommandListScreen(), callback=self._on_command_list_dismiss)

    def _on_command_list_dismiss(self, result: str | None) -> None:
        """Handle the result from the command list modal."""
        self.command_mode = False
        if result == "__quit__":
            self.exit()
        elif result is not None:
            self._switch_to(result)
        self._update_sub_title()

    def _update_sub_title(self) -> None:
        """Set sub_title to the current screen label."""
        for _key, sid, label, _cls in SCREEN_NAV:
            if sid == self._current_screen_id:
                self.sub_title = label
                break

    def on_key(self, event) -> None:
        """Suppress colon/escape/f1 in on_key since bindings handle them."""
        if event.key in ("colon", "escape", "f1"):
            return

    # -- Screen switching -----------------------------------------------------

    def _switch_to(self, screen_id: str) -> None:
        """Switch to a screen by ID."""
        if screen_id == self._current_screen_id:
            return
        self._current_screen_id = screen_id
        self.switch_screen(screen_id)
        self._update_sub_title()

    # -- Actions --------------------------------------------------------------

    def action_go_to(self, screen_id: str) -> None:
        """Navigate to a screen by ID."""
        self._switch_to(screen_id)
