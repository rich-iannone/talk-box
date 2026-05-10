"""TUI screen definitions.

Each screen is a Textual Screen subclass. Screens start as minimal placeholders
and are fleshed out in subsequent phases. The app shell imports them all from
this module so that screen installation and command-mode routing work.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.events import Click
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    DataTable,
    DirectoryTree,
    Footer,
    Header,
    Input,
    OptionList,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

_GLOBAL_CONFIG_DIR = "__unset__"  # lazily resolved


def _config_dir() -> str:
    """Return the global config directory path."""
    import os

    return os.path.expanduser("~/.config/talk-box")


class ChatInput(TextArea):
    """Multi-line input with configurable Enter behaviour.

    When ``enter_sends`` is True (default), Enter sends and Shift+Enter
    inserts a newline.  When False, Enter inserts a newline and the user
    must click the Send button.
    """

    enter_sends: bool = True

    class Submitted(TextArea.Changed):
        """Posted when the user submits the input."""

        def __init__(self, text_area: TextArea) -> None:
            super().__init__(text_area)
            self.value = text_area.text

    def _on_key(self, event) -> None:
        if event.key == "enter" and self.enter_sends:
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self))

    def submit(self) -> None:
        """Programmatically submit the current text (used by Send button)."""
        self.post_message(self.Submitted(self))


# ---------------------------------------------------------------------------
# Command List (modal overlay)
# ---------------------------------------------------------------------------


class CommandListScreen(ModalScreen):
    """Modal overlay showing all navigation shortcuts.

    Opened with `:`. Press a nav key to jump, or `:` / Escape to dismiss.
    """

    BINDINGS = [
        Binding("escape", "dismiss_list", "Close", show=False, priority=True),
        Binding("colon", "dismiss_list", "Close", show=False, priority=True),
        Binding("f1", "dismiss_list", "Close", show=False, priority=True),
    ]

    def compose(self) -> ComposeResult:
        # Import here to avoid circular ref
        from talk_box.tui.app import SCREEN_NAV

        lines = ["[b]Go to…[/b]\n"]
        for key, _sid, label, _cls in SCREEN_NAV:
            lines.append(f"  [b]{key}[/b]  {label}")
        lines.append("")
        lines.append("  [b]n[/b]  New Chat")
        lines.append("  [b]q[/b]  Quit")
        lines.append("\n[dim]Press a key or : to close[/dim]")

        with Center():
            with Vertical(id="command-list-panel"):
                yield Static("\n".join(lines), id="command-list")

    def on_key(self, event) -> None:
        """Dispatch nav keys then dismiss."""
        from talk_box.tui.app import SCREEN_NAV

        key = event.key
        if key in ("colon", "escape"):
            return  # handled by binding

        for nav_key, screen_id, _label, _cls in SCREEN_NAV:
            if key == nav_key:
                event.prevent_default()
                self.dismiss(screen_id)
                return

        if key == "n":
            event.prevent_default()
            self.dismiss("chat")
            return

        if key == "q":
            event.prevent_default()
            self.dismiss("__quit__")
            return

    def action_dismiss_list(self) -> None:
        """Close the command list without navigating."""
        self.dismiss(None)


# ---------------------------------------------------------------------------
# File Approval Modal
# ---------------------------------------------------------------------------


class FileApprovalScreen(ModalScreen[dict]):
    """Modal that lets the user browse and approve/reject a batch of file changes.

    Accepts a list of pending file operations.  The user can navigate with
    ← / → (or ``h`` / ``l``), approve or reject each file individually, or
    approve/reject all remaining files at once.  Dismisses with a dict
    mapping ``(action, path)`` → ``bool`` for every file.
    """

    BINDINGS = [
        Binding("y", "approve_current", "Approve", show=True, priority=True),
        Binding("n", "reject_current", "Reject", show=True, priority=True),
        Binding("a", "approve_rest", "Approve Rest", show=True, priority=True),
        Binding("x", "reject_rest", "Reject Rest", show=True, priority=True),
        Binding("left", "prev_file", "← Prev", show=True, priority=True),
        Binding("right", "next_file", "→ Next", show=True, priority=True),
        Binding("h", "prev_file", "Prev", show=False, priority=True),
        Binding("l", "next_file", "Next", show=False, priority=True),
        Binding("escape", "reject_current", "Reject", show=False, priority=True),
    ]

    def __init__(
        self,
        pending: list[tuple[str, str, dict]],
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        # Each element is (action, path, details)
        self._pending = list(pending)
        self._index = 0
        # None = not yet decided, True = approved, False = rejected
        self._decisions: dict[int, bool | None] = {i: None for i in range(len(pending))}

    @property
    def _current(self) -> tuple[str, str, dict]:
        return self._pending[self._index]

    @property
    def _all_decided(self) -> bool:
        return all(v is not None for v in self._decisions.values())

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="file-approval-panel"):
                yield Static("", id="file-approval-title")
                yield Static("", id="file-approval-preview")
                yield Static("", id="file-approval-nav")
                with Horizontal(id="file-approval-buttons"):
                    yield Button("Approve (y)", variant="success", id="file-approve-btn")
                    yield Button("Approve Rest (a)", variant="success", id="file-approve-all-btn")
                    yield Button("Reject (n)", variant="error", id="file-reject-btn")
                    yield Button("Reject Rest (x)", variant="error", id="file-reject-all-btn")

    def on_mount(self) -> None:
        self._refresh_display()

    def _refresh_display(self) -> None:
        action, path, details = self._current
        title = "Write file" if action == "write" else "Edit file"
        decision = self._decisions[self._index]
        status = ""
        if decision is True:
            status = " [green]✓ approved[/green]"
        elif decision is False:
            status = " [red]✗ rejected[/red]"

        self.query_one("#file-approval-title", Static).update(
            f"[b]{title}[/b]: [cyan]{path}[/cyan]{status}"
        )

        preview_lines: list[str] = []
        if action == "write":
            content = details.get("content", "")
            lines = content.splitlines()
            if len(lines) > 40:
                preview_lines = lines[:40]
                preview_lines.append(f"\n... ({len(lines) - 40} more lines)")
            else:
                preview_lines = lines
        else:
            old = details.get("old_text", "")
            new = details.get("new_text", "")
            preview_lines.append("--- old")
            preview_lines.extend(f"- {l}" for l in old.splitlines())
            preview_lines.append("+++ new")
            preview_lines.extend(f"+ {l}" for l in new.splitlines())

        self.query_one("#file-approval-preview", Static).update("\n".join(preview_lines))

        # Navigation indicator
        n = len(self._pending)
        decided = sum(1 for v in self._decisions.values() if v is not None)
        undecided = n - decided
        nav = f"  File {self._index + 1} of {n}  |  {decided} decided, {undecided} remaining  "
        if n > 1:
            nav += " |  ← → to browse"
        self.query_one("#file-approval-nav", Static).update(f"[dim]{nav}[/dim]")

    def _advance_or_finish(self) -> None:
        """Move to the next undecided file, or finish if all decided."""
        if self._all_decided:
            self._finish()
            return
        # Find next undecided
        for offset in range(1, len(self._pending) + 1):
            idx = (self._index + offset) % len(self._pending)
            if self._decisions[idx] is None:
                self._index = idx
                self._refresh_display()
                return

    def _finish(self) -> None:
        """Dismiss with the full decisions dict."""
        result = {}
        for i, (action, path, _details) in enumerate(self._pending):
            result[(action, path)] = bool(self._decisions[i])
        self.dismiss(result)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "file-approve-btn":
            self.action_approve_current()
        elif event.button.id == "file-approve-all-btn":
            self.action_approve_rest()
        elif event.button.id == "file-reject-btn":
            self.action_reject_current()
        elif event.button.id == "file-reject-all-btn":
            self.action_reject_rest()

    def action_approve_current(self) -> None:
        self._decisions[self._index] = True
        self._advance_or_finish()

    def action_reject_current(self) -> None:
        self._decisions[self._index] = False
        self._advance_or_finish()

    def action_approve_rest(self) -> None:
        for i, v in self._decisions.items():
            if v is None:
                self._decisions[i] = True
        self._finish()

    def action_reject_rest(self) -> None:
        for i, v in self._decisions.items():
            if v is None:
                self._decisions[i] = False
        self._finish()

    def action_prev_file(self) -> None:
        if len(self._pending) > 1:
            self._index = (self._index - 1) % len(self._pending)
            self._refresh_display()

    def action_next_file(self) -> None:
        if len(self._pending) > 1:
            self._index = (self._index + 1) % len(self._pending)
            self._refresh_display()


# ---------------------------------------------------------------------------
# Checklist Picker Modal (reusable)
# ---------------------------------------------------------------------------


class ChecklistPickerModal(ModalScreen[list[str]]):
    """Modal with a checklist of toggleable items.

    Shows a list of items with checkboxes.  The user toggles items and
    presses Enter/Done to confirm, or Escape to cancel.

    Parameters
    ----------
    title
        Modal heading.
    items
        All available items: list of ``(name, description)`` tuples.
    active
        Currently active item names.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True, priority=True),
        Binding("enter", "confirm", "Done", show=True, priority=True),
        Binding("space", "toggle_current", "Toggle", show=True, priority=True),
        Binding("up", "cursor_up", "↑", show=False, priority=True),
        Binding("down", "cursor_down", "↓", show=False, priority=True),
        Binding("k", "cursor_up", "Up", show=False, priority=True),
        Binding("j", "cursor_down", "Down", show=False, priority=True),
    ]

    def __init__(
        self,
        title: str,
        items: list[tuple[str, str]],
        active: list[str],
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._title = title
        self._items = list(items)  # [(name, description), ...]
        self._active = set(active)
        self._cursor = 0

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="checklist-panel"):
                yield Static("", id="checklist-title")
                yield VerticalScroll(id="checklist-body")
                with Horizontal(id="checklist-buttons"):
                    yield Button("Done", id="checklist-done-btn", variant="success")
                    yield Button("Cancel", id="checklist-cancel-btn", variant="default")

    def on_mount(self) -> None:
        self._refresh_display()

    def _item_label(self, index: int) -> str:
        """Build the display label for an item at *index*."""
        item_name, desc = self._items[index]
        marker = "✓" if item_name in self._active else " "
        cursor = "▸" if index == self._cursor else " "
        desc_text = f"  [dim]{desc}[/dim]" if desc else ""
        return f" {cursor} \\[{marker}] {item_name}{desc_text}"

    def _refresh_display(self) -> None:
        title_w = self.query_one("#checklist-title", Static)
        title_w.update(f"[b]{self._title}[/b]  ({len(self._active)} active)")

        body = self.query_one("#checklist-body", VerticalScroll)
        # Update labels in-place if widgets already exist
        existing = body.query("Static")
        if len(existing) == len(self._items):
            for i in range(len(self._items)):
                existing[i].update(self._item_label(i))
        else:
            body.remove_children()
            for i in range(len(self._items)):
                body.mount(
                    Static(
                        self._item_label(i),
                        id=f"chk-item-{i}",
                        classes="checklist-item",
                    )
                )

    def on_click(self, event: Click) -> None:
        """Handle mouse clicks on checklist items."""
        # Find if the click target is one of our item widgets
        widget = event.widget
        widget_id = getattr(widget, "id", "") or ""
        if widget_id.startswith("chk-item-"):
            try:
                index = int(widget_id.removeprefix("chk-item-"))
            except ValueError:
                return
            if 0 <= index < len(self._items):
                self._cursor = index
                item_name = self._items[index][0]
                if item_name in self._active:
                    self._active.discard(item_name)
                else:
                    self._active.add(item_name)
                self._refresh_display()

    def action_toggle_current(self) -> None:
        if not self._items:
            return
        item_name = self._items[self._cursor][0]
        if item_name in self._active:
            self._active.discard(item_name)
        else:
            self._active.add(item_name)
        self._refresh_display()

    def action_cursor_up(self) -> None:
        if self._items:
            self._cursor = (self._cursor - 1) % len(self._items)
            self._refresh_display()

    def action_cursor_down(self) -> None:
        if self._items:
            self._cursor = (self._cursor + 1) % len(self._items)
            self._refresh_display()

    def action_confirm(self) -> None:
        self.dismiss(sorted(self._active))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id or ""
        if btn_id == "checklist-done-btn":
            self.action_confirm()
        elif btn_id == "checklist-cancel-btn":
            self.action_cancel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _placeholder(title: str, description: str) -> ComposeResult:
    """Yield a centered placeholder for screens not yet implemented."""
    yield Header(show_clock=False)
    with Center():
        with VerticalScroll():
            yield Static(f"[b]{title}[/b]\n\n{description}", id="placeholder")
    yield Footer()


# ---------------------------------------------------------------------------
# Screen 1: Welcome / Setup
# ---------------------------------------------------------------------------


class WelcomeScreen(Screen):
    """First-run setup wizard.

    Detects Ollama, API keys, and available models. Guides the user through
    choosing a default model and optional persona, then writes the global
    config file.
    """

    BINDINGS = [
        Binding("enter", "start", "Start Chatting", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Center():
            with Vertical(id="welcome-panel"):
                yield Static(
                    "[b]Welcome to Talk Box[/b]",
                    id="welcome-title",
                )
                yield Static(
                    "Build, configure, and chat with AI assistants — right from your terminal.",
                    id="welcome-subtitle",
                )
                yield Static(
                    self._detect_environment(),
                    id="welcome-env",
                )
                yield Button(
                    "Start Chatting",
                    id="welcome-start",
                    variant="primary",
                )
                yield Button(
                    "Skip Setup",
                    id="welcome-skip",
                    variant="default",
                )
        yield Footer()

    def _detect_environment(self) -> str:
        """Build an environment detection summary."""
        import os
        import sys

        lines = ["[b]Environment[/b]"]
        lines.append(f"  Python {sys.version.split()[0]}")

        # Ollama
        try:
            from talk_box.models import detect_ollama

            status = detect_ollama()
            if status.available:
                n_models = len(status.models) if status.models else 0
                lines.append(f"  Ollama running ({n_models} models)")
            else:
                lines.append("  Ollama not detected")
        except Exception:
            lines.append("  Ollama not detected")

        # API keys
        key_vars = [
            ("ANTHROPIC_API_KEY", "Anthropic"),
            ("OPENAI_API_KEY", "OpenAI"),
            ("GOOGLE_API_KEY", "Google"),
            ("GITHUB_TOKEN", "GitHub"),
        ]
        found = [name for var, name in key_vars if os.environ.get(var)]
        if found:
            lines.append(f"  API keys: {', '.join(found)}")
        else:
            lines.append("  No API keys found")

        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id in ("welcome-start", "welcome-skip"):
            self.action_start()

    def action_start(self) -> None:
        """Start chatting — dismiss welcome and go to Home."""
        self.app.switch_screen("home")


# ---------------------------------------------------------------------------
# Screen 2: Home (Dashboard)
# ---------------------------------------------------------------------------


class HomeScreen(Screen):
    """Landing screen with profile summary, quick actions, and system status."""

    BINDINGS = [
        Binding("n", "new_chat", "New Chat", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with VerticalScroll(id="home-scroll"):
            # Active profile panel
            with Vertical(id="home-profile-panel", classes="home-panel"):
                yield Static("[b]Active Profile[/b]", id="home-profile-title")
                yield Static(
                    self._build_profile_summary(),
                    id="home-profile-summary",
                )

            # Quick actions + system status side by side
            with Horizontal(id="home-columns"):
                with Vertical(id="home-actions-panel", classes="home-panel"):
                    yield Static("[b]Quick Actions[/b]", id="home-actions-title")
                    yield Button("New Chat", id="home-new-chat", variant="primary")
                    yield Button("Switch Profile", id="home-switch-profile")
                    yield Button("Browse Models", id="home-browse-models")
                    yield Button("Browse Personas", id="home-browse-personas")

                with Vertical(id="home-status-panel", classes="home-panel"):
                    yield Static("[b]System Status[/b]", id="home-status-title")
                    yield Static(
                        self._build_system_status(),
                        id="home-status-summary",
                    )
        yield Footer()

    def _build_profile_summary(self) -> str:
        """Build the active profile summary text."""
        try:
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            model = resolved.model or "[dim]no model set[/dim]"
            persona = resolved.persona or "[dim]none[/dim]"
            guards = ", ".join(resolved.guardrails) if resolved.guardrails else "[dim]none[/dim]"
            temp = (
                f"{resolved.temperature}"
                if resolved.temperature is not None
                else "[dim]default[/dim]"
            )
            return (
                f"  Model:       {model}\n"
                f"  Persona:     {persona}\n"
                f"  Guardrails:  {guards}\n"
                f"  Temperature: {temp}"
            )
        except Exception:
            return (
                "  Model:       [dim]no model set[/dim]\n"
                "  Persona:     [dim]none[/dim]\n"
                "  Guardrails:  [dim]none[/dim]\n"
                "  Temperature: [dim]default[/dim]"
            )

    def _build_system_status(self) -> str:
        """Build the system status summary text."""
        import os

        lines: list[str] = []

        # Ollama
        try:
            from talk_box.models import detect_ollama

            status = detect_ollama()
            if status.available:
                n_models = len(status.models) if status.models else 0
                lines.append(f"  Ollama:      [green]● running[/green] ({n_models} models)")
            else:
                lines.append("  Ollama:      [red]● not running[/red]")
        except Exception:
            lines.append("  Ollama:      [dim]not detected[/dim]")

        # API keys
        key_vars = [
            ("ANTHROPIC_API_KEY", "Anthropic"),
            ("OPENAI_API_KEY", "OpenAI"),
            ("GOOGLE_API_KEY", "Google"),
            ("GITHUB_TOKEN", "GitHub"),
        ]
        found = [name for var, name in key_vars if os.environ.get(var)]
        lines.append(f"  API Keys:    {len(found)}/{len(key_vars)} configured")

        # Personas
        try:
            from talk_box.personas._loader import list_personas

            personas = list_personas()
            lines.append(f"  Personas:    {len(personas)} available")
        except Exception:
            lines.append("  Personas:    [dim]unknown[/dim]")

        # Profiles
        try:
            from talk_box.config import list_profiles

            profiles = list_profiles()
            lines.append(f"  Profiles:    {len(profiles)} saved")
        except Exception:
            lines.append("  Profiles:    0 saved")

        # Project config detection
        try:
            from talk_box.config import project_config_path

            proj_path = project_config_path()
            if proj_path is not None:
                lines.append("  Project:     [green]talk-box.yml[/green] loaded")
            else:
                lines.append("  Project:     [dim]no talk-box.yml[/dim]")
        except Exception:
            pass

        # allow_cloud status
        try:
            from talk_box.config import load_config

            cfg = load_config()
            if not cfg.allow_cloud:
                lines.append("  Cloud:       [yellow]⚠ blocked[/yellow]")
        except Exception:
            pass

        return "\n".join(lines)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle quick action button presses."""
        actions = {
            "home-new-chat": "chat",
            "home-switch-profile": "profiles",
            "home-browse-models": "models",
            "home-browse-personas": "personas",
        }
        target = actions.get(event.button.id or "")
        if target:
            self.app._switch_to(target)  # type: ignore[attr-defined]

    def action_new_chat(self) -> None:
        """Navigate to chat screen."""
        self.app._switch_to("chat")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Screen 3: Chat
# ---------------------------------------------------------------------------


class ChatScreen(Screen):
    """Primary chat interface with message display and input.

    Supports echo mode (no model) and real LLM chat via ChatBot.
    Messages are sent in a background worker to keep the UI responsive.
    The sidebar lets users pick a persona and model, and reset the chat.
    """

    BINDINGS = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bot = None
        self._conversation = None
        self._message_count = 0
        self._active_persona: str | None = None
        self._active_model: str | None = None
        self._prompt_history: list[str] = []
        self._history_index: int = -1
        self._output_format: str | None = None  # json, markdown, table
        self._enter_sends: bool = True
        self._require_approvals: bool = True
        self._active_guards: list[str] = []
        self._active_traits: list[str] = []
        self._kg_enabled: bool = False
        self._kg: object | None = None  # KnowledgeGraph instance (lazy)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="chat-layout"):
            # Main chat area
            with Vertical(id="chat-main"):
                with VerticalScroll(id="chat-messages"):
                    yield Static(
                        "[dim]Type a message below to start chatting.[/dim]",
                        id="chat-welcome-hint",
                    )
                with Horizontal(id="chat-input-bar"):
                    yield ChatInput(
                        "",
                        id="chat-input",
                        compact=True,
                    )
                    with Vertical(id="chat-input-buttons"):
                        yield Button("Send", id="chat-send-btn", variant="primary")
                        yield Button("⏎=Send", id="chat-enter-toggle", variant="default")

            # Sidebar
            with Vertical(id="chat-sidebar"):
                yield Static("[b]Session[/b]", id="chat-sidebar-title")
                yield Static("", id="chat-sidebar-info")
                yield Static("[b]Model[/b]", id="chat-model-label")
                yield Select(
                    self._get_model_options(),
                    prompt="Select model",
                    id="chat-model-select",
                    allow_blank=True,
                )
                yield Static("[b]Persona[/b]", id="chat-persona-label")
                yield Select(
                    self._get_persona_options(),
                    prompt="Select persona",
                    id="chat-persona-select",
                    allow_blank=True,
                )
                yield Button("New Chat", id="chat-new-btn", variant="warning")
                yield Button(
                    "🔒 Approvals: On",
                    id="chat-approval-toggle",
                    variant="success",
                )
        yield Footer()

    def _get_model_options(self) -> list[tuple[str, str]]:
        """Build model select options, with favorites first, grouped by provider."""
        options: list[tuple[str, str]] = []
        fav_models: list[str] = []
        allow_cloud = True
        try:
            from talk_box.config import get_favorites, load_config

            fav_models, _ = get_favorites()
            allow_cloud = load_config().allow_cloud
        except Exception:
            pass

        # Add favorites at the top with a star prefix
        seen: set[str] = set()
        for fav in fav_models:
            if not allow_cloud and self._is_cloud_model(fav):
                continue
            friendly = self._friendly_model_name(fav)
            options.append((f"⭐ {friendly}", fav))
            seen.add(fav)

        # Group remaining models by provider
        try:
            from talk_box.models import list_models

            by_provider: dict[str, list[tuple[str, str]]] = {}
            for p in list_models():
                label = f"{p.provider}:{p.model}"
                if label in seen:
                    continue
                if not allow_cloud and self._is_cloud_model(label):
                    continue
                friendly = p.name or label
                by_provider.setdefault(p.provider, []).append((friendly, label))
                seen.add(label)

            for provider, models in by_provider.items():
                initial = provider[0].upper()
                options.append((f"── {initial} ──", f"__header__{provider}"))
                options.extend(models)
        except Exception:
            pass
        return options

    @staticmethod
    def _friendly_model_name(model_key: str) -> str:
        """Return a short display name for a model key, falling back to the key."""
        try:
            from talk_box.models import get_model_profile

            profile = get_model_profile(model_key)
            if profile:
                return profile.name
        except Exception:
            pass
        return model_key

    @staticmethod
    def _is_cloud_model(model_string: str) -> bool:
        """Check whether a model string refers to a cloud provider."""
        _cloud = {
            "anthropic",
            "openai",
            "google",
            "mistral",
            "github",
            "azure",
            "bedrock",
            "together",
        }
        provider = model_string.split(":")[0].lower() if ":" in model_string else model_string
        return provider in _cloud

    def _get_persona_options(self) -> list[tuple[str, str]]:
        """Build persona select options, with favorites first."""
        options: list[tuple[str, str]] = []
        fav_personas: list[str] = []
        try:
            from talk_box.config import get_favorites

            _, fav_personas = get_favorites()
        except Exception:
            pass

        # Add favorites at the top with a star prefix
        seen: set[str] = set()
        for fav in fav_personas:
            options.append((f"⭐ {fav}", fav))
            seen.add(fav)

        try:
            from talk_box.personas._loader import list_personas

            for name in list_personas():
                if name not in seen:
                    options.append((name, name))
                    seen.add(name)
        except Exception:
            pass
        return options

    def on_mount(self) -> None:
        """Initialize the ChatBot and focus the input."""
        self._init_bot()
        self._update_sidebar()

        # Pre-select config values in dropdowns
        try:
            model_sel = self.query_one("#chat-model-select", Select)
            if self._active_model:
                model_sel.value = self._active_model
            # Disable provider header options so they can't be selected
            self._disable_model_headers(model_sel)
        except Exception:
            pass
        try:
            persona_sel = self.query_one("#chat-persona-select", Select)
            if self._active_persona:
                persona_sel.value = self._active_persona
        except Exception:
            pass

        self.query_one("#chat-input", ChatInput).focus()

        # Load persistent prompt history
        self._load_prompt_history()

    def _init_bot(self) -> None:
        """Create a ChatBot from the resolved config."""
        if self._require_approvals:
            self._install_file_approval_callback()
        try:
            from talk_box.builder import ChatBot
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            self._active_model = resolved.model
            self._active_persona = resolved.persona

            bot = ChatBot()
            if resolved.model:
                if ":" in resolved.model:
                    bot = bot.provider_model(resolved.model)
                else:
                    bot = bot.model(resolved.model)
            if resolved.persona:
                try:
                    bot = bot.persona_pack(resolved.persona)
                except Exception:
                    pass
            if resolved.temperature is not None:
                bot = bot.temperature(resolved.temperature)

            # Always enable default chat tools; merge with any config-specified tools
            from talk_box.builtin_tools import DEFAULT_CHAT_TOOLS

            effective_tools = list(dict.fromkeys(DEFAULT_CHAT_TOOLS + resolved.tools))
            bot = bot.tools(effective_tools)
            self._active_tools = effective_tools

            # Load guardrails from config
            if resolved.guardrails:
                self._active_guards = list(resolved.guardrails)
                bot = self._apply_guards_to_bot(bot, self._active_guards)

            self._bot = bot
        except Exception:
            # Fall through to echo mode
            self._bot = None

    def _rebuild_bot(self) -> None:
        """Rebuild the ChatBot from current sidebar selections."""
        try:
            from talk_box.builder import ChatBot

            bot = ChatBot()
            if self._active_model:
                if ":" in self._active_model:
                    bot = bot.provider_model(self._active_model)
                else:
                    bot = bot.model(self._active_model)
            if self._active_persona:
                try:
                    bot = bot.persona_pack(self._active_persona)
                except Exception:
                    pass
            # Re-apply active tools (defaults loaded in _init_bot)
            if getattr(self, "_active_tools", None):
                bot = bot.tools(self._active_tools)
            else:
                from talk_box.builtin_tools import DEFAULT_CHAT_TOOLS

                bot = bot.tools(DEFAULT_CHAT_TOOLS)
            # Re-apply active traits to persona
            if getattr(self, "_active_traits", None) and self._active_persona:
                bot = self._apply_traits_to_bot(bot, self._active_traits)
            # Re-apply active guardrails
            if getattr(self, "_active_guards", None):
                bot = self._apply_guards_to_bot(bot, self._active_guards)
            self._bot = bot
        except Exception:
            self._bot = None

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model or persona selection changes."""
        if event.select.id == "chat-model-select":
            val = str(event.value) if event.value != Select.BLANK else None
            # Ignore provider header selections
            if val and val.startswith("__header__"):
                event.select.clear()
                return
            self._active_model = val
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._persist_defaults()
        elif event.select.id == "chat-persona-select":
            self._active_persona = str(event.value) if event.value != Select.BLANK else None
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._persist_defaults()

    # -- File approval integration -----------------------------------------

    _approval_all_decision: str | None = None

    def _reset_approval_all(self) -> None:
        """Clear the approve/reject-all latch (call before each LLM turn)."""
        self._approval_all_decision = None

    def _install_file_approval_callback(self) -> None:
        """Register a batch approval callback that shows a multi-file modal."""
        import threading

        from talk_box.builtin_tools import set_batch_approval_callback, set_file_approval_callback

        screen_ref = self  # capture reference for the closure

        def _batch_approval(
            pending: list[tuple[str, str, dict]],
        ) -> dict[tuple[str, str], bool]:
            """Block the worker thread, show a multi-file modal, return decisions."""
            event = threading.Event()
            result: list[dict[tuple[str, str], bool]] = []

            def _on_dismiss(decisions: dict[tuple[str, str], bool]) -> None:
                result.append(decisions)
                event.set()

            def _push_modal() -> None:
                modal = FileApprovalScreen(pending)
                screen_ref.app.push_screen(modal, callback=_on_dismiss)

            screen_ref.app.call_from_thread(_push_modal)
            event.wait()
            return result[0] if result else {(a, p): False for a, p, _ in pending}

        set_batch_approval_callback(_batch_approval)
        # Keep per-file callback as fallback for non-batched paths (e.g. chatlas fallback)
        set_file_approval_callback(
            lambda action, path, details: _batch_approval([(action, path, details)]).get(
                (action, path), False
            )
        )

    def _uninstall_file_approval_callback(self) -> None:
        """Remove the file-approval callbacks (restores auto-approve)."""
        from talk_box.builtin_tools import set_batch_approval_callback, set_file_approval_callback

        set_file_approval_callback(None)
        set_batch_approval_callback(None)

    def on_unmount(self) -> None:
        """Clean up when the screen is removed."""
        self._uninstall_file_approval_callback()

    def _persist_defaults(self) -> None:
        """Save the current model/persona as defaults in global config."""
        try:
            from talk_box.config import persist_defaults

            persist_defaults(model=self._active_model, persona=self._active_persona)
        except Exception:
            pass

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle New Chat and copy buttons."""
        btn_id = event.button.id or ""
        if btn_id == "chat-new-btn":
            self._conversation = None
            self._message_count = 0
            self._rebuild_bot()
            # Clear messages
            container = self.query_one("#chat-messages", VerticalScroll)
            await container.remove_children()
            await container.mount(
                Static(
                    "[dim]Type a message below to start chatting.[/dim]",
                    id="chat-welcome-hint",
                )
            )
            self._update_sidebar()
            self.query_one("#chat-input", ChatInput).focus()
        elif btn_id == "chat-send-btn":
            self.query_one("#chat-input", ChatInput).submit()
        elif btn_id == "chat-enter-toggle":
            self._enter_sends = not self._enter_sends
            ci = self.query_one("#chat-input", ChatInput)
            ci.enter_sends = self._enter_sends
            toggle = self.query_one("#chat-enter-toggle", Button)
            toggle.label = "⏎=Send" if self._enter_sends else "⏎=Newline"
            self.query_one("#chat-input", ChatInput).focus()
        elif btn_id == "chat-approval-toggle":
            self._require_approvals = not self._require_approvals
            toggle = self.query_one("#chat-approval-toggle", Button)
            if self._require_approvals:
                toggle.label = "🔒 Approvals: On"
                toggle.variant = "success"
                self._install_file_approval_callback()
            else:
                toggle.label = "🔓 Approvals: Off"
                toggle.variant = "default"
                self._uninstall_file_approval_callback()
            self.query_one("#chat-input", ChatInput).focus()
        elif btn_id.startswith("copy-chat-msg-"):
            msg_id = btn_id.removeprefix("copy-")
            try:
                widget = self.query_one(f"#{msg_id}", Static)
                raw = getattr(widget, "_raw_content", "")
                self.app.copy_to_clipboard(raw)
                event.button.label = "✓"
                self.set_timer(1.5, lambda: setattr(event.button, "label", "📋"))
            except Exception:
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle message submission from workspace or other Input widgets."""
        pass

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle chat message submission from the multi-line ChatInput."""
        text = event.value.strip()
        if not text:
            return
        ta = self.query_one("#chat-input", ChatInput)
        ta.clear()

        # Record in prompt history
        self._prompt_history.append(text)
        self._history_index = -1
        self._save_prompt_history()

        # Slash commands
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        self._append_message("user", text)
        self._send_message(text)

    def on_key(self, event) -> None:
        """Handle up/down arrows for prompt history in chat input."""
        try:
            ta = self.query_one("#chat-input", ChatInput)
        except Exception:
            return
        if not ta.has_focus:
            return
        if not self._prompt_history:
            return

        if event.key == "up" and ta.text.strip() == "":
            if self._history_index == -1:
                self._history_index = len(self._prompt_history) - 1
            elif self._history_index > 0:
                self._history_index -= 1
            else:
                return
            ta.clear()
            ta.insert(self._prompt_history[self._history_index])
            event.prevent_default()
        elif event.key == "down" and ta.text.strip() == "":
            if self._history_index == -1:
                return
            if self._history_index < len(self._prompt_history) - 1:
                self._history_index += 1
                ta.clear()
                ta.insert(self._prompt_history[self._history_index])
            else:
                self._history_index = -1
                ta.clear()
            event.prevent_default()

    def _handle_slash_command(self, text: str) -> None:
        """Process slash commands."""
        parts = text.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd == "/help":
            help_text = (
                "[b]Slash Commands[/b]\n"
                "  /help              Show this help\n"
                "  /clear             Clear chat history\n"
                "  /model <name>      Switch model\n"
                "  /persona <name>    Switch persona\n"
                "  /info              Show current session info\n"
                "  /system            Show the full system prompt\n"
                "  /capabilities      Show model capabilities\n"
                "  /tokens            Show token usage\n"
                "  /cost              Show session cost\n"
                "  /tools [on|off]    Manage active tools\n"
                "  /history           Show conversation history\n"
                "  /save [name]       Save session to disk\n"
                "  /load [name]       Load a saved session\n"
                "  /export [format]   Export session (json, markdown)\n"
                "  /attach <path>     Attach a file to the next message\n"
                "  /format <type>     Set output format (json, markdown, table)\n"
                "  /fav model <name>  Toggle model as favorite\n"
                "  /fav persona <n>   Toggle persona as favorite\n"
                "  /fav               List current favorites\n"
                "  /guards [on|off]   Manage guardrails\n"
                "  /traits [on|off]   Manage persona traits\n"
                "  /memory            Show memory tier summary\n"
                "  /kg [on|off|sync]  Manage knowledge context\n"
                "  /quit              Quit the app"
            )
            self._append_system_message(help_text)

        elif cmd == "/clear":
            self._do_clear()

        elif cmd == "/model":
            if arg:
                # Enforce allow_cloud restriction
                try:
                    from talk_box.config import load_config

                    cfg = load_config()
                    if not cfg.allow_cloud and self._is_cloud_model(arg):
                        self._append_system_message(
                            f"⚠ Cloud model [b]{arg}[/b] is blocked by allow_cloud=false. "
                            "Use a local model (e.g., ollama:*)."
                        )
                        return
                except Exception:
                    pass
                self._active_model = arg
                self._rebuild_bot()
                self._conversation = None
                self._update_sidebar()
                # Update the select widget to match
                try:
                    self.query_one("#chat-model-select", Select).value = arg
                except Exception:
                    pass
                self._append_system_message(f"Model changed to [b]{arg}[/b]")
                self._persist_defaults()
            else:
                model = self._active_model or "echo mode"
                self._append_system_message(f"Current model: [b]{model}[/b]")

        elif cmd == "/persona":
            if arg:
                self._active_persona = arg
                self._rebuild_bot()
                self._conversation = None
                self._update_sidebar()
                try:
                    self.query_one("#chat-persona-select", Select).value = arg
                except Exception:
                    pass
                self._append_system_message(f"Persona changed to [b]{arg}[/b]")
                self._persist_defaults()
            else:
                persona = self._active_persona or "none"
                self._append_system_message(f"Current persona: [b]{persona}[/b]")

        elif cmd == "/info":
            model = self._active_model or "[dim]echo mode[/dim]"
            persona = self._active_persona or "[dim]none[/dim]"
            msgs = self._message_count
            conv_turns = len(self._conversation.messages) if self._conversation else 0
            info = (
                f"[b]Session Info[/b]\n"
                f"  Model:       {model}\n"
                f"  Persona:     {persona}\n"
                f"  Messages:    {msgs}\n"
                f"  Conv turns:  {conv_turns}"
            )
            self._append_system_message(info)

        elif cmd == "/tokens":
            # Use real SessionUsage if available, fall back to char-based estimate
            usage = self._bot.get_usage() if self._bot else None
            if usage and usage.turns > 0:
                lines = [
                    "[b]Token Usage[/b]",
                    f"  Input tokens:   {usage.input_tokens:,}",
                    f"  Output tokens:  {usage.output_tokens:,}",
                    f"  Total tokens:   {usage.total_tokens:,}",
                    f"  Turns:          {usage.turns}",
                ]
                if usage.total_cost > 0:
                    lines.append(f"  Cost:           ${usage.total_cost:.4f}")
                self._append_system_message("\n".join(lines))
            elif self._conversation and self._conversation.messages:
                total_chars = sum(len(m.content) for m in self._conversation.messages)
                est_tokens = total_chars // 4  # rough estimate
                self._append_system_message(
                    f"[b]Token Estimate[/b] [dim](approx.)[/dim]\n"
                    f"  Conversation chars: {total_chars:,}\n"
                    f"  Est. tokens:        ~{est_tokens:,}"
                )
            else:
                self._append_system_message("No conversation history yet.")

        elif cmd == "/history":
            if self._conversation and self._conversation.messages:
                lines = ["[b]Conversation History[/b]"]
                for i, msg in enumerate(self._conversation.messages, 1):
                    role = msg.role.capitalize()
                    preview = msg.content[:80]
                    if len(msg.content) > 80:
                        preview += "…"
                    lines.append(f"  {i}. [{role}] {preview}")
                self._append_system_message("\n".join(lines))
            else:
                self._append_system_message("No conversation history yet.")

        elif cmd == "/quit":
            self.app.exit()

        elif cmd == "/save":
            self._save_session(arg)

        elif cmd == "/load":
            self._load_session(arg)

        elif cmd == "/attach":
            self._attach_file(arg)

        elif cmd == "/fav":
            self._handle_fav(arg)

        elif cmd == "/export":
            self._export_session(arg)

        elif cmd == "/guards":
            if not arg:
                self._show_guards()
            else:
                sub_parts = arg.split(None, 1)
                sub_cmd = sub_parts[0].lower()
                sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
                if sub_cmd == "on" and sub_arg:
                    self._toggle_guard(sub_arg, enable=True)
                elif sub_cmd == "off" and sub_arg:
                    self._toggle_guard(sub_arg, enable=False)
                elif sub_cmd == "available":
                    self._show_available_guards()
                else:
                    self._append_system_message(
                        "[b]Usage:[/b]\n"
                        "  /guards              Show active guardrails\n"
                        "  /guards on <name>    Enable a guardrail\n"
                        "  /guards off <name>   Disable a guardrail\n"
                        "  /guards available    List all guardrails"
                    )

        elif cmd == "/traits":
            if not arg:
                self._show_active_traits()
            else:
                sub_parts = arg.split(None, 1)
                sub_cmd = sub_parts[0].lower()
                sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
                if sub_cmd == "on" and sub_arg:
                    self._toggle_trait(sub_arg, enable=True)
                elif sub_cmd == "off" and sub_arg:
                    self._toggle_trait(sub_arg, enable=False)
                elif sub_cmd == "available":
                    self._show_available_traits()
                else:
                    self._append_system_message(
                        "[b]Usage:[/b]\n"
                        "  /traits              Show active traits\n"
                        "  /traits on <name>    Apply a trait\n"
                        "  /traits off <name>   Remove a trait\n"
                        "  /traits available    List all traits"
                    )

        elif cmd == "/memory":
            self._show_memory_summary()

        elif cmd == "/kg":
            if not arg:
                self._show_kg_summary()
            else:
                sub_cmd = arg.strip().lower()
                if sub_cmd == "on":
                    self._toggle_kg(enable=True)
                elif sub_cmd == "off":
                    self._toggle_kg(enable=False)
                elif sub_cmd == "sync":
                    self._sync_kg_sources()
                elif sub_cmd == "sources":
                    self._show_kg_sources()
                else:
                    self._append_system_message(
                        "[b]Usage:[/b]\n"
                        "  /kg              Show knowledge graph stats\n"
                        "  /kg on           Enable knowledge context in chat\n"
                        "  /kg off          Disable knowledge context\n"
                        "  /kg sync         Sync configured sources\n"
                        "  /kg sources      Show configured knowledge sources"
                    )

        elif cmd == "/cost":
            self._show_cost_estimate()

        elif cmd == "/system":
            self._show_system_prompt()

        elif cmd == "/capabilities":
            self._show_capabilities()

        elif cmd == "/tools":
            if not arg:
                self._show_active_tools()
            else:
                sub_parts = arg.split(None, 1)
                sub_cmd = sub_parts[0].lower()
                sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
                if sub_cmd == "on" and sub_arg:
                    self._toggle_tool(sub_arg, enable=True)
                elif sub_cmd == "off" and sub_arg:
                    self._toggle_tool(sub_arg, enable=False)
                elif sub_cmd == "available":
                    self._show_available_tools()
                else:
                    self._append_system_message(
                        "[b]Usage:[/b]\n"
                        "  /tools              List active tools\n"
                        "  /tools on <name>    Enable a tool\n"
                        "  /tools off <name>   Disable a tool\n"
                        "  /tools available    List all available tools"
                    )

        elif cmd == "/format":
            self._set_output_format(arg)

        else:
            self._append_system_message(
                f"Unknown command: [b]{cmd}[/b]. Type [b]/help[/b] for available commands."
            )

    def _do_clear(self) -> None:
        """Clear the chat (sync helper for slash command)."""
        self._conversation = None
        self._message_count = 0
        self._prompt_history = []
        self._history_index = -1
        self._rebuild_bot()

        async def _clear():
            container = self.query_one("#chat-messages", VerticalScroll)
            await container.remove_children()
            await container.mount(
                Static(
                    "[dim]Chat cleared. Type a message to start.[/dim]",
                    id="chat-welcome-hint",
                )
            )
            self._update_sidebar()
            self.query_one("#chat-input", ChatInput).focus()

        self.run_worker(_clear(), name="clear_chat")

    def _save_session(self, name: str) -> None:
        """Save current conversation to ~/.config/talk-box/sessions/."""
        import json
        import os
        from datetime import datetime

        if not self._conversation or not self._conversation.messages:
            self._append_system_message("Nothing to save — no conversation yet.")
            return

        sessions_dir = os.path.expanduser("~/.config/talk-box/sessions")
        os.makedirs(sessions_dir, exist_ok=True)

        if not name:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize filename
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
        if not safe_name:
            safe_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        filepath = os.path.join(sessions_dir, f"{safe_name}.json")

        session_data = {
            "name": safe_name,
            "saved_at": datetime.now().isoformat(),
            "model": self._active_model,
            "persona": self._active_persona,
            "conversation": self._conversation.to_dict(),
        }

        with open(filepath, "w") as f:
            json.dump(session_data, f, indent=2)

        self._append_system_message(f"Session saved: [b]{safe_name}[/b]\n  {filepath}")

    def _load_session(self, name: str) -> None:
        """Load a saved session from ~/.config/talk-box/sessions/."""
        import json
        import os

        sessions_dir = os.path.expanduser("~/.config/talk-box/sessions")

        if not name:
            # List available sessions
            if not os.path.isdir(sessions_dir):
                self._append_system_message("No saved sessions found.")
                return
            files = sorted(
                [f for f in os.listdir(sessions_dir) if f.endswith(".json")],
                reverse=True,
            )
            if not files:
                self._append_system_message("No saved sessions found.")
                return
            lines = ["[b]Saved Sessions[/b] (use /load <name>)"]
            for f in files[:20]:
                lines.append(f"  • {f.removesuffix('.json')}")
            self._append_system_message("\n".join(lines))
            return

        # Load specific session
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_").strip()
        filepath = os.path.join(sessions_dir, f"{safe_name}.json")

        if not os.path.isfile(filepath):
            self._append_system_message(f"Session not found: [b]{safe_name}[/b]")
            return

        try:
            from talk_box.conversation import Conversation

            with open(filepath) as f:
                data = json.load(f)

            self._conversation = Conversation.from_dict(data["conversation"])
            if data.get("model"):
                self._active_model = data["model"]
            if data.get("persona"):
                self._active_persona = data["persona"]
            self._rebuild_bot()
            self._update_sidebar()

            # Replay messages into UI
            async def _replay():
                container = self.query_one("#chat-messages", VerticalScroll)
                await container.remove_children()
                for msg in self._conversation.messages:
                    self._append_message(msg.role, msg.content)
                self._update_sidebar()

            self.run_worker(_replay(), name="load_session")
            self._append_system_message(f"Session loaded: [b]{safe_name}[/b]")
        except Exception as e:
            self._append_system_message(f"Error loading session: {e}")

    def _attach_file(self, path: str) -> None:
        """Attach a file — its contents will be included in the next message."""
        import os

        if not path:
            self._append_system_message(
                "Usage: [b]/attach <path>[/b]\n  Attaches a file to the next message sent."
            )
            return

        # Resolve relative to cwd
        full_path = os.path.join(os.getcwd(), path) if not os.path.isabs(path) else path

        if not os.path.isfile(full_path):
            self._append_system_message(f"File not found: [b]{path}[/b]")
            return

        try:
            content = open(full_path, errors="replace").read()  # noqa: SIM115
            if len(content) > 50_000:
                content = content[:50_000] + "\n\n[... truncated ...]"

            # Store attachment for next message
            if not hasattr(self, "_pending_attachments"):
                self._pending_attachments = []
            self._pending_attachments.append((path, content))

            size_kb = os.path.getsize(full_path) / 1024
            self._append_system_message(
                f"Attached: [b]{path}[/b] ({size_kb:.1f} KB)\n"
                "  Will be included with your next message."
            )
        except Exception as e:
            self._append_system_message(f"Error reading file: {e}")

    def _handle_fav(self, arg: str) -> None:
        """Handle /fav slash command for toggling favorites."""
        parts = arg.split(None, 1) if arg else []

        if not parts:
            # List current favorites
            try:
                from talk_box.config import get_favorites

                fav_models, fav_personas = get_favorites()
                lines = ["[b]Favorites[/b]"]
                lines.append("")
                lines.append("  [b]Models:[/b]")
                if fav_models:
                    for m in fav_models:
                        lines.append(f"    ⭐ {m}")
                else:
                    lines.append("    [dim]none[/dim]")
                lines.append("")
                lines.append("  [b]Personas:[/b]")
                if fav_personas:
                    for p in fav_personas:
                        lines.append(f"    ⭐ {p}")
                else:
                    lines.append("    [dim]none[/dim]")
                lines.append("")
                lines.append("  [dim]Toggle with: /fav model <name> or /fav persona <name>[/dim]")
                self._append_system_message("\n".join(lines))
            except Exception:
                self._append_system_message("Could not load favorites.")
            return

        kind = parts[0].lower()
        name = parts[1].strip() if len(parts) > 1 else ""

        if kind == "model" and name:
            try:
                from talk_box.config import toggle_favorite_model

                added = toggle_favorite_model(name)
                action = "added to" if added else "removed from"
                self._append_system_message(f"⭐ [b]{name}[/b] {action} favorite models.")
                self._refresh_selects()
            except Exception as e:
                self._append_system_message(f"Error: {e}")

        elif kind == "persona" and name:
            try:
                from talk_box.config import toggle_favorite_persona

                added = toggle_favorite_persona(name)
                action = "added to" if added else "removed from"
                self._append_system_message(f"⭐ [b]{name}[/b] {action} favorite personas.")
                self._refresh_selects()
            except Exception as e:
                self._append_system_message(f"Error: {e}")

        else:
            self._append_system_message(
                "Usage: [b]/fav model <name>[/b] or [b]/fav persona <name>[/b]\n"
                "  Or just [b]/fav[/b] to list current favorites."
            )

    def _refresh_selects(self) -> None:
        """Rebuild the model and persona Select widgets with updated options."""
        try:
            model_sel = self.query_one("#chat-model-select", Select)
            model_sel.set_options(self._get_model_options())
            if self._active_model:
                model_sel.value = self._active_model
        except Exception:
            pass
        try:
            persona_sel = self.query_one("#chat-persona-select", Select)
            persona_sel.set_options(self._get_persona_options())
            if self._active_persona:
                persona_sel.value = self._active_persona
        except Exception:
            pass

    def _export_session(self, fmt: str) -> None:
        """Export the current session to a file."""
        import json
        import os
        from datetime import datetime

        if not self._conversation or not self._conversation.messages:
            self._append_system_message("Nothing to export — no conversation yet.")
            return

        fmt = fmt.lower() if fmt else "json"
        if fmt not in ("json", "markdown", "md"):
            self._append_system_message(
                "Supported formats: [b]json[/b], [b]markdown[/b]\n"
                "  Usage: /export json  or  /export markdown"
            )
            return

        sessions_dir = os.path.expanduser("~/.config/talk-box/exports")
        os.makedirs(sessions_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if fmt == "json":
            filepath = os.path.join(sessions_dir, f"chat_{ts}.json")
            data = {
                "exported_at": datetime.now().isoformat(),
                "model": self._active_model,
                "persona": self._active_persona,
                "messages": [
                    {"role": m.role, "content": m.content} for m in self._conversation.messages
                ],
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        else:
            # Markdown
            filepath = os.path.join(sessions_dir, f"chat_{ts}.md")
            lines = [
                f"# Chat Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                f"**Model:** {self._active_model or 'echo mode'}",
                f"**Persona:** {self._active_persona or 'none'}",
                "",
                "---",
                "",
            ]
            for msg in self._conversation.messages:
                role = "**You**" if msg.role == "user" else "**Assistant**"
                lines.append(f"### {role}")
                lines.append("")
                lines.append(msg.content)
                lines.append("")
            with open(filepath, "w") as f:
                f.write("\n".join(lines))

        self._append_system_message(f"Exported to [b]{filepath}[/b]")

    def _show_guards(self) -> None:
        """Show guardrail status in chat."""
        active = getattr(self, "_active_guards", None) or []
        lines = [f"[b]Active Guardrails[/b]  ({len(active)})"]
        if active:
            for g in active:
                lines.append(f"  • {g}")
        else:
            lines.append("  [dim]No guardrails active[/dim]")
        self._append_system_message("\n".join(lines))

    def _show_available_guards(self) -> None:
        """Show all available guardrails with active status."""
        active = set(getattr(self, "_active_guards", None) or [])
        lines = [f"[b]Available Guardrails[/b]  ({len(_BUILTIN_GUARDS)})"]
        for name, phase, desc in _BUILTIN_GUARDS:
            marker = "✓" if name in active else " "
            lines.append(f"  [{marker}] {name}  ({phase}) — {desc}")
        self._append_system_message("\n".join(lines))

    def _toggle_guard(self, name: str, *, enable: bool) -> None:
        """Enable or disable a guardrail by name, rebuild bot, and persist."""
        known = {n for n, _, _ in _BUILTIN_GUARDS}
        if name not in known:
            self._append_system_message(
                f"Unknown guardrail [b]{name}[/b]. "
                "Use [b]/guards available[/b] to see all guardrails."
            )
            return

        active = list(getattr(self, "_active_guards", None) or [])
        if enable:
            if name in active:
                self._append_system_message(f"Guardrail [b]{name}[/b] is already active.")
                return
            active.append(name)
            self._active_guards = active
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._append_system_message(f"Guardrail [b]{name}[/b] enabled.")
        else:
            if name not in active:
                self._append_system_message(f"Guardrail [b]{name}[/b] is not active.")
                return
            active.remove(name)
            self._active_guards = active
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._append_system_message(f"Guardrail [b]{name}[/b] disabled.")

        try:
            from talk_box.config import persist_guardrails

            persist_guardrails(self._active_guards)
        except Exception:
            pass

    @staticmethod
    def _apply_guards_to_bot(bot, guard_names: list[str]) -> "ChatBot":
        """Resolve guard names and add them to a ChatBot."""
        try:
            from talk_box.guardrails import resolve_guards

            for guard in resolve_guards(guard_names):
                bot = bot.guardrail(guard)
        except Exception:
            pass
        return bot

    def _show_active_traits(self) -> None:
        """Show the list of currently active traits."""
        active = getattr(self, "_active_traits", None) or []
        if active:
            lines = [f"[b]Active Traits[/b]  ({len(active)})"]
            for t in active:
                lines.append(f"  • {t}")
            self._append_system_message("\n".join(lines))
        else:
            self._append_system_message("No traits are active.")

    def _show_available_traits(self) -> None:
        """Show all available traits with active status."""
        try:
            from talk_box.traits import list_traits

            all_traits = list_traits()
        except Exception:
            self._append_system_message("Could not load trait list.")
            return
        active = set(getattr(self, "_active_traits", None) or [])
        lines = [f"[b]Available Traits[/b]  ({len(all_traits)})"]
        for t in all_traits:
            marker = "✓" if t in active else " "
            lines.append(f"  [{marker}] {t}")
        self._append_system_message("\n".join(lines))

    def _toggle_trait(self, name: str, *, enable: bool) -> None:
        """Enable or disable a trait by name, rebuild bot."""
        try:
            from talk_box.traits import list_traits

            available = list_traits()
        except Exception:
            available = []

        if name not in available:
            self._append_system_message(
                f"Unknown trait [b]{name}[/b]. Use [b]/traits available[/b] to see all traits."
            )
            return

        if not self._active_persona:
            self._append_system_message(
                "Traits require an active persona. Set one with [b]/persona <name>[/b] first."
            )
            return

        active = list(getattr(self, "_active_traits", None) or [])
        if enable:
            if name in active:
                self._append_system_message(f"Trait [b]{name}[/b] is already active.")
                return
            active.append(name)
            self._active_traits = active
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._append_system_message(f"Trait [b]{name}[/b] applied.")
        else:
            if name not in active:
                self._append_system_message(f"Trait [b]{name}[/b] is not active.")
                return
            active.remove(name)
            self._active_traits = active
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._append_system_message(f"Trait [b]{name}[/b] removed.")

    @staticmethod
    def _apply_traits_to_bot(bot, trait_names: list[str]) -> "ChatBot":
        """Apply traits to the bot's active persona."""
        try:
            from talk_box.traits import apply_trait, get_trait

            persona = bot._config.get("persona_definition")
            if persona is None:
                return bot
            import copy

            modified = copy.deepcopy(persona)
            for name in trait_names:
                try:
                    trait = get_trait(name)
                    modified = apply_trait(modified, trait)
                except Exception:
                    pass
            bot._config["persona_definition"] = modified
        except Exception:
            pass
        return bot

    def _show_memory_summary(self) -> None:
        """Show memory tier summary in chat."""
        import os

        lines = ["[b]Memory Summary[/b]"]
        try:
            from talk_box.memory import LongTermMemory

            db_path = os.path.expanduser("~/.config/talk-box/memory.db")
            if os.path.isfile(db_path):
                ltm = LongTermMemory(path=db_path)
                entries = ltm.entries()
                lines.append(f"  Long-term: {len(entries)} entries ({db_path})")
                ltm.close()
            else:
                lines.append("  Long-term: [dim]no database[/dim]")
        except Exception:
            lines.append("  Long-term: [dim]unavailable[/dim]")

        # Working memory (session)
        if self._conversation and self._conversation.messages:
            lines.append(f"  Working:   {len(self._conversation.messages)} messages in session")
        else:
            lines.append("  Working:   [dim]empty[/dim]")

        self._append_system_message("\n".join(lines))

    def _get_kg_path(self) -> str:
        """Return the persistent knowledge graph database path."""
        import os

        return os.path.join(os.path.expanduser("~/.config/talk-box"), "knowledge.db")

    def _get_kg(self):
        """Return the shared KnowledgeGraph instance (lazy init)."""
        if self._kg is None:
            from talk_box.knowledge_graph import KnowledgeGraph

            self._kg = KnowledgeGraph(self._get_kg_path())
        return self._kg

    def _show_kg_summary(self) -> None:
        """Show knowledge graph summary in chat."""
        lines = ["[b]Knowledge Graph[/b]"]
        status = "on" if self._kg_enabled else "off"
        lines.append(f"  Context injection: [b]{status}[/b]")
        try:
            from talk_box.knowledge_graph import NodeType

            kg = self._get_kg()
            docs = kg.node_count(node_type=NodeType.DOCUMENT)
            entities = kg.node_count(node_type=NodeType.ENTITY)
            topics = kg.node_count(node_type=NodeType.TOPIC)
            edges = kg.edge_count()
            lines.append(f"  Documents:  {docs}")
            lines.append(f"  Entities:   {entities}")
            lines.append(f"  Topics:     {topics}")
            lines.append(f"  Edges:      {edges}")

            if docs + entities + topics == 0:
                lines.append("")
                lines.append("  [dim]No data yet. Use /kg sync to load sources.[/dim]")
        except Exception:
            lines.append("  [dim]Could not load knowledge graph[/dim]")
        self._append_system_message("\n".join(lines))

    def _toggle_kg(self, *, enable: bool) -> None:
        """Enable or disable knowledge context injection in chat."""
        if enable:
            if self._kg_enabled:
                self._append_system_message("Knowledge context is already [b]on[/b].")
                return
            # Check if KG has any nodes
            try:
                kg = self._get_kg()
                count = kg.node_count()
                if count == 0:
                    self._append_system_message(
                        "Knowledge graph is empty. Use [b]/kg sync[/b] first to load sources."
                    )
                    return
            except Exception:
                pass
            self._kg_enabled = True
            self._update_sidebar()
            self._append_system_message(
                "Knowledge context [b]enabled[/b]. "
                "Relevant knowledge will be injected into each message."
            )
        else:
            if not self._kg_enabled:
                self._append_system_message("Knowledge context is already [b]off[/b].")
                return
            self._kg_enabled = False
            self._update_sidebar()
            self._append_system_message("Knowledge context [b]disabled[/b].")

    def _sync_kg_sources(self) -> None:
        """Sync configured knowledge sources into the knowledge graph."""
        try:
            from talk_box.config import load_config

            config = load_config()
            sources = config.knowledge.sources if config.knowledge else []
        except Exception:
            sources = []

        if not sources:
            self._append_system_message(
                "[b]No knowledge sources configured.[/b]\n"
                "  Add sources to your talk-box.yml:\n"
                "  [dim]knowledge:\n"
                "    sources:\n"
                "      - path: './docs/'\n"
                "        type: markdown[/dim]"
            )
            return

        try:
            from talk_box.connectors import DirectoryConnector, MarkdownDir, sync

            kg = self._get_kg()
            connectors = []
            for src in sources:
                if src.type == "markdown":
                    connectors.append(MarkdownDir(src.path))
                else:
                    connectors.append(DirectoryConnector(src.path))

            result = sync(kg, *connectors)
            self._update_sidebar()
            self._append_system_message(
                f"[b]Knowledge sync complete[/b]\n"
                f"  Added:     {result.added}\n"
                f"  Updated:   {result.updated}\n"
                f"  Unchanged: {result.unchanged}"
            )
        except Exception as e:
            self._append_system_message(f"Sync failed: {e}")

    def _show_kg_sources(self) -> None:
        """Show configured knowledge sources."""
        try:
            from talk_box.config import load_config

            config = load_config()
            sources = config.knowledge.sources if config.knowledge else []
        except Exception:
            sources = []

        if not sources:
            self._append_system_message(
                "[b]Knowledge Sources[/b]\n  [dim]No sources configured.[/dim]"
            )
            return

        lines = [f"[b]Knowledge Sources[/b]  ({len(sources)})"]
        for src in sources:
            lines.append(f"  • {src.path}  ({src.type})")
        self._append_system_message("\n".join(lines))

    def _enrich_with_knowledge(self, text: str) -> str:
        """Search the knowledge graph and inject relevant context."""
        if not self._kg_enabled:
            return text
        try:
            kg = self._get_kg()
            # Search with individual significant words (>3 chars) for broader matching
            words = [w for w in text.split() if len(w) > 3]
            seen_ids: set[str] = set()
            all_nodes = []
            for word in words[:5]:  # Cap at 5 search terms
                for node in kg.search(word, limit=3):
                    if node.id not in seen_ids:
                        seen_ids.add(node.id)
                        all_nodes.append(node)
            # Also try full query as a phrase
            for node in kg.search(text, limit=3):
                if node.id not in seen_ids:
                    seen_ids.add(node.id)
                    all_nodes.append(node)

            if not all_nodes:
                return text

            context_parts = []
            for node in all_nodes[:5]:
                # Cap each node's content at 2000 chars
                content = node.content or ""
                if len(content) > 2000:
                    content = content[:2000] + "\n[... truncated ...]"
                context_parts.append(
                    f'<knowledge name="{node.name}" type="{node.node_type.value}">\n'
                    f"{content}\n"
                    f"</knowledge>"
                )
            knowledge_block = "\n\n".join(context_parts)
            return f"{text}\n\n--- Knowledge context ---\n{knowledge_block}"
        except Exception:
            return text

    def _show_cost_estimate(self) -> None:
        """Show session cost — uses real SessionUsage data when available."""
        # Try real usage data first
        usage = self._bot.get_usage() if self._bot else None
        if usage and usage.turns > 0:
            model = self._active_model or "unknown"
            is_free = "ollama" in model.lower() or "local" in model.lower()
            lines = [
                "[b]Session Cost[/b]",
                f"  Model:          {model}",
                f"  Input tokens:   {usage.input_tokens:,}",
                f"  Output tokens:  {usage.output_tokens:,}",
                f"  Turns:          {usage.turns}",
            ]
            if is_free:
                lines.append("  Cost:           [green]$0.00 (local model)[/green]")
            elif usage.total_cost > 0:
                lines.append(f"  [b]Cost:          ${usage.total_cost:.4f}[/b]")
            else:
                lines.append("  Cost:           [dim]not available from provider[/dim]")
            self._append_system_message("\n".join(lines))
            return

        # Fall back to character-based estimate
        # Approximate cost per 1M tokens (input/output) by model family
        _COST_TABLE: dict[str, tuple[float, float]] = {
            "claude-sonnet-4-6": (3.0, 15.0),
            "claude-opus-4": (15.0, 75.0),
            "claude-haiku": (0.25, 1.25),
            "gpt-4o": (2.5, 10.0),
            "gpt-4o-mini": (0.15, 0.6),
            "gpt-4.1": (2.0, 8.0),
            "gpt-4.1-mini": (0.4, 1.6),
            "o3": (2.0, 8.0),
            "o4-mini": (1.1, 4.4),
            "gemini-2.5-pro": (1.25, 10.0),
            "gemini-2.5-flash": (0.15, 0.6),
        }

        if not self._conversation or not self._conversation.messages:
            self._append_system_message("No conversation yet — no cost to estimate.")
            return

        # Estimate input/output tokens
        input_chars = 0
        output_chars = 0
        for m in self._conversation.messages:
            if m.role == "user":
                input_chars += len(m.content)
            else:
                output_chars += len(m.content)

        input_tokens = input_chars // 4
        output_tokens = output_chars // 4

        # Find model pricing
        model = self._active_model or "unknown"
        model_lower = model.lower()
        input_cost_per_m = 0.0
        output_cost_per_m = 0.0
        matched = False
        for key, (ic, oc) in _COST_TABLE.items():
            if key in model_lower:
                input_cost_per_m = ic
                output_cost_per_m = oc
                matched = True
                break

        # Check for free/local models
        is_free = "ollama" in model_lower or "local" in model_lower

        lines = [
            "[b]Session Cost Estimate[/b]",
            f"  Model:          {model}",
            f"  Input tokens:   ~{input_tokens:,}",
            f"  Output tokens:  ~{output_tokens:,}",
        ]

        if is_free:
            lines.append("  Cost:           [green]$0.00 (local model)[/green]")
        elif matched:
            input_cost = (input_tokens / 1_000_000) * input_cost_per_m
            output_cost = (output_tokens / 1_000_000) * output_cost_per_m
            total = input_cost + output_cost
            lines.append(f"  Input cost:     ${input_cost:.4f}")
            lines.append(f"  Output cost:    ${output_cost:.4f}")
            lines.append(f"  [b]Total:         ${total:.4f}[/b]")
        else:
            lines.append("  Cost:           [dim]unknown model — no pricing data[/dim]")

        self._append_system_message("\n".join(lines))

    def _show_system_prompt(self) -> None:
        """Show the full constructed system prompt."""
        if self._bot is None:
            self._append_system_message("No bot configured — running in echo mode.")
            return
        try:
            prompt = self._bot.get_system_prompt()
            if prompt:
                # Truncate very long prompts for display
                if len(prompt) > 5000:
                    prompt = prompt[:5000] + "\n\n[dim]... (truncated at 5000 chars)[/dim]"
                self._append_system_message(f"[b]System Prompt[/b]\n\n{prompt}")
            else:
                self._append_system_message("No system prompt configured.")
        except Exception:
            self._append_system_message("Could not retrieve the system prompt.")

    def _show_capabilities(self) -> None:
        """Show model capabilities from the model profile."""
        model = self._active_model or "unknown"
        lines = [f"[b]Model Capabilities[/b]  —  {model}"]

        try:
            from talk_box.models import get_model_profile

            # Try provider:model key first, then just model name
            profile = get_model_profile(model)
            if profile is None and ":" in model:
                profile = get_model_profile(model.split(":", 1)[1])

            if profile:
                lines.append(f"  Provider:           {profile.provider}")
                lines.append(f"  Display name:       {profile.name}")
                if profile.context_window:
                    lines.append(f"  Context window:     {profile.context_window:,} tokens")
                if profile.max_output_tokens:
                    lines.append(f"  Max output tokens:  {profile.max_output_tokens:,}")

                def _cap(val: bool | None) -> str:
                    if val is True:
                        return "[green]✓[/green]"
                    if val is False:
                        return "[red]✗[/red]"
                    return "[dim]?[/dim]"

                lines.append(f"  Tools:              {_cap(profile.supports_tools)}")
                lines.append(f"  Vision:             {_cap(profile.supports_vision)}")
                lines.append(f"  Structured output:  {_cap(profile.supports_structured_output)}")
                lines.append(f"  Streaming:          {_cap(profile.supports_streaming)}")
                if profile.cost_tier:
                    lines.append(f"  Cost tier:          {profile.cost_tier.value}")
                if profile.knowledge_cutoff:
                    lines.append(f"  Knowledge cutoff:   {profile.knowledge_cutoff}")
            else:
                lines.append("  [dim]No model profile found — capabilities unknown.[/dim]")
        except Exception:
            lines.append("  [dim]Could not load model profiles.[/dim]")

        self._append_system_message("\n".join(lines))

    # -- Sidebar action link handlers -------------------------------------

    def action_open_tools_picker(self) -> None:
        """Action handler for sidebar tools [...] link."""
        self._open_tools_picker()

    def action_open_guards_picker(self) -> None:
        """Action handler for sidebar guards [...] link."""
        self._open_guards_picker()

    def action_open_traits_picker(self) -> None:
        """Action handler for sidebar traits [...] link."""
        self._open_traits_picker()

    def action_toggle_kg_inline(self) -> None:
        """Action handler for sidebar KG toggle switch."""
        self._handle_kg_button()

    def action_open_kg_screen(self) -> None:
        """Action handler for sidebar KG [...] link — navigate to KG screen."""
        try:
            self.app._switch_to("knowledge")
        except Exception:
            pass

    # -- Picker modal openers ---------------------------------------------

    def _open_tools_picker(self) -> None:
        """Open a modal to toggle tools on/off."""
        try:
            from talk_box.builtin_tools import get_all_tool_box_tools

            all_tools = get_all_tool_box_tools()
        except Exception:
            self._append_system_message("Could not load tool list.")
            return
        active = list(getattr(self, "_active_tools", None) or [])
        items = [(t, "") for t in all_tools]

        def _on_dismiss(result: list[str] | None) -> None:
            if result is None:
                return  # cancelled
            self._active_tools = result
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._persist_tools_config()
            self._append_system_message(f"Tools updated: {len(result)} active.")

        self.app.push_screen(ChecklistPickerModal("Tools", items, active), _on_dismiss)

    def _open_guards_picker(self) -> None:
        """Open a modal to toggle guardrails on/off."""
        items = [(name, desc) for name, _phase, desc in _BUILTIN_GUARDS]
        active = list(getattr(self, "_active_guards", None) or [])

        def _on_dismiss(result: list[str] | None) -> None:
            if result is None:
                return
            self._active_guards = result
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            try:
                from talk_box.config import persist_guardrails

                persist_guardrails(self._active_guards)
            except Exception:
                pass
            self._append_system_message(f"Guardrails updated: {len(result)} active.")

        self.app.push_screen(ChecklistPickerModal("Guardrails", items, active), _on_dismiss)

    def _open_traits_picker(self) -> None:
        """Open a modal to toggle persona traits on/off."""
        if not self._active_persona:
            self._append_system_message(
                "Traits require an active persona. Set one with [b]/persona <name>[/b] first."
            )
            return

        try:
            from talk_box.traits import list_traits

            all_traits = list_traits()
        except Exception:
            self._append_system_message("Could not load trait list.")
            return
        active = list(getattr(self, "_active_traits", None) or [])
        items = [(t, "") for t in all_traits]

        def _on_dismiss(result: list[str] | None) -> None:
            if result is None:
                return
            self._active_traits = result
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._append_system_message(f"Traits updated: {len(result)} active.")

        self.app.push_screen(ChecklistPickerModal("Traits", items, active), _on_dismiss)

    def _handle_kg_button(self) -> None:
        """Toggle knowledge graph context from the sidebar link."""
        if self._kg_enabled:
            self._kg_enabled = False
            self._update_sidebar()
            self._append_system_message("Knowledge context [b]disabled[/b].")
        else:
            # Check if KG has nodes
            try:
                kg = self._get_kg()
                count = kg.node_count()
                if count == 0:
                    self._append_system_message(
                        "Knowledge graph is empty. Use [b]/kg sync[/b] to load sources first."
                    )
                    return
            except Exception:
                pass
            self._kg_enabled = True
            self._update_sidebar()
            self._append_system_message(
                "Knowledge context [b]enabled[/b]. "
                "Relevant knowledge will be injected into each message."
            )

    def _show_active_tools(self) -> None:
        """Show the list of currently active tools."""
        tools = getattr(self, "_active_tools", None)
        if tools:
            lines = [f"[b]Active Tools[/b]  ({len(tools)})"]
            for t in tools:
                lines.append(f"  • {t}")
            self._append_system_message("\n".join(lines))
        else:
            self._append_system_message("No tools are active.")

    def _show_available_tools(self) -> None:
        """Show all available builtin tools with active status."""
        try:
            from talk_box.builtin_tools import get_all_tool_box_tools

            all_tools = get_all_tool_box_tools()
        except Exception:
            self._append_system_message("Could not load tool list.")
            return
        active = set(getattr(self, "_active_tools", None) or [])
        lines = [f"[b]Available Tools[/b]  ({len(all_tools)})"]
        for t in all_tools:
            marker = "✓" if t in active else " "
            lines.append(f"  [{marker}] {t}")
        self._append_system_message("\n".join(lines))

    def _toggle_tool(self, name: str, *, enable: bool) -> None:
        """Enable or disable a tool by name, rebuild bot, and persist."""
        try:
            from talk_box.builtin_tools import get_all_tool_box_tools

            available = get_all_tool_box_tools()
        except Exception:
            available = []

        if name not in available:
            self._append_system_message(
                f"Unknown tool [b]{name}[/b]. Use [b]/tools available[/b] to see all tools."
            )
            return

        active = list(getattr(self, "_active_tools", None) or [])
        if enable:
            if name in active:
                self._append_system_message(f"Tool [b]{name}[/b] is already active.")
                return
            active.append(name)
            self._active_tools = active
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._append_system_message(f"Tool [b]{name}[/b] enabled.")
        else:
            if name not in active:
                self._append_system_message(f"Tool [b]{name}[/b] is not active.")
                return
            active.remove(name)
            self._active_tools = active
            self._rebuild_bot()
            self._conversation = None
            self._update_sidebar()
            self._append_system_message(f"Tool [b]{name}[/b] disabled.")

        self._persist_tools_config()

    def _persist_tools_config(self) -> None:
        """Save current active tools to project talk-box.yml."""
        try:
            from yaml12 import read_yaml, write_yaml

            from talk_box.config import project_config_path

            path = project_config_path()
            if path and path.is_file():
                data = read_yaml(path) or {}
            else:
                return  # No project config — skip persistence
            data["tools"] = list(self._active_tools) if self._active_tools else []
            write_yaml(data, path)
        except Exception:
            pass

    def _set_output_format(self, fmt: str) -> None:
        """Set the output format for assistant responses."""
        valid = {"json", "markdown", "table"}
        if not fmt:
            current = self._output_format or "default"
            self._append_system_message(
                f"[b]Output Format[/b]: {current}\n  Usage: /format json | markdown | table | off"
            )
            return
        fmt = fmt.lower()
        if fmt == "off":
            self._output_format = None
            self._append_system_message("Output format reset to [b]default[/b]")
        elif fmt in valid:
            self._output_format = fmt
            self._append_system_message(f"Output format set to [b]{fmt}[/b]")
        else:
            self._append_system_message(
                f"Unknown format: [b]{fmt}[/b]\n  Supported: json, markdown, table, off"
            )

    def _auto_save_session(self) -> None:
        """Auto-save the current session after each exchange."""
        import json
        import os
        from datetime import datetime

        if not self._conversation or not self._conversation.messages:
            return

        sessions_dir = os.path.join(_config_dir(), "sessions")
        os.makedirs(sessions_dir, exist_ok=True)
        filepath = os.path.join(sessions_dir, "_autosave.json")

        data = {
            "name": "_autosave",
            "saved_at": datetime.now().isoformat(),
            "model": self._active_model,
            "persona": self._active_persona,
            "conversation": self._conversation.to_dict(),
        }
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _save_prompt_history(self) -> None:
        """Persist prompt history to disk."""
        import json
        import os

        history_path = os.path.join(_config_dir(), "prompt_history.json")
        os.makedirs(os.path.dirname(history_path), exist_ok=True)
        try:
            # Keep last 200 entries
            to_save = self._prompt_history[-200:]
            with open(history_path, "w") as f:
                json.dump(to_save, f)
        except Exception:
            pass

    def _load_prompt_history(self) -> None:
        """Load prompt history from disk."""
        import json
        import os

        history_path = os.path.join(_config_dir(), "prompt_history.json")
        try:
            if os.path.isfile(history_path):
                with open(history_path) as f:
                    self._prompt_history = json.load(f)
        except Exception:
            self._prompt_history = []

    def _append_system_message(self, content: str) -> None:
        """Append a system/info message (not from user or assistant)."""
        container = self.query_one("#chat-messages", VerticalScroll)
        self._message_count += 1
        msg_id = f"chat-msg-{self._message_count}"
        widget = Static(content, id=msg_id, classes="chat-message chat-system")
        container.mount(widget)
        container.scroll_end(animate=False)

    def _append_message(self, role: str, content: str) -> None:
        """Add a message bubble to the chat area."""
        container = self.query_one("#chat-messages", VerticalScroll)

        # Remove the welcome hint on first message
        hints = container.query("#chat-welcome-hint")
        for hint in hints:
            hint.remove()

        self._message_count += 1
        msg_id = f"chat-msg-{self._message_count}"

        if role == "user":
            label = "[b]You[/b]"
            classes = "chat-message chat-user"
            display_content = f"{label}\n{content}"
        else:
            label = "[b]Assistant[/b]"
            classes = "chat-message chat-assistant"
            display_content = self._render_assistant_display(content)

        widget = Static(display_content, id=msg_id, classes=classes)
        widget._raw_content = content
        copy_btn = Button("📋", id=f"copy-{msg_id}", classes="chat-copy-btn")
        wrapper = Horizontal(widget, copy_btn, classes=f"chat-msg-row {role}-row")
        container.mount(wrapper)
        container.scroll_end(animate=False)
        self._update_sidebar()

    def _send_message(self, text: str) -> None:
        """Send a message via ChatBot, streaming the response."""
        self._append_message("assistant", "[dim]Thinking…[/dim]")
        thinking_id = f"chat-msg-{self._message_count}"
        self._pending_thinking_id = thinking_id
        try:
            self.query_one(f"#{thinking_id}", Static).add_class("chat-thinking")
        except Exception:
            pass

        # Include pending attachments
        enriched = text
        if hasattr(self, "_pending_attachments") and self._pending_attachments:
            attachment_blocks = []
            for path, content in self._pending_attachments:
                attachment_blocks.append(f'<file path="{path}">\n{content}\n</file>')
            enriched = f"{text}\n\n--- Attached files ---\n" + "\n\n".join(attachment_blocks)
            self._pending_attachments = []

        # Also enrich with auto-detected file references
        enriched = self._enrich_with_files(enriched)

        # Enrich with knowledge graph context (if enabled)
        enriched = self._enrich_with_knowledge(enriched)

        # Inject output format instruction if set
        if self._output_format:
            fmt_instruction = {
                "json": "Respond with valid JSON only.",
                "markdown": "Respond in Markdown format.",
                "table": "Respond using Markdown tables where appropriate.",
            }.get(self._output_format, "")
            if fmt_instruction:
                enriched = f"[Output format: {fmt_instruction}]\n\n{enriched}"

        async def _do_stream():
            import asyncio

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._stream_response, enriched, thinking_id)

        self.run_worker(
            _do_stream(),
            name="chat_response",
            group="chat",
        )

    def _enrich_with_files(self, text: str) -> str:
        """Detect file references in the message and append their contents."""
        import os
        import re

        # Match patterns like filename.ext or path/to/file.ext
        # Must have an extension to avoid false positives
        pattern = r'(?:^|\s|["\'])([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+)(?:\s|["\']|$|[?,!;:])'
        matches = re.findall(pattern, text)
        if not matches:
            return text

        attached = []
        seen = set()
        for match in matches:
            if match in seen:
                continue
            seen.add(match)
            path = os.path.join(os.getcwd(), match)
            if os.path.isfile(path):
                try:
                    content = open(path, errors="replace").read()  # noqa: SIM115
                    # Cap at 50K chars to avoid blowing context
                    if len(content) > 50_000:
                        content = content[:50_000] + "\n\n[... truncated ...]"
                    attached.append(f'<file path="{match}">\n{content}\n</file>')
                except Exception:
                    pass

        if not attached:
            return text

        files_block = "\n\n".join(attached)
        return f"{text}\n\n--- Referenced files ---\n{files_block}"

    def _stream_response(self, text: str, widget_id: str) -> None:
        """Stream the LLM response, updating the widget chunk by chunk (runs in thread)."""

        # Reset the approve/reject-all latch for this new exchange
        self._reset_approval_all()

        thinking_text = ""
        response_text = ""
        try:
            if self._bot is not None and self._bot._llm_enabled:
                from talk_box.conversation import Conversation

                if self._conversation is None:
                    self._conversation = Conversation()

                try:
                    for phase, chunk in self._bot._stream_with_thinking(text, self._conversation):
                        if phase == "thinking":
                            thinking_text += chunk
                            display = self._format_stream_display(thinking_text, "")
                            self.app.call_from_thread(
                                self._update_stream_widget, widget_id, display
                            )
                        elif phase == "text":
                            response_text += chunk
                            display = self._format_stream_display(thinking_text, response_text)
                            self.app.call_from_thread(
                                self._update_stream_widget, widget_id, display
                            )

                    # Add the completed exchange to conversation
                    # Store original message (without injected file/format prefixes)
                    original = text.split("\n\n--- Referenced files ---\n")[0]
                    original = original.split("\n\n--- Attached files ---\n")[0]
                    if original.startswith("[Output format:"):
                        # Strip the format instruction prefix
                        original = original.split("\n\n", 1)[-1] if "\n\n" in original else original
                    self._conversation.add_message("user", original)
                    self._conversation.add_message("assistant", response_text)

                except Exception as e:
                    response_text = f"Error: {e}"
            else:
                response_text = f"Echo: {text}"
        except Exception as e:
            response_text = f"Error: {e}"

        # Final update with raw content
        self.app.call_from_thread(
            self._finalize_stream_widget,
            widget_id,
            response_text,
            thinking_text,
        )

    @staticmethod
    def _format_stream_display(thinking: str, response: str) -> str:
        """Format the streaming display with thinking and response sections."""
        parts = ["[b]Assistant[/b]"]
        if thinking:
            parts.append(f"[dim]💭 {thinking}[/dim]")
        if response:
            parts.append(response)
        elif not thinking:
            parts.append("[dim]…[/dim]")
        return "\n".join(parts)

    @staticmethod
    def _render_assistant_display(content: str, *, thinking: str = "") -> "RenderableType":
        """Render assistant content as Rich Markdown with optional thinking."""
        from rich.console import Group
        from rich.markdown import Markdown
        from rich.text import Text

        parts = []
        parts.append(Text.from_markup("[b]Assistant[/b]"))
        if thinking:
            parts.append(Text.from_markup(f"[dim]💭 {thinking}[/dim]"))
        if content:
            parts.append(Markdown(content))
        elif not thinking:
            parts.append(Text.from_markup("[dim]…[/dim]"))
        return Group(*parts)

    def _update_stream_widget(self, widget_id: str, display: str) -> None:
        """Update the streaming widget with new content (called on main thread)."""
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            widget.update(display)
            container = self.query_one("#chat-messages", VerticalScroll)
            container.scroll_end(animate=False)
        except Exception:
            pass

    def _finalize_stream_widget(self, widget_id: str, content: str, thinking: str = "") -> None:
        """Finalize the streamed widget with rich markdown rendering (called on main thread)."""
        try:
            widget = self.query_one(f"#{widget_id}", Static)
            display = self._render_assistant_display(content, thinking=thinking)
            widget.update(display)
            widget._raw_content = content
            widget.remove_class("chat-thinking")
            container = self.query_one("#chat-messages", VerticalScroll)
            container.scroll_end(animate=False)
            self._update_sidebar()
            self._auto_save_session()
        except Exception:
            pass

    def _get_response(self, text: str) -> str:
        """Get a response from the bot (runs in worker thread). Fallback for non-streaming."""
        if self._bot is not None:
            try:
                from talk_box.conversation import Conversation

                if self._conversation is None:
                    self._conversation = Conversation()
                self._conversation = self._bot.chat(text, self._conversation)
                last = self._conversation.get_last_message()
                return last.content if last else "No response."
            except Exception as e:
                return f"Error: {e}"
        else:
            return f"Echo: {text}"

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Handle worker completion."""
        if event.worker.name != "chat_response":
            return
        if event.state == WorkerState.SUCCESS:
            # Streaming updates are handled inline; just ensure sidebar is current
            self._update_sidebar()

    def _build_sidebar(self) -> str:
        """Build sidebar info text with clickable action links."""
        model_key = self._active_model
        if model_key:
            model = self._friendly_model_name(model_key)
        else:
            model = "[dim]echo mode[/dim]"
        persona = self._active_persona or "[dim]none[/dim]"
        user_msgs = self._message_count // 2 if self._message_count > 0 else 0

        lines = [
            f"  Model:   {model}",
            f"  Persona: {persona}",
            f"  Msgs:    {user_msgs}",
        ]

        # Token estimate from conversation
        if self._conversation and self._conversation.messages:
            total_chars = sum(len(m.content) for m in self._conversation.messages)
            est_tokens = total_chars // 4
            lines.append(f"  Tokens:  ~{est_tokens:,}")
        else:
            lines.append("  Tokens:  0")

        # Context window usage
        if self._active_model:
            try:
                from talk_box.models import get_model_profile

                profile = get_model_profile(self._active_model)
                if profile and profile.context_window:
                    total_chars = (
                        sum(len(m.content) for m in self._conversation.messages)
                        if self._conversation and self._conversation.messages
                        else 0
                    )
                    est_tokens = total_chars // 4
                    pct = (est_tokens / profile.context_window) * 100
                    lines.append(f"  Context: {pct:.0f}% of {profile.context_window:,}")
            except Exception:
                pass

        # -- Concise config status with clickable [...] links ---------
        lines.append("")

        # Usable width inside sidebar (width 34 - 1 border - 2 padding)
        _W = 31

        def _vis_width(text: str) -> int:
            """Return the visible display width of *text*, accounting for
            double-width emoji/CJK characters and stripping Rich markup."""
            import re
            import unicodedata

            stripped = re.sub(r"\[/?[^\]]*\]", "", text)
            w = 0
            for ch in stripped:
                cat = unicodedata.east_asian_width(ch)
                w += 2 if cat in ("W", "F") else 1
            return w

        def _config_line(
            label: str,
            value: str,
            click_action: str,
            *,
            extra_links: str = "",
        ) -> str:
            """Build a config line with value left-aligned and [...] right-aligned."""
            vis_label_w = _vis_width(label)
            vis_value_w = _vis_width(value)
            vis_extra_w = _vis_width(extra_links)
            # "[…]" → the ellipsis char is single-width = 3 visible cols ([…])
            link_vis = 3 + vis_extra_w
            pad = _W - vis_label_w - vis_value_w - link_vis
            if pad < 1:
                pad = 1
            link = f'[@click="screen.{click_action}"][…][/]'
            return f"{label}{value}{' ' * pad}{extra_links}{link}"

        # Tools
        tools = getattr(self, "_active_tools", None) or []
        lines.append(_config_line("  Tools:  ", str(len(tools)), "open_tools_picker"))

        # Guards
        guards = getattr(self, "_active_guards", None) or []
        guard_val = str(len(guards)) if guards else "[dim]0[/dim]"
        lines.append(_config_line("  Guards: ", guard_val, "open_guards_picker"))

        # Traits
        traits = getattr(self, "_active_traits", None) or []
        if traits:
            trait_val = ", ".join(traits)
        else:
            trait_val = "[dim]0[/dim]"
        lines.append(_config_line("  Traits: ", trait_val, "open_traits_picker"))

        # Knowledge graph
        if self._kg_enabled:
            try:
                kg = self._get_kg()
                count = kg.node_count()
                kg_val = f"ON ({count})"
            except Exception:
                kg_val = "ON"
        else:
            kg_val = "[dim]OFF[/dim]"
        kg_extra = '[@click="screen.toggle_kg_inline"]⏻[/] '
        lines.append(_config_line("  KG:     ", kg_val, "open_kg_screen", extra_links=kg_extra))

        return "\n".join(lines)

    def _update_sidebar(self) -> None:
        """Refresh the sidebar info."""
        try:
            info = self.query_one("#chat-sidebar-info", Static)
            info.update(self._build_sidebar())
        except Exception:
            pass

    def action_focus_input(self) -> None:
        """Refocus the chat input."""
        self.query_one("#chat-input", ChatInput).focus()


# ---------------------------------------------------------------------------
# Screen 4: Workspace (Agentic File Editing)
# ---------------------------------------------------------------------------


class WorkspaceScreen(Screen):
    """Agentic file editing with file tree, source viewer, agent panel, and chat."""

    BINDINGS = [
        Binding("ctrl+a", "accept_change", "Accept", show=True),
        Binding("ctrl+r", "reject_change", "Reject", show=True),
    ]

    _pending_changes: list[dict] = []
    _current_change_idx: int = 0

    def compose(self) -> ComposeResult:
        import os

        yield Header(show_clock=False)
        with Horizontal(id="workspace-layout"):
            # Left: File tree
            with Vertical(id="workspace-tree-panel"):
                yield Static("[b]Files[/b]", id="workspace-tree-title")
                yield DirectoryTree(os.getcwd(), id="workspace-tree")
                yield Static("", id="workspace-modified-list")
            # Center: Source / Diff viewer
            with Vertical(id="workspace-viewer-panel"):
                yield Static("[b]Source Viewer[/b]", id="workspace-viewer-title")
                with VerticalScroll(id="workspace-viewer-scroll"):
                    yield Static(
                        "[dim]Select a file from the tree to view its contents.[/dim]",
                        id="workspace-viewer-content",
                    )
                # Approval toolbar (hidden until changes pending)
                with Horizontal(id="workspace-approval-bar", classes="hidden"):
                    yield Button("Accept", id="workspace-accept-btn", variant="success")
                    yield Button("Reject", id="workspace-reject-btn", variant="error")
                    yield Button("Skip", id="workspace-skip-btn")
                    yield Static("", id="workspace-change-info")
            # Right: Agent panel
            with Vertical(id="workspace-agent-panel"):
                yield Static("[b]Agent[/b]", id="workspace-agent-title")
                yield Static("", id="workspace-agent-status")
                yield Static("[b]Plan[/b]", id="workspace-plan-title")
                with VerticalScroll(id="workspace-plan-scroll"):
                    yield Static(
                        "[dim]No active plan. Send a task below.[/dim]",
                        id="workspace-plan-content",
                    )
                yield Static("", id="workspace-agent-stats")
        # Bottom: Chat input
        yield Input(
            placeholder="Describe a task… (e.g., 'Refactor to use dataclasses')",
            id="workspace-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Set up the workspace with agent status."""
        import os
        from pathlib import Path

        from talk_box.workspace_tools import WorkspaceAgent

        self._pending_changes = []
        self._current_change_idx = 0

        # Load trusted commands from config if available
        trusted: list[str] = []
        try:
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            trusted = resolved.trusted_commands or []
        except Exception:
            pass

        self._agent = WorkspaceAgent(
            root=Path(os.getcwd()),
            trusted_commands=trusted
            or ["python", "uv", "pytest", "grep", "find", "cat", "ls", "wc"],
        )
        self._update_agent_status()

    def _update_agent_status(self) -> None:
        """Refresh agent status panel."""
        try:
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            model = resolved.model or "default"
            persona = resolved.persona or "none"
        except Exception:
            model = "default"
            persona = "none"

        status_lines = [
            f"  Model:   [b]{model}[/b]",
            f"  Persona: [b]{persona}[/b]",
            "  Status:  [dim]idle[/dim]",
        ]
        try:
            self.query_one("#workspace-agent-status", Static).update("\n".join(status_lines))
        except Exception:
            pass

    _BINARY_EXTENSIONS = frozenset(
        {
            ".pyc",
            ".pyo",
            ".so",
            ".dylib",
            ".dll",
            ".exe",
            ".bin",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".ico",
            ".webp",
            ".bmp",
            ".zip",
            ".gz",
            ".tar",
            ".bz2",
            ".xz",
            ".7z",
            ".jar",
            ".whl",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".ppt",
            ".pptx",
            ".db",
            ".sqlite",
            ".sqlite3",
            ".wasm",
            ".o",
            ".a",
            ".mp3",
            ".mp4",
            ".wav",
            ".avi",
            ".mov",
            ".mkv",
            ".ttf",
            ".otf",
            ".woff",
            ".woff2",
            ".eot",
        }
    )

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Show file contents when a file is selected in the tree."""
        path = event.path
        title = self.query_one("#workspace-viewer-title", Static)
        content = self.query_one("#workspace-viewer-content", Static)
        title.update(f"[b]{path.name}[/b]")

        if path.suffix.lower() in self._BINARY_EXTENSIONS:
            content.update(f"[dim]Binary file ({path.suffix})[/dim]")
            return

        try:
            text = path.read_text(errors="replace")
            # Check for binary content (null bytes)
            if "\x00" in text[:1024]:
                content.update("[dim]Binary file[/dim]")
                return
            # Truncate very large files
            lines = text.splitlines()
            if len(lines) > 500:
                display = "\n".join(lines[:500])
                display += f"\n\n… {len(lines) - 500} more lines …"
            else:
                display = text
            # Use plain Text to avoid Rich markup parsing of file contents
            from rich.text import Text

            content.update(Text(display))
        except Exception as e:
            from rich.text import Text

            content.update(Text(f"Cannot read file: {e}"))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle task submission — run workspace tool commands or show a plan."""
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""

        # Direct tool commands start with ">"
        if text.startswith(">"):
            self._run_tool_command(text[1:].strip())
            return

        # Natural language task — show plan and execute analysis step
        self._run_task(text)

    def _run_tool_command(self, cmd: str) -> None:
        """Execute a direct workspace tool command.

        Supported commands:
            >read PATH [START] [END]
            >write PATH CONTENT
            >edit PATH <<<OLD>>> <<<NEW>>>
            >search PATTERN [GLOB]
            >ls [PATH] [PATTERN]
            >exec COMMAND
        """
        plan = self.query_one("#workspace-plan-content", Static)
        parts = cmd.split(maxsplit=1)
        if not parts:
            plan.update("[red]Empty command[/red]")
            return

        action = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if action == "read":
            tokens = args.split()
            path = tokens[0] if tokens else ""
            start = int(tokens[1]) if len(tokens) > 1 else 1
            end = int(tokens[2]) if len(tokens) > 2 else None
            result = self._agent.file_read(path, start_line=start, end_line=end)

        elif action == "write":
            # >write path\ncontent
            lines = args.split("\n", 1)
            path = lines[0].strip()
            content = lines[1] if len(lines) > 1 else ""
            result = self._agent.file_write(path, content)

        elif action == "search":
            tokens = args.split(maxsplit=1)
            pattern = tokens[0] if tokens else ""
            glob = tokens[1] if len(tokens) > 1 else "**/*"
            result = self._agent.file_search(pattern, glob=glob)

        elif action in ("ls", "list"):
            tokens = args.split()
            path = tokens[0] if tokens else "."
            pattern = tokens[1] if len(tokens) > 1 else "*"
            result = self._agent.list_files(path, pattern=pattern)

        elif action in ("exec", "run", "shell"):
            result = self._agent.shell_exec(args)

        else:
            plan.update(
                f"[red]Unknown command: {action}[/red]\n\nAvailable: read, write, search, ls, exec"
            )
            return

        # Display result
        from rich.text import Text as RichText

        icon = "✅" if result.success else "❌"
        header = f"{icon} [b]>{action}[/b]"
        if result.path:
            header += f" {result.path}"
        plan.update(f"{header}\n\n")

        # Show output in the source viewer for read operations
        if action == "read" and result.success:
            viewer = self.query_one("#workspace-viewer-content", Static)
            title = self.query_one("#workspace-viewer-title", Static)
            title.update(f"[b]{result.path}[/b]")
            viewer.update(RichText(result.output))
            plan.update(f"{header}\n  {len(result.output.splitlines())} lines displayed")
        else:
            plan.update(f"{header}\n\n{result.output}")

        self._update_change_stats()

    def _run_task(self, text: str) -> None:
        """Execute a natural language task using workspace tools."""
        plan_content = self.query_one("#workspace-plan-content", Static)

        # Step 1: Analyze — list project files
        plan_content.update(
            f"[b]Task:[/b] {text}\n\n"
            "  1. 🔄 Analyze codebase\n"
            "  2. ⬜ Plan changes\n"
            "  3. ⬜ Generate edits\n"
            "  4. ⬜ Review & approve\n"
            "  5. ⬜ Run tests"
        )

        # Update status
        self._set_agent_status("analyzing")

        # Run analysis in a worker to avoid blocking UI
        self.run_worker(self._do_analyze(text), name="workspace-analyze")

    async def _do_analyze(self, task: str) -> None:
        """Worker: analyze the project and produce a plan."""
        import asyncio

        # Step 1: List project structure
        listing = await asyncio.to_thread(self._agent.list_files, ".", pattern="*")
        py_search = await asyncio.to_thread(
            self._agent.file_search, "def ", glob="**/*.py", max_results=20
        )

        # Build analysis summary
        file_list = listing.output if listing.success else "(could not list files)"
        code_hits = py_search.output if py_search.success else "(no Python files found)"

        plan_text = (
            f"[b]Task:[/b] {task}\n\n"
            "  1. ✅ Analyze codebase\n"
            "  2. ✅ Plan changes\n"
            "  3. ⬜ Generate edits — [dim]awaiting approval[/dim]\n"
            "  4. ⬜ Review & approve\n"
            "  5. ⬜ Run tests\n\n"
            "[b]Analysis:[/b]\n"
            f"  Files in root:\n"
        )
        # Show first 15 root entries
        for line in file_list.splitlines()[:15]:
            plan_text += f"    {line}\n"

        plan_text += (
            f"\n  Python definitions found: {len(code_hits.splitlines()) - 1}\n\n"
            "[b]Suggested approach:[/b]\n"
            f"  Use [b]>search[/b] to find relevant code, then [b]>edit[/b] to apply changes.\n"
            f"  Use [b]>exec pytest[/b] to verify.\n\n"
            "[dim]Use > commands to execute tools directly:[/dim]\n"
            "  [dim]>read PATH        — view a file[/dim]\n"
            "  [dim]>search PATTERN   — grep across project[/dim]\n"
            "  [dim]>ls [DIR]         — list directory[/dim]\n"
            "  [dim]>exec COMMAND     — run a shell command[/dim]\n"
            "  [dim]>edit PATH <<<OLD>>> <<<NEW>>> — edit a file[/dim]"
        )

        self.app.call_from_thread(self._update_plan, plan_text)
        self.app.call_from_thread(self._set_agent_status, "ready")
        self.app.call_from_thread(self._update_change_stats)

    def _update_plan(self, text: str) -> None:
        """Update the plan panel content (main thread)."""
        try:
            self.query_one("#workspace-plan-content", Static).update(text)
        except Exception:
            pass

    def _set_agent_status(self, status: str) -> None:
        """Update the agent status line."""
        colors = {
            "idle": "[dim]idle[/dim]",
            "analyzing": "[yellow]analyzing…[/yellow]",
            "ready": "[green]ready[/green]",
            "executing": "[yellow]executing…[/yellow]",
            "done": "[green]done[/green]",
            "error": "[red]error[/red]",
        }
        styled = colors.get(status, status)
        try:
            widget = self.query_one("#workspace-agent-status", Static)
            lines = str(widget._Static__content).splitlines()
            # Replace the Status line
            new_lines = []
            for line in lines:
                if "Status:" in line:
                    new_lines.append(f"  Status:  {styled}")
                else:
                    new_lines.append(line)
            widget.update("\n".join(new_lines))
        except Exception:
            pass

    def _update_change_stats(self) -> None:
        """Show change statistics in the agent stats area."""
        changes = self._agent.changes
        if not changes:
            return
        try:
            stats = self.query_one("#workspace-agent-stats", Static)
            writes = sum(1 for c in changes if c["action"] == "write")
            edits = sum(1 for c in changes if c["action"] == "edit")
            cmds = sum(1 for c in changes if c["action"] == "shell")
            parts = []
            if writes:
                parts.append(f"  Writes: {writes}")
            if edits:
                parts.append(f"  Edits:  {edits}")
            if cmds:
                parts.append(f"  Cmds:   {cmds}")
            stats.update("\n".join(parts))
        except Exception:
            pass

    def action_accept_change(self) -> None:
        """Accept the current proposed change."""
        if not self._pending_changes:
            return
        # Placeholder for accepting file changes
        self.notify("Change accepted", severity="information")

    def action_reject_change(self) -> None:
        """Reject the current proposed change."""
        if not self._pending_changes:
            return
        self.notify("Change rejected", severity="warning")


# ---------------------------------------------------------------------------
# Screen 5: Profile Editor
# ---------------------------------------------------------------------------


class ProfileScreen(Screen):
    """Browse named profiles (model + persona + guardrails)."""

    BINDINGS = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_profile: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="profile-layout"):
            with Vertical(id="profile-list-panel"):
                yield Static("[b]Profiles[/b]", id="profile-list-title")
                yield OptionList(id="profile-list")
            with Vertical(id="profile-detail-panel"):
                yield Static("[b]Profile Details[/b]", id="profile-detail-title")
                with VerticalScroll(id="profile-list-scroll"):
                    yield Static(
                        "[dim]Select a profile from the list.[/dim]",
                        id="profile-detail-content",
                    )
                yield Button("Use Profile", id="profile-use-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the profile list."""
        ol = self.query_one("#profile-list", OptionList)
        try:
            from talk_box.config import list_profiles

            names = list_profiles()
            if names:
                for name in names:
                    ol.add_option(Option(name, id=name))
            else:
                ol.add_option(
                    Option(
                        "[dim]No profiles — create in ~/.config/talk-box/profiles/[/dim]",
                        id="__empty",
                        disabled=True,
                    )
                )
        except Exception:
            pass

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Show detail for the highlighted profile."""
        if event.option.id is None or str(event.option.id).startswith("__"):
            return
        name = str(event.option.id)
        self._selected_profile = name
        self._show_profile_detail(name)

    def _show_profile_detail(self, name: str) -> None:
        """Display profile details in the detail panel."""
        try:
            from talk_box.config import load_profile

            profile = load_profile(name)
            lines = [
                f"[b]{profile.name}[/b]",
                "",
                f"  Model:       {profile.model or '(default)'}",
                f"  Persona:     {profile.persona or '(default)'}",
                f"  Temperature: {profile.temperature if profile.temperature is not None else '(default)'}",
            ]
            if profile.guardrails:
                lines.append(f"  Guardrails:  {', '.join(profile.guardrails)}")
            else:
                lines.append("  Guardrails:  (none)")
            self.query_one("#profile-detail-title", Static).update(f"[b]{profile.name}[/b]")
            self.query_one("#profile-detail-content", Static).update("\n".join(lines))
        except Exception:
            self.query_one("#profile-detail-content", Static).update(
                f"[dim]Could not load profile: {name}[/dim]"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Use Profile button — apply model/persona and go to Chat."""
        if event.button.id == "profile-use-btn" and self._selected_profile:
            try:
                from talk_box.config import load_profile, persist_defaults

                profile = load_profile(self._selected_profile)
                if profile.model:
                    persist_defaults(model=profile.model)
                if profile.persona:
                    persist_defaults(persona=profile.persona)

                self.app._switch_to("chat")  # type: ignore[attr-defined]
                try:
                    chat = self.app.screen
                    if profile.model:
                        chat._active_model = profile.model
                    if profile.persona:
                        chat._active_persona = profile.persona
                    chat._rebuild_bot()
                    chat._conversation = None
                    chat._update_sidebar()
                except Exception:
                    pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Screen 6: Persona Browser & Editor
# ---------------------------------------------------------------------------


class PersonaScreen(Screen):
    """Browse personas grouped by category with detail panel."""

    BINDINGS = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_persona: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="persona-layout"):
            with Vertical(id="persona-list-panel"):
                yield Static("[b]Personas[/b]", id="persona-list-title")
                yield OptionList(id="persona-list")
            with Vertical(id="persona-detail-panel"):
                yield Static("[b]Select a persona[/b]", id="persona-detail-title")
                with VerticalScroll(id="persona-detail-scroll"):
                    yield Static(
                        "[dim]Highlight a persona to see its details.[/dim]",
                        id="persona-detail-content",
                    )
                yield Button("Use in Chat", id="persona-use-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the persona option list."""
        ol = self.query_one("#persona-list", OptionList)
        try:
            from talk_box.personas._loader import get_persona, list_personas

            names = list_personas()
            categories: dict[str, list[str]] = {}
            for name in names:
                try:
                    p = get_persona(name)
                    cat = p.category or "other"
                except Exception:
                    cat = "other"
                categories.setdefault(cat, []).append(name)

            for cat in sorted(categories):
                ol.add_option(Option(f"── {cat} ──", id=f"__cat_{cat}", disabled=True))
                for name in sorted(categories[cat]):
                    ol.add_option(Option(name, id=name))
        except Exception:
            pass

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Show detail for the highlighted persona."""
        if event.option.id is None or str(event.option.id).startswith("__cat_"):
            return
        name = str(event.option.id)
        try:
            from talk_box.personas._loader import get_persona

            p = get_persona(name)
            lines = [
                f"[b]{p.display_name or p.name}[/b]",
                f"  Category: {p.category or '—'}",
                f"  {p.description}" if p.description else "",
                "",
                f"  Role:      {p.persona_role}" if p.persona_role else "",
                f"  Expertise: {p.expertise}" if p.expertise else "",
                f"  Temp:      {p.temperature}" if p.temperature is not None else "",
            ]
            if p.tags:
                lines.append(f"  Tags:      {', '.join(p.tags)}")
            if p.tools:
                lines.append(f"  Tools:     {', '.join(p.tools)}")
            if p.avoid_topics:
                lines.append(f"  Avoid:     {', '.join(p.avoid_topics)}")
            if p.recommended_models:
                lines.append(f"  Models:    {', '.join(str(m) for m in p.recommended_models)}")
            if p.default_guards:
                lines.append(f"  Guards:    {', '.join(str(g) for g in p.default_guards)}")
            if p.test_queries:
                lines.append("")
                lines.append("  [b]Test queries:[/b]")
                for q in p.test_queries[:5]:
                    lines.append(f"    • {q}")
            detail = "\n".join(line for line in lines if line is not None)
        except Exception:
            detail = f"[dim]{name}[/dim]"
        self._selected_persona = name
        self.query_one("#persona-detail-title", Static).update(f"[b]{name}[/b]")
        self.query_one("#persona-detail-content", Static).update(detail)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Use in Chat button."""
        if event.button.id == "persona-use-btn" and self._selected_persona:
            self.app._switch_to("chat")  # type: ignore[attr-defined]
            # Set persona on the chat screen
            try:
                chat = self.app.screen
                chat._active_persona = self._selected_persona
                chat._rebuild_bot()
                chat._conversation = None
                chat._update_sidebar()
                try:
                    chat.query_one("#chat-persona-select", Select).value = self._selected_persona
                except Exception:
                    pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Screen 7: Trait Browser & Composer
# ---------------------------------------------------------------------------


class TraitScreen(Screen):
    """Browse composable persona traits grouped by category."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="trait-layout"):
            with Vertical(id="trait-list-panel"):
                yield Static("[b]Traits[/b]", id="trait-list-title")
                yield OptionList(id="trait-list")
            with Vertical(id="trait-detail-panel"):
                yield Static("[b]Trait Details[/b]", id="trait-detail-title")
                with VerticalScroll(id="trait-detail-scroll"):
                    yield Static(
                        "[dim]Highlight a trait to see its details.[/dim]",
                        id="trait-detail-content",
                    )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the trait option list."""
        ol = self.query_one("#trait-list", OptionList)
        try:
            from talk_box.traits import trait_categories

            cats = trait_categories()
            for cat, names in cats.items():
                ol.add_option(Option(f"── {cat} ──", id=f"__cat_{cat}", disabled=True))
                for name in names:
                    ol.add_option(Option(name, id=name))
        except Exception:
            pass

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Show detail for the highlighted trait."""
        if event.option.id is None or str(event.option.id).startswith("__cat_"):
            return
        name = str(event.option.id)
        try:
            from talk_box.traits import get_trait

            t = get_trait(name)
            lines = [
                f"[b]{t.display_name or t.name}[/b]",
                f"  Category:    {t.category}",
                f"  {t.description}" if t.description else "",
            ]
            if t.constraints:
                lines.append("")
                lines.append("  [b]Constraints:[/b]")
                for c in t.constraints:
                    lines.append(f"    • {c}")
            if t.critical_constraints:
                lines.append("  [b]Critical:[/b]")
                for c in t.critical_constraints:
                    lines.append(f"    • {c}")
            if t.expertise_extra:
                lines.append(f"  Expertise:   {t.expertise_extra}")
            if t.tags:
                lines.append(f"  Tags:        {', '.join(t.tags)}")
            if t.temperature is not None:
                lines.append(f"  Temperature: {t.temperature}")
            detail = "\n".join(line for line in lines if line is not None)
        except Exception:
            detail = f"[dim]{name}[/dim]"
        self.query_one("#trait-detail-title", Static).update(f"[b]{name}[/b]")
        self.query_one("#trait-detail-content", Static).update(detail)


# ---------------------------------------------------------------------------
# Screen 8: Model Browser
# ---------------------------------------------------------------------------


class ModelScreen(Screen):
    """Browse model profiles with capability and cost comparison."""

    BINDINGS = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_model: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="model-layout"):
            yield Static("[b]Model Profiles[/b]", id="model-title")
            yield DataTable(id="model-table")
            with Vertical(id="model-detail-panel"):
                yield Static("", id="model-detail")
                yield Button("Use in Chat", id="model-use-btn", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the model table."""
        table = self.query_one("#model-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Provider", "Model", "Context", "Tools", "Vision", "Cost")

        # Check allow_cloud setting
        allow_cloud = True
        try:
            from talk_box.config import load_config

            allow_cloud = load_config().allow_cloud
        except Exception:
            pass

        _cloud = {
            "anthropic",
            "openai",
            "google",
            "mistral",
            "github",
            "azure",
            "bedrock",
            "together",
        }
        try:
            from talk_box.models import list_models

            for p in list_models():
                if not allow_cloud and p.provider.lower() in _cloud:
                    continue
                ctx = f"{p.context_window:,}" if p.context_window else "—"
                tools = "✓" if p.supports_tools else "✗"
                vision = "✓" if p.supports_vision else "✗"
                cost = p.cost_tier.value if p.cost_tier else "—"
                table.add_row(
                    p.provider,
                    p.model,
                    ctx,
                    tools,
                    vision,
                    cost,
                    key=p.key,
                )
        except Exception:
            pass

        # Ollama status row
        try:
            from talk_box.models import detect_ollama

            status = detect_ollama()
            if status.available and status.models:
                for name in status.models:
                    key = f"ollama:{name}"
                    # Skip if already in the table from built-in profiles
                    try:
                        table.get_row(key)
                    except Exception:
                        table.add_row(
                            "ollama",
                            name,
                            "—",
                            "—",
                            "—",
                            "free",
                            key=key,
                        )
        except Exception:
            pass

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show detail for the highlighted model."""
        if event.row_key is None:
            return
        key = str(event.row_key.value)
        self._selected_model = key
        try:
            from talk_box.models import get_model_profile

            profile = get_model_profile(key)
            if profile:
                lines = [
                    f"[b]{profile.display_name or profile.name}[/b]",
                    f"  Provider:  {profile.provider}",
                    f"  Context:   {profile.context_window:,} tokens"
                    if profile.context_window
                    else "",
                    f"  Output:    {profile.max_output_tokens:,} tokens"
                    if profile.max_output_tokens
                    else "",
                    f"  Tools:     {'Yes' if profile.supports_tools else 'No'}",
                    f"  Vision:    {'Yes' if profile.supports_vision else 'No'}",
                    f"  Streaming: {'Yes' if profile.supports_streaming else 'No'}",
                    f"  Cost:      {profile.cost_tier.value}" if profile.cost_tier else "",
                    f"  Cutoff:    {profile.knowledge_cutoff}" if profile.knowledge_cutoff else "",
                    f"  Notes:     {profile.notes}" if profile.notes else "",
                ]
                detail = "\n".join(line for line in lines if line)
            else:
                detail = f"[dim]No profile data for {key}[/dim]"
        except Exception:
            detail = f"[dim]{key}[/dim]"
        self.query_one("#model-detail", Static).update(detail)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle Use in Chat button."""
        if event.button.id == "model-use-btn" and self._selected_model:
            self.app._switch_to("chat")  # type: ignore[attr-defined]
            try:
                chat = self.app.screen
                chat._active_model = self._selected_model
                chat._rebuild_bot()
                chat._conversation = None
                chat._update_sidebar()
                try:
                    chat.query_one("#chat-model-select", Select).value = self._selected_model
                except Exception:
                    pass
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Screen 9: Guardrail Manager
# ---------------------------------------------------------------------------


_BUILTIN_GUARDS = [
    ("no_pii", "both", "Detects and redacts PII (emails, phones, SSNs, cards)"),
    ("max_response_length", "output", "Limits response length in characters"),
    ("max_input_length", "input", "Limits input length in characters"),
    ("tone_check", "output", "Validates response matches an expected tone"),
    ("disclaimer_required", "output", "Ensures responses include a disclaimer"),
    ("must_cite_sources", "output", "Requires citation patterns in responses"),
    ("keyword_block", "both", "Blocks messages containing specified keywords"),
]


class GuardrailScreen(Screen):
    """Browse built-in guardrails with active/inactive toggle."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="guard-layout"):
            yield Static("[b]Guardrails[/b]", id="guard-title")
            yield DataTable(id="guard-table")
            with Horizontal(id="guard-actions"):
                yield Button("Toggle Guard", id="guard-toggle-btn", variant="primary")
            with Vertical(id="guard-detail-panel"):
                yield Static("", id="guard-detail")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the guard table with active status."""
        self._active_guards = self._load_active_guards()
        self._rebuild_table()

    def _load_active_guards(self) -> set[str]:
        """Load active guardrails from config."""
        try:
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            return set(resolved.guardrails or [])
        except Exception:
            return set()

    def _rebuild_table(self) -> None:
        """Rebuild the guard table with current active status."""
        table = self.query_one("#guard-table", DataTable)
        table.clear(columns=True)
        table.cursor_type = "row"
        table.add_columns("Status", "Guard", "Phase", "Description")
        for name, phase, desc in _BUILTIN_GUARDS:
            status = "✅ Active" if name in self._active_guards else "⬜ Off"
            table.add_row(status, name, phase, desc, key=name)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show detail for the highlighted guard."""
        if event.row_key is None:
            return
        key = str(event.row_key.value)
        for name, phase, desc in _BUILTIN_GUARDS:
            if name == key:
                active = name in self._active_guards
                status = "[green]Active[/green]" if active else "[dim]Inactive[/dim]"
                lines = [
                    f"[b]{name}[/b]  {status}",
                    f"  Phase:       {phase}",
                    f"  Description: {desc}",
                    "",
                    "  Usage:",
                    f"    bot.guardrail(tb.{name}(...))",
                    "",
                    "  Press [b]Toggle Guard[/b] to enable/disable.",
                ]
                self.query_one("#guard-detail", Static).update("\n".join(lines))
                return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the toggle guard button."""
        if event.button.id != "guard-toggle-btn":
            return

        table = self.query_one("#guard-table", DataTable)
        row_key = table.cursor_row
        if row_key < 0 or row_key >= len(_BUILTIN_GUARDS):
            return

        name = _BUILTIN_GUARDS[row_key][0]

        if name in self._active_guards:
            self._active_guards.discard(name)
            self.notify(f"Disabled guard: {name}", severity="warning")
        else:
            self._active_guards.add(name)
            self.notify(f"Enabled guard: {name}", severity="information")

        # Persist to config
        self._save_active_guards()
        self._rebuild_table()

    def _save_active_guards(self) -> None:
        """Persist active guards to global config."""
        try:
            from talk_box.config import persist_guardrails

            persist_guardrails(list(self._active_guards))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Screen 10: Pathway Designer
# ---------------------------------------------------------------------------


class PathwayScreen(Screen):
    """Browse and inspect conversational pathways."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="pathway-layout"):
            yield Static(
                "[b]Pathways[/b]\n\n"
                "Pathways define structured conversation flows with states,\n"
                "branches, and fallbacks. Create pathways programmatically\n"
                "using the chainable builder API:\n\n"
                "  [dim]pathway = (\n"
                '      tb.Pathways(title="Support", ...)\n'
                '      .state("intake: gather info")\n'
                '      .required(["issue description"])\n'
                '      .next_state("triage")\n'
                '      .state("triage: route request")\n'
                '      .branch_on("Technical", id="tech")\n'
                '      .branch_on("Billing", id="billing")\n'
                "  )[/dim]\n\n"
                "State types: [b]chat[/b], [b]decision[/b], [b]collect[/b], "
                "[b]tool[/b], [b]summary[/b]",
                id="pathway-info",
            )
        yield Footer()


# ---------------------------------------------------------------------------
# Screen 11: Skill Browser
# ---------------------------------------------------------------------------


class SkillScreen(Screen):
    """Browse skill packs grouped by category."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="skill-layout"):
            with Vertical(id="skill-list-panel"):
                yield Static("[b]Skills[/b]", id="skill-list-title")
                yield OptionList(id="skill-list")
            with Vertical(id="skill-detail-panel"):
                yield Static("[b]Skill Details[/b]", id="skill-detail-title")
                with VerticalScroll(id="skill-detail-scroll"):
                    yield Static(
                        "[dim]Highlight a skill to see its details.[/dim]",
                        id="skill-detail-content",
                    )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the skill option list."""
        ol = self.query_one("#skill-list", OptionList)
        try:
            from talk_box.skills import skill_categories

            cats = skill_categories()
            for cat, names in cats.items():
                ol.add_option(Option(f"── {cat} ──", id=f"__cat_{cat}", disabled=True))
                for name in names:
                    ol.add_option(Option(name, id=name))
        except Exception:
            pass

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        """Show detail for the highlighted skill."""
        if event.option.id is None or str(event.option.id).startswith("__cat_"):
            return
        name = str(event.option.id)
        try:
            from talk_box.skills import get_skill

            s = get_skill(name)
            lines = [
                f"[b]{s.display_name or s.name}[/b]",
                f"  Category: {s.category}",
                f"  {s.description}" if s.description else "",
            ]
            if s.instructions:
                lines.append("")
                lines.append("  [b]Instructions:[/b]")
                # Show first 200 chars of instructions
                preview = s.instructions[:200]
                if len(s.instructions) > 200:
                    preview += "…"
                lines.append(f"    {preview}")
            if s.constraints:
                lines.append("")
                lines.append("  [b]Constraints:[/b]")
                for c in s.constraints:
                    lines.append(f"    • {c}")
            if s.tools:
                lines.append(f"  Tools: {', '.join(s.tools)}")
            if s.tags:
                lines.append(f"  Tags:  {', '.join(s.tags)}")
            detail = "\n".join(line for line in lines if line is not None)
        except Exception:
            detail = f"[dim]{name}[/dim]"
        self.query_one("#skill-detail-title", Static).update(f"[b]{name}[/b]")
        self.query_one("#skill-detail-content", Static).update(detail)


# ---------------------------------------------------------------------------
# Screen 12: Knowledge Explorer
# ---------------------------------------------------------------------------


class KnowledgeScreen(Screen):
    """Explore the knowledge graph and inspect nodes."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="kg-layout"):
            yield Static("[b]Knowledge Graph[/b]", id="kg-title")
            with Horizontal(id="kg-stats"):
                yield Static("", id="kg-stats-content")
            yield DataTable(id="kg-table")
            with VerticalScroll(id="kg-detail-scroll"):
                yield Static("", id="kg-detail")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the knowledge graph summary."""
        table = self.query_one("#kg-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Type", "Name")

        try:
            import os

            from talk_box.knowledge_graph import KnowledgeGraph, NodeType

            kg_path = os.path.join(os.path.expanduser("~/.config/talk-box"), "knowledge.db")
            self._kg = KnowledgeGraph(kg_path)
            docs = self._kg.node_count(node_type=NodeType.DOCUMENT)
            entities = self._kg.node_count(node_type=NodeType.ENTITY)
            topics = self._kg.node_count(node_type=NodeType.TOPIC)
            edges = self._kg.edge_count()

            stats = (
                f"  Documents: {docs}  |  Entities: {entities}  |  "
                f"Topics: {topics}  |  Edges: {edges}"
            )
            self.query_one("#kg-stats-content", Static).update(stats)

            for node in self._kg.list_nodes(limit=50):
                table.add_row(node.id, node.node_type.value, node.name, key=node.id)
        except Exception:
            self._kg = None
            self.query_one("#kg-stats-content", Static).update(
                "  [dim]No knowledge graph loaded[/dim]"
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show detail for the highlighted node."""
        if event.row_key is None or not hasattr(self, "_kg") or self._kg is None:
            return
        node_id = str(event.row_key.value)
        try:
            node = self._kg.get_node(node_id)
            if node is None:
                return
            lines = [
                f"[b]{node.name}[/b]",
                f"  ID:   {node.id}",
                f"  Type: {node.node_type.value}",
            ]
            if node.content:
                preview = node.content[:300]
                if len(node.content) > 300:
                    preview += "…"
                lines.append("")
                lines.append("  [b]Content:[/b]")
                lines.append(f"  {preview}")
            if node.metadata:
                lines.append("")
                lines.append("  [b]Metadata:[/b]")
                for k, v in list(node.metadata.items())[:10]:
                    lines.append(f"    {k}: {v}")
            # Show connected edges
            edges = self._kg.get_edges(node_id)
            if edges:
                lines.append("")
                lines.append(f"  [b]Connections:[/b] ({len(edges)})")
                for edge in edges[:8]:
                    other = edge.target if edge.source == node_id else edge.source
                    lines.append(f"    → {edge.relation}: {other}")
            self.query_one("#kg-detail", Static).update("\n".join(lines))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Screen 13: Memory Inspector
# ---------------------------------------------------------------------------


class MemoryScreen(Screen):
    """Inspect memory tiers and their contents."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="memory-layout"):
            yield Static("[b]Memory Inspector[/b]", id="memory-title")
            with VerticalScroll(id="memory-content"):
                yield Static("", id="memory-tiers")
        yield Footer()

    def on_mount(self) -> None:
        """Show memory tier overview with live data."""
        lines: list[str] = []

        # Try to load the default long-term memory store
        try:
            import os

            from talk_box.memory import LongTermMemory

            db_path = os.path.expanduser("~/.config/talk-box/memory.db")

            lines.append("[b]Working Memory[/b]")
            lines.append("  In-conversation key-value store. Ephemeral.")
            lines.append("")

            lines.append("[b]Short-Term Memory[/b]")
            lines.append("  Session memory with TTL and eviction.")
            lines.append("")

            lines.append("[b]Long-Term Memory[/b]")
            if os.path.isfile(db_path):
                ltm = LongTermMemory(path=db_path)
                entries = ltm.entries()
                lines.append(f"  Database: {db_path}")
                lines.append(f"  Entries:  {len(entries)}")
                if entries:
                    lines.append("")
                    lines.append("  [b]Recent entries:[/b]")
                    for entry in entries[:15]:
                        preview = str(entry.value)[:60]
                        if len(str(entry.value)) > 60:
                            preview += "…"
                        tags_str = f" [{', '.join(entry.tags)}]" if entry.tags else ""
                        lines.append(f"    {entry.key}{tags_str}: {preview}")
                ltm.close()
            else:
                lines.append(f"  Database: [dim]not found[/dim] ({db_path})")
                lines.append("  No persistent memories stored yet.")

            lines.append("")
            lines.append("[b]Usage[/b]")
            lines.append(
                "  [dim]store = tb.MemoryStore(long_term_path='~/.config/talk-box/memory.db')[/dim]"
            )
            lines.append(
                "  [dim]store.remember('key', 'value', tier=tb.MemoryTier.LONG_TERM)[/dim]"
            )
            lines.append("  [dim]store.recall('key')[/dim]")
        except Exception:
            lines = [
                "[b]Working Memory[/b]",
                "  In-conversation key-value store. Lost when the conversation ends.",
                "  API: WorkingMemory().set(key, value) / .get(key)",
                "",
                "[b]Short-Term Memory[/b]",
                "  Session memory with TTL and max-entry eviction.",
                "  API: ShortTermMemory(max_entries=100, default_ttl=3600)",
                "",
                "[b]Long-Term Memory[/b]",
                "  Persistent SQLite-backed memory.",
                "  API: LongTermMemory(path) / MemoryStore(path)",
            ]

        self.query_one("#memory-tiers", Static).update("\n".join(lines))


# ---------------------------------------------------------------------------
# Screen 14: Eval Dashboard
# ---------------------------------------------------------------------------


class EvalScreen(Screen):
    """Browse evaluation dimensions, scoring info, and scorecard history."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="eval-layout"):
            yield Static("[b]Eval Dashboard[/b]", id="eval-title")
            yield DataTable(id="eval-dim-table")
            with Vertical(id="eval-info-panel"):
                yield Static("", id="eval-info")
            yield Static("[b]Scorecard History[/b]", id="eval-history-title")
            yield DataTable(id="eval-history-table")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the eval dimensions table and scorecard history."""
        table = self.query_one("#eval-dim-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Dimension", "Description")

        dims = [
            ("relevance", "Is the response on-topic and directly helpful?"),
            ("safety", "Is the response free from harmful content?"),
            ("instruction_adherence", "Does the response follow persona constraints?"),
            ("tone", "Does the response match the expected communication style?"),
            ("completeness", "Does the response thoroughly address the query?"),
            ("conciseness", "Is the response appropriately concise?"),
        ]
        for name, desc in dims:
            table.add_row(name, desc, key=name)

        self.query_one("#eval-info", Static).update(
            "  Run evals with: [b]tb.eval_suite(bot, cases)[/b]\n"
            "  Compare variants: [b]tb.eval(bots, cases)[/b]\n"
            "  Export results: [b]results.to_scorecard(path)[/b]"
        )

        self._load_scorecard_history()

    def _load_scorecard_history(self) -> None:
        """Scan for scorecard JSON files and populate the history table."""
        import json
        import os
        from pathlib import Path

        history_table = self.query_one("#eval-history-table", DataTable)
        history_table.cursor_type = "row"
        history_table.add_columns("Persona", "Model", "Overall", "Queries", "Date")

        scorecards_dir = Path(os.getcwd()) / "scorecards"
        if not scorecards_dir.is_dir():
            history_table.add_row("—", "—", "—", "—", "[dim]No scorecards found[/dim]")
            return

        entries: list[tuple[str, str, str, str, str, str]] = []
        for json_file in sorted(scorecards_dir.rglob("*.json"), reverse=True):
            if json_file.parent.name == "_sweeps":
                continue
            try:
                data = json.loads(json_file.read_text())
                config = data.get("config", {})
                persona = config.get("persona", "—")
                variants = data.get("variants", {})
                generated = data.get("generated_at", "—")
                # Show date portion only
                date_str = generated[:19].replace("T", " ") if "T" in generated else generated

                for model_name, scores in variants.items():
                    overall = scores.get("overall", 0)
                    n_queries = scores.get("num_queries", config.get("num_queries", "?"))
                    overall_str = (
                        f"{overall:.1%}" if isinstance(overall, (int, float)) else str(overall)
                    )
                    entries.append(
                        (persona, model_name, overall_str, str(n_queries), date_str, str(json_file))
                    )
            except Exception:
                continue

        if not entries:
            history_table.add_row("—", "—", "—", "—", "[dim]No scorecards found[/dim]")
            return

        for persona, model, overall, n_queries, date_str, path in entries[:20]:
            row_key = f"sc-{len(history_table.rows)}"
            history_table.add_row(persona, model, overall, n_queries, date_str, key=row_key)
