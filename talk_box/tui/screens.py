"""TUI screen definitions.

Each screen is a Textual Screen subclass. Screens start as minimal placeholders
and are fleshed out in subsequent phases. The app shell imports them all from
this module so that screen installation and command-mode routing work.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
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
        yield Footer()

    def _get_model_options(self) -> list[tuple[str, str]]:
        """Build model select options, with favorites first."""
        options: list[tuple[str, str]] = []
        fav_models: list[str] = []
        try:
            from talk_box.config import get_favorites

            fav_models, _ = get_favorites()
        except Exception:
            pass

        # Add favorites at the top with a star prefix
        seen: set[str] = set()
        for fav in fav_models:
            options.append((f"⭐ {fav}", fav))
            seen.add(fav)

        try:
            from talk_box.models import list_models

            for p in list_models():
                label = f"{p.provider}:{p.model}"
                if label not in seen:
                    options.append((label, label))
                    seen.add(label)
        except Exception:
            pass
        return options

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
            self._bot = bot
        except Exception:
            self._bot = None

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model or persona selection changes."""
        if event.select.id == "chat-model-select":
            self._active_model = str(event.value) if event.value != Select.BLANK else None
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
                "  /tokens            Show token usage estimate\n"
                "  /history           Show conversation history\n"
                "  /save [name]       Save session to disk\n"
                "  /load [name]       Load a saved session\n"
                "  /export [format]   Export session (json, markdown)\n"
                "  /attach <path>     Attach a file to the next message\n"
                "  /format <type>     Set output format (json, markdown, table)\n"
                "  /fav model <name>  Toggle model as favorite\n"
                "  /fav persona <n>   Toggle persona as favorite\n"
                "  /fav               List current favorites\n"
                "  /guards            Show guardrail status\n"
                "  /memory            Show memory tier summary\n"
                "  /kg                Show knowledge graph stats\n"
                "  /cost              Show estimated session cost\n"
                "  /quit              Quit the app"
            )
            self._append_system_message(help_text)

        elif cmd == "/clear":
            self._do_clear()

        elif cmd == "/model":
            if arg:
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
            if self._conversation and self._conversation.messages:
                total_chars = sum(len(m.content) for m in self._conversation.messages)
                est_tokens = total_chars // 4  # rough estimate
                self._append_system_message(
                    f"[b]Token Estimate[/b]\n"
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
            self._show_guards()

        elif cmd == "/memory":
            self._show_memory_summary()

        elif cmd == "/kg":
            self._show_kg_summary()

        elif cmd == "/cost":
            self._show_cost_estimate()

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
        lines = ["[b]Guardrails[/b]"]
        try:
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            active = resolved.guardrails or []

            if active:
                for g in active:
                    lines.append(f"  ✅ {g}")
            else:
                lines.append("  [dim]No guardrails active[/dim]")

            lines.append("")
            lines.append("  Available guards:")
            for name, _phase, _desc in _BUILTIN_GUARDS:
                marker = "●" if name in active else "○"
                lines.append(f"    {marker} {name}")
        except Exception:
            lines.append("  [dim]Could not load guardrail info[/dim]")
        self._append_system_message("\n".join(lines))

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

    def _show_kg_summary(self) -> None:
        """Show knowledge graph summary in chat."""
        lines = ["[b]Knowledge Graph[/b]"]
        try:
            from talk_box.knowledge_graph import KnowledgeGraph, NodeType

            kg = KnowledgeGraph()
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
                lines.append("  [dim]No data yet. Sync with:[/dim]")
                lines.append("  [dim]tb.MarkdownDir('./docs/').sync(kg)[/dim]")
        except Exception:
            lines.append("  [dim]Could not load knowledge graph[/dim]")
        self._append_system_message("\n".join(lines))

    def _show_cost_estimate(self) -> None:
        """Show estimated session cost based on token usage and model pricing."""
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
        """Build sidebar info text."""
        model = self._active_model or "[dim]echo mode[/dim]"
        persona = self._active_persona or "[dim]none[/dim]"
        user_msgs = self._message_count // 2 if self._message_count > 0 else 0

        lines = [
            f"  Model:    {model}",
            f"  Persona:  {persona}",
            "",
            f"  Messages: {user_msgs}",
        ]

        # Token estimate from conversation
        if self._conversation and self._conversation.messages:
            total_chars = sum(len(m.content) for m in self._conversation.messages)
            est_tokens = total_chars // 4
            lines.append(f"  Tokens:   ~{est_tokens:,}")
        else:
            lines.append("  Tokens:   0")

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
                    lines.append(f"  Context:  {pct:.0f}% of {profile.context_window:,}")
            except Exception:
                pass

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
        try:
            from talk_box.models import list_models

            for p in list_models():
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
            from talk_box.knowledge_graph import KnowledgeGraph, NodeType

            self._kg = KnowledgeGraph()
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
