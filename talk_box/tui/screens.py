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
from textual.widgets import Button, Footer, Header, Input, Static
from textual.worker import Worker, WorkerState

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
    """

    BINDINGS = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bot = None
        self._conversation = None
        self._message_count = 0

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
                yield Input(
                    placeholder="Type a message… (Enter to send)",
                    id="chat-input",
                )

            # Sidebar
            with Vertical(id="chat-sidebar"):
                yield Static("[b]Session[/b]", id="chat-sidebar-title")
                yield Static(
                    self._build_sidebar(),
                    id="chat-sidebar-info",
                )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the ChatBot and focus the input."""
        self._init_bot()
        self.query_one("#chat-input", Input).focus()

    def _init_bot(self) -> None:
        """Create a ChatBot from the resolved config."""
        try:
            from talk_box.builder import ChatBot
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            bot = ChatBot()
            if resolved.model:
                provider_model = resolved.model
                if ":" in provider_model:
                    provider, model = provider_model.split(":", 1)
                    bot = bot.provider(provider).model(model)
                else:
                    bot = bot.model(provider_model)
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

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle message submission."""
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        self._append_message("user", text)
        self._send_message(text)

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
        else:
            label = "[b]Assistant[/b]"
            classes = "chat-message chat-assistant"

        widget = Static(f"{label}\n{content}", id=msg_id, classes=classes)
        container.mount(widget)
        container.scroll_end(animate=False)
        self._update_sidebar()

    def _send_message(self, text: str) -> None:
        """Send a message via ChatBot in a background worker."""
        self._append_message("assistant", "[dim]Thinking…[/dim]")
        thinking_id = f"chat-msg-{self._message_count}"
        self._pending_thinking_id = thinking_id

        async def _do_chat():
            import asyncio

            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._get_response, text)

        self.run_worker(
            _do_chat(),
            name="chat_response",
            group="chat",
        )

    def _get_response(self, text: str) -> str:
        """Get a response from the bot (runs in worker thread)."""
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
        """Handle worker completion — replace thinking indicator with response."""
        if event.worker.name != "chat_response":
            return
        if event.state == WorkerState.SUCCESS:
            result = event.worker.result
            thinking_id = getattr(self, "_pending_thinking_id", None)
            if thinking_id:
                try:
                    widget = self.query_one(f"#{thinking_id}", Static)
                    widget.update(f"[b]Assistant[/b]\n{result}")
                    widget.remove_class("chat-thinking")
                except Exception:
                    self._append_message("assistant", str(result))
            container = self.query_one("#chat-messages", VerticalScroll)
            container.scroll_end(animate=False)
            self._update_sidebar()

    def _build_sidebar(self) -> str:
        """Build sidebar info text."""
        try:
            from talk_box.config import load_config

            config = load_config()
            resolved = config.resolve()
            model = resolved.model or "[dim]echo mode[/dim]"
            persona = resolved.persona or "[dim]none[/dim]"
        except Exception:
            model = "[dim]echo mode[/dim]"
            persona = "[dim]none[/dim]"

        user_msgs = self._message_count // 2 if self._message_count > 0 else 0
        return (
            f"\n  Model:\n  {model}\n\n  Persona:\n  {persona}\n\n  Messages:\n  {user_msgs} sent"
        )

    def _update_sidebar(self) -> None:
        """Refresh the sidebar info."""
        try:
            info = self.query_one("#chat-sidebar-info", Static)
            info.update(self._build_sidebar())
        except Exception:
            pass

    def action_focus_input(self) -> None:
        """Refocus the chat input."""
        self.query_one("#chat-input", Input).focus()


# ---------------------------------------------------------------------------
# Screen 4: Workspace (Agentic File Editing)
# ---------------------------------------------------------------------------


class WorkspaceScreen(Screen):
    """Agentic file editing with diff view and approval toolbar."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Workspace",
            "File tree, diff view, agent task panel, and approval toolbar.\n\n"
            "[dim]Agentic workspace coming later in Phase 7b.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 5: Profile Editor
# ---------------------------------------------------------------------------


class ProfileScreen(Screen):
    """Build and manage named profiles (model + persona + guardrails)."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Profiles",
            "Create, edit, and switch named profiles.\n\n[dim]Profile editor coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 6: Persona Browser & Editor
# ---------------------------------------------------------------------------


class PersonaScreen(Screen):
    """Browse, search, and edit personas."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Personas",
            "Browse all personas by category. Preview, edit, and create new ones.\n\n"
            "[dim]Persona browser coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 7: Trait Browser & Composer
# ---------------------------------------------------------------------------


class TraitScreen(Screen):
    """Browse and compose persona traits."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Traits",
            "Browse traits and compose them onto personas.\n\n"
            "[dim]Trait composer coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 8: Model Browser
# ---------------------------------------------------------------------------


class ModelScreen(Screen):
    """Browse model profiles with capability and cost comparison."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Models",
            "Browse model profiles, compare capabilities, and check Ollama status.\n\n"
            "[dim]Model browser coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 9: Guardrail Manager
# ---------------------------------------------------------------------------


class GuardrailScreen(Screen):
    """Configure and test guardrails."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Guardrails",
            "Configure guard pipelines and test them with sample inputs.\n\n"
            "[dim]Guardrail manager coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 10: Pathway Designer
# ---------------------------------------------------------------------------


class PathwayScreen(Screen):
    """Visual pathway state-machine designer."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Pathways",
            "Design conversation pathways with states, branches, and fallbacks.\n\n"
            "[dim]Pathway designer coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 11: Skill Browser
# ---------------------------------------------------------------------------


class SkillScreen(Screen):
    """Browse and manage skills."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Skills",
            "Browse skill packs, install new skills, and manage registrations.\n\n"
            "[dim]Skill browser coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 12: Knowledge Explorer
# ---------------------------------------------------------------------------


class KnowledgeScreen(Screen):
    """Explore the knowledge graph, run syncs, and inspect nodes."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Knowledge",
            "Knowledge graph explorer with sync, search, and visualization.\n\n"
            "[dim]Knowledge explorer coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 13: Memory Inspector
# ---------------------------------------------------------------------------


class MemoryScreen(Screen):
    """Inspect memory tiers (working, short-term, long-term)."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Memory",
            "Inspect working, short-term, and long-term memory contents.\n\n"
            "[dim]Memory inspector coming soon.[/dim]",
        )


# ---------------------------------------------------------------------------
# Screen 14: Eval Dashboard
# ---------------------------------------------------------------------------


class EvalScreen(Screen):
    """Run evals and view results across models and personas."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield from _placeholder(
            "Eval",
            "Run eval suites, view scorecards, and track regressions.\n\n"
            "[dim]Eval dashboard coming soon.[/dim]",
        )
