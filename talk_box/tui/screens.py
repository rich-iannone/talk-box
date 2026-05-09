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
from textual.widgets import Button, DataTable, Footer, Header, Input, Static
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
    """Browse named profiles (model + persona + guardrails)."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="profile-layout"):
            with Vertical(id="profile-list-panel"):
                yield Static("[b]Profiles[/b]", id="profile-list-title")
                with VerticalScroll(id="profile-list-scroll"):
                    yield Static("", id="profile-list-content")
            with Vertical(id="profile-detail-panel"):
                yield Static("[b]Profile Details[/b]", id="profile-detail-title")
                yield Static(
                    "[dim]Select a profile from the list.[/dim]",
                    id="profile-detail-content",
                )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the profile list."""
        try:
            from talk_box.config import list_profiles, load_config

            config = load_config()
            active = config.default_profile
            names = list_profiles()

            lines: list[str] = []
            if active:
                lines.append(f"  Active: [b]{active}[/b]")
                lines.append("")

            if names:
                for name in names:
                    marker = "▸" if name == active else " "
                    lines.append(f"{marker} {name}")
            else:
                lines.append("[dim]No profiles found.[/dim]")
                lines.append("")
                lines.append("[dim]Create profiles in[/dim]")
                lines.append("[dim]~/.config/talk-box/profiles/[/dim]")

            self.query_one("#profile-list-content", Static).update("\n".join(lines))

            # Show first / active profile detail
            detail_name = active or (names[0] if names else None)
            if detail_name:
                self._show_profile_detail(detail_name)
        except Exception:
            self.query_one("#profile-list-content", Static).update(
                "[dim]Could not load profiles[/dim]"
            )

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


# ---------------------------------------------------------------------------
# Screen 6: Persona Browser & Editor
# ---------------------------------------------------------------------------


class PersonaScreen(Screen):
    """Browse personas grouped by category with detail panel."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="persona-layout"):
            with Vertical(id="persona-list-panel"):
                yield Static("[b]Personas[/b]", id="persona-list-title")
                with VerticalScroll(id="persona-list-scroll"):
                    yield Static(
                        self._build_persona_list(),
                        id="persona-list-content",
                    )
            with Vertical(id="persona-detail-panel"):
                yield Static("[b]Select a persona[/b]", id="persona-detail-title")
                yield Static(
                    "[dim]Use the Persona Browser to explore available personas.\n\n"
                    "Persona selection and detail view coming soon.[/dim]",
                    id="persona-detail-content",
                )
        yield Footer()

    def _build_persona_list(self) -> str:
        """Build the categorized persona list."""
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

            lines: list[str] = []
            for cat in sorted(categories):
                lines.append(f"\n[b]{cat}[/b] ({len(categories[cat])})")
                for name in sorted(categories[cat]):
                    lines.append(f"  {name}")
            return "\n".join(lines) if lines else "[dim]No personas found[/dim]"
        except Exception:
            return "[dim]Could not load personas[/dim]"


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
                with VerticalScroll(id="trait-list-scroll"):
                    yield Static("", id="trait-list-content")
            with Vertical(id="trait-detail-panel"):
                yield Static("[b]Trait Details[/b]", id="trait-detail-title")
                yield Static(
                    "[dim]Select a trait to see its details.[/dim]",
                    id="trait-detail-content",
                )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the trait list grouped by category."""
        try:
            from talk_box.traits import trait_categories

            cats = trait_categories()
            lines: list[str] = []
            for cat, names in cats.items():
                lines.append(f"\n[b]{cat}[/b] ({len(names)})")
                for name in names:
                    lines.append(f"  {name}")
            text = "\n".join(lines) if lines else "[dim]No traits found[/dim]"
        except Exception:
            text = "[dim]Could not load traits[/dim]"
        self.query_one("#trait-list-content", Static).update(text)


# ---------------------------------------------------------------------------
# Screen 8: Model Browser
# ---------------------------------------------------------------------------


class ModelScreen(Screen):
    """Browse model profiles with capability and cost comparison."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="model-layout"):
            yield Static("[b]Model Profiles[/b]", id="model-title")
            yield DataTable(id="model-table")
            with Vertical(id="model-detail-panel"):
                yield Static("", id="model-detail")
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
    """Browse built-in guardrails and active guard pipeline."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="guard-layout"):
            yield Static("[b]Guardrails[/b]", id="guard-title")
            yield DataTable(id="guard-table")
            with Vertical(id="guard-detail-panel"):
                yield Static("", id="guard-detail")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the guard table."""
        table = self.query_one("#guard-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("Guard", "Phase", "Description")
        for name, phase, desc in _BUILTIN_GUARDS:
            table.add_row(name, phase, desc, key=name)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show detail for the highlighted guard."""
        if event.row_key is None:
            return
        key = str(event.row_key.value)
        for name, phase, desc in _BUILTIN_GUARDS:
            if name == key:
                lines = [
                    f"[b]{name}[/b]",
                    f"  Phase:       {phase}",
                    f"  Description: {desc}",
                    "",
                    "  Usage:",
                    f"    bot.guardrail(tb.{name}(...))",
                ]
                self.query_one("#guard-detail", Static).update("\n".join(lines))
                return


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
                with VerticalScroll(id="skill-list-scroll"):
                    yield Static("", id="skill-list-content")
            with Vertical(id="skill-detail-panel"):
                yield Static("[b]Skill Details[/b]", id="skill-detail-title")
                yield Static(
                    "[dim]Select a skill to see its details.[/dim]",
                    id="skill-detail-content",
                )
        yield Footer()

    def on_mount(self) -> None:
        """Populate the skill list grouped by category."""
        try:
            from talk_box.skills import skill_categories

            cats = skill_categories()
            lines: list[str] = []
            for cat, names in cats.items():
                lines.append(f"\n[b]{cat}[/b] ({len(names)})")
                for name in names:
                    lines.append(f"  {name}")
            text = "\n".join(lines) if lines else "[dim]No skills found[/dim]"
        except Exception:
            text = "[dim]Could not load skills[/dim]"
        self.query_one("#skill-list-content", Static).update(text)


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
        yield Footer()

    def on_mount(self) -> None:
        """Populate the knowledge graph summary."""
        table = self.query_one("#kg-table", DataTable)
        table.cursor_type = "row"
        table.add_columns("ID", "Type", "Name")

        try:
            from talk_box.knowledge_graph import KnowledgeGraph, NodeType

            kg = KnowledgeGraph()
            docs = kg.node_count(node_type=NodeType.DOCUMENT)
            entities = kg.node_count(node_type=NodeType.ENTITY)
            topics = kg.node_count(node_type=NodeType.TOPIC)
            edges = kg.edge_count()

            stats = (
                f"  Documents: {docs}  |  Entities: {entities}  |  "
                f"Topics: {topics}  |  Edges: {edges}"
            )
            self.query_one("#kg-stats-content", Static).update(stats)

            for node in kg.list_nodes(limit=50):
                table.add_row(node.id, node.node_type.value, node.name, key=node.id)
        except Exception:
            self.query_one("#kg-stats-content", Static).update(
                "  [dim]No knowledge graph loaded[/dim]"
            )


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
        """Show memory tier overview."""
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
            "",
            "[b]Retention Policies[/b]",
            "  Control memory lifecycle with RetentionPolicy.",
            "  Personas can specify retention rules for automatic cleanup.",
        ]
        self.query_one("#memory-tiers", Static).update("\n".join(lines))


# ---------------------------------------------------------------------------
# Screen 14: Eval Dashboard
# ---------------------------------------------------------------------------


class EvalScreen(Screen):
    """Browse evaluation dimensions and scoring info."""

    BINDINGS = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="eval-layout"):
            yield Static("[b]Eval Dashboard[/b]", id="eval-title")
            yield DataTable(id="eval-dim-table")
            with Vertical(id="eval-info-panel"):
                yield Static("", id="eval-info")
        yield Footer()

    def on_mount(self) -> None:
        """Populate the eval dimensions table."""
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
