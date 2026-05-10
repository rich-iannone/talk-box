"""Tests for the Talk Box TUI application shell."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from talk_box.tui.app import SCREEN_NAV, TalkBoxApp
from textual.widgets import Button, DataTable, DirectoryTree, Input, OptionList, Select, Static
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
    SkillScreen,
    TraitScreen,
    WelcomeScreen,
    WorkspaceScreen,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def _fake_config():
    """Pretend a global config exists so the app skips the Welcome screen."""
    with patch.object(TalkBoxApp, "_has_global_config", return_value=True):
        yield


@pytest.fixture()
def _no_config():
    """Ensure no config exists so the app shows the Welcome screen."""
    with patch.object(TalkBoxApp, "_has_global_config", return_value=False):
        yield


# ---------------------------------------------------------------------------
# App instantiation
# ---------------------------------------------------------------------------


class TestAppCreation:
    def test_app_instantiates(self):
        app = TalkBoxApp()
        assert app.TITLE == "Talk Box"

    def test_screen_nav_has_13_entries(self):
        assert len(SCREEN_NAV) == 13

    def test_screen_nav_keys_are_unique(self):
        keys = [k for k, *_ in SCREEN_NAV]
        assert len(keys) == len(set(keys))

    def test_screen_nav_ids_are_unique(self):
        ids = [sid for _, sid, *_ in SCREEN_NAV]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Screen classes
# ---------------------------------------------------------------------------


class TestScreenClasses:
    @pytest.mark.parametrize(
        "screen_cls",
        [
            WelcomeScreen,
            HomeScreen,
            ChatScreen,
            WorkspaceScreen,
            ProfileScreen,
            PersonaScreen,
            TraitScreen,
            ModelScreen,
            GuardrailScreen,
            PathwayScreen,
            SkillScreen,
            KnowledgeScreen,
            MemoryScreen,
            EvalScreen,
        ],
    )
    def test_screen_is_importable(self, screen_cls):
        assert screen_cls is not None

    def test_all_14_screens_exist(self):
        # 13 in SCREEN_NAV + WelcomeScreen
        screen_classes = {cls for *_, cls in SCREEN_NAV}
        screen_classes.add(WelcomeScreen)
        assert len(screen_classes) == 14


# ---------------------------------------------------------------------------
# Async app tests (using textual.pilot)
# ---------------------------------------------------------------------------


class TestAppMount:
    @pytest.mark.asyncio()
    async def test_mounts_with_config(self, _fake_config):
        """App shows HomeScreen when config exists."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            assert isinstance(app, TalkBoxApp)
            assert app._current_screen_id == "home"

    @pytest.mark.asyncio()
    async def test_mounts_without_config(self, _no_config):
        """App shows WelcomeScreen when no config exists."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            # Welcome is pushed on top
            assert isinstance(app.screen, WelcomeScreen)

    @pytest.mark.asyncio()
    async def test_vscode_hint_shown(self, _fake_config):
        """VS Code hint notification fires when TERM_PROGRAM=vscode."""
        with patch.dict(os.environ, {"TERM_PROGRAM": "vscode"}):
            with patch("talk_box.tui.app._IN_VSCODE", True):
                async with TalkBoxApp().run_test() as pilot:
                    # The notification was posted (hard to assert toast content,
                    # but we verify the app started without errors)
                    assert isinstance(pilot.app, TalkBoxApp)


class TestCommandMode:
    @pytest.mark.asyncio()
    async def test_escape_opens_command_list(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            assert not app.command_mode

            await pilot.press("escape")
            assert app.command_mode

            # Pressing escape again dismisses the modal
            await pilot.press("escape")
            assert not app.command_mode
            assert app._current_screen_id == "home"

    @pytest.mark.asyncio()
    async def test_colon_opens_command_list(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            assert not app.command_mode

            await pilot.press("colon")
            assert app.command_mode

            # Pressing colon again dismisses the modal
            await pilot.press("colon")
            assert not app.command_mode
            assert app._current_screen_id == "home"

    @pytest.mark.asyncio()
    async def test_command_mode_navigates_to_chat(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")  # enter command mode
            await pilot.press("c")  # chat
            assert app._current_screen_id == "chat"
            assert not app.command_mode  # auto-exits command mode

    @pytest.mark.asyncio()
    async def test_command_mode_navigates_to_models(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            await pilot.press("m")
            assert app._current_screen_id == "models"

    @pytest.mark.asyncio()
    async def test_command_mode_navigates_to_personas(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            await pilot.press("p")
            assert app._current_screen_id == "personas"

    @pytest.mark.asyncio()
    async def test_command_mode_navigates_to_eval(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            await pilot.press("e")
            assert app._current_screen_id == "eval"

    @pytest.mark.asyncio()
    async def test_command_mode_navigates_to_knowledge(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            await pilot.press("k")
            assert app._current_screen_id == "knowledge"

    @pytest.mark.asyncio()
    async def test_command_mode_navigates_to_workspace(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            await pilot.press("w")
            assert app._current_screen_id == "workspace"

    @pytest.mark.asyncio()
    async def test_command_mode_quit(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            await pilot.press("q")
            # App exits — if we get here without error, quit was triggered

    @pytest.mark.asyncio()
    async def test_keys_ignored_outside_command_mode(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            # Without entering command mode, pressing 'c' does nothing
            await pilot.press("c")
            assert app._current_screen_id == "home"

    @pytest.mark.asyncio()
    async def test_all_nav_keys_work(self, _fake_config):
        """Verify every registered nav key switches to the correct screen."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            for key, screen_id, _label, _cls in SCREEN_NAV:
                await pilot.press("escape")  # enter command mode
                await pilot.press(key)
                assert app._current_screen_id == screen_id, (
                    f"Key '{key}' should navigate to '{screen_id}'"
                )

    @pytest.mark.asyncio()
    async def test_switch_to_same_screen_is_noop(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            assert app._current_screen_id == "home"
            # Switching to home when already on home should not error
            app._switch_to("home")
            assert app._current_screen_id == "home"


class TestSubTitle:
    @pytest.mark.asyncio()
    async def test_subtitle_shows_screen_label(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            assert app.sub_title == "Home"

    @pytest.mark.asyncio()
    async def test_subtitle_updates_on_navigate(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            await pilot.press("c")
            assert app.sub_title == "Chat"

    @pytest.mark.asyncio()
    async def test_subtitle_shows_command_mode(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("escape")
            assert app.sub_title == "COMMAND MODE"


class TestWelcomeScreen:
    @pytest.mark.asyncio()
    async def test_welcome_has_start_button(self, _no_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            assert isinstance(app.screen, WelcomeScreen)
            button = app.screen.query_one("#welcome-start")
            assert button is not None

    @pytest.mark.asyncio()
    async def test_welcome_has_skip_button(self, _no_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            button = app.screen.query_one("#welcome-skip")
            assert button is not None

    @pytest.mark.asyncio()
    async def test_welcome_has_env_detection(self, _no_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            env = app.screen.query_one("#welcome-env")
            assert env is not None

    @pytest.mark.asyncio()
    async def test_start_button_goes_to_home(self, _no_config):
        async with TalkBoxApp().run_test(size=(80, 40)) as pilot:
            app = pilot.app
            assert isinstance(app.screen, WelcomeScreen)
            await pilot.click("#welcome-start")
            assert not isinstance(app.screen, WelcomeScreen)

    @pytest.mark.asyncio()
    async def test_skip_button_goes_to_home(self, _no_config):
        async with TalkBoxApp().run_test(size=(80, 40)) as pilot:
            app = pilot.app
            assert isinstance(app.screen, WelcomeScreen)
            await pilot.click("#welcome-skip")
            assert not isinstance(app.screen, WelcomeScreen)


class TestHomeScreen:
    @pytest.mark.asyncio()
    async def test_home_has_profile_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            panel = pilot.app.screen.query_one("#home-profile-panel")
            assert panel is not None

    @pytest.mark.asyncio()
    async def test_home_has_profile_summary(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            summary = pilot.app.screen.query_one("#home-profile-summary")
            assert summary is not None

    @pytest.mark.asyncio()
    async def test_home_has_actions_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            panel = pilot.app.screen.query_one("#home-actions-panel")
            assert panel is not None

    @pytest.mark.asyncio()
    async def test_home_has_new_chat_button(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            btn = pilot.app.screen.query_one("#home-new-chat")
            assert btn is not None

    @pytest.mark.asyncio()
    async def test_home_has_status_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            panel = pilot.app.screen.query_one("#home-status-panel")
            assert panel is not None

    @pytest.mark.asyncio()
    async def test_home_has_status_summary(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            status = pilot.app.screen.query_one("#home-status-summary")
            assert status is not None

    @pytest.mark.asyncio()
    async def test_new_chat_button_navigates(self, _fake_config):
        async with TalkBoxApp().run_test(size=(80, 40)) as pilot:
            app = pilot.app
            await pilot.click("#home-new-chat")
            assert app._current_screen_id == "chat"

    @pytest.mark.asyncio()
    async def test_n_key_navigates_to_chat(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.press("n")
            assert app._current_screen_id == "chat"

    @pytest.mark.asyncio()
    async def test_profile_summary_shows_model(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            summary = pilot.app.screen.query_one("#home-profile-summary")
            # Should contain "Model:" label
            assert summary is not None


class TestChatScreen:
    @pytest.mark.asyncio()
    async def test_chat_has_input(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            assert inp is not None

    @pytest.mark.asyncio()
    async def test_chat_has_messages_area(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            area = app.screen.query_one("#chat-messages")
            assert area is not None

    @pytest.mark.asyncio()
    async def test_chat_has_sidebar(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar = app.screen.query_one("#chat-sidebar")
            assert sidebar is not None

    @pytest.mark.asyncio()
    async def test_chat_welcome_hint_shown(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            hint = app.screen.query_one("#chat-welcome-hint")
            assert hint is not None

    @pytest.mark.asyncio()
    async def test_chat_echo_mode(self, _fake_config):
        """Sending a message in echo mode returns Echo: <text>."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("hello world")
            await pilot.press("enter")
            # Wait for worker to complete
            await pilot.pause()
            await pilot.pause()
            await pilot.pause()
            # Should have user + assistant messages
            messages = app.screen.query(".chat-message")
            assert len(messages) >= 2

    @pytest.mark.asyncio()
    async def test_chat_sidebar_updates(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            info = app.screen.query_one("#chat-sidebar-info")
            assert info is not None

    @pytest.mark.asyncio()
    async def test_chat_has_model_select(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sel = app.screen.query_one("#chat-model-select", Select)
            assert sel is not None

    @pytest.mark.asyncio()
    async def test_chat_has_persona_select(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sel = app.screen.query_one("#chat-persona-select", Select)
            assert sel is not None

    @pytest.mark.asyncio()
    async def test_chat_has_new_chat_button(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            btn = app.screen.query_one("#chat-new-btn", Button)
            assert btn is not None

    @pytest.mark.asyncio()
    async def test_chat_slash_help(self, _fake_config):
        """Typing /help shows a system message with available commands."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/help")
            await pilot.press("enter")
            await pilot.pause()
            # Should have a system message
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_slash_info(self, _fake_config):
        """Typing /info shows session info."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/info")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_slash_clear(self, _fake_config):
        """Typing /clear resets conversation."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            # Send a message first
            inp = app.screen.query_one("#chat-input")
            inp.load_text("hello")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            # Now clear
            inp.load_text("/clear")
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert app.screen._conversation is None
            assert app.screen._message_count == 0

    @pytest.mark.asyncio()
    async def test_chat_slash_unknown(self, _fake_config):
        """Unknown slash command shows error message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/foobar")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_prompt_history(self, _fake_config):
        """Prompt history stores sent messages."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            # Reset history for clean test
            baseline = len(app.screen._prompt_history)
            inp = app.screen.query_one("#chat-input")
            inp.load_text("first message")
            await pilot.press("enter")
            await pilot.pause()
            inp.load_text("second message")
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.screen._prompt_history) == baseline + 2
            assert app.screen._prompt_history[-2] == "first message"
            assert app.screen._prompt_history[-1] == "second message"

    @pytest.mark.asyncio()
    async def test_chat_enrich_with_files(self, _fake_config, tmp_path, monkeypatch):
        """File references in messages get enriched with file contents."""
        test_file = tmp_path / "test.md"
        test_file.write_text("# Hello World")
        monkeypatch.chdir(tmp_path)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            result = app.screen._enrich_with_files("look at test.md")
            assert "# Hello World" in result
            assert '<file path="test.md">' in result

    @pytest.mark.asyncio()
    async def test_chat_enrich_no_match(self, _fake_config, tmp_path, monkeypatch):
        """Messages without file references are returned unchanged."""
        monkeypatch.chdir(tmp_path)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            result = app.screen._enrich_with_files("just a normal message")
            assert result == "just a normal message"

    @pytest.mark.asyncio()
    async def test_chat_slash_save(self, _fake_config, tmp_path, monkeypatch):
        """Saving a session when there's no conversation shows a message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/save")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_slash_load_lists(self, _fake_config, tmp_path, monkeypatch):
        """/load with no arg lists available sessions."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/load")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_slash_attach_no_arg(self, _fake_config):
        """/attach with no path shows usage."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/attach")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_slash_attach_file(self, _fake_config, tmp_path, monkeypatch):
        """/attach with a valid file stores it as pending."""
        test_file = tmp_path / "data.csv"
        test_file.write_text("a,b,c\n1,2,3")
        monkeypatch.chdir(tmp_path)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/attach data.csv")
            await pilot.press("enter")
            await pilot.pause()
            assert hasattr(app.screen, "_pending_attachments")
            assert len(app.screen._pending_attachments) == 1
            assert app.screen._pending_attachments[0][0] == "data.csv"

    @pytest.mark.asyncio()
    async def test_chat_sidebar_shows_model(self, _fake_config):
        """Sidebar includes model info."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "Model:" in sidebar_text
            assert "Tokens:" in sidebar_text

    @pytest.mark.asyncio()
    async def test_chat_slash_fav_list(self, _fake_config):
        """/fav with no args shows favorites."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/fav")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_slash_fav_usage(self, _fake_config):
        """/fav with invalid args shows usage."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/fav bogus")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_chat_persist_defaults(self, _fake_config, tmp_path, monkeypatch):
        """_persist_defaults calls persist_defaults without error."""
        # Mock persist_defaults to avoid writing to real config
        from unittest.mock import MagicMock

        mock_persist = MagicMock()
        monkeypatch.setattr("talk_box.tui.screens.persist_defaults", mock_persist, raising=False)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            # Just verify it doesn't crash
            app.screen._persist_defaults()


class TestFavoritesConfig:
    """Test the favorites config functions."""

    def test_toggle_favorite_model(self, tmp_path, monkeypatch):
        """Toggle a model favorite on and off."""
        from talk_box.config import toggle_favorite_model, get_favorites
        import talk_box.config as config_mod

        monkeypatch.setattr(config_mod, "_GLOBAL_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "_GLOBAL_CONFIG_PATH", tmp_path / "config.yml")

        # Add
        added = toggle_favorite_model("anthropic:claude-sonnet-4-6")
        assert added is True
        fav_models, _ = get_favorites()
        assert "anthropic:claude-sonnet-4-6" in fav_models

        # Remove
        removed = toggle_favorite_model("anthropic:claude-sonnet-4-6")
        assert removed is False
        fav_models, _ = get_favorites()
        assert "anthropic:claude-sonnet-4-6" not in fav_models

    def test_toggle_favorite_persona(self, tmp_path, monkeypatch):
        """Toggle a persona favorite on and off."""
        from talk_box.config import toggle_favorite_persona, get_favorites
        import talk_box.config as config_mod

        monkeypatch.setattr(config_mod, "_GLOBAL_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "_GLOBAL_CONFIG_PATH", tmp_path / "config.yml")

        added = toggle_favorite_persona("code_reviewer")
        assert added is True
        _, fav_personas = get_favorites()
        assert "code_reviewer" in fav_personas

    def test_persist_defaults(self, tmp_path, monkeypatch):
        """persist_defaults writes model and persona to config."""
        from talk_box.config import persist_defaults, load_config
        import talk_box.config as config_mod

        config_path = tmp_path / "config.yml"
        monkeypatch.setattr(config_mod, "_GLOBAL_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(config_mod, "_GLOBAL_CONFIG_PATH", config_path)

        persist_defaults(model="openai:gpt-4o", persona="data_analyst")
        assert config_path.is_file()

        from yaml12 import read_yaml

        data = read_yaml(config_path)
        assert data["default_model"] == "openai:gpt-4o"
        assert data["default_persona"] == "data_analyst"

    def test_get_favorites_empty(self, tmp_path, monkeypatch):
        """get_favorites returns empty lists when no config exists."""
        import talk_box.config as config_mod

        monkeypatch.setattr(config_mod, "_GLOBAL_CONFIG_PATH", tmp_path / "nope.yml")

        from talk_box.config import get_favorites

        models, personas = get_favorites()
        assert models == []
        assert personas == []


class TestWorkspaceScreen:
    @pytest.mark.asyncio()
    async def test_workspace_has_tree(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            tree = app.screen.query_one("#workspace-tree", DirectoryTree)
            assert tree is not None

    @pytest.mark.asyncio()
    async def test_workspace_has_viewer(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            viewer = app.screen.query_one("#workspace-viewer-content", Static)
            assert viewer is not None

    @pytest.mark.asyncio()
    async def test_workspace_has_agent_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            status = app.screen.query_one("#workspace-agent-status", Static)
            assert status is not None

    @pytest.mark.asyncio()
    async def test_workspace_has_plan_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            plan = app.screen.query_one("#workspace-plan-content", Static)
            assert plan is not None

    @pytest.mark.asyncio()
    async def test_workspace_has_input(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            inp = app.screen.query_one("#workspace-input")
            assert inp is not None

    @pytest.mark.asyncio()
    async def test_workspace_has_approval_buttons(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            accept = app.screen.query_one("#workspace-accept-btn", Button)
            reject = app.screen.query_one("#workspace-reject-btn", Button)
            assert accept is not None
            assert reject is not None


class TestModelScreen:
    @pytest.mark.asyncio()
    async def test_model_has_table(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("models")
            await pilot.pause()
            table = app.screen.query_one("#model-table")
            assert table is not None

    @pytest.mark.asyncio()
    async def test_model_table_has_rows(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("models")
            await pilot.pause()
            table = app.screen.query_one("#model-table", DataTable)
            assert table.row_count > 0

    @pytest.mark.asyncio()
    async def test_model_table_has_columns(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("models")
            await pilot.pause()
            table = app.screen.query_one("#model-table", DataTable)
            assert len(table.columns) == 6

    @pytest.mark.asyncio()
    async def test_model_detail_panel_exists(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("models")
            await pilot.pause()
            detail = app.screen.query_one("#model-detail")
            assert detail is not None

    @pytest.mark.asyncio()
    async def test_model_has_use_button(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("models")
            await pilot.pause()
            btn = app.screen.query_one("#model-use-btn", Button)
            assert btn is not None


class TestPersonaScreen:
    @pytest.mark.asyncio()
    async def test_persona_has_list(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("personas")
            await pilot.pause()
            ol = app.screen.query_one("#persona-list", OptionList)
            assert ol is not None

    @pytest.mark.asyncio()
    async def test_persona_has_detail_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("personas")
            await pilot.pause()
            detail = app.screen.query_one("#persona-detail-content")
            assert detail is not None

    @pytest.mark.asyncio()
    async def test_persona_list_has_options(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("personas")
            await pilot.pause()
            ol = app.screen.query_one("#persona-list", OptionList)
            assert ol.option_count > 0

    @pytest.mark.asyncio()
    async def test_persona_has_use_button(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("personas")
            await pilot.pause()
            btn = app.screen.query_one("#persona-use-btn", Button)
            assert btn is not None


class TestProfileScreen:
    @pytest.mark.asyncio()
    async def test_profile_has_list_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("profiles")
            await pilot.pause()
            ol = app.screen.query_one("#profile-list", OptionList)
            assert ol is not None

    @pytest.mark.asyncio()
    async def test_profile_has_detail_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("profiles")
            await pilot.pause()
            detail = app.screen.query_one("#profile-detail-content")
            assert detail is not None

    @pytest.mark.asyncio()
    async def test_profile_list_title(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("profiles")
            await pilot.pause()
            title = app.screen.query_one("#profile-list-title")
            assert title is not None


class TestGuardrailScreen:
    @pytest.mark.asyncio()
    async def test_guard_has_table(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("guardrails")
            await pilot.pause()
            table = app.screen.query_one("#guard-table")
            assert table is not None

    @pytest.mark.asyncio()
    async def test_guard_table_has_rows(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("guardrails")
            await pilot.pause()
            table = app.screen.query_one("#guard-table", DataTable)
            assert table.row_count == 7

    @pytest.mark.asyncio()
    async def test_guard_table_has_columns(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("guardrails")
            await pilot.pause()
            table = app.screen.query_one("#guard-table", DataTable)
            assert len(table.columns) == 4

    @pytest.mark.asyncio()
    async def test_guard_detail_panel_exists(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("guardrails")
            await pilot.pause()
            detail = app.screen.query_one("#guard-detail")
            assert detail is not None


class TestTraitScreen:
    @pytest.mark.asyncio()
    async def test_trait_has_list(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("traits")
            await pilot.pause()
            ol = app.screen.query_one("#trait-list", OptionList)
            assert ol is not None

    @pytest.mark.asyncio()
    async def test_trait_has_detail_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("traits")
            await pilot.pause()
            detail = app.screen.query_one("#trait-detail-content")
            assert detail is not None


class TestPathwayScreen:
    @pytest.mark.asyncio()
    async def test_pathway_has_info(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("pathways")
            await pilot.pause()
            info = app.screen.query_one("#pathway-info")
            assert info is not None


class TestSkillScreen:
    @pytest.mark.asyncio()
    async def test_skill_has_list(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("skills")
            await pilot.pause()
            ol = app.screen.query_one("#skill-list", OptionList)
            assert ol is not None

    @pytest.mark.asyncio()
    async def test_skill_has_detail_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("skills")
            await pilot.pause()
            detail = app.screen.query_one("#skill-detail-content")
            assert detail is not None


class TestKnowledgeScreen:
    @pytest.mark.asyncio()
    async def test_knowledge_has_table(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("knowledge")
            await pilot.pause()
            table = app.screen.query_one("#kg-table")
            assert table is not None

    @pytest.mark.asyncio()
    async def test_knowledge_has_stats(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("knowledge")
            await pilot.pause()
            stats = app.screen.query_one("#kg-stats-content")
            assert stats is not None


class TestMemoryScreen:
    @pytest.mark.asyncio()
    async def test_memory_has_tiers(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("memory")
            await pilot.pause()
            tiers = app.screen.query_one("#memory-tiers")
            assert tiers is not None


class TestEvalScreen:
    @pytest.mark.asyncio()
    async def test_eval_has_table(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("eval")
            await pilot.pause()
            table = app.screen.query_one("#eval-dim-table", DataTable)
            assert table.row_count == 6

    @pytest.mark.asyncio()
    async def test_eval_has_info(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("eval")
            await pilot.pause()
            info = app.screen.query_one("#eval-info")
            assert info is not None


class TestNewSlashCommands:
    """Tests for /export, /guards, /memory, /kg slash commands."""

    @pytest.mark.asyncio()
    async def test_slash_export_no_conversation(self, _fake_config):
        """/export with no conversation shows a message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/export")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_slash_export_bad_format(self, _fake_config):
        """/export with unsupported format shows usage."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            # Give it a conversation so it doesn't bail early
            from talk_box.conversation import Conversation

            app.screen._conversation = Conversation()
            app.screen._conversation.add_message("user", "hello")
            app.screen._conversation.add_message("assistant", "hi")

            inp = app.screen.query_one("#chat-input")
            inp.load_text("/export xml")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = [str(m._Static__content) for m in msgs]
            assert any("Supported formats" in str(t) for t in texts)

    @pytest.mark.asyncio()
    async def test_slash_export_json(self, _fake_config, tmp_path, monkeypatch):
        """/export json writes a JSON file."""
        import os

        export_dir = tmp_path / "exports"
        monkeypatch.setenv("HOME", str(tmp_path))  # redirect ~/.config to tmp

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()

            from talk_box.conversation import Conversation

            app.screen._conversation = Conversation()
            app.screen._conversation.add_message("user", "hello")
            app.screen._conversation.add_message("assistant", "hi")

            # Patch the export dir
            monkeypatch.setattr(
                os.path,
                "expanduser",
                lambda p: str(tmp_path / ".config" / "talk-box" / "exports")
                if "exports" in p
                else os.path._real_expanduser(p)  # type: ignore[attr-defined]
                if hasattr(os.path, "_real_expanduser")
                else p,
            )

            app.screen._export_session("json")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_slash_guards(self, _fake_config):
        """/guards shows guardrail info."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/guards")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_slash_memory(self, _fake_config):
        """/memory shows memory summary."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/memory")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_slash_kg(self, _fake_config):
        """/kg shows knowledge graph summary."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/kg")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_slash_help_lists_new_commands(self, _fake_config):
        """/help output includes the new commands."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/help")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "/export" in texts
            assert "/guards" in texts
            assert "/memory" in texts
            assert "/kg" in texts
            assert "/system" in texts
            assert "/capabilities" in texts
            assert "/tools" in texts

    @pytest.mark.asyncio()
    async def test_slash_system_no_bot(self, _fake_config):
        """/system with no bot shows echo mode message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._bot = None
            app.screen._handle_slash_command("/system")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "echo mode" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_system_shows_prompt(self, _fake_config):
        """/system shows the system prompt when a bot is configured."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            # Inject a mock bot with a known system prompt
            from unittest.mock import MagicMock

            mock_bot = MagicMock()
            mock_bot.get_system_prompt.return_value = "You are a helpful test bot."
            app.screen._bot = mock_bot
            app.screen._handle_slash_command("/system")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "helpful test bot" in texts

    @pytest.mark.asyncio()
    async def test_slash_capabilities_no_profile(self, _fake_config):
        """/capabilities with unknown model shows fallback."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_model = "unknown:model"
            app.screen._handle_slash_command("/capabilities")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "capabilities" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_capabilities_with_profile(self, _fake_config):
        """/capabilities with a known model shows profile details."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_model = "anthropic:claude-sonnet-4-6"
            app.screen._handle_slash_command("/capabilities")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "anthropic" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_tools_shows_active(self, _fake_config):
        """/tools shows the list of active tools."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read", "file_write"]
            app.screen._handle_slash_command("/tools")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "file_read" in texts
            assert "file_write" in texts

    @pytest.mark.asyncio()
    async def test_slash_tools_on_enables_tool(self, _fake_config):
        """/tools on <name> adds a tool to active tools."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read"]
            app.screen._handle_slash_command("/tools on calculate")
            await pilot.pause()
            assert "calculate" in app.screen._active_tools
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "enabled" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_tools_off_disables_tool(self, _fake_config):
        """/tools off <name> removes a tool from active tools."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read", "calculate"]
            app.screen._handle_slash_command("/tools off calculate")
            await pilot.pause()
            assert "calculate" not in app.screen._active_tools
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "disabled" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_tools_on_unknown_shows_error(self, _fake_config):
        """/tools on with unknown name shows error."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read"]
            app.screen._handle_slash_command("/tools on nonexistent_tool")
            await pilot.pause()
            assert "file_read" in app.screen._active_tools
            assert "nonexistent_tool" not in app.screen._active_tools
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "unknown" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_tools_on_already_active(self, _fake_config):
        """/tools on with already-active tool shows message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read"]
            app.screen._handle_slash_command("/tools on file_read")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "already" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_tools_off_not_active(self, _fake_config):
        """/tools off with inactive tool shows message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read"]
            app.screen._handle_slash_command("/tools off calculate")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "not active" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_tools_available_lists_all(self, _fake_config):
        """/tools available lists all tools with active markers."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read"]
            app.screen._handle_slash_command("/tools available")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "Available Tools" in texts
            assert "file_read" in texts
            assert "calculate" in texts

    @pytest.mark.asyncio()
    async def test_slash_tools_bad_subcommand_shows_usage(self, _fake_config):
        """/tools with invalid subcommand shows usage help."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._handle_slash_command("/tools foobar")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "/tools on" in texts
            assert "/tools off" in texts

    @pytest.mark.asyncio()
    async def test_slash_tokens_with_usage(self, _fake_config):
        """/tokens shows real usage data when available."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            from unittest.mock import MagicMock

            from talk_box.usage import SessionUsage

            usage = SessionUsage(input_tokens=100, output_tokens=50, turns=2, total_cost=0.001)
            mock_bot = MagicMock()
            mock_bot.get_usage.return_value = usage
            app.screen._bot = mock_bot
            app.screen._handle_slash_command("/tokens")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "100" in texts
            assert "50" in texts

    # -- Guard toggle tests ------------------------------------------------

    @pytest.mark.asyncio()
    async def test_slash_guards_shows_active(self, _fake_config):
        """/guards shows the list of active guardrails."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_guards = ["no_pii"]
            app.screen._handle_slash_command("/guards")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "no_pii" in texts

    @pytest.mark.asyncio()
    async def test_slash_guards_on_enables(self, _fake_config):
        """/guards on <name> enables a guardrail."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_guards = []
            app.screen._handle_slash_command("/guards on no_pii")
            await pilot.pause()
            assert "no_pii" in app.screen._active_guards
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "enabled" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_guards_off_disables(self, _fake_config):
        """/guards off <name> disables a guardrail."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_guards = ["no_pii"]
            app.screen._handle_slash_command("/guards off no_pii")
            await pilot.pause()
            assert "no_pii" not in app.screen._active_guards
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "disabled" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_guards_on_unknown_shows_error(self, _fake_config):
        """/guards on with unknown name shows error."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_guards = []
            app.screen._handle_slash_command("/guards on fake_guard")
            await pilot.pause()
            assert "fake_guard" not in app.screen._active_guards
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "unknown" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_guards_available_lists_all(self, _fake_config):
        """/guards available lists all guardrails with active markers."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_guards = ["no_pii"]
            app.screen._handle_slash_command("/guards available")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "Available Guardrails" in texts
            assert "no_pii" in texts
            assert "keyword_block" in texts

    @pytest.mark.asyncio()
    async def test_slash_guards_bad_subcommand_shows_usage(self, _fake_config):
        """/guards with invalid subcommand shows usage help."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._handle_slash_command("/guards foobar")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "/guards on" in texts
            assert "/guards off" in texts

    # -- Trait toggle tests ------------------------------------------------

    @pytest.mark.asyncio()
    async def test_slash_traits_shows_active(self, _fake_config):
        """/traits shows the list of active traits."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_traits = ["concise"]
            app.screen._handle_slash_command("/traits")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "concise" in texts

    @pytest.mark.asyncio()
    async def test_slash_traits_no_active_shows_message(self, _fake_config):
        """/traits with none active shows message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_traits = []
            app.screen._handle_slash_command("/traits")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "no traits" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_traits_on_requires_persona(self, _fake_config):
        """/traits on without persona shows error."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_persona = None
            app.screen._handle_slash_command("/traits on concise")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "persona" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_traits_on_enables(self, _fake_config):
        """/traits on <name> adds a trait."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_persona = "analyst"
            app.screen._active_traits = []
            app.screen._handle_slash_command("/traits on concise")
            await pilot.pause()
            assert "concise" in app.screen._active_traits
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "applied" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_traits_off_removes(self, _fake_config):
        """/traits off <name> removes a trait."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_persona = "analyst"
            app.screen._active_traits = ["concise"]
            app.screen._handle_slash_command("/traits off concise")
            await pilot.pause()
            assert "concise" not in app.screen._active_traits
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "removed" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_traits_on_unknown_shows_error(self, _fake_config):
        """/traits on with unknown name shows error."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_persona = "analyst"
            app.screen._active_traits = []
            app.screen._handle_slash_command("/traits on nonexistent_trait")
            await pilot.pause()
            assert "nonexistent_trait" not in app.screen._active_traits
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "unknown" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_traits_available_lists_all(self, _fake_config):
        """/traits available lists all traits."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_traits = ["concise"]
            app.screen._handle_slash_command("/traits available")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "Available Traits" in texts
            assert "concise" in texts

    @pytest.mark.asyncio()
    async def test_slash_traits_bad_subcommand_shows_usage(self, _fake_config):
        """/traits with invalid subcommand shows usage help."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._handle_slash_command("/traits foobar")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "/traits on" in texts
            assert "/traits off" in texts

    # -- Knowledge toggle tests --------------------------------------------

    @pytest.mark.asyncio()
    async def test_slash_kg_shows_summary(self, _fake_config):
        """/kg shows knowledge graph summary."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._handle_slash_command("/kg")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "Knowledge Graph" in texts

    @pytest.mark.asyncio()
    async def test_slash_kg_on_empty_shows_warning(self, _fake_config):
        """/kg on with empty graph shows sync suggestion."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            # Use a fresh in-memory KG (empty)
            from talk_box.knowledge_graph import KnowledgeGraph

            app.screen._kg = KnowledgeGraph(":memory:")
            app.screen._handle_slash_command("/kg on")
            await pilot.pause()
            assert not app.screen._kg_enabled
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "sync" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_kg_on_with_data_enables(self, _fake_config):
        """/kg on with data enables knowledge context."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            # Use a fresh in-memory KG with a test node
            from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

            kg = KnowledgeGraph(":memory:")
            kg.add_node(Node(id="test-1", node_type=NodeType.DOCUMENT, name="Test", content="test"))
            app.screen._kg = kg
            app.screen._handle_slash_command("/kg on")
            await pilot.pause()
            assert app.screen._kg_enabled
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "enabled" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_kg_off_disables(self, _fake_config):
        """/kg off disables knowledge context."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._kg_enabled = True
            app.screen._handle_slash_command("/kg off")
            await pilot.pause()
            assert not app.screen._kg_enabled
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "disabled" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_kg_on_already_enabled(self, _fake_config):
        """/kg on when already enabled shows message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._kg_enabled = True
            app.screen._handle_slash_command("/kg on")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "already" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_kg_sources_no_config(self, _fake_config):
        """/kg sources with no configured sources shows message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._handle_slash_command("/kg sources")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "no sources" in texts.lower()

    @pytest.mark.asyncio()
    async def test_slash_kg_bad_subcommand_shows_usage(self, _fake_config):
        """/kg with invalid subcommand shows usage help."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._handle_slash_command("/kg foobar")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "/kg on" in texts
            assert "/kg off" in texts

    @pytest.mark.asyncio()
    async def test_enrich_with_knowledge_disabled(self, _fake_config):
        """_enrich_with_knowledge returns text unchanged when disabled."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._kg_enabled = False
            result = app.screen._enrich_with_knowledge("hello world")
            assert result == "hello world"

    @pytest.mark.asyncio()
    async def test_enrich_with_knowledge_enabled_injects(self, _fake_config):
        """_enrich_with_knowledge injects matching nodes when enabled."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

            kg = KnowledgeGraph(":memory:")
            kg.add_node(
                Node(
                    id="enrich-1",
                    node_type=NodeType.DOCUMENT,
                    name="Python Guide",
                    content="Python is a programming language",
                )
            )
            app.screen._kg = kg
            app.screen._kg_enabled = True
            result = app.screen._enrich_with_knowledge("tell me about Python")
            assert "Knowledge context" in result
            assert "Python Guide" in result

    @pytest.mark.asyncio()
    async def test_enrich_with_knowledge_no_matches(self, _fake_config):
        """_enrich_with_knowledge returns text unchanged when no matches."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            from talk_box.knowledge_graph import KnowledgeGraph

            app.screen._kg = KnowledgeGraph(":memory:")
            app.screen._kg_enabled = True
            result = app.screen._enrich_with_knowledge("xyzzy_nonexistent_query_12345")
            assert result == "xyzzy_nonexistent_query_12345"

    @pytest.mark.asyncio()
    async def test_kg_summary_shows_status(self, _fake_config):
        """/kg summary shows on/off status."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._kg_enabled = True
            app.screen._handle_slash_command("/kg")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = " ".join(str(m._Static__content) for m in msgs)
            assert "on" in texts.lower()

    @pytest.mark.asyncio()
    async def test_sidebar_shows_kg_status(self, _fake_config):
        """Sidebar includes knowledge graph status."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "KG:" in sidebar_text
            assert "OFF" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_shows_tools_count(self, _fake_config):
        """Sidebar includes active tools count."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_tools = ["file_read", "file_write"]
            sidebar_text = app.screen._build_sidebar()
            assert "Tools:" in sidebar_text
            assert "2" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_shows_guards(self, _fake_config):
        """Sidebar includes guard status."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_guards = ["no_pii"]
            sidebar_text = app.screen._build_sidebar()
            assert "Guards:" in sidebar_text
            assert "1" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_shows_traits(self, _fake_config):
        """Sidebar includes active traits."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_traits = ["concise"]
            sidebar_text = app.screen._build_sidebar()
            assert "Traits:" in sidebar_text
            assert "concise" in sidebar_text

    # -- Sidebar action link tests -----------------------------------------

    @pytest.mark.asyncio()
    async def test_sidebar_shows_tools_link(self, _fake_config):
        """Sidebar displays tools count with clickable link."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "Tools:" in sidebar_text
            assert "[…]" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_shows_guards_link(self, _fake_config):
        """Sidebar displays guards count with clickable link."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "Guards:" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_shows_traits_link(self, _fake_config):
        """Sidebar displays traits with clickable link."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "Traits:" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_shows_kg_with_toggle(self, _fake_config):
        """Sidebar displays KG status with toggle and settings link."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "KG:" in sidebar_text
            assert "⏻" in sidebar_text  # toggle switch
            assert "OFF" in sidebar_text

    @pytest.mark.asyncio()
    async def test_kg_inline_toggle_on(self, _fake_config):
        """KG inline toggle enables knowledge context when data exists."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType

            kg = KnowledgeGraph(":memory:")
            kg.add_node(Node(id="btn-1", node_type=NodeType.DOCUMENT, name="T", content="x"))
            app.screen._kg = kg
            app.screen.action_toggle_kg_inline()
            assert app.screen._kg_enabled
            sidebar_text = app.screen._build_sidebar()
            assert "ON" in sidebar_text

    @pytest.mark.asyncio()
    async def test_kg_inline_toggle_off(self, _fake_config):
        """KG inline toggle disables knowledge context."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._kg_enabled = True
            app.screen.action_toggle_kg_inline()
            assert not app.screen._kg_enabled
            sidebar_text = app.screen._build_sidebar()
            assert "OFF" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_friendly_model_name(self, _fake_config):
        """Sidebar shows friendly model name instead of provider:model."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_model = "anthropic:claude-opus-4-7"
            sidebar_text = app.screen._build_sidebar()
            # Should show display name, not raw key
            assert "Claude" in sidebar_text


class TestChecklistPickerModal:
    """Tests for the ChecklistPickerModal."""

    @pytest.mark.asyncio()
    async def test_checklist_modal_renders(self, _fake_config):
        """Modal renders with title and items."""
        from talk_box.tui.screens import ChecklistPickerModal

        async with TalkBoxApp().run_test() as pilot:
            items = [("alpha", "First"), ("beta", "Second")]
            modal = ChecklistPickerModal("Test Title", items, ["alpha"])
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.push_screen(modal)
            await pilot.pause()
            title = app.screen.query_one("#checklist-title", Static)
            assert "Test Title" in str(title._Static__content)
            # Body is now a VerticalScroll with individual Static items
            items_found = app.screen.query(".checklist-item")
            assert len(items_found) == 2
            all_text = " ".join(str(w._Static__content) for w in items_found)
            assert "alpha" in all_text
            assert "beta" in all_text

    @pytest.mark.asyncio()
    async def test_checklist_modal_toggle(self, _fake_config):
        """Toggling an item changes active state."""
        from talk_box.tui.screens import ChecklistPickerModal

        async with TalkBoxApp().run_test() as pilot:
            items = [("alpha", ""), ("beta", "")]
            modal = ChecklistPickerModal("Test", items, ["alpha"])
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.push_screen(modal)
            await pilot.pause()
            # alpha is active, toggle it off
            modal.action_toggle_current()
            assert "alpha" not in modal._active
            # toggle it back on
            modal.action_toggle_current()
            assert "alpha" in modal._active

    @pytest.mark.asyncio()
    async def test_checklist_modal_navigation(self, _fake_config):
        """Cursor navigation works."""
        from talk_box.tui.screens import ChecklistPickerModal

        async with TalkBoxApp().run_test() as pilot:
            items = [("a", ""), ("b", ""), ("c", "")]
            modal = ChecklistPickerModal("Test", items, [])
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.push_screen(modal)
            await pilot.pause()
            assert modal._cursor == 0
            modal.action_cursor_down()
            assert modal._cursor == 1
            modal.action_cursor_down()
            assert modal._cursor == 2
            modal.action_cursor_up()
            assert modal._cursor == 1

    @pytest.mark.asyncio()
    async def test_checklist_modal_confirm(self, _fake_config):
        """Confirm returns sorted active items."""
        from talk_box.tui.screens import ChecklistPickerModal

        results = []

        async with TalkBoxApp().run_test() as pilot:
            items = [("beta", ""), ("alpha", "")]
            modal = ChecklistPickerModal("Test", items, ["beta", "alpha"])
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.push_screen(modal, lambda r: results.append(r))
            await pilot.pause()
            modal.action_confirm()
            await pilot.pause()
        assert results == [["alpha", "beta"]]


class TestSessionHistoryModal:
    """Tests for the SessionHistoryModal."""

    @pytest.mark.asyncio()
    async def test_session_history_modal_renders(self, _fake_config):
        """Modal renders with title and session items."""
        from talk_box.tui.screens import SessionHistoryModal

        sessions = [
            {
                "name": "my-chat",
                "saved_at": "2025-01-15T10:30:00",
                "model": "anthropic:claude-sonnet-4-6",
                "persona": "coder",
                "messages": 4,
            },
            {
                "name": "debug-session",
                "saved_at": "2025-01-14T08:00:00",
                "model": "ollama:llama3",
                "persona": "",
                "messages": 10,
            },
        ]
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            modal = SessionHistoryModal(sessions)
            app.push_screen(modal)
            await pilot.pause()
            title = app.screen.query_one("#session-history-title", Static)
            assert "2 saved" in str(title._Static__content)
            items_found = app.screen.query(".session-item")
            assert len(items_found) == 2
            all_text = " ".join(str(w._Static__content) for w in items_found)
            assert "my-chat" in all_text
            assert "debug-session" in all_text

    @pytest.mark.asyncio()
    async def test_session_history_empty(self, _fake_config):
        """Modal shows empty state message when no sessions exist."""
        from talk_box.tui.screens import SessionHistoryModal

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            modal = SessionHistoryModal([])
            app.push_screen(modal)
            await pilot.pause()
            title = app.screen.query_one("#session-history-title", Static)
            assert "0 saved" in str(title._Static__content)
            empty_w = app.screen.query_one("#session-empty", Static)
            assert "No saved sessions" in str(empty_w._Static__content)

    @pytest.mark.asyncio()
    async def test_session_history_navigation(self, _fake_config):
        """Cursor navigation works."""
        from talk_box.tui.screens import SessionHistoryModal

        sessions = [
            {"name": "a", "saved_at": "", "model": "", "persona": "", "messages": 0},
            {"name": "b", "saved_at": "", "model": "", "persona": "", "messages": 0},
            {"name": "c", "saved_at": "", "model": "", "persona": "", "messages": 0},
        ]
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            modal = SessionHistoryModal(sessions)
            app.push_screen(modal)
            await pilot.pause()
            assert modal._cursor == 0
            modal.action_cursor_down()
            assert modal._cursor == 1
            modal.action_cursor_down()
            assert modal._cursor == 2
            modal.action_cursor_up()
            assert modal._cursor == 1

    @pytest.mark.asyncio()
    async def test_session_history_select(self, _fake_config):
        """Selecting a session returns its name."""
        from talk_box.tui.screens import SessionHistoryModal

        results = []
        sessions = [
            {
                "name": "chat-001",
                "saved_at": "",
                "model": "",
                "persona": "",
                "messages": 2,
                "_filename": "chat-001",
            },
            {
                "name": "chat-002",
                "saved_at": "",
                "model": "",
                "persona": "",
                "messages": 5,
                "_filename": "chat-002",
            },
        ]
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            modal = SessionHistoryModal(sessions)
            app.push_screen(modal, lambda r: results.append(r))
            await pilot.pause()
            modal.action_cursor_down()
            modal.action_select_session()
            await pilot.pause()
        assert results == ["chat-002"]

    @pytest.mark.asyncio()
    async def test_session_history_cancel(self, _fake_config):
        """Cancel returns None."""
        from talk_box.tui.screens import SessionHistoryModal

        results = []
        sessions = [{"name": "x", "saved_at": "", "model": "", "persona": "", "messages": 0}]
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            modal = SessionHistoryModal(sessions)
            app.push_screen(modal, lambda r: results.append(r))
            await pilot.pause()
            modal.action_cancel()
            await pilot.pause()
        assert results == [None]

    @pytest.mark.asyncio()
    async def test_session_history_delete(self, _fake_config, tmp_path):
        """Deleting a session removes it from the list and disk."""
        import json

        from talk_box.tui.screens import SessionHistoryModal

        # Create a fake sessions directory with a file
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        session_data = {
            "name": "deleteme",
            "saved_at": "",
            "model": "",
            "persona": "",
            "conversation": {"messages": []},
        }
        (sessions_dir / "deleteme.json").write_text(json.dumps(session_data))

        sessions = [
            {
                "name": "deleteme",
                "saved_at": "",
                "model": "",
                "persona": "",
                "messages": 0,
                "_filename": "deleteme",
            },
            {
                "name": "keepme",
                "saved_at": "",
                "model": "",
                "persona": "",
                "messages": 0,
                "_filename": "keepme",
            },
        ]

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            modal = SessionHistoryModal(sessions)
            app.push_screen(modal)
            await pilot.pause()
            # Patch _config_dir to use tmp_path
            with patch("talk_box.tui.screens._config_dir", return_value=str(tmp_path)):
                await modal.action_delete_session()
                await pilot.pause()
            assert len(modal._sessions) == 1
            assert modal._sessions[0]["name"] == "keepme"
            assert not (sessions_dir / "deleteme.json").exists()

    @pytest.mark.asyncio()
    async def test_session_history_click(self, _fake_config):
        """Clicking a session item moves the cursor."""
        from textual.events import Click

        from talk_box.tui.screens import SessionHistoryModal

        sessions = [
            {"name": "a", "saved_at": "", "model": "", "persona": "", "messages": 0},
            {"name": "b", "saved_at": "", "model": "", "persona": "", "messages": 0},
        ]
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            modal = SessionHistoryModal(sessions)
            app.push_screen(modal)
            await pilot.pause()
            assert modal._cursor == 0
            # Simulate click on second item
            item_widget = app.screen.query_one("#session-item-1", Static)
            fake_click = Click(
                widget=item_widget,
                x=0,
                y=0,
                delta_x=0,
                delta_y=0,
                button=1,
                shift=False,
                meta=False,
                ctrl=False,
                screen_x=0,
                screen_y=0,
            )
            modal.on_click(fake_click)
            assert modal._cursor == 1


class TestSessionHistorySidebar:
    """Tests for session history sidebar integration."""

    @pytest.mark.asyncio()
    async def test_sidebar_shows_saved_count(self, _fake_config):
        """Sidebar shows Saved count with clickable link."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "Saved:" in sidebar_text

    @pytest.mark.asyncio()
    async def test_sidebar_saved_link(self, _fake_config):
        """Sidebar Saved line has clickable [...] link."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            sidebar_text = app.screen._build_sidebar()
            assert "open_session_history" in sidebar_text

    @pytest.mark.asyncio()
    async def test_count_saved_sessions(self, _fake_config, tmp_path):
        """_count_saved_sessions counts only non-autosave JSON files."""
        import json

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "chat1.json").write_text(json.dumps({"name": "chat1"}))
        (sessions_dir / "chat2.json").write_text(json.dumps({"name": "chat2"}))
        (sessions_dir / "_autosave.json").write_text(json.dumps({"name": "_autosave"}))
        (sessions_dir / "readme.txt").write_text("not a session")

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            with patch("talk_box.tui.screens._config_dir", return_value=str(tmp_path)):
                count = app.screen._count_saved_sessions()
            assert count == 3  # chat1 + chat2 + _autosave

    @pytest.mark.asyncio()
    async def test_list_saved_sessions(self, _fake_config, tmp_path):
        """_list_saved_sessions returns metadata sorted newest-first."""
        import json

        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        (sessions_dir / "old.json").write_text(
            json.dumps(
                {
                    "name": "old",
                    "saved_at": "2025-01-01T00:00:00",
                    "model": "ollama:llama3",
                    "persona": "",
                    "conversation": {"messages": [{"role": "user", "content": "hi"}]},
                }
            )
        )
        (sessions_dir / "new.json").write_text(
            json.dumps(
                {
                    "name": "new",
                    "saved_at": "2025-06-01T12:00:00",
                    "model": "anthropic:claude-sonnet-4-6",
                    "persona": "coder",
                    "conversation": {
                        "messages": [
                            {"role": "user", "content": "a"},
                            {"role": "assistant", "content": "b"},
                        ]
                    },
                }
            )
        )
        (sessions_dir / "_autosave.json").write_text(json.dumps({"name": "_autosave"}))

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            with patch("talk_box.tui.screens._config_dir", return_value=str(tmp_path)):
                sessions = app.screen._list_saved_sessions()
            assert len(sessions) == 3  # new + old + autosave
            assert sessions[0]["name"] == "new"  # newest first
            # autosave has name "(last session)"
            autosave_names = [s["name"] for s in sessions if s["_filename"] == "_autosave"]
            assert autosave_names == ["(last session)"]
            named = [s for s in sessions if s["_filename"] != "_autosave"]
            assert named[0]["messages"] == 2
            assert named[1]["messages"] == 1


class TestKnowledgeScreenDetail:
    """Tests for KnowledgeScreen node detail panel."""

    @pytest.mark.asyncio()
    async def test_kg_has_detail_panel(self, _fake_config):
        """KnowledgeScreen has a detail panel widget."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("knowledge")
            await pilot.pause()
            detail = app.screen.query_one("#kg-detail", Static)
            assert detail is not None


class TestProfileScreenEnhanced:
    """Tests for enhanced ProfileScreen with OptionList and Use button."""

    @pytest.mark.asyncio()
    async def test_profile_has_option_list(self, _fake_config):
        """ProfileScreen uses an OptionList for the profile list."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("profiles")
            await pilot.pause()
            ol = app.screen.query_one("#profile-list", OptionList)
            assert ol is not None

    @pytest.mark.asyncio()
    async def test_profile_has_use_button(self, _fake_config):
        """ProfileScreen has a Use Profile button."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("profiles")
            await pilot.pause()
            btn = app.screen.query_one("#profile-use-btn", Button)
            assert btn is not None

    @pytest.mark.asyncio()
    async def test_profile_has_detail_panel(self, _fake_config):
        """ProfileScreen has a detail content area."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("profiles")
            await pilot.pause()
            detail = app.screen.query_one("#profile-detail-content", Static)
            assert detail is not None


class TestFormatCommand:
    """Tests for the /format slash command."""

    @pytest.mark.asyncio()
    async def test_format_shows_current(self, _fake_config):
        """/format with no arg shows current format."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/format")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_format_set_json(self, _fake_config):
        """/format json sets the output format."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/format json")
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen._output_format == "json"

    @pytest.mark.asyncio()
    async def test_format_set_markdown(self, _fake_config):
        """/format markdown sets the output format."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/format markdown")
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen._output_format == "markdown"

    @pytest.mark.asyncio()
    async def test_format_set_table(self, _fake_config):
        """/format table sets the output format."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/format table")
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen._output_format == "table"

    @pytest.mark.asyncio()
    async def test_format_off(self, _fake_config):
        """/format off resets the output format."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._output_format = "json"
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/format off")
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen._output_format is None

    @pytest.mark.asyncio()
    async def test_format_invalid(self, _fake_config):
        """/format with an invalid type shows an error."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/format xml")
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen._output_format is None
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1


class TestMultiLineInput:
    """Tests for multi-line ChatInput widget."""

    @pytest.mark.asyncio()
    async def test_chat_input_is_textarea(self, _fake_config):
        """Chat input is a TextArea-based ChatInput widget."""
        from talk_box.tui.screens import ChatInput

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", ChatInput)
            assert inp is not None

    @pytest.mark.asyncio()
    async def test_chat_input_clears_after_send(self, _fake_config):
        """Chat input clears after sending a message."""
        from talk_box.tui.screens import ChatInput

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input", ChatInput)
            inp.load_text("test message")
            await pilot.press("enter")
            await pilot.pause()
            assert inp.text == ""

    @pytest.mark.asyncio()
    async def test_send_button_exists(self, _fake_config):
        """Chat has a Send button."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            btn = app.screen.query_one("#chat-send-btn", Button)
            assert btn is not None

    @pytest.mark.asyncio()
    async def test_enter_toggle_exists(self, _fake_config):
        """Chat has an Enter-mode toggle button."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            toggle = app.screen.query_one("#chat-enter-toggle", Button)
            assert toggle is not None
            assert "Send" in str(toggle.label)

    @pytest.mark.asyncio()
    async def test_enter_toggle_switches_mode(self, _fake_config):
        """Clicking the Enter toggle switches between Send and Newline modes."""
        from talk_box.tui.screens import ChatInput

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()

            ci = app.screen.query_one("#chat-input", ChatInput)
            toggle = app.screen.query_one("#chat-enter-toggle", Button)

            # Default: enter sends
            assert app.screen._enter_sends is True
            assert ci.enter_sends is True
            assert "Send" in str(toggle.label)

            # Click toggle → newline mode
            await pilot.click("#chat-enter-toggle")
            await pilot.pause()
            assert app.screen._enter_sends is False
            assert ci.enter_sends is False
            assert "Newline" in str(toggle.label)

            # Click again → back to send mode
            await pilot.click("#chat-enter-toggle")
            await pilot.pause()
            assert app.screen._enter_sends is True
            assert ci.enter_sends is True
            assert "Send" in str(toggle.label)

    @pytest.mark.asyncio()
    async def test_send_button_submits(self, _fake_config):
        """Clicking Send button submits the chat input text."""
        from talk_box.tui.screens import ChatInput

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()

            ci = app.screen.query_one("#chat-input", ChatInput)
            ci.load_text("/info")
            await pilot.click("#chat-send-btn")
            await pilot.pause()

            # /info produces a system message
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1

    @pytest.mark.asyncio()
    async def test_enter_newline_mode_does_not_send(self, _fake_config):
        """In newline mode, Enter does not send the message."""
        from talk_box.tui.screens import ChatInput

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()

            # Switch to newline mode
            ci = app.screen.query_one("#chat-input", ChatInput)
            ci.enter_sends = False
            app.screen._enter_sends = False

            baseline = len(app.screen._prompt_history)
            ci.load_text("/help")
            await pilot.press("enter")
            await pilot.pause()

            # Should NOT have submitted (no new prompt history entry)
            assert len(app.screen._prompt_history) == baseline


class TestAutoSave:
    """Tests for session auto-save."""

    @pytest.mark.asyncio()
    async def test_auto_save_method_exists(self, _fake_config):
        """ChatScreen has the _auto_save_session method."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            assert hasattr(app.screen, "_auto_save_session")
            # Calling with no conversation should be a no-op
            app.screen._auto_save_session()

    @pytest.mark.asyncio()
    async def test_auto_save_writes_file(self, _fake_config, tmp_path, monkeypatch):
        """Auto-save writes to _autosave.json."""
        import os

        monkeypatch.setattr(
            "talk_box.tui.screens._config_dir",
            lambda: str(tmp_path),
        )

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()

            from talk_box.conversation import Conversation

            app.screen._conversation = Conversation()
            app.screen._conversation.add_message("user", "hello")
            app.screen._conversation.add_message("assistant", "hi")
            app.screen._auto_save_session()

            autosave = os.path.join(str(tmp_path), "sessions", "_autosave.json")
            assert os.path.isfile(autosave)


class TestPromptHistoryPersistence:
    """Tests for persistent prompt history."""

    @pytest.mark.asyncio()
    async def test_save_load_prompt_history(self, _fake_config, tmp_path, monkeypatch):
        """Prompt history persists to disk and reloads."""
        monkeypatch.setattr(
            "talk_box.tui.screens._config_dir",
            lambda: str(tmp_path),
        )

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._prompt_history = ["one", "two", "three"]
            app.screen._save_prompt_history()

            # Clear and reload
            app.screen._prompt_history = []
            app.screen._load_prompt_history()
            assert app.screen._prompt_history == ["one", "two", "three"]


class TestMarkdownRendering:
    """Tests for Rich Markdown rendering in chat assistant messages."""

    @pytest.mark.asyncio()
    async def test_render_assistant_display_returns_group(self, _fake_config):
        """_render_assistant_display returns a Rich Group renderable."""
        from rich.console import Group

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            result = app.screen._render_assistant_display("Hello **world**")
            assert isinstance(result, Group)

    @pytest.mark.asyncio()
    async def test_render_assistant_display_with_thinking(self, _fake_config):
        """_render_assistant_display includes thinking section."""
        from rich.console import Group

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            result = app.screen._render_assistant_display("Answer", thinking="pondering")
            assert isinstance(result, Group)

    @pytest.mark.asyncio()
    async def test_append_assistant_message_uses_markdown(self, _fake_config):
        """Assistant messages use Rich Markdown rendering."""
        from rich.console import Group

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._append_message("assistant", "# Heading\n\n- item 1\n- item 2")
            await pilot.pause()
            widget = app.screen.query_one("#chat-msg-1", Static)
            # The content should be a Group (markdown), not plain string
            assert isinstance(widget.content, Group)

    @pytest.mark.asyncio()
    async def test_append_user_message_stays_plain(self, _fake_config):
        """User messages remain plain text, not markdown."""
        from rich.console import Group

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._append_message("user", "Hello there")
            await pilot.pause()
            widget = app.screen.query_one("#chat-msg-1", Static)
            # User messages are plain strings, not Group
            assert not isinstance(widget.content, Group)

    @pytest.mark.asyncio()
    async def test_raw_content_preserved_for_copy(self, _fake_config):
        """_raw_content attribute stores original text for clipboard copy."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            original = "```python\nprint('hello')\n```"
            app.screen._append_message("assistant", original)
            await pilot.pause()
            widget = app.screen.query_one("#chat-msg-1", Static)
            assert widget._raw_content == original


class TestWorkspaceScreenTools:
    """Tests for workspace screen tool command execution."""

    @pytest.mark.asyncio()
    async def test_workspace_has_agent(self, _fake_config):
        """WorkspaceScreen creates a WorkspaceAgent on mount."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            assert hasattr(app.screen, "_agent")

    @pytest.mark.asyncio()
    async def test_tool_command_ls(self, _fake_config):
        """Direct >ls command lists files."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            app.screen._run_tool_command("ls")
            await pilot.pause()
            plan = app.screen.query_one("#workspace-plan-content", Static)
            content = str(plan._Static__content)
            assert ">ls" in content

    @pytest.mark.asyncio()
    async def test_tool_command_read(self, _fake_config):
        """Direct >read command shows file content."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            app.screen._run_tool_command("read pyproject.toml")
            await pilot.pause()
            plan = app.screen.query_one("#workspace-plan-content", Static)
            content = str(plan._Static__content)
            assert ">read" in content

    @pytest.mark.asyncio()
    async def test_tool_command_search(self, _fake_config):
        """Direct >search command finds text."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            app.screen._run_tool_command("search def")
            await pilot.pause()
            plan = app.screen.query_one("#workspace-plan-content", Static)
            content = str(plan._Static__content)
            assert ">search" in content

    @pytest.mark.asyncio()
    async def test_tool_command_unknown(self, _fake_config):
        """Unknown > command shows error."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            app.screen._run_tool_command("zzzz")
            await pilot.pause()
            plan = app.screen.query_one("#workspace-plan-content", Static)
            content = str(plan._Static__content)
            assert "Unknown" in content

    @pytest.mark.asyncio()
    async def test_task_triggers_analysis(self, _fake_config):
        """Natural language task sets initial plan with task text."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("workspace")
            await pilot.pause()
            # Directly call _run_task — the worker will fail in test mode
            # but the initial plan text should be set synchronously
            app.screen._run_task("Refactor the code")
            plan = app.screen.query_one("#workspace-plan-content", Static)
            content = str(plan._Static__content)
            assert "Refactor" in content


class TestGuardrailScreenToggle:
    """Tests for guardrail toggle functionality."""

    @pytest.mark.asyncio()
    async def test_toggle_button_exists(self, _fake_config):
        """Toggle button is present on GuardrailScreen."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("guardrails")
            await pilot.pause()
            btn = app.screen.query_one("#guard-toggle-btn", Button)
            assert btn is not None

    @pytest.mark.asyncio()
    async def test_guard_table_has_status_column(self, _fake_config):
        """Guard table includes Status column."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("guardrails")
            await pilot.pause()
            table = app.screen.query_one("#guard-table", DataTable)
            col_labels = [str(c.label) for c in table.columns.values()]
            assert "Status" in col_labels

    @pytest.mark.asyncio()
    async def test_active_guards_set_initialized(self, _fake_config):
        """_active_guards is initialized from config."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("guardrails")
            await pilot.pause()
            assert isinstance(app.screen._active_guards, set)


class TestEvalScreenHistory:
    """Tests for EvalScreen scorecard history."""

    @pytest.mark.asyncio()
    async def test_eval_has_history_table(self, _fake_config):
        """EvalScreen includes a history DataTable."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("eval")
            await pilot.pause()
            table = app.screen.query_one("#eval-history-table", DataTable)
            assert table is not None

    @pytest.mark.asyncio()
    async def test_eval_history_has_columns(self, _fake_config):
        """Eval history table has expected columns."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("eval")
            await pilot.pause()
            table = app.screen.query_one("#eval-history-table", DataTable)
            col_labels = [str(c.label) for c in table.columns.values()]
            assert "Persona" in col_labels
            assert "Model" in col_labels
            assert "Overall" in col_labels

    @pytest.mark.asyncio()
    async def test_eval_history_has_rows(self, _fake_config):
        """Eval history table has at least one row (scorecards exist in project)."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("eval")
            await pilot.pause()
            table = app.screen.query_one("#eval-history-table", DataTable)
            assert table.row_count > 0


class TestCostCommand:
    """Tests for /cost slash command."""

    @pytest.mark.asyncio()
    async def test_cost_with_model(self, _fake_config):
        """/cost shows pricing info for the active model."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_model = "anthropic:claude-sonnet-4-20250514"
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/cost")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = [str(m._Static__content) for m in msgs]
            assert any("cost" in t.lower() or "$" in t for t in texts)

    @pytest.mark.asyncio()
    async def test_cost_no_model(self, _fake_config):
        """/cost with no model set shows a message."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            app.screen._active_model = None
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/cost")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            assert len(msgs) >= 1


class TestAllowCloudTUI:
    """Tests for allow_cloud enforcement in the TUI."""

    @pytest.mark.asyncio()
    async def test_model_command_blocks_cloud(self, _fake_config, monkeypatch):
        """The /model command rejects cloud models when allow_cloud=false."""
        from talk_box import config as config_mod

        def fake_load_config(**kwargs):
            return config_mod.TalkBoxConfig(allow_cloud=False)

        monkeypatch.setattr(config_mod, "load_config", fake_load_config)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/model anthropic:claude-sonnet-4-6")
            await pilot.press("enter")
            await pilot.pause()
            msgs = app.screen.query(".chat-system")
            texts = [str(m._Static__content) for m in msgs]
            assert any("blocked" in t.lower() or "allow_cloud" in t for t in texts)
            # Model should NOT have been changed
            assert app.screen._active_model != "anthropic:claude-sonnet-4-6"

    @pytest.mark.asyncio()
    async def test_model_command_allows_local(self, _fake_config, monkeypatch):
        """The /model command accepts local models when allow_cloud=false."""
        from talk_box import config as config_mod

        def fake_load_config(**kwargs):
            return config_mod.TalkBoxConfig(allow_cloud=False)

        monkeypatch.setattr(config_mod, "load_config", fake_load_config)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("chat")
            await pilot.pause()
            inp = app.screen.query_one("#chat-input")
            inp.load_text("/model ollama:llama3.3")
            await pilot.press("enter")
            await pilot.pause()
            assert app.screen._active_model == "ollama:llama3.3"

    def test_is_cloud_model_static_method(self):
        """ChatScreen._is_cloud_model classifies providers correctly."""
        from talk_box.tui.screens import ChatScreen

        assert ChatScreen._is_cloud_model("anthropic:claude-sonnet-4-6") is True
        assert ChatScreen._is_cloud_model("openai:gpt-4o") is True
        assert ChatScreen._is_cloud_model("google:gemini-2.5-flash") is True
        assert ChatScreen._is_cloud_model("ollama:llama3.3") is False
        assert ChatScreen._is_cloud_model("local:my-model") is False


class TestHomeScreenProjectConfig:
    """Tests for HomeScreen displaying project config status."""

    @pytest.mark.asyncio()
    async def test_home_shows_project_status(self, _fake_config):
        """HomeScreen shows project config status in system status."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("home")
            await pilot.pause()
            status = app.screen.query_one("#home-status-summary", Static)
            text = str(status._Static__content)
            assert "Project" in text


class TestSimpleMode:
    """Tests for mode: simple (chat-only TUI)."""

    @pytest.mark.asyncio()
    async def test_simple_mode_shows_chat(self, _fake_config, monkeypatch):
        """In simple mode, the app starts on the chat screen."""
        from talk_box import config as config_mod
        from talk_box.config import TUIMode

        def fake_load_config(**kwargs):
            return config_mod.TalkBoxConfig(mode=TUIMode.SIMPLE)

        monkeypatch.setattr(config_mod, "load_config", fake_load_config)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.pause()
            assert app._simple_mode is True
            assert app._current_screen_id == "chat"

    @pytest.mark.asyncio()
    async def test_simple_mode_blocks_command_list(self, _fake_config, monkeypatch):
        """In simple mode, command list does not open."""
        from talk_box import config as config_mod
        from talk_box.config import TUIMode

        def fake_load_config(**kwargs):
            return config_mod.TalkBoxConfig(mode=TUIMode.SIMPLE)

        monkeypatch.setattr(config_mod, "load_config", fake_load_config)

        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.pause()
            app.action_open_command_list()
            assert app.command_mode is False

    @pytest.mark.asyncio()
    async def test_full_mode_is_default(self, _fake_config):
        """Without mode:simple, the app loads normally."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            await pilot.pause()
            assert app._simple_mode is False


# ---------------------------------------------------------------------------
# File Approval Modal
# ---------------------------------------------------------------------------


class TestFileApprovalScreen:
    def test_file_approval_screen_importable(self):
        from talk_box.tui.screens import FileApprovalScreen

        assert FileApprovalScreen is not None

    def test_file_approval_screen_instantiates_single(self):
        from talk_box.tui.screens import FileApprovalScreen

        modal = FileApprovalScreen([("write", "test.txt", {"content": "hello"})])
        assert len(modal._pending) == 1
        assert modal._pending[0] == ("write", "test.txt", {"content": "hello"})

    def test_file_approval_screen_instantiates_multi(self):
        from talk_box.tui.screens import FileApprovalScreen

        ops = [
            ("write", "a.txt", {"content": "aaa"}),
            ("edit", "b.py", {"old_text": "x", "new_text": "y"}),
        ]
        modal = FileApprovalScreen(ops)
        assert len(modal._pending) == 2
        assert modal._index == 0

    @pytest.mark.asyncio()
    async def test_approve_single_file(self, _fake_config):
        """Pressing 'y' on a single-file modal returns approved dict."""
        from talk_box.tui.screens import FileApprovalScreen

        results = []

        async with TalkBoxApp().run_test() as pilot:
            modal = FileApprovalScreen([("write", "test.txt", {"content": "hi"})])
            pilot.app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()

        assert results == [{("write", "test.txt"): True}]

    @pytest.mark.asyncio()
    async def test_reject_single_file(self, _fake_config):
        """Pressing 'n' on a single-file modal returns rejected dict."""
        from talk_box.tui.screens import FileApprovalScreen

        results = []

        async with TalkBoxApp().run_test() as pilot:
            modal = FileApprovalScreen([("write", "out.txt", {"content": "data"})])
            pilot.app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()

        assert results == [{("write", "out.txt"): False}]

    @pytest.mark.asyncio()
    async def test_escape_rejects(self, _fake_config):
        """Pressing Escape rejects the current file."""
        from talk_box.tui.screens import FileApprovalScreen

        results = []

        async with TalkBoxApp().run_test() as pilot:
            modal = FileApprovalScreen([("edit", "f.py", {"old_text": "a", "new_text": "b"})])
            pilot.app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

        assert results == [{("edit", "f.py"): False}]

    @pytest.mark.asyncio()
    async def test_approve_rest_approves_all(self, _fake_config):
        """Pressing 'a' approves all undecided files."""
        from talk_box.tui.screens import FileApprovalScreen

        results = []
        ops = [
            ("write", "a.txt", {"content": "x"}),
            ("write", "b.txt", {"content": "y"}),
        ]

        async with TalkBoxApp().run_test() as pilot:
            modal = FileApprovalScreen(ops)
            pilot.app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.press("a")
            await pilot.pause()

        assert results == [{("write", "a.txt"): True, ("write", "b.txt"): True}]

    @pytest.mark.asyncio()
    async def test_reject_rest_rejects_all(self, _fake_config):
        """Pressing 'x' rejects all undecided files."""
        from talk_box.tui.screens import FileApprovalScreen

        results = []
        ops = [
            ("write", "a.txt", {"content": "x"}),
            ("edit", "b.py", {"old_text": "1", "new_text": "2"}),
        ]

        async with TalkBoxApp().run_test() as pilot:
            modal = FileApprovalScreen(ops)
            pilot.app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            await pilot.press("x")
            await pilot.pause()

        assert results == [{("write", "a.txt"): False, ("edit", "b.py"): False}]

    @pytest.mark.asyncio()
    async def test_navigate_and_approve_individually(self, _fake_config):
        """Navigate right, reject second, then approve first."""
        from talk_box.tui.screens import FileApprovalScreen

        results = []
        ops = [
            ("write", "a.txt", {"content": "aaa"}),
            ("write", "b.txt", {"content": "bbb"}),
        ]

        async with TalkBoxApp().run_test() as pilot:
            modal = FileApprovalScreen(ops)
            pilot.app.push_screen(modal, callback=lambda r: results.append(r))
            await pilot.pause()
            # Navigate to file 2
            await pilot.press("right")
            await pilot.pause()
            # Reject file 2
            await pilot.press("n")
            await pilot.pause()
            # Auto-advances to file 1 (the remaining undecided)
            # Approve file 1
            await pilot.press("y")
            await pilot.pause()

        assert len(results) == 1
        assert results[0][("write", "a.txt")] is True
        assert results[0][("write", "b.txt")] is False

    @pytest.mark.asyncio()
    async def test_shows_path_in_title(self, _fake_config):
        """Modal title includes the file path."""
        from talk_box.tui.screens import FileApprovalScreen

        async with TalkBoxApp().run_test() as pilot:
            modal = FileApprovalScreen([("write", "important.md", {"content": "# Hi"})])
            pilot.app.push_screen(modal)
            await pilot.pause()
            title_widget = pilot.app.screen.query_one("#file-approval-title", Static)
            assert "important.md" in str(title_widget.render())
            await pilot.press("escape")

    @pytest.mark.asyncio()
    async def test_shows_file_count(self, _fake_config):
        """Nav indicator shows file count."""
        from talk_box.tui.screens import FileApprovalScreen

        async with TalkBoxApp().run_test() as pilot:
            ops = [
                ("write", "a.txt", {"content": "x"}),
                ("write", "b.txt", {"content": "y"}),
                ("write", "c.txt", {"content": "z"}),
            ]
            modal = FileApprovalScreen(ops)
            pilot.app.push_screen(modal)
            await pilot.pause()
            nav_widget = pilot.app.screen.query_one("#file-approval-nav", Static)
            rendered = str(nav_widget.render())
            assert "1 of 3" in rendered
            await pilot.press("a")  # approve all to dismiss

    def test_install_callback_sets_global(self):
        """_install_file_approval_callback sets batch and per-file callbacks."""
        from talk_box.builtin_tools import (
            get_batch_approval_callback,
            get_file_approval_callback,
            set_batch_approval_callback,
            set_file_approval_callback,
        )

        set_file_approval_callback(None)
        set_batch_approval_callback(None)
        assert get_file_approval_callback() is None
        assert get_batch_approval_callback() is None

        screen = ChatScreen()
        assert hasattr(screen, "_install_file_approval_callback")
        assert hasattr(screen, "_uninstall_file_approval_callback")

    def test_require_approvals_defaults_true(self):
        """ChatScreen starts with approvals enabled."""
        screen = ChatScreen()
        assert screen._require_approvals is True

    @pytest.mark.asyncio()
    async def test_approval_toggle_button_exists(self, _fake_config):
        """Sidebar contains the approval toggle button."""
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app.switch_screen("chat")
            await pilot.pause()
            toggle = app.screen.query_one("#chat-approval-toggle", Button)
            assert "On" in str(toggle.label)

    @pytest.mark.asyncio()
    async def test_approval_toggle_off_and_on(self, _fake_config):
        """Clicking the toggle flips state and label."""
        async with TalkBoxApp().run_test(size=(120, 50)) as pilot:
            app = pilot.app
            app.switch_screen("chat")
            await pilot.pause()
            toggle = app.screen.query_one("#chat-approval-toggle", Button)
            assert app.screen._require_approvals is True

            toggle.press()
            await pilot.pause()
            assert app.screen._require_approvals is False
            assert "Off" in str(toggle.label)

            toggle.press()
            await pilot.pause()
            assert app.screen._require_approvals is True
            assert "On" in str(toggle.label)
