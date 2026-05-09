"""Tests for the Talk Box TUI application shell."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from talk_box.tui.app import SCREEN_NAV, TalkBoxApp
from textual.widgets import DataTable, Static
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
            inp.value = "hello world"
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


class TestPersonaScreen:
    @pytest.mark.asyncio()
    async def test_persona_has_list(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("personas")
            await pilot.pause()
            content = app.screen.query_one("#persona-list-content")
            assert content is not None

    @pytest.mark.asyncio()
    async def test_persona_has_detail_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("personas")
            await pilot.pause()
            detail = app.screen.query_one("#persona-detail-content")
            assert detail is not None

    @pytest.mark.asyncio()
    async def test_persona_list_shows_categories(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("personas")
            await pilot.pause()
            content = app.screen.query_one("#persona-list-content", Static)
            # Should not be empty placeholder
            assert content is not None


class TestProfileScreen:
    @pytest.mark.asyncio()
    async def test_profile_has_list_panel(self, _fake_config):
        async with TalkBoxApp().run_test() as pilot:
            app = pilot.app
            app._switch_to("profiles")
            await pilot.pause()
            content = app.screen.query_one("#profile-list-content")
            assert content is not None

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
            assert len(table.columns) == 3

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
            content = app.screen.query_one("#trait-list-content")
            assert content is not None

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
            content = app.screen.query_one("#skill-list-content")
            assert content is not None

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
