import pytest
from yaml12 import format_yaml
from pathlib import Path

from talk_box.personas._loader import (
    PersonaDefinition,
    ModelRecommendation,
    _parse_persona,
    _reset_cache,
    create_persona,
    get_persona,
    list_personas,
    load_persona,
    persona_categories,
    register_persona,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_persona_cache():
    """Reset the persona cache before each test."""
    _reset_cache()
    yield
    _reset_cache()


@pytest.fixture
def sample_persona_dict():
    """A minimal valid persona YAML dict."""
    return {
        "name": "test_persona",
        "display_name": "Test Persona",
        "category": "testing",
        "description": "A test persona.",
        "persona_role": "test assistant",
        "expertise": "testing",
        "task_context": "You help with tests.",
        "critical_constraints": ["Never break tests"],
        "constraints": ["Be accurate"],
        "core_analysis": ["Identify test gaps"],
        "output_format": ["Use bullet points"],
        "final_emphasis": "Tests are important.",
        "avoid_topics": ["production secrets"],
        "tools": ["calculate"],
        "recommended_models": [
            {
                "provider_model": "openai:gpt-4o-mini",
                "context": "Fast and cheap",
                "temperature": 0.3,
            }
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "tags": ["test", "demo"],
        "test_queries": ["Run a test"],
    }


@pytest.fixture
def sample_persona_yaml(tmp_path, sample_persona_dict):
    """Write a sample persona YAML to a temp file."""
    path = tmp_path / "test_persona.yaml"
    path.write_text(format_yaml(sample_persona_dict), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# PersonaDefinition tests
# ---------------------------------------------------------------------------


class TestPersonaDefinition:
    """Tests for PersonaDefinition dataclass."""

    def test_creation_minimal(self):
        persona = PersonaDefinition(
            name="minimal",
            display_name="Minimal",
            category="test",
            description="Minimal persona.",
            persona_role="assistant",
        )
        assert persona.name == "minimal"
        assert persona.critical_constraints == []
        assert persona.avoid_topics == []
        assert persona.tools == []
        assert persona.temperature is None

    def test_creation_full(self, sample_persona_dict):
        persona = _parse_persona(sample_persona_dict)
        assert persona.name == "test_persona"
        assert persona.display_name == "Test Persona"
        assert persona.category == "testing"
        assert persona.persona_role == "test assistant"
        assert persona.expertise == "testing"
        assert persona.temperature == 0.2
        assert persona.max_tokens == 500
        assert "production secrets" in persona.avoid_topics
        assert "calculate" in persona.tools
        assert len(persona.recommended_models) == 1
        assert persona.recommended_models[0].provider_model == "openai:gpt-4o-mini"

    def test_build_system_prompt(self, sample_persona_dict):
        persona = _parse_persona(sample_persona_dict)
        prompt = persona.build_system_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        # The prompt should contain key persona elements
        assert "test assistant" in prompt.lower() or "test" in prompt.lower()

    def test_build_system_prompt_minimal(self):
        persona = PersonaDefinition(
            name="min",
            display_name="Min",
            category="test",
            description="Minimal.",
            persona_role="helper",
        )
        prompt = persona.build_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestModelRecommendation:
    """Tests for ModelRecommendation dataclass."""

    def test_creation(self):
        rec = ModelRecommendation(
            provider_model="openai:gpt-4o",
            context="General use",
            temperature=0.5,
        )
        assert rec.provider_model == "openai:gpt-4o"
        assert rec.context == "General use"
        assert rec.temperature == 0.5

    def test_defaults(self):
        rec = ModelRecommendation(provider_model="ollama:llama3")
        assert rec.context == ""
        assert rec.temperature is None


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


class TestParsing:
    """Tests for YAML parsing."""

    def test_parse_persona_full(self, sample_persona_dict):
        persona = _parse_persona(sample_persona_dict)
        assert persona.name == "test_persona"
        assert len(persona.recommended_models) == 1

    def test_parse_persona_minimal(self):
        data = {
            "name": "bare",
            "persona_role": "helper",
        }
        persona = _parse_persona(data)
        assert persona.name == "bare"
        assert persona.display_name == "Bare"  # auto-generated
        assert persona.category == "general"  # default
        assert persona.constraints == []
        assert persona.recommended_models == []

    def test_parse_persona_auto_display_name(self):
        data = {
            "name": "my_cool_persona",
            "persona_role": "cool helper",
        }
        persona = _parse_persona(data)
        assert persona.display_name == "My Cool Persona"


# ---------------------------------------------------------------------------
# Registry / loader tests
# ---------------------------------------------------------------------------


class TestRegistry:
    """Tests for the persona registry."""

    def test_list_personas_returns_sorted_list(self):
        names = list_personas()
        assert isinstance(names, list)
        assert names == sorted(names)
        # Should have our 23 built-in personas
        assert len(names) >= 20

    def test_get_persona_valid(self):
        persona = get_persona("code_reviewer")
        assert persona.name == "code_reviewer"
        assert persona.category == "technical"
        assert isinstance(persona.persona_role, str)
        assert len(persona.persona_role) > 0

    def test_get_persona_invalid(self):
        with pytest.raises(KeyError, match="not found"):
            get_persona("nonexistent_persona_xyz")

    def test_persona_categories(self):
        cats = persona_categories()
        assert isinstance(cats, dict)
        assert "business" in cats
        assert "technical" in cats
        assert "creative" in cats
        assert "education" in cats
        assert "data" in cats
        # Each category should have personas
        for cat, names in cats.items():
            assert len(names) > 0
            assert names == sorted(names)

    def test_all_personas_load_successfully(self):
        """Every built-in persona YAML should load and produce a valid system prompt."""
        names = list_personas()
        for name in names:
            persona = get_persona(name)
            assert persona.name == name
            assert len(persona.persona_role) > 0
            # Build the system prompt to verify PromptBuilder integration
            prompt = persona.build_system_prompt()
            assert isinstance(prompt, str)
            assert len(prompt) > 50  # Should be substantial

    def test_all_personas_have_required_fields(self):
        """All built-in personas should have core fields populated."""
        names = list_personas()
        for name in names:
            persona = get_persona(name)
            assert persona.display_name, f"{name} missing display_name"
            assert persona.category, f"{name} missing category"
            assert persona.description, f"{name} missing description"
            assert persona.persona_role, f"{name} missing persona_role"
            assert len(persona.recommended_models) > 0, f"{name} missing recommended_models"
            assert len(persona.tags) > 0, f"{name} missing tags"
            assert len(persona.test_queries) > 0, f"{name} missing test_queries"


class TestLoadPersona:
    """Tests for loading personas from arbitrary paths."""

    def test_load_from_file(self, sample_persona_yaml):
        persona = load_persona(sample_persona_yaml)
        assert persona.name == "test_persona"
        assert persona.temperature == 0.2

    def test_load_from_string_path(self, sample_persona_yaml):
        persona = load_persona(str(sample_persona_yaml))
        assert persona.name == "test_persona"

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_persona(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# Built-in persona spot checks
# ---------------------------------------------------------------------------


class TestBuiltInPersonas:
    """Spot checks on specific built-in personas."""

    def test_customer_support_tier1(self):
        p = get_persona("customer_support_tier1")
        assert p.category == "business"
        assert p.temperature == 0.3
        assert len(p.avoid_topics) > 0
        assert len(p.critical_constraints) > 0

    def test_code_reviewer(self):
        p = get_persona("code_reviewer")
        assert p.category == "technical"
        assert p.temperature == 0.1
        assert "text_stats" in p.tools

    def test_tutor(self):
        p = get_persona("tutor")
        assert p.category == "education"
        assert "socratic" in [t.lower() for t in p.tags]

    def test_data_analyst(self):
        p = get_persona("data_analyst")
        assert p.category == "data"
        assert "calculate" in p.tools

    def test_brainstorming_partner(self):
        p = get_persona("brainstorming_partner")
        assert p.category == "creative"
        assert p.temperature == 0.8  # High for creativity


# ---------------------------------------------------------------------------
# ChatBot.persona_pack() integration
# ---------------------------------------------------------------------------


class TestChatBotPersonaPack:
    """Tests for ChatBot.persona_pack() integration."""

    def test_persona_pack_sets_system_prompt(self):
        from talk_box.builder import ChatBot

        bot = ChatBot().persona_pack("code_reviewer")
        assert bot._config["system_prompt"] is not None
        assert len(bot._config["system_prompt"]) > 50

    def test_persona_pack_stores_prompt_builder(self):
        from talk_box.builder import ChatBot
        from talk_box.prompt_builder import PromptBuilder

        bot = ChatBot().persona_pack("code_reviewer")
        assert isinstance(bot._config.get("system_prompt_builder"), PromptBuilder)

    def test_persona_pack_sets_avoid_topics(self):
        from talk_box.builder import ChatBot

        bot = ChatBot().persona_pack("customer_support_tier1")
        assert "avoid_topics" in bot._config
        assert len(bot._config["avoid_topics"]) > 0

    def test_persona_pack_sets_temperature(self):
        from talk_box.builder import ChatBot

        bot = ChatBot().persona_pack("code_reviewer")
        assert bot._config["temperature"] == 0.1

    def test_persona_pack_sets_tools(self):
        from talk_box.builder import ChatBot

        bot = ChatBot().persona_pack("code_reviewer")
        assert "text_stats" in bot._config.get("tools", [])

    def test_persona_pack_stores_metadata(self):
        from talk_box.builder import ChatBot

        bot = ChatBot().persona_pack("sql_helper")
        assert bot._config["persona_pack"] == "sql_helper"
        assert isinstance(bot._config["persona_definition"], PersonaDefinition)

    def test_persona_pack_chaining(self):
        from talk_box.builder import ChatBot

        bot = ChatBot().persona_pack("tutor").temperature(0.6)
        assert bot._config["persona_pack"] == "tutor"
        assert bot._config["temperature"] == 0.6

    def test_persona_pack_invalid_name(self):
        from talk_box.builder import ChatBot

        with pytest.raises(KeyError, match="not found"):
            ChatBot().persona_pack("nonexistent_xyz")

    def test_persona_pack_preserves_existing_avoid_topics(self):
        from talk_box.builder import ChatBot

        bot = ChatBot()
        bot._config["avoid_topics"] = ["custom_topic"]
        bot.persona_pack("customer_support_tier1")
        topics = bot._config["avoid_topics"]
        assert "custom_topic" in topics
        assert len(topics) > 1  # Should have merged

    def test_user_temperature_not_overridden(self):
        from talk_box.builder import ChatBot

        # Set temperature first, then apply persona — user setting should win
        bot = ChatBot()
        bot._config["temperature"] = 0.9
        bot.persona_pack("code_reviewer")
        assert bot._config["temperature"] == 0.9


# ---------------------------------------------------------------------------
# build_prompt_builder() tests
# ---------------------------------------------------------------------------


class TestBuildPromptBuilder:
    """Tests for PersonaDefinition.build_prompt_builder()."""

    def test_returns_prompt_builder(self, sample_persona_dict):
        from talk_box.prompt_builder import PromptBuilder

        persona = _parse_persona(sample_persona_dict)
        builder = persona.build_prompt_builder()
        assert isinstance(builder, PromptBuilder)

    def test_prompt_builder_produces_string(self, sample_persona_dict):
        persona = _parse_persona(sample_persona_dict)
        builder = persona.build_prompt_builder()
        prompt = str(builder)
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_build_system_prompt_matches_builder(self, sample_persona_dict):
        persona = _parse_persona(sample_persona_dict)
        assert persona.build_system_prompt() == str(persona.build_prompt_builder())

    def test_builder_can_be_extended(self, sample_persona_dict):
        persona = _parse_persona(sample_persona_dict)
        builder = persona.build_prompt_builder()
        builder.constraint("Always use type annotations")
        prompt = str(builder)
        assert "type annotations" in prompt.lower()

    def test_all_builtin_personas_produce_builders(self):
        from talk_box.prompt_builder import PromptBuilder

        for name in list_personas():
            persona = get_persona(name)
            builder = persona.build_prompt_builder()
            assert isinstance(builder, PromptBuilder), f"{name} failed"
            assert len(str(builder)) > 50, f"{name} prompt too short"


# ---------------------------------------------------------------------------
# create_persona() tests
# ---------------------------------------------------------------------------


class TestCreatePersona:
    """Tests for create_persona() Python-only API."""

    def test_minimal(self):
        persona = create_persona("test_py", "test helper")
        assert persona.name == "test_py"
        assert persona.persona_role == "test helper"
        assert persona.display_name == "Test Py"
        assert persona.category == "custom"

    def test_full(self):
        persona = create_persona(
            "full_test",
            persona_role="senior advisor",
            display_name="Full Test Persona",
            category="testing",
            description="A complete test.",
            expertise="everything",
            task_context="Help with tests.",
            critical_constraints=["Never lie"],
            constraints=["Be thorough"],
            core_analysis=["Check coverage"],
            output_format=["Bullet points"],
            final_emphasis="Quality matters.",
            avoid_topics=["politics"],
            tools=["calculate"],
            recommended_models=[
                {"provider_model": "openai:gpt-4o", "context": "General"},
            ],
            temperature=0.3,
            max_tokens=1000,
            tags=["test"],
            test_queries=["Run tests"],
        )
        assert persona.display_name == "Full Test Persona"
        assert persona.category == "testing"
        assert persona.expertise == "everything"
        assert persona.critical_constraints == ["Never lie"]
        assert persona.temperature == 0.3
        assert len(persona.recommended_models) == 1
        assert persona.recommended_models[0].provider_model == "openai:gpt-4o"

    def test_builds_prompt(self):
        from talk_box.prompt_builder import PromptBuilder

        persona = create_persona(
            "builder_test",
            persona_role="code reviewer",
            critical_constraints=["Never approve insecure code"],
            final_emphasis="Security first.",
        )
        builder = persona.build_prompt_builder()
        assert isinstance(builder, PromptBuilder)
        prompt = str(builder)
        assert "code reviewer" in prompt.lower()


# ---------------------------------------------------------------------------
# register_persona() tests
# ---------------------------------------------------------------------------


class TestRegisterPersona:
    """Tests for register_persona()."""

    def test_register_and_retrieve(self):
        persona = create_persona("custom_reg", "registered helper")
        register_persona(persona)
        retrieved = get_persona("custom_reg")
        assert retrieved.name == "custom_reg"
        assert retrieved.persona_role == "registered helper"

    def test_register_duplicate_raises(self):
        persona = create_persona("dup_test", "helper")
        register_persona(persona)
        with pytest.raises(ValueError, match="already exists"):
            register_persona(persona)

    def test_registered_persona_in_list(self):
        persona = create_persona("listed_custom", "helper")
        register_persona(persona)
        assert "listed_custom" in list_personas()

    def test_registered_persona_in_categories(self):
        persona = create_persona("cat_custom", "helper", category="my_category")
        register_persona(persona)
        cats = persona_categories()
        assert "my_category" in cats
        assert "cat_custom" in cats["my_category"]

    def test_persona_pack_with_registered(self):
        from talk_box.builder import ChatBot

        persona = create_persona(
            "pack_custom",
            persona_role="custom pack helper",
            temperature=0.4,
            avoid_topics=["secrets"],
        )
        register_persona(persona)
        bot = ChatBot().persona_pack("pack_custom")
        assert bot._config["persona_pack"] == "pack_custom"
        assert bot._config["temperature"] == 0.4
