import pytest

from talk_box.personas import create_persona
from talk_box.traits import (
    TraitDefinition,
    _reset_cache,
    apply_trait,
    create_trait,
    get_trait,
    list_traits,
    load_trait,
    register_trait,
    trait_categories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_persona():
    """Create a minimal persona for testing."""
    return create_persona(
        "test_persona",
        persona_role="test assistant",
        description="A test persona.",
        expertise="testing",
        constraints=["Be helpful"],
        critical_constraints=["Never lie"],
        avoid_topics=["politics"],
        tools=["text_stats"],
        tags=["test"],
        temperature=0.7,
        task_context="You help with tests.",
        output_format=["Use bullet points"],
        final_emphasis="Be accurate.",
    )


# ---------------------------------------------------------------------------
# TraitDefinition
# ---------------------------------------------------------------------------


class TestTraitDefinition:
    def test_basic_creation(self):
        t = TraitDefinition(name="my_trait")
        assert t.name == "my_trait"
        assert t.display_name == "My Trait"
        assert t.category == "general"
        assert t.constraints == []

    def test_display_name_auto(self):
        t = TraitDefinition(name="security_focused")
        assert t.display_name == "Security Focused"

    def test_display_name_explicit(self):
        t = TraitDefinition(name="my_trait", display_name="Custom Name")
        assert t.display_name == "Custom Name"

    def test_all_fields(self):
        t = TraitDefinition(
            name="full",
            display_name="Full Trait",
            category="test",
            description="Testing all fields.",
            constraints=["c1"],
            critical_constraints=["cc1"],
            expertise_extra="extra exp",
            avoid_topics=["bad"],
            tools=["tool1"],
            tags=["tag1"],
            temperature=0.3,
            task_context_extra="extra context",
            output_format=["format1"],
            final_emphasis="emphasis",
            metadata={"key": "val"},
        )
        assert t.constraints == ["c1"]
        assert t.critical_constraints == ["cc1"]
        assert t.expertise_extra == "extra exp"
        assert t.temperature == 0.3
        assert t.metadata == {"key": "val"}


# ---------------------------------------------------------------------------
# apply_trait
# ---------------------------------------------------------------------------


class TestApplyTrait:
    def test_adds_constraints(self):
        persona = _base_persona()
        trait = create_trait("t", constraints=["New constraint"])
        result = apply_trait(persona, trait)
        assert "New constraint" in result.constraints
        assert "Be helpful" in result.constraints

    def test_adds_critical_constraints(self):
        persona = _base_persona()
        trait = create_trait("t", critical_constraints=["Critical new"])
        result = apply_trait(persona, trait)
        assert "Critical new" in result.critical_constraints
        assert "Never lie" in result.critical_constraints

    def test_adds_avoid_topics(self):
        persona = _base_persona()
        trait = create_trait("t", avoid_topics=["violence"])
        result = apply_trait(persona, trait)
        assert "violence" in result.avoid_topics
        assert "politics" in result.avoid_topics

    def test_adds_tools(self):
        persona = _base_persona()
        trait = create_trait("t", tools=["new_tool"])
        result = apply_trait(persona, trait)
        assert "new_tool" in result.tools
        assert "text_stats" in result.tools

    def test_adds_tags(self):
        persona = _base_persona()
        trait = create_trait("t", tags=["new_tag"])
        result = apply_trait(persona, trait)
        assert "new_tag" in result.tags
        assert "test" in result.tags

    def test_adds_output_format(self):
        persona = _base_persona()
        trait = create_trait("t", output_format=["Use headings"])
        result = apply_trait(persona, trait)
        assert "Use headings" in result.output_format
        assert "Use bullet points" in result.output_format

    def test_appends_expertise(self):
        persona = _base_persona()
        trait = create_trait("t", expertise_extra="security")
        result = apply_trait(persona, trait)
        assert result.expertise == "testing, security"

    def test_appends_expertise_empty(self):
        persona = create_persona("bare", persona_role="bare")
        trait = create_trait("t", expertise_extra="security")
        result = apply_trait(persona, trait)
        assert result.expertise == "security"

    def test_appends_task_context(self):
        persona = _base_persona()
        trait = create_trait("t", task_context_extra="Also do X.")
        result = apply_trait(persona, trait)
        assert result.task_context == "You help with tests. Also do X."

    def test_appends_task_context_empty(self):
        persona = create_persona("bare", persona_role="bare")
        trait = create_trait("t", task_context_extra="Do X.")
        result = apply_trait(persona, trait)
        assert result.task_context == "Do X."

    def test_overrides_temperature(self):
        persona = _base_persona()
        assert persona.temperature == 0.7
        trait = create_trait("t", temperature=0.2)
        result = apply_trait(persona, trait)
        assert result.temperature == 0.2

    def test_no_temperature_override_when_none(self):
        persona = _base_persona()
        trait = create_trait("t", constraints=["x"])
        result = apply_trait(persona, trait)
        assert result.temperature == 0.7

    def test_overrides_final_emphasis(self):
        persona = _base_persona()
        trait = create_trait("t", final_emphasis="New emphasis.")
        result = apply_trait(persona, trait)
        assert result.final_emphasis == "New emphasis."

    def test_no_final_emphasis_override_when_empty(self):
        persona = _base_persona()
        trait = create_trait("t", constraints=["x"])
        result = apply_trait(persona, trait)
        assert result.final_emphasis == "Be accurate."

    def test_does_not_mutate_original(self):
        persona = _base_persona()
        original_constraints = list(persona.constraints)
        trait = create_trait("t", constraints=["Added"])
        apply_trait(persona, trait)
        assert persona.constraints == original_constraints

    def test_deduplicates_lists(self):
        persona = _base_persona()
        trait = create_trait("t", constraints=["Be helpful"])  # Already exists
        result = apply_trait(persona, trait)
        assert result.constraints.count("Be helpful") == 1

    def test_stack_multiple_traits(self):
        persona = _base_persona()
        t1 = create_trait("t1", constraints=["C1"], tags=["tag_a"])
        t2 = create_trait("t2", constraints=["C2"], tags=["tag_b"])
        result = apply_trait(apply_trait(persona, t1), t2)
        assert "C1" in result.constraints
        assert "C2" in result.constraints
        assert "tag_a" in result.tags
        assert "tag_b" in result.tags


# ---------------------------------------------------------------------------
# create_trait
# ---------------------------------------------------------------------------


class TestCreateTrait:
    def test_minimal(self):
        t = create_trait("minimal")
        assert t.name == "minimal"
        assert t.category == "custom"
        assert t.constraints == []

    def test_all_params(self):
        t = create_trait(
            "full",
            display_name="Full",
            category="test",
            description="desc",
            constraints=["c"],
            critical_constraints=["cc"],
            expertise_extra="exp",
            avoid_topics=["bad"],
            tools=["tool"],
            tags=["tag"],
            temperature=0.5,
            task_context_extra="ctx",
            output_format=["fmt"],
            final_emphasis="emph",
            metadata={"k": "v"},
        )
        assert t.display_name == "Full"
        assert t.description == "desc"
        assert t.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# Registry: register, get, list, categories
# ---------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self):
        _reset_cache()

    def teardown_method(self):
        _reset_cache()

    def test_register_and_get(self):
        t = create_trait("custom_trait", description="Custom.")
        register_trait(t)
        retrieved = get_trait("custom_trait")
        assert retrieved.description == "Custom."

    def test_register_duplicate_raises(self):
        t = create_trait("dup")
        register_trait(t)
        with pytest.raises(ValueError, match="already exists"):
            register_trait(t)

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_trait("nonexistent_trait_xyz")

    def test_list_traits_includes_builtins(self):
        names = list_traits()
        assert "concise" in names
        assert "verbose" in names
        assert "formal" in names
        assert "security_focused" in names
        assert "junior_friendly" in names

    def test_list_traits_sorted(self):
        names = list_traits()
        assert names == sorted(names)

    def test_trait_categories_has_tone(self):
        cats = trait_categories()
        assert "tone" in cats
        assert "concise" in cats["tone"]

    def test_trait_categories_has_compliance(self):
        cats = trait_categories()
        assert "compliance" in cats
        assert "security_focused" in cats["compliance"]

    def test_register_custom_appears_in_list(self):
        register_trait(create_trait("zzz_custom"))
        assert "zzz_custom" in list_traits()


# ---------------------------------------------------------------------------
# Built-in traits
# ---------------------------------------------------------------------------


class TestBuiltinTraits:
    def setup_method(self):
        _reset_cache()

    def teardown_method(self):
        _reset_cache()

    def test_security_focused_has_constraints(self):
        t = get_trait("security_focused")
        assert len(t.critical_constraints) > 0
        assert len(t.constraints) > 0
        assert t.expertise_extra != ""

    def test_junior_friendly_has_constraints(self):
        t = get_trait("junior_friendly")
        assert len(t.constraints) > 0
        assert t.task_context_extra != ""

    def test_concise_has_final_emphasis(self):
        t = get_trait("concise")
        assert t.final_emphasis != ""
        assert len(t.constraints) > 0

    def test_verbose_has_final_emphasis(self):
        t = get_trait("verbose")
        assert t.final_emphasis != ""

    def test_formal_has_avoid_topics(self):
        t = get_trait("formal")
        assert len(t.avoid_topics) > 0

    def test_apply_builtin_to_persona(self):
        persona = _base_persona()
        sec = get_trait("security_focused")
        result = apply_trait(persona, sec)
        # Security constraints merged in
        assert any("owasp" in c.lower() or "validation" in c.lower() for c in result.constraints)
        assert any(
            "security" in c.lower() or "vulnerabilit" in c.lower()
            for c in result.critical_constraints
        )

    def test_stack_concise_and_security(self):
        persona = _base_persona()
        result = apply_trait(
            apply_trait(persona, get_trait("concise")), get_trait("security_focused")
        )
        assert result.final_emphasis != "Be accurate."  # concise overwrote it
        assert any("owasp" in c.lower() or "validation" in c.lower() for c in result.constraints)


# ---------------------------------------------------------------------------
# load_trait
# ---------------------------------------------------------------------------


class TestLoadTrait:
    def test_load_from_file(self, tmp_path):
        yaml_content = """
name: loaded_trait
display_name: "Loaded Trait"
category: test
description: "A trait loaded from a file."
constraints:
  - "Be loaded"
"""
        f = tmp_path / "loaded.yaml"
        f.write_text(yaml_content)
        t = load_trait(f)
        assert t.name == "loaded_trait"
        assert t.constraints == ["Be loaded"]

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_trait("/nonexistent/path/trait.yaml")


# ---------------------------------------------------------------------------
# Integration: trait + persona → Agent
# ---------------------------------------------------------------------------


class TestTraitAgentIntegration:
    def test_agent_with_traited_persona(self):
        from talk_box.agent import Agent

        persona = _base_persona()
        traited = apply_trait(persona, create_trait("sec", constraints=["Check security"]))
        agent = Agent(name="secure_agent", persona=traited)
        assert "Check security" in agent.persona.constraints
        # Agent's chatbot should be configured from the traited persona
        assert agent.chatbot is not None


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        import talk_box

        for name in [
            "TraitDefinition",
            "apply_trait",
            "create_trait",
            "get_trait",
            "list_traits",
            "register_trait",
            "load_trait",
            "trait_categories",
        ]:
            assert hasattr(talk_box, name), f"talk_box.{name} not found"

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "TraitDefinition",
            "apply_trait",
            "create_trait",
            "get_trait",
            "list_traits",
            "register_trait",
            "load_trait",
            "trait_categories",
        ]:
            assert name in talk_box.__all__, f"{name} not in __all__"
