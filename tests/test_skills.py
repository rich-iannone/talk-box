import pytest

from talk_box.skills import (
    SkillDefinition,
    _reset_cache,
    create_skill,
    discover_skills,
    get_skill,
    list_skills,
    load_skill,
    register_skill,
    skill_categories,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_cache():
    """Reset the skill cache before each test so registrations don't leak."""
    _reset_cache()
    yield
    _reset_cache()


def _make_skill(**kwargs):
    """Create a skill with sensible defaults."""
    defaults = {"name": "test_skill", "description": "A test skill."}
    defaults.update(kwargs)
    return create_skill(**defaults)


# ---------------------------------------------------------------------------
# SkillDefinition
# ---------------------------------------------------------------------------


class TestSkillDefinition:
    def test_fields(self):
        skill = SkillDefinition(
            name="s",
            display_name="S",
            category="cat",
            description="desc",
            instructions="do stuff",
            constraints=["c1"],
            tools=["t1"],
            tags=["tag1"],
            metadata={"v": 1},
        )
        assert skill.name == "s"
        assert skill.display_name == "S"
        assert skill.category == "cat"
        assert skill.description == "desc"
        assert skill.instructions == "do stuff"
        assert skill.constraints == ["c1"]
        assert skill.tools == ["t1"]
        assert skill.tags == ["tag1"]
        assert skill.metadata == {"v": 1}

    def test_defaults(self):
        skill = SkillDefinition(name="s", display_name="S", category="c", description="d")
        assert skill.instructions == ""
        assert skill.constraints == []
        assert skill.tools == []
        assert skill.tags == []
        assert skill.metadata == {}


# ---------------------------------------------------------------------------
# create_skill
# ---------------------------------------------------------------------------


class TestCreateSkill:
    def test_minimal(self):
        skill = create_skill("my_skill")
        assert skill.name == "my_skill"
        assert skill.display_name == "My Skill"
        assert skill.category == "custom"

    def test_all_params(self):
        skill = create_skill(
            "analyzer",
            display_name="Analyzer Pro",
            category="data",
            description="Analyze things",
            instructions="Step 1: analyze",
            constraints=["Be accurate"],
            tools=["reader"],
            tags=["data"],
            metadata={"version": "2.0"},
        )
        assert skill.display_name == "Analyzer Pro"
        assert skill.category == "data"
        assert skill.description == "Analyze things"
        assert skill.instructions == "Step 1: analyze"
        assert skill.constraints == ["Be accurate"]
        assert skill.tools == ["reader"]
        assert skill.tags == ["data"]
        assert skill.metadata == {"version": "2.0"}

    def test_display_name_auto_generated(self):
        skill = create_skill("my_cool_skill")
        assert skill.display_name == "My Cool Skill"


# ---------------------------------------------------------------------------
# register / get / list
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        skill = _make_skill()
        register_skill(skill)
        assert get_skill("test_skill") is skill

    def test_get_missing_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_skill("nonexistent")

    def test_list_skills_empty_custom(self):
        # Built-in packs are loaded, so list should include them
        names = list_skills()
        assert isinstance(names, list)

    def test_register_overwrites(self):
        s1 = _make_skill(description="v1")
        s2 = _make_skill(description="v2")
        register_skill(s1)
        register_skill(s2)
        assert get_skill("test_skill").description == "v2"

    def test_list_includes_registered(self):
        register_skill(_make_skill(name="custom_one"))
        names = list_skills()
        assert "custom_one" in names

    def test_list_is_sorted(self):
        register_skill(_make_skill(name="zebra"))
        register_skill(_make_skill(name="alpha"))
        names = list_skills()
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Built-in packs
# ---------------------------------------------------------------------------


class TestBuiltInPacks:
    def test_sql_analysis_exists(self):
        skill = get_skill("sql_analysis")
        assert skill.category == "data"
        assert skill.instructions != ""

    def test_summarization_exists(self):
        skill = get_skill("summarization")
        assert skill.category == "writing"

    def test_code_explanation_exists(self):
        skill = get_skill("code_explanation")
        assert skill.category == "engineering"

    def test_api_documentation_exists(self):
        skill = get_skill("api_documentation")
        assert skill.category == "engineering"

    def test_data_cleaning_exists(self):
        skill = get_skill("data_cleaning")
        assert skill.category == "data"

    def test_five_built_in_packs(self):
        # At minimum, the 5 we shipped
        names = list_skills()
        expected = [
            "api_documentation",
            "code_explanation",
            "data_cleaning",
            "sql_analysis",
            "summarization",
        ]
        for name in expected:
            assert name in names


# ---------------------------------------------------------------------------
# skill_categories
# ---------------------------------------------------------------------------


class TestSkillCategories:
    def test_returns_grouped(self):
        cats = skill_categories()
        assert "data" in cats
        assert "engineering" in cats
        assert "writing" in cats

    def test_sorted_within_categories(self):
        cats = skill_categories()
        for names in cats.values():
            assert names == sorted(names)

    def test_includes_registered(self):
        register_skill(_make_skill(name="my_custom", category="testing"))
        cats = skill_categories()
        assert "testing" in cats
        assert "my_custom" in cats["testing"]


# ---------------------------------------------------------------------------
# load_skill
# ---------------------------------------------------------------------------


class TestLoadSkill:
    def test_load_from_file(self, tmp_path):
        yaml_content = """
name: loaded_skill
display_name: "Loaded Skill"
category: custom
description: "A skill loaded from a file."
instructions: "Do the thing."
constraints:
  - "Be careful"
tags:
  - loaded
"""
        path = tmp_path / "my_skill.yaml"
        path.write_text(yaml_content)
        skill = load_skill(path)
        assert skill.name == "loaded_skill"
        assert skill.display_name == "Loaded Skill"
        assert skill.instructions == "Do the thing."
        assert skill.constraints == ["Be careful"]
        assert skill.tags == ["loaded"]

    def test_load_missing_file(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_skill("/nonexistent/path.yaml")

    def test_load_minimal_yaml(self, tmp_path):
        yaml_content = """
name: minimal
"""
        path = tmp_path / "minimal.yaml"
        path.write_text(yaml_content)
        skill = load_skill(path)
        assert skill.name == "minimal"
        assert skill.display_name == "Minimal"
        assert skill.category == "general"
        assert skill.constraints == []

    def test_load_with_metadata(self, tmp_path):
        yaml_content = """
name: versioned
metadata:
  version: "3.0"
  author: "test"
"""
        path = tmp_path / "versioned.yaml"
        path.write_text(yaml_content)
        skill = load_skill(path)
        assert skill.metadata == {"version": "3.0", "author": "test"}


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        import talk_box

        for name in [
            "SkillDefinition",
            "create_skill",
            "get_skill",
            "list_skills",
            "register_skill",
            "load_skill",
            "skill_categories",
        ]:
            assert hasattr(talk_box, name)

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "SkillDefinition",
            "create_skill",
            "discover_skills",
            "get_skill",
            "list_skills",
            "register_skill",
            "load_skill",
            "skill_categories",
        ]:
            assert name in talk_box.__all__


class TestDiscoverSkills:
    """Tests for discover_skills() SKILL.md scanning."""

    def test_discover_from_directory(self, tmp_path):
        from talk_box.skills import _reset_cache, discover_skills

        _reset_cache()

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "SKILL.md").write_text(
            "---\n"
            "name: test_skill\n"
            "category: testing\n"
            "description: A test skill\n"
            "---\n\n"
            "# Instructions\n\n"
            "Do something useful.\n"
        )

        found = discover_skills(str(skills_dir), scan_cwd=False)
        assert len(found) == 1
        assert found[0].name == "test_skill"
        assert found[0].category == "testing"
        assert "Do something useful" in found[0].instructions

    def test_discover_with_constraints(self, tmp_path):
        from talk_box.skills import _reset_cache, discover_skills

        _reset_cache()

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "SKILL.md").write_text(
            "---\n"
            "name: constrained_skill\n"
            "description: Has constraints\n"
            "constraints:\n"
            "  - Be concise\n"
            "  - Use examples\n"
            "tags:\n"
            "  - writing\n"
            "---\n\n"
            "Write clearly.\n"
        )

        found = discover_skills(str(skills_dir), scan_cwd=False)
        assert len(found) == 1
        assert found[0].constraints == ["Be concise", "Use examples"]
        assert found[0].tags == ["writing"]

    def test_discover_nested_skill_md(self, tmp_path):
        from talk_box.skills import _reset_cache, discover_skills

        _reset_cache()

        nested = tmp_path / "skills" / "sub"
        nested.mkdir(parents=True)
        (nested / "SKILL.md").write_text(
            "---\nname: nested_skill\ndescription: Found nested\n---\n\nNested instructions.\n"
        )

        found = discover_skills(str(tmp_path / "skills"), scan_cwd=False)
        assert len(found) == 1
        assert found[0].name == "nested_skill"

    def test_discover_empty_dir(self, tmp_path):
        from talk_box.skills import _reset_cache, discover_skills

        _reset_cache()

        empty = tmp_path / "empty"
        empty.mkdir()
        found = discover_skills(str(empty), scan_cwd=False)
        assert found == []

    def test_discover_no_frontmatter_skipped(self, tmp_path):
        from talk_box.skills import _reset_cache, discover_skills

        _reset_cache()

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "SKILL.md").write_text("No frontmatter here.\n")

        found = discover_skills(str(skills_dir), scan_cwd=False)
        assert found == []

    def test_discover_stores_source_path(self, tmp_path):
        from talk_box.skills import _reset_cache, discover_skills

        _reset_cache()

        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "SKILL.md").write_text(
            "---\nname: sourced\ndescription: Has source\n---\n\nBody.\n"
        )

        found = discover_skills(str(skills_dir), scan_cwd=False)
        assert "source" in found[0].metadata
        assert "SKILL.md" in found[0].metadata["source"]


class TestWrapCallable:
    """Tests for wrap_callable() cross-tool orchestration."""

    def test_wrap_simple_function(self):
        from talk_box.skills import _reset_cache, wrap_callable

        _reset_cache()

        def word_count(text: str) -> int:
            return len(text.split())

        skill = wrap_callable(word_count, description="Count words")
        assert skill.name == "word_count"
        assert skill.description == "Count words"
        assert skill.category == "wrapped"
        assert "wrapped_callable" in skill.metadata
        assert "params_schema" in skill.metadata

    def test_wrap_uses_custom_name(self):
        from talk_box.skills import _reset_cache, wrap_callable

        _reset_cache()

        def my_fn(x: int) -> int:
            return x * 2

        skill = wrap_callable(my_fn, name="doubler", description="Double a number")
        assert skill.name == "doubler"

    def test_wrap_extracts_params_schema(self):
        from talk_box.skills import _reset_cache, wrap_callable

        _reset_cache()

        def analyze(text: str, verbose: bool = False, limit: int = 10) -> dict:
            return {}

        skill = wrap_callable(analyze)
        schema = skill.metadata["params_schema"]
        assert schema["properties"]["text"]["type"] == "string"
        assert schema["properties"]["verbose"]["type"] == "boolean"
        assert schema["properties"]["verbose"]["default"] is False
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["limit"]["default"] == 10
        assert "text" in schema["required"]
        assert "verbose" not in schema["required"]

    def test_wrap_uses_docstring(self):
        from talk_box.skills import _reset_cache, wrap_callable

        _reset_cache()

        def helper(x: str) -> str:
            """Transform the input string."""
            return x.upper()

        skill = wrap_callable(helper)
        assert "Transform the input string" in skill.instructions

    def test_wrap_custom_schema(self):
        from talk_box.skills import _reset_cache, wrap_callable

        _reset_cache()

        custom = {"type": "object", "properties": {"q": {"type": "string"}}}
        skill = wrap_callable(lambda q: q, name="custom", params_schema=custom)
        assert skill.metadata["params_schema"] == custom

    def test_wrap_registers_in_cache(self):
        from talk_box.skills import _reset_cache, get_skill, wrap_callable

        _reset_cache()

        def registered_fn(a: str) -> str:
            return a

        wrap_callable(registered_fn, register=True)
        assert get_skill("registered_fn") is not None

    def test_wrap_no_register(self):
        from talk_box.skills import _reset_cache, get_skill, wrap_callable

        _reset_cache()

        def unregistered_fn(a: str) -> str:
            return a

        wrap_callable(unregistered_fn, register=False)
        with pytest.raises(KeyError):
            get_skill("unregistered_fn")

    def test_all_contains_wrap_callable(self):
        import talk_box

        assert "wrap_callable" in talk_box.__all__


class TestDiscoverEntryPointSkills:
    """Tests for discover_entry_point_skills."""

    def test_discover_returns_list(self):
        from talk_box.skills import _reset_cache, discover_entry_point_skills

        _reset_cache()
        # No third-party packages with talk_box.skills entry points installed,
        # so this should return an empty list without error
        result = discover_entry_point_skills()
        assert isinstance(result, list)

    def test_discover_entry_point_skills_in_all(self):
        import talk_box

        assert "discover_entry_point_skills" in talk_box.__all__

    def test_discover_registers_in_cache(self, monkeypatch):
        """Mock entry points to verify skills are cached."""
        from talk_box.skills import (
            SkillDefinition,
            _reset_cache,
            discover_entry_point_skills,
            get_skill,
        )

        _reset_cache()

        mock_skill = SkillDefinition(
            name="ep_skill",
            display_name="EP Skill",
            category="test",
            description="From entry point",
            instructions="Do the thing",
            constraints=[],
            tools=[],
            tags=[],
            metadata={},
        )

        class FakeEP:
            def load(self):
                return lambda: mock_skill

        class FakeEPs:
            def select(self, group):
                return [FakeEP()]

        monkeypatch.setattr(
            "talk_box.skills.entry_points",
            lambda: FakeEPs(),
        )

        result = discover_entry_point_skills()
        assert len(result) == 1
        assert result[0].name == "ep_skill"
        assert get_skill("ep_skill").description == "From entry point"

    def test_discover_handles_list_return(self, monkeypatch):
        """Entry point can return a list of skills."""
        from talk_box.skills import (
            SkillDefinition,
            _reset_cache,
            discover_entry_point_skills,
        )

        _reset_cache()

        skills = [
            SkillDefinition(
                name=f"ep_{i}",
                display_name=f"EP {i}",
                category="test",
                description="",
                instructions="",
                constraints=[],
                tools=[],
                tags=[],
                metadata={},
            )
            for i in range(3)
        ]

        class FakeEP:
            def load(self):
                return lambda: skills

        class FakeEPs:
            def select(self, group):
                return [FakeEP()]

        monkeypatch.setattr(
            "talk_box.skills.entry_points",
            lambda: FakeEPs(),
        )

        result = discover_entry_point_skills()
        assert len(result) == 3

    def test_discover_skips_broken_entry_points(self, monkeypatch):
        """Broken entry points are skipped without error."""
        from talk_box.skills import _reset_cache, discover_entry_point_skills

        _reset_cache()

        class FakeEP:
            def load(self):
                raise ImportError("broken package")

        class FakeEPs:
            def select(self, group):
                return [FakeEP()]

        monkeypatch.setattr(
            "talk_box.skills.entry_points",
            lambda: FakeEPs(),
        )

        result = discover_entry_point_skills()
        assert result == []


# ---------------------------------------------------------------------------
# SKILL.md file discovery tests
# ---------------------------------------------------------------------------


class TestDiscoverSkillsMd:
    def test_discover_from_directory(self, tmp_path):
        """discover_skills() finds SKILL.md files in search paths."""
        _reset_cache()

        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: test_skill\ndisplay_name: Test Skill\n"
            "category: testing\ndescription: A test skill\n---\n\n"
            "# Instructions\n\nDo testing things.\n",
            encoding="utf-8",
        )

        result = discover_skills(tmp_path, scan_cwd=False)
        assert len(result) >= 1
        names = [s.name for s in result]
        assert "test_skill" in names

        skill = get_skill("test_skill")
        assert skill.category == "testing"
        assert "testing things" in skill.instructions

    def test_discover_skips_bad_files(self, tmp_path):
        """Malformed SKILL.md files are silently skipped."""
        _reset_cache()

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "SKILL.md").write_text("not valid frontmatter", encoding="utf-8")

        result = discover_skills(tmp_path, scan_cwd=False)
        assert result == []

    def test_first_party_skill_md_parseable(self):
        """The first-party SKILL.md files in skills/ can be parsed."""
        _reset_cache()
        from pathlib import Path

        skills_dir = Path(__file__).parent.parent / "skills"
        if not skills_dir.is_dir():
            pytest.skip("No skills/ directory in project root")

        result = discover_skills(skills_dir, scan_cwd=False)
        assert len(result) >= 1
        names = [s.name for s in result]
        # At least one of our first-party skills should be found
        assert any(n in names for n in ["great_tables", "pointblank", "shiny_python"])
