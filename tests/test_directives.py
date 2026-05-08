"""Tests for talk_box.directives."""

from __future__ import annotations

import pytest

from talk_box.directives import (
    ApplyResult,
    ConfidentialDirective,
    ContextDirective,
    ExpiresDirective,
    ParsedDirectives,
    RelatesToDirective,
    apply_directives,
    parse_directives,
    strip_directives,
)
from talk_box.knowledge_graph import KnowledgeGraph, Node, NodeType


# ---------------------------------------------------------------------------
# ContextDirective
# ---------------------------------------------------------------------------


class TestContextDirective:
    def test_creation(self):
        d = ContextDirective(value="Q3 Planning")
        assert d.value == "Q3 Planning"
        assert d.line == 0

    def test_with_line(self):
        d = ContextDirective(value="test", line=5)
        assert d.line == 5

    def test_frozen(self):
        d = ContextDirective(value="x")
        with pytest.raises(AttributeError):
            d.value = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RelatesToDirective
# ---------------------------------------------------------------------------


class TestRelatesToDirective:
    def test_creation(self):
        d = RelatesToDirective(target="Project Alpha")
        assert d.target == "Project Alpha"

    def test_frozen(self):
        d = RelatesToDirective(target="x")
        with pytest.raises(AttributeError):
            d.target = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ConfidentialDirective
# ---------------------------------------------------------------------------


class TestConfidentialDirective:
    def test_creation(self):
        d = ConfidentialDirective()
        assert d.line == 0

    def test_with_line(self):
        d = ConfidentialDirective(line=10)
        assert d.line == 10


# ---------------------------------------------------------------------------
# ExpiresDirective
# ---------------------------------------------------------------------------


class TestExpiresDirective:
    def test_creation(self):
        d = ExpiresDirective(date_str="2026-12-31")
        assert d.date_str == "2026-12-31"

    def test_is_expired_past_date(self):
        d = ExpiresDirective(date_str="2020-01-01")
        assert d.is_expired() is True

    def test_is_expired_future_date(self):
        d = ExpiresDirective(date_str="2099-12-31")
        assert d.is_expired() is False

    def test_is_expired_with_now(self):
        from datetime import datetime

        # Use a known timestamp
        jan_2025 = datetime(2025, 1, 15).timestamp()
        d = ExpiresDirective(date_str="2025-06-01")
        assert d.is_expired(now=jan_2025) is False

        jul_2025 = datetime(2025, 7, 1).timestamp()
        assert d.is_expired(now=jul_2025) is True

    def test_is_expired_invalid_date(self):
        d = ExpiresDirective(date_str="not-a-date")
        assert d.is_expired() is False


# ---------------------------------------------------------------------------
# ParsedDirectives
# ---------------------------------------------------------------------------


class TestParsedDirectives:
    def test_empty(self):
        d = ParsedDirectives()
        assert d.directive_count == 0
        assert d.is_confidential is False
        assert d.is_expired is False
        assert d.context_values == []
        assert d.relates_to_targets == []
        assert d.all_directives == []

    def test_is_confidential(self):
        d = ParsedDirectives(confidentials=[ConfidentialDirective()])
        assert d.is_confidential is True

    def test_context_values(self):
        d = ParsedDirectives(
            contexts=[
                ContextDirective(value="A", line=1),
                ContextDirective(value="B", line=2),
            ]
        )
        assert d.context_values == ["A", "B"]

    def test_relates_to_targets(self):
        d = ParsedDirectives(relates_to=[RelatesToDirective(target="X", line=1)])
        assert d.relates_to_targets == ["X"]

    def test_is_expired(self):
        d = ParsedDirectives(expires=[ExpiresDirective(date_str="2020-01-01", line=1)])
        assert d.is_expired is True

    def test_not_expired(self):
        d = ParsedDirectives(expires=[ExpiresDirective(date_str="2099-12-31", line=1)])
        assert d.is_expired is False

    def test_directive_count(self):
        d = ParsedDirectives(
            contexts=[ContextDirective(value="a", line=1)],
            relates_to=[RelatesToDirective(target="b", line=2)],
            confidentials=[ConfidentialDirective(line=3)],
            expires=[ExpiresDirective(date_str="2026-01-01", line=4)],
        )
        assert d.directive_count == 4

    def test_all_directives_sorted(self):
        d = ParsedDirectives(
            contexts=[ContextDirective(value="a", line=3)],
            confidentials=[ConfidentialDirective(line=1)],
            expires=[ExpiresDirective(date_str="2026-01-01", line=5)],
        )
        items = d.all_directives
        lines = [i.line for i in items]
        assert lines == [1, 3, 5]

    def test_to_metadata_full(self):
        d = ParsedDirectives(
            contexts=[ContextDirective(value="topic1", line=1)],
            relates_to=[RelatesToDirective(target="X", line=2)],
            confidentials=[ConfidentialDirective(line=3)],
            expires=[ExpiresDirective(date_str="2026-12-31", line=4)],
        )
        meta = d.to_metadata()
        assert meta["_contexts"] == ["topic1"]
        assert meta["_relates_to"] == ["X"]
        assert meta["_confidential"] is True
        assert meta["_expires"] == ["2026-12-31"]

    def test_to_metadata_empty(self):
        d = ParsedDirectives()
        meta = d.to_metadata()
        assert meta == {}

    def test_repr(self):
        d = ParsedDirectives(
            contexts=[ContextDirective(value="a", line=1)],
            confidentials=[ConfidentialDirective(line=2)],
        )
        r = repr(d)
        assert "contexts=1" in r
        assert "confidential=True" in r


# ---------------------------------------------------------------------------
# parse_directives
# ---------------------------------------------------------------------------


class TestParseDirectives:
    def test_context(self):
        d = parse_directives("@context: Q3 Planning")
        assert len(d.contexts) == 1
        assert d.contexts[0].value == "Q3 Planning"

    def test_relates_to(self):
        d = parse_directives("@relates-to: Project Alpha")
        assert len(d.relates_to) == 1
        assert d.relates_to[0].target == "Project Alpha"

    def test_confidential(self):
        d = parse_directives("@confidential")
        assert d.is_confidential is True

    def test_expires(self):
        d = parse_directives("@expires: 2026-12-31")
        assert len(d.expires) == 1
        assert d.expires[0].date_str == "2026-12-31"

    def test_multiple_directives(self):
        text = """Meeting notes.
@context: Q3 Planning
@relates-to: Project Alpha
@confidential
@expires: 2026-12-31
More content here.
"""
        d = parse_directives(text)
        assert d.directive_count == 4
        assert d.context_values == ["Q3 Planning"]
        assert d.relates_to_targets == ["Project Alpha"]
        assert d.is_confidential is True

    def test_no_directives(self):
        d = parse_directives("Just a normal note with no directives.")
        assert d.directive_count == 0

    def test_empty_text(self):
        d = parse_directives("")
        assert d.directive_count == 0

    def test_case_insensitive(self):
        d = parse_directives("@CONTEXT: Test")
        assert len(d.contexts) == 1
        assert d.contexts[0].value == "Test"

    def test_leading_whitespace(self):
        d = parse_directives("   @context: indented")
        assert len(d.contexts) == 1
        assert d.contexts[0].value == "indented"

    def test_multiple_contexts(self):
        text = "@context: Topic A\n@context: Topic B"
        d = parse_directives(text)
        assert len(d.contexts) == 2
        assert d.context_values == ["Topic A", "Topic B"]

    def test_multiple_relates_to(self):
        text = "@relates-to: X\n@relates-to: Y"
        d = parse_directives(text)
        assert len(d.relates_to) == 2

    def test_line_numbers(self):
        text = "Line 1\n@context: test\nLine 3\n@confidential"
        d = parse_directives(text)
        assert d.contexts[0].line == 2
        assert d.confidentials[0].line == 4

    def test_directive_in_middle_of_text(self):
        text = "Start of note.\n@context: Important Topic\nEnd of note."
        d = parse_directives(text)
        assert d.context_values == ["Important Topic"]

    def test_inline_directive_not_matched(self):
        # Directives must be at the start of a line
        d = parse_directives("This is not @confidential at all")
        # Note: after stripping, this becomes "This is not @confidential at all"
        # which doesn't match ^@confidential$ — correct behavior
        assert d.is_confidential is False


# ---------------------------------------------------------------------------
# strip_directives
# ---------------------------------------------------------------------------


class TestStripDirectives:
    def test_strips_all_directive_types(self):
        text = "Hello\n@context: test\n@relates-to: X\n@confidential\n@expires: 2026-01-01\nWorld"
        result = strip_directives(text)
        assert "@context" not in result
        assert "@relates-to" not in result
        assert "@confidential" not in result
        assert "@expires" not in result
        assert "Hello" in result
        assert "World" in result

    def test_preserves_non_directive_lines(self):
        text = "Line 1\nLine 2\nLine 3"
        result = strip_directives(text)
        assert result == text

    def test_empty_text(self):
        assert strip_directives("") == ""

    def test_only_directives(self):
        text = "@confidential\n@context: test"
        result = strip_directives(text)
        assert result.strip() == ""

    def test_preserves_indentation(self):
        text = "  normal line\n  @confidential\n  another line"
        result = strip_directives(text)
        assert "  normal line" in result
        assert "  another line" in result
        assert "@confidential" not in result


# ---------------------------------------------------------------------------
# ApplyResult
# ---------------------------------------------------------------------------


class TestApplyResult:
    def test_defaults(self):
        r = ApplyResult()
        assert r.metadata_updated is False
        assert r.edges_created == 0
        assert r.directives_applied == 0

    def test_repr(self):
        r = ApplyResult(metadata_updated=True, edges_created=1, directives_applied=3)
        s = repr(r)
        assert "metadata_updated=True" in s
        assert "edges=1" in s

    def test_frozen(self):
        r = ApplyResult()
        with pytest.raises(AttributeError):
            r.edges_created = 5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# apply_directives
# ---------------------------------------------------------------------------


class TestApplyDirectives:
    def _add_doc(self, kg: KnowledgeGraph, doc_id: str, name: str, content: str) -> None:
        kg.add_node(
            Node(
                id=doc_id,
                node_type=NodeType.DOCUMENT,
                name=name,
                content=content,
            )
        )

    def test_updates_metadata(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content")

        directives = ParsedDirectives(
            contexts=[ContextDirective(value="Q3 Planning", line=1)],
            confidentials=[ConfidentialDirective(line=2)],
        )
        result = apply_directives(kg, "doc-1", directives)

        assert result.metadata_updated is True
        node = kg.get_node("doc-1")
        assert node is not None
        assert node.metadata["_contexts"] == ["Q3 Planning"]
        assert node.metadata["_confidential"] is True
        assert node.metadata["_has_directives"] is True

    def test_creates_relates_to_edges(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "My notes")
        self._add_doc(kg, "doc-2", "Project Alpha", "Alpha project details")

        directives = ParsedDirectives(
            relates_to=[RelatesToDirective(target="Project Alpha", line=1)]
        )
        result = apply_directives(kg, "doc-1", directives)

        assert result.edges_created == 1
        edges = kg.get_edges("doc-1", direction="outgoing", relation="relates_to")
        assert len(edges) == 1
        assert edges[0].target == "doc-2"
        assert edges[0].metadata["directive"] is True

    def test_no_self_edge(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Alpha", "@relates-to: Alpha")

        directives = ParsedDirectives(relates_to=[RelatesToDirective(target="Alpha", line=1)])
        result = apply_directives(kg, "doc-1", directives)

        # Should not create self-edge
        assert result.edges_created == 0

    def test_missing_node(self):
        kg = KnowledgeGraph(":memory:")
        directives = ParsedDirectives(contexts=[ContextDirective(value="test", line=1)])
        result = apply_directives(kg, "nonexistent", directives)

        assert result.metadata_updated is False
        assert result.directives_applied == 0

    def test_relates_to_target_not_found(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content")

        directives = ParsedDirectives(
            relates_to=[RelatesToDirective(target="Nonexistent Project", line=1)]
        )
        result = apply_directives(kg, "doc-1", directives)

        # Edge not created because target not found
        assert result.edges_created == 0

    def test_expires_metadata(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content")

        directives = ParsedDirectives(expires=[ExpiresDirective(date_str="2026-12-31", line=1)])
        apply_directives(kg, "doc-1", directives)

        node = kg.get_node("doc-1")
        assert node is not None
        assert node.metadata["_expires"] == ["2026-12-31"]

    def test_empty_directives(self):
        kg = KnowledgeGraph(":memory:")
        self._add_doc(kg, "doc-1", "Notes", "Content")

        directives = ParsedDirectives()
        result = apply_directives(kg, "doc-1", directives)

        assert result.metadata_updated is True
        assert result.directives_applied == 0
        node = kg.get_node("doc-1")
        assert node is not None
        assert node.metadata["_has_directives"] is False


# ---------------------------------------------------------------------------
# Integration: parse + apply
# ---------------------------------------------------------------------------


class TestParseAndApply:
    def test_full_roundtrip(self):
        kg = KnowledgeGraph(":memory:")
        kg.add_node(
            Node(
                id="doc-1",
                node_type=NodeType.DOCUMENT,
                name="Meeting Notes",
                content="Notes from today.\n@context: Q3 Planning\n@confidential\n@expires: 2026-12-31",
            )
        )
        kg.add_node(
            Node(
                id="doc-2",
                node_type=NodeType.DOCUMENT,
                name="Q3 Planning",
                content="The Q3 plan details.",
            )
        )

        node = kg.get_node("doc-1")
        assert node is not None
        directives = parse_directives(node.content)
        result = apply_directives(kg, "doc-1", directives)

        assert result.metadata_updated is True
        assert result.directives_applied == 3

        updated = kg.get_node("doc-1")
        assert updated is not None
        assert updated.metadata["_confidential"] is True
        assert updated.metadata["_contexts"] == ["Q3 Planning"]
        assert updated.metadata["_expires"] == ["2026-12-31"]


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        from talk_box import (
            ApplyResult,
            ConfidentialDirective,
            ContextDirective,
            ExpiresDirective,
            ParsedDirectives,
            RelatesToDirective,
            apply_directives,
            parse_directives,
            strip_directives,
        )

        assert ApplyResult is not None
        assert ConfidentialDirective is not None
        assert ContextDirective is not None
        assert ExpiresDirective is not None
        assert ParsedDirectives is not None
        assert RelatesToDirective is not None
        assert apply_directives is not None
        assert parse_directives is not None
        assert strip_directives is not None

    def test_all_contains_exports(self):
        import talk_box

        for name in [
            "ApplyResult",
            "ConfidentialDirective",
            "ContextDirective",
            "ExpiresDirective",
            "ParsedDirectives",
            "RelatesToDirective",
            "apply_directives",
            "parse_directives",
            "strip_directives",
        ]:
            assert name in talk_box.__all__, f"{name} missing from __all__"
