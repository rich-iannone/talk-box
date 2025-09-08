"""Test the new constructor parameters."""

import pytest
from talk_box.pathways import Pathways


def test_constructor_with_completion_criteria():
    """Test that completion_criteria can be passed to constructor."""

    # Test with string
    pathway1 = Pathways(
        title="Test Pathway",
        desc="Test description",
        activation="Test activation",
        completion_criteria="Single completion criterion",
    )

    data1 = pathway1._build()
    assert data1["completion_criteria"] == ["Single completion criterion"]

    # Test with list
    pathway2 = Pathways(
        title="Test Pathway",
        desc="Test description",
        activation="Test activation",
        completion_criteria=["Criterion 1", "Criterion 2"],
    )

    data2 = pathway2._build()
    assert data2["completion_criteria"] == ["Criterion 1", "Criterion 2"]

    # Test with None (default)
    pathway3 = Pathways(title="Test Pathway", desc="Test description")

    data3 = pathway3._build()
    assert data3["completion_criteria"] == []


def test_constructor_with_fallback_strategy():
    """Test that fallback_strategy can be passed to constructor."""

    # Test with string
    pathway1 = Pathways(
        title="Test Pathway",
        desc="Test description",
        activation="Test activation",
        fallback_strategy="Test fallback strategy",
    )

    data1 = pathway1._build()
    assert data1["fallback_strategy"] == "Test fallback strategy"

    # Test with None (default)
    pathway2 = Pathways(title="Test Pathway", desc="Test description")

    data2 = pathway2._build()
    assert data2["fallback_strategy"] is None


def test_complete_constructor():
    """Test constructor with all parameters."""

    pathway = Pathways(
        title="Complete Test",
        desc="Full test of constructor",
        activation=["Activation 1", "Activation 2"],
        completion_criteria=["Complete 1", "Complete 2"],
        fallback_strategy="Fallback approach",
    )

    data = pathway._build()
    assert data["title"] == "Complete Test"
    assert data["description"] == "Full test of constructor"
    assert data["activation_conditions"] == ["Activation 1", "Activation 2"]
    assert data["completion_criteria"] == ["Complete 1", "Complete 2"]
    assert data["fallback_strategy"] == "Fallback approach"


def test_old_chaining_methods_removed():
    """Test that the old chaining methods are no longer available."""

    pathway = Pathways(title="Test Pathway", desc="Test description")

    # These methods should not exist anymore
    assert not hasattr(pathway, "completion_criteria"), (
        "completion_criteria method should be removed"
    )
    assert not hasattr(pathway, "fallback_strategy"), "fallback_strategy method should be removed"


def test_pathway_building_still_works():
    """Test that the pathway building with states still works."""

    pathway = (
        Pathways(
            title="Support Flow",
            desc="Customer support pathway",
            activation="User needs help",
            completion_criteria=["Issue resolved", "User satisfied"],
            fallback_strategy="Escalate to human support",
        )
        .state("Gather information", id="intake")
        .required(["issue_description", "contact_info"])
        .next_state("resolution")
        .state("Provide solution", id="resolution")
        .success_condition("User's issue is resolved")
    )

    data = pathway._build()

    # Check pathway-level data
    assert data["title"] == "Support Flow"
    assert data["completion_criteria"] == ["Issue resolved", "User satisfied"]
    assert data["fallback_strategy"] == "Escalate to human support"

    # Check states still work
    assert len(data["states"]) == 2
    assert "intake" in data["states"]
    assert "resolution" in data["states"]
