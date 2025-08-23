import pytest

from talk_box.presets import Preset, PresetManager


class TestPreset:
    """Test cases for the Preset class."""

    def test_preset_creation(self):
        """Test basic Preset creation."""
        preset = Preset(
            name="test_preset",
            tone="friendly",
            expertise="testing",
            verbosity="medium",
            constraints=["no_spam"],
            system_prompt="You are a test assistant.",
        )

        assert preset.name == "test_preset"
        assert preset.tone == "friendly"
        assert preset.expertise == "testing"
        assert preset.verbosity == "medium"
        assert preset.constraints == ["no_spam"]
        assert preset.system_prompt == "You are a test assistant."

    def test_preset_to_dict(self):
        """Test converting preset to dictionary."""
        preset = Preset(
            name="test",
            tone="casual",
            expertise="general",
            verbosity="brief",
            constraints=[],
        )

        preset_dict = preset.to_dict()
        expected = {
            "name": "test",
            "tone": "casual",
            "expertise": "general",
            "verbosity": "brief",
            "constraints": [],
            "system_prompt": None,
        }

        assert preset_dict == expected


class TestPresetManager:
    """Test cases for the PresetManager class."""

    def test_preset_manager_initialization(self):
        """Test PresetManager initialization with default presets."""
        manager = PresetManager()
        presets = manager.list_presets()

        # Should have our default presets
        expected_presets = [
            "customer_support",
            "technical_advisor",
            "creative_writer",
            "data_analyst",
            "legal_advisor",
        ]

        for preset_name in expected_presets:
            assert preset_name in presets

    def test_get_preset(self):
        """Test getting a preset by name."""
        manager = PresetManager()

        # Test getting existing preset
        technical = manager.get_preset("technical_advisor")
        assert technical is not None
        assert technical.name == "technical_advisor"
        assert technical.tone == "authoritative"

        # Test getting non-existent preset
        missing = manager.get_preset("nonexistent")
        assert missing is None

    def test_add_preset(self):
        """Test adding a new preset."""
        manager = PresetManager()
        initial_count = len(manager.list_presets())

        new_preset = Preset(
            name="custom_preset",
            tone="enthusiastic",
            expertise="custom",
            verbosity="detailed",
            constraints=["family_friendly"],
        )

        manager.add_preset(new_preset)

        # Should have one more preset
        assert len(manager.list_presets()) == initial_count + 1
        assert "custom_preset" in manager.list_presets()

        # Should be able to retrieve it
        retrieved = manager.get_preset("custom_preset")
        assert retrieved == new_preset

    def test_remove_preset(self):
        """Test removing a preset."""
        manager = PresetManager()

        # Add a test preset first
        test_preset = Preset(
            name="temp_preset",
            tone="neutral",
            expertise="temp",
            verbosity="medium",
            constraints=[],
        )
        manager.add_preset(test_preset)

        # Verify it was added
        assert "temp_preset" in manager.list_presets()

        # Remove it
        result = manager.remove_preset("temp_preset")
        assert result is True
        assert "temp_preset" not in manager.list_presets()

        # Try to remove non-existent preset
        result = manager.remove_preset("nonexistent")
        assert result is False

    def test_apply_preset(self):
        """Test applying a preset to a configuration."""
        manager = PresetManager()

        # Test with empty config
        config = {}
        updated = manager.apply_preset("technical_advisor", config)

        assert updated["name"] == "technical_advisor"
        assert updated["tone"] == "authoritative"
        assert updated["expertise"] == "python,ml"

        # Test with existing config values (should not override)
        config = {"tone": "casual", "custom_field": "value"}
        updated = manager.apply_preset("technical_advisor", config)

        # Should keep existing values
        assert updated["tone"] == "casual"
        assert updated["custom_field"] == "value"
        # Should add missing values from preset
        assert updated["expertise"] == "python,ml"

    def test_apply_nonexistent_preset(self):
        """Test applying a non-existent preset raises error."""
        manager = PresetManager()

        with pytest.raises(ValueError, match="Preset 'nonexistent' not found"):
            manager.apply_preset("nonexistent", {})

    def test_list_presets(self):
        """Test listing all presets."""
        manager = PresetManager()
        presets = manager.list_presets()

        assert isinstance(presets, list)
        assert len(presets) >= 5  # Should have at least our default presets
        assert all(isinstance(name, str) for name in presets)
