import pytest
from talk_box import ChatBot, PresetNames


class TestPresetNames:
    """Test suite for PresetNames constants."""

    def test_preset_names_constants_exist(self):
        """Test that all expected PresetNames constants exist."""
        expected_constants = [
            "CUSTOMER_SUPPORT",
            "TECHNICAL_ADVISOR",
            "DATA_ANALYST",
            "CREATIVE_WRITER",
            "LEGAL_ADVISOR",
        ]

        for constant in expected_constants:
            assert hasattr(PresetNames, constant), f"PresetNames.{constant} should exist"

    def test_preset_names_values(self):
        """Test that PresetNames constants have correct string values."""
        expected_values = {
            PresetNames.CUSTOMER_SUPPORT: "customer_support",
            PresetNames.TECHNICAL_ADVISOR: "technical_advisor",
            PresetNames.DATA_ANALYST: "data_analyst",
            PresetNames.CREATIVE_WRITER: "creative_writer",
            PresetNames.LEGAL_ADVISOR: "legal_advisor",
        }

        for constant, expected_value in expected_values.items():
            assert str(constant) == expected_value

    def test_preset_with_string(self):
        """Test ChatBot preset method with string value."""
        bot = ChatBot().preset("customer_support")
        assert bot._config.get("preset") == "customer_support"

    def test_preset_with_constant(self):
        """Test ChatBot preset method with PresetNames constant."""
        bot = ChatBot().preset(PresetNames.CUSTOMER_SUPPORT)
        assert bot._config.get("preset") == "customer_support"

    def test_preset_string_and_constant_equivalence(self):
        """Test that string and constant approaches produce same result."""
        bot1 = ChatBot().preset("technical_advisor")
        bot2 = ChatBot().preset(PresetNames.TECHNICAL_ADVISOR)

        assert bot1._config.get("preset") == bot2._config.get("preset")

    def test_all_preset_constants_work(self):
        """Test that all PresetNames constants work with ChatBot."""
        constants = [
            PresetNames.CUSTOMER_SUPPORT,
            PresetNames.TECHNICAL_ADVISOR,
            PresetNames.DATA_ANALYST,
            PresetNames.CREATIVE_WRITER,
            PresetNames.LEGAL_ADVISOR,
        ]

        for constant in constants:
            bot = ChatBot().preset(constant)
            expected_value = str(constant)
            assert bot._config.get("preset") == expected_value

    def test_preset_method_chaining_with_constants(self):
        """Test method chaining works with PresetNames constants."""
        bot = ChatBot().preset(PresetNames.DATA_ANALYST).temperature(0.3).max_tokens(500)

        assert bot._config.get("preset") == "data_analyst"
        assert bot._config.get("temperature") == 0.3
        assert bot._config.get("max_tokens") == 500
