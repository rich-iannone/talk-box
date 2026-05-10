"""Tests for SessionUsage token and cost tracking."""

import pytest

from talk_box.usage import SessionUsage


class TestSessionUsage:
    def test_default_values(self):
        usage = SessionUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_cost == 0.0
        assert usage.turns == 0
        assert usage.total_tokens == 0

    def test_add_turn(self):
        usage = SessionUsage()
        usage.add_turn(input_tokens=100, output_tokens=200, cost=0.003)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 200
        assert usage.total_tokens == 300
        assert usage.total_cost == 0.003
        assert usage.turns == 1

    def test_add_multiple_turns(self):
        usage = SessionUsage()
        usage.add_turn(input_tokens=100, output_tokens=200, cost=0.003)
        usage.add_turn(input_tokens=500, output_tokens=800, cost=0.008)
        assert usage.input_tokens == 600
        assert usage.output_tokens == 1000
        assert usage.total_tokens == 1600
        assert usage.total_cost == pytest.approx(0.011)
        assert usage.turns == 2

    def test_to_dict(self):
        usage = SessionUsage()
        usage.add_turn(input_tokens=120, output_tokens=350, cost=0.002)
        d = usage.to_dict()
        assert d["input_tokens"] == 120
        assert d["output_tokens"] == 350
        assert d["total_tokens"] == 470
        assert d["total_cost"] == 0.002
        assert d["turns"] == 1

    def test_reset(self):
        usage = SessionUsage()
        usage.add_turn(input_tokens=500, output_tokens=800, cost=0.01)
        usage.reset()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_cost == 0.0
        assert usage.turns == 0
        assert usage._turn_details == []

    def test_update_from_chat_session_with_data(self):
        """update_from_chat_session extracts tokens and cost from chatlas."""
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.get_tokens.return_value = [{"input": 50, "output": 120}]
        mock_session.get_cost.return_value = 0.005

        usage = SessionUsage()
        usage.update_from_chat_session(mock_session)
        assert usage.input_tokens == 50
        assert usage.output_tokens == 120
        assert usage.total_cost == 0.005
        assert usage.turns == 1

    def test_update_from_chat_session_empty_tokens(self):
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.get_tokens.return_value = []
        mock_session.get_cost.return_value = 0.0

        usage = SessionUsage()
        usage.update_from_chat_session(mock_session)
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.turns == 1

    def test_update_from_chat_session_exception_handling(self):
        from unittest.mock import MagicMock

        mock_session = MagicMock()
        mock_session.get_tokens.side_effect = Exception("not supported")
        mock_session.get_cost.side_effect = Exception("not supported")

        usage = SessionUsage()
        usage.update_from_chat_session(mock_session)
        # Should not raise, defaults to zero
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_cost == 0.0
        assert usage.turns == 1

    def test_turn_details_tracked(self):
        usage = SessionUsage()
        usage.add_turn(input_tokens=10, output_tokens=20, cost=0.001)
        usage.add_turn(input_tokens=30, output_tokens=40, cost=0.002)
        assert len(usage._turn_details) == 2
        assert usage._turn_details[0]["turn"] == 1
        assert usage._turn_details[1]["turn"] == 2


class TestChatBotUsage:
    def test_chatbot_has_usage(self):
        from talk_box.builder import ChatBot

        bot = ChatBot()
        usage = bot.get_usage()
        assert isinstance(usage, SessionUsage)
        assert usage.turns == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
