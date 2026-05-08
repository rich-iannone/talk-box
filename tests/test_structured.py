import json

import pytest
from pydantic import BaseModel

from talk_box.builder import ChatBot
from talk_box.structured import (
    ExtractResult,
    extract,
    schema_to_dict,
)


# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class City(BaseModel):
    name: str
    country: str
    population: int


class Sentiment(BaseModel):
    label: str
    score: float


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class Nested(BaseModel):
    name: str
    tags: list[str]
    scores: dict[str, float]


# ---------------------------------------------------------------------------
# ExtractResult
# ---------------------------------------------------------------------------


class TestExtractResult:
    def test_frozen(self):
        result = ExtractResult(data={"a": 1}, model_name="M", duration=0.5)
        with pytest.raises(AttributeError):
            result.data = {}  # type: ignore[misc]

    def test_fields(self):
        result = ExtractResult(data={"x": 1}, model_name="MyModel", duration=1.23)
        assert result.data == {"x": 1}
        assert result.model_name == "MyModel"
        assert result.duration == 1.23


# ---------------------------------------------------------------------------
# extract (mock path)
# ---------------------------------------------------------------------------


class TestExtractMock:
    def test_basic_extraction(self):
        bot = ChatBot()
        bot.mock_responses(
            [json.dumps({"name": "Paris", "country": "France", "population": 2161000})]
        )
        result = extract(bot, "Tell me about Paris", data_model=City)
        assert result.data == {"name": "Paris", "country": "France", "population": 2161000}
        assert result.model_name == "City"
        assert result.duration >= 0

    def test_sentiment_extraction(self):
        bot = ChatBot()
        bot.mock_responses([json.dumps({"label": "positive", "score": 0.95})])
        result = extract(bot, "I love it", data_model=Sentiment)
        assert result.data["label"] == "positive"
        assert result.data["score"] == 0.95

    def test_nested_model(self):
        bot = ChatBot()
        data = {"name": "test", "tags": ["a", "b"], "scores": {"x": 1.0, "y": 2.0}}
        bot.mock_responses([json.dumps(data)])
        result = extract(bot, "complex data", data_model=Nested)
        assert result.data["tags"] == ["a", "b"]
        assert result.data["scores"]["x"] == 1.0

    def test_invalid_json_raises(self):
        bot = ChatBot()
        bot.mock_responses(["not json at all"])
        with pytest.raises(ValueError, match="not valid JSON"):
            extract(bot, "parse this", data_model=City)

    def test_invalid_schema_raises(self):
        bot = ChatBot()
        # Missing required field "population"
        bot.mock_responses([json.dumps({"name": "Paris", "country": "France"})])
        with pytest.raises(Exception):
            extract(bot, "tell me", data_model=City)

    def test_extra_fields_ignored(self):
        bot = ChatBot()
        data = {"name": "London", "country": "UK", "population": 8982000, "extra": "ignored"}
        bot.mock_responses([json.dumps(data)])
        result = extract(bot, "London", data_model=City)
        assert "extra" not in result.data
        assert result.data["name"] == "London"

    def test_consumes_one_mock(self):
        bot = ChatBot()
        bot.mock_responses(
            [
                json.dumps({"name": "A", "country": "B", "population": 1}),
                json.dumps({"name": "C", "country": "D", "population": 2}),
            ]
        )
        r1 = extract(bot, "first", data_model=City)
        r2 = extract(bot, "second", data_model=City)
        assert r1.data["name"] == "A"
        assert r2.data["name"] == "C"

    def test_duration_is_measured(self):
        bot = ChatBot()
        bot.mock_responses([json.dumps({"label": "neutral", "score": 0.5})])
        result = extract(bot, "meh", data_model=Sentiment)
        assert isinstance(result.duration, float)
        assert result.duration >= 0

    def test_with_agent(self):
        from talk_box.agent import Agent
        from talk_box.personas import create_persona

        persona = create_persona("test", persona_role="assistant", description="Test.")
        agent = Agent(name="test", persona=persona)
        agent.chatbot.mock_responses([json.dumps({"label": "happy", "score": 0.9})])
        result = extract(agent.chatbot, "I'm happy", data_model=Sentiment, agent=agent)
        assert result.data["label"] == "happy"

    def test_agent_param_overrides_chatbot(self):
        from talk_box.agent import Agent
        from talk_box.personas import create_persona

        persona = create_persona("test", persona_role="assistant", description="Test.")
        agent = Agent(name="test", persona=persona)
        other_bot = ChatBot()
        # Mock on agent's chatbot, not other_bot
        agent.chatbot.mock_responses([json.dumps({"label": "x", "score": 0.1})])
        result = extract(other_bot, "msg", data_model=Sentiment, agent=agent)
        assert result.data["label"] == "x"


# ---------------------------------------------------------------------------
# schema_to_dict
# ---------------------------------------------------------------------------


class TestSchemaToDict:
    def test_basic_schema(self):
        schema = schema_to_dict(City)
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "country" in schema["properties"]
        assert "population" in schema["properties"]

    def test_required_fields(self):
        schema = schema_to_dict(City)
        assert set(schema["required"]) == {"name", "country", "population"}

    def test_nested_schema(self):
        schema = schema_to_dict(Nested)
        props = schema["properties"]
        assert "name" in props
        assert "tags" in props
        assert "scores" in props

    def test_returns_dict(self):
        schema = schema_to_dict(Sentiment)
        assert isinstance(schema, dict)

    def test_title_matches_model(self):
        schema = schema_to_dict(City)
        assert schema["title"] == "City"


# ---------------------------------------------------------------------------
# Top-level imports
# ---------------------------------------------------------------------------


class TestTopLevelImport:
    def test_import_from_package(self):
        import talk_box

        for name in ["ExtractResult", "extract", "schema_to_dict"]:
            assert hasattr(talk_box, name)

    def test_all_contains_exports(self):
        import talk_box

        for name in ["ExtractResult", "extract", "schema_to_dict"]:
            assert name in talk_box.__all__
