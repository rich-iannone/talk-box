"""Structured outputs: extract typed data from LLM responses."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, TypeVar

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractResult:
    """Result of extracting structured data from an LLM response.

    Parameters
    ----------
    data
        The extracted data as a dictionary matching the schema.
    model_name
        The model class name used for extraction.
    duration
        Wall-clock time in seconds the extraction took.

    Examples
    --------
    ```python
    import talk_box as tb
    from pydantic import BaseModel

    class Sentiment(BaseModel):
        label: str
        score: float

    bot = tb.ChatBot().model("gpt-4o-mini")
    result = tb.extract(bot, "I love this product!", data_model=Sentiment)
    result.data       # {"label": "positive", "score": 0.95}
    result.model_name # "Sentiment"
    result.duration   # 1.23
    ```
    """

    data: dict[str, Any]
    model_name: str
    duration: float


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def extract(
    chatbot: Any,
    message: str,
    *,
    data_model: type,
    agent: Any | None = None,
) -> ExtractResult:
    """Extract structured data from an LLM response.

    Sends a message to a `ChatBot` or `Agent` and parses the response into
    a typed dictionary matching a Pydantic model schema.

    When the ChatBot has mock responses configured, the first mock response
    is consumed and parsed as JSON.  When a real LLM is configured, the
    chatlas ``extract_data()`` method is used for provider-native structured
    output support.

    Parameters
    ----------
    chatbot
        A `ChatBot` instance (or the ``chatbot`` attribute of an ``Agent``).
    message
        The prompt to send.
    data_model
        A Pydantic ``BaseModel`` subclass describing the desired output schema.
    agent
        An optional ``Agent`` whose ``chatbot`` will be used. If provided,
        *chatbot* is ignored.

    Returns
    -------
    ExtractResult
        The extracted data, model name, and timing.

    Raises
    ------
    ValueError
        If the mock response is not valid JSON matching the schema.

    Examples
    --------
    With mock responses (for testing):

    ```python
    import talk_box as tb
    from pydantic import BaseModel

    class City(BaseModel):
        name: str
        country: str
        population: int

    bot = tb.ChatBot()
    bot.mock_responses(['{"name": "Paris", "country": "France", "population": 2161000}'])
    result = tb.extract(bot, "Tell me about Paris", data_model=City)
    result.data  # {"name": "Paris", "country": "France", "population": 2161000}
    ```

    With an Agent:

    ```python
    agent = tb.Agent.from_persona("data_analyst")
    agent.chatbot.mock_responses(['{"name": "Tokyo", "country": "Japan", "population": 13960000}'])
    result = tb.extract(agent.chatbot, "Tell me about Tokyo", data_model=City, agent=agent)
    ```
    """
    if agent is not None:
        chatbot = agent.chatbot

    model_name = data_model.__name__

    start = time.monotonic()

    # Mock path: parse JSON from the mock response
    if chatbot._mock_responses:
        raw = chatbot._mock_responses.pop(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Mock response is not valid JSON for {model_name}: {exc}") from exc

        # Validate against the Pydantic model
        instance = data_model.model_validate(data)
        data = instance.model_dump()

        duration = time.monotonic() - start
        return ExtractResult(data=data, model_name=model_name, duration=duration)

    # Real LLM path: use chatlas extract_data
    from talk_box._utils_chatlas import ChatlasAdapter

    provider = chatbot._config.get("provider")
    model = chatbot._config.get("model")

    adapter = ChatlasAdapter(provider=provider, model=model)
    chat_session = adapter.create_chat_session(chatbot._config)

    data = chat_session.extract_data(message, data_model=data_model)

    duration = time.monotonic() - start
    return ExtractResult(data=data, model_name=model_name, duration=duration)


# ---------------------------------------------------------------------------
# schema_to_dict
# ---------------------------------------------------------------------------


def schema_to_dict(data_model: type) -> dict[str, Any]:
    """Convert a Pydantic model class to a JSON Schema dictionary.

    Useful for inspecting what schema will be sent to the LLM, logging,
    or building dynamic schemas.

    Parameters
    ----------
    data_model
        A Pydantic ``BaseModel`` subclass.

    Returns
    -------
    dict[str, Any]
        The JSON Schema representation of the model.

    Examples
    --------
    ```python
    from pydantic import BaseModel
    import talk_box as tb

    class Weather(BaseModel):
        city: str
        temp_f: float
        condition: str

    schema = tb.schema_to_dict(Weather)
    schema["properties"]  # {"city": {...}, "temp_f": {...}, "condition": {...}}
    schema["required"]    # ["city", "temp_f", "condition"]
    ```
    """
    return data_model.model_json_schema()
