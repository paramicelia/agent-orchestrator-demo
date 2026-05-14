"""Intent classifier tests — sanitises agent picks and falls back safely."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.agents.intent_classifier import VALID_AGENTS, classify_intent
from backend.llm.groq_client import GroqClient


@pytest.fixture
def client_with_response():
    def _make(payload: dict) -> GroqClient:
        c = GroqClient(api_key="test")
        c.lite_json = AsyncMock(return_value=payload)  # type: ignore[method-assign]
        return c

    return _make


async def test_single_agent_pick(client_with_response):
    client = client_with_response({"selected_agents": ["event"], "reasoning": "weekend out"})
    result = await classify_intent("what should I do this weekend?", client)
    assert result["selected_agents"] == ["event"]


async def test_multiple_agents_pick(client_with_response):
    client = client_with_response(
        {"selected_agents": ["topic", "event"], "reasoning": "ideas + place"}
    )
    result = await classify_intent("help me plan an evening with my partner", client)
    assert set(result["selected_agents"]) == {"topic", "event"}


async def test_unknown_agent_dropped(client_with_response):
    client = client_with_response({"selected_agents": ["weather", "event"], "reasoning": "x"})
    result = await classify_intent("blah", client)
    assert result["selected_agents"] == ["event"]


async def test_empty_picks_fallback_to_topic(client_with_response):
    client = client_with_response({"selected_agents": [], "reasoning": "unknown"})
    result = await classify_intent("???", client)
    assert result["selected_agents"] == ["topic"]


async def test_malformed_response_fallback(client_with_response):
    client = client_with_response({})  # missing keys entirely
    result = await classify_intent("hello", client)
    assert result["selected_agents"] == ["topic"]


def test_valid_agents_constant():
    # If this changes someone must update the routing table in supervisor.py.
    assert set(VALID_AGENTS) == {"topic", "people", "event"}
