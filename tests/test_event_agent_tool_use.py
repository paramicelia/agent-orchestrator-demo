"""Event agent — tool-use loop with a mocked Groq client."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.agents import event_agent
from backend.llm.groq_client import GroqClient


class _ScriptedClient:
    """Replays a fixed sequence of ``call_with_tools`` responses."""

    def __init__(self, responses: list[dict[str, Any]]):
        self.smart_model = "test-smart"
        self.lite_model = "test-lite"
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def call_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **_: Any,
    ) -> dict[str, Any]:
        self.calls.append({"messages": list(messages), "tools": list(tools)})
        return self._responses.pop(0)


@pytest.mark.asyncio
async def test_event_agent_no_tool_path():
    """If the model returns no tool calls, agent returns text immediately."""
    client = _ScriptedClient(
        [
            {
                "content": "Try Vanguard Trio tonight.",
                "tool_calls": [],
                "finish_reason": "stop",
            }
        ]
    )
    text, tools = await event_agent.run_with_tools(
        "any plans for tonight?", [], "u1", client  # type: ignore[arg-type]
    )
    assert text == "Try Vanguard Trio tonight."
    assert tools == []


@pytest.mark.asyncio
async def test_event_agent_search_then_answer():
    """Standard 2-round loop: search_events -> final text."""
    client = _ScriptedClient(
        [
            # Round 1: model asks for search_events
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "search_events",
                        "arguments": {"query": "jazz", "location": "New York"},
                    }
                ],
                "finish_reason": "tool_calls",
            },
            # Round 2: model writes its reply using the result
            {
                "content": "Vanguard Trio at Smoke Jazz Club, $35.",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        ]
    )
    text, tools = await event_agent.run_with_tools(
        "jazz tonight in NY", [], "u1", client  # type: ignore[arg-type]
    )
    assert "Vanguard Trio" in text
    assert len(tools) == 1
    assert tools[0]["name"] == "search_events"
    # Tool actually ran — output is a list of dicts from the stub catalogue
    assert isinstance(tools[0]["output"], list)
    assert tools[0]["output"][0]["title"] == "Vanguard Trio — Late Set"


@pytest.mark.asyncio
async def test_event_agent_search_then_book_then_answer():
    """3-round loop: search -> book -> final."""
    client = _ScriptedClient(
        [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_events",
                        "arguments": {"query": "indie film", "location": "New York"},
                    }
                ],
                "finish_reason": "tool_calls",
            },
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c2",
                        "name": "book_event",
                        "arguments": {"event_id": "evt_film_001", "user_id": "u1"},
                    }
                ],
                "finish_reason": "tool_calls",
            },
            {
                "content": "Booked — confirmation forthcoming.",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        ]
    )
    text, tools = await event_agent.run_with_tools(
        "book that indie premiere", [], "u1", client  # type: ignore[arg-type]
    )
    assert "Booked" in text
    assert [t["name"] for t in tools] == ["search_events", "book_event"]
    assert tools[1]["output"]["status"] == "confirmed"


@pytest.mark.asyncio
async def test_event_agent_node_writes_tool_calls(mock_groq, memory):
    """The LangGraph node attaches tool_calls to state."""

    # mock_groq.call_with_tools default returns no tool calls. Override:
    async def two_round(*args: Any, **kwargs: Any) -> dict[str, Any]:
        # First call: ask for search; subsequent: answer.
        seq = two_round._seq  # type: ignore[attr-defined]
        if seq["i"] == 0:
            seq["i"] += 1
            return {
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "name": "search_events",
                        "arguments": {"query": "ai", "location": "online"},
                    }
                ],
                "finish_reason": "tool_calls",
            }
        return {
            "content": "Check the LangGraph meetup.",
            "tool_calls": [],
            "finish_reason": "stop",
        }

    two_round._seq = {"i": 0}  # type: ignore[attr-defined]
    mock_groq.call_with_tools = AsyncMock(side_effect=two_round)  # type: ignore[method-assign]
    state = {
        "user_id": "u9",
        "message": "any AI events online?",
        "memory_context": [],
        "agent_outputs": [],
        "tool_calls": [],
        "trace": [],
    }
    out = await event_agent.event_agent_node(state, mock_groq)
    assert "LangGraph" in out["agent_outputs"][0]["content"]
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["name"] == "search_events"


@pytest.mark.asyncio
async def test_event_agent_unknown_tool_does_not_crash():
    """A made-up tool call should land as an error in the tool log, not raise."""
    client = _ScriptedClient(
        [
            {
                "content": None,
                "tool_calls": [
                    {"id": "x", "name": "delete_universe", "arguments": {}}
                ],
                "finish_reason": "tool_calls",
            },
            {
                "content": "Sorry, I can't do that.",
                "tool_calls": [],
                "finish_reason": "stop",
            },
        ]
    )
    text, tools = await event_agent.run_with_tools(
        "delete the universe", [], "u1", client  # type: ignore[arg-type]
    )
    assert text  # didn't blow up
    assert tools[0]["output"]["error"]


def test_event_agent_uses_groq_client_signature():
    """Verify the real GroqClient exposes call_with_tools (regression guard)."""
    c = GroqClient(api_key="x")
    assert hasattr(c, "call_with_tools")


def test_infer_query_picks_known_keywords():
    assert event_agent._infer_query_from_message("Find me a jazz event tonight") == "jazz"
    assert event_agent._infer_query_from_message("Any indie film premieres?") == "film"
    assert event_agent._infer_query_from_message("AI meetups online") == "ai"
    assert event_agent._infer_query_from_message("Korean BBQ this week") == "food"


def test_infer_query_skips_generic_intent_words():
    """plan / weekend / bored / help should not become the query."""
    assert event_agent._infer_query_from_message("Plan my weekend") != "plan"
    assert event_agent._infer_query_from_message("I'm bored") != "bored"
    assert event_agent._infer_query_from_message("Help me decide") != "help"
    # Fallback returns the safe generic query
    assert event_agent._infer_query_from_message("plan my weekend") == event_agent._GENERIC_FALLBACK_QUERY


def test_infer_location_picks_known_city():
    assert event_agent._infer_location_from_message("jazz in NYC tonight") == "New York"
    assert event_agent._infer_location_from_message("free online event") == "online"
    assert event_agent._infer_location_from_message("something fun") == "any"
