"""Supervisor graph topology + end-to-end fan-out/fan-in test."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.agents.supervisor import build_graph, route_to_agents


def test_route_to_agents_default():
    assert route_to_agents({}) == ["topic_agent"]


def test_route_to_agents_multiple():
    state = {"selected_agents": ["topic", "event"]}
    nodes = route_to_agents(state)
    assert "topic_agent" in nodes
    assert "event_agent" in nodes
    assert "people_agent" not in nodes


def test_route_to_agents_unknown_falls_back():
    assert route_to_agents({"selected_agents": ["weather"]}) == ["topic_agent"]


def test_graph_compiles(mock_groq, memory):
    graph = build_graph(mock_groq, memory)
    # smoke check the compiled graph has the right nodes
    nodes = set(graph.get_graph().nodes.keys())
    expected = {
        "load_memory",
        "classify_intent",
        "topic_agent",
        "people_agent",
        "event_agent",
        "aggregate",
        "save_memory",
    }
    assert expected.issubset(nodes), f"missing nodes: {expected - nodes}"


@pytest.mark.asyncio
async def test_graph_end_to_end_single_agent(mock_groq, memory):
    # classifier returns topic only
    mock_groq.lite_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"selected_agents": ["topic"], "reasoning": "ok"}
    )
    graph = build_graph(mock_groq, memory)
    out = await graph.ainvoke(
        {
            "user_id": "u1",
            "message": "got any cool conversation ideas?",
            "agent_outputs": [],
            "trace": [],
        }
    )
    assert out["selected_agents"] == ["topic"]
    assert len(out["agent_outputs"]) == 1
    assert out["agent_outputs"][0]["agent"] == "topic"
    assert out["final_response"].startswith("[smart-reply]")


@pytest.mark.asyncio
async def test_graph_end_to_end_fan_out(mock_groq, memory):
    # classifier returns multiple agents -> fan-out
    mock_groq.lite_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"selected_agents": ["topic", "event"], "reasoning": "both"}
    )
    graph = build_graph(mock_groq, memory)
    out = await graph.ainvoke(
        {
            "user_id": "u2",
            "message": "plan my weekend",
            "agent_outputs": [],
            "trace": [],
        }
    )
    agents_run = {o["agent"] for o in out["agent_outputs"]}
    assert agents_run == {"topic", "event"}
    assert out["final_response"]
