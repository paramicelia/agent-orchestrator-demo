"""FastAPI smoke tests using TestClient with a mocked graph."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from backend.api.routes import router
from backend.memory.mem0_client import Mem0Client


def _make_app(memory: Mem0Client, fake_result: dict):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    app.state.memory = memory
    app.state.client = None

    class FakeGraph:
        ainvoke = AsyncMock(return_value=fake_result)

    app.state.graph = FakeGraph()
    return app


def test_healthz(memory):
    app = _make_app(memory, {})
    with TestClient(app) as client:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_personas_endpoint(memory):
    app = _make_app(memory, {})
    with TestClient(app) as client:
        r = client.get("/personas")
        assert r.status_code == 200
        body = r.json()
        assert "neutral" in body["personas"]
        assert "gen-z" in body["personas"]


def test_chat_returns_full_payload(memory):
    fake = {
        "final_response": "Try a jazz bar.",
        "aggregated_response": "Try a jazz bar (aggregated).",
        "selected_agents": ["event"],
        "memory_context": [{"id": "m1", "text": "loves jazz", "metadata": {}, "score": 0.9}],
        "agent_outputs": [{"agent": "event", "content": "jazz bar"}],
        "tool_calls": [
            {
                "name": "search_events",
                "arguments": {"query": "jazz", "location": "New York"},
                "output": [{"event_id": "evt_jazz_001", "title": "Vanguard Trio"}],
            }
        ],
        "trace": ["memory.load: 1 hits", "intent=['event']"],
    }
    app = _make_app(memory, fake)
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={"user_id": "demo", "message": "where to go tonight?", "persona": "casual"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["final_response"] == "Try a jazz bar."
        assert body["selected_agents"] == ["event"]
        assert len(body["memory_used"]) == 1
        assert body["trace"]
        assert body["persona"] == "casual"
        assert body["tool_calls"][0]["name"] == "search_events"


def test_chat_persona_defaults_neutral(memory):
    fake = {
        "final_response": "x",
        "aggregated_response": "x",
        "selected_agents": ["topic"],
        "memory_context": [],
        "agent_outputs": [],
        "tool_calls": [],
        "trace": [],
    }
    app = _make_app(memory, fake)
    with TestClient(app) as client:
        # No persona field at all
        r = client.post("/chat", json={"user_id": "u", "message": "hi"})
        assert r.status_code == 200
        assert r.json()["persona"] == "neutral"


def test_chat_persona_invalid_falls_back(memory):
    fake = {
        "final_response": "x",
        "aggregated_response": "x",
        "selected_agents": ["topic"],
        "memory_context": [],
        "agent_outputs": [],
        "tool_calls": [],
        "trace": [],
    }
    app = _make_app(memory, fake)
    with TestClient(app) as client:
        r = client.post(
            "/chat", json={"user_id": "u", "message": "hi", "persona": "shakespearean"}
        )
        assert r.status_code == 200
        assert r.json()["persona"] == "neutral"


def test_chat_validation(memory):
    app = _make_app(memory, {})
    with TestClient(app) as client:
        r = client.post("/chat", json={"user_id": "", "message": "x"})
        assert r.status_code == 422


def test_memory_routes(memory):
    memory.add("u1", "Loves Korean BBQ")
    app = _make_app(memory, {})
    with TestClient(app) as client:
        r = client.get("/memory/u1")
        assert r.status_code == 200
        body = r.json()
        assert body["user_id"] == "u1"
        assert any("Korean" in m["text"] for m in body["memories"])

        r2 = client.post("/reset/u1")
        assert r2.status_code == 200
        assert r2.json()["removed"] == 1
