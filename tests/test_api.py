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


def test_chat_returns_full_payload(memory):
    fake = {
        "final_response": "Try a jazz bar.",
        "selected_agents": ["event"],
        "memory_context": [{"id": "m1", "text": "loves jazz", "metadata": {}, "score": 0.9}],
        "agent_outputs": [{"agent": "event", "content": "jazz bar"}],
        "trace": ["memory.load: 1 hits", "intent=['event']"],
    }
    app = _make_app(memory, fake)
    with TestClient(app) as client:
        r = client.post("/chat", json={"user_id": "demo", "message": "where to go tonight?"})
        assert r.status_code == 200
        body = r.json()
        assert body["final_response"] == "Try a jazz bar."
        assert body["selected_agents"] == ["event"]
        assert len(body["memory_used"]) == 1
        assert body["trace"]


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
