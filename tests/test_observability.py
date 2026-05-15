"""Observability shim — no LangSmith key means a clean no-op."""

from __future__ import annotations

import importlib
import os


def test_tracing_disabled_without_key(monkeypatch):
    """init_tracing returns False when API key is absent."""
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    from backend.config import get_settings

    get_settings.cache_clear()

    import backend.observability as obs

    importlib.reload(obs)
    assert obs.init_tracing() is False
    assert obs.is_tracing_enabled() is False


def test_tracing_enabled_when_key_present(monkeypatch):
    """When both flag and key are set, env is propagated and True returned."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake_key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "aod-test")

    from backend.config import get_settings

    get_settings.cache_clear()

    import backend.observability as obs

    importlib.reload(obs)
    assert obs.init_tracing() is True
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGCHAIN_API_KEY") == "lsv2_fake_key"
    assert os.environ.get("LANGCHAIN_PROJECT") == "aod-test"


def test_tracing_disabled_when_flag_false_even_with_key(monkeypatch):
    """Both flag AND key must be present — flag-only or key-only is disabled."""
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_fake_key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

    from backend.config import get_settings

    get_settings.cache_clear()

    import backend.observability as obs

    importlib.reload(obs)
    assert obs.init_tracing() is False


async def test_traceable_node_appends_latency_line():
    """`traceable_node` must inject a `node:<name> took Xms` line into trace."""
    from backend.observability import traceable_node

    @traceable_node("dummy_node")
    async def fake_node(state):
        return {"trace": ["hello"], "selected_agents": ["topic"]}

    out = await fake_node({"message": "x"})
    assert "selected_agents" in out
    # original trace line is preserved
    assert out["trace"][0] == "hello"
    # latency line is appended
    assert any(line.startswith("node:dummy_node took") for line in out["trace"])
    # structured latency entry is present for the UI
    assert out["node_latencies"]
    assert out["node_latencies"][0]["name"] == "dummy_node"
    assert isinstance(out["node_latencies"][0]["ms"], int)


async def test_traceable_node_with_no_existing_trace():
    """If the wrapped node returns no trace key, the wrapper still injects one."""
    from backend.observability import traceable_node

    @traceable_node("solo_node")
    async def fake_node(state):
        return {"final_response": "done"}

    out = await fake_node({})
    assert out["final_response"] == "done"
    assert any("node:solo_node took" in line for line in out["trace"])
