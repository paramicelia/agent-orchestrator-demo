"""Shared pytest fixtures."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

# Compatibility shim: some environments have a newer `langchain` package
# installed globally that no longer exposes `langchain.debug`, while
# `langchain-core` 0.3.x still probes for it. Set it to False up-front so
# `langchain_core.globals.get_debug()` does not raise AttributeError when
# the LangGraph callback manager initialises.
try:  # pragma: no cover - environment shim
    import langchain  # type: ignore[import-not-found]

    if not hasattr(langchain, "debug"):
        langchain.debug = False
    if not hasattr(langchain, "verbose"):
        langchain.verbose = False
    if not hasattr(langchain, "llm_cache"):
        langchain.llm_cache = None
except Exception:  # noqa: BLE001
    pass

from backend.llm.groq_client import GroqClient
from backend.memory.mem0_client import Mem0Client


@pytest.fixture
def tmp_chroma_dir() -> Iterator[str]:
    d = tempfile.mkdtemp(prefix="chroma_test_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def memory(tmp_chroma_dir: str) -> Mem0Client:
    return Mem0Client(db_path=tmp_chroma_dir)


@pytest.fixture
def mock_groq() -> GroqClient:
    """Groq client with all network methods replaced by AsyncMock."""
    client = GroqClient(api_key="test-key")

    async def fake_smart(prompt: str, *, system: str | None = None, **_: Any) -> str:
        return f"[smart-reply] {prompt[:60]}"

    async def fake_lite(prompt: str, *, system: str | None = None, **_: Any) -> str:
        return "[lite-reply]"

    async def fake_lite_json(prompt: str, *, system: str | None = None, **_: Any) -> dict[str, Any]:
        # default: only topic
        return {"selected_agents": ["topic"], "reasoning": "stub"}

    async def fake_call(
        model: str,
        messages: list[dict[str, Any]],
        **_: Any,
    ) -> str:
        return "[call-reply]"

    async def fake_call_with_tools(
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **_: Any,
    ) -> dict[str, Any]:
        # Default: model returns no tool calls and a short reply. Tests that
        # need a real tool-use loop should patch this fixture directly.
        return {
            "content": "[event-reply] no tools needed",
            "tool_calls": [],
            "finish_reason": "stop",
        }

    client.smart = AsyncMock(side_effect=fake_smart)  # type: ignore[method-assign]
    client.lite = AsyncMock(side_effect=fake_lite)  # type: ignore[method-assign]
    client.lite_json = AsyncMock(side_effect=fake_lite_json)  # type: ignore[method-assign]
    client.call = AsyncMock(side_effect=fake_call)  # type: ignore[method-assign]
    client.call_with_tools = AsyncMock(side_effect=fake_call_with_tools)  # type: ignore[method-assign]
    return client


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolate_settings(monkeypatch: pytest.MonkeyPatch, tmp_chroma_dir: str) -> None:
    """Don't read the real .env during tests."""
    monkeypatch.setenv("MEMORY_DB_PATH", tmp_chroma_dir)
    # Clear the cached settings so the fixture wins.
    from backend.config import get_settings

    get_settings.cache_clear()


def _expect_json(payload: dict[str, Any]) -> str:
    """Helper used in some tests."""
    return json.dumps(payload)
