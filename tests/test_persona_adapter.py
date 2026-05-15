"""Persona adapter — neutral is a no-op, other personas hit the lite tier."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.agents import persona_adapter


def test_normalise_persona_known():
    assert persona_adapter.normalise_persona("formal") == "formal"
    assert persona_adapter.normalise_persona("GEN-Z") == "gen-z"


def test_normalise_persona_unknown_falls_back():
    assert persona_adapter.normalise_persona("klingon") == "neutral"
    assert persona_adapter.normalise_persona(None) == "neutral"
    assert persona_adapter.normalise_persona("") == "neutral"


@pytest.mark.asyncio
async def test_adapt_neutral_is_noop(mock_groq):
    text = "Original text suggesting jazz club."
    out = await persona_adapter.adapt(text, "neutral", mock_groq)
    assert out == text
    # Crucially: didn't call the LLM at all.
    mock_groq.lite.assert_not_awaited()


@pytest.mark.asyncio
async def test_adapt_genz_calls_lite(mock_groq):
    mock_groq.lite = AsyncMock(return_value="lowkey go check the jazz spot, fr")  # type: ignore[method-assign]
    out = await persona_adapter.adapt("Go to the jazz club.", "gen-z", mock_groq)
    assert "lowkey" in out
    mock_groq.lite.assert_awaited_once()


@pytest.mark.asyncio
async def test_adapt_falls_back_on_llm_error(mock_groq):
    async def boom(*_a, **_k):
        raise RuntimeError("groq down")

    mock_groq.lite = AsyncMock(side_effect=boom)  # type: ignore[method-assign]
    original = "Try the indie film premiere tonight."
    out = await persona_adapter.adapt(original, "casual", mock_groq)
    assert out == original  # never raises


@pytest.mark.asyncio
async def test_adapt_empty_input_short_circuits(mock_groq):
    out = await persona_adapter.adapt("", "formal", mock_groq)
    assert out == ""
    mock_groq.lite.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_adapt_node_writes_final_response(mock_groq):
    mock_groq.lite = AsyncMock(return_value="Formal rewrite.")  # type: ignore[method-assign]
    state = {
        "aggregated_response": "Go grab some jazz.",
        "persona": "formal",
    }
    out = await persona_adapter.persona_adapt_node(state, mock_groq)
    assert out["final_response"] == "Formal rewrite."
    assert any("persona_adapt" in t for t in out["trace"])


@pytest.mark.asyncio
async def test_persona_adapt_node_neutral_preserves(mock_groq):
    state = {
        "aggregated_response": "Original reply.",
        "persona": "neutral",
    }
    out = await persona_adapter.persona_adapt_node(state, mock_groq)
    assert out["final_response"] == "Original reply."


def test_supported_personas_constant():
    assert "neutral" in persona_adapter.SUPPORTED_PERSONAS
    assert "formal" in persona_adapter.SUPPORTED_PERSONAS
    assert "casual" in persona_adapter.SUPPORTED_PERSONAS
    assert "gen-z" in persona_adapter.SUPPORTED_PERSONAS
    assert "elderly-friendly" in persona_adapter.SUPPORTED_PERSONAS
