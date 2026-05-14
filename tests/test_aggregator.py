"""Aggregator behaviour — empty / single / multiple drafts."""

from __future__ import annotations

import pytest

from backend.agents.aggregator import aggregate


@pytest.mark.asyncio
async def test_no_drafts_returns_friendly_message(mock_groq):
    out = await aggregate("hi", [], mock_groq)
    assert "couldn't" in out.lower() or "try" in out.lower()


@pytest.mark.asyncio
async def test_single_draft_is_polished(mock_groq):
    drafts = [{"agent": "topic", "content": "- jazz history\n- Murakami novels"}]
    out = await aggregate("got ideas?", drafts, mock_groq)
    # mocked smart() prefix
    assert out.startswith("[smart-reply]")


@pytest.mark.asyncio
async def test_multiple_drafts_are_merged(mock_groq):
    drafts = [
        {"agent": "topic", "content": "- chess openings\n- crypto philosophy"},
        {"agent": "event", "content": "- jazz bar\n- farmers market"},
    ]
    out = await aggregate("plan my weekend", drafts, mock_groq)
    # Aggregator should still produce a polished reply via smart() mock.
    assert out
    # Verify the underlying smart() call received both drafts.
    last_call = mock_groq.smart.call_args
    full_prompt = last_call.args[0] if last_call.args else last_call.kwargs.get("prompt", "")
    assert "TOPIC DRAFT" in full_prompt
    assert "EVENT DRAFT" in full_prompt
