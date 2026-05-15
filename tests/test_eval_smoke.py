"""Eval harness — smoke test on 3 hardcoded turns with a mocked Groq.

CI must never hit the real Groq API, so this file exercises the eval flow
end-to-end against ``conftest.mock_groq``. The judge is wired to return a
fixed score so we assert plumbing, not LLM behaviour.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.agents.supervisor import build_graph
from eval import run_eval
from eval.judge import intent_match_score


def test_intent_match_perfect():
    assert intent_match_score(["topic"], ["topic"]) == 1.0


def test_intent_match_partial():
    s = intent_match_score(["topic", "event"], ["topic"])
    # Jaccard: 1/2 = 0.5
    assert s == 0.5


def test_intent_match_disjoint():
    assert intent_match_score(["topic"], ["event"]) == 0.0


def test_intent_match_expected_empty_is_one():
    assert intent_match_score([], ["topic"]) == 1.0


def test_composite_score_math():
    # intent=1.0, helpfulness=5 (-> 1.0 normalised), groundedness=1.0
    assert run_eval.composite_score(1.0, 5, 1.0) == 1.0
    # intent=0.0, helpfulness=1 (-> 0.0), groundedness=0.0
    assert run_eval.composite_score(0.0, 1, 0.0) == 0.0
    # mid
    s = run_eval.composite_score(0.5, 3, 0.5)
    assert 0.3 < s < 0.55


def test_to_ten_scaling():
    assert run_eval.to_ten(1.0) == 10.0
    assert run_eval.to_ten(0.0) == 0.0
    assert run_eval.to_ten(0.55) == 5.5


@pytest.mark.asyncio
async def test_run_turn_smoke(mock_groq, memory):
    """Wire the whole flow on 3 hardcoded turns with the mocked Groq."""

    # Classifier — different agent pick per turn to force interesting paths.
    classify_sequence = [
        {"selected_agents": ["topic"], "reasoning": "x"},
        {"selected_agents": ["event"], "reasoning": "x"},
        {"selected_agents": ["topic", "event"], "reasoning": "x"},
    ]
    idx = {"i": 0}

    async def fake_classify(*_a: Any, **_k: Any) -> dict[str, Any]:
        out = classify_sequence[idx["i"] % len(classify_sequence)]
        idx["i"] += 1
        return out

    mock_groq.lite_json = AsyncMock(side_effect=fake_classify)  # type: ignore[method-assign]

    # Judge — return a fixed JSON payload via the raw `call` method
    async def fake_judge_call(model: str, messages: list[dict[str, Any]], **_: Any) -> str:
        return '{"helpfulness": 4, "groundedness": 0.8, "reason": "fine"}'

    mock_groq.call = AsyncMock(side_effect=fake_judge_call)  # type: ignore[method-assign]

    graph = build_graph(mock_groq, memory)

    turns = [
        {
            "id": "smoke_1",
            "user_id": "smoke_u",
            "message": "ideas for me?",
            "persona": "neutral",
            "expected_agents": ["topic"],
            "category": "single",
        },
        {
            "id": "smoke_2",
            "user_id": "smoke_u",
            "message": "what to do tonight?",
            "persona": "neutral",
            "expected_agents": ["event"],
            "category": "single",
        },
        {
            "id": "smoke_3",
            "user_id": "smoke_u",
            "message": "weekend plans",
            "persona": "neutral",
            "expected_agents": ["topic", "event"],
            "category": "multi",
        },
    ]
    results = []
    for turn in turns:
        scored = await run_eval.run_turn(graph, memory, mock_groq, turn)
        results.append(scored)

    assert len(results) == 3
    for r in results:
        s = r["scores"]
        assert 0.0 <= s["intent_match"] <= 1.0
        assert 1 <= s["helpfulness"] <= 5
        assert 0.0 <= s["groundedness"] <= 1.0
        assert 0.0 <= s["composite_0_1"] <= 1.0
        assert 0.0 <= s["composite_0_10"] <= 10.0

    summary = run_eval.summarise(results, "test-judge", 1.23)
    assert summary["n"] == 3
    assert summary["judge_model"] == "test-judge"
    assert 0.0 <= summary["composite_0_10"] <= 10.0


def test_render_markdown_is_well_formed():
    """Smoke check the renderer produces a markdown table."""
    fake_results = [
        {
            "id": "x1",
            "category": "single",
            "expected_agents": ["topic"],
            "actual_agents": ["topic"],
            "tool_calls": [],
            "scores": {
                "intent_match": 1.0,
                "helpfulness": 4,
                "groundedness": 0.9,
                "composite_0_1": 0.85,
                "composite_0_10": 8.5,
            },
        }
    ]
    summary = {
        "n": 1,
        "intent_match": 1.0,
        "helpfulness": 4.0,
        "groundedness": 0.9,
        "composite_0_1": 0.85,
        "composite_0_10": 8.5,
        "judge_model": "fake",
        "wall_time_sec": 0.1,
    }
    md = run_eval.render_markdown(fake_results, summary)
    assert "Composite score" in md
    assert "| x1 |" in md
    assert "Per-turn breakdown" in md


def test_dataset_loads_and_has_required_fields(repo_root):
    """The shipped dataset.json must parse and have the contract fields."""
    import json

    data = json.loads((repo_root / "eval" / "dataset.json").read_text(encoding="utf-8"))
    turns = data["turns"]
    assert len(turns) >= 10
    for t in turns:
        for field in ("id", "user_id", "message", "expected_agents", "category"):
            assert field in t, f"turn {t.get('id')} missing field {field}"
