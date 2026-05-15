"""CI gating eval — runs the full eval framework end-to-end with mocked Groq.

This file is wired into ``.github/workflows/ci.yml`` as its OWN gating step
(after ``pytest tests/``). Every check here uses canned Groq responses, so
the gate is free to run on every PR — no quota burn.

What this gate validates
------------------------

1. **Dataset schema is intact.** ``eval/dataset.json`` parses, has at least
   10 turns, and every turn has the contract fields the runner depends on.
2. **Judge contract holds.** ``judge_response`` returns the right shape
   (``helpfulness`` int 1..5, ``groundedness`` float 0..1, ``reason`` str)
   on canned input.
3. **End-to-end ``run_eval.main()`` works.** With a mocked Groq client, the
   runner produces a ``results.json`` whose summary + per-turn entries
   match the schema the README claims is shipped.

If this file goes red, merging is blocked even if the unit tests pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "eval" / "dataset.json"


# --------------------------------------------------------------------------- #
# (1) Dataset schema
# --------------------------------------------------------------------------- #


REQUIRED_TURN_FIELDS: tuple[str, ...] = (
    "id",
    "user_id",
    "message",
    "expected_agents",
    "category",
)

ALLOWED_AGENT_NAMES: frozenset[str] = frozenset({"topic", "people", "event"})


def test_dataset_loads() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "turns" in data
    assert isinstance(data["turns"], list)


def test_dataset_has_at_least_ten_turns() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assert len(data["turns"]) >= 10, "dataset must ship with >=10 turns"


def test_dataset_turn_schema() -> None:
    """Each dataset turn must satisfy the runner's contract."""
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    for turn in data["turns"]:
        for field in REQUIRED_TURN_FIELDS:
            assert field in turn, f"turn {turn.get('id')} missing field {field}"
        assert isinstance(turn["id"], str) and turn["id"]
        assert isinstance(turn["user_id"], str) and turn["user_id"]
        assert isinstance(turn["message"], str) and turn["message"]
        assert isinstance(turn["expected_agents"], list)
        for a in turn["expected_agents"]:
            assert a in ALLOWED_AGENT_NAMES, (
                f"turn {turn['id']}: unknown agent {a!r} in expected_agents"
            )


def test_dataset_ids_unique() -> None:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    ids = [t["id"] for t in data["turns"]]
    assert len(ids) == len(set(ids)), "duplicate turn ids in dataset"


# --------------------------------------------------------------------------- #
# (2) Judge contract — scores fall inside the documented ranges
# --------------------------------------------------------------------------- #


async def test_judge_returns_valid_floats() -> None:
    """``judge_response`` with a canned LLM payload yields the right shape."""
    from backend.llm.groq_client import GroqClient
    from eval.judge import judge_response

    canned = '{"helpfulness": 4, "groundedness": 0.83, "reason": "fine"}'
    client = GroqClient(api_key="test")
    client.call = AsyncMock(return_value=canned)  # type: ignore[method-assign]

    out = await judge_response(
        user_message="Plan my Saturday.",
        reply="How about a jazz set at Smoke?",
        memory_context=[],
        tool_calls=[{"name": "search_events", "arguments": {"query": "jazz"}, "output": []}],
        judge_client=client,
    )

    assert isinstance(out["helpfulness"], int)
    assert 1 <= out["helpfulness"] <= 5
    assert isinstance(out["groundedness"], float)
    assert 0.0 <= out["groundedness"] <= 1.0
    assert isinstance(out["reason"], str)


async def test_judge_clamps_out_of_range_values() -> None:
    """Even if the LLM returns junk, the judge clamps into the contract range."""
    from backend.llm.groq_client import GroqClient
    from eval.judge import judge_response

    # Helpfulness 99 should clamp to 5. Groundedness -3 should clamp to 0.
    canned = '{"helpfulness": 99, "groundedness": -3.0, "reason": "junk"}'
    client = GroqClient(api_key="test")
    client.call = AsyncMock(return_value=canned)  # type: ignore[method-assign]

    out = await judge_response(
        user_message="x",
        reply="y",
        memory_context=[],
        tool_calls=[],
        judge_client=client,
    )
    assert out["helpfulness"] == 5
    assert out["groundedness"] == 0.0


def test_intent_match_score_in_unit_range() -> None:
    """Property check: intent_match is always 0..1."""
    from eval.judge import intent_match_score

    pairs: list[tuple[list[str], list[str]]] = [
        ([], []),
        (["topic"], []),
        ([], ["event"]),
        (["topic"], ["topic"]),
        (["topic", "event"], ["event", "people"]),
    ]
    for expected, actual in pairs:
        s = intent_match_score(expected, actual)
        assert 0.0 <= s <= 1.0, f"out of range for {expected} vs {actual}: {s}"


def test_composite_score_math_in_unit_range() -> None:
    from eval import run_eval

    samples = [
        (0.0, 1, 0.0),
        (1.0, 5, 1.0),
        (0.5, 3, 0.5),
        (0.33, 4, 0.7),
    ]
    for intent, helpfulness, ground in samples:
        s = run_eval.composite_score(intent, helpfulness, ground)
        assert 0.0 <= s <= 1.0, f"out of range: {s}"
    assert run_eval.to_ten(1.0) == 10.0
    assert run_eval.to_ten(0.0) == 0.0


# --------------------------------------------------------------------------- #
# (3) end-to-end run_eval.main() with mocked Groq writes a valid results.json
# --------------------------------------------------------------------------- #


RESULT_SUMMARY_KEYS: frozenset[str] = frozenset(
    {
        "n",
        "intent_match",
        "helpfulness",
        "groundedness",
        "composite_0_1",
        "composite_0_10",
        "judge_model",
        "wall_time_sec",
    }
)

RESULT_TURN_KEYS: frozenset[str] = frozenset(
    {
        "id",
        "category",
        "persona",
        "user_message",
        "expected_agents",
        "actual_agents",
        "tool_calls",
        "memory_used",
        "reply",
        "scores",
        "judge_reason",
        "duration_ms",
    }
)


@pytest.mark.asyncio
async def test_run_eval_main_writes_valid_results_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: mock Groq, run eval.run_eval.main, assert results.json schema."""
    from backend.llm.groq_client import GroqClient
    from eval import run_eval

    # 1. Stop run_eval from bailing out — it requires GROQ_API_KEY in env.
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-used")
    monkeypatch.setenv("EVAL_TURN_DELAY", "0")  # no sleeps in CI

    # 2. Redirect the results files into a throwaway sub-dir INSIDE the repo
    #    so run_eval's ``RESULTS_JSON.relative_to(REPO_ROOT)`` print does not
    #    raise — it logs the path relative to the repo at the end.
    out_dir = REPO_ROOT / "eval" / "_ci_smoke_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_json = out_dir / "results.json"
    results_md = out_dir / "results.md"
    # Wipe any previous run so the assertion below is fresh.
    for f in (results_json, results_md):
        if f.exists():
            f.unlink()
    monkeypatch.setattr(run_eval, "RESULTS_JSON", results_json)
    monkeypatch.setattr(run_eval, "RESULTS_MD", results_md)

    # 3. Patch the GroqClient so absolutely no network is touched.
    def make_fake_client(*_a: Any, **_kw: Any) -> GroqClient:
        client = GroqClient(api_key="test")

        async def fake_smart(prompt: str, *, system: str | None = None, **_k: Any) -> str:
            return "Try the Vanguard Trio at Smoke Jazz Club tonight."

        async def fake_lite(prompt: str, *, system: str | None = None, **_k: Any) -> str:
            return "SKIP"  # so save_memory short-circuits

        async def fake_lite_json(prompt: str, *, system: str | None = None, **_k: Any) -> dict[str, Any]:
            return {"selected_agents": ["topic"], "reasoning": "ci"}

        async def fake_call(
            model: str, messages: list[dict[str, Any]], **_k: Any
        ) -> str:
            # Judge call — return a fixed JSON.
            return '{"helpfulness": 4, "groundedness": 0.9, "reason": "ok"}'

        async def fake_call_with_tools(
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            **_k: Any,
        ) -> dict[str, Any]:
            return {"content": "no tools needed", "tool_calls": [], "finish_reason": "stop"}

        client.smart = AsyncMock(side_effect=fake_smart)  # type: ignore[method-assign]
        client.lite = AsyncMock(side_effect=fake_lite)  # type: ignore[method-assign]
        client.lite_json = AsyncMock(side_effect=fake_lite_json)  # type: ignore[method-assign]
        client.call = AsyncMock(side_effect=fake_call)  # type: ignore[method-assign]
        client.call_with_tools = AsyncMock(side_effect=fake_call_with_tools)  # type: ignore[method-assign]
        return client

    with patch.object(run_eval, "GroqClient", side_effect=make_fake_client):
        rc = await run_eval.main()

    assert rc == 0, "run_eval.main() returned a non-zero exit code"

    # 4. results.json schema check.
    payload = json.loads(results_json.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "results" in payload

    summary = payload["summary"]
    assert RESULT_SUMMARY_KEYS.issubset(summary.keys()), (
        f"summary missing fields: {RESULT_SUMMARY_KEYS - set(summary.keys())}"
    )
    assert isinstance(summary["n"], int)
    assert summary["n"] >= 10  # dataset has >= 10 turns
    assert 0.0 <= summary["composite_0_1"] <= 1.0
    assert 0.0 <= summary["composite_0_10"] <= 10.0

    results = payload["results"]
    assert len(results) == summary["n"]
    for r in results:
        assert RESULT_TURN_KEYS.issubset(r.keys()), (
            f"result {r.get('id')} missing fields: {RESULT_TURN_KEYS - set(r.keys())}"
        )
        s = r["scores"]
        assert 0.0 <= s["intent_match"] <= 1.0
        assert 1 <= s["helpfulness"] <= 5
        assert 0.0 <= s["groundedness"] <= 1.0
        assert 0.0 <= s["composite_0_1"] <= 1.0
        assert 0.0 <= s["composite_0_10"] <= 10.0

    # 5. results.md also written and contains the composite badge line.
    md = results_md.read_text(encoding="utf-8")
    assert "Composite score" in md
    assert "Per-turn breakdown" in md

    # 6. Cleanup — leave no temp files in tree.
    for f in (results_json, results_md):
        if f.exists():
            f.unlink()
    try:
        out_dir.rmdir()
    except OSError:
        pass
