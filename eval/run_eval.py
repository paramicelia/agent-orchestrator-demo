"""Async eval runner — loads dataset, runs the graph, scores, writes results.

Usage::

    python -m eval.run_eval

Requires ``GROQ_API_KEY`` in the environment (or .env). Writes:

* ``eval/results.json`` — full machine-readable scoring per turn.
* ``eval/results.md``   — human-readable table with the overall score.

The overall score is the mean of:

* ``intent_match``  (normalised 0..1, weight 1)
* ``helpfulness``   (normalised 1-5 → 0..1, weight 2)
* ``groundedness``  (0..1, weight 1)

Composite ranges 0..1; we also publish a 0..10 score for the README badge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Compat shim — must match backend/main.py.
import langchain  # type: ignore[import-not-found]

for _attr, _default in (("debug", False), ("verbose", False), ("llm_cache", None)):
    if not hasattr(langchain, _attr):
        setattr(langchain, _attr, _default)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from backend.agents.supervisor import build_graph  # noqa: E402
from backend.llm.groq_client import GroqClient  # noqa: E402
from backend.memory.mem0_client import Mem0Client  # noqa: E402
from eval.judge import intent_match_score, judge_response  # noqa: E402

logger = logging.getLogger("eval.run_eval")

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "eval" / "dataset.json"
RESULTS_JSON = REPO_ROOT / "eval" / "results.json"
RESULTS_MD = REPO_ROOT / "eval" / "results.md"


# --------------------------------------------------------------------------- #
# Scoring math
# --------------------------------------------------------------------------- #


def composite_score(intent: float, helpfulness: int, groundedness: float) -> float:
    """Weighted mean of the three metrics, normalised to 0..1."""
    helpfulness_n = (helpfulness - 1) / 4.0
    weighted = (intent * 1.0) + (helpfulness_n * 2.0) + (groundedness * 1.0)
    return round(weighted / 4.0, 3)


def to_ten(score_unit: float) -> float:
    return round(score_unit * 10.0, 1)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


async def run_turn(
    graph: Any,
    memory: Mem0Client,
    judge_client: GroqClient,
    turn: dict[str, Any],
) -> dict[str, Any]:
    """Run one dataset turn through the graph and score it."""
    user_id = turn["user_id"]

    # Seed prior memories if the dataset asked for them.
    for mem_text in turn.get("prior_memories", []) or []:
        memory.add(user_id, mem_text, metadata={"source": "eval_seed"})

    initial_state: dict[str, Any] = {
        "user_id": user_id,
        "message": turn["message"],
        "persona": turn.get("persona", "neutral"),
        "agent_outputs": [],
        "tool_calls": [],
        "trace": [],
    }
    started = time.monotonic()
    result = await graph.ainvoke(initial_state)
    duration_ms = int((time.monotonic() - started) * 1000)

    expected_agents = turn.get("expected_agents") or []
    actual_agents = result.get("selected_agents") or []
    im = intent_match_score(expected_agents, actual_agents)

    scores = await judge_response(
        user_message=turn["message"],
        reply=result.get("final_response", ""),
        memory_context=result.get("memory_context", []),
        tool_calls=result.get("tool_calls", []),
        judge_client=judge_client,
    )

    composite = composite_score(im, scores["helpfulness"], scores["groundedness"])

    return {
        "id": turn["id"],
        "category": turn.get("category"),
        "persona": initial_state["persona"],
        "user_message": turn["message"],
        "expected_agents": expected_agents,
        "actual_agents": actual_agents,
        "tool_calls": [
            {"name": tc.get("name"), "arguments": tc.get("arguments")}
            for tc in result.get("tool_calls", [])
        ],
        "memory_used": [
            m.get("text", "") for m in result.get("memory_context", [])
        ],
        "reply": result.get("final_response", ""),
        "scores": {
            "intent_match": round(im, 3),
            "helpfulness": scores["helpfulness"],
            "groundedness": scores["groundedness"],
            "composite_0_1": composite,
            "composite_0_10": to_ten(composite),
        },
        "judge_reason": scores["reason"],
        "duration_ms": duration_ms,
    }


def render_markdown(results: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Eval results")
    lines.append("")
    lines.append(f"- **Composite score:** {summary['composite_0_10']} / 10")
    lines.append(f"- **Intent match (mean):** {summary['intent_match']:.2f}")
    lines.append(f"- **Helpfulness (mean):** {summary['helpfulness']:.2f} / 5")
    lines.append(f"- **Groundedness (mean):** {summary['groundedness']:.2f}")
    lines.append(f"- **Turns scored:** {summary['n']}")
    lines.append(f"- **Judge model:** `{summary['judge_model']}`")
    lines.append(f"- **Run wall-time:** {summary['wall_time_sec']:.1f}s")
    lines.append("")
    lines.append("## Per-turn breakdown")
    lines.append("")
    lines.append(
        "| ID | Category | Expected | Actual | Intent | Helpful | Ground | Composite |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        s = r["scores"]
        lines.append(
            "| {id} | {cat} | {exp} | {act} | {im:.2f} | {h}/5 | {g:.2f} | {c:.2f}/10 |".format(
                id=r["id"],
                cat=r.get("category", "-"),
                exp=", ".join(r["expected_agents"]) or "-",
                act=", ".join(r["actual_agents"]) or "-",
                im=s["intent_match"],
                h=s["helpfulness"],
                g=s["groundedness"],
                c=s["composite_0_10"],
            )
        )
    lines.append("")
    lines.append("## Tool use observed")
    lines.append("")
    for r in results:
        if r["tool_calls"]:
            names = ", ".join(tc["name"] for tc in r["tool_calls"])
            lines.append(f"- **{r['id']}** -> {names}")
    if not any(r["tool_calls"] for r in results):
        lines.append("- (no tool calls observed)")
    return "\n".join(lines) + "\n"


def summarise(results: list[dict[str, Any]], judge_model: str, wall_time_sec: float) -> dict[str, Any]:
    if not results:
        return {
            "n": 0,
            "intent_match": 0.0,
            "helpfulness": 0.0,
            "groundedness": 0.0,
            "composite_0_1": 0.0,
            "composite_0_10": 0.0,
            "judge_model": judge_model,
            "wall_time_sec": wall_time_sec,
        }
    n = len(results)
    im = sum(r["scores"]["intent_match"] for r in results) / n
    hp = sum(r["scores"]["helpfulness"] for r in results) / n
    gr = sum(r["scores"]["groundedness"] for r in results) / n
    composite = sum(r["scores"]["composite_0_1"] for r in results) / n
    return {
        "n": n,
        "intent_match": round(im, 3),
        "helpfulness": round(hp, 3),
        "groundedness": round(gr, 3),
        "composite_0_1": round(composite, 3),
        "composite_0_10": to_ten(composite),
        "judge_model": judge_model,
        "wall_time_sec": round(wall_time_sec, 2),
    }


async def main() -> int:
    logging.basicConfig(
        level=os.environ.get("EVAL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Aborting eval.", file=sys.stderr)
        return 1

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    turns: list[dict[str, Any]] = dataset["turns"]

    # Use a throwaway Chroma dir so prior_memories from this run never leak
    # into the persistent demo store.
    tmp_chroma = tempfile.mkdtemp(prefix="chroma_eval_")
    try:
        client = GroqClient()
        memory = Mem0Client(db_path=tmp_chroma)
        graph = build_graph(client, memory)

        results: list[dict[str, Any]] = []
        started = time.monotonic()
        delay_s = float(os.environ.get("EVAL_TURN_DELAY", "1.5"))
        for i, turn in enumerate(turns):
            if i > 0 and delay_s > 0:
                # Stay under the Groq free-tier rate limit.
                await asyncio.sleep(delay_s)
            logger.info("eval turn %s (%s)", turn["id"], turn.get("category"))
            try:
                scored = await run_turn(graph, memory, client, turn)
            except Exception as exc:  # noqa: BLE001
                logger.exception("turn %s failed: %s", turn["id"], exc)
                scored = {
                    "id": turn["id"],
                    "category": turn.get("category"),
                    "user_message": turn["message"],
                    "expected_agents": turn.get("expected_agents", []),
                    "actual_agents": [],
                    "tool_calls": [],
                    "memory_used": [],
                    "reply": "",
                    "scores": {
                        "intent_match": 0.0,
                        "helpfulness": 1,
                        "groundedness": 0.0,
                        "composite_0_1": 0.0,
                        "composite_0_10": 0.0,
                    },
                    "judge_reason": f"runtime_error: {exc}",
                    "duration_ms": 0,
                }
            results.append(scored)
            logger.info(
                "  -> composite %.2f / 10 (intent=%.2f, h=%d, g=%.2f)",
                scored["scores"]["composite_0_10"],
                scored["scores"]["intent_match"],
                scored["scores"]["helpfulness"],
                scored["scores"]["groundedness"],
            )

        wall = time.monotonic() - started
        summary = summarise(results, client.smart_model, wall)

        RESULTS_JSON.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )
        RESULTS_MD.write_text(render_markdown(results, summary), encoding="utf-8")

        print()
        print("=" * 72)
        print(f" Composite: {summary['composite_0_10']} / 10")
        print(f" Intent:    {summary['intent_match']:.2f}")
        print(f" Helpful:   {summary['helpfulness']:.2f} / 5")
        print(f" Ground:    {summary['groundedness']:.2f}")
        print(f" Wall time: {summary['wall_time_sec']:.1f}s")
        print(f" Wrote:     {RESULTS_JSON.relative_to(REPO_ROOT)}")
        print(f" Wrote:     {RESULTS_MD.relative_to(REPO_ROOT)}")
        print("=" * 72)
    finally:
        shutil.rmtree(tmp_chroma, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
