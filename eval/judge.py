"""LLM-as-judge — scores a turn against expected behaviour.

Three metrics
-------------

* ``intent_match`` (0..1) — Jaccard between the agents the graph actually
  picked and the agents the dataset said should fire.
* ``helpfulness`` (1..5) — Likert score returned by the judge model
  (``llama-3.3-70b-versatile``) reading the user message + reply.
* ``groundedness`` (0..1) — judge model decides whether the reply uses
  memory and tool output when relevant.

The judge's structured output is requested in JSON mode to avoid the usual
prose-parsing failure modes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)


JUDGE_SYSTEM = """You are an evaluator for an AI social-concierge assistant.

Score the assistant's reply on two axes:

1. helpfulness (integer 1..5):
   1 = useless or off-topic
   2 = weak / generic
   3 = OK
   4 = good
   5 = excellent — specific, on-topic, actionable

2. groundedness (float 0.0..1.0):
   - If the user has prior memories listed, the reply MUST use them when
     they are relevant. If it ignores obviously relevant memory, score < 0.5.
   - If the assistant called tools, the reply MUST reference the tool
     results (e.g. real event titles). If it invents facts that contradict
     the tools, score < 0.3.
   - If there are no memories and no tools, score 1.0 by default unless
     the reply hallucinates an obvious fake fact.

Return STRICT JSON of the form:

  {"helpfulness": <int 1-5>, "groundedness": <float 0-1>, "reason": "<one short sentence>"}

No preamble, no markdown, no extra keys.
"""


def intent_match_score(expected: list[str], actual: list[str]) -> float:
    """Jaccard similarity between expected and actual agent picks.

    Empty expected falls back to "any agent is fine" → score 1.0.
    """
    if not expected:
        return 1.0
    a = set(expected)
    b = set(actual)
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _format_memory(mems: list[dict[str, Any]]) -> str:
    if not mems:
        return "(none)"
    return "\n".join(f"- {m.get('text', '')}" for m in mems)


def _format_tools(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return "(no tools called)"
    blocks = []
    for t in tools:
        blocks.append(
            f"- {t.get('name')}({json.dumps(t.get('arguments', {}))}) -> "
            f"{json.dumps(t.get('output'))[:300]}"
        )
    return "\n".join(blocks)


async def judge_response(
    *,
    user_message: str,
    reply: str,
    memory_context: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    judge_client: GroqClient,
) -> dict[str, Any]:
    """Score a single turn. Returns {helpfulness, groundedness, reason}."""
    prompt = (
        f"User message:\n{user_message}\n\n"
        f"Memories available to assistant:\n{_format_memory(memory_context)}\n\n"
        f"Tools the assistant called:\n{_format_tools(tool_calls)}\n\n"
        f"Assistant reply:\n{reply}\n\n"
        "Score it now."
    )
    try:
        raw = await judge_client.call(
            model=judge_client.smart_model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge_response: parse failed: %s", exc)
        return {"helpfulness": 3, "groundedness": 0.5, "reason": f"judge_error: {exc}"}

    # Clamp + coerce.
    helpfulness = int(max(1, min(5, int(data.get("helpfulness", 3)))))
    grd_raw = float(data.get("groundedness", 0.5))
    groundedness = max(0.0, min(1.0, grd_raw))
    return {
        "helpfulness": helpfulness,
        "groundedness": round(groundedness, 3),
        "reason": str(data.get("reason", ""))[:240],
    }
