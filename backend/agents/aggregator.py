"""Aggregator — merges specialist outputs into one coherent reply."""

from __future__ import annotations

from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient

SYSTEM_PROMPT = """You are the final response writer for a social concierge.

You will receive 1-3 specialist drafts (topics / people / events). Merge them
into a single short, warm reply addressed to the user. Keep it conversational
and useful — under 180 words. Preserve concrete suggestions; drop filler.

Structure:
- 1 short opening line acknowledging the request.
- The merged suggestions, organised by theme with light headings if more
  than one specialist contributed.
- 1 short closing line inviting the user to pick one.

Do not mention "agents" or internal mechanics.
"""


def _format_drafts(agent_outputs: list[dict[str, Any]]) -> str:
    if not agent_outputs:
        return "(no specialist drafts)"
    blocks = []
    for item in agent_outputs:
        agent = item.get("agent", "?")
        content = item.get("content", "").strip()
        blocks.append(f"[{agent.upper()} DRAFT]\n{content}")
    return "\n\n".join(blocks)


async def aggregate(
    message: str,
    agent_outputs: list[dict[str, Any]],
    client: GroqClient,
) -> str:
    """Merge specialist outputs into one response."""
    if not agent_outputs:
        return "I couldn't generate suggestions right now. Try rephrasing?"
    if len(agent_outputs) == 1:
        # Single specialist — still polish, but cheaper.
        drafts = _format_drafts(agent_outputs)
        prompt = (
            f"User asked: {message}\n\n"
            f"Single specialist draft below — lightly polish it into a warm reply.\n\n{drafts}"
        )
        return await client.smart(prompt, system=SYSTEM_PROMPT, temperature=0.4, max_tokens=400)
    drafts = _format_drafts(agent_outputs)
    prompt = f"User asked: {message}\n\nSpecialist drafts:\n\n{drafts}"
    return await client.smart(prompt, system=SYSTEM_PROMPT, temperature=0.5, max_tokens=500)


async def aggregator_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    text = await aggregate(state["message"], state.get("agent_outputs", []), client)
    return {
        # Pre-persona text. The persona_adapt node writes the final user-facing
        # text into ``final_response`` (and short-circuits to this text when the
        # persona is "neutral").
        "aggregated_response": text,
        "final_response": text,
        "trace": [f"aggregate: {len(state.get('agent_outputs', []))} drafts -> {len(text)} chars"],
    }
