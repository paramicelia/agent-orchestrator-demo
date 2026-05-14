"""Event specialist — suggests events, places, activities."""

from __future__ import annotations

from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient

NAME = "event"

SYSTEM_PROMPT = """You are the Event Agent inside a social concierge assistant.

Suggest 2-3 specific events, places or activities that fit the user's
request. Prefer concrete archetypes ("a small jazz bar", "Sunday morning
farmers market") over generic categories. If location is not stated, give
two options that work in any mid-to-large city.

If memory shows the user already attended something similar, do not
suggest the exact same thing — offer a fresh variation.

Format: 2-3 short bullets. No preamble. No outro.
"""


def _build_prompt(message: str, memory_context: list[dict[str, Any]]) -> str:
    mem_block = ""
    if memory_context:
        lines = [f"- {item.get('text', '')}" for item in memory_context]
        mem_block = "Known about the user:\n" + "\n".join(lines) + "\n\n"
    return f"{mem_block}User message: {message}"


async def run(message: str, memory_context: list[dict[str, Any]], client: GroqClient) -> str:
    prompt = _build_prompt(message, memory_context)
    return await client.smart(prompt, system=SYSTEM_PROMPT, temperature=0.6, max_tokens=400)


async def event_agent_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    text = await run(state["message"], state.get("memory_context", []), client)
    return {
        "agent_outputs": [{"agent": NAME, "content": text}],
        "trace": [f"agent:{NAME} produced {len(text)} chars"],
    }
