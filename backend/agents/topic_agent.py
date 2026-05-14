"""Topic specialist — suggests conversation topics, reading, learning."""

from __future__ import annotations

from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient

NAME = "topic"

SYSTEM_PROMPT = """You are the Topic Agent inside a social concierge assistant.

Suggest 2-3 interesting conversation topics, articles or ideas tailored to
the user's request. Be specific and avoid generic platitudes.

If the user has prior memories (preferences, past interests), weave them in
explicitly so the suggestions feel personal.

Format: 2-3 short bullets. No preamble. No outro. Just the bullets.
"""


def _build_prompt(message: str, memory_context: list[dict[str, Any]]) -> str:
    mem_block = ""
    if memory_context:
        lines = [f"- {item.get('text', '')}" for item in memory_context]
        mem_block = "Known about the user:\n" + "\n".join(lines) + "\n\n"
    return f"{mem_block}User message: {message}"


async def run(message: str, memory_context: list[dict[str, Any]], client: GroqClient) -> str:
    """Pure functional entry point — easy to unit-test."""
    prompt = _build_prompt(message, memory_context)
    return await client.smart(prompt, system=SYSTEM_PROMPT, temperature=0.6, max_tokens=400)


async def topic_agent_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    """LangGraph node wrapper."""
    text = await run(state["message"], state.get("memory_context", []), client)
    return {
        "agent_outputs": [{"agent": NAME, "content": text}],
        "trace": [f"agent:{NAME} produced {len(text)} chars"],
    }
