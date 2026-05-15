"""People specialist — suggests who to meet, reconnect with, introduce."""

from __future__ import annotations

from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient

NAME = "people"

SYSTEM_PROMPT = """You are the People Agent inside a social concierge assistant.

Suggest 2-3 ideas for who the user could meet, reconnect with or introduce
to each other, based on their request. Use ARCHETYPES, never invent specific
names, jobs or histories you weren't given. Good examples:

  - "a friend from your university years you haven't seen in a while"
  - "someone in your field who recently changed jobs"
  - "the colleague you mentioned enjoying lunch with"

If the user named real people or the memory layer surfaced names, you may
reference them. Otherwise stay on archetypes — do NOT make up first names,
employers, or past events.

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


async def people_agent_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    text = await run(state["message"], state.get("memory_context", []), client)
    return {
        "agent_outputs": [{"agent": NAME, "content": text}],
        "trace": [f"agent:{NAME} produced {len(text)} chars"],
    }
