"""Topic specialist — suggests conversation topics, reading, learning."""

from __future__ import annotations

import asyncio
from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient
from backend.observability import traceable_node

NAME = "topic"

SYSTEM_PROMPT = """You are the Topic Agent inside a social concierge assistant.

Suggest 2-3 interesting conversation topics, articles or ideas tailored to
the user's request. Be specific and avoid generic platitudes.

ONLY use the prior memories that are explicitly provided to you. NEVER
invent past conversations, locations the user has visited, articles they
have read, or anything else that wasn't given to you. If you have no
relevant memory, just give good general suggestions without pretending to
know the user.

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


@traceable_node("topic_agent")
async def topic_agent_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    """LangGraph node wrapper."""
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    text = await run(state["message"], state.get("memory_context", []), client)
    elapsed_ms = int((loop.time() - t0) * 1000)
    return {
        "agent_outputs": [{"agent": NAME, "content": text}],
        "trace": [f"agent:{NAME} produced {len(text)} chars in {elapsed_ms}ms"],
    }
