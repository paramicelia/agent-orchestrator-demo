"""Intent classifier — picks 1..N specialist agents using the lite tier."""

from __future__ import annotations

import logging
from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

VALID_AGENTS: tuple[str, ...] = ("topic", "people", "event")

SYSTEM_PROMPT = """You are an intent router for a social concierge assistant.

Your job is to read the user's message and pick which specialist agents should
handle it. There are exactly three specialists:

- "topic"  — suggests interesting conversation topics, articles to read, things to learn
- "people" — suggests people to meet, reconnect with, or introduce to each other
- "event"  — suggests events, places to go, activities to do

You may pick 1, 2, or all 3. Pick the MINIMUM number that fully serves the user.

Reply with strict JSON of the form:
{"selected_agents": ["topic", "event"], "reasoning": "<one short sentence>"}

Use only the agent names above. Do not invent new agents.
"""


async def classify_intent(message: str, client: GroqClient) -> dict[str, Any]:
    """Run the lite classifier and return {selected_agents, reasoning}."""
    user_prompt = f'User message: """{message}"""\n\nRespond with JSON.'
    raw = await client.lite_json(user_prompt, system=SYSTEM_PROMPT, temperature=0.0)
    selected = raw.get("selected_agents") or []
    if not isinstance(selected, list):
        selected = []
    # sanitise: only keep known agents, preserve order, dedupe
    cleaned: list[str] = []
    for name in selected:
        if isinstance(name, str) and name in VALID_AGENTS and name not in cleaned:
            cleaned.append(name)
    if not cleaned:
        # safe fallback — always pick topic so the user gets *something*
        cleaned = ["topic"]
        logger.info("intent classifier produced no valid agents; falling back to ['topic']")
    return {
        "selected_agents": cleaned,
        "reasoning": raw.get("reasoning", ""),
    }


async def intent_classifier_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    """LangGraph node wrapper."""
    intent = await classify_intent(state["message"], client)
    return {
        "intent": intent,
        "selected_agents": intent["selected_agents"],
        "trace": [f"intent={intent['selected_agents']} reason={intent['reasoning'][:60]}"],
    }
