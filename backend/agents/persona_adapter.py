"""Persona adapter — rewrites the aggregated reply in a target tone.

This node demonstrates the "context-aware translation systems" pattern the
target JD asks for: same factual content, different audience-adapted
surface form. It runs on the cheap 8B tier because the rewrite is mostly
mechanical.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

SUPPORTED_PERSONAS: tuple[str, ...] = (
    "neutral",
    "formal",
    "casual",
    "gen-z",
    "elderly-friendly",
)

PERSONA_INSTRUCTIONS: dict[str, str] = {
    "neutral": "Keep the original wording. Only fix obvious typos.",
    "formal": (
        "Rewrite in polished, professional English. Use complete sentences, "
        "no contractions, and a respectful register suitable for business email."
    ),
    "casual": (
        "Rewrite in friendly, conversational English. Use contractions, light "
        "humour, and feel like a text from a friend who knows the city well."
    ),
    "gen-z": (
        "Rewrite with Gen-Z energy — short punchy lines, lowercase preferred, "
        "occasional slang ('lowkey', 'fr', 'vibe', 'no cap'). Don't overdo it; "
        "stay readable. No emojis."
    ),
    "elderly-friendly": (
        "Rewrite for an older reader. Use simple words, short sentences, no "
        "slang and no jargon. Spell out abbreviations. Avoid any reference "
        "the reader might not recognise without explanation."
    ),
}


def normalise_persona(persona: str | None) -> str:
    """Coerce arbitrary input to a known persona, defaulting to ``neutral``."""
    if not persona:
        return "neutral"
    p = persona.strip().lower()
    return p if p in SUPPORTED_PERSONAS else "neutral"


SYSTEM_PROMPT_TEMPLATE = """You rewrite assistant replies for a specific audience.

PRESERVE all facts, names, suggestions and recommendations exactly. Do not
add new advice. Do not drop any of the original suggestions.

CHANGE only the tone and phrasing.

Target persona: {persona}
{instructions}

Return only the rewritten reply — no preamble, no quotes, no labels.
"""


async def adapt(
    text: str,
    persona: str,
    client: GroqClient,
) -> str:
    """Rewrite ``text`` in the target persona's tone."""
    persona = normalise_persona(persona)
    if persona == "neutral" or not text.strip():
        # No-op for the default — saves a Groq call on the common path.
        return text
    system = SYSTEM_PROMPT_TEMPLATE.format(
        persona=persona,
        instructions=PERSONA_INSTRUCTIONS[persona],
    )
    try:
        rewritten = await client.lite(
            text,
            system=system,
            temperature=0.5,
            max_tokens=600,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("persona_adapt: lite rewrite failed: %s", exc)
        return text
    return rewritten.strip() or text


async def persona_adapt_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    """LangGraph node wrapper."""
    aggregated = state.get("aggregated_response", "")
    persona = normalise_persona(state.get("persona"))
    rewritten = await adapt(aggregated, persona, client)
    return {
        "final_response": rewritten,
        "trace": [f"persona_adapt: {persona} -> {len(rewritten)} chars"],
    }
