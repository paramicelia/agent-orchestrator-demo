"""LangGraph supervisor — wires memory + intent + specialists + aggregator.

Topology
--------

    load_memory ─► classify_intent ─► topic_agent  ──┐
                                  └►  people_agent ──┼─► aggregate ─► persona_adapt ─► save_memory ─► END
                                  └►  event_agent  ──┘
                                       (tool-use loop:
                                        search_events / book_event)

The conditional edge after ``classify_intent`` fans out to 1..3 specialists
based on the intent classifier's ``selected_agents``. Each specialist appends
to ``agent_outputs`` (reducer = ``operator.add``). Aggregate fans them back
into ``aggregated_response``. ``persona_adapt`` rewrites the text in the
requested tone and writes ``final_response``. ``save_memory`` persists a
single takeaway to mem0.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from backend.agents import aggregator as agg
from backend.agents import (
    event_agent,
    intent_classifier,
    people_agent,
    persona_adapter,
    topic_agent,
)
from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient
from backend.memory import MemoryClient

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Memory nodes
# --------------------------------------------------------------------------- #


def make_load_memory_node(memory: MemoryClient):
    async def load_memory_node(state: AgentState) -> dict[str, Any]:
        hits = memory.search(state["user_id"], state["message"], limit=5)
        return {
            "memory_context": hits,
            "trace": [f"memory.load: {len(hits)} hits for user={state['user_id']}"],
        }

    return load_memory_node


def make_save_memory_node(memory: MemoryClient, client: GroqClient):
    """Extract one short takeaway from the turn and persist it."""

    EXTRACT_SYSTEM = (
        "Extract ONE short fact or preference about the user from the turn "
        "below, suitable to remember across sessions. Return only the fact "
        "as a single sentence, no preamble. If nothing is worth remembering, "
        'reply with exactly: SKIP'
    )

    async def save_memory_node(state: AgentState) -> dict[str, Any]:
        turn = (
            f"User: {state['message']}\n"
            f"Assistant: {state.get('final_response', '')}"
        )
        try:
            fact = await client.lite(turn, system=EXTRACT_SYSTEM, temperature=0.0, max_tokens=80)
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_memory: LLM extract failed: %s", exc)
            return {"trace": [f"memory.save: skipped (extract failed: {exc})"]}
        fact = fact.strip()
        if not fact or fact.upper().startswith("SKIP"):
            return {"trace": ["memory.save: skipped (nothing notable)"]}
        try:
            memory.add(state["user_id"], fact, metadata={"source": "auto_extract"})
            return {"trace": [f"memory.save: stored fact ({len(fact)} chars)"]}
        except Exception as exc:  # noqa: BLE001
            logger.warning("save_memory: add failed: %s", exc)
            return {"trace": [f"memory.save: failed ({exc})"]}

    return save_memory_node


# --------------------------------------------------------------------------- #
# Conditional dispatch
# --------------------------------------------------------------------------- #


def route_to_agents(state: AgentState) -> list[str]:
    """Conditional-edge router — fans out to selected specialists.

    Returns a list of node names; LangGraph executes them in parallel.
    """
    selected = state.get("selected_agents") or ["topic"]
    mapping = {
        "topic": "topic_agent",
        "people": "people_agent",
        "event": "event_agent",
    }
    nodes = [mapping[name] for name in selected if name in mapping]
    return nodes or ["topic_agent"]


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #


def build_graph(client: GroqClient, memory: MemoryClient):
    """Compile and return a runnable LangGraph supervisor."""
    graph = StateGraph(AgentState)

    # Bind dependencies into node callables.
    async def classify_node(state: AgentState) -> dict[str, Any]:
        return await intent_classifier.intent_classifier_node(state, client)

    async def topic_node(state: AgentState) -> dict[str, Any]:
        return await topic_agent.topic_agent_node(state, client)

    async def people_node(state: AgentState) -> dict[str, Any]:
        return await people_agent.people_agent_node(state, client)

    async def event_node(state: AgentState) -> dict[str, Any]:
        return await event_agent.event_agent_node(state, client)

    async def aggregate_node(state: AgentState) -> dict[str, Any]:
        return await agg.aggregator_node(state, client)

    async def persona_node(state: AgentState) -> dict[str, Any]:
        return await persona_adapter.persona_adapt_node(state, client)

    graph.add_node("load_memory", make_load_memory_node(memory))
    graph.add_node("classify_intent", classify_node)
    graph.add_node("topic_agent", topic_node)
    graph.add_node("people_agent", people_node)
    graph.add_node("event_agent", event_node)
    graph.add_node("aggregate", aggregate_node)
    graph.add_node("persona_adapt", persona_node)
    graph.add_node("save_memory", make_save_memory_node(memory, client))

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_to_agents,
        {
            "topic_agent": "topic_agent",
            "people_agent": "people_agent",
            "event_agent": "event_agent",
        },
    )

    graph.add_edge("topic_agent", "aggregate")
    graph.add_edge("people_agent", "aggregate")
    graph.add_edge("event_agent", "aggregate")
    graph.add_edge("aggregate", "persona_adapt")
    graph.add_edge("persona_adapt", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()
