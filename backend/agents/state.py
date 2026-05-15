"""Shared LangGraph state for the supervisor graph."""

from __future__ import annotations

# Use operator.add as a reducer so fan-out agent nodes can each write
# into agent_outputs without clobbering each other.
import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    """State that flows through every node in the supervisor graph."""

    # Inputs
    user_id: str
    message: str
    persona: str

    # Memory layer outputs
    memory_context: list[dict[str, Any]]

    # Intent classifier outputs
    intent: dict[str, Any]
    selected_agents: list[str]

    # Specialist agent outputs — each agent appends one dict.
    agent_outputs: Annotated[list[dict[str, Any]], operator.add]

    # Tool-call audit log produced by the event_agent (and any future
    # specialist that uses tool-use). Each entry: {name, arguments, output}.
    tool_calls: Annotated[list[dict[str, Any]], operator.add]

    # Aggregator output (pre-persona)
    aggregated_response: str

    # Persona-adapted output — what the user actually sees
    final_response: str

    # Bookkeeping
    trace: Annotated[list[str], operator.add]

    # Per-node latency log emitted by `backend.observability.traceable_node`.
    # Each entry: {"name": "<node_name>", "ms": <int>}. List rather than dict
    # so the operator.add reducer can merge fan-out branches without losing
    # entries.
    node_latencies: Annotated[list[dict[str, Any]], operator.add]
