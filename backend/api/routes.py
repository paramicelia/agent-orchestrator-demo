"""FastAPI routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.agents.persona_adapter import SUPPORTED_PERSONAS, normalise_persona

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)
    persona: str | None = Field(
        default="neutral",
        description=f"Reply tone. One of: {', '.join(SUPPORTED_PERSONAS)}",
    )


class ChatResponse(BaseModel):
    user_id: str
    message: str
    persona: str
    final_response: str
    aggregated_response: str
    selected_agents: list[str]
    memory_used: list[dict[str, Any]]
    agent_outputs: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    trace: list[str]
    # Per-node wall-clock latencies, injected by
    # `backend.observability.traceable_node`. Frontend renders these as a
    # breakdown so a slow turn can be diagnosed without leaving the page.
    node_latencies: list[dict[str, Any]] = []


class MemoryItem(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]


class MemoryListResponse(BaseModel):
    user_id: str
    memories: list[MemoryItem]


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/personas")
async def personas() -> dict[str, list[str]]:
    """List the personas the persona_adapt node understands."""
    return {"personas": list(SUPPORTED_PERSONAS)}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    if graph is None:
        raise HTTPException(503, "Graph not initialised")
    persona = normalise_persona(req.persona)
    initial_state: dict[str, Any] = {
        "user_id": req.user_id,
        "message": req.message,
        "persona": persona,
        "agent_outputs": [],
        "tool_calls": [],
        "trace": [],
    }
    try:
        result = await graph.ainvoke(initial_state)
    except RuntimeError as exc:
        # Most common: GROQ_API_KEY missing.
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("graph.ainvoke failed")
        raise HTTPException(500, f"Agent error: {exc}") from exc

    return ChatResponse(
        user_id=req.user_id,
        message=req.message,
        persona=persona,
        final_response=result.get("final_response", ""),
        aggregated_response=result.get("aggregated_response", ""),
        selected_agents=result.get("selected_agents", []),
        memory_used=result.get("memory_context", []),
        agent_outputs=result.get("agent_outputs", []),
        tool_calls=result.get("tool_calls", []),
        trace=result.get("trace", []),
        node_latencies=result.get("node_latencies", []),
    )


@router.get("/memory/{user_id}", response_model=MemoryListResponse)
async def get_memory(user_id: str, request: Request) -> MemoryListResponse:
    memory = request.app.state.memory
    if memory is None:
        raise HTTPException(503, "Memory not initialised")
    items = memory.get_all(user_id)
    return MemoryListResponse(
        user_id=user_id,
        memories=[MemoryItem(**item) for item in items],
    )


@router.post("/reset/{user_id}")
async def reset_memory(user_id: str, request: Request) -> dict[str, Any]:
    memory = request.app.state.memory
    if memory is None:
        raise HTTPException(503, "Memory not initialised")
    removed = memory.reset(user_id)
    return {"user_id": user_id, "removed": removed}
