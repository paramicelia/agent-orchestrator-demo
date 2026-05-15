"""FastAPI routes."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.agents.persona_adapter import SUPPORTED_PERSONAS, normalise_persona
from backend.tenants import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=64)
    message: str = Field(..., min_length=1, max_length=2000)
    persona: str | None = Field(
        default="neutral",
        description=f"Reply tone. One of: {', '.join(SUPPORTED_PERSONAS)}",
    )
    tenant_id: str | None = Field(
        default=DEFAULT_TENANT_ID,
        min_length=1,
        max_length=64,
        description=(
            "Which tenant config to apply. Unknown ids fall back to the "
            "built-in default tenant — see GET /tenants for the list."
        ),
    )


class ChatResponse(BaseModel):
    user_id: str
    message: str
    persona: str
    tenant_id: str
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


class TenantSummary(BaseModel):
    """Subset of TenantConfig safe to return on /tenants endpoints.

    Drops free-form ``metadata`` and ``agent_prompts`` so neither internal
    branding labels nor system-prompt overrides leak through a public
    endpoint by accident.
    """

    # Same protected-namespace opt-out as TenantConfig: the legit field
    # name ``model_tier`` collides with pydantic v2's default reserved
    # ``model_`` prefix.
    model_config = {"protected_namespaces": ()}

    tenant_id: str
    display_name: str
    allowed_personas: list[str]
    default_persona: str
    model_tier: dict[str, str]
    memory_retention_days: int


class TenantListResponse(BaseModel):
    tenants: list[TenantSummary]


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/personas")
async def personas() -> dict[str, list[str]]:
    """List the personas the persona_adapt node understands."""
    return {"personas": list(SUPPORTED_PERSONAS)}


@router.get("/tenants", response_model=TenantListResponse)
async def list_tenants(request: Request) -> TenantListResponse:
    """List every tenant available to the frontend selector."""
    registry = getattr(request.app.state, "tenant_registry", None)
    if not registry:
        # No registry wired -> only the built-in default is reachable.
        from backend.tenants import DEFAULT_TENANT

        registry = {DEFAULT_TENANT.tenant_id: DEFAULT_TENANT}
    summaries = [
        TenantSummary(
            tenant_id=cfg.tenant_id,
            display_name=cfg.display_name,
            allowed_personas=list(cfg.allowed_personas),
            default_persona=cfg.default_persona,
            model_tier=dict(cfg.model_tier),
            memory_retention_days=cfg.memory_retention_days,
        )
        for cfg in sorted(registry.values(), key=lambda c: c.tenant_id)
    ]
    return TenantListResponse(tenants=summaries)


@router.get("/tenants/{tenant_id}", response_model=TenantSummary)
async def get_tenant_config(tenant_id: str, request: Request) -> TenantSummary:
    """Return one tenant's public-facing summary or 404."""
    registry = getattr(request.app.state, "tenant_registry", None) or {}
    cfg = registry.get(tenant_id)
    if cfg is None:
        raise HTTPException(404, f"Unknown tenant_id: {tenant_id!r}")
    return TenantSummary(
        tenant_id=cfg.tenant_id,
        display_name=cfg.display_name,
        allowed_personas=list(cfg.allowed_personas),
        default_persona=cfg.default_persona,
        model_tier=dict(cfg.model_tier),
        memory_retention_days=cfg.memory_retention_days,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    graph = request.app.state.graph
    if graph is None:
        raise HTTPException(503, "Graph not initialised")
    persona = normalise_persona(req.persona)
    tenant_id = (req.tenant_id or DEFAULT_TENANT_ID).strip() or DEFAULT_TENANT_ID
    initial_state: dict[str, Any] = {
        "user_id": req.user_id,
        "message": req.message,
        "persona": persona,
        "tenant_id": tenant_id,
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
        tenant_id=result.get("tenant_id", tenant_id),
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
async def get_memory(
    user_id: str, request: Request, tenant_id: str = DEFAULT_TENANT_ID
) -> MemoryListResponse:
    memory = request.app.state.memory
    if memory is None:
        raise HTTPException(503, "Memory not initialised")
    items = memory.get_all(user_id, tenant_id=tenant_id)
    return MemoryListResponse(
        user_id=user_id,
        memories=[MemoryItem(**item) for item in items],
    )


@router.post("/reset/{user_id}")
async def reset_memory(
    user_id: str, request: Request, tenant_id: str = DEFAULT_TENANT_ID
) -> dict[str, Any]:
    memory = request.app.state.memory
    if memory is None:
        raise HTTPException(503, "Memory not initialised")
    removed = memory.reset(user_id, tenant_id=tenant_id)
    return {"user_id": user_id, "tenant_id": tenant_id, "removed": removed}
