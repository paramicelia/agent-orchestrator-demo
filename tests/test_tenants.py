"""Multi-tenancy — YAML loader, schema validation, memory isolation, API.

The repo ships three reference tenants (acme, zenith, kids_safe). These
tests pin both the YAML→Pydantic contract and the runtime behaviour
(persona enforcement, memory namespacing, API endpoint shape) so a
regression in any of them breaks CI before it can ship.

No real Groq calls are made; the graph-level tests use the ``mock_groq``
fixture from conftest.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api.routes import router
from backend.memory.mem0_client import Mem0Client
from backend.tenants import (
    DEFAULT_TENANT,
    DEFAULT_TENANT_ID,
    TenantConfig,
    get_tenant,
    load_all_tenants,
    load_tenant_file,
)

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def repo_tenants_dir() -> Path:
    """The shipped tenants/ directory at repo root."""
    return Path(__file__).resolve().parent.parent / "tenants"


def _write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 1. Schema validation
# --------------------------------------------------------------------------- #


def test_tenantconfig_minimal_valid() -> None:
    cfg = TenantConfig(
        tenant_id="t1",
        display_name="Test",
        allowed_personas=["neutral"],
        default_persona="neutral",
    )
    assert cfg.tenant_id == "t1"
    assert cfg.model_tier == {"classifier": "lite", "specialist": "smart"}
    assert cfg.memory_retention_days == 90
    assert cfg.eval_thresholds == {}


def test_tenantconfig_rejects_unknown_persona_in_allow_list() -> None:
    with pytest.raises(ValidationError) as ei:
        TenantConfig(
            tenant_id="t1",
            display_name="Test",
            allowed_personas=["neutral", "klingon"],
            default_persona="neutral",
        )
    assert "klingon" in str(ei.value)


def test_tenantconfig_rejects_default_not_in_allow_list() -> None:
    with pytest.raises(ValidationError) as ei:
        TenantConfig(
            tenant_id="t1",
            display_name="Test",
            allowed_personas=["neutral", "casual"],
            default_persona="formal",  # not in allow list
        )
    assert "default_persona" in str(ei.value)


def test_tenantconfig_rejects_negative_retention() -> None:
    with pytest.raises(ValidationError):
        TenantConfig(
            tenant_id="t1",
            display_name="Test",
            allowed_personas=["neutral"],
            default_persona="neutral",
            memory_retention_days=0,
        )


def test_tenantconfig_rejects_unknown_agent_override() -> None:
    with pytest.raises(ValidationError) as ei:
        TenantConfig(
            tenant_id="t1",
            display_name="Test",
            allowed_personas=["neutral"],
            default_persona="neutral",
            agent_prompts={"weather_agent": "do weather"},
        )
    assert "weather_agent" in str(ei.value)


def test_tenantconfig_rejects_bad_model_tier_value() -> None:
    with pytest.raises(ValidationError):
        TenantConfig(
            tenant_id="t1",
            display_name="Test",
            allowed_personas=["neutral"],
            default_persona="neutral",
            model_tier={"classifier": "gigabrain"},  # unknown tier
        )


def test_tenantconfig_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TenantConfig(
            tenant_id="t1",
            display_name="Test",
            allowed_personas=["neutral"],
            default_persona="neutral",
            this_field_does_not_exist=True,  # type: ignore[call-arg]
        )


def test_tenantconfig_rejects_bad_id_format() -> None:
    with pytest.raises(ValidationError):
        TenantConfig(
            tenant_id="Has Spaces",
            display_name="Test",
            allowed_personas=["neutral"],
            default_persona="neutral",
        )


def test_tenantconfig_resolve_persona_fallback() -> None:
    cfg = TenantConfig(
        tenant_id="t1",
        display_name="Test",
        allowed_personas=["neutral", "casual"],
        default_persona="casual",
    )
    # Allowed persona returns as-is (normalised).
    assert cfg.resolve_persona("casual") == "casual"
    assert cfg.resolve_persona("CASUAL") == "casual"
    # Disallowed (even if globally valid) falls back to default.
    assert cfg.resolve_persona("formal") == "casual"
    # Unknown also falls back.
    assert cfg.resolve_persona("klingon") == "casual"
    # None falls back through normalise() -> "neutral" -> not in allow list -> default.
    assert cfg.resolve_persona(None) == "casual"


def test_tenantconfig_system_prompt_for_uses_override() -> None:
    cfg = TenantConfig(
        tenant_id="t1",
        display_name="Test",
        allowed_personas=["neutral"],
        default_persona="neutral",
        agent_prompts={"topic_agent": "branded topic prompt"},
    )
    assert cfg.system_prompt_for("topic_agent", "default") == "branded topic prompt"
    assert cfg.system_prompt_for("event_agent", "default fallback") == "default fallback"


# --------------------------------------------------------------------------- #
# 2. YAML loader — shipped tenants + edge cases
# --------------------------------------------------------------------------- #


def test_load_shipped_tenants(repo_tenants_dir: Path) -> None:
    registry = load_all_tenants(repo_tenants_dir)
    # default + 3 shipped
    assert set(registry.keys()) >= {"default", "acme", "zenith", "kids_safe"}
    assert registry["acme"].display_name == "Acme Corp"
    assert registry["acme"].memory_retention_days == 365
    assert registry["zenith"].default_persona == "casual"
    assert registry["zenith"].agent_prompts is not None
    assert "event_agent" in registry["zenith"].agent_prompts
    # kids_safe is locked to a single persona on the lite tier
    assert registry["kids_safe"].allowed_personas == ["elderly-friendly"]
    assert registry["kids_safe"].model_tier["specialist"] == "lite"


def test_load_each_shipped_yaml_individually(repo_tenants_dir: Path) -> None:
    for path in sorted(repo_tenants_dir.glob("*.yaml")):
        cfg = load_tenant_file(path)
        assert cfg.tenant_id == path.stem


def test_get_tenant_unknown_returns_none() -> None:
    registry = {"default": DEFAULT_TENANT}
    assert get_tenant(registry, "unknown") is None
    assert get_tenant(registry, None) is None
    assert get_tenant(registry, "default") is DEFAULT_TENANT


def test_load_all_tenants_missing_dir_returns_default_only(tmp_path: Path) -> None:
    registry = load_all_tenants(tmp_path / "does_not_exist")
    assert list(registry.keys()) == [DEFAULT_TENANT_ID]


def test_load_all_tenants_skips_underscore_prefixed(tmp_path: Path) -> None:
    """Files starting with `_` are intentionally skipped (secrets/disabled)."""
    good = tmp_path / "acme.yaml"
    skipped = tmp_path / "_disabled.yaml"
    _write_yaml(
        good,
        {
            "tenant_id": "acme",
            "display_name": "Acme",
            "allowed_personas": ["neutral"],
            "default_persona": "neutral",
        },
    )
    _write_yaml(
        skipped,
        {
            "tenant_id": "ghost",
            "display_name": "Ghost",
            "allowed_personas": ["neutral"],
            "default_persona": "neutral",
        },
    )
    registry = load_all_tenants(tmp_path)
    assert "acme" in registry
    assert "ghost" not in registry


def test_load_all_tenants_duplicate_id_raises(tmp_path: Path) -> None:
    body = {
        "tenant_id": "twin",
        "display_name": "Twin",
        "allowed_personas": ["neutral"],
        "default_persona": "neutral",
    }
    _write_yaml(tmp_path / "a.yaml", body)
    _write_yaml(tmp_path / "b.yaml", body)
    with pytest.raises(ValueError, match="duplicate tenant_id"):
        load_all_tenants(tmp_path)


def test_load_all_tenants_bad_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("tenant_id: t1\n  bad indent: yes", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid"):
        load_all_tenants(tmp_path)


# --------------------------------------------------------------------------- #
# 3. Memory isolation — tenant scoping
# --------------------------------------------------------------------------- #


def test_memory_isolation_across_tenants(tmp_chroma_dir: str) -> None:
    """A memory written to tenant A must never surface in tenant B's search."""
    mem = Mem0Client(db_path=tmp_chroma_dir)

    mem.add("u1", "Acme user prefers cold brew", tenant_id="acme")
    mem.add("u1", "Zenith user lives in Brooklyn", tenant_id="zenith")

    acme_hits = mem.search("u1", "coffee", limit=5, tenant_id="acme")
    zenith_hits = mem.search("u1", "coffee", limit=5, tenant_id="zenith")

    assert any("cold brew" in h["text"] for h in acme_hits)
    assert all("cold brew" not in h["text"] for h in zenith_hits)
    assert any("Brooklyn" in h["text"] for h in zenith_hits)
    assert all("Brooklyn" not in h["text"] for h in acme_hits)


def test_memory_get_all_scoped_by_tenant(tmp_chroma_dir: str) -> None:
    mem = Mem0Client(db_path=tmp_chroma_dir)
    mem.add("u1", "alpha", tenant_id="acme")
    mem.add("u1", "beta", tenant_id="zenith")
    assert [m["text"] for m in mem.get_all("u1", tenant_id="acme")] == ["alpha"]
    assert [m["text"] for m in mem.get_all("u1", tenant_id="zenith")] == ["beta"]


def test_memory_reset_scoped_by_tenant(tmp_chroma_dir: str) -> None:
    mem = Mem0Client(db_path=tmp_chroma_dir)
    mem.add("u1", "alpha", tenant_id="acme")
    mem.add("u1", "beta", tenant_id="zenith")
    removed = mem.reset("u1", tenant_id="acme")
    assert removed == 1
    assert mem.get_all("u1", tenant_id="acme") == []
    assert [m["text"] for m in mem.get_all("u1", tenant_id="zenith")] == ["beta"]


def test_memory_default_tenant_backward_compat(tmp_chroma_dir: str) -> None:
    """Calls without tenant_id collapse to the 'default' tenant."""
    mem = Mem0Client(db_path=tmp_chroma_dir)
    mem.add("u1", "no-tenant-arg")
    hits = mem.search("u1", "no-tenant-arg", limit=5, tenant_id="default")
    assert any("no-tenant-arg" in h["text"] for h in hits)


# --------------------------------------------------------------------------- #
# 4. Persona enforcement at the supervisor layer
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_persona_falls_back_when_disallowed(mock_groq) -> None:
    """``persona_adapt_node`` swaps in the tenant default when persona is blocked."""
    from backend.agents import persona_adapter

    # Kids-safe-shaped tenant: only "elderly-friendly" allowed.
    cfg = TenantConfig(
        tenant_id="kids_safe",
        display_name="Kids Safe",
        allowed_personas=["elderly-friendly"],
        default_persona="elderly-friendly",
    )
    state = {
        "aggregated_response": "Go grab some jazz.",
        "persona": "gen-z",  # NOT allowed for this tenant
        "tenant_config": cfg,
    }
    # lite() must be called with the elderly-friendly system prompt — we
    # only check the return value here; the prompt switch is exercised
    # by the trace line below.
    mock_groq.lite = AsyncMock(return_value="simple rewrite")  # type: ignore[method-assign]
    out = await persona_adapter.persona_adapt_node(state, mock_groq)
    assert out["final_response"] == "simple rewrite"
    fallback_trace = " ".join(out["trace"])
    assert "kids_safe" in fallback_trace
    assert "gen-z" in fallback_trace
    assert "elderly-friendly" in fallback_trace


@pytest.mark.asyncio
async def test_persona_allowed_is_used_as_is(mock_groq) -> None:
    from backend.agents import persona_adapter

    cfg = TenantConfig(
        tenant_id="acme",
        display_name="Acme",
        allowed_personas=["neutral", "formal", "casual", "gen-z", "elderly-friendly"],
        default_persona="formal",
    )
    state = {
        "aggregated_response": "Visit the jazz bar.",
        "persona": "casual",
        "tenant_config": cfg,
    }
    mock_groq.lite = AsyncMock(return_value="hey, hit up the jazz bar tonight")  # type: ignore[method-assign]
    out = await persona_adapter.persona_adapt_node(state, mock_groq)
    assert "jazz" in out["final_response"]
    # No fallback trace line for an allowed persona.
    assert not any("falling back" in line for line in out["trace"])


# --------------------------------------------------------------------------- #
# 5. Supervisor graph picks up tenant config
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_graph_load_tenant_resolves_known(mock_groq, memory) -> None:
    """A request with tenant_id=acme should resolve to the acme TenantConfig."""
    from backend.agents.supervisor import build_graph

    cfg = TenantConfig(
        tenant_id="acme",
        display_name="Acme",
        allowed_personas=["neutral", "formal"],
        default_persona="formal",
    )
    registry = {DEFAULT_TENANT_ID: DEFAULT_TENANT, "acme": cfg}
    mock_groq.lite_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"selected_agents": ["topic"], "reasoning": "ok"}
    )
    graph = build_graph(mock_groq, memory, tenant_registry=registry)
    out = await graph.ainvoke(
        {
            "user_id": "u",
            "message": "hi",
            "tenant_id": "acme",
            "agent_outputs": [],
            "trace": [],
        }
    )
    assert out["tenant_id"] == "acme"
    assert out["tenant_config"].tenant_id == "acme"
    assert any("tenant.load: acme" in line for line in out["trace"])


@pytest.mark.asyncio
async def test_graph_load_tenant_falls_back_on_unknown(mock_groq, memory) -> None:
    from backend.agents.supervisor import build_graph

    registry = {DEFAULT_TENANT_ID: DEFAULT_TENANT}
    mock_groq.lite_json = AsyncMock(  # type: ignore[method-assign]
        return_value={"selected_agents": ["topic"], "reasoning": "ok"}
    )
    graph = build_graph(mock_groq, memory, tenant_registry=registry)
    out = await graph.ainvoke(
        {
            "user_id": "u",
            "message": "hi",
            "tenant_id": "does_not_exist",
            "agent_outputs": [],
            "trace": [],
        }
    )
    # Fell back to default; trace line records the warning.
    assert out["tenant_id"] == DEFAULT_TENANT_ID
    assert any("unknown tenant_id" in line for line in out["trace"])


# --------------------------------------------------------------------------- #
# 6. API endpoints — /tenants list + /tenants/{id} + chat with tenant_id
# --------------------------------------------------------------------------- #


def _make_app_with_tenants(memory, fake_result, registry):
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    app.state.memory = memory
    app.state.client = None
    app.state.tenant_registry = registry

    class FakeGraph:
        async def ainvoke(self, state):
            # Echo tenant_id back in the result so the route shape can be
            # verified end-to-end.
            payload = dict(fake_result)
            payload["tenant_id"] = state.get("tenant_id", DEFAULT_TENANT_ID)
            return payload

    app.state.graph = FakeGraph()
    return app


def test_api_list_tenants(memory) -> None:
    cfg = TenantConfig(
        tenant_id="acme",
        display_name="Acme Corp",
        allowed_personas=["neutral", "formal"],
        default_persona="formal",
        model_tier={"classifier": "lite", "specialist": "smart"},
        memory_retention_days=365,
    )
    registry = {DEFAULT_TENANT_ID: DEFAULT_TENANT, "acme": cfg}
    app = _make_app_with_tenants(memory, {}, registry)
    with TestClient(app) as client:
        r = client.get("/tenants")
        assert r.status_code == 200
        body = r.json()
        ids = [t["tenant_id"] for t in body["tenants"]]
        assert "default" in ids
        assert "acme" in ids
        acme = next(t for t in body["tenants"] if t["tenant_id"] == "acme")
        assert acme["display_name"] == "Acme Corp"
        assert acme["allowed_personas"] == ["neutral", "formal"]
        assert acme["memory_retention_days"] == 365
        # Public summary must not leak agent_prompts or metadata.
        assert "agent_prompts" not in acme
        assert "metadata" not in acme


def test_api_get_tenant_known(memory) -> None:
    cfg = TenantConfig(
        tenant_id="zenith",
        display_name="Zenith",
        allowed_personas=["neutral", "casual"],
        default_persona="casual",
    )
    registry = {DEFAULT_TENANT_ID: DEFAULT_TENANT, "zenith": cfg}
    app = _make_app_with_tenants(memory, {}, registry)
    with TestClient(app) as client:
        r = client.get("/tenants/zenith")
        assert r.status_code == 200
        assert r.json()["tenant_id"] == "zenith"


def test_api_get_tenant_unknown_returns_404(memory) -> None:
    registry = {DEFAULT_TENANT_ID: DEFAULT_TENANT}
    app = _make_app_with_tenants(memory, {}, registry)
    with TestClient(app) as client:
        r = client.get("/tenants/ghost")
        assert r.status_code == 404


def test_api_chat_accepts_tenant_id(memory) -> None:
    fake = {
        "final_response": "ok",
        "aggregated_response": "ok",
        "selected_agents": ["topic"],
        "memory_context": [],
        "agent_outputs": [],
        "tool_calls": [],
        "trace": [],
    }
    registry = {DEFAULT_TENANT_ID: DEFAULT_TENANT}
    app = _make_app_with_tenants(memory, fake, registry)
    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={
                "user_id": "demo",
                "message": "hi",
                "persona": "neutral",
                "tenant_id": "default",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["tenant_id"] == "default"


def test_api_chat_tenant_id_defaults_when_missing(memory) -> None:
    fake = {
        "final_response": "ok",
        "aggregated_response": "ok",
        "selected_agents": ["topic"],
        "memory_context": [],
        "agent_outputs": [],
        "tool_calls": [],
        "trace": [],
    }
    registry = {DEFAULT_TENANT_ID: DEFAULT_TENANT}
    app = _make_app_with_tenants(memory, fake, registry)
    with TestClient(app) as client:
        r = client.post("/chat", json={"user_id": "demo", "message": "hi"})
        assert r.status_code == 200
        assert r.json()["tenant_id"] == DEFAULT_TENANT_ID
