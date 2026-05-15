"""Built-in fallback tenant.

A request that omits ``tenant_id`` or carries an unknown one is routed to
this in-process default. Keeping it in code rather than in a YAML means
the demo always boots cleanly even if the ``tenants/`` directory is empty
or unreadable (e.g. a misconfigured Docker volume mount).

The defaults below mirror the pre-multi-tenancy behaviour exactly so the
backward-compat tests pass without code-path branching.
"""

from __future__ import annotations

from backend.tenants.schemas import TenantConfig

DEFAULT_TENANT_ID: str = "default"

DEFAULT_TENANT: TenantConfig = TenantConfig(
    tenant_id=DEFAULT_TENANT_ID,
    display_name="Default (built-in fallback)",
    allowed_personas=["neutral", "formal", "casual", "gen-z", "elderly-friendly"],
    default_persona="neutral",
    agent_prompts=None,
    model_tier={"classifier": "lite", "specialist": "smart"},
    memory_retention_days=90,
    eval_thresholds={},
    metadata={"builtin": True},
)
