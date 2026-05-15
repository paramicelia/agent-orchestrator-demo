"""Pydantic schema for a tenant configuration.

A ``TenantConfig`` is the single source of truth for everything the
supervisor graph reads on a per-tenant basis: persona allow-list, agent
prompt overrides, model tier, memory retention, and the eval-gate
thresholds CI uses before promoting a build.

The schema is intentionally strict — unknown fields are rejected so a
typo in a tenant YAML fails loudly at boot instead of silently routing
traffic to the wrong tier in production.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Importing the persona allow-list from the adapter keeps the validator in
# lockstep with whatever personas the adapter actually knows how to render.
from backend.agents.persona_adapter import SUPPORTED_PERSONAS

# Model-tier names the supervisor recognises. ``GroqClient.smart()`` and
# ``GroqClient.lite()`` are the only two tiers wired today — keeping the
# enum surface small means a misconfigured tenant cannot ask for a tier
# that does not exist.
VALID_MODEL_TIERS: frozenset[str] = frozenset({"smart", "lite"})

# Agents whose system prompts can be overridden per tenant. Keeping this
# closed prevents a YAML typo from creating an "agent" that never runs.
OVERRIDEABLE_AGENTS: frozenset[str] = frozenset({"topic_agent", "people_agent", "event_agent"})

# Eval metrics callers can gate on. Mirror the metric names the eval
# harness emits in eval/run_eval.py.
VALID_EVAL_METRICS: frozenset[str] = frozenset(
    {"intent_match", "helpfulness", "groundedness", "composite"}
)


class TenantConfig(BaseModel):
    """Schema for one tenant's YAML configuration."""

    model_config = ConfigDict(
        extra="forbid",  # typos in YAML fail loudly
        str_strip_whitespace=True,
        # The field is called ``model_tier`` and pydantic v2 reserves the
        # ``model_`` namespace by default; opt out so the legit field
        # name does not collide with framework internals.
        protected_namespaces=(),
    )

    tenant_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="URL-safe tenant identifier (lower-case alnum + - + _).",
    )
    display_name: str = Field(
        ..., min_length=1, max_length=128, description="Human-readable label for the UI."
    )

    allowed_personas: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Personas this tenant is allowed to request. Must be a non-empty "
            f"subset of {sorted(SUPPORTED_PERSONAS)}."
        ),
    )
    default_persona: str = Field(
        ...,
        description=(
            "Persona used when the request omits one OR asks for a persona "
            "not in ``allowed_personas``. Must itself be in ``allowed_personas``."
        ),
    )

    agent_prompts: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional system-prompt overrides per agent name. Keys must be a "
            f"subset of {sorted(OVERRIDEABLE_AGENTS)}. Unset agents use the "
            "built-in prompt."
        ),
    )

    model_tier: dict[str, str] = Field(
        default_factory=lambda: {"classifier": "lite", "specialist": "smart"},
        description=(
            "Tier override per coarse role. Recognised keys are 'classifier' "
            "and 'specialist'; recognised values are 'lite' and 'smart'. A "
            "cost-sensitive tenant can flip both to 'lite' to halve cost."
        ),
    )

    memory_retention_days: int = Field(
        default=90,
        ge=1,
        le=3650,
        description="Soft retention window for stored memories (operational only).",
    )

    eval_thresholds: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-tenant minimum eval scores. CI promotion is blocked when the "
            "shipped eval drops below any threshold for this tenant. Keys must "
            f"be a subset of {sorted(VALID_EVAL_METRICS)}."
        ),
    )

    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Free-form labels (industry, region, contract tier, ...).",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #

    @field_validator("allowed_personas")
    @classmethod
    def _personas_subset(cls, v: list[str]) -> list[str]:
        bad = [p for p in v if p not in SUPPORTED_PERSONAS]
        if bad:
            raise ValueError(
                f"allowed_personas contains unknown persona(s): {bad}. "
                f"Allowed: {sorted(SUPPORTED_PERSONAS)}"
            )
        # De-dupe while preserving order.
        seen: set[str] = set()
        out: list[str] = []
        for p in v:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out

    @field_validator("agent_prompts")
    @classmethod
    def _prompts_keys(cls, v: dict[str, str] | None) -> dict[str, str] | None:
        if v is None:
            return None
        bad = [k for k in v.keys() if k not in OVERRIDEABLE_AGENTS]
        if bad:
            raise ValueError(
                f"agent_prompts has unknown agent key(s): {bad}. "
                f"Allowed: {sorted(OVERRIDEABLE_AGENTS)}"
            )
        for k, prompt in v.items():
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"agent_prompts[{k!r}] must be a non-empty string")
        return v

    @field_validator("model_tier")
    @classmethod
    def _model_tier_values(cls, v: dict[str, str]) -> dict[str, str]:
        bad_keys = [k for k in v.keys() if k not in {"classifier", "specialist"}]
        if bad_keys:
            raise ValueError(
                f"model_tier has unknown role(s): {bad_keys}. "
                "Allowed: ['classifier', 'specialist']"
            )
        bad_vals = [(k, val) for k, val in v.items() if val not in VALID_MODEL_TIERS]
        if bad_vals:
            raise ValueError(
                f"model_tier has unknown tier(s): {bad_vals}. "
                f"Allowed values: {sorted(VALID_MODEL_TIERS)}"
            )
        return v

    @field_validator("eval_thresholds")
    @classmethod
    def _eval_keys(cls, v: dict[str, float]) -> dict[str, float]:
        bad = [k for k in v.keys() if k not in VALID_EVAL_METRICS]
        if bad:
            raise ValueError(
                f"eval_thresholds has unknown metric(s): {bad}. "
                f"Allowed: {sorted(VALID_EVAL_METRICS)}"
            )
        for k, val in v.items():
            if not isinstance(val, int | float):
                raise ValueError(f"eval_thresholds[{k!r}] must be numeric")
            if val < 0:
                raise ValueError(f"eval_thresholds[{k!r}] must be >= 0, got {val}")
        return v

    @model_validator(mode="after")
    def _default_persona_in_allow_list(self) -> TenantConfig:
        if self.default_persona not in self.allowed_personas:
            raise ValueError(
                f"default_persona={self.default_persona!r} is not in "
                f"allowed_personas={self.allowed_personas!r}"
            )
        return self

    # ------------------------------------------------------------------ #
    # Convenience accessors
    # ------------------------------------------------------------------ #

    def resolve_persona(self, requested: str | None) -> str:
        """Return the persona to use, falling back to ``default_persona``.

        The persona adapter calls this so a tenant can effectively whitelist
        which tones are reachable. Resolution rules, in order:

        1. ``None`` / empty string -> ``self.default_persona``.
        2. A globally-known persona that is in ``allowed_personas``
           (case-insensitive) -> that persona.
        3. Anything else (unknown name, globally-known but disallowed) ->
           ``self.default_persona``.

        Never raises — the request still flows even on a typo.
        """
        # Imported here (rather than module top) to avoid a startup cycle:
        # persona_adapter is what defines SUPPORTED_PERSONAS which this
        # module imports at the top.
        from backend.agents.persona_adapter import SUPPORTED_PERSONAS

        if not requested or not requested.strip():
            return self.default_persona
        candidate = requested.strip().lower()
        # Unknown globally -> tenant default. (We do NOT collapse to the
        # global "neutral" sentinel here, otherwise a tenant that does not
        # allow "neutral" would never receive a typo as a default-tenant
        # request and end up with a disallowed persona.)
        if candidate not in SUPPORTED_PERSONAS:
            return self.default_persona
        if candidate in self.allowed_personas:
            return candidate
        return self.default_persona

    def system_prompt_for(self, agent: str, fallback: str) -> str:
        """Return the tenant override for ``agent`` or ``fallback``."""
        if self.agent_prompts and agent in self.agent_prompts:
            return self.agent_prompts[agent]
        return fallback
