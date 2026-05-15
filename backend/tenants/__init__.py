"""Multi-tenancy layer.

The platform is a single shared deployment that serves many customers
("tenants"). Each tenant's behaviour — which personas are allowed, which
agents get smart vs. lite tier, what eval thresholds gate a deploy, how
long memory is retained — is configured by a single YAML file under the
top-level ``tenants/`` directory. Onboarding a new tenant means dropping
in a new YAML and restarting the app. Zero code change.

Public surface
--------------

* :class:`TenantConfig` — Pydantic v2 model for a single tenant config.
* :func:`load_all_tenants` — read every ``*.yaml`` in a directory, validate,
  return a dict keyed by ``tenant_id``.
* :func:`get_tenant` — convenience accessor on a loaded registry.
* :data:`DEFAULT_TENANT` — built-in fallback used when an incoming request
  carries an unknown ``tenant_id`` so the demo never 4xx's on a typo.
"""

from __future__ import annotations

from backend.tenants.defaults import DEFAULT_TENANT, DEFAULT_TENANT_ID
from backend.tenants.loader import (
    TenantRegistry,
    get_tenant,
    load_all_tenants,
    load_tenant_file,
)
from backend.tenants.schemas import TenantConfig

__all__ = [
    "DEFAULT_TENANT",
    "DEFAULT_TENANT_ID",
    "TenantConfig",
    "TenantRegistry",
    "get_tenant",
    "load_all_tenants",
    "load_tenant_file",
]
