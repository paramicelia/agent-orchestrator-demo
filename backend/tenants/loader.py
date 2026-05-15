"""Load tenant YAML configs from disk into a validated registry.

Usage::

    registry = load_all_tenants(Path("tenants"))
    cfg = get_tenant(registry, "acme")

The loader is intentionally boring: it scans a directory, runs every
``*.yaml`` / ``*.yml`` file through :class:`TenantConfig` validation, and
returns a dict keyed by ``tenant_id``. There is **no** code that reads
tenant-specific behaviour anywhere else in the codebase; everything goes
through the validated ``TenantConfig`` object so adding a new tenant is
truly a YAML-only change.

A CLI front-end (``python -m backend.tenants.loader --validate FILE``) is
useful in CI to lint a PR that adds a new tenant before merge — see the
RUNBOOK for the onboarding flow.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from backend.tenants.defaults import DEFAULT_TENANT, DEFAULT_TENANT_ID
from backend.tenants.schemas import TenantConfig

logger = logging.getLogger(__name__)

#: A loaded set of tenants, keyed by ``tenant_id``.
TenantRegistry = dict[str, TenantConfig]


def load_tenant_file(path: Path) -> TenantConfig:
    """Load and validate a single tenant YAML file.

    Raises :class:`ValueError` (wrapping the underlying ``yaml`` or
    ``ValidationError``) so callers can collect errors across many files
    without catching multiple exception types.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path}: could not read file: {exc}") from exc

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping, got {type(data).__name__}"
        )

    try:
        return TenantConfig(**data)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid tenant config:\n{exc}") from exc


def load_all_tenants(tenants_dir: Path) -> TenantRegistry:
    """Load every ``*.yaml`` / ``*.yml`` under ``tenants_dir``.

    Returns a registry keyed by ``tenant_id``. The :data:`DEFAULT_TENANT`
    is always injected at key ``"default"`` so the supervisor has a safe
    fallback even for an empty directory.

    Behaviour notes
    ---------------
    * Missing directory => warning + registry containing only the default.
    * Duplicate ``tenant_id`` => :class:`ValueError`. Two YAMLs cannot
      register the same id silently overwriting each other.
    * Per-file validation failure => :class:`ValueError` with the offending
      file path included, so a CI run pinpoints the broken YAML.
    * Filenames starting with ``_`` are skipped — useful for ``_secrets/``
      or ``_disabled.yaml`` while iterating.
    """
    registry: TenantRegistry = {DEFAULT_TENANT_ID: DEFAULT_TENANT}

    if not tenants_dir.exists():
        logger.warning(
            "tenants directory %s does not exist — using built-in default only",
            tenants_dir,
        )
        return registry

    if not tenants_dir.is_dir():
        raise ValueError(f"{tenants_dir} is not a directory")

    candidate_files = sorted(
        [
            p
            for p in tenants_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in (".yaml", ".yml")
            and not p.name.startswith("_")
        ]
    )

    errors: list[str] = []
    for path in candidate_files:
        try:
            cfg = load_tenant_file(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if cfg.tenant_id in registry and cfg.tenant_id != DEFAULT_TENANT_ID:
            errors.append(
                f"{path}: duplicate tenant_id={cfg.tenant_id!r} "
                f"(already loaded from another file)"
            )
            continue

        registry[cfg.tenant_id] = cfg
        logger.info("tenants: loaded %s from %s", cfg.tenant_id, path.name)

    if errors:
        raise ValueError(
            "Failed to load one or more tenant configs:\n  - "
            + "\n  - ".join(errors)
        )

    logger.info(
        "tenants: registry ready (%d tenants: %s)",
        len(registry),
        ", ".join(sorted(registry.keys())),
    )
    return registry


def get_tenant(registry: TenantRegistry, tenant_id: str | None) -> TenantConfig | None:
    """Look up a tenant by id. Returns ``None`` if unknown.

    Callers that want the safe fallback should do
    ``registry.get(tid) or registry[DEFAULT_TENANT_ID]``; this function
    intentionally returns ``None`` so unknown ids are observable in tests
    and in the supervisor's ``trace`` output.
    """
    if not tenant_id:
        return None
    return registry.get(tenant_id)


# --------------------------------------------------------------------------- #
# CLI: `python -m backend.tenants.loader --validate path/to/tenants/`
# --------------------------------------------------------------------------- #


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="backend.tenants.loader",
        description="Validate tenant YAML configs. Returns non-zero on any error.",
    )
    parser.add_argument(
        "--validate",
        metavar="PATH",
        type=Path,
        required=True,
        help="Path to a tenants/ directory or a single tenant YAML file.",
    )
    args = parser.parse_args(argv)

    target: Path = args.validate
    try:
        if target.is_file():
            cfg = load_tenant_file(target)
            print(f"OK {target}: tenant_id={cfg.tenant_id}")
        else:
            registry = load_all_tenants(target)
            print(f"OK {target}: {len(registry)} tenants loaded")
            for tid, cfg in sorted(registry.items()):
                print(f"  - {tid}: {cfg.display_name}")
    except ValueError as exc:
        print(f"FAIL {target}:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(_main())
