"""Long-term memory layer.

Two interchangeable backends ship in this package, both implementing the
mem0-style ``add`` / ``search`` / ``get_all`` / ``reset`` surface:

* :class:`Mem0Client` — local-first Chroma + sentence-transformers. Default.
* :class:`PgVectorClient` — PostgreSQL with the pgvector extension. Pick
  this when the host already runs Postgres for app data and a second vector
  database isn't worth the operational cost.

Pick via the ``MEMORY_BACKEND`` env var (``chroma`` / ``pgvector``) — see
``build_memory_client`` for the wiring used by ``backend/main.py``.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from backend.memory.mem0_client import Mem0Client
from backend.memory.pgvector_client import PgVectorClient

logger = logging.getLogger(__name__)


class MemoryClient(Protocol):
    """Structural type both backends satisfy.

    Exists so callers (supervisor, eval, tests) can rely on the shape
    without importing a concrete class. The ``tenant_id`` keyword is
    optional and defaults to ``"default"`` for backward compatibility
    with the pre-multi-tenancy callers.
    """

    def add(
        self,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = ...,
        *,
        tenant_id: str | None = ...,
    ) -> str: ...

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = ...,
        *,
        tenant_id: str | None = ...,
    ) -> list[dict[str, Any]]: ...

    def get_all(
        self, user_id: str, *, tenant_id: str | None = ...
    ) -> list[dict[str, Any]]: ...

    def reset(self, user_id: str, *, tenant_id: str | None = ...) -> int: ...


def build_memory_client(backend: str | None = None) -> MemoryClient:
    """Build the memory client requested by config or by explicit arg.

    Unknown values fall back to ``chroma`` with a warning so a typo in the
    env var never takes the demo offline.
    """
    from backend.config import get_settings

    settings = get_settings()
    name = (backend or settings.memory_backend or "chroma").strip().lower()
    if name == "pgvector":
        logger.info("memory backend: pgvector")
        return PgVectorClient()
    if name != "chroma":
        logger.warning("Unknown MEMORY_BACKEND=%r, falling back to chroma", name)
    logger.info("memory backend: chroma")
    return Mem0Client()


__all__ = ["MemoryClient", "Mem0Client", "PgVectorClient", "build_memory_client"]
