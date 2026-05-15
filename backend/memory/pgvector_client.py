"""PostgreSQL + pgvector long-term memory backend.

This module implements the same public API as ``Mem0Client``
(``add`` / ``search`` / ``get_all`` / ``reset``) but persists vectors into a
Postgres database with the ``pgvector`` extension instead of Chroma. It is
the swap target for deployments that already run Postgres for application
data and want a single, durable store for everything — including long-term
agent memory — without standing up a second vector database.

Design notes
------------
* Embeddings are produced by the same ``sentence-transformers`` model the
  Chroma backend uses (``all-MiniLM-L6-v2``, 384-dim, cosine), so memories
  written by one backend can be re-indexed into the other without retraining.
* Connection is lazy: the constructor does NOT open a connection. The first
  ``add`` / ``search`` / ``get_all`` / ``reset`` call dials Postgres and runs
  the migration in ``backend/memory/migrations/001_memories_pgvector.sql``
  inside an idempotent ``CREATE TABLE IF NOT EXISTS`` + ``CREATE INDEX IF
  NOT EXISTS`` block, so this client is safe to swap in on a fresh database.
* Multi-tenancy is enforced at the query layer: every read or write is
  scoped by ``(tenant, user_id)`` exactly like the Chroma backend. The
  ``tenant`` column has a ``DEFAULT 'default'`` so a single-tenant deploy
  that ignores the new keyword keeps writing readable rows.
* Distance is cosine (``<=>``) and we expose it as ``score = 1.0 - distance``
  to keep the return-shape identical to ``Mem0Client.search``.

CI / tests
----------
The companion test file ``tests/test_memory_pgvector.py`` exercises this
client with a fully mocked ``psycopg.connect`` so neither the test suite
nor CI requires a running Postgres instance.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from backend.config import get_settings

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
SCHEMA_MIGRATION = MIGRATIONS_DIR / "001_memories_pgvector.sql"

# Sentinel tenant id used for the pre-multi-tenancy code path. Memories
# written without an explicit tenant_id collapse onto this so existing
# rows (and callers that don't pass tenant_id) stay reachable.
_DEFAULT_TENANT = "default"


def _normalise_tenant(tenant_id: str | None) -> str:
    if not tenant_id or not tenant_id.strip():
        return _DEFAULT_TENANT
    return tenant_id.strip()


class PgVectorClient:
    """Postgres + pgvector implementation of the mem0-style memory API."""

    TABLE_NAME = "agent_memories"

    def __init__(
        self,
        dsn: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.dsn = dsn or settings.postgres_dsn
        if not self.dsn:
            raise RuntimeError(
                "POSTGRES_DSN is not set. For the pgvector backend, point it at "
                "your Postgres instance, e.g. "
                "postgresql://agent:agent@localhost:5432/agent_demo"
            )
        self.embed_model_name = embed_model or settings.memory_embed_model
        self._embedder: SentenceTransformer | None = None
        self._schema_ready = False

    # ----- internals --------------------------------------------------- #

    def _connect(self) -> psycopg.Connection:
        """Open a fresh connection and register the vector adapter on it.

        We do not pool here — the demo is single-process and connection cost
        on Postgres is negligible compared to LLM latency. A pool can be
        added later by swapping this method for ``psycopg_pool.ConnectionPool``.
        """
        conn = psycopg.connect(self.dsn)
        register_vector(conn)
        return conn

    def _ensure_schema(self, conn: psycopg.Connection) -> None:
        """Idempotently run the schema migration on this connection."""
        if self._schema_ready:
            return
        sql = SCHEMA_MIGRATION.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        self._schema_ready = True
        logger.info("pgvector: schema ready on %s", self._safe_dsn())

    def _safe_dsn(self) -> str:
        """Redact the password from the DSN for log lines."""
        if "@" not in self.dsn:
            return self.dsn
        head, tail = self.dsn.rsplit("@", 1)
        if "//" in head and ":" in head.split("//", 1)[1]:
            scheme, rest = head.split("//", 1)
            user, _pw = rest.split(":", 1)
            return f"{scheme}//{user}:***@{tail}"
        return f"***@{tail}"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Lazy-load the embedder and compute normalised vectors."""
        if self._embedder is None:
            logger.info("Loading sentence-transformers model: %s", self.embed_model_name)
            self._embedder = SentenceTransformer(self.embed_model_name)
        vectors = self._embedder.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    # ----- public API (mirrors Mem0Client) ----------------------------- #

    def add(
        self,
        user_id: str,
        text: str,
        metadata: dict[str, Any] | None = None,
        *,
        tenant_id: str | None = None,
    ) -> str:
        """Store one memory item for a user. Returns the memory id."""
        if not text.strip():
            raise ValueError("Memory text must be non-empty")
        tenant = _normalise_tenant(tenant_id)
        mem_id = str(uuid.uuid4())
        created_at = time.time()
        meta = {
            "user_id": user_id,
            "tenant": tenant,
            "created_at": created_at,
            **(metadata or {}),
        }
        embedding = self._embed([text])[0]

        conn = self._connect()
        try:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"INSERT INTO {self.TABLE_NAME} "
                    "(id, tenant, user_id, text, embedding, metadata, created_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, to_timestamp(%s))",
                    (
                        mem_id,
                        tenant,
                        user_id,
                        text,
                        embedding,
                        psycopg.types.json.Jsonb(meta),
                        created_at,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        logger.debug(
            "pgvector.add tenant=%s user=%s id=%s text=%r",
            tenant,
            user_id,
            mem_id,
            text[:80],
        )
        return mem_id

    def search(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k cosine similarity search scoped to a single (tenant, user)."""
        if not query.strip():
            return []
        tenant = _normalise_tenant(tenant_id)
        embedding = self._embed([query])[0]
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, text, metadata, embedding <=> %s AS distance "
                    f"FROM {self.TABLE_NAME} "
                    "WHERE tenant = %s AND user_id = %s "
                    "ORDER BY distance ASC "
                    "LIMIT %s",
                    (embedding, tenant, user_id, int(limit)),
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        items: list[dict[str, Any]] = []
        for row in rows:
            mem_id, text, meta, distance = row
            items.append(
                {
                    "id": str(mem_id),
                    "text": text,
                    "metadata": meta or {},
                    "score": (1.0 - float(distance)) if distance is not None else None,
                }
            )
        return items

    def get_all(
        self, user_id: str, *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return every memory for a (tenant, user), newest first."""
        tenant = _normalise_tenant(tenant_id)
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, text, metadata FROM {self.TABLE_NAME} "
                    "WHERE tenant = %s AND user_id = %s "
                    "ORDER BY created_at DESC",
                    (tenant, user_id),
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            {"id": str(row[0]), "text": row[1], "metadata": row[2] or {}}
            for row in rows
        ]

    def reset(self, user_id: str, *, tenant_id: str | None = None) -> int:
        """Delete every memory for the given (tenant, user). Returns count removed."""
        tenant = _normalise_tenant(tenant_id)
        conn = self._connect()
        try:
            self._ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.TABLE_NAME} WHERE tenant = %s AND user_id = %s",
                    (tenant, user_id),
                )
                removed = cur.rowcount
            conn.commit()
        finally:
            conn.close()
        return int(removed or 0)
