"""Mem0-style long-term memory layer.

Design note
-----------
The official ``mem0ai`` package does not ship a first-class HuggingFace
embedder provider as of this writing, and pinning paid OpenAI embeddings is
not acceptable for an open-source portfolio demo. So we implement a thin
mem0-style wrapper directly on top of ``chromadb`` + ``sentence-transformers``:

    add(user_id, text)     -> upsert memory item with embedded vector
    search(user_id, query) -> top-k semantic search scoped per user
    get_all(user_id)       -> list memories for debug/demo
    reset(user_id)         -> wipe user's namespace

This matches the surface area of ``mem0.Memory`` while staying free, local
and reproducible in CI.

Multi-tenancy
-------------
All four methods accept an optional ``tenant_id`` keyword. When provided,
the namespace key becomes ``{tenant_id}:{user_id}`` (stored in the
``tenant`` + ``user_id`` metadata fields), so a memory written under one
tenant is never returned to another. The legacy single-tenant key format
is preserved when ``tenant_id`` is ``None`` or ``"default"`` so existing
callers and stored data keep working unchanged.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from backend.config import get_settings

logger = logging.getLogger(__name__)

# Sentinel tenant id used for the pre-multi-tenancy code path. Memories
# written without an explicit tenant_id collapse onto this so old data
# stays reachable.
_DEFAULT_TENANT = "default"


def _normalise_tenant(tenant_id: str | None) -> str:
    """Coerce ``None`` / empty string to the default-tenant sentinel."""
    if not tenant_id or not tenant_id.strip():
        return _DEFAULT_TENANT
    return tenant_id.strip()


class Mem0Client:
    """Local mem0-style memory layer (Chroma + sentence-transformers)."""

    COLLECTION_NAME = "agent_memories"

    def __init__(
        self,
        db_path: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self.db_path = db_path or settings.memory_db_path
        self.embed_model_name = embed_model or settings.memory_embed_model

        self._client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # Embedder is heavy — load lazily.
        self._embedder: SentenceTransformer | None = None

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._embedder is None:
            logger.info("Loading sentence-transformers model: %s", self.embed_model_name)
            self._embedder = SentenceTransformer(self.embed_model_name)
        vectors = self._embedder.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    @staticmethod
    def _where(user_id: str, tenant_id: str) -> dict[str, Any]:
        """Chroma where-clause scoping queries to a single (tenant, user)."""
        return {"$and": [{"user_id": user_id}, {"tenant": tenant_id}]}

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
        meta = {
            "user_id": user_id,
            "tenant": tenant,
            "created_at": time.time(),
            **(metadata or {}),
        }
        embedding = self._embed([text])[0]
        self._collection.add(
            ids=[mem_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[meta],
        )
        logger.debug(
            "memory.add tenant=%s user=%s id=%s text=%r",
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
        """Top-k semantic search scoped to a single (tenant, user)."""
        if not query.strip():
            return []
        tenant = _normalise_tenant(tenant_id)
        embedding = self._embed([query])[0]
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            where=self._where(user_id, tenant),
        )
        items: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for mem_id, doc, meta, dist in zip(ids, docs, metas, distances, strict=False):
            items.append(
                {
                    "id": mem_id,
                    "text": doc,
                    "metadata": meta or {},
                    "score": 1.0 - float(dist) if dist is not None else None,
                }
            )
        return items

    def get_all(
        self, user_id: str, *, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Return all memories for a (tenant, user) (for debug/demo)."""
        tenant = _normalise_tenant(tenant_id)
        result = self._collection.get(where=self._where(user_id, tenant))
        items: list[dict[str, Any]] = []
        ids = result.get("ids", []) or []
        docs = result.get("documents", []) or []
        metas = result.get("metadatas", []) or []
        for mem_id, doc, meta in zip(ids, docs, metas, strict=False):
            items.append({"id": mem_id, "text": doc, "metadata": meta or {}})
        # newest first
        items.sort(key=lambda x: x["metadata"].get("created_at", 0), reverse=True)
        return items

    def reset(self, user_id: str, *, tenant_id: str | None = None) -> int:
        """Delete every memory for the given (tenant, user). Returns count."""
        existing = self.get_all(user_id, tenant_id=tenant_id)
        if not existing:
            return 0
        self._collection.delete(ids=[item["id"] for item in existing])
        return len(existing)
