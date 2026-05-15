-- 001_memories_pgvector.sql
-- Initial schema for the pgvector long-term memory backend.
-- Idempotent: safe to run on every startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_memories (
    id          UUID PRIMARY KEY,
    user_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    -- 384 = sentence-transformers/all-MiniLM-L6-v2 dimensionality.
    -- Keep this in sync with PgVectorClient._embed() / Mem0Client._embed().
    embedding   vector(384) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Per-user reads are the hot path (search, get_all, reset all scope by
-- user_id), so the partition key gets its own b-tree index.
CREATE INDEX IF NOT EXISTS agent_memories_user_id_idx
    ON agent_memories (user_id);

-- IVF-Flat is the right ANN choice for this scale (thousands of vectors per
-- user). lists=100 is a sensible default; tune up for >1M total vectors.
-- The index is created on the cosine distance operator class so the
-- query in PgVectorClient.search (``embedding <=> :q``) can use it.
CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx
    ON agent_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
