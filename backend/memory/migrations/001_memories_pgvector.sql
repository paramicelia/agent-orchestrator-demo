-- 001_memories_pgvector.sql
-- Initial schema for the pgvector long-term memory backend.
-- Idempotent: safe to run on every startup.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_memories (
    id          UUID PRIMARY KEY,
    -- Tenant scope. Defaults to 'default' so a fresh single-tenant
    -- deployment behaves exactly like the pre-multi-tenancy schema.
    tenant      TEXT NOT NULL DEFAULT 'default',
    user_id     TEXT NOT NULL,
    text        TEXT NOT NULL,
    -- 384 = sentence-transformers/all-MiniLM-L6-v2 dimensionality.
    -- Keep this in sync with PgVectorClient._embed() / Mem0Client._embed().
    embedding   vector(384) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotent ALTER for pre-multi-tenancy databases: add the tenant
-- column to an existing table if it isn't there yet so a rolling upgrade
-- on a populated DB does not need a separate migration.
ALTER TABLE agent_memories
    ADD COLUMN IF NOT EXISTS tenant TEXT NOT NULL DEFAULT 'default';

-- Per (tenant, user) reads are the hot path (search, get_all, reset all
-- scope by both keys), so the composite gets its own b-tree index.
CREATE INDEX IF NOT EXISTS agent_memories_tenant_user_idx
    ON agent_memories (tenant, user_id);

-- IVF-Flat is the right ANN choice for this scale (thousands of vectors per
-- user). lists=100 is a sensible default; tune up for >1M total vectors.
-- The index is created on the cosine distance operator class so the
-- query in PgVectorClient.search (``embedding <=> :q``) can use it.
CREATE INDEX IF NOT EXISTS agent_memories_embedding_idx
    ON agent_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
