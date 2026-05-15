"""PgVectorClient — SQL + schema tests with a fully mocked psycopg connection.

The whole point of these tests is to validate the SQL contract this client
relies on **without** requiring a running Postgres in CI. Each test fakes
``psycopg.connect`` with ``unittest.mock`` and asserts:

* The schema migration runs once and only once per client instance.
* INSERT carries every required column in the right order.
* SELECT scopes by ``user_id`` and orders by cosine distance.
* DELETE returns the rowcount returned by the cursor.
* The DSN is required.

The companion runtime-only smoke test is in ``backend/memory/pgvector_client.py``
docstring — to exercise the live path, boot ``compose.pgvector.yml`` and
flip ``MEMORY_BACKEND=pgvector`` locally.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


class _FakeCursor:
    """Tiny stand-in for psycopg's cursor context manager."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, Any]] = []
        self._next_fetchall: list[tuple[Any, ...]] = []
        self.rowcount = 0

    # Used by the client as `with conn.cursor() as cur:`.
    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        self.executed.append((sql, params))

    def fetchall(self) -> list[tuple[Any, ...]]:
        return list(self._next_fetchall)

    def queue_fetchall(self, rows: list[tuple[Any, ...]]) -> None:
        self._next_fetchall = rows


class _FakeConn:
    """Minimal psycopg-shaped fake. The client only uses cursor / commit / close."""

    def __init__(self) -> None:
        self.cursor_instances: list[_FakeCursor] = []
        self.committed = False
        self.closed = False

    def cursor(self) -> _FakeCursor:
        cur = _FakeCursor()
        self.cursor_instances.append(cur)
        return cur

    def commit(self) -> None:
        self.committed = True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_conn() -> _FakeConn:
    return _FakeConn()


@pytest.fixture
def pg_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the pgvector backend on with a fake DSN."""
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://agent:agent@localhost:5432/agent_demo")
    monkeypatch.setenv("MEMORY_BACKEND", "pgvector")
    from backend.config import get_settings

    get_settings.cache_clear()


@pytest.fixture
def patched_client(pg_settings: None, fake_conn: _FakeConn):
    """Build a PgVectorClient whose connect() returns ``fake_conn`` and
    whose embedder is stubbed with a deterministic 384-dim zero vector."""
    from backend.memory.pgvector_client import PgVectorClient

    client = PgVectorClient()
    with patch.object(client, "_connect", return_value=fake_conn), patch.object(
        client, "_embed", return_value=[[0.0] * 384]
    ):
        # Also register_vector should never be invoked because _connect is mocked.
        yield client, fake_conn


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_missing_dsn_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Constructing the client without a DSN must fail loudly."""
    monkeypatch.setenv("POSTGRES_DSN", "")
    from backend.config import get_settings

    get_settings.cache_clear()
    from backend.memory.pgvector_client import PgVectorClient

    with pytest.raises(RuntimeError, match="POSTGRES_DSN"):
        PgVectorClient()


def test_add_rejects_empty_text(patched_client) -> None:
    client, _conn = patched_client
    with pytest.raises(ValueError):
        client.add("u1", "   ")


def test_add_inserts_with_expected_columns(patched_client) -> None:
    client, conn = patched_client
    mem_id = client.add("alice", "Alice loves jazz", metadata={"source": "test"})
    assert isinstance(mem_id, str) and len(mem_id) > 0

    # Cursor 0 = schema migration. Cursor 1 = the INSERT itself.
    assert len(conn.cursor_instances) == 2
    insert_sql, insert_params = conn.cursor_instances[1].executed[0]
    assert "INSERT INTO agent_memories" in insert_sql
    assert "(id, user_id, text, embedding, metadata, created_at)" in insert_sql
    # 6 bound params: id, user_id, text, embedding, metadata, created_at.
    assert len(insert_params) == 6
    assert insert_params[1] == "alice"
    assert insert_params[2] == "Alice loves jazz"
    assert isinstance(insert_params[3], list) and len(insert_params[3]) == 384
    assert conn.committed is True
    assert conn.closed is True


def test_schema_migration_runs_once(patched_client) -> None:
    """Two writes on the same client => migration runs exactly once."""
    client, conn = patched_client
    client.add("u", "first")
    client.add("u", "second")

    schema_runs = 0
    for cur in conn.cursor_instances:
        for sql, _ in cur.executed:
            if "CREATE TABLE IF NOT EXISTS agent_memories" in sql:
                schema_runs += 1
    assert schema_runs == 1, f"expected schema migration to run once, got {schema_runs}"


def test_search_scopes_by_user_and_orders_by_distance(patched_client) -> None:
    client, conn = patched_client
    # Queue a fake fetchall on the SELECT cursor — cursor index depends on
    # whether schema has run yet; here it has not (fresh client).
    # The client opens cursor 0 for migration, cursor 1 for SELECT.
    # We need to set fetchall *after* the cursor is created, so we use a
    # patcher hook on the connection.
    def cursor_with_rows() -> _FakeCursor:
        cur = _FakeCursor()
        # Distance ascending: best (0.1) first.
        cur.queue_fetchall(
            [
                ("00000000-0000-0000-0000-000000000001", "I love jazz", {"source": "x"}, 0.1),
                ("00000000-0000-0000-0000-000000000002", "Disliked clubs", {}, 0.6),
            ]
        )
        conn.cursor_instances.append(cur)
        return cur

    # Replace cursor() so the SECOND call (the SELECT) returns rows.
    original_cursor = conn.cursor
    call_count = {"n": 0}

    def cursor_dispatch() -> _FakeCursor:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original_cursor()  # migration cursor
        return cursor_with_rows()

    conn.cursor = cursor_dispatch  # type: ignore[method-assign]

    results = client.search("alice", "music tonight", limit=3)

    select_cur = conn.cursor_instances[-1]
    sql, params = select_cur.executed[0]
    assert "FROM agent_memories" in sql
    assert "WHERE user_id = %s" in sql
    assert "embedding <=> %s" in sql
    assert "ORDER BY distance ASC" in sql
    assert params[1] == "alice"
    assert params[2] == 3

    assert len(results) == 2
    # 1 - distance => higher score for closer match
    assert results[0]["score"] == pytest.approx(0.9)
    assert results[1]["score"] == pytest.approx(0.4)
    # Identity preserved in shape (same keys Mem0Client returns)
    assert set(results[0].keys()) == {"id", "text", "metadata", "score"}


def test_search_empty_query_short_circuits(patched_client) -> None:
    client, conn = patched_client
    # Need to make _embed raise so we *prove* it isn't called.
    with patch.object(client, "_embed", side_effect=AssertionError("embed must not run")):
        assert client.search("u", "   ") == []
    # No SQL was issued.
    assert conn.cursor_instances == []


def test_get_all_orders_newest_first(patched_client) -> None:
    client, conn = patched_client

    def cursor_with_rows() -> _FakeCursor:
        cur = _FakeCursor()
        cur.queue_fetchall(
            [
                ("11111111-1111-1111-1111-111111111111", "newer", {}),
                ("22222222-2222-2222-2222-222222222222", "older", {}),
            ]
        )
        conn.cursor_instances.append(cur)
        return cur

    original_cursor = conn.cursor
    call_count = {"n": 0}

    def cursor_dispatch() -> _FakeCursor:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original_cursor()  # migration
        return cursor_with_rows()

    conn.cursor = cursor_dispatch  # type: ignore[method-assign]

    items = client.get_all("alice")

    select_cur = conn.cursor_instances[-1]
    sql, _ = select_cur.executed[0]
    assert "WHERE user_id = %s" in sql
    assert "ORDER BY created_at DESC" in sql
    assert [i["text"] for i in items] == ["newer", "older"]


def test_reset_returns_rowcount(patched_client) -> None:
    client, conn = patched_client

    def cursor_for_delete() -> _FakeCursor:
        cur = _FakeCursor()
        cur.rowcount = 3
        conn.cursor_instances.append(cur)
        return cur

    original_cursor = conn.cursor
    call_count = {"n": 0}

    def cursor_dispatch() -> _FakeCursor:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return original_cursor()  # migration
        return cursor_for_delete()

    conn.cursor = cursor_dispatch  # type: ignore[method-assign]

    removed = client.reset("alice")
    assert removed == 3

    delete_cur = conn.cursor_instances[-1]
    sql, params = delete_cur.executed[0]
    assert "DELETE FROM agent_memories" in sql
    assert "WHERE user_id = %s" in sql
    assert params[0] == "alice"
    assert conn.committed is True


def test_safe_dsn_redacts_password(patched_client) -> None:
    client, _ = patched_client
    safe = client._safe_dsn()
    assert "agent" in safe  # username still visible
    assert "agent:agent" not in safe  # password redacted
    assert "***" in safe


def test_build_memory_client_factory_selects_pgvector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory in backend.memory must wire pgvector when env says so."""
    monkeypatch.setenv("POSTGRES_DSN", "postgresql://agent:agent@localhost:5432/agent_demo")
    monkeypatch.setenv("MEMORY_BACKEND", "pgvector")
    from backend.config import get_settings

    get_settings.cache_clear()

    # Stop the real connect() from running — we only care that the factory
    # returns the right *type*.
    with patch("backend.memory.pgvector_client.psycopg.connect", return_value=MagicMock()):
        from backend.memory import PgVectorClient, build_memory_client

        client = build_memory_client()
        assert isinstance(client, PgVectorClient)


def test_build_memory_client_factory_unknown_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMORY_BACKEND", "redisvec")  # nonsense
    monkeypatch.setenv("POSTGRES_DSN", "")
    from backend.config import get_settings

    get_settings.cache_clear()

    from backend.memory import Mem0Client, build_memory_client

    client = build_memory_client()
    assert isinstance(client, Mem0Client)
