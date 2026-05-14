"""Mem0Client persistence + scoping tests."""

from __future__ import annotations

import pytest


def test_add_and_search(memory):
    memory.add("u1", "I love jazz music and small live shows")
    memory.add("u1", "I dislike crowded clubs")
    hits = memory.search("u1", "where should I go for music tonight?", limit=3)
    assert len(hits) >= 1
    # the jazz memory should rank above the dislike one for a music query
    assert any("jazz" in h["text"].lower() for h in hits)


def test_search_scoped_per_user(memory):
    memory.add("alice", "Alice's favourite cuisine is Korean")
    memory.add("bob", "Bob is allergic to peanuts")
    alice_hits = memory.search("alice", "food", limit=5)
    bob_hits = memory.search("bob", "food", limit=5)
    assert any("Korean" in h["text"] for h in alice_hits)
    assert all("Korean" not in h["text"] for h in bob_hits)
    assert any("peanuts" in h["text"] for h in bob_hits)


def test_get_all_orders_newest_first(memory):
    memory.add("u3", "first")
    memory.add("u3", "second")
    items = memory.get_all("u3")
    assert [i["text"] for i in items[:2]] == ["second", "first"]


def test_reset_removes_only_one_user(memory):
    memory.add("a", "alpha")
    memory.add("b", "beta")
    removed = memory.reset("a")
    assert removed == 1
    assert memory.get_all("a") == []
    assert any(i["text"] == "beta" for i in memory.get_all("b"))


def test_add_rejects_empty_text(memory):
    with pytest.raises(ValueError):
        memory.add("u", "   ")
