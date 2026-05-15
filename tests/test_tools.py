"""Tool layer tests — Pydantic schemas + handler dispatch + filters."""

from __future__ import annotations

import pytest

from backend.tools import TOOL_SCHEMAS, book_event, handle_tool_call, search_events
from backend.tools.events import Booking, Event


def test_search_events_returns_jazz_in_ny():
    hits = search_events("jazz", "New York")
    assert hits, "expected at least one jazz event in NY"
    assert all(isinstance(h, dict) for h in hits)
    assert any("jazz" in (h["title"].lower() + " ".join(h["tags"])) for h in hits)
    # all must be in New York
    assert all(h["location"].lower() == "new york" for h in hits)


def test_search_events_default_location_online():
    hits = search_events("ai")
    assert hits
    assert all(h["location"].lower() == "online" for h in hits)


def test_search_events_any_location_widens_results():
    only_ny = search_events("jazz", "New York")
    any_loc = search_events("jazz", "any")
    assert len(any_loc) >= len(only_ny)


def test_search_events_unknown_query_empty():
    hits = search_events("underwater basket weaving", "Mars")
    assert hits == []


def test_search_events_caps_at_three():
    # very broad — should still cap at 3
    hits = search_events("", "any")
    assert len(hits) <= 3


def test_book_event_returns_confirmation():
    booking = book_event("evt_jazz_001", "user_42")
    assert booking["event_id"] == "evt_jazz_001"
    assert booking["user_id"] == "user_42"
    assert booking["status"] == "confirmed"
    assert booking["confirmation_code"].startswith("AOD-")
    assert booking["booking_id"].startswith("bk_")


def test_handle_tool_call_dispatch_search():
    out = handle_tool_call("search_events", {"query": "jazz", "location": "New York"})
    assert isinstance(out, list)


def test_handle_tool_call_dispatch_book():
    out = handle_tool_call(
        "book_event", {"event_id": "evt_film_001", "user_id": "u"}
    )
    assert out["status"] == "confirmed"


def test_handle_tool_call_unknown_tool():
    with pytest.raises(ValueError):
        handle_tool_call("delete_universe", {})


def test_tool_schemas_shape():
    # Must satisfy OpenAI/Groq tool-use protocol
    assert isinstance(TOOL_SCHEMAS, list)
    assert len(TOOL_SCHEMAS) == 2
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert names == {"search_events", "book_event"}
    for schema in TOOL_SCHEMAS:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn and "description" in fn and "parameters" in fn
        assert fn["parameters"]["type"] == "object"


def test_event_pydantic_round_trip():
    ev = Event(
        event_id="evt_x",
        title="t",
        venue="v",
        location="online",
        starts_at="2026-01-01T00:00:00+00:00",
        price_usd=0.0,
        tags=["a", "b"],
    )
    d = ev.model_dump()
    assert d["event_id"] == "evt_x"
    ev2 = Event.model_validate(d)
    assert ev2 == ev


def test_booking_pydantic_round_trip():
    b = Booking(
        booking_id="bk_1",
        event_id="evt_x",
        user_id="u",
        status="confirmed",
        confirmation_code="AOD-AAAAAA",
        booked_at="2026-01-01T00:00:00+00:00",
    )
    assert Booking.model_validate(b.model_dump()) == b
