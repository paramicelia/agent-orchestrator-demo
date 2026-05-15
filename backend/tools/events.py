"""Event tools — Pydantic-typed callables exposed to the LLM via Groq tool-use.

These are deliberately deterministic stubs (no external API) so:

1. CI can run them without any network.
2. The eval harness gets reproducible behaviour for groundedness scoring.
3. Recruiters can run the demo end-to-end without third-party credentials.

In production these would back onto a real events provider (Ticketmaster /
Eventbrite / Foursquare) but the calling convention from the agent's POV
stays identical.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #


class Event(BaseModel):
    """One bookable event returned by ``search_events``."""

    event_id: str = Field(..., description="Stable identifier used by book_event")
    title: str
    venue: str
    location: str
    starts_at: str = Field(..., description="ISO-8601 timestamp")
    price_usd: float
    tags: list[str] = Field(default_factory=list)


class Booking(BaseModel):
    """Confirmation payload returned by ``book_event``."""

    booking_id: str
    event_id: str
    user_id: str
    status: str
    confirmation_code: str
    booked_at: str


# --------------------------------------------------------------------------- #
# Stub data — picked so each query has obvious matches
# --------------------------------------------------------------------------- #

_NOW = datetime(2026, 5, 15, 19, 0, tzinfo=UTC)


def _iso(offset_hours: int) -> str:
    return (_NOW + timedelta(hours=offset_hours)).isoformat()


_FAKE_EVENTS: list[Event] = [
    Event(
        event_id="evt_jazz_001",
        title="Vanguard Trio — Late Set",
        venue="Smoke Jazz Club",
        location="New York",
        starts_at=_iso(4),
        price_usd=35.0,
        tags=["jazz", "live music", "intimate", "nightlife"],
    ),
    Event(
        event_id="evt_jazz_002",
        title="Sunday Brunch Jazz Quartet",
        venue="Blue Note Cafe",
        location="online",
        starts_at=_iso(36),
        price_usd=0.0,
        tags=["jazz", "livestream", "free", "online"],
    ),
    Event(
        event_id="evt_film_001",
        title="A24 Indie Premiere — Q&A with director",
        venue="Metrograph",
        location="New York",
        starts_at=_iso(24),
        price_usd=22.0,
        tags=["film", "indie", "premiere"],
    ),
    Event(
        event_id="evt_food_001",
        title="Korean BBQ Tasting Night",
        venue="Cote Steakhouse",
        location="New York",
        starts_at=_iso(48),
        price_usd=85.0,
        tags=["food", "korean", "bbq", "tasting"],
    ),
    Event(
        event_id="evt_tech_001",
        title="AI Agents Meetup — LangGraph deep dive",
        venue="Brooklyn Brewery",
        location="online",
        starts_at=_iso(72),
        price_usd=0.0,
        tags=["tech", "ai", "meetup", "online", "free"],
    ),
]


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #


def search_events(query: str, location: str = "online") -> list[dict[str, Any]]:
    """Search the fake catalogue and return at most 3 matching events.

    Matching rule: query term hits ``title`` or any ``tag``; location must match
    case-insensitively unless ``location == "any"``.

    Returned as a list of dicts (not Pydantic models) so the result serialises
    cleanly into the LLM message stream.
    """
    q = (query or "").lower().strip()
    loc = (location or "online").lower().strip()
    matches: list[Event] = []
    for ev in _FAKE_EVENTS:
        loc_match = loc == "any" or ev.location.lower() == loc
        text_match = (
            not q
            or q in ev.title.lower()
            or any(q in tag.lower() or tag.lower() in q for tag in ev.tags)
        )
        if loc_match and text_match:
            matches.append(ev)
    matches = matches[:3]
    logger.debug("search_events(query=%r, location=%r) -> %d hits", query, location, len(matches))
    return [m.model_dump() for m in matches]


def book_event(event_id: str, user_id: str) -> dict[str, Any]:
    """Pretend to book an event. Returns a confirmation dict."""
    booking = Booking(
        booking_id=f"bk_{uuid.uuid4().hex[:8]}",
        event_id=event_id,
        user_id=user_id,
        status="confirmed",
        confirmation_code=f"AOD-{uuid.uuid4().hex[:6].upper()}",
        booked_at=_NOW.isoformat(),
    )
    logger.debug("book_event(event_id=%r, user_id=%r) -> %s", event_id, user_id, booking.booking_id)
    return booking.model_dump()


# --------------------------------------------------------------------------- #
# Dispatcher used by the tool-call loop in event_agent
# --------------------------------------------------------------------------- #


def handle_tool_call(name: str, arguments: dict[str, Any]) -> Any:
    """Route a tool call from the LLM to the matching Python function.

    Keeps the tool registry colocated so adding a new tool is one entry.
    """
    if name == "search_events":
        return search_events(
            query=str(arguments.get("query", "")),
            location=str(arguments.get("location", "online")),
        )
    if name == "book_event":
        return book_event(
            event_id=str(arguments.get("event_id", "")),
            user_id=str(arguments.get("user_id", "anonymous")),
        )
    raise ValueError(f"Unknown tool: {name}")
