"""OpenAI-compatible tool schemas advertised to the Groq tool-use endpoint.

Groq's chat completions API accepts the OpenAI ``tools=[{type:"function", ...}]``
shape, so we hand-write the JSON schemas here. Keeping them in one file makes
it easy to confirm the prompt → tool contract during code review.
"""

from __future__ import annotations

from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": (
                "Search the events catalogue for happenings matching the user's "
                "interest. Use this whenever the user asks for an event, a place "
                "to go, a thing to do, or a recommendation tied to a time/place."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text interest, e.g. 'jazz', 'indie film', 'korean bbq'.",
                    },
                    "location": {
                        "type": "string",
                        "description": "City name or 'online'. Default 'online' if not specified.",
                        "default": "online",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_event",
            "description": (
                "Book an event for the user. Call this ONLY after the user has "
                "agreed to book a specific event_id returned by search_events."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_id": {
                        "type": "string",
                        "description": "The event_id returned by a prior search_events call.",
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Identifier of the user making the booking.",
                    },
                },
                "required": ["event_id", "user_id"],
            },
        },
    },
]
