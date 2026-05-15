"""Event specialist — uses Groq tool-calling to invoke real functions.

Workflow
--------

    user msg ─► model (with tools advertised) ─► tool_call(s) ─► run tools
                                                      │
                                                      ▼
                                                tool results
                                                      │
                                                      ▼
                                            model (final answer)

The agent advertises two tools — ``search_events`` and ``book_event`` — and
runs the standard OpenAI/Groq tool-use loop. Each tool invocation is
captured into ``state.tool_calls`` so the frontend can render an audit
trail and the eval harness can verify groundedness.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from backend.agents.state import AgentState
from backend.llm.groq_client import GroqClient
from backend.observability import _traceable, traceable_node
from backend.tools import TOOL_SCHEMAS, handle_tool_call

logger = logging.getLogger(__name__)

NAME = "event"
MAX_TOOL_LOOPS = 3  # hard cap so a misbehaving model can't burn through tokens

SYSTEM_PROMPT = """You are the Event Agent inside a social concierge.

You have two tools available:

- search_events(query, location): find events. ALWAYS call this first to
  get real options before suggesting anything. Use a single keyword for
  `query` (e.g. "jazz", "ai", "film", "food"). For `location`, infer the
  city from the user's message; if no city is mentioned, pass "any" so
  the search covers both online and in-person results.
- book_event(event_id, user_id): book a specific event. Call this ONLY if
  the user has explicitly asked to book or confirm a specific event.

After getting tool results, write a short reply (2-3 sentences or <=3
bullet points). ALWAYS reference each suggestion by its real `title` and
include the `venue`. If search_events returned no results, suggest a
different angle and stop — do not call the tool again with the same query.

If you booked anything, surface the confirmation_code.

If memory shows the user already attended something similar, prefer a
different suggestion. No preamble. No outro.
"""


# Keywords we know the stub catalogue indexes — used by the fallback path.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("jazz", ("jazz", "live music", "music", "blues")),
    ("film", ("film", "movie", "cinema", "indie", "premiere")),
    ("ai", ("ai", "artificial intelligence", "ml", "tech", "meetup")),
    ("food", ("food", "bbq", "dinner", "tasting", "eat", "restaurant", "korean")),
    ("free", ("free", "complimentary", "no cost", "no charge")),
)

# Broad-intent words that don't map to a specific catalogue tag — when the
# user just says "plan my weekend" or "I'm bored" we default to the
# safest, most general query.
_GENERIC_INTENT_WORDS: frozenset[str] = frozenset(
    {
        "plan", "planning", "fun", "thing", "things", "activity", "activities",
        "weekend", "tonight", "evening", "night", "morning", "afternoon",
        "bored", "suggest", "suggestion", "ideas", "idea", "help", "recommend",
        "recommendation", "anything", "something", "date",
    }
)

# Generic-intent words map to whichever catalogue tag is most useful as a
# starting recommendation — "live music" tends to win because every event
# in our stub catalogue has at least music-adjacent tags.
_GENERIC_FALLBACK_QUERY = "jazz"

_LOCATIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("New York", ("new york", "nyc", "manhattan", "brooklyn")),
    ("online", ("online", "stream", "virtual", "remote", "free online")),
)


def _infer_query_from_message(message: str) -> str:
    m = (message or "").lower()
    for canonical, hints in _KEYWORDS:
        if any(h in m for h in hints):
            return canonical
    # Fall through: use the first content-bearing word longer than 3 chars
    # that isn't a generic intent word.
    skip = _GENERIC_INTENT_WORDS | {"find", "want", "show", "give"}
    for token in m.split():
        token = token.strip(".,?!")
        if len(token) > 3 and token not in skip:
            return token
    # Pure generic intent — return our safest catalogue-friendly query.
    return _GENERIC_FALLBACK_QUERY


def _infer_location_from_message(message: str) -> str:
    m = (message or "").lower()
    for canonical, hints in _LOCATIONS:
        if any(h in m for h in hints):
            return canonical
    return "any"


def _build_user_message(message: str, memory_context: list[dict[str, Any]], user_id: str) -> str:
    mem_block = ""
    if memory_context:
        lines = [f"- {item.get('text', '')}" for item in memory_context]
        mem_block = "Known about the user:\n" + "\n".join(lines) + "\n\n"
    return f"{mem_block}user_id: {user_id}\nUser message: {message}"


@_traceable(name="event_agent.run_with_tools", run_type="chain")
async def run_with_tools(
    message: str,
    memory_context: list[dict[str, Any]],
    user_id: str,
    client: GroqClient,
) -> tuple[str, list[dict[str, Any]]]:
    """Run the tool-use loop and return (final_text, tool_call_log).

    ``tool_call_log`` is a list of ``{name, arguments, output}`` dicts in the
    order they fired, suitable for rendering in the UI side panel.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_message(message, memory_context, user_id)},
    ]
    tool_log: list[dict[str, Any]] = []

    for loop in range(MAX_TOOL_LOOPS):
        try:
            result = await client.call_with_tools(
                model=client.smart_model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                temperature=0.3,
                max_tokens=600,
            )
        except Exception as exc:  # noqa: BLE001
            # Groq occasionally returns 400 ``tool_use_failed`` when the
            # model emits malformed function-call syntax instead of the
            # structured tool_calls field. Recover by running search_events
            # ourselves with the user's keywords, then asking the model to
            # turn the results into prose.
            err = str(exc)
            logger.warning("event_agent: tool round %d failed: %s", loop, err[:200])
            if "tool_use_failed" in err and loop == 0:
                from backend.tools import search_events as _search_events

                inferred_query = _infer_query_from_message(message)
                inferred_location = _infer_location_from_message(message)
                hits = _search_events(inferred_query, inferred_location)
                tool_log.append(
                    {
                        "name": "search_events",
                        "arguments": {
                            "query": inferred_query,
                            "location": inferred_location,
                            "_via": "fallback",
                        },
                        "output": hits,
                    }
                )
                try:
                    hit_block = json.dumps(hits, indent=2) if hits else "(no matches)"
                    fallback = await client.smart(
                        f"User asked: {message}\n\nsearch_events results:\n{hit_block}\n\n"
                        "Write a short 2-3 sentence reply suggesting one or two of "
                        "these events by title and venue. If none match, suggest a "
                        "different angle.",
                        system=SYSTEM_PROMPT,
                        temperature=0.4,
                        max_tokens=400,
                    )
                    return fallback or "Try a small jazz bar or the local farmers market.", tool_log
                except Exception as exc2:  # noqa: BLE001
                    logger.warning("event_agent: fallback smart failed: %s", exc2)
            return (
                "I hit a backend hiccup trying to look up events — try again in a moment.",
                tool_log,
            )
        tool_calls = result.get("tool_calls") or []
        content = result.get("content") or ""

        if not tool_calls:
            # Model decided it's done — return its text.
            if not content.strip():
                content = "I couldn't find anything matching that — try a different angle?"
            logger.debug("event_agent: tool loop finished after %d round(s)", loop)
            return content, tool_log

        # Append the assistant turn (including its tool_calls metadata) so
        # the follow-up call to the model has the right history.
        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"]),
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Run each tool call and append its result as a tool message.
        for tc in tool_calls:
            try:
                output = handle_tool_call(tc["name"], tc["arguments"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("tool %s failed: %s", tc["name"], exc)
                output = {"error": str(exc)}
            tool_log.append(
                {"name": tc["name"], "arguments": tc["arguments"], "output": output}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(output),
                }
            )

    # Hit the loop cap — ask the model one more time for a wrap-up.
    messages.append(
        {
            "role": "user",
            "content": (
                "You've called enough tools. Now respond to the original request "
                "in 2-3 sentences using the tool results above."
            ),
        }
    )
    wrap = await client.call_with_tools(
        model=client.smart_model,
        messages=messages,
        tools=TOOL_SCHEMAS,
        temperature=0.3,
        max_tokens=400,
        tool_choice="none",
    )
    return wrap.get("content") or "Try one of the options above.", tool_log


@traceable_node("event_agent")
async def event_agent_node(state: AgentState, client: GroqClient) -> dict[str, Any]:
    """LangGraph node wrapper."""
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    text, tool_log = await run_with_tools(
        state["message"],
        state.get("memory_context", []),
        state.get("user_id", "anonymous"),
        client,
    )
    elapsed_ms = int((loop.time() - t0) * 1000)
    trace = [
        f"agent:{NAME} produced {len(text)} chars, used {len(tool_log)} tool calls in {elapsed_ms}ms"
    ]
    for entry in tool_log:
        trace.append(f"  tool:{entry['name']} args={entry['arguments']}")
    return {
        "agent_outputs": [{"agent": NAME, "content": text, "tool_calls": tool_log}],
        "tool_calls": tool_log,
        "trace": trace,
    }
