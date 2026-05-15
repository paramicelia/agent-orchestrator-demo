# Architecture

Detailed walk-through of the moving parts. For the elevator pitch see
[`README.md`](README.md).

---

## 1. LangGraph supervisor topology

```
              ┌───────────────┐
              │  load_memory  │   reads top-k memory hits for the user
              └──────┬────────┘
                     │  memory_context populated
                     ▼
              ┌────────────────────┐
              │  classify_intent   │   8B "lite" call → selected_agents = [...]
              └────────┬───────────┘
                       │  conditional edge: route_to_agents()
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌───────────┐  ┌────────────┐  ┌───────────────────┐
  │ topic_ag  │  │ people_ag  │  │ event_ag          │
  │ (70B)     │  │ (70B)      │  │ (70B + TOOL LOOP) │     1..3 fire in parallel
  └─────┬─────┘  └─────┬──────┘  └─────────┬─────────┘
        │              │                   │
        └──────────────┼───────────────────┘
                       ▼
              ┌─────────────────┐
              │   aggregate     │   merges drafts → aggregated_response (70B)
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  persona_adapt  │   rewrites in target tone (8B)
              │                 │   → final_response
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  save_memory    │   extracts a single fact, persists to mem0
              └────────┬────────┘
                       ▼
                      END
```

### Why conditional edges + a reducer?

The classifier returns *a list*, not a single label. We want the chosen
specialists to run **in parallel** and then converge on a single aggregator
node — classic LangGraph fan-out / fan-in.

This works because `AgentState.agent_outputs` is annotated with
`operator.add` as a reducer:

```python
class AgentState(TypedDict, total=False):
    agent_outputs: Annotated[list[dict[str, Any]], operator.add]
    tool_calls:    Annotated[list[dict[str, Any]], operator.add]
```

When `topic_agent` returns `[{"agent": "topic", ...}]` and `event_agent`
returns `[{"agent": "event", ...}]` simultaneously, LangGraph merges them
into `[{"agent": "topic", ...}, {"agent": "event", ...}]` before
`aggregate` runs. No locks, no race conditions.

---

## 2. Tool use inside `event_agent`

The `event_agent` is the one specialist that takes **action**, not just
talks. It uses Groq's OpenAI-compatible function-calling endpoint to run
the standard tool-use loop:

```
   ┌─────────────────┐
   │  user message   │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────────────────────────────────┐
   │  call_with_tools(messages, [search, book])  │   round 1
   └────────┬────────────────────────────────────┘
            │
            ▼
   ┌─────────────────┐
   │  tool_calls?    │
   └─┬───────────┬───┘
     │ yes       │ no
     ▼           ▼
   run tool   return final text
     │
     ▼
   append tool_result to messages
     │
     └─────► loop up to MAX_TOOL_LOOPS=3
```

Two tools are advertised (`backend/tools/schemas.py`):

```python
search_events(query: str, location: str = "online") -> list[Event]
book_event(event_id: str, user_id: str) -> Booking
```

Both have Pydantic schemas (`backend/tools/events.py`) so every output is
type-checked before it hits the model. Each round of the loop appends one
entry to `state.tool_calls` with `{name, arguments, output}`, which the
frontend renders in a side panel and the eval harness uses to score
groundedness.

### Graceful degradation

Llama-3.3-70b occasionally emits malformed `<function=name args>` syntax
instead of structured `tool_calls`. Groq rejects this with HTTP 400
`tool_use_failed`. The agent catches that, infers query + location from
the message via a small keyword table, runs `search_events` directly, and
asks the model to write prose around the real tool output. End user never
sees the underlying failure.

---

## 3. Persona adapter

```
aggregator output ──► persona_adapt (8B) ──► final_response
                          │
                          └─ persona ∈ {neutral, formal, casual, gen-z, elderly-friendly}
```

The aggregator writes its merged reply to `aggregated_response`. The
`persona_adapt` node reads that, looks up the requested persona's
rewrite instructions, and asks the 8B model to re-render the same content
in the target tone. The output is written to `final_response`.

**`neutral` is a no-op short-circuit** — the node returns
`aggregated_response` verbatim without making an LLM call. This keeps the
default path cheap; only users who actively pick a persona pay the extra
8B call (~$0.0001).

The pattern closes the JD's "context-aware translation systems" ask:
identical factual content rendered for different audiences without
re-running the upstream specialists.

---

## 4. Model tiering

```
┌─────────┬─────────────────────────────┬───────────────────────────────┐
│ Tier    │ Model                       │ Used for                      │
├─────────┼─────────────────────────────┼───────────────────────────────┤
│ lite    │ llama-3.1-8b-instant        │ intent_classifier             │
│         │                             │ save_memory extract           │
│         │                             │ persona_adapt rewrite         │
├─────────┼─────────────────────────────┼───────────────────────────────┤
│ smart   │ llama-3.3-70b-versatile     │ topic_agent                   │
│         │                             │ people_agent                  │
│         │                             │ event_agent + tool calls      │
│         │                             │ aggregate                     │
│         │                             │ eval/judge.py LLM-as-judge    │
└─────────┴─────────────────────────────┴───────────────────────────────┘
```

Decision rule: anything that requires **judgement under context** goes to
smart. Anything that produces **structured routing** or a **mechanical
rewrite** goes to lite. With a 3-specialist run we make roughly 3 lite
calls + 5 smart calls per turn (4-6 if the event_agent loops on tools).

---

## 5. Memory read / write cycle

```
                    ┌─────────────────────────┐
   load_memory  ──► │ embed(user_message)     │
                    │ → Chroma .query()       │
                    │ → top-k where user_id=X │
                    └──────────┬──────────────┘
                               │ memory_context = [{text, score, ...}]
                               ▼
                       ...graph runs...
                               │
                    ┌──────────┴──────────────┐
   save_memory  ──► │ lite("extract one fact")│
                    │ → "User loves jazz"     │
                    │ → embed + Chroma .add() │
                    └─────────────────────────┘
```

### Per-user scoping

Every record carries `metadata.user_id = X` and every query passes
`where={"user_id": X}` to Chroma. Tested explicitly in
`tests/test_memory.py::test_search_scoped_per_user` — Alice and Bob cannot
see each other's memories.

---

## 6. LLM-as-judge eval harness

```
   eval/dataset.json (10 turns)
       │
       ▼
   ┌──────────────────────────────┐
   │  for each turn:              │
   │    seed prior_memories       │
   │    state = {user, msg, ...}  │
   │    result = await graph.ainvoke(state)
   │    ▼                         │
   │    intent_match = Jaccard(expected, actual)
   │    ▼                         │
   │    judge.judge_response()    │   ◄── 70B judge in JSON mode
   │      → helpfulness 1-5       │
   │      → groundedness 0-1      │
   │    ▼                         │
   │    composite = weighted_mean │
   └──────────┬───────────────────┘
              ▼
       eval/results.json
       eval/results.md
```

The judge is wired to read the user message, the available memories, the
tool calls actually made, and the final reply. It returns strict JSON so
parsing is reliable; bad responses fall back to a 3/0.5 neutral score.

CI never runs the real judge. `tests/test_eval_smoke.py` runs the same
plumbing against a mocked Groq client to validate the scoring math and
markdown rendering without spending quota.

---

## 7. Async data flow inside a single turn

```
client POST /chat  (FastAPI async handler)
      │
      ▼
graph.ainvoke(state)
      │
      ├── await load_memory_node(state)
      │        └── memory.search(...)
      │
      ├── await classify_node(state)
      │        └── await client.lite_json(...)         ← 8B
      │
      ├── parallel:
      │        await topic_agent_node(state)            ← 70B   ┐
      │        await event_agent_node(state)            ← 70B   │
      │            └── tool loop (search_events, ...)            ├ asyncio.gather
      │        await people_agent_node(state)           ← 70B   ┘
      │
      ├── await aggregate_node(state)
      │        └── await client.smart(...)              ← 70B
      │
      ├── await persona_adapt_node(state)
      │        └── await client.lite(...)               ← 8B (skipped if neutral)
      │
      └── await save_memory_node(state)
               └── await client.lite(...)               ← 8B
               └── memory.add(...)
      │
      ▼
return ChatResponse(...)
```

Everything I/O-bound is `async`, so the FastAPI worker can keep serving
other requests while a long 70B call is in flight.

---

## 8. Observability (LangSmith)

```
┌────────────────────────────────────────────┐
│  backend/observability.py::init_tracing()  │
│                                            │
│  if LANGCHAIN_TRACING_V2 and LANGSMITH_KEY │
│      → propagate env vars                  │
│      → LangChain callback manager picks    │
│        them up automatically               │
│      → every node in the graph appears in  │
│        the LangSmith UI                    │
│                                            │
│  else                                      │
│      → return False, no-op                 │
│      → graph still runs locally            │
└────────────────────────────────────────────┘
```

No code path requires LangSmith credentials. CI never sets the API key,
so tracing is dormant in CI and any local run without `.env` set up.

---

## 9. Failure modes & graceful degradation

| Failure                              | Behaviour                                                  |
|--------------------------------------|-------------------------------------------------------------|
| `GROQ_API_KEY` not set at import     | App starts; first `/chat` returns HTTP 503 with clear msg  |
| Intent classifier returns garbage    | Falls back to `["topic"]` so the user still gets a reply   |
| Groq returns 400 `tool_use_failed`   | event_agent infers query + location and runs the tool itself |
| Specialist agent raises              | LangGraph surfaces error in `/chat` 500; other fan-out agents complete |
| Persona LLM rewrite fails            | Falls back to `aggregated_response` verbatim — never blocks reply |
| `save_memory` extract fails          | Trace logs the failure, turn still returns to the user     |
| Chroma directory missing             | `PersistentClient` creates it on first call                |
| LangSmith key absent                 | `init_tracing()` returns False, no callback manager change |

---

## 10. What this maps to on the JD

> Building production-grade, low-latency, multi-agent solutions on top of an
> orchestration framework (LangGraph, AutoGen, Google ADK).
> Action-level chatbots executing workflows.
> Context-aware translation systems.
> Designing intent recognition logic.
> Long-term memory + RAG.
> Deploying ML features to production with measurable results.
> Async services with FastAPI.
> Docker + observability + CI.

Each row above is one folder in this repo.
