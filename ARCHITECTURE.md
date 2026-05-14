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
              │  classify_intent   │   8B "lite" call → selected_agents = ["topic","event"]
              └────────┬───────────┘
                       │  conditional edge: route_to_agents()
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌───────────┐  ┌────────────┐  ┌───────────┐
  │ topic_ag  │  │ people_ag  │  │ event_ag  │     (any 1..3 of these fire in parallel)
  └─────┬─────┘  └─────┬──────┘  └─────┬─────┘
        │              │               │
        └──────────────┼───────────────┘
                       ▼
              ┌─────────────────┐
              │   aggregate     │   merges drafts → final_response (70B)
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
```

So when `topic_agent` returns `[{"agent": "topic", ...}]` and `event_agent`
returns `[{"agent": "event", ...}]` simultaneously, LangGraph merges them
into `[{"agent": "topic", ...}, {"agent": "event", ...}]` before
`aggregate` runs. No locks, no race conditions.

---

## 2. Model tiering

```
┌────────────────────────────────────────────────────────────┐
│ Tier   │ Model                       │ Used for             │
├────────┼─────────────────────────────┼──────────────────────┤
│ lite   │ llama-3.1-8b-instant        │ intent_classifier    │
│        │                             │ save_memory extract  │
├────────┼─────────────────────────────┼──────────────────────┤
│ smart  │ llama-3.3-70b-versatile     │ topic_agent          │
│        │                             │ people_agent         │
│        │                             │ event_agent          │
│        │                             │ aggregate            │
└────────┴─────────────────────────────┴──────────────────────┘
```

Decision rule: anything that requires **judgement under context** goes to
smart. Anything that produces **structured routing** or a **single short
fact** goes to lite. With a 3-specialist run we make roughly 2 lite calls
+ 4 smart calls per turn, so the lite tier cuts our bill by ~30 % vs a
smart-everywhere design.

Both models are accessed via the same `GroqClient.smart()` / `.lite()`
helpers, so swapping providers (Anthropic, OpenAI, OpenRouter) is one file.

---

## 3. Memory read / write cycle

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

### Why a custom mem0-style wrapper instead of the `mem0ai` SDK?

The `mem0ai` SDK at the time of writing requires either an OpenAI / Cohere /
Together API key for embeddings, or a HuggingFace endpoint, both of which
would either (a) cost money to run the demo or (b) require a hosted
inference URL. For a portfolio demo we want **zero paid dependencies** and
**full offline reproducibility** so CI can run on a free GitHub Actions
runner.

So we wrap Chroma + `sentence-transformers/all-MiniLM-L6-v2` directly with
the same `add` / `search` / `get_all` / `reset` surface as `mem0.Memory`:

```python
class Mem0Client:
    def add(self, user_id: str, text: str, metadata: dict | None = None) -> str: ...
    def search(self, user_id: str, query: str, limit: int = 5) -> list[dict]: ...
    def get_all(self, user_id: str) -> list[dict]: ...
    def reset(self, user_id: str) -> int: ...
```

Switching back to the upstream SDK is one import line; the call sites do
not change.

### Per-user scoping

Every record carries `metadata.user_id = X` and every query passes
`where={"user_id": X}` to Chroma. Tested explicitly in
`tests/test_memory.py::test_search_scoped_per_user` — Alice and Bob cannot
see each other's memories.

---

## 4. Async data flow inside a single turn

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
      │        └── await client.lite_json(...)  ← 8B
      │
      ├── parallel:
      │        await topic_agent_node(state)    ← 70B  ┐
      │        await event_agent_node(state)    ← 70B  ├─ asyncio.gather under the hood
      │        await people_agent_node(state)   ← 70B  ┘
      │
      ├── await aggregate_node(state)
      │        └── await client.smart(...)      ← 70B
      │
      └── await save_memory_node(state)
               └── await client.lite(...)       ← 8B
               └── memory.add(...)
      │
      ▼
return ChatResponse(...)
```

Everything I/O-bound is `async`, so the FastAPI worker can keep serving
other requests while a long 70B call is in flight.

---

## 5. Failure modes & graceful degradation

| Failure                              | Behaviour                                             |
|--------------------------------------|--------------------------------------------------------|
| `GROQ_API_KEY` not set at import     | App starts; first `/chat` call returns HTTP 503 with a clear message |
| Intent classifier returns garbage    | Falls back to `["topic"]` so the user still gets a reply |
| Specialist agent raises              | LangGraph surfaces the error in `/chat` 500; other agents in the same fan-out still complete |
| `save_memory` extract fails          | Trace logs the failure, turn still returns to the user |
| Chroma directory missing             | `PersistentClient` creates it on first call          |

---

## 6. What this maps to on the JD

> Building production-grade, low-latency, multi-agent solutions on top of an
> orchestration framework (LangGraph, AutoGen, Google ADK).
> Designing and implementing intent recognition logic.
> Building tiered model architectures.
> Managing long-term memory.
> Async services with FastAPI and asyncpg.
> Docker + observability + CI.

Each row above is one folder in this repo.
