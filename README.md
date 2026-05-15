# Agent Orchestrator Demo

[![CI](https://github.com/paramicelia/agent-orchestrator-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/paramicelia/agent-orchestrator-demo/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Eval](https://img.shields.io/badge/eval-8.3%2F10-brightgreen.svg)](eval/results.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A production-shaped **multi-agent chat system** that closes the gap between
> a toy LangGraph demo and the patterns AI Engineer / ML Engineer job
> descriptions actually ask for:
>
> - **Tool use / function calling** for action-level workflows
> - **LangGraph supervisor + intent routing** for multi-agent orchestration
> - **Long-term memory** (mem0-style, Chroma + sentence-transformers, no paid embeddings)
> - **Persona adapter** as a context-aware translation step
> - **LLM-as-judge eval harness** with a fixed dataset
> - **LangSmith tracing** (opt-in, graceful no-op without a key)
> - **Model tiering** between a cheap 8B router and a 70B reasoner

---

## Why this exists

Production assistant products are built from six moving parts:

1. An **intent layer** that decides *which* specialists to wake up.
2. A pool of **specialist agents** that produce focused suggestions.
3. **Tool use** so at least one specialist can take *action*, not just talk.
4. An **aggregator** that fuses drafts into one reply.
5. A **persona adapter** that re-renders the reply for the audience.
6. A **memory layer** that survives sessions and personalises everything.

This repo implements all six in ~1100 lines of Python, ships them as a
runnable FastAPI service with a one-page chat UI, **and** provides an
LLM-as-judge eval harness that scores composite quality at **8.3 / 10** on
the shipped 10-turn dataset.

---

## Architecture

```
                ┌─────────────────────────────┐
   user msg ──► │  FastAPI  POST /chat        │
                └──────────────┬──────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │  LangGraph supervisor                              │
        │                                                    │
        │  load_memory ─► classify_intent (8B lite) ──┐      │
        │                                             │      │
        │   ┌──────────────────────────────┐          │      │
        │   │ topic_agent  (70B smart)     │ ◄────────┤      │
        │   │ people_agent (70B smart)     │ ◄────────┤      │
        │   │ event_agent  (70B + TOOLS)   │ ◄────────┘      │
        │   │   ├─ search_events()                           │
        │   │   └─ book_event()                              │
        │   └────────────┬──────────────────┘                │
        │                ▼                                   │
        │            aggregate    (70B smart)                │
        │                ▼                                   │
        │            persona_adapt (8B lite)                 │
        │                ▼                                   │
        │            save_memory ──► END                     │
        └────────────────┬──────────────────────────────────┘
                         │
                         ▼               ┌───────────────────────┐
                ┌────────────────┐  ◄──► │ LangSmith (optional)  │
                │ mem0-style     │       │ tracing for every turn│
                │ memory layer   │       └───────────────────────┘
                │ Chroma + ST    │
                └────────────────┘
```

Full topology, fan-out/fan-in semantics, tool-use loop and persona pipeline
are documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Agent orchestration | **LangGraph** (`StateGraph`) | Fan-out / fan-in, conditional edges, typed state |
| Tool use | **Groq function calling** | OpenAI-compatible protocol, sub-second latency |
| LLM provider | **Groq** | Cheapest 8B + competitive 70B |
| Lite tier | `llama-3.1-8b-instant` | Intent routing, memory extraction, persona rewrite |
| Smart tier | `llama-3.3-70b-versatile` | Specialist agents + aggregator + LLM judge |
| Long-term memory | **Chroma + sentence-transformers** | Local persistence, no paid embeddings, mem0-style API |
| API | **FastAPI** async + Pydantic v2 | Type-safe, OpenAPI for free |
| Container | Multi-stage **Dockerfile** + `docker-compose.yml` | Reproducible runtime |
| Eval | **LLM-as-judge** over fixed dataset | Reproducible quality bar, runs locally on demand |
| Tracing | **LangSmith** (opt-in) | Per-turn graph traces in the cloud, no-op without a key |
| CI | **GitHub Actions**: ruff + mypy + pytest | Green badge on every push |

> **Honest design note:** the official `mem0ai` SDK does not ship a HuggingFace
> embedder provider, and this repo refuses to pin paid OpenAI embeddings for a
> public demo. So the memory layer is a **mem0-style wrapper** on Chroma +
> sentence-transformers (`backend/memory/mem0_client.py`). API surface
> (`add` / `search` / `get_all` / `reset`) matches `mem0.Memory` so swapping
> back to the SDK is a one-import change.

---

## Quick start

### 1. Local (Python 3.11+)

```bash
git clone https://github.com/paramicelia/agent-orchestrator-demo.git
cd agent-orchestrator-demo

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your GROQ_API_KEY from https://console.groq.com/keys

make dev          # or: uvicorn backend.main:app --reload
```

Open <http://localhost:8000> for the chat UI, or hit the API directly:

```bash
curl -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"demo","message":"Find me a jazz event tonight in New York.","persona":"casual"}'
```

### 2. Docker

```bash
cp .env.example .env  # set GROQ_API_KEY
docker compose up --build
```

### 3. Smoke demo (proves memory persistence)

```bash
GROQ_API_KEY=gsk_... make smoke
```

---

## Tool use

The `event_agent` is a fully agentic specialist — it runs the standard
**model → tool call → tool result → model** loop using Groq's OpenAI-compatible
function-calling API. Two tools are exposed:

| Tool | Signature | Output |
|---|---|---|
| `search_events` | `(query: str, location: str = "online") -> list[Event]` | Up to 3 matching events from a stub catalogue |
| `book_event`    | `(event_id: str, user_id: str) -> Booking` | Mock confirmation payload |

Schemas live in [`backend/tools/schemas.py`](backend/tools/schemas.py);
implementations + Pydantic models in [`backend/tools/events.py`](backend/tools/events.py).

**Sample turn from a live eval run:**

```text
User:   Find me a jazz event tonight in New York.

→ event_agent issues tool call:
    search_events(query="jazz", location="New York")
→ tool returns:
    [{"event_id":"evt_jazz_001",
      "title":"Vanguard Trio — Late Set",
      "venue":"Smoke Jazz Club", ...}]
→ event_agent uses the result:
    "The 'Vanguard Trio — Late Set' at Smoke Jazz Club is a great option
     for live jazz, $35, intimate venue..."
```

In production these stubs would back onto Ticketmaster / Eventbrite / a
SQL catalogue — the agent contract stays identical.

---

## Persona adaptation

After `aggregate` produces a reply, `persona_adapt` rewrites it in the
requested tone using the 8B tier. Supported personas:

| Persona | Use |
|---|---|
| `neutral` (default) | No-op, returns aggregator output verbatim — saves a Groq call |
| `formal` | Polished business-email register |
| `casual` | Friendly, contractions, conversational |
| `gen-z` | Short punchy lines, lowercase, light slang |
| `elderly-friendly` | Simple words, short sentences, no jargon |

Pass via `POST /chat`:

```json
{"user_id":"demo","message":"What should I read?","persona":"gen-z"}
```

This closes the JD's **"context-aware translation systems"** ask: same
factual content, different audience-adapted surface form.

---

## Eval harness

LLM-as-judge over a fixed 10-turn dataset (`eval/dataset.json`) covering
single-intent, multi-intent, tool-use, memory-grounded, ambiguous,
off-topic and persona-rewrite scenarios.

Metrics:

- `intent_match` (0..1) — Jaccard between expected vs. actual selected agents
- `helpfulness` (1..5) — Likert score from `llama-3.3-70b-versatile` judge
- `groundedness` (0..1) — does the reply use memory and tool output when relevant
- `composite` — weighted mean, scaled to 0..10 for the README badge

Run locally:

```bash
GROQ_API_KEY=gsk_... make eval
# writes eval/results.json (gitignored) + eval/results.md (committed)
```

Current shipped score: **8.3 / 10** composite. CI runs
`tests/test_eval_smoke.py` with mocked Groq calls instead, so no quota is
spent in GitHub Actions.

### Latest results

| ID | Category | Expected | Actual | Intent | Helpful | Ground | Composite |
|---|---|---|---|---|---|---|---|
| t01 | single_intent_tool_use | event | event | 1.00 | 5/5 | 1.00 | 10.00/10 |
| t02 | single_intent | topic | topic | 1.00 | 5/5 | 1.00 | 10.00/10 |
| t03 | single_intent | people | people | 1.00 | 4/5 | 1.00 | 8.80/10 |
| t04 | multi_intent | event, people | event | 0.50 | 4/5 | 1.00 | 7.50/10 |
| t05 | multi_intent | topic, people, event | event, people | 0.67 | 3/5 | 1.00 | 6.70/10 |
| t06 | ambiguous | topic | topic, event | 0.50 | 4/5 | 1.00 | 7.50/10 |
| t07 | off_topic_safe_fallback | topic | topic | 1.00 | 5/5 | 1.00 | 10.00/10 |
| t08 | memory_grounded | event | event, topic | 0.50 | 5/5 | 0.80 | 8.20/10 |
| t09 | persona_rewrite | topic | topic | 1.00 | 2/5 | 1.00 | 6.20/10 |
| t10 | persona_rewrite_tool_use | event | event | 1.00 | 4/5 | 0.80 | 8.20/10 |

Full per-turn breakdown including tool calls and judge reasoning: [`eval/results.md`](eval/results.md).

---

## Observability (LangSmith)

LangSmith tracing is opt-in. The default `.env.example` ships with it
disabled so the demo runs anywhere.

To enable:

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=agent-orchestrator-demo
```

`backend/observability.py` propagates these into the process environment
on startup; the LangChain callback manager auto-instruments every node
in the LangGraph supervisor. With no key set, `init_tracing()` returns
`False` and is a complete no-op — local runs and CI never hit the network.

---

## Demo flow

A representative 4-turn session showing the full stack:

| Turn | User message | Persona | Agents fired | Tools called | What it shows |
|---|---|---|---|---|---|
| 1 | "Find me a jazz event tonight in New York." | neutral | event | `search_events(query=jazz, location=New York)` | Tool use grounds reply in real catalogue data |
| 2 | "Book the first one." | neutral | event | `book_event(event_id=evt_jazz_001, ...)` | Action-level workflow — confirmation code |
| 3 | "Who from my circle would enjoy that?" | casual | people | – | Memory recall + persona rewrite (casual tone) |
| 4 | "Summarise everything for my grandparents." | elderly-friendly | topic | – | Same content, simpler audience-adapted phrasing |

---

## Project structure

```
agent-orchestrator-demo/
├── backend/
│   ├── agents/
│   │   ├── supervisor.py          ← LangGraph StateGraph wiring
│   │   ├── intent_classifier.py   ← 8B router → list[agent]
│   │   ├── topic_agent.py         ← 70B specialist
│   │   ├── people_agent.py        ← 70B specialist
│   │   ├── event_agent.py         ← 70B + tool-use loop
│   │   ├── aggregator.py          ← merges drafts → one reply
│   │   ├── persona_adapter.py     ← 8B re-renders in target tone
│   │   └── state.py               ← TypedDict + reducers
│   ├── tools/
│   │   ├── events.py              ← Pydantic-typed search/book stubs
│   │   └── schemas.py             ← OpenAI-compat tool schemas
│   ├── memory/mem0_client.py      ← Chroma + sentence-transformers
│   ├── llm/groq_client.py         ← async Groq wrapper + tiering + tools
│   ├── api/routes.py              ← /chat /memory /healthz /reset /personas
│   ├── observability.py           ← LangSmith opt-in tracing
│   ├── config.py                  ← pydantic-settings
│   └── main.py                    ← FastAPI app + lifespan
├── frontend/index.html            ← single-file chat UI w/ tool & persona panels
├── eval/
│   ├── dataset.json               ← 10 fixed turns covering every path
│   ├── judge.py                   ← LLM-as-judge scoring fn
│   ├── run_eval.py                ← async runner → results.json + .md
│   └── results.md                 ← latest eval table
├── tests/                         ← 68 unit tests, all mocked
├── scripts/smoke_demo.py          ← runnable 3-turn demo
├── Dockerfile + docker-compose.yml
├── Makefile                       ← make {dev,test,lint,eval,smoke,...}
└── .github/workflows/ci.yml       ← ruff + mypy + pytest
```

---

## Tests

```bash
make test          # or: pytest tests/ -v
```

68 unit tests, all mocked. The eval **smoke** test (`test_eval_smoke.py`)
runs three hardcoded turns through the graph with a mocked Groq client and
the mocked judge, validating the full scoring pipeline without touching
the network. The full LLM-as-judge eval is reserved for `make eval`.

Coverage by file:

- `test_intent_classifier.py` — single / multi / unknown / empty / malformed.
- `test_supervisor_graph.py` — topology, fan-out end-to-end.
- `test_memory.py` — per-user scoping, ordering, reset, empty rejection.
- `test_aggregator.py` — empty / single / multi-draft merge.
- `test_api.py` — all endpoints + persona validation.
- `test_tools.py` — Pydantic schemas, dispatch, filtering, caps.
- `test_event_agent_tool_use.py` — full tool-use loop + recovery paths.
- `test_persona_adapter.py` — neutral no-op + each persona rewrite.
- `test_observability.py` — LangSmith init gating.
- `test_eval_smoke.py` — end-to-end scoring with mocked Groq.

---

## Roadmap

- [ ] Real events backend (Ticketmaster / Eventbrite) behind the tool layer.
- [ ] WebSocket streaming of agent tokens to the frontend.
- [ ] `recall_agent` — pure memory-only path ("what did we talk about last week?").
- [ ] CI nightly eval against Groq (manually triggered, not on every PR).
- [ ] Swap mem0-style wrapper to Vertex Memory Bank for GCP deployments.

---

## Author

Built by **Danilo Polishchuk** as a portfolio piece for AI Agent / ML
Engineer roles. Other deployed work: a voice loan-application bot in
production for Hay Credito, a multi-agent bridge that orchestrates 4
Claude instances, and a RAG support assistant. Reach out via GitHub.
