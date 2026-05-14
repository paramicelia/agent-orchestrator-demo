# Agent Orchestrator Demo

[![CI](https://github.com/paramicelia/agent-orchestrator-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/paramicelia/agent-orchestrator-demo/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A small but production-shaped **multi-agent social concierge** that demonstrates
> the patterns most AI-engineering job descriptions are now asking for:
> **LangGraph supervisor + intent routing**, **long-term memory** (mem0-style,
> running locally on Chroma + sentence-transformers, no paid embeddings), and
> **model tiering** between a cheap 8B router and a 70B reasoner.

---

## Why this exists

Real assistant products like Unikoom's "Bits" (and any "second-brain" social
agent) are built from four moving parts:

1. An **intent layer** that decides *which* specialists to wake up.
2. A pool of **specialist agents** that produce focused suggestions.
3. An **aggregator** that fuses their drafts into one reply.
4. A **memory layer** that survives across sessions and personalises everything.

This repo implements all four in ~700 lines of Python so the architecture stays
readable, and ships them as a runnable FastAPI service with a one-page chat UI.

---

## Architecture

```
                ┌─────────────────────────────┐
   user msg ──► │  FastAPI  POST /chat        │
                └──────────────┬──────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────┐
        │  LangGraph supervisor                          │
        │                                                │
        │  load_memory ─► classify_intent (8B lite) ──┐  │
        │                                             │  │
        │   ┌───────────────────────────────┐         │  │
        │   │  topic_agent  (70B smart)     │ ◄───────┤  │
        │   │  people_agent (70B smart)     │ ◄───────┤  │
        │   │  event_agent  (70B smart)     │ ◄───────┘  │
        │   └────────────┬──────────────────┘            │
        │                ▼                               │
        │            aggregate  (70B smart)              │
        │                ▼                               │
        │            save_memory ──► END                 │
        └─────────────────┬──────────────────────────────┘
                          │
                          ▼
                ┌─────────────────────────────┐
                │  mem0-style memory layer    │
                │  (Chroma + ST embeddings)   │
                └─────────────────────────────┘
```

Full topology, fan-out/fan-in semantics, and the memory read/write cycle are
documented in [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Tech stack

| Concern | Choice | Why |
|---|---|---|
| Agent orchestration | **LangGraph** (`StateGraph`) | First-class fan-out / fan-in, conditional edges, typed state |
| LLM provider | **Groq** | Cheapest 8B + competitive 70B; sub-second latency |
| Lite tier | `llama-3.1-8b-instant` | Intent routing + memory extraction (~$0.05 / 1M tokens) |
| Smart tier | `llama-3.3-70b-versatile` | Specialist agents + aggregator |
| Long-term memory | **Chroma + sentence-transformers** | Local persistence, no paid embeddings, mem0-style API |
| API | **FastAPI** async + Pydantic v2 | Type-safe, OpenAPI for free, easy CORS |
| Container | Multi-stage **Dockerfile** + `docker-compose.yml` | Reproducible runtime |
| CI | **GitHub Actions**: ruff + mypy + pytest | Green badge on every push |

> **Honest design note:** the official `mem0ai` SDK does not ship a HuggingFace
> embedder provider, and this repo refuses to pin paid OpenAI embeddings for a
> public demo. So the memory layer is a **mem0-style wrapper** built directly
> on Chroma + sentence-transformers (`backend/memory/mem0_client.py`). API
> surface (`add` / `search` / `get_all` / `reset`) matches `mem0.Memory` so
> swapping back to the SDK is a one-import change. See `ARCHITECTURE.md` for
> the rationale.

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

uvicorn backend.main:app --reload
```

Open <http://localhost:8000> for the chat UI, or hit the API directly:

```bash
curl -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"demo","message":"plan my Saturday — I like jazz"}'
```

### 2. Docker

```bash
cp .env.example .env  # set GROQ_API_KEY
docker compose up --build
```

### 3. Smoke demo (proves memory persistence)

```bash
GROQ_API_KEY=gsk_... python scripts/smoke_demo.py
```

Three turns are sent under the same `user_id`; the script prints which agents
fired, which memories were retrieved, and what got persisted at the end.

---

## Demo flow

A representative session:

| Turn | User message | Agents fired | Memory after |
|---|---|---|---|
| 1 | "Plan my weekend, I love jazz" | `event`, `topic` | "User loves jazz" |
| 2 | "Who from my circle would enjoy that?" | `people` | + "User has a social circle they want to involve" |
| 3 | "What event should I book *tonight*?" | `event` | + "User prefers concrete, bookable suggestions" |

By turn 3 the event agent receives *all three* memories from turn 1 and 2 in
its context, so its suggestion is shaped by the user's earlier taste.

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
│   │   ├── event_agent.py         ← 70B specialist
│   │   ├── aggregator.py          ← merges drafts → one reply
│   │   └── state.py               ← TypedDict + reducers
│   ├── memory/mem0_client.py      ← Chroma + sentence-transformers
│   ├── llm/groq_client.py         ← async Groq wrapper + tiering
│   ├── api/routes.py              ← /chat /memory /healthz /reset
│   ├── config.py                  ← pydantic-settings
│   └── main.py                    ← FastAPI app + lifespan
├── frontend/index.html            ← single-file chat UI (vanilla JS)
├── tests/                         ← 20+ unit tests, all mocked
├── scripts/smoke_demo.py          ← runnable 3-turn demo
├── Dockerfile + docker-compose.yml
└── .github/workflows/ci.yml       ← ruff + mypy + pytest
```

---

## Tests

```bash
pytest tests/ -v
```

Unit tests **never call the real Groq API** — every LLM call is replaced by
`unittest.mock.AsyncMock`. Memory tests run against an isolated, per-test
temporary Chroma directory.

Coverage:

- `test_intent_classifier.py` — single pick, multi pick, unknown agents
  dropped, empty payload fallback.
- `test_supervisor_graph.py` — graph compiles, every node present, fan-out
  end-to-end, conditional router.
- `test_memory.py` — per-user scoping, ordering, reset isolation,
  empty-text rejection.
- `test_aggregator.py` — empty / single / multi-draft merging.
- `test_api.py` — `/healthz`, `/chat`, `/memory/{user_id}`, `/reset`,
  Pydantic validation.

---

## Roadmap

- [ ] Swap the local Chroma memory layer back onto `mem0ai` once it ships a
      HuggingFace embedder, *or* onto Vertex Memory Bank for GCP deployments.
- [ ] LangSmith tracing (env-gated, scaffolded but disabled by default).
- [ ] WebSocket streaming of agent tokens to the frontend.
- [ ] One more specialist (`recall_agent` — "what did we talk about last week?")
      to demo a memory-only path.
- [ ] Eval harness (LLM-as-judge over a fixed prompt set) wired into CI.

---

## Author

Built by **Danilo Polishchuk** as a portfolio piece for AI Agent Engineer
roles. Other deployed work: a voice loan-application bot in production for
Hay Credito, a multi-agent bridge that orchestrates 4 Claude instances, and
a RAG support assistant. Reach out via GitHub.
