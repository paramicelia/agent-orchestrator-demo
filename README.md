# Agent Orchestrator Demo

[![CI](https://github.com/paramicelia/agent-orchestrator-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/paramicelia/agent-orchestrator-demo/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org)
[![Eval](https://img.shields.io/badge/eval-8.3%2F10-brightgreen.svg)](eval/results.md)
[![Smoke-eval gate](https://img.shields.io/badge/smoke--eval-gating-blue.svg)](.github/workflows/ci.yml)
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
| Long-term memory | **Chroma + sentence-transformers** (default) or **Postgres + pgvector** | Local persistence by default, swap to pgvector with one env var for prod |
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

## Memory backends

Two interchangeable backends ship in `backend/memory/`. Both implement the
same `add` / `search` / `get_all` / `reset` surface (see
[`MemoryClient`](backend/memory/__init__.py) Protocol).

| Backend | Module | When to pick it |
|---|---|---|
| **Chroma** (default) | `backend/memory/mem0_client.py` | Local-first, zero infra. Best for the demo, local dev, and small single-process deployments. |
| **pgvector** | `backend/memory/pgvector_client.py` | Production. Pick this when the host already runs Postgres for app data and a second vector database isn't worth the operational cost. |

Swap is **one env var**:

```bash
# Chroma (default)
MEMORY_BACKEND=chroma

# Postgres + pgvector
MEMORY_BACKEND=pgvector
POSTGRES_DSN=postgresql://agent:agent@localhost:5432/agent_demo
```

The pgvector schema (`agent_memories` table, 384-dim vector column, IVF-Flat
cosine index, b-tree on `user_id`) is in
[`backend/memory/migrations/001_memories_pgvector.sql`](backend/memory/migrations/001_memories_pgvector.sql)
and is applied idempotently on the first call after process start.

### Boot pgvector locally

```bash
# Start Postgres + pgvector on :5432
docker compose -f compose.pgvector.yml up -d

# Run the app on the host against it
export MEMORY_BACKEND=pgvector
export POSTGRES_DSN=postgresql://agent:agent@localhost:5432/agent_demo
make dev
```

Or run app + DB together inside the docker network:

```bash
docker compose -f compose.pgvector.yml --profile app up --build
```

Embeddings stay on `sentence-transformers/all-MiniLM-L6-v2` (384-dim, cosine)
in both backends so a corpus indexed in one is re-ingestible into the other.

---

## Multi-tenancy

This is a single shared platform layer that serves many customers
("tenants") at once. Each tenant's behaviour — persona allow-list,
which agents run on smart vs. lite tier, per-tenant eval thresholds,
memory-retention window — is configured by a **YAML file under
`tenants/`**. Onboarding a new tenant is a YAML-only change: drop a
file, restart, done. Zero code change.

### Why multi-tenancy

Production AI assistants almost always need to serve more than one
brand / segment / contract tier from one codebase. The naive approach
(if/elses scattered through agents) becomes a maintenance nightmare
the moment the third tenant lands. The pattern here — a **core shared
platform layer parameterised by config** — is what the ORIL JD calls
out explicitly: "designing a core shared platform layer where new
customers are onboarded via config-driven parameterization (YAML/JSON)
without code changes."

### Tenant YAML schema

Minimal example:

```yaml
# tenants/newco.yaml
tenant_id: newco
display_name: "NewCo"
allowed_personas: [neutral, formal, casual]
default_persona: casual
model_tier:
  classifier: lite
  specialist: smart
memory_retention_days: 90
eval_thresholds:
  composite: 7.5
```

Full schema (with validators) is in
[`backend/tenants/schemas.py`](backend/tenants/schemas.py). Unknown
fields are rejected — a typo fails loudly at boot rather than silently
mis-routing traffic.

The repo ships three reference tenants:

| Tenant | Personas | Tier | Retention | Why |
|---|---|---|---|---|
| `acme` | all 5 | smart everywhere | 365 days | Enterprise white-glove |
| `zenith` | neutral / casual / gen-z | lite classifier + smart specialist | 90 days | Consumer startup, branded `event_agent` prompt |
| `kids_safe` | elderly-friendly only | all-lite (cost-minimised) | 30 days | School deployments, tightest eval gates |

### Onboarding flow

```bash
# 1. Drop a new YAML and validate it
cp tenants/acme.yaml tenants/newco.yaml
$EDITOR tenants/newco.yaml
python -m backend.tenants.loader --validate tenants/newco.yaml

# 2. Open a PR. CI runs the same validator across the whole directory
#    plus the tenant test suite (tests/test_tenants.py).

# 3. Merge + deploy. No code change.
```

Detailed runbook section: [`RUNBOOK.md` → "Onboard new tenant"](RUNBOOK.md#onboard-new-tenant).

### What flows through `tenant_id` on the wire

```bash
curl -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"user_id":"demo","tenant_id":"acme","message":"Plan my weekend.","persona":"formal"}'
```

- `tenant_id` defaults to `"default"` for backward compatibility with
  the pre-multi-tenancy callers.
- Unknown `tenant_id` silently falls back to the built-in default
  tenant and emits a `trace` line — never 4xx's.
- A request with a persona not in the tenant's `allowed_personas`
  collapses to `tenant_config.default_persona` (also visible in trace).
- Memory writes use `(tenant_id, user_id)` as the namespace key in
  both Chroma and pgvector — Acme's memory is never returned to Zenith.

Frontend (`frontend/index.html`) renders a tenant dropdown populated
from `GET /tenants` and constrains the persona dropdown to whatever
the selected tenant allows.

---

## Operations runbook

Step-by-step deploy / rollback / tenant onboarding / incident-response
procedures live in [`RUNBOOK.md`](RUNBOOK.md). It's structured so an
on-call engineer with no prior exposure to this repo can execute
deploys, rollbacks and common ops tasks without paging the original
author — directly addressing the "deployment SOPs or runbooks that
allow others to execute deployments without direct involvement" ask.

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

### CI gating: smoke-eval

Beyond the standard unit + integration suite, every PR has to clear a
separate **smoke-eval gate** step in `.github/workflows/ci.yml`:

```yaml
- name: Smoke-eval gate
  env:
    GROQ_API_KEY: "test-key-not-used"
  run: pytest tests/test_eval_smoke_ci.py -v
```

`tests/test_eval_smoke_ci.py` exercises the eval framework end-to-end with
a fully mocked Groq client and a redirected results path. It asserts:

1. `eval/dataset.json` parses, has >=10 turns, and every turn has the
   contract fields the runner depends on.
2. `judge_response` returns `{helpfulness: int 1..5, groundedness: float 0..1, reason: str}`.
3. `run_eval.main()` end-to-end produces a `results.json` whose
   `summary` + per-turn entries match the schema the README publishes.

This is what makes the eval harness load-bearing — a dataset typo or a
silent schema regression fails CI before it can ship.

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

## Observability — node-level tracing for multi-step workflows

Two layers ship out of the box, both controlled by
[`backend/observability.py`](backend/observability.py):

### 1. Per-node latency in the response (always on)

Every async LangGraph node is wrapped with `@traceable_node("<name>")` which:

* measures wall-clock latency around the inner call
* appends a structured `{"name": "<node>", "ms": <int>}` entry to a new
  `node_latencies` field on `AgentState`
* appends a human-readable line to the existing `trace` list, e.g.
  `node:event_agent took 1820ms`
* each specialist also embeds the timing directly in its trace string so the
  side-panel UI shows `agent:event produced 412 chars, used 1 tool calls in 1820ms`.

The `/chat` response now carries `node_latencies` alongside `trace`, and the
frontend side panel renders an aggregate breakdown so a slow turn is
diagnosable without leaving the page.

**Example trace from one turn:**

```text
memory.load: 1 hits for user=demo
node:intent_classifier took 124ms
intent=['event'] reason=user asked for a specific event with city
node:event_agent took 1820ms
agent:event produced 412 chars, used 1 tool calls in 1820ms
  tool:search_events args={'query': 'jazz', 'location': 'New York'}
node:aggregator took 612ms
aggregate: 1 drafts -> 287 chars
node:persona_adapt took 0ms
persona_adapt: neutral -> 287 chars
node:save_memory took 184ms
memory.save: stored fact (47 chars)
```

### 2. LangSmith export (opt-in)

`@traceable_node` also wraps each node with `langsmith.traceable(run_type="chain")`,
so when LangSmith is configured the full per-node call tree (including the
inner `event_agent.run_with_tools` tool-use loop) is exported to the cloud
dashboard. Without a key, the decorator is a clean no-op.

To enable:

```bash
# .env
LANGCHAIN_TRACING_V2=true
LANGSMITH_API_KEY=lsv2_pt_...
LANGSMITH_PROJECT=agent-orchestrator-demo
```

`init_tracing()` propagates these into the process environment on startup
so the LangChain callback manager auto-instruments every node. With no key
set, both the env-var bootstrap and every `@traceable_node` decorator
no-op — local runs and CI never hit the network.

> Want a screenshot of the LangSmith trace tree? Wire your key, hit `/chat`
> once, then check the LangSmith console for the run named
> `LangGraph` with child runs `intent_classifier` →
> `topic_agent / people_agent / event_agent` → `aggregator` →
> `persona_adapt`. See
> [`screenshots/01-chat-ui.png`](screenshots/01-chat-ui.png) for the local
> side-panel rendering of `node_latencies`.

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
│   ├── api/routes.py              ← /chat /memory /healthz /reset /personas /tenants
│   ├── observability.py           ← LangSmith opt-in tracing
│   ├── tenants/                   ← YAML-driven multi-tenancy
│   │   ├── schemas.py             ← TenantConfig (Pydantic v2)
│   │   ├── loader.py              ← YAML loader + CLI validator
│   │   └── defaults.py            ← built-in fallback tenant
│   ├── config.py                  ← pydantic-settings
│   └── main.py                    ← FastAPI app + lifespan
├── tenants/                       ← per-tenant YAML configs (drop a file = onboard)
│   ├── acme.yaml                  ← enterprise: all personas, smart tier
│   ├── zenith.yaml                ← startup: narrow personas, branded event_agent
│   └── kids_safe.yaml             ← school: elderly-friendly only, all-lite
├── frontend/index.html            ← single-file chat UI w/ tool & persona panels
├── RUNBOOK.md                     ← deploy / rollback / onboarding / on-call SOPs
├── eval/
│   ├── dataset.json               ← 10 fixed turns covering every path
│   ├── judge.py                   ← LLM-as-judge scoring fn
│   ├── run_eval.py                ← async runner → results.json + .md
│   └── results.md                 ← latest eval table
├── tests/                         ← 90+ unit tests, all mocked (incl. pgvector + smoke-eval gate)
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

90+ unit tests, all mocked. The eval **smoke** test (`test_eval_smoke.py`)
runs three hardcoded turns through the graph with a mocked Groq client and
the mocked judge, validating the full scoring pipeline without touching
the network. The full LLM-as-judge eval is reserved for `make eval`.

Coverage by file:

- `test_intent_classifier.py` — single / multi / unknown / empty / malformed.
- `test_supervisor_graph.py` — topology, fan-out end-to-end.
- `test_memory.py` — per-user scoping, ordering, reset, empty rejection (Chroma).
- `test_memory_pgvector.py` — pgvector SQL contract + schema migration with mocked psycopg.
- `test_aggregator.py` — empty / single / multi-draft merge.
- `test_api.py` — all endpoints + persona validation.
- `test_tools.py` — Pydantic schemas, dispatch, filtering, caps.
- `test_event_agent_tool_use.py` — full tool-use loop + recovery paths.
- `test_persona_adapter.py` — neutral no-op + each persona rewrite.
- `test_observability.py` — LangSmith init gating + `traceable_node` latency injection.
- `test_eval_smoke.py` — end-to-end scoring with mocked Groq.
- `test_eval_smoke_ci.py` — **CI gating step** — dataset schema + judge contract + run_eval results.json schema.

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
