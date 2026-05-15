# Operations Runbook

Step-by-step procedures for shipping, operating and recovering the
agent-orchestrator-demo service in production. Each section is written
so an on-call engineer who has never touched this repo before can
execute it without paging the original author.

---

## Deploy (production)

### 1. Prerequisites

Before running any deploy step, confirm you have:

- Push access to the container registry (`ghcr.io/paramicelia/agent-orchestrator-demo`)
- Deploy credentials for the target environment (ECS task role / Cloud Run
  invoker / Railway service token — pick the one your env uses)
- `GROQ_API_KEY` in your local shell for the post-deploy smoke check
- The PR you intend to ship is merged to `main`

### 2. Pre-deploy checks

All three checks below MUST pass before promoting an image. Each one
gates the next.

```bash
# 2.1 Latest CI on main is green
gh run list --branch main --workflow ci.yml --limit 1
# expected: "completed  success"

# 2.2 Full eval on the shipping commit clears the per-tenant gates.
#     This is the only step that spends real Groq quota.
GROQ_API_KEY=$GROQ_API_KEY make eval
python -c "import json; r = json.load(open('eval/results.json'))
print('composite', r['summary']['avg_composite'])
assert r['summary']['avg_composite'] >= 8.0, 'composite below 8.0 — block deploy'"

# 2.3 Smoke test the local container against a real key
docker build -t agent-orchestrator:smoke .
docker run --rm -d -p 8000:8000 -e GROQ_API_KEY=$GROQ_API_KEY \
    --name aod-smoke agent-orchestrator:smoke
sleep 5
curl -sf localhost:8000/healthz || (docker logs aod-smoke; exit 1)
curl -sf localhost:8000/tenants | jq -e '.tenants | length >= 3'
docker stop aod-smoke
```

If any of the three fails: STOP. Do not promote. File a ticket and
investigate.

### 3. Deploy steps

```bash
# 3.1 Tag the image with the short SHA — never deploy :latest.
TAG=$(git rev-parse --short HEAD)
docker build -t ghcr.io/paramicelia/agent-orchestrator-demo:$TAG .
docker push ghcr.io/paramicelia/agent-orchestrator-demo:$TAG

# 3.2 Update the running service to the new tag. Pick ONE of:

# --- ECS ---
aws ecs update-service \
    --cluster aod-prod \
    --service aod-api \
    --force-new-deployment \
    --task-definition aod-api:$(aws ecs register-task-definition \
        --cli-input-json file://infra/ecs/task-def.json \
        --query 'taskDefinition.revision' --output text)

# --- Cloud Run ---
gcloud run deploy aod-api \
    --image ghcr.io/paramicelia/agent-orchestrator-demo:$TAG \
    --region us-central1 \
    --platform managed \
    --set-env-vars MEMORY_BACKEND=pgvector

# --- Railway ---
railway up --service aod-api --detach
```

### 4. Post-deploy verification

```bash
# 4.1 Health endpoint returns 200 within 60s of deploy
for i in 1 2 3 4 5 6; do
    curl -sf https://aod.example.com/healthz && break
    echo "attempt $i failed, retrying in 10s"; sleep 10
done

# 4.2 Smoke chat call against the default tenant
curl -sf -X POST https://aod.example.com/chat \
    -H 'content-type: application/json' \
    -d '{"user_id":"deploy_smoke","message":"Find me a jazz event tonight in New York.","tenant_id":"default"}' \
    | jq -e '.final_response | length > 0'

# 4.3 Tenants endpoint still lists every YAML — catches a bad volume mount
curl -sf https://aod.example.com/tenants | jq -e '[.tenants[].tenant_id] | contains(["acme","zenith","kids_safe"])'

# 4.4 LangSmith trace check (only if LANGSMITH_API_KEY is wired)
echo "Open https://smith.langchain.com/o/<org>/projects/p/agent-orchestrator-demo and"
echo "verify a fresh run for user_id=deploy_smoke shows the full node tree:"
echo "load_tenant -> load_memory -> classify_intent -> event_agent -> aggregate -> persona_adapt -> save_memory"
```

If 4.1 or 4.2 fails, go straight to **Section 5 (Rollback)**.

### 5. Rollback procedure

Trip the rollback when ANY of these is true within the first 15 minutes
post-deploy:

- P95 turn latency > 2000ms (was sub-1500ms before)
- 5xx rate > 1% over a 5-minute window
- Average eval composite score drops by > 0.5 points vs. previous run
- A user-reported reproducible regression

Steps:

```bash
# 5.1 Identify the previous good revision.
# ECS: list task definition revisions
aws ecs describe-services --cluster aod-prod --services aod-api \
    --query 'services[0].deployments[*].[taskDefinition,runningCount]' --output table

# 5.2 Roll the service back. Pick ONE of:

# --- ECS ---
aws ecs update-service --cluster aod-prod --service aod-api \
    --task-definition aod-api:<PREV_REVISION> --force-new-deployment

# --- Cloud Run ---
gcloud run services update-traffic aod-api --to-revisions=<PREV_REVISION>=100

# --- Railway ---
railway redeploy --service aod-api --deployment <PREV_DEPLOY_ID>

# 5.3 Validate rollback. Re-run section 4.1 and 4.2. Both must pass.

# 5.4 Open a postmortem ticket from the template at:
#     docs/postmortem-template.md
#     Fill: timeline, root cause, customer impact, action items.
```

---

## Onboard new tenant

Onboarding a new customer is a YAML-only change. Zero code is touched.

### 1. Create `tenants/{tenant_id}.yaml`

Copy any of the shipped configs (`tenants/acme.yaml` for enterprise,
`tenants/zenith.yaml` for startup, `tenants/kids_safe.yaml` for tight
allow-lists) and edit the values. Minimum required fields:

```yaml
tenant_id: newco
display_name: "NewCo"
allowed_personas: [neutral, formal]
default_persona: neutral
# Everything else has sensible defaults — see backend/tenants/schemas.py.
```

`tenant_id` must match the filename stem and use only `[a-z0-9_-]`.

### 2. Validate locally before pushing

```bash
# Validate a single file. Non-zero exit => CI will fail on the PR too.
python -m backend.tenants.loader --validate tenants/newco.yaml

# Validate the whole directory at once.
python -m backend.tenants.loader --validate tenants/
```

### 3. PR + CI eval pass

Open a PR with the new YAML. CI runs:

- `ruff` — catches accidental Python in the YAML directory
- `pytest tests/test_tenants.py` — validates all YAMLs against the
  `TenantConfig` schema and asserts loader behaviour
- Smoke-eval gate — confirms the eval framework still scores >= the
  tenant's `eval_thresholds`

A red CI is your first signal something is off. Read the failure
output — `TenantConfig` validation errors are explicit
(`allowed_personas contains unknown persona(s): ['gen-zee']`).

### 4. Deploy (no code change)

```bash
# Merge the PR, then run the standard deploy flow above. The
# tenants/ directory is baked into the Docker image, so a deploy is
# the only step that makes the new tenant reachable.
```

The new tenant appears in `GET /tenants` and is selectable from the
frontend dropdown the moment the new revision is healthy.

---

## Common operations

### Reset tenant memory (data corruption / GDPR)

```bash
# Wipe one (tenant, user) pair. Use this for GDPR right-to-erasure
# requests and for clearing a corrupted user namespace in incident response.
curl -X POST "https://aod.example.com/reset/<USER_ID>?tenant_id=<TENANT_ID>"

# Wipe an entire tenant via direct DB access (pgvector backend only).
# This is destructive — DO NOT run without a fresh backup.
psql $POSTGRES_DSN -c "DELETE FROM agent_memories WHERE tenant = '<TENANT_ID>';"
```

### Update persona allow-list for tenant

1. Edit `tenants/<tenant_id>.yaml`, modify `allowed_personas` and/or
   `default_persona`.
2. `python -m backend.tenants.loader --validate tenants/<tenant_id>.yaml`
3. Open PR, get CI green, merge, deploy.
4. Frontend dropdown auto-refreshes on next page load (no FE deploy).

### Scale up / down

```bash
# ECS desired count
aws ecs update-service --cluster aod-prod --service aod-api --desired-count 4

# Cloud Run min/max instances
gcloud run services update aod-api --min-instances 2 --max-instances 20

# Tune the heavy knobs via env vars in the task def / service config:
#   GROQ_SMART_MODEL  override the 70B model id
#   GROQ_LITE_MODEL   override the 8B model id
#   MEMORY_BACKEND    chroma | pgvector
#   POSTGRES_DSN      pgvector connection string
#   APP_LOG_LEVEL     info | debug
```

### Investigate slow turn

1. Note the `user_id` and approximate timestamp from the complaint.
2. Open the LangSmith console for the project, filter to the user_id.
3. Click into the slowest turn. Expand the `LangGraph` parent run.
4. Read child-run latencies top-to-bottom:
   - `load_tenant` > 50ms => tenant registry size or disk read issue
   - `load_memory` > 500ms => Chroma compaction / pgvector index bloat
   - `event_agent` > 3000ms => tool loop is firing > 1 round; check tool args
   - `aggregate` > 1500ms => 70B latency spike, check Groq status page
5. Cross-reference with `node_latencies` in the `/chat` response — every
   turn carries the same numbers inline.

---

## Monitoring & alerts

### Latency dashboard

Track at the load-balancer layer:

- P50 turn latency (target < 1000ms)
- P95 turn latency (target < 2000ms; alert at 2500ms for 5 min)
- P99 turn latency (target < 4000ms; alert at 6000ms for 5 min)

The `/chat` response already carries `node_latencies` per turn — feed
these into your APM (Datadog, New Relic, Honeycomb) for per-node breakdowns.

### Eval drift detection

Run `make eval` nightly via cron / GitHub scheduled action. Compare
`results.json.summary.avg_composite` against the previous 7-day rolling
mean. Alert if the new score drops by more than 0.5 points.

### Error rate threshold

```yaml
# Example Datadog monitor
- name: aod-api-5xx-rate
  query: "sum:trace.fastapi.request.errors{service:aod-api}.as_count() / sum:trace.fastapi.request.hits{service:aod-api}.as_count() > 0.01"
  for: 5m
  severity: page
```

---

## On-call

### Pager rotation

Primary / secondary rotation managed in PagerDuty schedule `aod-prod`.
Hand-off Mondays at 09:00 UTC. Document any incident handed off
mid-shift in the rotation channel.

### Severity definitions

- **SEV-1** — Service down, all tenants affected. Page primary
  immediately. SLO target: ack < 5 min, mitigation < 30 min.
- **SEV-2** — One tenant down OR > 5% error rate. Page primary. SLO
  target: ack < 15 min, mitigation < 2 hours.
- **SEV-3** — Degraded performance, no user-visible failure. Email
  primary. Fix within next business day.

### Escalation path

1. On-call primary — first responder.
2. On-call secondary — if primary is unreachable for 15 min.
3. Engineering manager — for SEV-1 lasting > 1 hour, or any incident
   involving data loss / privacy.
4. CTO — for SEV-1 lasting > 4 hours, or any incident affecting > 50%
   of paying tenants.
