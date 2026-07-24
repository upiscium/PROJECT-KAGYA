# PROJECT-KAGYA

## Local Runbook

- Start the API with `just api`; it serves FastAPI on the `api.host` and `api.port` values from `config.yaml`.
- Run backend tests with `uv run pytest`.
- Run frontend tests from `frontend/` with `nix develop --command npm test`.
- Build the frontend from `frontend/` with `nix develop --command npm run build`.

The single-subject state layers, authority boundaries, persistent schemas, decision flow, and regression-test mapping are documented in [SUBJECT_ARCHITECTURE.md](SUBJECT_ARCHITECTURE.md).

## Private First Run

Use this path when bringing up PROJECT-KAGYA for yourself on localhost, LAN, VPN, or an SSH tunnel.

1. Install dependencies:

```bash
uv sync
cd frontend && npm ci && cd ..
```

2. Run the fast local sanity checks with the default `dummy` model:

```bash
uv run pytest
just config-check
```

3. Create a private real-model config by copying `config.yaml`, then set `model.provider: transformers`. The primary training-compatible default is `google/gemma-4-12B-it`; the fallback remains `google/gemma-4-E2B-it`. Pin `model.revision` and `model.processor_revision` to immutable commits before using split deployment. Base variants can treat the prompt as document continuation and produce repetitive or unrelated text. Keep the committed `config.yaml` on `dummy` unless you want every local run to load real models.

If you previously used `.kagya/chroma` with another embedding backend, PROJECT-KAGYA creates embedding-versioned Chroma collections for non-legacy embeddings. Old collections are kept on disk but are not mixed with new embedding dimensions.

4. Smoke test the real model before starting the app:

```bash
just transformers-smoke /path/to/transformers-config.yaml
just transformers-smoke-fallback /path/to/transformers-config.yaml
```

5. Create local env files. The backend does not auto-load `.env`, so source it before `just api`. Next.js auto-loads `frontend/.env.local` for `npm run dev`.

```bash
ADMIN_TOKEN="$(openssl rand -hex 32)"
cat > .env <<EOF
KAGYA_ADMIN_TOKEN=${ADMIN_TOKEN}
KAGYA_CONFIG_PATH=/path/to/transformers-config.yaml
EOF

cat > frontend/.env.local <<EOF
KAGYA_ADMIN_TOKEN=${ADMIN_TOKEN}
KAGYA_BACKEND_URL=http://127.0.0.1:8000
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:3000
EOF
```

6. Start the API and frontend in separate terminals:

```bash
set -a; source .env; set +a
just api
```

```bash
cd frontend
npm run dev
```

7. Open the frontend on the private origin, use `/chat` first, then inspect `/debug`, `/memory`, `/sleep`, `/adapters`, and `/evaluations` as needed. Optional browser authentication is disabled by default, so keep this token-only mode on loopback, a private LAN/VPN, or an SSH tunnel and off the public internet.

8. After the first useful session, create a backup because `.kagya/` contains private runtime state:

```bash
KAGYA_BACKUP_DIR=.kagya/backups scripts/private-backup.sh
```

## Admin Access

- Normal chat remains public at `POST /api/chat`.
- Debug, memory inspection, sleep, and adapter endpoints require `X-KAGYA-Admin-Token`.
- The expected token is read from the env var named by `api.admin_token_env`; the default is `KAGYA_ADMIN_TOKEN`.
- Frontend admin pages call the Next.js `/admin-proxy/*` route, which injects `KAGYA_ADMIN_TOKEN` server-side; the token is not included in browser bundles.
- Chat responses accept typed feedback at `POST /api/feedback`. Public submissions must identify the response's episode, experience, and context; arbitrary memory, decision, and context targets are admin-only.
- Operators can audit `GET /api/feedback`, create broader target feedback at `POST /api/feedback/admin`, append revisions at `POST /api/feedback/{id}/revisions`, and withdraw effects at `POST /api/feedback/{id}/withdraw`.
- Every mutation requires an idempotency key. Feedback remains categorical and versioned: correction and expected-answer text becomes provenance-linked memory, never a free-form reward or hidden-thought update.
- `do_not_remember`, negative quality/safety signals, and explicit training exclusion remove the target episode from retrieval/consolidation/training while preserving it for audit. Corrections supersede rather than delete the original memory. Withdrawal restores the prior lifecycle and training policy when the feedback owns those effects.
- Operators can inspect evidence-backed pre/post Decision self-assessments at `GET /api/metacognition` and `GET /api/metacognition/assessments/{id}`. Confidence combines structured capability evidence with observed past accuracy; operator feedback is retained as explicit calibration provenance.
- `api.admin_auth.enabled` defaults to `false`. In that mode the existing admin-token behavior is unchanged; Origin, session, role, CSRF, and re-authentication checks are not applied. This mode is private/loopback only and must not be exposed directly to the public internet.

### Optional Admin Identity

Enable identity only behind a reverse proxy that has already authenticated the operator through SSO or WebAuthn. The reverse proxy must remove browser-supplied copies of its assertion headers before adding trusted values. Do not expose the Next.js port around that proxy.

1. Set `api.admin_auth.enabled: true` in the backend config. Roles are `read_only`, `approval_only`, and `full_admin`. Read-only actors may use safe methods only. Approval-only actors may perform explicit approval/rejection/review workflows, but cannot edit general state or start training. Full admins retain all admin operations.
2. Set matching frontend environment values:

```bash
KAGYA_ADMIN_AUTH_ENABLED=true
NEXT_PUBLIC_KAGYA_ADMIN_AUTH_ENABLED=true
KAGYA_SSO_TRUST_TOKEN=<random reverse-proxy-to-Next secret>
KAGYA_SSO_TRUST_HEADER=x-kagya-sso-secret
KAGYA_SSO_ACTOR_HEADER=x-forwarded-user
KAGYA_SSO_ROLE_HEADER=x-kagya-role
KAGYA_SSO_REAUTH_HEADER=x-kagya-reauthenticated-at
KAGYA_ADMIN_ALLOWED_ORIGINS=https://kagya.private.example
# Set these only when the matching api.admin_auth header names are customized:
KAGYA_BACKEND_ACTOR_HEADER=X-KAGYA-Actor
KAGYA_BACKEND_ROLE_HEADER=X-KAGYA-Role
KAGYA_BACKEND_REAUTH_HEADER=X-KAGYA-Reauthenticated-At
KAGYA_ADMIN_CSRF_HEADER=X-KAGYA-CSRF-Token
```

3. After SSO/WebAuthn succeeds, the reverse proxy injects the trust token, stable actor ID, role, and optional Unix re-authentication timestamp. `GET /admin-proxy/auth/session` exchanges that assertion for a signed, `HttpOnly`, `SameSite=Strict` session cookie and a separate `SameSite=Strict` double-submit CSRF token. The frontend initializes this session automatically when `NEXT_PUBLIC_KAGYA_ADMIN_AUTH_ENABLED=true`.
4. Browser mutations require an allowed `Origin`, non-cross-site Fetch Metadata, the signed session, and matching CSRF cookie/header. FastAPI repeats Origin, Fetch Metadata, CSRF, role, and configured re-authentication checks behind the proxy.
5. `api.admin_auth.reauthentication_paths` uses shell-style path patterns. Matching mutations require the re-authentication timestamp to be no older than `reauthentication_max_age_seconds`. Configure expensive or destructive paths explicitly; the committed defaults include state restore/reset, training job operations, cleanup, adapter activation/rollback, and identity/value rollbacks.

The backend token remains the emergency local recovery path when `allow_loopback_recovery` is enabled: a token-authenticated, non-browser request from loopback with no asserted actor is attributed as `local-recovery` and receives full-admin access. Keep this path local, rotate the token after emergency use, and disable it if operational recovery is provided another way.

Authorized mutations append a hash-chained Journal audit record containing only a sanitized actor ID, role, re-authentication status, and method/path target. Admin tokens, SSO trust tokens, session signatures, CSRF values, WebAuthn material, and other credentials are never written to agent state, traces, Journal records, or backup manifests. All secrets remain environment values or transient cookies/headers.

## Model Provider

- `config.yaml` defaults to the safe `dummy` provider.
- Real model smoke should use the Transformers provider and model IDs from `config.yaml` only.
- Real model smoke is opt-in and does not run in normal tests. Set `model.provider: transformers` in the target config, then run `just transformers-smoke /path/to/config.yaml`. Use `just transformers-smoke-fallback /path/to/config.yaml` to also load and generate with `model.fallback_id`.
- FastAPI startup preloads the primary Transformers model and processor so the first chat request does not pay the full model-load cost. The provider still falls back to `model.fallback_id` when primary generation fails.
- Chat responses include `model.fallback_used`; fallback responses report the fallback model ID and no active adapter.
- External LLM providers such as Ollama, OpenAI, Gemini API, and Claude API are intentionally unsupported.

## Deployment Topology

- `standalone + all + local` is the normal single-host configuration.
- `split + inference + ssh` requires `training.remote_worker`, an SSH identity path, a known-hosts path, and an expected worker model whose ID and exact revisions match the inference model.
- `split + training_worker + worker` requires isolated inbox, work, and result directories plus explicit allowed submitter node IDs.
- Other mode, role, and backend combinations are rejected during config validation.
- `node.id` is a stable application identifier and is distinct from the OS hostname. Set `enforce_hostname_match: true` only with an explicit `expected_hostname`.
- Passwords, private key contents, Hugging Face tokens, and worker token values do not belong in YAML. Configuration stores credential file paths or environment-variable names only.
- Legacy configs without `deployment` are explicitly migrated to `standalone/all/local`; `just config-check` reports the migration note. Split configs never use that migration as an implicit topology fallback.

Identity-changing state uses typed origin provenance. Admin `POST /api/goals` accepts external-request proposals only; it cannot declare a Goal intrinsic on the subject's behalf. Adoption is a separate subject event that records endorsement. Value update requests are retained as operator feedback/evidence and cannot spoof self-origin through a caller-provided source label. Legacy Goal, Commitment, Value, and identity-proposal records migrate as inherited and uncertain rather than self-originated.

Active Goals can be decomposed through the admin-only `/api/plans` API into strict schema-v1 Plans and Steps. Plans retain immutable revision snapshots and operator change reasons; reject unknown dependencies, dependency cycles, unknown/private fields, and raw model prose; and require expected-observation evidence before Step or Goal completion. Retry, timeout, verification, and rollback policies are structured state only. The scheduler never executes a tool or rollback. Only dependency-ready Steps enter Working Memory or become Plan-linked ActionCandidates, and all lifecycle changes persist through `AgentRuntime`, the authoritative snapshot, and the private state WAL for restart recovery.

## Responsibility Model

Desire, Intention, and Commitment are separate durable layers. Desire remains a decaying `MotivationRecord`; an adopted Goal is an Intention that can reference the Desires that prompted it; a Commitment is an accepted responsibility whose lifetime is independent of those Desires. Desire decay or loss therefore never releases, fulfills, or breaches an accepted Commitment.

`POST /api/commitments` records an external proposal only. It stores typed origin, beneficiary, scope, deadline, cost, burden, fulfillability, Relationship references, and unresolved Desire/Value/Commitment conflicts, but it cannot create a Goal or active Commitment. `POST /api/commitments/{id}/accept` requires a separate explicit self-endorsement and creates the linked Intention. Acceptance, fulfillability reassessment, renegotiation, fulfillment, release, breach, repair, and accountability evidence remain in append-only lifecycle records.

An impossible fulfillability reassessment retains the Commitment and creates a structured Decision containing at least `renegotiate_commitment` and `notify_beneficiary_of_impossibility` options; it does not silently abandon the responsibility. At-risk or impossible commitments receive Scheduler reevaluation wake-ups. Breach and repair evidence updates linked Relationship conflict/repair history and Narrative Self commitment events. Admin lifecycle routes are `POST /api/commitments/{id}/reassess`, `/renegotiate`, `/transition`, and `/repair`; inspection remains protected by the admin token.

## Metacognition

Metacognitive boundaries distinguish `unknown`, `uncertain`, `unable`, and `needs_help` and can steer explicitly scoped Decisions toward `request_information`, `defer`, `observe`, or `delegate`. Assessments record Self Model revisions, Narrative Self references, cognitive load, attention saturation, emotion influence, prediction/outcome evidence, and recurring error/bias hypotheses. Generated apologies, hidden thought, and model self-report are not accepted as competence or calibration authority.

## Relationship Continuity

Each stable `interlocutor_key` maps to one versioned subjective Relationship across contexts. Relationship state keeps trust, familiarity, closeness, and caution as independent bounded axes, alongside perceived role, expectations, boundaries, reciprocity, shared Experience references, commitments, unresolved matters, conflict/repair history, uncertainty, and revisions. The other person's reported values and beliefs remain under `other_values` and `other_beliefs`; they are never copied into the subject's Value or Belief stores.

Experience-derived changes require two consistent observations before an axis moves and each accepted observation is capped at `0.08`. A new alias also requires two independent evidence references and cannot be attached when it already belongs to another Relationship. This deliberately favors a missed merge over combining two people. Operators can inspect `GET /api/relationships`, correct a Relationship with `POST /api/relationships/{id}/corrections`, attach a corroborated alias with `POST /api/relationships/{id}/aliases`, and split an alias with `POST /api/relationships/{id}/split`. All routes require admin authorization and all mutations preserve evidence-linked revision history.

Relationship state is part of the authoritative agent snapshot. Current relationship caution and uncertainty affect appraisal and Emotion; continuity summaries carry relationship-linked Goals, Commitments, and unresolved matters into later contexts; relationship-targeted Goals receive a bounded reciprocity/trust/caution adjustment; and Decision candidates retain explicit relationship appraisal contributions. Relationship APIs and snapshots contain structured references and state only, never chat text, prompts, hidden thoughts, or raw memory content.

## Multimodal Attachments

The first real multimodal milestone supports one local image attachment for capable Transformers image-text models.

- Supported attachment type: `image`.
- Supported URLs: local `file://` URLs only.
- Supported content types: `image/png`, `image/jpeg`, and `image/webp`.
- Size limit: 5 MiB per image.
- Limit: one image per chat request.
- Dummy and other text-only providers remain metadata-only; they do not load attachment files.
- Transformers validates and decodes supported image attachments before passing them to the processor.
- Public chat responses never include raw attachment metadata or local file paths.

Unsupported attachment types or invalid images are rejected by capable multimodal providers with clear errors. Keep local image paths private; prompt metadata records only safe fields such as type, name, content type, and URL scheme.

## Safe Tools

- Approved static `text_template` tools can render deterministic strings from supplied arguments.
- Approved static `metadata_lookup` tools can read one key from human-approved tool metadata.
- Shell tools, generated code execution, unapproved tools, disabled tools, and unknown tools remain blocked.
- Tool execution attempts record audit events for both allowed and blocked requests.

## Production QLoRA Boundary

The committed config keeps `qlora.dry_run: true`. Non-dry-run QLoRA training is an explicit production path and should only be enabled on a host that passes the preflight check:

```bash
just qlora-prod-check /path/to/production-qlora-config.yaml
```

Production QLoRA assumptions:

- `model.provider` is `transformers`, and `just transformers-smoke-fallback /path/to/config.yaml` passes first.
- CUDA is available; CPU training is not part of the supported production path.
- `model.load_in_4bit` remains enabled for the supported QLoRA path.
- `datasets`, `peft`, `torch`, `transformers`, and `trl` are installed in the runtime environment.
- `transformers>=5.14.1` is required. Version 5.9.0 exposes multimodal auto classes but cannot resolve the `gemma4_unified` config used by `google/gemma-4-12B-it`.
- `adapter_registry.eval_sets` points to at least one existing eval set so trained adapters can be evaluated before promotion.
- `adapter_registry.manual_approval_required` remains true; no trained adapter is activated without explicit operator approval.
- Interrupted or failed training artifacts should be treated as incomplete and must not be manually registered as approved/active adapters.
- Training inputs and outputs use immutable `training-<job-id>/` and `result-<job-id>/` directories. Payload checksums exclude only `checksums.sha256` itself; unknown schemas, unsafe paths, symlinks, revision mismatches, partial artifacts, and overwrite attempts are rejected before transport or import.
- The RTX 3090 24GB baseline uses NF4, BF16 compute, gradient checkpointing, accumulation of 8, sequence length 512, `paged_adamw_8bit`, and batch size 1. Reduce sequence length before changing quantization when resolving OOM.
- `resume_policy: never` intentionally restarts failed jobs from the immutable bundle. Parent-adapter continuation is rejected until parent hash and base compatibility validation are implemented.

Training flow for real adapters:

1. Run the real-model smoke checks.
2. Run `just qlora-prod-check /path/to/config.yaml`.
3. Generate a dream dataset from the private runtime.
4. Run non-dry-run training with `qlora.dry_run: false`.
5. Register only the produced candidate adapter.
6. Evaluate the candidate and inspect evaluation history/regressions.
7. Manually approve and activate only if the evaluation result is acceptable.

For the manual RTX 3090 integration check, pin `model.revision` and `model.processor_revision` to immutable commits on both nodes, set `model.provider: transformers` and `qlora.dry_run: false`, run the production preflight, then submit a short `dream-v2`/`gemma-v1` bundle through `kagya-worker run`. Preserve the resulting `result.json`, `training_metrics.json`, checksums, GPU name, CUDA/package versions, peak VRAM observation, and a successful adapter-load generation as the execution record. This hardware check is intentionally opt-in and is not run in CI.

The public `google/gemma-4-12B-it` repository is not gated and may be downloaded anonymously. The compatibility-verified immutable revision is `12ace6d648d72bd41519e140f1185f34d38c7e3d`; use the same revision for `model.revision` and `model.processor_revision` on both nodes.

Distributed training operations are available through admin-token-protected endpoints:

- `GET /api/training/nodes` reports worker reachability, heartbeat, revisions, capacity, and GPU environment.
- `GET /api/sleep/jobs/{job_id}` includes phase durations, transfer bytes, last contact, failure category, retryability, metrics, and import status.
- `POST /api/sleep/jobs/{job_id}/reconcile` and `POST /api/sleep/reconcile` rediscover remote state and report orphan jobs/results without duplicating submissions.
- `POST /api/sleep/cleanup` applies `sleep.artifact_retention_days` to terminal bundle/result/work artifacts. Imported ACTIVE and rollback adapter directories are outside this cleanup boundary.
- `GET /api/adapters/{adapter_id}/provenance` reports model/job/node provenance and activation/rollback history.
- `GET /api/system/journal` returns admin-only, operator-safe durable event lifecycle records and hash continuity metadata. It never returns event payloads, prompts, generated text, attachments, credentials, or hidden thoughts.
- `agent_state_wal.path` is a separate mode-`0600` private authoritative state log. Keep it with the state snapshot in encrypted backups; do not publish it as operational telemetry or an operator-safe Journal.
- Admin-only `GET /api/state/reconstruct/{sequence}` and `POST /api/state/restore/{sequence}/dry-run` reconstruct and compare retained state without replaying side effects. `POST /api/state/restore/{sequence}` commits the selected state as a new runtime event; tools, network, notifications, training, Chroma, and adapter-registry operations are never replayed.

### Operational Observability

`GET /health/live` is a process liveness check. `GET /health/ready` separately verifies the role-specific runtime and, on inference nodes, journal integrity; it returns `503` while dependencies are not ready. The legacy `GET /health` remains a basic deployment identity check.

Admin-token-protected observability exports are dependency-free:

- `GET /api/system/metrics` returns Prometheus text metrics.
- `GET /api/system/telemetry` returns OTLP JSON-compatible metrics and spans for collectors that accept OTLP JSON payloads.
- `GET /api/system/traces?event_id=...` follows an authoritative journal event into its bounded causal span. Correlation and causation IDs are records, never metric labels.

Metrics persist at `observability.metrics_path`; the newest `observability.max_traces` spans persist at `observability.traces_path`. `observability.max_series` is a hard series limit. Labels are fixed categorical dimensions: prompts, generated text, hidden thoughts, credentials, IDs, model paths, job names, and user metadata are never labels or span content. Unsafe trace identifiers are one-way hashed. Treat both files as operational state and retain them across service restarts for before/after comparison.

The single-active `AutonomyLoop` restores versioned wake-ups from the subject snapshot and derives deadline/reassessment checks from Goals, Commitments, and unresolved Decisions. Each due occurrence is idempotently completed by `AgentRuntime` and the durable Journal. `autonomy.max_events_per_cycle`, `max_inferences_per_cycle`, and `max_wall_seconds_per_cycle` bound every cycle; excess work remains pending. An idle cycle returns `no_action` without model inference or a journal event. Shutdown stops new wake-ups before draining accepted runtime work.

Admin operators can inspect `GET /api/autonomy/status` and create an internal wake-up with `POST /api/autonomy/wake-ups`. Action retry, outbox, sleep/consolidation, and operator wake-ups are signals only: the scheduler never invokes tools, sends outbox data, starts training, or performs any other external effect directly.

Latency, queue depth, generation rate, memory retrieval, fallback/quarantine, storage, sleep, adapter, and runtime lifecycle metrics describe processing. `kagya_active_goals`, `kagya_unresolved_decisions`, and `kagya_attention_focus_items` describe current subjective state; do not interpret them as throughput or performance scores. RAM is sampled during export, and accelerator allocated/reserved memory is exported when CUDA is available.

### Split Training Worker Runbook

The worker is invoked as a one-shot command over SSH; it is not a long-running systemd daemon. Install `deploy/bin/kagya-worker-remote` at a stable absolute path on the worker and create its restricted environment file:

```bash
sudo install -o kagya -g kagya -m 0755 deploy/bin/kagya-worker-remote /opt/project-kagya/bin/kagya-worker-remote
sudo install -o kagya -g kagya -m 0600 deploy/env/training-worker.env.example /etc/project-kagya/training-worker.env
```

Set `KAGYA_CONFIG_PATH` to a private worker config, set `CUDA_VISIBLE_DEVICES` to the dedicated training GPU, and enable `KAGYA_WORKER_NIX_DEVELOP=1` on NixOS when the flake supplies CUDA libraries. The worker config must use `split + training_worker + worker`, `model.provider: transformers`, the exact model and processor revision, `qlora.dry_run: false`, and distinct absolute runtime directories:

```yaml
deployment:
  mode: split
  node:
    id: training-eve-01
    role: training_worker
    expected_hostname: Eve
    enforce_hostname_match: true
  training:
    backend: worker
    worker:
      inbox_directory: /var/lib/project-kagya/worker/inbox
      work_directory: /var/lib/project-kagya/worker/work
      result_directory: /var/lib/project-kagya/worker/results
      max_concurrent_jobs: 1
      retain_failed_jobs: true
      allowed_submitters:
        - inference-01
```

On the inference node, verify the worker host-key fingerprint out of band before adding it to the dedicated known-hosts file. Configure `split + inference + ssh` with the same immutable revisions:

```yaml
deployment:
  mode: split
  node:
    id: inference-01
    role: inference
    expected_hostname: null
    enforce_hostname_match: false
  training:
    backend: ssh
    remote_worker:
      node_id: training-eve-01
      host: eve.internal
      port: 22
      user: kagya
      identity_file: /etc/project-kagya/ssh/training-worker
      known_hosts_file: /etc/project-kagya/ssh/known_hosts
      remote_inbox: /var/lib/project-kagya/worker/inbox
      remote_results: /var/lib/project-kagya/worker/results
      command: /opt/project-kagya/bin/kagya-worker-remote
      connect_timeout_seconds: 10
      job_timeout_seconds: 86400
      poll_interval_seconds: 30
      expected_worker_model:
        model_id: google/gemma-4-12B-it
        revision: 12ace6d648d72bd41519e140f1185f34d38c7e3d
        processor_revision: 12ace6d648d72bd41519e140f1185f34d38c7e3d
```

After both configs pass `just config-check`, run the explicit end-to-end smoke from the inference node. The work directory must be empty and should be retained as the execution record until its result and provenance have been reviewed:

```bash
KAGYA_CONFIG_PATH=/etc/project-kagya/inference.yaml \
  uv run python -m kagya.training.split_smoke \
  --work-dir /var/lib/project-kagya/smoke/$(date -u +%Y%m%dT%H%M%SZ) \
  --confirm RUN-SPLIT-TRAINING
```

The smoke checks strict SSH worker health, CUDA visibility, duplicate-submit idempotency, immutable bundle/result checksums, backend restart recovery, completed-job discovery, local PEFT candidate import, provenance, and event-boundary activation/rollback. It never prunes remote artifacts and never writes to the configured authoritative adapter registry; local import state is isolated under `--work-dir`.

## Configuration Field Status

Most `config.yaml` fields are active runtime settings. The fields below are intentionally retained but have limited or future-facing behavior:

| Field | Status |
| --- | --- |
| `model.fallback_id` | Active for Transformers per-request fallback generation. Dummy provider ignores it. |
| `model.revision` / `model.processor_revision` | Passed to Transformers model and processor loading. Split inference requires immutable exact revisions that match the worker expectation. |
| `model.fallback_revision` | Passed to fallback model and processor loading. |
| `memory.embedding_model_id` | Active for the default sentence-transformers memory embedding backend. Tests and bootstrap flows can still inject deterministic embeddings explicitly. |
| `adapter_registry.allowed_states` | Reserved as an operator-visible lifecycle contract. Runtime transitions are enforced by `AdapterStatus`. |
| `adapter_registry.manual_approval_required` | Reserved for future automatic approval policy. Current lifecycle always requires explicit approval before activation. |
| `qlora.alpha` / `qlora.dropout` | Legacy aliases retained for compatibility; `qlora.lora_alpha` and `qlora.lora_dropout` are the training-facing names. Dry-run manifests include both. |
| `qlora.max_steps` | Active for the minimal non-dry-run trainer and included in dry-run manifests for auditability. |
| `qlora.gradient_checkpointing` / `gradient_accumulation_steps` / `max_sequence_length` | Active RTX 3090 memory controls for real QLoRA. |
| `qlora.optimizer` / `seed` / `target_modules` / `resume_policy` | Strict reproducibility and Gemma adapter contract. Only paged 8-bit AdamW and restart-from-bundle are currently supported. |

### Configuration Compatibility Policy

- Unknown configuration keys are rejected by the typed schema. Add new fields to `kagya/config/schema.py` and document them here before relying on them.
- Compatibility aliases may remain in `config.yaml` when they protect existing local deployments. `qlora.alpha` and `qlora.dropout` are retained as legacy aliases, while `qlora.lora_alpha` and `qlora.lora_dropout` are the training-facing values.
- Reserved fields are accepted only when they are documented as operator-visible contracts. `adapter_registry.manual_approval_required` is accepted for future policy work, but current lifecycle transitions still require explicit approval in code.
- If a compatibility alias diverges from its canonical field, the canonical field wins for runtime behavior unless the field status table says otherwise.
- Validate a deployment config with `just config-check` or `uv run python -m kagya.config.check /path/to/config.yaml` before upgrading services.

## Release Checklist

- `uv run pytest`
- `uv run ruff check kagya tests`
- `just config-check`
- `just schema-check`
- `just transformers-smoke /path/to/transformers-config.yaml` on hosts intended to run real models.
- `just qlora-prod-check /path/to/production-qlora-config.yaml` before enabling non-dry-run training.
- `npm test -- --run` from `frontend/`
- `npm run build` from `frontend/`
- `timeout 5s just api || test $? -eq 124 -o $? -eq 143`
- `KAGYA_ADMIN_TOKEN=... scripts/smoke-private-deploy.sh http://127.0.0.1:8080` for private deployments.
- Search for forbidden provider implementation paths.
- Verify normal API/UI responses do not expose `hidden_thought`, raw prompts, retrieved memory, or `<think>` tags.
- Verify public chat responses do not expose local attachment paths.

## Memory And Learning Safety

Episodic memories retain source event, processing sequence, provider/model identity, validation state, content hash, and generation-health metadata. Empty, repetitive, prompt-leaking, or non-finite generations are quarantined: operators can inspect and review them through the memory admin API, but normal retrieval and sleep learning exclude them.

Semantic memory uses schema-versioned DB2 records. Normalized exact duplicates reuse one record and merge source provenance; merge proposals, contradictions, supersessions, and corrections retain inspectable lineage and an operation audit trail. Retrieval excludes archived, forgotten, expired, invalid, superseded, corrected, source-rejected, unpublished, and fully decayed records. Rejecting or archiving the last viable source episode automatically removes its derived semantic records from retrieval, and restoring source validity reevaluates them. Archive/restore is reversible cold storage, logical forgetting preserves audit history, and admin `DELETE /api/memory/semantic/{id}` is the separate irreversible physical deletion path. Existing DB2 records are backfilled to semantic schema version 2 when the memory system starts.

Semantic records remain evidence-bearing memory, not accepted Beliefs. Belief adoption and retraction continue through the separate Belief store and APIs.

Sleep consolidation uses an explicit pipeline version and attempt ID. A completed episode is not processed twice by the same pipeline version, datasets are written to immutable `dreams/runs/<attempt-id>/` paths, raw hidden thoughts are not copied into new datasets, and semantic memories remain staged until the adapter candidate is registered. Adapter registry mutations use an exclusive sidecar lock and atomic fsynced replacement. Real adapter evaluation runs paired baseline and candidate providers; a candidate that regresses against baseline is not promoted.

Sleep training bundles are built from governed dataset revisions under `.kagya/training_artifacts/datasets/revisions/`. Every record carries its inclusion decision, consent/privacy classification, and source event/memory/decision/feedback lineage. Private, rejected, and do-not-train records are excluded; PII, credentials, secrets, poisoning indicators, exact duplicates, and near-duplicates are quarantined. Scanner failures fail closed. Content hashes retain an immutable train/validation/test assignment across revisions, and cross-split duplicates are rejected. Each revision has a checksummed immutable manifest, while bundle and adapter training manifests pin its revision and manifest hash. Admin operators can browse `GET /api/training/datasets`, inspect `GET /api/training/datasets/{revision}`, compare `GET /api/training/datasets/diff?from=...&to=...`, or use the `/datasets` UI.

## Working Memory

The active subject uses finite attention-based working memory instead of an unbounded conversation transcript. `working_memory.item_capacity` limits resident items and `working_memory.token_capacity` limits the selected view rendered into prompts. Recent episodes and semantic memories are retained as DB references, while goals, commitments, unresolved items, and current emotion receive explicit retention priority. Eviction only removes an item from working memory; it never deletes long-term DB1/DB2 records. Admin debug chat reports selection scores and reasons without exposing selected content.

Successful chat events also create a versioned first-person Experience record under the durable subject snapshot. `GET /api/experiences` and `GET /api/experiences/{experience_id}` are admin-only. Experience records separate external observation references from internal structured interpretation and appraisal; they do not duplicate chat text, generated responses, prompts, attachments, or hidden thoughts. Their subjective salience influences Working Memory retention, episodic retrieval ordering, and consolidation eligibility. Evidence-backed reassessment appends revisions instead of rewriting history.

Beliefs are stored separately from episodic and semantic memory under the subject snapshot. Admin `POST /api/beliefs` creates an Experience-backed proposal; `POST /api/beliefs/{belief_id}/resolve` explicitly accepts or rejects it after evidence review. Proposed, disputed, superseded, retracted, expired, rejected, or out-of-scope Beliefs are excluded from normal Working Memory and Decision inputs. `GET /api/beliefs?active_only=true` returns only currently usable Beliefs. Memory rendered into prompts is labelled as a record, not as an adopted current fact.

Internal motivation is persisted separately from externally proposed Goals. Repeated Experience signals can form curiosity Interest, closure Drive, or Aversion; one transient event is insufficient. Admin `GET /api/motivation` inspects structured records, while `POST /api/motivation/reevaluate` runs a bounded internal reevaluation without accepting a caller-supplied task. Generated Goals remain candidate intrinsic Goals with self-origin provenance and motivation/Experience references. `POST /api/motivation/decay` advances explicit elapsed-time decay.

## Context Model

Chat responses return an opaque `context_id`; clients resend it to resume the same situation. Context frames distinguish channel, client session, participants, topic/task, parent/related contexts, and active/suspended/closed lifecycle state without splitting the subject's global emotion or identity. Retrieval keeps semantic relevance and context compatibility as separate scores. Cross-context memories remain available but are explicitly marked with their source context and relationship. Interlocutor keys are unverified correlation hints, not authenticated identity claims. Context lifecycle administration is available under `/api/contexts/*` with the admin token.

## Appraisal And Emotion

Model loss is treated as a calibrated, model-specific novelty measurement rather than a direct emotion value. Invalid, non-finite, or unavailable loss becomes an explicit invalid measurement and contributes no novelty; it is never converted to zero-loss wellbeing. Structured appraisal separates novelty, goal progress, threat, controllability, certainty, social relevance, and effort cost before updating valence/arousal. Admin debug responses and runtime events expose numeric contributions and safe reason codes. Optional `appraisal.timer_enabled` recovery submits serialized `emotion_tick` events; the timer never mutates subject state outside the agent queue.

## Private Deployment

Production builds must inject immutable source provenance. Set
`KAGYA_SOURCE_COMMIT_SHA` to the 40-hex source commit,
`KAGYA_SOURCE_TREE_HASH` to its Git tree hash, and `KAGYA_BUILD_ID` to a bounded
build identifier. Development runs resolve Git metadata locally and record a
`dirty` or `unknown` status rather than inventing a source revision. Production
behavioral evaluation and activation reject anything other than verified source
metadata.

PROJECT-KAGYA is intended to run as a private/local application, not as a public website. The deployment target is a single Linux host with FastAPI bound to `127.0.0.1:8000`, Next.js bound to `127.0.0.1:3000`, and nginx or Caddy bound to loopback for local or SSH-tunnel access.

The backend uses one process-local agent event queue to serialize chat, sleep, memory administration, and adapter lifecycle operations. Run exactly one Uvicorn worker; multiple workers would create independent subjects and independent event sequences. `api.agent_queue_capacity` bounds waiting work. A full queue returns HTTP 429, shutdown stops accepting work and drains accepted events, and an accepted event continues even if its client disconnects.

The subject's versioned internal-state snapshot defaults to `.kagya/agent_state.json`. Each accepted event is returned only after its sequence and internal state have been atomically written and fsynced. Startup validates or migrates the snapshot and safely falls back to baseline state if it is corrupt. Admin endpoints under `/api/state` provide snapshot, export, restore, and reset operations. Snapshots intentionally exclude conversation turns, event payloads, prompts, attachments, generated text, and hidden thoughts.

### 1. Prepare Host

Create a service user and install the required runtime tools:

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin kagya
sudo mkdir -p /opt/project-kagya /etc/project-kagya
sudo chown -R kagya:kagya /opt/project-kagya /etc/project-kagya
```

Install `git`, `uv`, `nodejs` 22, `npm`, and either `nginx` or `caddy` using your OS package manager or Nix profile. For the real Transformers provider, verify NVIDIA drivers/CUDA before switching away from the default `dummy` provider.

### 2. Install Application

```bash
sudo -u kagya git clone <repo-url> /opt/project-kagya
cd /opt/project-kagya
sudo -u kagya git switch develop
sudo -u kagya uv sync
cd frontend
sudo -u kagya npm ci
```

### 3. Configure Environment

```bash
sudo cp deploy/env/backend.env.example /etc/project-kagya/backend.env
sudo cp deploy/env/frontend.env.example /etc/project-kagya/frontend.env
sudo chmod 600 /etc/project-kagya/*.env
sudo chown kagya:kagya /etc/project-kagya/*.env
```

Edit both env files and set the same long random `KAGYA_ADMIN_TOKEN`. Set `NEXT_PUBLIC_API_BASE_URL` to the browser-visible private origin and keep `KAGYA_BACKEND_URL` pointed at the private FastAPI listener.

For SSH-tunnel access, run `ssh -L 18080:127.0.0.1:8080 user@host`, set `NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:18080`, rebuild the frontend, and open `http://127.0.0.1:18080` locally.

Admin warning: the frontend does not expose the admin token to browser bundles, but admin pages do not have browser login/session handling. This is acceptable for the intended private LAN/VPN/SSH-tunnel deployment model. Keep this listener bound to loopback or behind private network access; if you expose it outside that boundary, add SSO, basic auth, or another access-control layer first.

Build the frontend after the env file is configured because `NEXT_PUBLIC_API_BASE_URL` is embedded at build time:

```bash
sudo -u kagya bash -lc 'set -a; source /etc/project-kagya/frontend.env; set +a; cd /opt/project-kagya/frontend && npm run build'
```

### 4. Install Services

```bash
sudo cp deploy/systemd/kagya-api.service /etc/systemd/system/kagya-api.service
sudo cp deploy/systemd/kagya-frontend.service /etc/systemd/system/kagya-frontend.service
sudo systemctl daemon-reload
sudo systemctl enable --now kagya-api kagya-frontend
sudo systemctl status kagya-api kagya-frontend
```

### 5. Configure Reverse Proxy

For nginx:

```bash
sudo cp deploy/nginx/kagya.conf /etc/nginx/sites-available/kagya.conf
sudo ln -s /etc/nginx/sites-available/kagya.conf /etc/nginx/sites-enabled/kagya.conf
sudo nginx -t
sudo systemctl reload nginx
```

For Caddy, copy `deploy/caddy/Caddyfile` into your Caddy config path. The provided example binds to `127.0.0.1:8080` and is meant for local or SSH-tunnel access.

### 6. Verify Deployment

```bash
KAGYA_ADMIN_TOKEN=replace-with-long-random-token scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```

The smoke script verifies `/health`, public `/api/chat`, direct admin API rejection without a token, direct admin API success with `X-KAGYA-Admin-Token`, and `/admin-proxy/*` forwarding through the frontend. Set `CHECK_ADMIN_PROXY=0` if you are checking only the FastAPI reverse proxy without the frontend service.

Normal chat is unauthenticated on the private listener at `POST /api/chat`; direct debug, memory, sleep, and adapter APIs require the admin token header. Frontend admin pages use `/admin-proxy/*` and should remain behind your private access boundary.

### 7. Back Up Runtime Data

Backups must include both runtime data and private environment files:

- `.kagya/chroma`: Chroma memory storage.
- `.kagya/dreams`: generated dream datasets.
- `.kagya/adapters`: QLoRA dry-run or trained adapter artifacts.
- `.kagya/eval_results`: adapter evaluation outputs.
- `.kagya/adapter_registry.json`: adapter lifecycle state.
- `.kagya/agent_state.json`: versioned subject state and last processed event sequence.
- `.kagya/agent_journal.jsonl*`: fsynced event lifecycle journal and retained hash-chain rotation files.
- `.kagya/operational_metrics.json`: atomic persistent bounded-cardinality metric aggregates.
- `.kagya/operational_traces.json`: atomic bounded recent correlation/causation spans.
- `/etc/project-kagya/*.env`: backend/frontend env files, including `KAGYA_ADMIN_TOKEN`.

Create a restricted archive on the deployment host:

```bash
cd /opt/project-kagya
sudo -u kagya KAGYA_BACKUP_DIR=/var/backups/project-kagya scripts/private-backup.sh
```

Treat backup archives as secrets. They may contain private memories, hidden training thoughts, local paths, adapter artifacts, and admin tokens. Store them with `0600` permissions, encrypt them before off-host transfer, and avoid attaching them to issues or logs.

Before large maintenance, inspect disk usage:

```bash
sudo du -sh /opt/project-kagya/.kagya /etc/project-kagya
sudo du -sh /home/kagya/.cache/huggingface 2>/dev/null || true
```

Model caches are usually reproducible and large; back them up only if bandwidth or model availability requires it. The `.kagya/` directory is application state and should be backed up.

### 8. Restore Runtime Data

Restore onto a host with the same runtime tools installed and the application checkout present. Stop services before replacing local data:

```bash
sudo systemctl stop kagya-api kagya-frontend
cd /opt/project-kagya
sudo KAGYA_APP_DIR=/opt/project-kagya KAGYA_CONFIG_DIR=/etc/project-kagya scripts/private-backup.sh --restore /var/backups/project-kagya/project-kagya-YYYYMMDDTHHMMSSZ.tar.gz
sudo chown -R kagya:kagya /opt/project-kagya/.kagya /etc/project-kagya
sudo chmod 600 /etc/project-kagya/*.env
```

Rebuild the frontend if `frontend.env` changed, then restart services:

```bash
sudo -u kagya bash -lc 'set -a; source /etc/project-kagya/frontend.env; set +a; cd /opt/project-kagya/frontend && npm run build'
sudo systemctl start kagya-api kagya-frontend
```

Run the private deployment smoke test after restore:

```bash
KAGYA_ADMIN_TOKEN=replace-with-restored-token scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```

If restore changes model provider settings, verify model cache availability before switching away from `dummy`. If restore changes adapter artifacts or registry state, inspect `/adapters` in the admin UI before activating any adapter.

### 9. Retention And Pruning

Use backups before pruning. `.kagya/` contains private memories, training data, adapter artifacts, and lifecycle state. The default policy is conservative:

- Never prune `.kagya/chroma` with filesystem commands. Use future memory archive/tag tooling instead so DB1/DB2 invariants are preserved.
- Never prune `.kagya/adapter_registry.json`; it is the source of adapter lifecycle truth.
- Never prune `.kagya/agent_journal.jsonl*` with generic filesystem-age rules. Journal rotation and retention are controlled by `agent_journal.max_bytes` and `agent_journal.retained_files`.
- Do not prune active, approved, trial, candidate, or rejected adapter directories. Only archived adapter artifact directories are eligible for manual pruning.
- Preserve the immediate rollback target recorded for the ACTIVE adapter. If activation history is missing or unreadable, `private-prune.sh` protects every archived adapter rather than guessing.
- Evaluation results under `.kagya/eval_results` and dream datasets under `.kagya/dreams` may be pruned after they are older than the operator-selected retention window and a backup exists.
- Runtime lifecycle events are currently in-memory and disappear on process restart. Tool registry definitions persist in `.kagya/tool_registry.json`, and tool audit events persist in inspectable JSONL at `.kagya/tool_audit.jsonl`.
- Hugging Face/model caches are outside `.kagya` and are reproducible. Prune them with provider-specific cache tools only when disk pressure requires it.

Inspect pruning candidates without deleting anything:

```bash
cd /opt/project-kagya
sudo -u kagya KAGYA_RETENTION_DAYS=30 scripts/private-prune.sh
```

After a successful backup, apply pruning with explicit confirmation:

```bash
cd /opt/project-kagya
sudo -u kagya KAGYA_RETENTION_DAYS=30 scripts/private-prune.sh --apply --confirm PRUNE
```

Run the private smoke test after pruning if adapter artifacts or dream/eval files were removed:

```bash
KAGYA_ADMIN_TOKEN=replace-with-token scripts/smoke-private-deploy.sh http://127.0.0.1:8080
```
