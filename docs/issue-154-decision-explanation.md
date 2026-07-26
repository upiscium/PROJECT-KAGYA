# Issue 154: Public-Safe Decision Explanation

## Boundary

`PublicDecisionExplanation` schema v1 is an immutable structured projection of one exact `DecisionRecord` revision and status. It is not chain-of-thought and cannot become decision authority. The authoritative inputs remain the DecisionRecord, selected and considered candidates, explicit candidate source references, linked current boundary assessment, and action/outcome records.

The public deployment boundary currently exposes explanations only through authenticated admin routes. Public chat does not prove that a caller owns a persisted context or interlocutor identity, so adding an unauthenticated endpoint would permit cross-context disclosure. Admin responses use the same public-safe projection and do not include a separate private-content mode.

## Source Admission

- Values appear only when the selected candidate has a recorded per-value contribution and the Decision froze that Value revision.
- Goals, Commitments, Beliefs, and evidence appear only through explicit `goal_refs`, `commitment_refs`, `belief_refs`, and `evidence_refs` on a considered candidate and exact corresponding Decision revision maps.
- The builder never scans all active identity or belief state. Missing, stale, unendorsed, private, context-incompatible, or interlocutor-incompatible references are omitted completely and produce only bounded information-gap codes and counts.
- A context mismatch, or a missing/nonparticipant interlocutor when the Decision context has participants, strips every causal reference except Decision and explanation identity/revision.
- Care or appeasement reasons are copied only from the exact linked, digest-matching current `IdentityBoundaryAssessment` revision.
- Action risk, policy, approval, validation, intent, receipt, Observation, and Verification IDs are projected only from records causally bound to the Decision.

## Lifecycle And Rendering

Create, revise, and render operations require `AgentRuntime` events and idempotency keys. State is retained under `extensions.decision_explanations`, therefore normal snapshot, private WAL reconstruction, and restart behavior apply. DecisionRecords and safely extensible action intents/receipts retain explanation revision references.

A Decision outcome revises every attached context-specific explanation. Prior revisions remain immutable and the new revision records changed fields, outcome status, prediction error, post-assessment reference, event ID, and processing sequence.

Deterministic rendering uses neutral versioned clause templates. Optional natural rendering receives only `public_json()` and may select or reorder offered immutable clause IDs. It must return exactly:

```json
{"explanation_id":"...","explanation_revision":1,"ordered_clause_ids":["disposition.no_op.v1"]}
```

Unknown, duplicate, or mismatched clause IDs, unknown fields, mismatched identity/revision, malformed JSON, provider fallback, and provider failures fail closed. Visible text is always rendered locally; the model cannot provide prose or claims. A failed later render clears prior model ordering and restores deterministic ordering and text.

Schema-valid policy/risk-budget denial is persisted as an `ActionPolicyRejectionRecord`; malformed arguments remain validation failures and are explained as unable rather than policy-blocked. Action status and outcomes are projected only from the latest selected-candidate intent and matching receipt, Observation, and Verification chain.

## Admin API

- `GET /api/decisions/explanations`
- `GET /api/decisions/explanations/{explanation_id}?revision=N`
- `POST /api/decisions/{decision_id}/explanations`
- `POST /api/decisions/explanations/{explanation_id}/revisions`
- `POST /api/decisions/explanations/{explanation_id}/render`

The frontend `/decisions` view displays deterministic structure, uncertainty/information gaps, source availability, risk/policy/approval status, renderer state, outcomes, and revision links without hidden internals.
