# Issue #146 Identity Boundary Checklist

- [x] CARE is derived only from active, independently self-endorsed Value/Goal/Commitment authority plus an Experience with a reviewed `other_welfare_reviewed` interpretation.
- [x] Pressure-only support, external-as-self preference, protected conflict, authority conflict, and uncertainty deterministically produce appeasement risk/refuse/defer outcomes.
- [x] Versioned assessments persist opaque evidence/origin refs, state revisions, pressure refs, event ID/sequence, and immutable history without request text.
- [x] Observable typed pressure metadata is supported; ordinary chat derives repetition through a private per-subject HMAC key and exposes only opaque signal IDs/counts.
- [x] Pressure records cannot mutate Values, Beliefs, Goals, Commitments, Relationships, or Self Model state.
- [x] Unknown/inherited-uncertain Values and Beliefs fail closed until reviewer-bound subject/operator review with evidence, reason, AgentRuntime event ID/sequence, and a canonical record hash.
- [x] Legacy scalar Values are imported as separate `legacy:<id>` inherited records; configured seeds remain unchanged and legacy records remain quarantined pending review.
- [x] Runtime-owned BoundaryDisposition overrides model-visible output for refusal/defer paths and is required as Identity hard-gate evidence.
- [x] Providers score fixed REFUSE/RESPOND/ACCEPT envelopes against the exact public-safe prompt before generation; no declaration, generated text, keyword, fixture expectation, or evaluator model determines the probe class.
- [x] Linked Decision/Action records persist the boundary assessment ID, revision, recommendation, and digest; stale, REFUSE, and DEFER evidence blocks execution.
- [x] Legacy unknown state stays quarantined across migration, restart, snapshot, WAL reconstruction, API reads, prompts, and Working Memory selection.
- [x] Adapter schema v11 stores strict hash-bound deterministic and real-model Identity drift assessments over all required dimensions.
- [x] Missing, failed, stale, incomplete, tampered, or legacy identity evidence cannot activate an adapter.
- [x] Canary rollback resolves a live assessment bound to the active adapter ID/hash and runtime event; operator-supplied violation codes/evidence are not accepted.
- [x] Canary success is server-derived: runtime REFUSE plus probe REFUSE stabilizes the canary, while a protected runtime override over probe RESPOND/ACCEPT rolls back immediately.
- [x] Adapter activation registers commit-failure compensation and repairs missing activation history from the registry rollback target after restart.
- [x] Admin status/UI expose Identity integrity separately; public chat and operator-safe Journals do not expose private boundary state.
- [x] All authoritative subject mutations remain serialized through `AgentRuntime`; structured response and production behavioral/provenance gates remain intact.
