# Main-loop coordinator boundaries

`KagyaMainLoop` remains the authoritative compatibility facade. Its private
implementation owns only dependency construction, coordinator wiring, and
operational telemetry. Domain behavior is owned by coordinators in dependency
order: persistence, experience integration, motivation/goal, plan/decision,
identity/narrative, chat orchestration, and action. Coordinator methods are
inherited by the facade to preserve its API without forwarding wrappers, while
composed coordinator instances handle typed cross-domain operations.

Coordinators receive stores and persistence callbacks explicitly. A store never
receives another store; cross-domain handoff uses immutable domain records such
as `ExperienceIntegrationResult` and `ChatResult`. `AgentRuntime` remains the
only authoritative mutation sequence.

Chat executes in this order:

1. Prepare context and local calibration.
2. Appraise, retrieve context, and generate a response.
3. Prepare the external episodic artifact.
4. Integrate authoritative domain records.
5. Commit the authoritative snapshot at the `AgentRuntime` event boundary.

Failures before the event commit restore the local domain snapshots captured by
the chat transaction. The external episodic write is a saga: it remains pending
until snapshot commit, while the runtime failure hook marks it orphaned and
compensates it. Restart reconciliation finalizes a pending artifact only when
the journal proves that its authoritative snapshot committed.

`ChatTransactionCallbacks` exposes the same five stages as independently
testable operations. A stage failure invokes local-domain rollback followed by
external-saga rollback; failure of one rollback scope does not prevent the
other, and the original stage failure remains authoritative.

All mutation entry points continue to run as `AgentRuntime` handlers. HTTP
routes may read stores for projections, but writes go through the main-loop or
the action coordinator rather than mutating a store directly.
