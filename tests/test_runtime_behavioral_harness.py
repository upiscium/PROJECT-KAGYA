from datetime import UTC, datetime
from pathlib import Path
import json

import pytest

from kagya.actions import ACTION_STATE_KEY, ActionState, ActionValidationRecord
from kagya.config import load_settings
from kagya.learning import (
    AuthoritativeTransitionCollector,
    BehavioralArtifactStore,
    ControlledClock,
    FailureInjector,
    SubjectRuntimeHarness,
)
from kagya.identity import OriginActor
from kagya.motivation import CommitmentStatus
from kagya.runtime import AgentEventType, JournalLifecycle, hash_snapshot
from kagya.outbox import OutboxMessageKind
from kagya.learning.runtime_behavioral_runner import (
    _create_action_decision,
    _create_runtime_intent,
)
from kagya.training import DatasetCandidate, DatasetGovernanceStore, DatasetProvenance


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _harness(
    tmp_path: Path, injector: FailureInjector | None = None
) -> SubjectRuntimeHarness:
    return SubjectRuntimeHarness(
        tmp_path / "subject",
        load_settings(CONFIG_PATH),
        subject_id="candidate",
        failure_injector=injector,
    )


def test_collector_derives_runtime_diff_and_durable_evidence(tmp_path: Path) -> None:
    harness = _harness(tmp_path).create().start()
    collector = AuthoritativeTransitionCollector(harness)

    harness.execute(
        AgentEventType.GOAL_UPDATE,
        lambda loop: loop.create_commitment(
            commitment_id="collector-commitment",
            description="External proposal",
            origin_actor=OriginActor.OPERATOR,
            origin_source_ref="test:external-proposal",
        ),
    )
    harness.execute(
        AgentEventType.GOAL_UPDATE,
        lambda loop: loop.accept_commitment(
            "collector-commitment",
            self_endorsement="subject_acceptance:collector-commitment",
        ),
    )
    trace = harness.capture_trace(collector)
    harness.shutdown()

    transition = next(
        item
        for item in trace.transitions
        if item.path == ("domains", "commitments", "collector-commitment")
    )
    assert transition.event_sequence == 2
    assert "test:external-proposal" in transition.evidence_refs
    assert any("accept" in reference for reference in transition.evidence_refs)
    assert not any(
        reference.startswith(("journal:", "wal:"))
        for reference in transition.evidence_refs
    )


def test_restart_builds_fresh_graph_and_restores_filesystem_state(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path).create().start()
    harness.execute(
        AgentEventType.GOAL_UPDATE,
        lambda loop: loop.create_commitment(
            commitment_id="restart-commitment",
            description="Persist responsibility",
        ),
    )
    old_graph = harness.graph
    old_loop = old_graph.main_loop if old_graph is not None else None

    harness.restart()

    assert harness.graph is not old_graph
    assert harness.graph is not None
    assert harness.graph.main_loop is not old_loop
    commitments = harness.capture_authoritative_state()["domains"]["commitments"]
    assert commitments[0]["commitment_id"] == "restart-commitment"
    assert commitments[0]["status"] == CommitmentStatus.PROPOSED.value
    harness.shutdown()


def test_homeostatic_motivation_uses_controlled_clock_across_restart(
    tmp_path: Path,
) -> None:
    clock = ControlledClock(datetime(2001, 1, 1, tzinfo=UTC))
    harness = SubjectRuntimeHarness(
        tmp_path / "subject",
        load_settings(CONFIG_PATH),
        subject_id="candidate",
        clock=clock,
    ).create().start()

    def reinforce_homeostasis(loop):
        loop.record_homeostatic_state(valence=-0.8, arousal=0.8)
        return loop.derive_structured_motivations()[0]

    first = harness.execute(AgentEventType.MOTIVATION_UPDATE, reinforce_homeostasis).value
    harness.execute(AgentEventType.MOTIVATION_UPDATE, reinforce_homeostasis)
    assert first.created_at == clock.now().isoformat()
    assert harness.graph is not None
    signal = harness.graph.main_loop.persistent_state.motivation_extensions[
        "homeostatic_signal"
    ]
    assert signal["observed_at"] == clock.now().isoformat()

    clock.advance(59)
    _, goals = harness.execute(
        AgentEventType.INTRINSIC_GOAL_PROPOSE,
        lambda loop: loop.reevaluate_motivation(),
    ).value
    assert goals == []

    harness.restart()
    clock.advance(1)
    _, goals = harness.execute(
        AgentEventType.INTRINSIC_GOAL_PROPOSE,
        lambda loop: loop.reevaluate_motivation(),
    ).value
    assert len(goals) == 1
    harness.shutdown()


def test_rejected_action_validation_survives_restart_and_wal_without_raw_args(
    tmp_path: Path,
) -> None:
    harness = _harness(tmp_path).create().start()
    private_path = "../private-validation-sentinel"
    harness.execute(
        AgentEventType.DECISION_UPDATE,
        lambda loop: _create_action_decision(
            loop,
            "rejected-validation",
            "document_search",
            {"query": "secret-query", "relative_path": private_path},
        ),
    )
    outcome = harness.execute(
        AgentEventType.ACTION_INTENT,
        lambda _loop: _create_runtime_intent(harness, "rejected-validation"),
    )

    record = outcome.value
    assert isinstance(record, ActionValidationRecord)
    assert record.arguments_valid is False
    assert record.validated_event_id == outcome.event.event_id
    assert record.validated_event_sequence == outcome.event.processing_sequence
    assert harness.graph is not None
    assert harness.graph.action_execution.list_intents() == ()
    assert harness.graph.action_execution.list_receipts() == ()
    assert harness.action_attempts()[0].arguments_valid is False
    reconstructed = harness.graph.state_wal.reconstruct().snapshot
    wal_state = ActionState.model_validate(
        reconstructed.extensions[ACTION_STATE_KEY]
    )
    assert wal_state.validation_records == (record,)
    serialized_action_state = json.dumps(wal_state.model_dump(mode="json"), sort_keys=True)
    assert private_path not in serialized_action_state
    assert "secret-query" not in serialized_action_state
    public_result = json.dumps(harness.capture_authoritative_state(), sort_keys=True)
    journal = harness.graph.event_journal.path.read_text(encoding="utf-8")
    assert private_path not in public_result
    assert "secret-query" not in public_result
    assert private_path not in journal
    assert "secret-query" not in journal

    harness.restart()

    assert harness.graph is not None
    assert harness.graph.action_execution.list_validation_records() == (record,)
    assert harness.action_attempts()[0].arguments_valid is False
    harness.shutdown()


def test_snapshot_committed_before_journal_completion_recovers_without_replay(
    tmp_path: Path,
) -> None:
    injector = FailureInjector({"before_journal_completed"})
    harness = _harness(tmp_path, injector).create().start()

    with pytest.raises(Exception, match="completion could not be committed"):
        harness.execute(
            AgentEventType.STATE_SNAPSHOT,
            lambda loop: loop.persistent_state.extensions.update(
                {"exactly_once": {"count": 1}}
            ),
        )
    harness.abrupt_stop()
    harness = _harness(tmp_path).create().start()

    state = harness.capture_authoritative_state()
    assert state["extensions"]["exactly_once"]["count"] == 1
    assert state["last_processed_event_sequence"] == 1
    assert any(
        record.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
        and record.failure_category == "committed_before_crash"
        for record in harness.journal_records()
    )
    harness.shutdown()


def test_wal_reconstructs_corrupt_snapshot_and_verifies_hash(tmp_path: Path) -> None:
    harness = _harness(tmp_path).create().start()
    harness.execute(
        AgentEventType.STATE_SNAPSHOT,
        lambda loop: loop.persistent_state.extensions.update({"wal_probe": "retained"}),
    )
    assert harness.graph is not None
    expected_hash = harness.graph.state_wal.reconstruct().snapshot_hash
    snapshot_path = harness.graph.state_store.path
    harness.shutdown()
    snapshot_path.write_text("{corrupt", encoding="utf-8")

    restored = _harness(tmp_path).create().start()

    assert restored.graph is not None
    assert restored.graph.state_store.last_snapshot is not None
    assert hash_snapshot(restored.graph.state_store.last_snapshot) == expected_hash
    assert (
        restored.capture_authoritative_state()["extensions"]["wal_probe"] == "retained"
    )
    restored.shutdown()


def test_corrupt_journal_fails_closed_and_rejects_events(tmp_path: Path) -> None:
    harness = _harness(tmp_path).create().start()
    harness.execute(AgentEventType.STATE_SNAPSHOT, lambda _loop: None)
    assert harness.graph is not None
    journal_path = harness.graph.event_journal.path
    harness.shutdown()
    with journal_path.open("a", encoding="utf-8") as output:
        output.write("{corrupt\n")

    failed = _harness(tmp_path).create()

    assert failed.readiness is False
    assert failed.startup_error is not None
    with pytest.raises(RuntimeError, match="journal"):
        failed.start()


def test_wal_append_failure_leaves_snapshot_uncommitted(tmp_path: Path) -> None:
    injector = FailureInjector()
    harness = _harness(tmp_path, injector).create().start()
    injector.arm("before_wal_append")

    with pytest.raises(Exception, match="completion could not be committed"):
        harness.execute(
            AgentEventType.STATE_SNAPSHOT,
            lambda loop: loop.persistent_state.extensions.update(
                {"must_not_commit": True}
            ),
        )
    harness.abrupt_stop()
    harness = _harness(tmp_path).create().start()

    assert "must_not_commit" not in harness.capture_authoritative_state()["extensions"]
    assert any(
        record.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
        and record.failure_category == "uncommitted_after_crash"
        for record in harness.journal_records()
    )
    harness.shutdown()


def test_external_prepare_then_snapshot_failure_is_compensated_on_restart(
    tmp_path: Path,
) -> None:
    injector = FailureInjector()
    harness = _harness(tmp_path, injector).create().start()
    injector.arm("snapshot_temp_fsynced")

    with pytest.raises(Exception, match="completion could not be committed"):
        harness.execute(
            AgentEventType.CHAT,
            lambda loop: loop.chat(
                "stage an external observation", create_context=True
            ),
        )
    harness.abrupt_stop()
    harness = _harness(tmp_path).create().start()

    assert harness.graph is not None
    records = harness.graph.external_transactions.records()
    assert records
    assert {record.status.value for record in records} == {"compensated"}
    harness.shutdown()


def test_abrupt_stop_terminates_threads_before_fresh_reconstruction(
    tmp_path: Path,
) -> None:
    crashed = _harness(tmp_path).create().start()
    assert crashed.graph is not None
    old_graph = crashed.graph
    crashed.execute(
        AgentEventType.GOAL_UPDATE,
        lambda loop: loop.create_commitment(
            commitment_id="abrupt-commitment",
            description="Survive abrupt stop",
        ),
    )

    crashed.abrupt_stop()

    assert old_graph.runtime.is_alive is False
    assert old_graph.autonomy_loop._thread is None
    recovered = _harness(tmp_path).create().start()
    assert recovered is not crashed
    assert recovered.graph is not old_graph
    assert (
        recovered.capture_authoritative_state()["domains"]["commitments"][0][
            "commitment_id"
        ]
        == "abrupt-commitment"
    )
    recovered.shutdown()


def test_duplicate_outbox_delivery_input_has_exactly_one_effect(tmp_path: Path) -> None:
    harness = _harness(tmp_path).create().start()

    def enqueue_twice(_loop: object) -> None:
        assert harness.graph is not None
        for _ in range(2):
            harness.graph.outbox.enqueue(
                OutboxMessageKind.ACTION_RESULT,
                title="one",
                body="one",
                deduplication_key="delivery:one",
            )

    harness.execute(AgentEventType.OUTBOX_ENQUEUE, enqueue_twice)

    assert harness.graph is not None
    assert len(harness.graph.outbox.list_messages()) == 1
    assert harness.graph.outbox.list_messages()[0].deduplication_key == "delivery:one"
    harness.shutdown()


@pytest.mark.parametrize(
    ("checkpoint", "expected_classification", "committed"),
    (
        ("journal_accepted", "accepted_not_started", False),
        ("journal_started", "uncommitted_after_crash", False),
        ("external_write", "failed_before_commit", False),
        ("external_prepare", "uncommitted_after_crash", False),
        ("before_wal_append", "uncommitted_after_crash", False),
        ("after_wal_append", "uncommitted_after_crash", False),
        ("snapshot_temp_write", "uncommitted_after_crash", False),
        ("snapshot_temp_fsync", "uncommitted_after_crash", False),
        ("snapshot_atomic_replace", "committed_before_crash", True),
        ("before_finalize", "committed_before_crash", True),
        ("after_finalize", "committed_before_crash", True),
        ("before_journal_completed", "committed_before_crash", True),
    ),
)
def test_every_runtime_commit_checkpoint_recovers_with_fresh_harness(
    tmp_path: Path,
    checkpoint: str,
    expected_classification: str,
    committed: bool,
) -> None:
    injector = FailureInjector()
    crashed = _harness(tmp_path, injector).create().start()
    injector.arm(checkpoint)
    crashed.provider.queue_responses("A bounded public response.")

    with pytest.raises(Exception):
        crashed.execute(
            AgentEventType.CHAT,
            lambda loop: loop.chat(
                "checkpoint observation",
                create_context=True,
                interlocutor_key="checkpoint-operator",
            ),
        )
    crashed.abrupt_stop()

    recovered = _harness(tmp_path).create().start()
    assert recovered.graph is not None
    snapshot = recovered.graph.state_store.last_snapshot
    assert snapshot is not None
    wal = recovered.wal_records()
    assert wal[-1].state_hash_after == hash_snapshot(snapshot)
    assert wal[-1].processing_sequence == snapshot.last_processed_event_sequence
    classifications = [
        record.failure_category
        for record in recovered.journal_records()
        if record.lifecycle == JournalLifecycle.RECOVERY_CLASSIFIED
    ]
    if any(
        record.lifecycle == JournalLifecycle.FAILED
        for record in recovered.journal_records()
    ):
        classifications.append("failed_before_commit")
    assert expected_classification in classifications
    assert snapshot.last_processed_event_sequence == (1 if committed else 0)

    transactions = recovered.graph.external_transactions.records()
    assert len({record.transaction_id for record in transactions}) == len(transactions)
    assert len(transactions) <= 1
    if committed and transactions:
        assert {record.status.value for record in transactions} == {"committed"}
    elif transactions:
        assert {record.status.value for record in transactions} == {"compensated"}
    recovered.shutdown()


def test_private_provider_sentinels_never_cross_persistence_boundaries(
    tmp_path: Path,
) -> None:
    sentinels = (
        "hidden-thought-sentinel",
        "private-prompt-sentinel",
        "private-context-sentinel",
        "tool-secret-sentinel",
    )
    harness = _harness(tmp_path).create().start()
    harness.provider.generate(" ".join(sentinels))
    harness.provider.queue_responses("A public bounded response.")
    collector = AuthoritativeTransitionCollector(harness)
    harness.execute(
        AgentEventType.CHAT,
        lambda loop: loop.chat("ordinary public observation", create_context=True),
    )
    trace = harness.capture_trace(collector, "A public bounded response.")
    BehavioralArtifactStore(harness.root / "results").commit(
        "privacy-scan", trace.model_dump(mode="json")
    )
    DatasetGovernanceStore(harness.root / "datasets").create_revision(
        [
            DatasetCandidate(
                input="ordinary public observation",
                output="A public bounded response.",
                provenance=DatasetProvenance("episode", "privacy-episode"),
            )
        ],
        source_job_id="privacy-scan",
    )
    harness.shutdown()

    persisted = b"".join(
        path.read_bytes() for path in harness.root.rglob("*") if path.is_file()
    )
    for sentinel in sentinels:
        assert sentinel.encode() not in persisted
