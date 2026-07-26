from pathlib import Path
from typing import cast

from kagya.config import load_settings
from kagya.experience import ExperienceAppraisal
from kagya.external_transaction import ExternalTransactionCoordinator
from kagya.feedback import FeedbackSignal, FeedbackTarget, FeedbackTargetType
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    AgentStateStore,
    EventJournal,
    KagyaMainLoop,
    StateWAL,
    hash_snapshot,
)
from kagya.runtime.agent_runtime import DurableEventJournal


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_runtime_experience_salience_drives_cognition_and_survives_wal_restart(
    tmp_path: Path,
) -> None:
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "experience_runtime_db1",
                    "db2_collection": "experience_runtime_db2",
                    "db1_top_k": 2,
                    "consolidation_min_arousal": 0.6,
                    "consolidation_min_subjective_salience": 0.6,
                }
            ),
            "working_memory": settings.working_memory.model_copy(
                update={"item_capacity": 4, "token_capacity": 45}
            ),
        }
    )
    memory = DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )
    loop = KagyaMainLoop(settings, DummyProvider(), memory)
    state_store = AgentStateStore(tmp_path / "agent-state.json")
    journal = EventJournal(tmp_path / "agent-journal.jsonl")
    wal = StateWAL(tmp_path / "private" / "state-wal.jsonl")
    external = ExternalTransactionCoordinator([memory])
    previous = state_store.save(state_store.capture(loop, 0))
    wal.bootstrap(previous)

    def checkpoint(event: object) -> str:
        nonlocal previous
        event_id = str(getattr(event, "event_id"))
        sequence = int(getattr(event, "processing_sequence"))
        after = state_store.capture(loop, sequence)
        journal.prepared(
            event,  # type: ignore[arg-type]
            state_hash_before=hash_snapshot(previous),
            state_hash_after=hash_snapshot(after),
        )
        wal.append_transition(event, previous, after)  # type: ignore[arg-type]
        state_store.save(after)
        external.finalize_event(event_id, sequence)
        previous = after
        return hash_snapshot(after)

    runtime = AgentRuntime(
        queue_capacity=8,
        event_journal=cast(DurableEventJournal, journal),
        completion_hook=checkpoint,
    )
    runtime.start()
    try:
        high_chat = runtime.execute(
            AgentEventType.CHAT,
            source="test.experience.high",
            handler=lambda: loop.chat("shared retrieval cue"),
        ).value
        low_chat = runtime.execute(
            AgentEventType.CHAT,
            source="test.experience.low",
            handler=lambda: loop.chat("shared retrieval cue"),
        ).value
        high_feedback = runtime.execute(
            AgentEventType.FEEDBACK_UPDATE,
            source="test.experience.high-feedback",
            handler=lambda: loop.submit_feedback(
                target=FeedbackTarget(
                    target_type=FeedbackTargetType.EPISODE,
                    target_id=high_chat.episode_id,
                    episode_id=high_chat.episode_id,
                    experience_id=high_chat.experience_id,
                    context_id=high_chat.context_id,
                ),
                signals=(FeedbackSignal.GOOD,),
                idempotency_key="high-experience-review",
                actor_type="operator",
                actor_id="reviewer",
                source="test.experience.high-feedback",
                feedback_id="high-experience-review",
            ),
        ).value
        low_feedback = runtime.execute(
            AgentEventType.FEEDBACK_UPDATE,
            source="test.experience.low-feedback",
            handler=lambda: loop.submit_feedback(
                target=FeedbackTarget(
                    target_type=FeedbackTargetType.EPISODE,
                    target_id=low_chat.episode_id,
                    episode_id=low_chat.episode_id,
                    experience_id=low_chat.experience_id,
                    context_id=low_chat.context_id,
                ),
                signals=(FeedbackSignal.STYLE_PROBLEM,),
                idempotency_key="low-experience-review",
                actor_type="operator",
                actor_id="reviewer",
                source="test.experience.low-feedback",
                feedback_id="low-experience-review",
            ),
        ).value
        high = runtime.execute(
            AgentEventType.EXPERIENCE_UPDATE,
            source="test.experience.high-revision",
            handler=lambda: loop.reassess_experience(
                high_chat.experience_id,
                appraisal=_appraisal(high=True),
                reason_code="verified_high_salience",
                evidence_refs=(
                    f"feedback:{high_feedback.feedback_id}@"
                    f"{high_feedback.current_revision}",
                ),
                interpretation_codes=("verified_high_salience",),
            ),
        ).value
        low = runtime.execute(
            AgentEventType.EXPERIENCE_UPDATE,
            source="test.experience.low-revision",
            handler=lambda: loop.reassess_experience(
                low_chat.experience_id,
                appraisal=_appraisal(high=False),
                reason_code="verified_low_salience",
                evidence_refs=(
                    f"feedback:{low_feedback.feedback_id}@"
                    f"{low_feedback.current_revision}",
                ),
                interpretation_codes=("verified_low_salience",),
            ),
        ).value
    finally:
        runtime.shutdown()

    decisions = {
        item.item_id: item
        for item in loop.working_memory.select(
            resolver=loop._resolve_working_memory
        ).decisions
    }
    assert high.subjective_salience > low.subjective_salience
    assert (
        decisions[f"episode:{high_chat.episode_id}"].score
        > decisions[f"episode:{low_chat.episode_id}"].score
    )
    assert loop.attention_system.focus.candidate_ids[0] == (
        f"experience:{high.experience_id}"
    )

    retrieved = memory.retrieve_context(
        "shared retrieval cue", current_context_id=high.context_id
    )
    assert retrieved.db1_results[0].id == high_chat.episode_id
    semantic_ids = memory.consolidate_to_semantic(DummyProvider())
    assert len(semantic_ids) == 1
    assert memory.get_episodic(high_chat.episode_id).archived is True  # type: ignore[union-attr]
    assert memory.get_episodic(low_chat.episode_id).archived is False  # type: ignore[union-attr]

    reconstruction = StateWAL(wal.path).reconstruct()
    assert reconstruction.sequence == 6
    assert any(record.event_type == "experience_update" for record in wal.verify())
    assert journal.verify()
    restored = KagyaMainLoop(settings, DummyProvider(), memory)
    state_store.restore_into(restored, reconstruction.snapshot)
    assert restored.get_experience(high.experience_id).subjective_salience == (
        high.subjective_salience
    )
    assert restored.self_model.history
    assert "hidden_thought" not in str(reconstruction.snapshot.model_dump())
    assert "raw_prompt" not in str(reconstruction.snapshot.model_dump())

    restart_previous = reconstruction.snapshot

    def restart_checkpoint(event: object) -> str:
        nonlocal restart_previous
        sequence = int(getattr(event, "processing_sequence"))
        after = state_store.capture(restored, sequence)
        journal.prepared(
            event,  # type: ignore[arg-type]
            state_hash_before=hash_snapshot(restart_previous),
            state_hash_after=hash_snapshot(after),
        )
        wal.append_transition(event, restart_previous, after)  # type: ignore[arg-type]
        state_store.save(after)
        restart_previous = after
        return hash_snapshot(after)

    restarted = AgentRuntime(
        queue_capacity=1,
        initial_sequence=reconstruction.sequence,
        event_journal=cast(DurableEventJournal, journal),
        completion_hook=restart_checkpoint,
    )
    restarted.start()
    try:
        outcome = restarted.execute(
            AgentEventType.EXPERIENCE_READ,
            source="test.experience.restart-read",
            handler=lambda: restored.get_experience(high.experience_id),
        )
    finally:
        restarted.shutdown()
    assert outcome.event.processing_sequence == 7


def _appraisal(*, high: bool) -> ExperienceAppraisal:
    level = 1.0 if high else 0.0
    return ExperienceAppraisal(
        valence=level,
        arousal=level,
        novelty=level,
        novelty_valid=True,
        goal_progress=level,
        threat=level,
        controllability=0.8,
        certainty=0.0 if high else 1.0,
        social_relevance=level,
        effort_cost=0.0,
        reason_codes=("verified_high" if high else "verified_low",),
    )
