import pytest

from kagya.runtime import (
    ContextRegistry,
    ContextStatus,
    InferredAttribute,
    InterlocutorModel,
)


def test_context_lifecycle_create_suspend_resume_and_end() -> None:
    registry = ContextRegistry()
    created = registry.create(
        context_type="conversation",
        source_channel="api.chat",
        source_session_id="session-1",
        participant_ids=("person-1",),
        active_topic="gardens",
        active_task="plan",
    )

    assert created.status == ContextStatus.ACTIVE
    assert registry.suspend(created.context_id).status == ContextStatus.SUSPENDED
    assert registry.resume(created.context_id).status == ContextStatus.ACTIVE
    assert registry.end(created.context_id).status == ContextStatus.CLOSED
    with pytest.raises(ValueError, match="cannot be resumed"):
        registry.resume(created.context_id)


def test_context_compatibility_distinguishes_relationships() -> None:
    registry = ContextRegistry()
    current = registry.create(context_id="current", participant_ids=("person-1",))
    related = registry.create(context_id="related")
    registry.create(context_id="shared", participant_ids=("person-1",))
    registry.create(context_id="unrelated", participant_ids=("person-2",))
    registry.relate(current.context_id, related.context_id)

    assert registry.compatibility("current", current) == (1.0, "same_context")
    assert registry.compatibility("related", current) == (0.75, "related")
    assert registry.compatibility("shared", current) == (0.65, "shared_interlocutor")
    assert registry.compatibility("unrelated", current) == (0.2, "unrelated")
    assert registry.compatibility(None, current) == (0.45, "legacy_unknown")


def test_interlocutor_inference_keeps_confidence_and_evidence() -> None:
    registry = ContextRegistry()
    model = InterlocutorModel(
        identity_key="person-1",
        inferred_preferences={
            "language": InferredAttribute(
                value="Japanese",
                confidence=0.6,
                evidence_references=("episode-1",),
            )
        },
        uncertainties={
            "timezone": InferredAttribute(value="unknown", confidence=0.1)
        },
    )

    registry.register_interlocutor(model)
    registry.record_shared_history(("person-1",), "episode:one")

    restored = registry.interlocutors[0]
    assert restored.inferred_preferences["language"].confidence == 0.6
    assert restored.inferred_preferences["language"].evidence_references == (
        "episode-1",
    )
    assert restored.uncertainties["timezone"].confidence == 0.1
    assert restored.shared_history_references == ("episode:one",)
