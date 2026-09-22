"""Focused tests for the pure R12 Belief authority boundary."""

from datetime import UTC, datetime
from dataclasses import replace

import pytest

from kagya.belief import (
    AdmissionReason,
    BeliefEvidence,
    BeliefEvidenceType,
    BeliefLifecycle,
    BeliefProposition,
    BeliefRecord,
    BeliefRevisionOperation,
    BeliefRevisionReason,
    BeliefRevisionRecord,
    BeliefSubjectAdmission,
    EpistemicStatus,
    belief_record_digest,
    build_conflict_candidate,
    is_conflict_candidate,
)


NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_proposition_normalization_and_literal_golden_digest() -> None:
    proposition = BeliefProposition("  Alice\tlikes  tea ")
    assert proposition.canonical_text == "Alice likes tea"
    assert proposition.proposition_digest == (
        "cefd66d195cffa9a08fb839a9da2bcebaae42ba8cca2a1fce01c753e7f977972"
    )
    with pytest.raises(AttributeError):
        proposition.canonical_text = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        BeliefProposition("x" * 2_001)
    with pytest.raises(ValueError):
        BeliefProposition("a\x01b")


def test_structured_projection_is_optional_and_not_identity() -> None:
    plain = BeliefProposition("same")
    first = BeliefProposition("same", "Alice", "likes", "tea")
    second = BeliefProposition("same", "Bob", "hates", "coffee")
    assert plain.structured_projection is None
    assert first.proposition_digest == second.proposition_digest == plain.proposition_digest
    with pytest.raises(ValueError):
        BeliefProposition("partial", subject="Alice")
    with pytest.raises(ValueError):
        BeliefProposition("component", "a", "p", "x" * 257)


def test_same_identity_with_different_projection_is_not_a_conflict() -> None:
    left = BeliefProposition("same", "Alice", "likes", "tea")
    right = BeliefProposition("same", "Alice", "likes", "coffee")
    assert left.proposition_digest == right.proposition_digest
    assert build_conflict_candidate(left, right) is None


def test_conflict_candidate_is_a_pure_hint_only() -> None:
    left = BeliefProposition("Alice likes tea", "Alice", "likes", "tea")
    right = BeliefProposition("Alice likes coffee", "Alice", "likes", "coffee")
    assert is_conflict_candidate(left, right, ("context:home",), ("context:home",))
    assert not is_conflict_candidate(left, right, ("context:home",), ("context:work",))
    candidate = build_conflict_candidate(left, right, ("context:home",), ("context:home",))
    assert candidate is not None
    assert left.object == "tea" and right.object == "coffee"
    assert build_conflict_candidate(BeliefProposition("plain"), right) is None


def test_admission_is_reference_only_and_has_stable_digest() -> None:
    proposition = BeliefProposition("claim")
    evidence = BeliefEvidence("event:1", BeliefEvidenceType.EXTERNAL_CLAIM)
    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        (evidence.evidence_ref,),
        "event:2",
        3,
        AdmissionReason.SUBJECT_ENDORSEMENT,
    )
    assert admission.admission_digest == (
        "58604d8bc29dbd2962fa48edeaf2eb5605dbb0075ed48847a52395db7296773b"
    )
    assert admission.evidence_refs == ("event:1",)
    with pytest.raises(ValueError):
        BeliefSubjectAdmission(
            proposition.proposition_digest,
            ("event:2", "event:1"),
            "event:2",
            3,
            AdmissionReason.SUBJECT_ENDORSEMENT,
        )
    with pytest.raises(ValueError):
        BeliefSubjectAdmission(
            proposition.proposition_digest,
            ("event:1",),
            "event:2",
            0,
            AdmissionReason.SUBJECT_ENDORSEMENT,
        )
    with pytest.raises(ValueError):
        BeliefSubjectAdmission(
            proposition.proposition_digest,
            ("event:1",),
            "event:2",
            True,  # type: ignore[arg-type]
            AdmissionReason.SUBJECT_ENDORSEMENT,
        )


def test_external_evidence_does_not_adopt_a_belief() -> None:
    proposition = BeliefProposition("claim")
    record = BeliefRecord(
        "belief:1",
        proposition,
        BeliefLifecycle.PROPOSED,
        EpistemicStatus.ESTABLISHED,
        1.0,
        evidence=(BeliefEvidence("external:1", BeliefEvidenceType.EXTERNAL_CLAIM),),
    )
    assert not record.is_active
    assert not record.is_ordinary_active()


def test_adopted_uncertain_is_valid_but_not_ordinary_active() -> None:
    proposition = BeliefProposition("claim")
    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        ("event:1",),
        "event:2",
        3,
        AdmissionReason.SUBJECT_REVIEW,
    )
    record = BeliefRecord(
        "belief:1",
        proposition,
        BeliefLifecycle.ADOPTED,
        EpistemicStatus.UNCERTAIN,
        0.5,
        evidence=(BeliefEvidence("event:1", BeliefEvidenceType.EXPERIENCE),),
        subject_admission=admission,
    )
    assert not record.is_active
    assert not record.is_ordinary_active()


def test_adopted_admission_evidence_must_exactly_match_record_evidence() -> None:
    proposition = BeliefProposition("claim")
    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        ("event:1",),
        "event:2",
        3,
        AdmissionReason.SUBJECT_ENDORSEMENT,
    )
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.ADOPTED,
            EpistemicStatus.PROBABLE,
            0.5,
            subject_admission=admission,
        )
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.ADOPTED,
            EpistemicStatus.PROBABLE,
            0.5,
            evidence=(
                BeliefEvidence("event:1", BeliefEvidenceType.EXPERIENCE),
                BeliefEvidence("event:3", BeliefEvidenceType.EXPERIENCE),
            ),
            subject_admission=admission,
        )
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.ADOPTED,
            EpistemicStatus.PROBABLE,
            0.5,
            evidence=(BeliefEvidence("event:3", BeliefEvidenceType.EXPERIENCE),),
            subject_admission=admission,
        )


def test_ordinary_active_projection_is_bounded_and_pure() -> None:
    proposition = BeliefProposition("claim")
    admission = BeliefSubjectAdmission(
        proposition.proposition_digest,
        ("event:1",),
        "event:2",
        3,
        AdmissionReason.SUBJECT_ENDORSEMENT,
    )
    evidence = BeliefEvidence("event:1", BeliefEvidenceType.EXPERIENCE)
    record = BeliefRecord(
        "belief:1",
        proposition,
        BeliefLifecycle.ADOPTED,
        EpistemicStatus.PROBABLE,
        0.8,
        context_scope=("context:home",),
        valid_from=NOW,
        valid_until=datetime(2026, 1, 2, tzinfo=UTC),
        evidence=(evidence,),
        subject_admission=admission,
    )
    assert record.is_ordinary_active(at=NOW, context_id="context:home")
    assert not record.is_ordinary_active(at=NOW, context_id="context:work")
    assert not record.is_ordinary_active(at=datetime(2026, 1, 3, tzinfo=UTC), context_id="context:home")
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.PROPOSED,
            EpistemicStatus.UNKNOWN,
            0.5,
            context_scope=tuple(f"context:{index}" for index in range(17)),
        )
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.PROPOSED,
            EpistemicStatus.UNKNOWN,
            0.5,
            evidence=tuple(
                BeliefEvidence(f"event:{index}", BeliefEvidenceType.EXPERIENCE)
                for index in range(33)
            ),
        )
    proposed = BeliefRecord(
        "belief:1",
        proposition,
        BeliefLifecycle.PROPOSED,
        EpistemicStatus.UNKNOWN,
        0.5,
    )
    with pytest.raises(ValueError):
        replace(proposed, revision=2**31)
    with pytest.raises(ValueError):
        replace(proposed, revision=1)
    with pytest.raises(ValueError):
        replace(proposed, history_anchor_digest="0" * 64)


def test_revision_records_and_genesis_anchors_are_strict() -> None:
    genesis = BeliefRevisionRecord(
        "belief:1",
        0,
        BeliefRevisionOperation.CREATE,
        BeliefRevisionReason.CREATION,
        NOW,
    )
    with pytest.raises(ValueError):
        BeliefRevisionRecord(
            "belief:1",
            0,
            BeliefRevisionOperation.CREATE,
            BeliefRevisionReason.CREATION,
            NOW,
            previous_revision_digest="0" * 64,
        )
    with pytest.raises(ValueError):
        BeliefRevisionRecord(
            "belief:1",
            1,
            BeliefRevisionOperation.CORRECT,
            BeliefRevisionReason.CORRECTION,
            NOW,
        )
    corrected = BeliefRevisionRecord(
        "belief:1",
        1,
        BeliefRevisionOperation.CORRECT,
        BeliefRevisionReason.CORRECTION,
        NOW,
        previous_revision_digest=genesis.record_digest,
    )
    assert corrected.previous_revision_digest == genesis.record_digest


def test_supersession_references_obey_local_lifecycle_invariants() -> None:
    proposition = BeliefProposition("claim")
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.SUPERSEDED,
            EpistemicStatus.PROBABLE,
            0.5,
        )
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.PROPOSED,
            EpistemicStatus.UNKNOWN,
            0.5,
            superseded_by_id="belief:2",
        )
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.SUPERSEDED,
            EpistemicStatus.PROBABLE,
            0.5,
            superseded_by_id="belief:1",
        )
    with pytest.raises(ValueError):
        BeliefRecord(
            "belief:1",
            proposition,
            BeliefLifecycle.PROPOSED,
            EpistemicStatus.UNKNOWN,
            0.5,
            supersedes_id="belief:1",
        )


def test_whole_record_digest_binds_projection_but_not_proposition_identity() -> None:
    plain = BeliefRecord(
        "belief:1",
        BeliefProposition("same"),
        BeliefLifecycle.PROPOSED,
        EpistemicStatus.UNKNOWN,
        0.5,
    )
    projected = BeliefRecord(
        "belief:1",
        BeliefProposition("same", "Alice", "likes", "tea"),
        BeliefLifecycle.PROPOSED,
        EpistemicStatus.UNKNOWN,
        0.5,
    )
    assert plain.proposition.proposition_digest == projected.proposition.proposition_digest
    assert belief_record_digest(plain) != belief_record_digest(projected)
