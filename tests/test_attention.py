from kagya.attention import (
    AttentionCandidateStatus,
    AttentionSource,
    AttentionSystem,
)


def test_salience_drive_urgency_and_commitment_compete_with_finite_capacity() -> None:
    attention = AttentionSystem(capacity=2, high_arousal_cap=1, switch_cost=0.1)
    attention.observe(
        candidate_id="experience:urgent",
        target_ref="experience:urgent",
        source=AttentionSource.EXPERIENCE,
        source_refs=("experience:urgent",),
        salience=1.0,
        novelty=0.8,
        arousal=0.9,
    )
    attention.observe(
        candidate_id="motivation:closure",
        target_ref="context:one",
        source=AttentionSource.MOTIVATION,
        source_refs=("motivation:closure", "experience:prior"),
        drive=0.9,
        persistence=0.8,
    )
    attention.observe(
        candidate_id="goal:durable",
        target_ref="goal:durable",
        source=AttentionSource.GOAL,
        source_refs=("goal:durable", "value:care"),
        urgency=0.8,
        value_relevance=0.9,
        commitment_cost=1.0,
        persistence=1.0,
    )

    focus = attention.compete(event_id="event:one", event_sequence=1)

    assert len(focus.candidate_ids) == 2
    assert "goal:durable" in focus.candidate_ids
    assert focus.provenance_refs
    assert attention.history[-1].reason_codes


def test_focus_persistence_switch_cost_unfinished_state_and_habituation() -> None:
    attention = AttentionSystem(capacity=1, switch_cost=0.3)
    attention.observe(
        candidate_id="goal:long-term",
        target_ref="goal:long-term",
        source=AttentionSource.GOAL,
        source_refs=("goal:long-term",),
        urgency=0.7,
        persistence=1.0,
    )
    assert attention.compete().candidate_ids == ("goal:long-term",)
    attention.observe(
        candidate_id="experience:transient",
        target_ref="experience:transient",
        source=AttentionSource.EXPERIENCE,
        source_refs=("experience:transient",),
        salience=0.75,
        novelty=0.8,
    )
    assert attention.compete().candidate_ids == ("goal:long-term",)
    before = attention.get("goal:long-term").habituation

    focus = attention.refocus(
        ("experience:transient",),
        reason_code="operator_priority",
        provenance_refs=("decision:review",),
    )

    assert focus.unfinished_candidate_ids == ("goal:long-term",)
    assert attention.history[-1].switch_cost == 0.3
    assert attention.get("goal:long-term").habituation >= before


def test_high_arousal_cap_deliberate_controls_idle_and_round_trip() -> None:
    attention = AttentionSystem(capacity=2, high_arousal_cap=1)
    for identifier in ("one", "two"):
        attention.observe(
            candidate_id=f"experience:{identifier}",
            target_ref=f"experience:{identifier}",
            source=AttentionSource.EXPERIENCE,
            source_refs=(f"experience:{identifier}",),
            salience=1.0,
            arousal=1.0,
        )
    focus = attention.compete()
    assert len(focus.candidate_ids) == 1
    assert "high_arousal_cap" in focus.reason_codes

    attention.defer(
        focus.candidate_ids[0],
        reason_code="not_now",
        provenance_refs=("decision:defer",),
    )
    remaining = next(
        item.candidate_id
        for item in attention.list_candidates()
        if item.status == AttentionCandidateStatus.AVAILABLE
    )
    attention.ignore(
        remaining,
        reason_code="not_relevant",
        provenance_refs=("decision:ignore",),
    )
    assert attention.compete().idle is True

    restored = AttentionSystem(capacity=2)
    restored.restore(attention.to_json())
    assert restored.to_json() == attention.to_json()


def test_candidate_rejects_content_instead_of_persisting_raw_prompt() -> None:
    attention = AttentionSystem(capacity=1)

    try:
        attention.observe(
            candidate_id="candidate with raw content",
            target_ref="prompt:secret text",
            source=AttentionSource.EXTERNAL,
            source_refs=("prompt:secret text",),
        )
    except ValueError as exc:
        assert "opaque safe reference" in str(exc)
    else:
        raise AssertionError("free-form attention content must be rejected")


def test_source_synchronization_retires_stale_candidates() -> None:
    attention = AttentionSystem(capacity=1)
    attention.observe(
        candidate_id="goal:completed",
        target_ref="goal:completed",
        source=AttentionSource.GOAL,
        source_refs=("goal:completed",),
        urgency=1.0,
        persistence=1.0,
    )
    assert attention.compete().candidate_ids == ("goal:completed",)

    attention.synchronize_source(AttentionSource.GOAL, set())
    focus = attention.compete()

    assert attention.get("goal:completed").status == AttentionCandidateStatus.INACTIVE
    assert focus.idle is True
    assert focus.unfinished_candidate_ids == ("goal:completed",)
