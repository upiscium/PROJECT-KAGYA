import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionStore,
    DecisionExplanationStore,
    PublicDecisionExplanation,
    RendererState,
    build_explanation,
    render_natural,
)


@pytest.mark.parametrize(
    ("action_type", "disposition"),
    [
        (ActionType.NO_OP, "no_op"),
        (ActionType.REFUSE, "refuse"),
        (ActionType.DEFER, "defer"),
        (ActionType.REQUEST_INFORMATION, "request_information"),
        (ActionType.UNABLE, "unable"),
        (ActionType.REPLAN, "replan"),
    ],
)
def test_fallback_dispositions_are_structured_without_keyword_inference(
    action_type: ActionType, disposition: str
) -> None:
    decision = DecisionStore().create(
        [_candidate("selected", action_type)],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
    )

    explanation = build_explanation(
        _main_loop(),
        decision,
        event_id="event-2",
        event_sequence=2,
    )

    assert explanation.disposition.value == disposition
    assert explanation.selected.action_type == action_type.value
    assert explanation.information_gap_codes == ()


def test_builder_uses_only_selected_explicit_references_and_marks_missing() -> None:
    selected = ActionCandidate(
        **{
            **_candidate("selected", ActionType.NO_OP).__dict__,
            "value_effects": {"referenced-value": 0.5},
            "belief_refs": ("missing-belief",),
            "evidence_refs": ("evidence-1",),
        }
    )
    unrelated = _candidate("unrelated", ActionType.DEFER)
    store = DecisionStore()
    decision = store.create(
        [selected, unrelated],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id="context-1",
        active_goal_ids=(),
        value_revision_refs={"referenced-value": 2, "unrelated-value": 9},
        belief_revision_refs={},
        emotion_snapshot={},
        value_evaluator=lambda _options: {
            "selected": {"referenced-value": 0.4},
            "unrelated": {},
        },
    )
    main_loop = _main_loop()
    main_loop.value_system.values = {
        "referenced-value": SimpleNamespace(
            revision=2,
            supporting_evidence_ids=("evidence-1",),
            opposing_evidence_ids=(),
            origin_provenance=SimpleNamespace(
                endorsement=SimpleNamespace(value="endorsed")
            ),
        ),
        "unrelated-value": SimpleNamespace(
            revision=9,
            supporting_evidence_ids=(),
            opposing_evidence_ids=(),
            origin_provenance=SimpleNamespace(
                endorsement=SimpleNamespace(value="endorsed")
            ),
        ),
    }

    explanation = build_explanation(
        main_loop,
        decision,
        event_id="event-2",
        event_sequence=2,
        context_id="context-1",
        interlocutor_id="participant-1",
    )

    assert [item.source_id for item in explanation.contributions] == [
        "referenced-value"
    ]
    assert explanation.contributions[0].evidence_refs == ("evidence-1",)
    assert "belief_reference_missing" in explanation.information_gap_codes
    assert explanation.omitted_reference_count == 1
    assert "missing-belief" not in str(explanation.public_json())
    assert "unrelated-value" not in str(explanation.public_json())


def test_cross_context_projection_omits_sources_and_renderer_fails_closed() -> None:
    selected = ActionCandidate(
        **{
            **_candidate("selected", ActionType.NO_OP).__dict__,
            "value_effects": {"private-context-value": 0.5},
        }
    )
    decision = DecisionStore().create(
        [selected],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id="context-1",
        active_goal_ids=(),
        value_revision_refs={"private-context-value": 1},
        emotion_snapshot={},
        value_evaluator=lambda _options: {"selected": {"private-context-value": 0.4}},
    )
    explanation = build_explanation(
        _main_loop(),
        decision,
        event_id="event-2",
        event_sequence=2,
        context_id="context-other",
    )

    assert explanation.compatibility == "context_filtered"
    assert explanation.contributions == ()
    assert "source_context_incompatible" in explanation.information_gap_codes
    rendered = render_natural(
        explanation,
        lambda _prompt: (
            '{"explanation_id":"other","explanation_revision":1,"visible_explanation":"<think>secret</think>"}'
        ),
    )
    assert rendered.renderer.state == RendererState.FAILED
    assert (
        rendered.renderer.visible_explanation
        == explanation.renderer.visible_explanation
    )
    serialized = str(explanation.public_json())
    assert "private-context-value" not in serialized
    assert "event-2" not in serialized
    assert explanation.created_event_sequence is None
    assert explanation.selected.candidate_id == "filtered"
    assert explanation.context_id is None


def test_renderer_can_only_reorder_offered_immutable_clauses() -> None:
    decision = DecisionStore().create(
        [_candidate("selected", ActionType.NO_OP)],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
    )
    explanation = build_explanation(
        _main_loop(), decision, event_id="event-2", event_sequence=2
    )
    offered = explanation.renderer.offered_clause_ids
    rendered = render_natural(
        explanation,
        lambda _prompt: json.dumps(
            {
                "explanation_id": explanation.explanation_id,
                "explanation_revision": explanation.revision,
                "ordered_clause_ids": list(reversed(offered)),
            }
        ),
    )
    assert rendered.renderer.state == RendererState.SUCCEEDED
    assert rendered.renderer.ordered_clause_ids == tuple(reversed(offered))
    assert "honesty" not in rendered.renderer.visible_explanation.lower()
    assert "belief" not in rendered.renderer.visible_explanation.lower()

    invented = render_natural(
        rendered,
        lambda _prompt: json.dumps(
            {
                "explanation_id": rendered.explanation_id,
                "explanation_revision": rendered.revision,
                "ordered_clause_ids": ["belief.honesty.v1"],
                "visible_explanation": "I acted from honesty.",
            }
        ),
    )
    assert invented.renderer.state == RendererState.FAILED
    assert invented.renderer.failure_code == "renderer_failed"
    assert invented.renderer.ordered_clause_ids == offered
    assert invented.renderer.visible_explanation == explanation.renderer.visible_explanation


def test_store_empty_and_legacy_restore_are_ghost_free() -> None:
    decision = DecisionStore().create(
        [_candidate("selected", ActionType.NO_OP)],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
    )
    explanation = build_explanation(
        _main_loop(), decision, event_id="event-2", event_sequence=2
    )
    store = DecisionExplanationStore()
    store.append(explanation, idempotency_key="create", input_digest="digest")
    legacy = store.to_json()
    legacy_renderer = legacy["records"][explanation.explanation_id][0]["renderer"]
    legacy_renderer.pop("offered_clause_ids")
    legacy_renderer.pop("ordered_clause_ids")
    legacy_renderer["visible_explanation"] = "invented legacy prose"

    store.restore(legacy)
    restored = store.get(explanation.explanation_id)
    assert restored.renderer.state == RendererState.DETERMINISTIC
    assert "invented" not in restored.renderer.visible_explanation
    store.restore(None)
    assert store.list_latest() == ()
    store.restore({})
    assert store.list_latest() == ()


def test_schema_recursively_rejects_private_and_unknown_fields() -> None:
    decision = DecisionStore().create(
        [_candidate("selected", ActionType.NO_OP)],
        triggering_event_id="event-1",
        triggering_event_sequence=1,
        context_id=None,
        active_goal_ids=(),
        value_revision_refs={},
        emotion_snapshot={},
    )
    explanation = build_explanation(
        _main_loop(), decision, event_id="event-2", event_sequence=2
    )
    payload = explanation.public_json()
    payload["hidden_thought"] = "private sentinel"

    with pytest.raises(ValidationError):
        PublicDecisionExplanation.model_validate(payload)

    payload = explanation.public_json()
    payload["selected"]["candidate_id"] = "/home/private/secret.txt"
    with pytest.raises(ValidationError):
        PublicDecisionExplanation.model_validate(payload)


def _candidate(identifier: str, action_type: ActionType) -> ActionCandidate:
    return ActionCandidate(
        candidate_id=identifier,
        candidate_type=action_type,
        proposed_action="opaque_action_code",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(),
        uncertainty=0.2,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )


def _main_loop() -> SimpleNamespace:
    return SimpleNamespace(
        value_system=SimpleNamespace(values={}),
        goal_manager=SimpleNamespace(goals={}),
        commitment_store=SimpleNamespace(commitments={}),
        belief_store=SimpleNamespace(records={}),
        context_registry=SimpleNamespace(
            get=lambda context_id: SimpleNamespace(
                participant_ids=("participant-1",)
            )
            if context_id == "context-1"
            else None
        ),
        identity_boundary_store=SimpleNamespace(),
        _action_execution=None,
    )
