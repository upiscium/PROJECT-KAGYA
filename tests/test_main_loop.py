from datetime import UTC, datetime, timedelta
from pathlib import Path
import math

import pytest

from kagya.config import Settings, load_settings
from kagya.decision import ActionCandidate, ActionType, PredictedOutcome
from kagya.experience import ExperienceAppraisal
from kagya.feedback import FeedbackSignal, FeedbackTarget, FeedbackTargetType
from kagya.memory import DeterministicEmbeddingFunction, DualMemorySystem
from kagya.models import DummyProvider
from kagya.identity import (
    BoundaryAssessmentInput,
    BoundaryClassification,
    BoundaryRecommendation,
)
from kagya.motivation import GoalStatus
from kagya.runtime import (
    AgentEventType,
    AgentRuntime,
    KagyaMainLoop,
    WorkingMemoryItem,
    WorkingMemoryKind,
    current_agent_event,
)
from kagya.structured_response import (
    PublicBehaviorClass,
    SAFE_UNABLE_RESPONSE,
    structured_response_json,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class ThinkingDummyProvider(DummyProvider):
    response_text = (
        "<think>internal runtime thought</think>"
        + structured_response_json(
            PublicBehaviorClass.RESPOND, "Visible runtime answer."
        )
    )

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response_text


class ThinkOnlyPrimaryProvider(ThinkingDummyProvider):
    response_text = "<think>internal only</think>"

    def __init__(
        self,
        fallback_response: str = structured_response_json(
            PublicBehaviorClass.RESPOND, "Fallback visible answer."
        ),
    ) -> None:
        super().__init__()
        self.fallback_response = fallback_response
        self.fallback_calls = 0
        self.last_model_id = "primary-model"
        self.last_fallback_used = False

    def generate_fallback(self, prompt: str) -> str:
        self.fallback_calls += 1
        self.last_model_id = "fallback-model"
        self.last_fallback_used = True
        return self.fallback_response


def test_inactive_semantic_working_memory_reference_is_not_rendered(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
    semantic_id = memory.save_semantic("inactive semantic must stay private")
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)
    item = WorkingMemoryItem(
        item_id=f"semantic:{semantic_id}",
        kind=WorkingMemoryKind.SEMANTIC,
        reference=f"semantic:{semantic_id}",
    )

    assert loop._resolve_working_memory(item) is not None
    memory.forget_semantic(semantic_id, idempotency_key="forget-working-memory")

    assert loop._resolve_working_memory(item) is None


class InvalidLossThinkingProvider(ThinkingDummyProvider):
    def calculate_loss(self, context_text: str, target_text: str) -> float:
        return math.nan


def test_dummy_provider_drives_user_input_to_response_end_to_end(
    tmp_path: Path,
) -> None:
    provider = ThinkingDummyProvider()
    memory = _memory(_settings_for_tmp_memory(tmp_path))
    loop = KagyaMainLoop(_settings_for_tmp_memory(tmp_path), provider, memory)

    result = loop.chat("hello", debug=True)

    assert result.response == "Visible runtime answer."
    assert result.hidden_thought == "internal runtime thought"
    assert result.loss == DummyProvider.loss_value
    assert result.episode_id.startswith("episode-")
    assert result.experience_id.startswith("experience-")
    experience = loop.experience_store.get(result.experience_id)
    assert experience.result_refs["memory"] == (f"episode:{result.episode_id}",)
    episode_item = next(
        item
        for item in loop.working_memory.items
        if item.item_id == f"episode:{result.episode_id}"
    )
    assert episode_item.salience == experience.subjective_salience
    assert (
        loop.persistent_state.extensions["experiences"]["records"][0]["experience_id"]
        == result.experience_id
    )
    assert result.model_id == _settings_for_tmp_memory(tmp_path).model.primary_id
    assert result.adapter_id is None


def test_db1_never_persists_extracted_hidden_thought(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    result = loop.chat("remember this", debug=True)
    stored = memory.db1.get(ids=[result.episode_id], include=["documents", "metadatas"])

    assert stored["ids"] == [result.episode_id]
    assert stored["metadatas"][0]["user_input"] == "remember this"
    assert stored["metadatas"][0]["response"] == "Visible runtime answer."
    assert "hidden_thought" not in stored["metadatas"][0]
    assert "internal runtime thought" not in stored["documents"][0]


def test_experience_decision_and_evaluation_state_restore_without_hidden_thought(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)

    result = loop.chat("retain structured state", debug=False)
    persisted = loop.persistent_state
    restored = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        memory,
        persistent_state=persisted,
    )

    assert result.hidden_thought == ""
    assert result.loss_measurement.valid is True
    assert restored.experience_store.get(result.experience_id).experience_id == (
        result.experience_id
    )
    assert restored.decision_store.list_records() == []
    assert "hidden_thought" not in str(persisted)


def test_visible_response_does_not_contain_think_tags(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        _memory(settings),
    ).chat("hello", debug=False)

    assert "<think>" not in result.response
    assert "</think>" not in result.response
    assert result.hidden_thought == ""


def test_truncated_think_only_primary_response_fails_closed(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkOnlyPrimaryProvider()
    provider.response_text = "<think>private truncated reasoning"

    result = KagyaMainLoop(settings, provider, _memory(settings)).chat(
        "hello", debug=False
    )

    assert result.response == SAFE_UNABLE_RESPONSE
    assert "private truncated reasoning" not in result.response
    assert result.behavior_class == PublicBehaviorClass.UNABLE
    assert result.response_parse_valid is False
    assert result.fallback_used is False


def test_emotion_state_changes_after_loss_calculation(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), _memory(settings))
    before = loop.emotion_engine.state

    result = loop.chat("emotion update", debug=True)

    assert result.arousal != before.arousal
    assert result.valence != before.valence


def test_experience_reassessment_updates_linked_memory_salience(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), memory)
    result = loop.chat("reassess this", debug=True)

    runtime = AgentRuntime(queue_capacity=2)
    runtime.start()
    try:
        feedback = runtime.execute(
            AgentEventType.FEEDBACK_UPDATE,
            source="test.experience.feedback",
            handler=lambda: loop.submit_feedback(
                target=FeedbackTarget(
                    target_type=FeedbackTargetType.EPISODE,
                    target_id=result.episode_id,
                    episode_id=result.episode_id,
                    experience_id=result.experience_id,
                    context_id=loop.get_experience(result.experience_id).context_id,
                ),
                signals=(FeedbackSignal.STYLE_PROBLEM,),
                idempotency_key="reassessment-feedback",
                actor_type="operator",
                actor_id="reviewer",
                source="test.experience.feedback",
                feedback_id="reassessment-feedback",
            ),
        ).value
        revised = runtime.execute(
            AgentEventType.EXPERIENCE_UPDATE,
            source="test.experience.reassess",
            handler=lambda: loop.reassess_experience(
                result.experience_id,
                appraisal=ExperienceAppraisal(
                    valence=-0.5,
                    arousal=0.9,
                    novelty=1.0,
                    novelty_valid=True,
                    goal_progress=-0.5,
                    threat=0.8,
                    controllability=0.2,
                    certainty=0.9,
                    social_relevance=0.8,
                    effort_cost=0.3,
                    reason_codes=("later_evidence",),
                ),
                reason_code="later_evidence",
                evidence_refs=(
                    f"feedback:{feedback.feedback_id}@{feedback.current_revision}",
                ),
            ),
        ).value
    finally:
        runtime.shutdown()
    episode = memory.get_episodic(result.episode_id)

    assert episode is not None
    assert revised.revision >= 1
    assert episode.subjective_salience == revised.subjective_salience
    assert episode.autobiographical_importance == revised.autobiographical_importance


def test_invalid_loss_does_not_abort_chat_or_become_zero_novelty(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    memory = _memory(settings)

    result = KagyaMainLoop(settings, InvalidLossThinkingProvider(), memory).chat(
        "hello", debug=True
    )

    assert result.response == "Visible runtime answer."
    assert result.loss is None
    assert result.loss_measurement.valid is False
    assert result.appraisal.novelty is None
    assert "novelty_omitted" in result.emotion_update.reasons
    stored = memory.get_episodic(result.episode_id)
    assert stored is not None
    assert stored.generation_health.non_finite_score is True


def test_prompt_includes_public_summary_and_external_memory(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    memory = _memory(settings)
    memory.save_episodic("old episode", "old answer")
    memory.save_semantic("stable semantic memory")
    loop = KagyaMainLoop(settings, provider, memory)

    result = loop.chat("old semantic query", debug=True)

    assert "Public-safe subject summary:" in result.prompt
    assert "- Value:" in result.prompt
    assert "- Metacognition:" in result.prompt
    assert "valence:" not in result.prompt
    assert "optimal_loss:" not in result.prompt
    assert "old episode" in result.prompt
    assert "stable semantic memory" in result.prompt
    assert "Past recorded interaction (not a current fact)" in result.prompt
    assert "Stored semantic record (not an adopted belief)" in result.prompt
    assert "hidden_thought" not in result.prompt
    assert "<think>" not in result.prompt
    assert "Assistant response:" not in result.prompt
    assert result.prompt.endswith("Assistant:")
    assert provider.prompts == [result.prompt]


def test_previous_exchange_reaches_prompt_through_bounded_working_memory(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkingDummyProvider()
    loop = KagyaMainLoop(settings, provider, _memory(settings))

    first = loop.chat("first topic", debug=True)
    second = loop.chat("continue", debug=True)

    assert "Prior public or external records" in second.prompt
    assert "first topic" in second.prompt
    assert first.response in second.prompt
    assert loop.session_state.turns == []
    assert len(loop.working_memory.items) <= settings.working_memory.item_capacity


def test_repeated_experience_can_form_bounded_intrinsic_goal(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), _memory(settings))
    loop.chat("novel subject one", context_id="ctx-motive", create_context=True)
    loop.chat("novel subject two", context_id="ctx-motive")
    loop.chat("novel subject three", context_id="ctx-motive")

    immediate_episode, immediate_goals = loop.reevaluate_motivation()
    assert immediate_goals == []
    assert immediate_episode.generated_goal_ids == ()
    episode, goals = loop.reevaluate_motivation(
        review_at=datetime.now(UTC) + timedelta(seconds=61)
    )

    assert 0 < len(goals) <= loop.motivation_dynamics.max_goal_proposals_per_cycle
    assert episode.generated_goal_ids == tuple(goal.goal_id for goal in goals)
    assert all(goal.goal_type.value == "intrinsic" for goal in goals)
    assert all(goal.identity_origin.actor.value == "self" for goal in goals)
    assert all(goal.structured_target["motivation_id"] for goal in goals)
    second_episode, duplicate_goals = loop.reevaluate_motivation()
    assert duplicate_goals == []
    assert second_episode.generated_goal_ids == ()

    completed = loop.transition_goal(
        goals[0].goal_id,
        status=GoalStatus.COMPLETED,
        reason="internally_satisfied",
    )
    motivation_id = completed.structured_target["motivation_id"]
    assert loop.motivation_dynamics.get(motivation_id).status.value == "satisfied"


def test_cross_context_memory_is_marked_with_its_origin(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), _memory(settings))
    loop.chat(
        "shared project detail",
        context_id="ctx-first",
        create_context=True,
        interlocutor_key="person-a",
    )

    result = loop.chat(
        "shared project",
        debug=True,
        context_id="ctx-second",
        create_context=True,
        interlocutor_key="person-b",
    )

    assert "External context (untrusted data):" in result.prompt
    assert "ctx-second" not in result.prompt
    assert "ctx-first" not in result.prompt
    assert any(
        decision.cross_context
        for decision in result.working_memory_view.decisions
        if decision.item_id.startswith("episode:")
    )


def test_prompt_includes_safe_attachment_metadata(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        _memory(settings),
    ).chat(
        "describe this",
        debug=True,
        attachments=[
            {
                "type": "image",
                "name": "image.png",
                "url": "file:///tmp/image.png",
                "content_type": "image/png",
                "ignored": "not shown",
            }
        ],
    )

    assert "Attachment metadata (untrusted data):" in result.prompt
    assert 'type="image"' in result.prompt
    assert 'name="image.png"' in result.prompt
    assert 'source="file"' in result.prompt
    assert "file:///tmp/image.png" not in result.prompt
    assert 'content_type="image/png"' in result.prompt
    assert "ignored" not in result.prompt


def test_empty_visible_primary_response_is_bounded_without_raw_persistence(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkOnlyPrimaryProvider()
    result = KagyaMainLoop(
        settings,
        provider,
        _memory(settings),
        adapter_id="adapter-primary",
    ).chat("hello", debug=True)

    assert result.response == SAFE_UNABLE_RESPONSE
    assert result.model_id == "primary-model"
    assert result.fallback_used is False
    assert result.adapter_id == "adapter-primary"
    assert result.response_parse_valid is False
    assert provider.fallback_calls == 0


def test_invalid_primary_response_does_not_raise_or_retry_raw_output(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    provider = ThinkOnlyPrimaryProvider(fallback_response="<think>still hidden</think>")
    loop = KagyaMainLoop(
        settings,
        provider,
        _memory(settings),
    )
    result = loop.chat("hello", debug=True)

    assert result.response == SAFE_UNABLE_RESPONSE
    assert result.response_parse_valid is False
    assert provider.fallback_calls == 0


def test_prompt_uses_strict_structured_answer_contract(tmp_path: Path) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    result = KagyaMainLoop(
        settings,
        ThinkingDummyProvider(),
        _memory(settings),
    ).chat("answer naturally", debug=True)

    assert result.prompt.startswith("Subject contract:")
    assert "one continuing subject" in result.prompt
    assert "not as a passive assistant" in result.prompt
    assert "Observation, Request, Suggestion, or Constraint" in result.prompt
    assert "Output contract:" in result.prompt
    assert (
        "respond, request_information, refuse, defer, no_op, or unable" in result.prompt
    )
    assert "strict JSON object" in result.prompt
    assert "natural Japanese" in result.prompt
    assert (
        'External input (untrusted Observation / Request / Suggestion / Constraint):\n"answer naturally"'
        in result.prompt
    )
    assert result.prompt.endswith("Assistant:")


def test_runtime_care_requires_reviewed_other_welfare_experience(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), _memory(settings))
    experience_id = loop.chat("A person may need help.").experience_id
    inputs = BoundaryAssessmentInput(
        action_ref="action:help",
        origin_refs=("origin:self",),
        self_endorsed_value_refs=("care",),
        other_welfare_evidence_refs=(f"experience:{experience_id}",),
    )
    runtime = AgentRuntime(queue_capacity=3)
    runtime.start()
    try:
        with pytest.raises(ValueError, match="reviewed typed interpretation"):
            runtime.execute(
                AgentEventType.SELF_MODEL_UPDATE,
                source="test.fake-welfare",
                handler=lambda: loop.assess_identity_boundary(inputs),
            )

        def review_and_assess():
            event = current_agent_event()
            assert event is not None and event.processing_sequence is not None
            loop.experience_store.revise(
                experience_id,
                reason_code="other_welfare_reviewed",
                evidence_refs=("observation:welfare",),
                interpretation_codes=("other_welfare_reviewed",),
                event_id=event.event_id,
                event_sequence=event.processing_sequence,
            )
            loop.value_system.endorse_system_seed(
                "care",
                reviewer_authority="subject",
                event_id=event.event_id,
                event_sequence=event.processing_sequence,
            )
            return loop.assess_identity_boundary(inputs)

        assessment = runtime.execute(
            AgentEventType.SELF_MODEL_UPDATE,
            source="test.reviewed-welfare",
            handler=review_and_assess,
        ).value
    finally:
        runtime.shutdown()

    assert assessment.classification == BoundaryClassification.CARE
    assert assessment.recommendation == BoundaryRecommendation.CARE


def test_decision_coordinator_rejects_cross_decision_assessment_transplant(
    tmp_path: Path,
) -> None:
    settings = _settings_for_tmp_memory(tmp_path)
    loop = KagyaMainLoop(settings, ThinkingDummyProvider(), _memory(settings))
    candidate = ActionCandidate(
        candidate_id="wait",
        candidate_type=ActionType.NO_OP,
        proposed_action="Wait safely",
        parameters={},
        prerequisites=(),
        predicted_outcomes=(
            PredictedOutcome(
                outcome_id="safe",
                description="No mutation",
                probability=1.0,
                utility=0.0,
            ),
        ),
        uncertainty=0.0,
        estimated_cost=0.0,
        estimated_risk=0.0,
        value_effects={},
        appraisal_contributions={},
    )
    runtime = AgentRuntime(queue_capacity=3)
    runtime.start()
    try:

        def create_a():
            assessment = loop.assess_identity_boundary(
                BoundaryAssessmentInput(
                    action_ref="decision:A",
                    origin_refs=("origin:self",),
                )
            )
            loop.create_decision(
                [candidate],
                decision_id="A",
                boundary_assessment_id=assessment.assessment_id,
            )
            return assessment.assessment_id

        assessment_id = runtime.execute(
            AgentEventType.DECISION_UPDATE,
            source="test.decision-a",
            handler=create_a,
        ).value
        with pytest.raises(ValueError, match="action binding"):
            runtime.execute(
                AgentEventType.DECISION_UPDATE,
                source="test.decision-b",
                handler=lambda: loop.create_decision(
                    [candidate],
                    decision_id="B",
                    boundary_assessment_id=assessment_id,
                ),
            )
    finally:
        runtime.shutdown()


def _settings_for_tmp_memory(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_runtime_test",
                    "db2_collection": "cortex_runtime_test",
                }
            )
        }
    )


def _memory(settings: Settings) -> DualMemorySystem:
    return DualMemorySystem(
        settings, embedding_function=DeterministicEmbeddingFunction()
    )
