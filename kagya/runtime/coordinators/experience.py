from dataclasses import dataclass
from typing import Any

from kagya.body import EmotionUpdate
from kagya.attention import (
    AttentionCandidate,
    AttentionFocus,
    AttentionSource,
)
from kagya.belief import (
    BeliefEvidence,
    BeliefRecord,
    EpistemicStatus,
    Proposition,
)
from kagya.cognition import (
    ActionScore,
    AppraisalResult,
    ValueEvidence,
    ValueState,
    ValueUpdateKind,
    ValueUpdateRecord,
)
from kagya.identity import (
    AutobiographicalEpisode,
    NarrativeSelf,
    OriginActor,
    OriginInputKind,
    new_identity_origin,
)
from kagya.experience import (
    ExperienceAppraisal,
    ExperienceRecord,
    ExperienceStore,
)
from kagya.motivation import (
    ACCEPTED_COMMITMENT_STATUSES,
    GoalStatus,
    MotivationDynamics,
    MotivationStatus,
)
from kagya.persona import (
    PublicSubjectSummary,
)
from kagya.relationship import RelationshipState, RelationshipStore
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.coordinators._shared import RuntimeDomainMixin
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemoryKind,
    working_memory_item,
)

from typing import Callable


@dataclass(frozen=True)
class ExperienceIntegrationResult:
    experience: ExperienceRecord
    relationships: tuple[RelationshipState, ...]
    narrative_episode: AutobiographicalEpisode | None


class ExperienceIntegrationCoordinator(RuntimeDomainMixin):
    """Integrates one experience by passing records, never stores, between domains."""

    def __init__(
        self,
        experience_store: ExperienceStore,
        relationship_store: RelationshipStore,
        narrative_self: NarrativeSelf,
        motivation_dynamics: MotivationDynamics,
        *,
        persist_experience: Callable[[], None],
        persist_narrative: Callable[[], None],
        persist_motivation: Callable[[], None],
    ) -> None:
        self._experiences = experience_store
        self._relationships = relationship_store
        self._narrative = narrative_self
        self._motivation = motivation_dynamics
        self._persist_experience = persist_experience
        self._persist_narrative = persist_narrative
        self._persist_motivation = persist_motivation

    def integrate(
        self,
        experience: ExperienceRecord,
        *,
        active_commitment_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> ExperienceIntegrationResult:
        integrated = self._experiences.integrate(experience)
        self._persist_experience()
        relationships = tuple(
            self._relationships.observe_experience(
                integrated,
                active_commitment_refs=active_commitment_refs,
                event_id=event_id,
                event_sequence=event_sequence,
            )
        )
        for relationship in relationships:
            integrated = self._experiences.link_result(
                integrated.experience_id,
                kind="relationship",
                reference=(
                    f"relationship:{relationship.relationship_id}@"
                    f"{relationship.revision}"
                ),
                evidence_refs=(f"experience:{integrated.experience_id}",),
                event_id=event_id,
                event_sequence=event_sequence,
            )
        self._persist_experience()
        narrative_episode = self._narrative.observe_experience(integrated)
        if narrative_episode is not None:
            integrated = self._experiences.link_result(
                integrated.experience_id,
                kind="narrative",
                reference=f"narrative:{narrative_episode.episode_id}",
                evidence_refs=(f"experience:{integrated.experience_id}",),
                event_id=event_id,
                event_sequence=event_sequence,
            )
            self._persist_experience()
        self._persist_narrative()
        self._motivation.observe_experience(integrated)
        self._persist_motivation()
        return ExperienceIntegrationResult(
            experience=integrated,
            relationships=relationships,
            narrative_episode=narrative_episode,
        )

    def advance_time(self, elapsed_seconds: float) -> EmotionUpdate:
        update = self.emotion_engine.advance_time(elapsed_seconds)
        self.working_memory.admit(
            working_memory_item(
                item_id="emotion:current",
                kind=WorkingMemoryKind.EMOTION,
                content=(
                    f"Current valence={update.state.valence:.3f}, "
                    f"arousal={update.state.arousal:.3f}"
                ),
                activation=1.0,
                salience=max(0.5, update.state.arousal),
                retention_reason=RetentionReason.CURRENT_EMOTION,
                source="runtime.emotion_tick",
            )
        )
        return update

    def restore_appraisal_state(self) -> None:
        self.surprisal_calculator.restore_history(
            self.persistent_state.extensions.get("appraisal_calibration")
        )

    def _persist_appraisal_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "appraisal_calibration",
            self.surprisal_calculator.export_history,
        )

    def restore_experience_state(self) -> None:
        self.persistence_coordinator.restore(
            self.persistent_state.extensions,
            "experiences",
            self.experience_store.restore,
            self._persist_experience_state,
        )

    def _persist_experience_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "experiences",
            self.experience_store.to_json,
        )

    def restore_narrative_self_state(self) -> None:
        self.narrative_self.restore(
            self.persistent_state.identity_extensions.get("narrative_self")
        )
        self._sync_self_references()
        self._persist_narrative_self_state()

    def _persist_narrative_self_state(self) -> None:
        self.persistent_state.identity_extensions["narrative_self"] = (
            self.narrative_self.to_json()
        )

    def restore_attention_state(self) -> None:
        self.persistence_coordinator.restore(
            self.persistent_state.extensions,
            "attention",
            self.attention_system.restore,
            self._persist_attention_state,
        )

    def _persist_attention_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "attention",
            self.attention_system.to_json,
        )

    def refresh_attention(self, *, compete: bool = False) -> AttentionFocus:
        """Refresh structured candidates from authoritative internal stores."""
        experience_candidate_ids: set[str] = set()
        for experience in self.experience_store.list_records():
            experience_candidate_ids.add(f"experience:{experience.experience_id}")
            memory_refs = experience.result_refs.get("memory", ())
            self.attention_system.observe(
                candidate_id=f"experience:{experience.experience_id}",
                target_ref=f"experience:{experience.experience_id}",
                source=AttentionSource.EXPERIENCE,
                source_refs=(f"experience:{experience.experience_id}",),
                working_memory_ref=memory_refs[0] if memory_refs else None,
                salience=experience.subjective_salience,
                novelty=(experience.appraisal.novelty or 0.0)
                if experience.appraisal.novelty_valid
                else 0.0,
                value_relevance=experience.self_relevance,
                arousal=experience.appraisal.arousal,
                persistence=experience.autobiographical_importance,
            )
        self.attention_system.synchronize_source(
            AttentionSource.EXPERIENCE, experience_candidate_ids
        )
        active_motivations = [
            item
            for item in self.motivation_dynamics.list_records()
            if item.status == MotivationStatus.ACTIVE
        ]
        motivation_candidate_ids = {
            f"motivation:{item.motivation_id}" for item in active_motivations
        }
        for motivation in active_motivations:
            self.attention_system.observe(
                candidate_id=f"motivation:{motivation.motivation_id}",
                target_ref=motivation.target_ref,
                source=AttentionSource.MOTIVATION,
                source_refs=tuple(
                    dict.fromkeys(
                        (
                            f"motivation:{motivation.motivation_id}",
                            *(
                                f"experience:{item}"
                                for item in motivation.related_experience_ids
                            ),
                        )
                    )
                ),
                drive=motivation.strength,
                urgency=motivation.strength * (1.0 - motivation.satiation),
                persistence=motivation.persistence,
            )
        self.attention_system.synchronize_source(
            AttentionSource.MOTIVATION, motivation_candidate_ids
        )
        active_commitments = {
            item.related_goal_id: item
            for item in self.commitment_store.list_commitments()
            if item.status in ACCEPTED_COMMITMENT_STATUSES
        }
        active_goals = [
            goal
            for goal in self.goal_manager.list_goals()
            if goal.status
            in {GoalStatus.ACTIVE, GoalStatus.CANDIDATE, GoalStatus.SUSPENDED}
        ]
        goal_candidate_ids = {f"goal:{goal.goal_id}" for goal in active_goals}
        for goal in active_goals:
            commitment = active_commitments.get(goal.goal_id)
            value_relevance = min(
                1.0,
                max((abs(item) for item in goal.value_effects.values()), default=0.0),
            )
            self.attention_system.observe(
                candidate_id=f"goal:{goal.goal_id}",
                target_ref=f"goal:{goal.goal_id}",
                source=AttentionSource.GOAL,
                source_refs=tuple(
                    ref
                    for ref in (
                        f"goal:{goal.goal_id}",
                        None
                        if goal.origin_value_id is None
                        else f"value:{goal.origin_value_id}",
                    )
                    if ref is not None
                ),
                working_memory_ref=f"goal:{goal.goal_id}",
                urgency=goal.urgency,
                value_relevance=value_relevance,
                commitment_cost=0.8 if commitment is not None else 0.0,
                persistence=max(
                    goal.priority, 0.8 if goal.status == GoalStatus.ACTIVE else 0.0
                ),
            )
        self.attention_system.synchronize_source(
            AttentionSource.GOAL, goal_candidate_ids
        )
        active_commitment_records = self.commitment_store.list_commitments()
        active_commitment_records = [
            item
            for item in active_commitment_records
            if item.status in ACCEPTED_COMMITMENT_STATUSES
        ]
        commitment_candidate_ids = {
            f"commitment:{item.commitment_id}" for item in active_commitment_records
        }
        for commitment in active_commitment_records:
            if commitment.related_goal_id is None:
                continue
            self.attention_system.observe(
                candidate_id=f"commitment:{commitment.commitment_id}",
                target_ref=f"commitment:{commitment.commitment_id}",
                source=AttentionSource.COMMITMENT,
                source_refs=(
                    f"commitment:{commitment.commitment_id}",
                    f"goal:{commitment.related_goal_id}",
                ),
                working_memory_ref=f"commitment:{commitment.commitment_id}",
                urgency=self.goal_manager.get(commitment.related_goal_id).urgency,
                commitment_cost=1.0,
                persistence=1.0,
            )
        self.attention_system.synchronize_source(
            AttentionSource.COMMITMENT, commitment_candidate_ids
        )
        event = current_agent_event()
        if compete:
            self.attention_system.compete(
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            )
        self._persist_attention_state()
        return self.attention_system.focus

    def refocus_attention(
        self,
        candidate_ids: tuple[str, ...],
        *,
        reason_code: str,
        provenance_refs: tuple[str, ...],
    ) -> AttentionFocus:
        event = current_agent_event()
        focus = self.attention_system.refocus(
            candidate_ids,
            reason_code=reason_code,
            provenance_refs=provenance_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_attention_state()
        return focus

    def defer_attention(
        self, candidate_id: str, *, reason_code: str, provenance_refs: tuple[str, ...]
    ) -> AttentionFocus:
        event = current_agent_event()
        focus = self.attention_system.defer(
            candidate_id,
            reason_code=reason_code,
            provenance_refs=provenance_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_attention_state()
        return focus

    def ignore_attention(
        self, candidate_id: str, *, reason_code: str, provenance_refs: tuple[str, ...]
    ) -> AttentionFocus:
        event = current_agent_event()
        focus = self.attention_system.ignore(
            candidate_id,
            reason_code=reason_code,
            provenance_refs=provenance_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_attention_state()
        return focus

    def resume_attention(self, candidate_id: str) -> AttentionCandidate:
        candidate = self.attention_system.resume(candidate_id)
        self._persist_attention_state()
        return candidate

    def get_experience(self, experience_id: str) -> ExperienceRecord:
        return self.experience_store.get(experience_id)

    def list_experiences(self) -> list[ExperienceRecord]:
        return self.experience_store.list_records()

    def restore_belief_state(self) -> None:
        self.persistence_coordinator.restore(
            self.persistent_state.extensions,
            "beliefs",
            self.belief_store.restore,
            self._persist_belief_state,
        )

    def _persist_belief_state(self) -> None:
        self.persistence_coordinator.persist(
            self.persistent_state.extensions,
            "beliefs",
            self.belief_store.to_json,
        )

    def propose_belief_from_experience(
        self,
        experience_id: str,
        *,
        proposition: Proposition,
        source_trust: float,
        confidence: float,
        context_scope: tuple[str, ...] = (),
        valid_from: str | None = None,
        valid_until: str | None = None,
        belief_id: str | None = None,
    ) -> BeliefRecord:
        experience = self.experience_store.get(experience_id)
        record = self.belief_store.propose(
            proposition,
            identity_origin=experience.identity_origin,
            evidence=(
                BeliefEvidence(
                    reference=f"experience:{experience_id}",
                    evidence_type="external_claim",
                    source_trust=source_trust,
                    observed_at=experience.created_at,
                ),
            ),
            confidence=confidence,
            context_scope=context_scope,
            valid_from=valid_from,
            valid_until=valid_until,
            belief_id=belief_id,
        )
        self.link_experience_result(
            experience_id,
            kind="belief",
            reference=f"belief:{record.belief_id}",
            evidence_refs=(f"experience:{experience_id}",),
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return record

    def resolve_belief(
        self,
        belief_id: str,
        *,
        accept: bool,
        confidence: float,
        epistemic_status: EpistemicStatus,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> BeliefRecord:
        event = current_agent_event()
        record = self.belief_store.resolve(
            belief_id,
            accept=accept,
            confidence=confidence,
            epistemic_status=epistemic_status,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return record

    def retract_belief(
        self, belief_id: str, *, reason_code: str, evidence_refs: tuple[str, ...]
    ) -> BeliefRecord:
        event = current_agent_event()
        record = self.belief_store.retract(
            belief_id,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return record

    def supersede_belief(
        self,
        old_belief_id: str,
        new_belief_id: str,
        *,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> tuple[BeliefRecord, BeliefRecord]:
        event = current_agent_event()
        records = self.belief_store.supersede(
            old_belief_id,
            new_belief_id,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return records

    def expire_beliefs(self) -> list[BeliefRecord]:
        records = self.belief_store.expire()
        self._persist_belief_state()
        self._sync_belief_working_memory(None)
        return records

    def list_beliefs(self) -> list[BeliefRecord]:
        return self.belief_store.list_records()

    def _sync_belief_working_memory(self, context_id: str | None) -> None:
        for item in tuple(self.working_memory.items):
            if item.kind == WorkingMemoryKind.BELIEF:
                self.working_memory.forget(item.item_id)
        for belief in self.belief_store.active(context_id=context_id):
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"belief:{belief.belief_id}",
                    kind=WorkingMemoryKind.BELIEF,
                    content=(
                        f"Adopted belief ({belief.epistemic_status.value}, "
                        f"confidence={belief.confidence:.3f}): "
                        f"{belief.proposition.normalized}"
                    ),
                    activation=0.8,
                    salience=belief.confidence,
                    retention_reason=RetentionReason.ESTABLISHED_BELIEF,
                    source="runtime.belief_store",
                )
            )

    def reassess_experience(
        self,
        experience_id: str,
        *,
        appraisal: ExperienceAppraisal,
        reason_code: str,
        evidence_refs: tuple[str, ...],
    ) -> ExperienceRecord:
        event = current_agent_event()
        record = self.experience_store.revise_appraisal(
            experience_id,
            appraisal=appraisal,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        for reference in record.result_refs.get("memory", ()):
            if reference.startswith("episode:"):
                self.memory_system.link_experience(
                    reference.removeprefix("episode:"),
                    experience_id=record.experience_id,
                    subjective_salience=record.subjective_salience,
                    autobiographical_importance=record.autobiographical_importance,
                )
        self._persist_experience_state()
        return record

    def link_experience_result(
        self,
        experience_id: str,
        *,
        kind: str,
        reference: str,
        evidence_refs: tuple[str, ...],
    ) -> ExperienceRecord:
        event = current_agent_event()
        record = self.experience_store.link_result(
            experience_id,
            kind=kind,
            reference=reference,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.narrative_self.observe_experience(record)
        self._persist_experience_state()
        self._persist_narrative_self_state()
        return record

    def restore_value_state(self) -> None:
        self.value_system.restore(self.persistent_state.values)
        self._persist_value_state()

    def _persist_value_state(self) -> None:
        self.persistent_state.values = self.value_system.to_json()

    def apply_value_impacts(
        self,
        appraisal: AppraisalResult,
        impacts: dict[str, float],
        *,
        kind: ValueUpdateKind,
        memory_ids: tuple[str, ...] = (),
        source: str = "runtime",
        proposal_id: str | None = None,
        origin_actor: OriginActor = OriginActor.SELF,
        origin_input_kind: OriginInputKind = OriginInputKind.INTERNAL_STATE,
        experience_ids: tuple[str, ...] = (),
        decision_id: str | None = None,
        context_id: str | None = None,
    ) -> list[ValueUpdateRecord]:
        event = current_agent_event()
        evidence = ValueEvidence(
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            memory_ids=memory_ids,
            source=source,
            experience_ids=experience_ids,
            decision_id=decision_id,
            context_id=context_id,
            identity_origin=new_identity_origin(
                origin_actor,
                origin_input_kind,
                source_ref=source,
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            ),
        )
        proposals = self.value_system.proposals_from_appraisal(
            appraisal,
            impacts,
            kind=kind,
            evidence=evidence,
            proposal_id=proposal_id,
        )
        records = self.value_system.apply(proposals)
        self._persist_value_state()
        return records

    def apply_value_evidence_from_experience(
        self,
        experience_id: str,
        impacts: dict[str, float],
        *,
        proposal_id: str | None = None,
    ) -> list[ValueUpdateRecord]:
        experience = self.experience_store.get(experience_id)
        proposals = self.value_system.proposals_from_experience(
            experience, impacts, proposal_id=proposal_id
        )
        records = self.value_system.apply(proposals)
        for record in records:
            self.link_experience_result(
                experience_id,
                kind="value",
                reference=f"value:{record.value_id}@{record.after['revision']}",
                evidence_refs=record.evidence_ids,
            )
        self._sync_self_references()
        self._persist_self_model_state()
        self._persist_value_state()
        return records

    def evaluate_value_options(
        self, options: dict[str, dict[str, float]], *, context_id: str | None = None
    ) -> list[ActionScore]:
        return self.value_system.evaluate(options, context_id=context_id)

    def freeze_value(self, value_id: str, *, frozen: bool) -> ValueState:
        state = self.value_system.freeze(value_id, frozen=frozen)
        self._persist_value_state()
        return state

    def rollback_value(self, value_id: str, *, target_revision: int) -> ValueState:
        state = self.value_system.rollback(value_id, target_revision=target_revision)
        self._persist_value_state()
        return state

    def reset_values(
        self, value_ids: tuple[str, ...] | None = None
    ) -> list[ValueState]:
        states = self.value_system.reset(value_ids)
        self._persist_value_state()
        return states

    def correct_relationship(
        self,
        relationship_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
        **updates: Any,
    ) -> RelationshipState:
        event = current_agent_event()
        return self.relationship_store.correct(
            relationship_id,
            reason=reason,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            **updates,
        )

    def attach_relationship_alias(
        self,
        relationship_id: str,
        interlocutor_key: str,
        *,
        evidence_refs: tuple[str, ...],
    ) -> RelationshipState:
        event = current_agent_event()
        return self.relationship_store.attach_alias(
            relationship_id,
            interlocutor_key,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )

    def split_relationship_alias(
        self,
        relationship_id: str,
        interlocutor_key: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
    ) -> RelationshipState:
        event = current_agent_event()
        return self.relationship_store.split_alias(
            relationship_id,
            interlocutor_key,
            reason=reason,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )

    def _relationship_for_target(
        self, target: dict[str, Any]
    ) -> RelationshipState | None:
        key = target.get("interlocutor_key")
        return (
            self.relationship_store.for_interlocutor(key)
            if isinstance(key, str)
            else None
        )

    def _public_subject_summary(
        self, context_id: str, interlocutor_keys: tuple[str, ...]
    ) -> PublicSubjectSummary:
        relationships: list[str] = []
        for key in interlocutor_keys:
            state = self.relationship_store.for_interlocutor(key)
            if state is None:
                continue
            relationships.append(
                f"trust={state.axes.trust:.3f}; familiarity={state.axes.familiarity:.3f}; "
                f"closeness={state.axes.closeness:.3f}; caution={state.axes.caution:.3f}; "
                f"reciprocity={state.reciprocity:.3f}; uncertainty={state.uncertainty:.3f}"
            )
        quality = self._current_cognitive_quality()
        return PublicSubjectSummary(
            values=tuple(
                f"{value.name}; importance={value.weight:.3f}; "
                f"confidence={value.confidence:.3f}; protectedness={value.protectedness:.3f}"
                for value in self.value_system.list_values()[:5]
            ),
            goals=tuple(
                f"{goal.description}; status={goal.status.value}; "
                f"priority={goal.priority:.3f}; confidence={goal.confidence:.3f}"
                for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)[:5]
            ),
            commitments=tuple(
                f"{commitment.description}; status={commitment.status.value}; "
                f"fulfillability={commitment.fulfillability.value}"
                for commitment in self.commitment_store.list_commitments()
                if commitment.status in ACCEPTED_COMMITMENT_STATUSES
            )[:5],
            relationships=tuple(relationships[:5]),
            beliefs=tuple(
                f"{belief.proposition.normalized}; status={belief.epistemic_status.value}; "
                f"confidence={belief.confidence:.3f}"
                for belief in self.belief_store.active(context_id=context_id)[:5]
            ),
            metacognition=(
                f"estimated_quality={quality.estimated_quality:.3f}; "
                f"cognitive_load={quality.cognitive_load:.3f}; "
                f"attention_saturation={quality.attention_saturation:.3f}",
            ),
        )
