from datetime import UTC, datetime
import json
from typing import Any
from uuid import uuid4

from kagya.cognition import (
    AppraisalResult,
)
from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionStatus,
    PredictedOutcome,
)
from kagya.identity import (
    EndorsementStatus,
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    new_identity_origin,
)
from kagya.motivation import (
    ACCEPTED_COMMITMENT_STATUSES,
    Commitment,
    CommitmentFulfillability,
    CommitmentLifecycleAction,
    CommitmentStatus,
    Goal,
    GoalDecision,
    GoalDecisionAction,
    GoalDecisionInput,
    GoalManager,
    GoalStatus,
    GoalType,
    IntrinsicGoalAction,
    IntrinsicGoalDeliberation,
    IntrinsicGoalStatus,
    MotivationDynamics,
    MotivationEpisode,
    MotivationKind,
    MotivationRecord,
    MotivationSource,
    MotivationStatus,
)
from kagya.planning import (
    PLAN_STATE_KEY,
    Plan,
    PlanCandidate,
    PlanCondition,
    PlanStatus,
    ExpectedObservation,
    StepDefinition,
    VerificationPolicy,
)
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemoryKind,
    working_memory_item,
)
from kagya.runtime.coordinators._shared import (
    RuntimeDomainMixin,
    string_values,
    unit_target,
)

from typing import Callable


class MotivationGoalCoordinator(RuntimeDomainMixin):
    def __init__(
        self,
        goals: GoalManager,
        motivations: MotivationDynamics,
        *,
        persist: Callable[[], None],
    ) -> None:
        self._goals = goals
        self._motivations = motivations
        self._persist = persist

    def resolve_goal_motivation(self, goal: Goal, status: GoalStatus) -> None:
        if status in {GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED}:
            self._motivations.resolve_goal(
                goal.goal_id, success=status == GoalStatus.COMPLETED
            )
        self._persist()

    def list_active_goal_ids(self) -> tuple[str, ...]:
        return tuple(goal.goal_id for goal in self._goals.list_goals(GoalStatus.ACTIVE))

    def restore_motivation_state(self) -> None:
        decisions = self.persistent_state.motivation_extensions.get(
            "goal_decisions", []
        )
        intrinsic_deliberations = self.persistent_state.motivation_extensions.get(
            "intrinsic_goal_deliberations", []
        )
        self.goal_manager.restore(
            self.persistent_state.active_goals,
            decisions if isinstance(decisions, list) else [],
            intrinsic_deliberations
            if isinstance(intrinsic_deliberations, list)
            else [],
        )
        self.commitment_store.restore(self.persistent_state.commitments)
        self.plan_store.restore(
            self.persistent_state.motivation_extensions.get(PLAN_STATE_KEY)
        )
        self.motivation_dynamics.restore(
            self.persistent_state.motivation_extensions.get("dynamics")
        )
        self._persist_motivation_state()

    def _persist_motivation_state(self) -> None:
        self.persistent_state.active_goals = self.goal_manager.goals_json()
        self.persistent_state.commitments = self.commitment_store.to_json()
        self.persistent_state.motivation_extensions["goal_decisions"] = (
            self.goal_manager.decisions_json()
        )
        self.persistent_state.motivation_extensions["intrinsic_goal_deliberations"] = (
            self.goal_manager.intrinsic_deliberations_json()
        )
        self.persistent_state.motivation_extensions["dynamics"] = (
            self.motivation_dynamics.to_json()
        )
        self.persistent_state.motivation_extensions[PLAN_STATE_KEY] = (
            self.plan_store.to_json()
        )

    def derive_structured_motivations(self) -> list[MotivationRecord]:
        """Translate existing structured state into evidence-bound motives."""
        observed: list[MotivationRecord] = []
        homeostatic = self.persistent_state.motivation_extensions.get(
            "homeostatic_signal"
        )
        if isinstance(homeostatic, dict):
            revision = homeostatic.get("revision")
            observed_at = homeostatic.get("observed_at")
            tension = homeostatic.get("tension")
            valence = homeostatic.get("valence")
            arousal = homeostatic.get("arousal")
            if (
                isinstance(revision, int)
                and isinstance(observed_at, str)
                and isinstance(tension, (int, float))
                and isinstance(valence, (int, float))
                and isinstance(arousal, (int, float))
            ):
                source_ref = f"homeostatic-state@{revision}"
                if tension >= 0.6:
                    observed.append(
                        self.motivation_dynamics.observe_structured_signal(
                            MotivationKind.DRIVE,
                            MotivationSource.HOMEOSTATIC,
                            "homeostasis:emotion",
                            signal=float(tension),
                            uncertainty=0.1,
                            source_refs=(source_ref,),
                            observed_at=observed_at,
                            measurements=(
                                ("tension", float(tension)),
                                ("negative_valence", max(0.0, -float(valence))),
                                ("arousal", float(arousal)),
                            ),
                        )
                    )
                else:
                    self.motivation_dynamics.retire_structured_signal(
                        MotivationSource.HOMEOSTATIC,
                        "homeostasis:emotion",
                        source_state_ref=source_ref,
                    )
        for episode in self.narrative_self.episodes.values():
            if episode.unresolved_tension < 0.4:
                continue
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.CLOSURE,
                    f"narrative:{episode.episode_id}",
                    signal=episode.unresolved_tension,
                    uncertainty=0.3,
                    source_refs=(
                        f"narrative:{episode.episode_id}",
                        *(f"experience:{item}" for item in episode.experience_ids),
                    ),
                    observed_at=episode.created_at,
                    measurements=(("tension", episode.unresolved_tension),),
                )
            )
        for self_conflict in self.narrative_self.conflicts.values():
            if self_conflict.resolved_at is not None:
                continue
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.DELIBERATION,
                    f"self-conflict:{self_conflict.conflict_id}",
                    signal=0.7,
                    uncertainty=0.6,
                    source_refs=(
                        f"self-conflict:{self_conflict.conflict_id}",
                        *self_conflict.evidence_refs,
                    ),
                    observed_at=self_conflict.created_at,
                    measurements=(("tension", 0.7),),
                )
            )
        for projection in self.narrative_self.future_self.values():
            if projection.gap <= 0.0:
                continue
            motivation = self.motivation_dynamics.observe_structured_signal(
                MotivationKind.DESIRE,
                MotivationSource.LEARNING,
                f"future-self:{projection.projection_id}",
                signal=projection.gap,
                uncertainty=max(0.0, 1.0 - projection.current_level),
                source_refs=(
                    f"future-self:{projection.projection_id}@{projection.updated_at}",
                    *projection.evidence_refs,
                ),
                observed_at=projection.updated_at,
                measurements=(("gap", projection.gap),),
            )
            self.narrative_self.link_future_motivation(
                projection.projection_id, motivation.motivation_id
            )
            observed.append(motivation)
        for relationship in self.relationship_store.list_relationships():
            evidence_refs = tuple(
                dict.fromkeys(
                    (
                        f"relationship:{relationship.relationship_id}@{relationship.revision}",
                        *relationship.unresolved_matter_refs,
                        *relationship.conflict_refs,
                        *relationship.commitment_refs,
                    )
                )
            )
            if len(evidence_refs) == 1:
                continue
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.SOCIAL,
                    f"relationship:{relationship.relationship_id}",
                    signal=max(0.5, relationship.axes.closeness),
                    uncertainty=relationship.uncertainty,
                    source_refs=evidence_refs,
                )
            )
        for tradeoff in self.value_system.tradeoffs:
            if not tradeoff.conflict_names:
                continue
            values = [self.value_system.get(item) for item in tradeoff.value_ids]
            evidence_refs = tuple(
                dict.fromkeys(
                    (
                        f"value-tradeoff:{tradeoff.tradeoff_id}",
                        *(f"value-conflict:{item}" for item in tradeoff.conflict_names),
                        *(
                            f"value:{value.value_id}@{value.revision}"
                            for value in values
                        ),
                    )
                )
            )
            motives = [
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DESIRE,
                    MotivationSource.DELIBERATION,
                    f"value:{value.value_id}",
                    signal=min(
                        1.0,
                        max(
                            value.weight,
                            abs(
                                tradeoff.contribution_by_value.get(value.value_id, 0.0)
                            ),
                        ),
                    ),
                    uncertainty=1.0 - value.confidence,
                    source_refs=evidence_refs,
                    value_ids=(value.value_id,),
                )
                for value in values
            ]
            for index, left_motive in enumerate(motives):
                for right_motive in motives[index + 1 :]:
                    self.motivation_dynamics.register_conflict(
                        left_motive.motivation_id, right_motive.motivation_id
                    )
            observed.extend(motives)
        for limitation in self.self_model.state.known_limitations.values():
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.DRIVE,
                    MotivationSource.LEARNING,
                    f"limitation:{limitation.limitation_id}",
                    signal=limitation.confidence,
                    uncertainty=1.0 - limitation.confidence,
                    source_refs=(
                        (
                            f"limitation:{limitation.limitation_id}@"
                            f"{self.self_model.state.revision}"
                        ),
                        *limitation.evidence_refs,
                    ),
                )
            )
        for uncertainty in self.self_model.state.epistemic_uncertainties.values():
            observed.append(
                self.motivation_dynamics.observe_structured_signal(
                    MotivationKind.INTEREST,
                    MotivationSource.LEARNING,
                    f"uncertainty:{uncertainty.uncertainty_id}",
                    signal=max(0.4, 1.0 - uncertainty.confidence),
                    uncertainty=uncertainty.confidence,
                    source_refs=(
                        (
                            f"uncertainty:{uncertainty.uncertainty_id}@"
                            f"{self.self_model.state.revision}"
                        ),
                        *uncertainty.evidence_refs,
                    ),
                )
            )
        self._persist_narrative_self_state()
        self._persist_motivation_state()
        return observed

    def record_homeostatic_state(
        self, *, valence: float, arousal: float, observed_at: datetime | None = None
    ) -> None:
        if not -1.0 <= valence <= 1.0 or not 0.0 <= arousal <= 1.0:
            raise ValueError("homeostatic state must be bounded")
        current = self.persistent_state.motivation_extensions.get("homeostatic_signal")
        revision = (
            int(current.get("revision", 0)) + 1 if isinstance(current, dict) else 0
        )
        self.persistent_state.motivation_extensions["homeostatic_signal"] = {
            "revision": revision,
            "observed_at": (observed_at or datetime.now(UTC)).isoformat(),
            "valence": valence,
            "arousal": arousal,
            "tension": min(1.0, max(arousal, max(0.0, -valence))),
            "source": "persisted_emotion_state",
        }
        self._persist_motivation_state()

    def reevaluate_motivation(
        self,
        *,
        max_goal_proposals: int | None = None,
        review_at: datetime | None = None,
    ) -> tuple[MotivationEpisode, list[Goal]]:
        event = current_agent_event()
        self.derive_structured_motivations()
        candidates, held_ids = self.motivation_dynamics.goal_candidates(
            max_goal_proposals, review_at=review_at
        )
        goals: list[Goal] = []
        selected_ids: list[str] = []
        for candidate in candidates:
            goal = self.propose_goal(
                goal_type=GoalType.INTRINSIC,
                description=candidate.description,
                structured_target={
                    "motivation_id": candidate.motivation_id,
                    "target_ref": candidate.target_ref,
                    "source_refs": list(candidate.source_refs),
                    "motivation_revision": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).revision,
                    "motivation_strength": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).strength,
                    "motivation_persistence": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).persistence,
                    "motivation_uncertainty": self.motivation_dynamics.get(
                        candidate.motivation_id
                    ).uncertainty,
                },
                origin_actor=OriginActor.SELF,
                origin_input_kind=OriginInputKind.INTERNAL_STATE,
                origin_source_ref=f"motivation:{candidate.motivation_id}",
                priority=candidate.priority,
                urgency=candidate.urgency,
                expected_utility=candidate.priority,
                confidence=candidate.confidence,
                motivation_revision_ref=(
                    f"motivation:{candidate.motivation_id}@"
                    f"{self.motivation_dynamics.get(candidate.motivation_id).revision}"
                ),
                goal_id=f"intrinsic:{candidate.motivation_id}",
            )
            motivation = self.motivation_dynamics.link_goal(
                candidate.motivation_id, goal.goal_id
            )
            for experience_id in motivation.related_experience_ids:
                self.link_experience_result(
                    experience_id,
                    kind="goal",
                    reference=f"goal:{goal.goal_id}",
                    evidence_refs=(f"motivation:{motivation.motivation_id}",),
                )
            goals.append(goal)
            selected_ids.append(candidate.motivation_id)
        episode = self.motivation_dynamics.record_episode(
            selected_ids=tuple(selected_ids),
            held_ids=held_ids,
            generated_goal_ids=tuple(goal.goal_id for goal in goals),
            budget=max_goal_proposals,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        unresolved_intrinsic = any(
            goal.intrinsic_status
            in {
                IntrinsicGoalStatus.PROPOSAL,
                IntrinsicGoalStatus.DELIBERATING,
                IntrinsicGoalStatus.DEFERRED,
                IntrinsicGoalStatus.ENDORSED,
            }
            for goal in self.goal_manager.goals.values()
            if goal.goal_type == GoalType.INTRINSIC
        )
        if not candidates and not unresolved_intrinsic:
            self.goal_manager.record_no_intrinsic_goal(
                provenance_refs=("motivation-dynamics:no-eligible-proposal",),
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
            )
        self._persist_motivation_state()
        return episode, goals

    def decay_motivation(self, elapsed_hours: float) -> list[MotivationRecord]:
        records = self.motivation_dynamics.decay(elapsed_hours)
        self._persist_motivation_state()
        return records

    def decay_motivation_record(
        self, motivation_id: str, elapsed_hours: float
    ) -> MotivationRecord | None:
        record = self.motivation_dynamics.decay_record(motivation_id, elapsed_hours)
        self._persist_motivation_state()
        return record

    def deliberate_intrinsic_goal(self, goal_id: str) -> IntrinsicGoalDeliberation:
        """Resolve an intrinsic proposal from structured authoritative state only."""
        event = current_agent_event()
        goal = self.goal_manager.get(goal_id)
        if goal.intrinsic_status not in {
            IntrinsicGoalStatus.PROPOSAL,
            IntrinsicGoalStatus.DEFERRED,
        }:
            raise ValueError("Intrinsic proposal is not available for deliberation")
        target = goal.structured_target or {}
        motivation_id = target.get("motivation_id")
        if not isinstance(motivation_id, str):
            raise ValueError("Intrinsic proposal requires an originating Motivation")
        motivation = self.motivation_dynamics.get(motivation_id)
        motivation_score = (
            0.5 * float(target.get("motivation_strength", motivation.strength))
            + 0.3 * float(target.get("motivation_persistence", motivation.persistence))
            + 0.2
            * (
                1.0
                - float(target.get("motivation_uncertainty", motivation.uncertainty))
            )
        )

        deliberation_value_effects = dict(goal.value_effects)
        for value_id in motivation.related_value_ids:
            deliberation_value_effects.setdefault(value_id, 1.0)
        value_scores = self.value_system.evaluate(
            {goal.goal_id: deliberation_value_effects}
        )
        value_score = max(-1.0, min(1.0, value_scores[0].total_score))
        value_conflicts = value_scores[0].conflicts
        value_ids = set(deliberation_value_effects)
        if goal.origin_value_id is not None:
            value_ids.add(goal.origin_value_id)
        values = [self.value_system.get(value_id) for value_id in sorted(value_ids)]
        value_revisions = {value.value_id: value.revision for value in values}

        active_goal_conflicts = tuple(
            conflict_id
            for conflict_id in goal.conflict_ids
            if self.goal_manager.get(conflict_id).status == GoalStatus.ACTIVE
        )
        active_commitments = [
            item
            for item in self.commitment_store.list_commitments()
            if item.status in ACCEPTED_COMMITMENT_STATUSES
        ]
        proposal_conflict_refs = {
            motivation_id,
            f"motivation:{motivation_id}",
            *goal.related_desire_ids,
            *value_ids,
            *(f"value:{item}" for item in value_ids),
            *goal.conflict_ids,
            *(f"goal:{item}" for item in goal.conflict_ids),
        }
        commitment_conflicts = tuple(
            item.commitment_id
            for item in active_commitments
            if any(
                conflict.subject_ref in proposal_conflict_refs
                for conflict in item.unresolved_conflicts
            )
            or (
                item.related_goal_id is not None
                and item.related_goal_id in goal.conflict_ids
            )
        )
        conflict_score = -1.0 if active_goal_conflicts or commitment_conflicts else 1.0

        attention_capacity = self.attention_system.capacity
        proposal_has_attention = any(
            candidate_id
            in {
                f"goal:{goal.goal_id}",
                f"motivation:{motivation_id}",
            }
            for candidate_id in self.attention_system.focus.candidate_ids
        )
        available_attention = max(
            0, attention_capacity - len(self.attention_system.focus.candidate_ids)
        )
        if proposal_has_attention:
            available_attention = max(1, available_attention)
        attention_score = available_attention / attention_capacity
        cost = unit_target(target.get("estimated_cost", 0.0), "estimated_cost")
        risk = unit_target(target.get("estimated_risk", 0.0), "estimated_risk")
        cost_risk_score = 1.0 - 0.5 * (cost + risk)

        cognitive_load = min(
            1.0, len(self.working_memory.items) / self.working_memory.item_capacity
        )
        attention_saturation = len(self.attention_system.focus.candidate_ids) / max(
            1, attention_capacity
        )
        quality = self.metacognition.current_quality(
            cognitive_load=cognitive_load,
            attention_saturation=attention_saturation,
            emotion_valence=self.emotion_engine.state.valence,
            emotion_arousal=self.emotion_engine.state.arousal,
            provenance_refs=(
                f"working-memory:{len(self.working_memory.items)}",
                f"attention-focus:{self.attention_system.focus.revision}",
                "emotion:current",
            ),
        )

        relationship = self._relationship_for_target(target)
        relationship_score = (
            0.5
            if relationship is None
            else max(
                -1.0,
                min(
                    1.0,
                    relationship.axes.trust
                    + relationship.reciprocity
                    - relationship.axes.caution
                    - relationship.uncertainty,
                ),
            )
        )
        relationship_revisions = (
            {}
            if relationship is None
            else {relationship.relationship_id: relationship.revision}
        )

        themes = set(string_values(target.get("theme_codes"))) | set(
            string_values(target.get("topic_tags"))
        )
        narrative_conflicts = tuple(
            item.conflict_id
            for item in self.narrative_self.conflicts.values()
            if item.resolved_at is None
            and any(
                themes.intersection(self.narrative_self.get_claim(claim_id).theme_codes)
                for claim_id in item.claim_ids
            )
        )
        narrative_score = (
            -1.0 if narrative_conflicts else (0.7 if goal.narrative_self_refs else 0.5)
        )

        capability_ids = string_values(target.get("capability_ids"))
        capabilities = self.self_model.state.capabilities
        missing_capabilities = tuple(
            identifier
            for identifier in capability_ids
            if identifier not in capabilities
        )
        matching_limitations = tuple(
            item
            for item in self.self_model.state.known_limitations.values()
            if set(item.capability_ids).intersection(capability_ids)
            or themes.intersection(item.tags)
        )
        matching_uncertainties = tuple(
            item
            for item in self.self_model.state.epistemic_uncertainties.values()
            if themes.intersection(item.tags)
        )
        capability_score = (
            0.5
            if not capability_ids
            else sum(
                capabilities[item].confidence
                for item in capability_ids
                if item in capabilities
            )
            / max(1, len(capability_ids))
        )
        capability_score -= max(
            (item.confidence for item in matching_limitations), default=0.0
        )
        missing_information = tuple(
            dict.fromkeys(
                (
                    *string_values(target.get("missing_information")),
                    *(f"capability:{item}" for item in missing_capabilities),
                    *(
                        f"uncertainty:{item.uncertainty_id}"
                        for item in matching_uncertainties
                    ),
                    *(("goal:needs_information",) if goal.needs_information else ()),
                )
            )
        )
        information_score = -1.0 if missing_information else 1.0

        factors = {
            "motivation": motivation_score,
            "values": value_score,
            "goal_commitment_conflicts": conflict_score,
            "attention_capacity": attention_score,
            "cost_risk": cost_risk_score,
            "metacognitive_quality": quality.estimated_quality,
            "relationship": relationship_score,
            "narrative_continuity": narrative_score,
            "capability": max(-1.0, min(1.0, capability_score)),
            "information": information_score,
        }
        score = max(
            -1.0,
            min(
                1.0,
                0.2 * motivation_score
                + 0.15 * value_score
                + 0.12 * conflict_score
                + 0.08 * attention_score
                + 0.1 * cost_risk_score
                + 0.1 * quality.estimated_quality
                + 0.06 * relationship_score
                + 0.07 * narrative_score
                + 0.07 * capability_score
                + 0.05 * information_score,
            ),
        )
        reasons: list[str] = ["structured_multi_factor_deliberation"]
        if motivation.status != MotivationStatus.ACTIVE:
            action = IntrinsicGoalAction.REJECT
            reasons.append("originating_motivation_inactive")
        elif (
            active_goal_conflicts
            or commitment_conflicts
            or value_conflicts
            or narrative_conflicts
        ):
            action = IntrinsicGoalAction.DEFER
            reasons.append("major_unresolved_conflict")
        elif missing_information:
            action = IntrinsicGoalAction.DEFER
            reasons.append("missing_information_or_capability")
        elif available_attention == 0 or quality.estimated_quality < 0.4:
            action = IntrinsicGoalAction.DEFER
            reasons.append("insufficient_deliberative_capacity")
        elif risk >= 0.85 or cost >= 0.9 or score < 0.2:
            action = IntrinsicGoalAction.REJECT
            reasons.append("cost_risk_or_utility_unacceptable")
        elif score >= 0.45:
            action = IntrinsicGoalAction.ENDORSE
            reasons.append("multi_factor_threshold_satisfied")
        else:
            action = IntrinsicGoalAction.DEFER
            reasons.append("endorsement_threshold_not_met")

        provenance = tuple(
            dict.fromkeys(
                (
                    goal.motivation_revision_ref
                    or f"motivation:{motivation_id}@{motivation.revision}",
                    *(f"value:{item.value_id}@{item.revision}" for item in values),
                    *(
                        f"value-evidence:{item}"
                        for value in values
                        for item in (
                            *value.supporting_evidence_ids,
                            *value.opposing_evidence_ids,
                        )
                    ),
                    *(f"goal:{item}" for item in active_goal_conflicts),
                    *(f"commitment:{item}" for item in commitment_conflicts),
                    f"attention-focus:{self.attention_system.focus.revision}",
                    *quality.provenance_refs,
                    *(
                        f"relationship:{key}@{revision}"
                        for key, revision in relationship_revisions.items()
                    ),
                    *goal.narrative_self_refs,
                    *(f"self-conflict:{item}" for item in narrative_conflicts),
                    f"self-model:{self.self_model.state.revision}",
                    *(
                        f"limitation:{item.limitation_id}"
                        for item in matching_limitations
                    ),
                    *missing_information,
                )
            )
        )
        self.goal_manager.begin_intrinsic_deliberation(goal_id)
        record = self.goal_manager.resolve_intrinsic_deliberation(
            goal_id,
            action=action,
            score=score,
            factor_scores=factors,
            reason_codes=tuple(reasons),
            provenance_refs=provenance,
            value_revision_refs=value_revisions,
            self_model_revision=self.self_model.state.revision,
            attention_revision=self.attention_system.focus.revision,
            relationship_revision_refs=relationship_revisions,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return record

    def schedule_motivation_reviews(
        self, review_at: datetime
    ) -> list[MotivationRecord]:
        records = self.motivation_dynamics.schedule_next_reviews(review_at)
        self._persist_motivation_state()
        return records

    def record_no_intrinsic_goal(self) -> IntrinsicGoalDeliberation:
        event = current_agent_event()
        record = self.goal_manager.record_no_intrinsic_goal(
            provenance_refs=("motivation-dynamics:no-eligible-proposal",),
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return record

    def generate_intrinsic_plan(self, goal_id: str) -> Plan:
        goal = self.goal_manager.get(goal_id)
        if goal.intrinsic_status != IntrinsicGoalStatus.ENDORSED:
            raise ValueError("Plan generation requires an endorsed intrinsic Goal")
        existing = self.plan_store.list_plans(goal_id=goal_id)
        if existing:
            return existing[0]
        target = goal.structured_target or {}
        motivation_id = str(target.get("motivation_id", goal_id))
        candidate = PlanCandidate(
            plan_id=f"intrinsic-plan:{motivation_id}",
            goal_id=goal_id,
            success_condition=PlanCondition(
                condition_code="intrinsic_goal_context_observed",
                required_evidence_types=("observation",),
            ),
            failure_condition=PlanCondition(
                condition_code="intrinsic_goal_blocked",
                required_evidence_types=("failure",),
            ),
            abandonment_condition=PlanCondition(
                condition_code="intrinsic_goal_abandoned",
                required_evidence_types=("abandonment",),
            ),
            steps=(
                StepDefinition(
                    step_id="observe_progress",
                    action_type=ActionType.INTERNAL,
                    action_code="observe_intrinsic_goal_progress",
                    parameters={
                        "action": {
                            "tool_name": "restricted_metadata_read",
                            "arguments": {"namespace": "project", "key": "name"},
                        },
                        "value_effects": {"honesty": 0.2},
                    },
                    expected_observation=ExpectedObservation(
                        observation_code="restricted_metadata_read",
                        evidence_types=("observation",),
                    ),
                    verification=VerificationPolicy(
                        verification_code="structured_observation",
                        required_evidence_types=("observation",),
                    ),
                    timeout_seconds=3600.0,
                ),
            ),
        )
        return self.create_plan(candidate, actor_id="subject_scheduler")

    def activate_endorsed_intrinsic_goal(self, goal_id: str) -> GoalDecision:
        goal = self.goal_manager.get(goal_id)
        plans = self.plan_store.list_plans(goal_id=goal_id)
        if goal.intrinsic_status != IntrinsicGoalStatus.ENDORSED or not plans:
            raise ValueError(
                "Intrinsic adoption requires endorsement and a generated Plan"
            )
        decision = self.adopt_goal(goal_id)
        if decision.action in {GoalDecisionAction.ACTIVATE, GoalDecisionAction.RESUME}:
            self.activate_plan(plans[0].plan_id)
        return decision

    def propose_goal(
        self,
        *,
        goal_type: GoalType,
        description: str,
        structured_target: dict[str, Any] | None = None,
        origin_value_id: str | None = None,
        origin_actor: OriginActor | None = None,
        origin_input_kind: OriginInputKind | None = None,
        origin_source_ref: str | None = None,
        priority: float = 0.5,
        urgency: float = 0.5,
        expected_utility: float = 0.5,
        confidence: float = 0.5,
        dependency_ids: tuple[str, ...] = (),
        conflict_ids: tuple[str, ...] = (),
        deadline: str | None = None,
        value_effects: dict[str, float] | None = None,
        related_desire_ids: tuple[str, ...] = (),
        motivation_revision_ref: str | None = None,
        needs_information: bool = False,
        goal_id: str | None = None,
    ) -> Goal:
        event = current_agent_event()
        if origin_value_id is not None:
            self.value_system.get(origin_value_id)
        if value_effects:
            self.value_system.evaluate({"goal_proposal": value_effects})
        target = structured_target or {}
        relationship = self._relationship_for_target(target)
        if relationship is not None:
            expected_utility = max(
                0.0,
                min(
                    1.0,
                    expected_utility
                    + 0.1
                    * (
                        relationship.reciprocity
                        + relationship.axes.trust
                        - relationship.axes.caution
                        - 0.5
                    ),
                ),
            )
        narrative_selection = self.narrative_self.select_relevant(
            theme_codes=string_values(target.get("theme_codes"))
            + string_values(target.get("topic_tags")),
            capability_ids=string_values(target.get("capability_ids")),
        )
        narrative_refs = tuple(
            f"identity-claim:{claim_id}@{self.narrative_self.get_claim(claim_id).revision}"
            for claim_id in narrative_selection.claim_ids
        )
        projection_id = target.get("future_self_projection_id")
        if (
            isinstance(projection_id, str)
            and projection_id in self.narrative_self.future_self
        ):
            narrative_refs = (*narrative_refs, f"future-self:{projection_id}")
        identity_origin: IdentityOrigin | None = None
        if origin_actor is not None:
            identity_origin = new_identity_origin(
                origin_actor,
                origin_input_kind or OriginInputKind.SUGGESTION,
                source_ref=origin_source_ref,
                event_id=None if event is None else event.event_id,
                event_sequence=None if event is None else event.processing_sequence,
                endorsement=(
                    EndorsementStatus.PENDING
                    if goal_type == GoalType.INTRINSIC
                    and motivation_revision_ref is not None
                    else None
                ),
            )
        goal = self.goal_manager.propose(
            goal_type=goal_type,
            description=description,
            structured_target=structured_target,
            origin_event_id=None if event is None else event.event_id,
            origin_value_id=origin_value_id,
            identity_origin=identity_origin,
            priority=priority,
            urgency=urgency,
            expected_utility=expected_utility,
            confidence=confidence,
            dependency_ids=dependency_ids,
            conflict_ids=conflict_ids,
            deadline=deadline,
            value_effects=value_effects,
            value_revision_refs={
                value_id: self.value_system.get(value_id).revision
                for value_id in set(value_effects or {})
                | ({origin_value_id} if origin_value_id is not None else set())
            },
            narrative_self_refs=narrative_refs,
            related_desire_ids=related_desire_ids,
            motivation_revision_ref=motivation_revision_ref,
            needs_information=needs_information,
            goal_id=goal_id,
        )
        self._persist_motivation_state()
        return goal

    def adopt_goal(self, goal_id: str) -> GoalDecision:
        event = current_agent_event()
        value_scores = self._goal_value_scores()
        previous_status = self.goal_manager.get(goal_id).status
        decision = self.goal_manager.adopt(
            goal_id,
            value_score=value_scores.get(goal_id, 0.0),
            value_scores=value_scores,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        goal = self.goal_manager.get(goal_id)
        if previous_status != GoalStatus.FAILED and goal.status == GoalStatus.FAILED:
            self._apply_goal_outcome_appraisal(goal, GoalStatus.FAILED)
            self._sync_commitment_from_goal(
                goal, GoalStatus.FAILED, "deadline_expired", None
            )
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return decision

    def transition_goal(
        self,
        goal_id: str,
        status: GoalStatus,
        *,
        reason: str,
        outcome: str | None = None,
    ) -> Goal:
        if status == GoalStatus.COMPLETED:
            plans = self.plan_store.list_plans(goal_id=goal_id)
            if plans and not any(plan.status == PlanStatus.COMPLETED for plan in plans):
                raise ValueError(
                    "Goal cannot complete before its Plan success condition"
                )
        event = current_agent_event()
        goal = self.goal_manager.transition(
            goal_id,
            status,
            reason=reason,
            outcome=outcome,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._apply_goal_outcome_appraisal(goal, status)
        self._sync_commitment_from_goal(goal, status, reason, outcome)
        if status in {
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        }:
            self.motivation_dynamics.resolve_goal(
                goal.goal_id, success=status == GoalStatus.COMPLETED
            )
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return goal

    def reevaluate_goals(self) -> list[GoalDecision]:
        event = current_agent_event()
        previous_statuses = {
            goal.goal_id: goal.status for goal in self.goal_manager.goals.values()
        }
        decisions = self.goal_manager.reevaluate(
            value_scores=self._goal_value_scores(),
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        for goal in self.goal_manager.goals.values():
            if (
                previous_statuses.get(goal.goal_id) != GoalStatus.FAILED
                and goal.status == GoalStatus.FAILED
            ):
                self._apply_goal_outcome_appraisal(goal, GoalStatus.FAILED)
                self._sync_commitment_from_goal(
                    goal, GoalStatus.FAILED, "deadline_expired", None
                )
                self.motivation_dynamics.resolve_goal(goal.goal_id, success=False)
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return decisions

    def goal_decision_input(self) -> GoalDecisionInput:
        return self.goal_manager.decision_input(
            value_scores=self._goal_value_scores(),
            active_commitment_ids=(
                item.commitment_id
                for item in self.commitment_store.list_commitments()
                if item.status in ACCEPTED_COMMITMENT_STATUSES
            ),
        )

    def create_commitment(
        self,
        *,
        description: str,
        deadline: str | None = None,
        beneficiary: str = "self",
        scope: str | None = None,
        cost: float = 0.0,
        burden: float = 0.0,
        fulfillability: CommitmentFulfillability = CommitmentFulfillability.UNKNOWN,
        fulfillability_reason: str | None = None,
        related_desire_ids: tuple[str, ...] = (),
        conflicting_desire_ids: tuple[str, ...] = (),
        conflicting_value_ids: tuple[str, ...] = (),
        conflicting_commitment_ids: tuple[str, ...] = (),
        commitment_id: str | None = None,
        origin_actor: OriginActor | None = None,
        origin_source_ref: str | None = None,
        interlocutor_key: str | None = None,
    ) -> Commitment:
        event = current_agent_event()
        identifier = commitment_id or str(uuid4())
        if deadline is not None:
            parsed_deadline = datetime.fromisoformat(deadline)
            if parsed_deadline.tzinfo is None:
                raise ValueError("Commitment deadline must include a timezone")
            if parsed_deadline <= datetime.now(UTC):
                raise ValueError("Commitment deadline has expired")
        for desire_id in (*related_desire_ids, *conflicting_desire_ids):
            motivation = self.motivation_dynamics.get(desire_id)
            if motivation.kind.value != "desire":
                raise ValueError(f"Motivation is not a Desire: {desire_id}")
        for value_id in conflicting_value_ids:
            self.value_system.get(value_id)
        for other_id in conflicting_commitment_ids:
            self.commitment_store.get(other_id)
        identity_origin = new_identity_origin(
            origin_actor or OriginActor.SELF,
            OriginInputKind.REQUEST
            if origin_actor not in {None, OriginActor.SELF}
            else OriginInputKind.INTERNAL_STATE,
            source_ref=origin_source_ref or "runtime:commitment_proposal",
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        relationship_refs: tuple[str, ...] = ()
        if interlocutor_key is not None:
            relationship = self.relationship_store.ensure_interlocutor(interlocutor_key)
            relationship_refs = (f"relationship:{relationship.relationship_id}",)
        commitment = self.commitment_store.request(
            description=description,
            beneficiary=beneficiary,
            scope=scope or description,
            origin_event_id=None if event is None else event.event_id,
            identity_origin=identity_origin,
            deadline=deadline,
            cost=cost,
            burden=burden,
            fulfillability=fulfillability,
            fulfillability_reason=fulfillability_reason,
            related_desire_ids=related_desire_ids,
            relationship_refs=relationship_refs,
            conflicting_desire_ids=conflicting_desire_ids,
            conflicting_value_ids=conflicting_value_ids,
            conflicting_commitment_ids=conflicting_commitment_ids,
            commitment_id=identifier,
        )
        self._persist_motivation_state()
        return commitment

    def accept_commitment(
        self,
        commitment_id: str,
        *,
        self_endorsement: str,
        priority: float = 0.7,
        urgency: float = 0.7,
        expected_utility: float = 0.7,
        confidence: float = 0.8,
        value_effects: dict[str, float] | None = None,
        conflict_ids: tuple[str, ...] = (),
    ) -> Commitment:
        event = current_agent_event()
        commitment = self.commitment_store.get(commitment_id)
        if commitment.status != CommitmentStatus.PROPOSED:
            raise ValueError("Only proposed commitments can be accepted")
        if commitment.deadline is not None and datetime.fromisoformat(
            commitment.deadline
        ) <= datetime.now(UTC):
            raise ValueError("Commitment deadline has expired")
        commitment.identity_origin.endorse(
            self_endorsement,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        interlocutor_key = None
        if commitment.relationship_refs:
            relationship_id = commitment.relationship_refs[0].removeprefix(
                "relationship:"
            )
            relationship = self.relationship_store.get(relationship_id)
            interlocutor_key = relationship.interlocutor_keys[0]
        goal = self.propose_goal(
            goal_type=GoalType.COMMITMENT,
            description=commitment.description,
            structured_target={
                "commitment_id": commitment_id,
                **(
                    {}
                    if interlocutor_key is None
                    else {"interlocutor_key": interlocutor_key}
                ),
            },
            origin_actor=commitment.identity_origin.actor,
            origin_input_kind=commitment.identity_origin.input_kind,
            origin_source_ref=commitment.identity_origin.source_ref,
            priority=priority,
            urgency=urgency,
            expected_utility=expected_utility,
            confidence=confidence,
            conflict_ids=conflict_ids,
            deadline=commitment.deadline,
            value_effects=value_effects,
            related_desire_ids=commitment.related_desire_ids,
            goal_id=f"commitment:{commitment_id}",
        )
        self.adopt_goal(goal.goal_id)
        if self.goal_manager.get(goal.goal_id).status != GoalStatus.ACTIVE:
            raise ValueError("Commitment intention was not endorsed for activation")
        accepted = self.commitment_store.accept(
            commitment_id,
            acceptance_ref=self_endorsement,
            related_goal_id=goal.goal_id,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        if interlocutor_key is not None:
            self.relationship_store.link_commitment(
                interlocutor_key, f"commitment:{commitment_id}"
            )
        self._sync_self_references()
        self._persist_self_model_state()
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return accepted

    def reassess_commitment(
        self,
        commitment_id: str,
        *,
        fulfillability: CommitmentFulfillability,
        reason: str,
    ) -> Commitment:
        event = current_agent_event()
        current = self.commitment_store.get(commitment_id)
        decision_ref = None
        unresolved_decision = any(
            (record := self.decision_store.records.get(ref.removeprefix("decision:")))
            is not None
            and record.status == DecisionStatus.AWAITING_OUTCOME
            for ref in current.decision_refs
        )
        if (
            fulfillability == CommitmentFulfillability.IMPOSSIBLE
            and not unresolved_decision
        ):
            candidates = [
                ActionCandidate(
                    candidate_id=f"commitment:{commitment_id}:renegotiate",
                    candidate_type=ActionType.RESPOND,
                    proposed_action="renegotiate_commitment",
                    parameters={"commitment_id": commitment_id},
                    prerequisites=(),
                    predicted_outcomes=(
                        PredictedOutcome(
                            outcome_id="revised_terms",
                            description="Beneficiary agrees to revised terms",
                            probability=0.6,
                            utility=0.6,
                        ),
                    ),
                    uncertainty=0.4,
                    estimated_cost=0.3,
                    estimated_risk=0.2,
                    value_effects={},
                    appraisal_contributions={"accountability": 0.7},
                ),
                ActionCandidate(
                    candidate_id=f"commitment:{commitment_id}:notify",
                    candidate_type=ActionType.RESPOND,
                    proposed_action="notify_beneficiary_of_impossibility",
                    parameters={"commitment_id": commitment_id},
                    prerequisites=(),
                    predicted_outcomes=(
                        PredictedOutcome(
                            outcome_id="beneficiary_informed",
                            description="Beneficiary receives timely notice",
                            probability=0.9,
                            utility=0.4,
                        ),
                    ),
                    uncertainty=0.1,
                    estimated_cost=0.1,
                    estimated_risk=0.1,
                    value_effects={},
                    appraisal_contributions={"accountability": 0.8},
                ),
                ActionCandidate(
                    candidate_id=f"commitment:{commitment_id}:defer",
                    candidate_type=ActionType.DEFER,
                    proposed_action="defer_for_new_evidence",
                    parameters={"commitment_id": commitment_id},
                    prerequisites=(),
                    predicted_outcomes=(),
                    uncertainty=0.8,
                    estimated_cost=0.4,
                    estimated_risk=0.8,
                    value_effects={},
                    appraisal_contributions={"accountability": -0.5},
                ),
            ]
            decision = self.create_decision(
                candidates,
                decision_id=f"commitment-impossible:{commitment_id}:{uuid4()}",
            )
            decision_ref = f"decision:{decision.decision_id}"
        updated = self.commitment_store.reassess(
            commitment_id,
            fulfillability=fulfillability,
            reason=reason,
            decision_ref=decision_ref,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return updated

    def renegotiate_commitment(
        self,
        commitment_id: str,
        *,
        reason: str,
        proposed_scope: str | None = None,
        proposed_deadline: str | None = None,
    ) -> Commitment:
        event = current_agent_event()
        updated = self.commitment_store.renegotiate(
            commitment_id,
            reason=reason,
            proposed_scope=proposed_scope,
            proposed_deadline=proposed_deadline,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        return updated

    def repair_commitment(
        self,
        commitment_id: str,
        *,
        reason: str,
        evidence_refs: tuple[str, ...],
    ) -> Commitment:
        event = current_agent_event()
        current = self.commitment_store.get(commitment_id)
        if current.status != CommitmentStatus.BREACHED:
            raise ValueError("Only breached commitments can be repaired")
        updated = self.commitment_store.record_accountability(
            commitment_id,
            action=CommitmentLifecycleAction.REPAIR,
            reason=reason,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.relationship_store.transition_commitment(
            f"commitment:{commitment_id}",
            status="repaired",
            evidence_ref=evidence_refs[0],
        )
        self.narrative_self.record_commitment_event(
            f"commitment:{commitment_id}",
            kind="repair",
            description=reason,
            evidence_refs=evidence_refs,
            relationship_refs=current.relationship_refs,
        )
        self._persist_narrative_self_state()
        self._persist_motivation_state()
        return updated

    def transition_commitment(
        self,
        commitment_id: str,
        status: CommitmentStatus,
        *,
        reason: str,
        outcome: str | None = None,
    ) -> Commitment:
        if status not in {
            CommitmentStatus.FULFILLED,
            CommitmentStatus.RELEASED,
            CommitmentStatus.BREACHED,
        }:
            raise ValueError("Use the dedicated commitment lifecycle operation")
        event = current_agent_event()
        commitment = self.commitment_store.get(commitment_id)
        goal = (
            None
            if commitment.related_goal_id is None
            else self.goal_manager.goals.get(commitment.related_goal_id)
        )
        if goal is not None and goal.status not in {
            GoalStatus.COMPLETED,
            GoalStatus.ABANDONED,
            GoalStatus.FAILED,
        }:
            mapped_status = {
                CommitmentStatus.FULFILLED: GoalStatus.COMPLETED,
                CommitmentStatus.RELEASED: GoalStatus.ABANDONED,
                CommitmentStatus.BREACHED: GoalStatus.FAILED,
            }[status]
            self.transition_goal(
                goal.goal_id,
                mapped_status,
                reason=f"commitment_{status.value}:{reason}",
                outcome=outcome,
            )
            return self.commitment_store.get(commitment_id)
        commitment = self.commitment_store.transition(
            commitment_id,
            status,
            reason=reason,
            outcome=outcome,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.relationship_store.transition_commitment(
            f"commitment:{commitment_id}",
            status=status.value,
            evidence_ref=f"commitment:{commitment_id}:{status.value}",
        )
        if status == CommitmentStatus.BREACHED:
            self.narrative_self.record_commitment_event(
                f"commitment:{commitment_id}",
                kind="breach",
                description=reason,
                evidence_refs=(f"commitment:{commitment_id}:breached",),
                relationship_refs=commitment.relationship_refs,
            )
            self._persist_narrative_self_state()
        self._sync_motivation_working_memory()
        self._persist_motivation_state()
        return commitment

    def _goal_value_scores(self) -> dict[str, float]:
        options = {
            goal.goal_id: goal.value_effects
            for goal in self.goal_manager.goals.values()
            if goal.value_effects
        }
        if not options:
            return {}
        return {
            score.option_id: score.total_score
            for score in self.value_system.evaluate(options)
        }

    def _apply_goal_outcome_appraisal(self, goal: Goal, status: GoalStatus) -> None:
        progress = {
            GoalStatus.COMPLETED: 1.0,
            GoalStatus.FAILED: -1.0,
            GoalStatus.ABANDONED: -0.5,
        }.get(status)
        if progress is None:
            return
        self.emotion_engine.update_from_appraisal(
            AppraisalResult(
                novelty=None,
                goal_progress=progress,
                threat=0.0,
                controllability=0.5,
                certainty=goal.confidence,
                social_relevance=0.0,
                effort_cost=0.0,
                novelty_valid=False,
                reasons=(f"goal_{status.value}",),
            )
        )

    def _sync_commitment_from_goal(
        self,
        goal: Goal,
        status: GoalStatus,
        reason: str,
        outcome: str | None,
    ) -> None:
        commitment = next(
            (
                item
                for item in self.commitment_store.commitments.values()
                if item.related_goal_id == goal.goal_id
                and item.status
                in {CommitmentStatus.ACTIVE, CommitmentStatus.RENEGOTIATING}
            ),
            None,
        )
        mapped = {
            GoalStatus.COMPLETED: CommitmentStatus.FULFILLED,
            GoalStatus.ABANDONED: CommitmentStatus.RELEASED,
            GoalStatus.FAILED: CommitmentStatus.BREACHED,
        }.get(status)
        if commitment is None or mapped is None:
            return
        event = current_agent_event()
        self.commitment_store.transition(
            commitment.commitment_id,
            mapped,
            reason=f"goal_{status.value}:{reason}",
            outcome=outcome,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self.relationship_store.transition_commitment(
            f"commitment:{commitment.commitment_id}",
            status=mapped.value,
            evidence_ref=f"commitment:{commitment.commitment_id}:{mapped.value}",
        )
        if mapped == CommitmentStatus.BREACHED:
            self.narrative_self.record_commitment_event(
                f"commitment:{commitment.commitment_id}",
                kind="breach",
                description=reason,
                evidence_refs=(f"commitment:{commitment.commitment_id}:breached",),
                relationship_refs=commitment.relationship_refs,
            )
            self._persist_narrative_self_state()

    def _sync_motivation_working_memory(self) -> None:
        self._sync_plan_goal_statuses()
        for goal in self.goal_manager.goals.values():
            item_id = f"goal:{goal.goal_id}"
            if goal.status == GoalStatus.ACTIVE:
                self.working_memory.admit(
                    working_memory_item(
                        item_id=item_id,
                        kind=WorkingMemoryKind.GOAL,
                        content=goal.description,
                        activation=0.9,
                        salience=max(goal.priority, goal.urgency),
                        retention_reason=RetentionReason.ONGOING_GOAL,
                        source="runtime.goal_manager",
                    )
                )
            else:
                self.working_memory.forget(item_id)
        for commitment in self.commitment_store.commitments.values():
            item_id = f"commitment:{commitment.commitment_id}"
            if commitment.status in ACCEPTED_COMMITMENT_STATUSES:
                self.working_memory.admit(
                    working_memory_item(
                        item_id=item_id,
                        kind=WorkingMemoryKind.COMMITMENT,
                        content=commitment.description,
                        activation=0.9,
                        salience=0.9,
                        retention_reason=RetentionReason.ACTIVE_COMMITMENT,
                        source="runtime.commitment_store",
                    )
                )
            else:
                self.working_memory.forget(item_id)
        self._sync_plan_working_memory()

    def _require_active_plan_goal(self, plan_id: str) -> None:
        plan = self.plan_store.get(plan_id)
        if self.goal_manager.get(plan.goal_id).status != GoalStatus.ACTIVE:
            raise ValueError("Step lifecycle requires an active Goal")

    def _sync_plan_goal_statuses(self) -> None:
        event = current_agent_event()
        event_id = None if event is None else event.event_id
        event_sequence = None if event is None else event.processing_sequence
        for plan in self.plan_store.list_plans():
            goal = self.goal_manager.get(plan.goal_id)
            if plan.status == PlanStatus.ACTIVE and goal.status != GoalStatus.ACTIVE:
                self.plan_store.pause(
                    plan.plan_id,
                    event_id=event_id,
                    event_sequence=event_sequence,
                )
            elif plan.status == PlanStatus.PAUSED and goal.status == GoalStatus.ACTIVE:
                self.plan_store.resume(
                    plan.plan_id,
                    event_id=event_id,
                    event_sequence=event_sequence,
                )

    def _sync_plan_working_memory(self) -> None:
        for item in self.working_memory.items:
            if item.kind == WorkingMemoryKind.STEP:
                self.working_memory.forget(item.item_id)
        for plan, step, _state in self.plan_store.actionable_steps():
            self.working_memory.admit(
                working_memory_item(
                    item_id=f"plan-step:{plan.plan_id}:{step.step_id}",
                    kind=WorkingMemoryKind.STEP,
                    content=json.dumps(
                        {
                            "plan_id": plan.plan_id,
                            "plan_revision": plan.revision,
                            "step_id": step.step_id,
                            "action_type": step.action_type.value,
                            "action_code": step.action_code,
                            "expected_observation": (
                                step.expected_observation.observation_code
                            ),
                            "verification": step.verification.verification_code,
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    ),
                    activation=1.0,
                    salience=1.0,
                    retention_reason=RetentionReason.ACTIONABLE_STEP,
                    source="runtime.plan_store",
                )
            )
