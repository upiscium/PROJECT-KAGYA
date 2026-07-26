from typing import Any
from uuid import NAMESPACE_URL, uuid5

from kagya.agency import (
    AgencyAttribution,
    AttributionTarget,
    CausalContributor,
    CausalContributorKind,
)
from kagya.belief import EpistemicStatus, Proposition
from kagya.cognition import (
    AppraisalResult,
    ValueUpdateKind,
)
from kagya.counterfactual import (
    AlternativeOutcome,
    CounterfactualSignal,
    CounterfactualSimulation,
    CounterfactualTarget,
    EvidenceStatus,
)
from kagya.decision import (
    ActionCandidate,
    ActionType,
    DecisionRecord,
    PredictedOutcome,
    parse_candidate_output,
    schema_candidate_prompt,
)
from kagya.identity import (
    IdentityClaimKind,
    IdentityClaimStatus,
    OriginActor,
    OriginInputKind,
    new_identity_origin,
)
from kagya.experience import (
    ExperienceAppraisal,
    ExperienceRecord,
    build_observation_experience,
)
from kagya.metacognition import CognitiveQuality
from kagya.motivation import (
    ACCEPTED_COMMITMENT_STATUSES,
    GoalStatus,
    MotivationKind,
    MotivationSource,
)
from kagya.planning import (
    EvidenceReference,
    Plan,
    StepStatus,
)
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.coordinators._shared import RuntimeDomainMixin, string_values
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemoryKind,
    working_memory_item,
)


class ActionCoordinator(RuntimeDomainMixin):
    def __init__(self, action_execution: Any | None = None) -> None:
        self.action_execution = action_execution

    def bind(self, action_execution: Any | None) -> None:
        self.action_execution = action_execution

    def create_intent(
        self,
        decision_id: str,
        *,
        idempotency_key: str,
        dry_run: bool = False,
        budget: Any | None = None,
    ) -> Any:
        if self.action_execution is None:
            raise RuntimeError("Action execution is not configured")
        return self.action_execution.create_from_decision(
            decision_id,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
            budget=budget,
        )

    @property
    def action_execution(self) -> Any | None:
        return self._action_execution

    @action_execution.setter
    def action_execution(self, execution: Any | None) -> None:
        self._action_execution = execution
        if hasattr(self, "action_coordinator"):
            self.action_coordinator.bind(execution)
        if not hasattr(self, "agency_attribution_store"):
            return
        if execution is not None:
            for record in self.agency_attribution_store.state.records:
                self._validate_attribution_chain(record)
            for simulation in self.counterfactual_store.state.records:
                self._validate_counterfactual_chain(simulation)

    def generate_decision_candidates(self, situation: str) -> list[ActionCandidate]:
        raw = self.provider.generate(schema_candidate_prompt(situation))
        return parse_candidate_output(raw)

    def current_plan_candidates(self) -> list[ActionCandidate]:
        return [
            self.plan_store.action_candidate(plan.plan_id, step.step_id)
            for plan, step, _state in self.plan_store.actionable_steps()
        ]

    def create_plan_action_decision(self, plan_id: str, step_id: str) -> DecisionRecord:
        plan = self.plan_store.get(plan_id)
        decision_id = f"plan-action:{uuid5(NAMESPACE_URL, f'{plan_id}@{plan.revision}/{step_id}')}"
        existing = self.decision_store.records.get(decision_id)
        if existing is not None:
            return existing
        candidate = self.plan_store.action_candidate(plan_id, step_id)
        fallback = ActionCandidate(
            candidate_id=f"{decision_id}:no-op",
            candidate_type=ActionType.NO_OP,
            proposed_action="do_not_execute_plan_step",
            parameters={},
            prerequisites=(),
            predicted_outcomes=(
                PredictedOutcome(
                    outcome_id="plan_step_not_executed",
                    description="Plan step remains pending",
                    probability=1.0,
                    utility=-1.0,
                ),
            ),
            uncertainty=0.0,
            estimated_cost=0.0,
            estimated_risk=0.0,
            value_effects={},
            appraisal_contributions={},
        )
        goal = self.goal_manager.get(plan.goal_id)
        target_ref = str((goal.structured_target or {}).get("target_ref", ""))
        context_id = (
            target_ref.removeprefix("context:")
            if target_ref.startswith("context:")
            else None
        )
        return self.create_decision(
            [candidate, fallback],
            decision_id=decision_id,
            context_id=context_id,
        )

    def start_action_plan_step(self, plan_id: str, step_id: str) -> None:
        state = self.plan_store.get(plan_id).step_state(step_id)
        if state.status == StepStatus.READY:
            self.start_plan_step(plan_id, step_id)
        elif state.status != StepStatus.IN_PROGRESS:
            raise ValueError("Action Plan step is not executable")

    def record_action_plan_observation(
        self,
        plan_id: str,
        step_id: str,
        observation_id: str,
        observation_code: str,
    ) -> Plan:
        plan = self.plan_store.get(plan_id)
        definition = plan.step_definition(step_id)
        evidence_types = tuple(
            dict.fromkeys(
                (
                    *definition.expected_observation.evidence_types,
                    *definition.verification.required_evidence_types,
                )
            )
        )
        return self.complete_plan_step(
            plan_id,
            step_id,
            tuple(
                EvidenceReference(
                    reference=f"action-observation:{observation_id}",
                    evidence_type=evidence_type,
                    observation_code=observation_code,
                )
                for evidence_type in evidence_types
            ),
        )

    def record_decision_outcome(
        self,
        decision_id: str,
        *,
        description: str,
        utility: float,
        success: bool,
    ) -> DecisionRecord:
        event = current_agent_event()
        record = self.decision_store.record_outcome(
            decision_id,
            description=description,
            utility=utility,
            success=success,
            observed_event_id=None if event is None else event.event_id,
            observed_event_sequence=None
            if event is None
            else event.processing_sequence,
        )
        action_intents = (
            ()
            if self._action_execution is None
            else self._action_execution.list_intents()
        )
        if not any(
            intent.provenance.decision_id == decision_id for intent in action_intents
        ):
            record = self._apply_metacognitive_outcome(record)
            source_experiences = tuple(
                item
                for item in self.experience_store.list_records()
                if f"decision:{decision_id}" in item.result_refs.get("decision", ())
            )
            proposals = self.value_system.proposals_from_decision_outcome(
                record,
                experience_ids=tuple(item.experience_id for item in source_experiences),
            )
            updates = self.value_system.apply(proposals)
            for experience in source_experiences:
                for update in updates:
                    self.link_experience_result(
                        experience.experience_id,
                        kind="value",
                        reference=f"value:{update.value_id}@{update.after['revision']}",
                        evidence_refs=update.evidence_ids,
                    )
            self.value_system.record_reassessment(record, updates)
            self._persist_value_state()
            self._sync_self_references()
            self._persist_self_model_state()
            self._persist_metacognition_state()
        self._persist_decision_state()
        return record

    def record_verified_action_experience(self, intent_id: str) -> ExperienceRecord:
        if current_agent_event() is None:
            raise RuntimeError("Action Experience integration requires AgentRuntime")
        execution = self._action_execution
        if execution is None:
            raise ValueError("Action execution layer is unavailable")
        intent = execution.get_intent(intent_id)
        if intent.receipt_id is None:
            raise ValueError("Action intent has no verified receipt")
        receipt = execution.get_receipt(intent.receipt_id)
        if receipt.observation_id is None or receipt.verification_id is None:
            raise ValueError("Action receipt has no verified observation")
        observation = execution.get_observation(receipt.observation_id)
        verification = next(
            item
            for item in execution.list_verifications()
            if item.verification_id == receipt.verification_id
        )
        decision = self.decision_store.get(intent.provenance.decision_id)
        context_id = decision.context_id or f"action:{decision.decision_id}"
        context = (
            None
            if decision.context_id is None
            else self.context_registry.get(decision.context_id)
        )
        event = current_agent_event()
        assert event is not None
        success = verification.success
        appraisal = ExperienceAppraisal(
            valence=0.35 if success else -0.45,
            arousal=0.5 if success else 0.75,
            novelty=None,
            novelty_valid=False,
            goal_progress=1.0 if success else -1.0,
            threat=0.0 if success else 0.7,
            controllability=0.8,
            certainty=1.0,
            social_relevance=0.0
            if context is None or not context.participant_ids
            else 0.6,
            effort_cost=min(
                1.0, intent.cost_units_used / max(1, intent.budget.max_cost_units)
            ),
            reason_codes=(
                "verified_action_outcome",
                "action_succeeded" if success else "action_failed",
                verification.reason,
            ),
        )
        proposal = build_observation_experience(
            source_event_id=event.event_id,
            source_event_sequence=event.processing_sequence,
            external_observation_refs=(
                f"observation:{observation.observation_id}",
                f"outcome-verification:{verification.verification_id}",
            ),
            subject_action_refs=(
                f"action-intent:{intent.intent_id}@{intent.revision}",
            ),
            identity_origin=new_identity_origin(
                OriginActor.SELF,
                OriginInputKind.OBSERVATION,
                source_ref=f"observation:{observation.observation_id}",
                event_id=event.event_id,
                event_sequence=event.processing_sequence,
            ),
            context_id=context_id,
            interlocutor_ids=() if context is None else context.participant_ids,
            situation_codes=("verified_action_observation", intent.tool_name),
            interpretation_codes=appraisal.reason_codes,
            appraisal=appraisal,
            prediction_error=None,
            value_revision_refs={
                value.value_id: value.revision
                for value in self.value_system.active_values()
            },
            active_goal_refs=tuple(
                goal.goal_id for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)
            ),
            self_model_revision=self.self_model.state.revision,
            result_refs={"decision": (f"decision:{decision.decision_id}",)},
        )
        integration = self.experience_coordinator.integrate(
            proposal,
            active_commitment_refs=tuple(
                f"commitment:{item.commitment_id}"
                for item in self.commitment_store.list_commitments()
                if item.status in ACCEPTED_COMMITMENT_STATUSES
            ),
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            observe_motivation=False,
        )
        experience = integration.experience
        if integration.narrative_episode is not None:
            self._sync_self_references()
            self._persist_self_model_state()
            experience = self.link_experience_result(
                experience.experience_id,
                kind="self_model",
                reference=f"self-model@{self.self_model.state.revision}",
                evidence_refs=(f"experience:{experience.experience_id}",),
            )
        self.working_memory.admit(
            working_memory_item(
                item_id=f"experience:{experience.experience_id}",
                kind=WorkingMemoryKind.EPISODIC,
                content="Structured verified action outcome: "
                + ", ".join(experience.interpretation_codes),
                source_event_id=event.event_id,
                source_event_sequence=event.processing_sequence,
                context_id=context_id,
                source=event.source,
                activation=0.5 + 0.5 * experience.subjective_salience,
                salience=experience.subjective_salience,
                retention_reason=RetentionReason.RECENT_CONTEXT,
            )
        )
        self.refresh_attention(compete=True)
        return experience

    def attribute_action_outcome(self, intent_id: str) -> AgencyAttribution:
        if current_agent_event() is None:
            raise RuntimeError("Agency attribution application requires AgentRuntime")
        execution = self._action_execution
        if execution is None:
            raise ValueError("Action execution layer is unavailable")
        existing = self.agency_attribution_store.current_for_intent(intent_id)
        if existing is not None:
            return existing
        intent = execution.get_intent(intent_id)
        if intent.receipt_id is None:
            raise ValueError("Action intent has no execution receipt")
        receipt = execution.get_receipt(intent.receipt_id)
        if receipt.observation_id is None or receipt.verification_id is None:
            raise ValueError("Action outcome has not been autonomously verified")
        observation = execution.get_observation(receipt.observation_id)
        verification = next(
            (
                item
                for item in execution.list_verifications()
                if item.verification_id == receipt.verification_id
            ),
            None,
        )
        if verification is None:
            raise ValueError("Action outcome verification is missing")
        decision = self.decision_store.get(intent.provenance.decision_id)
        decision_context = (
            None
            if decision.context_id is None
            else self.context_registry.get(decision.context_id)
        )
        self_share = 0.25 if verification.success else 0.3
        participant_ids = (
            () if decision_context is None else decision_context.participant_ids
        )
        environment_share = (
            (0.45 if verification.success else 0.4)
            if participant_ids
            else (0.65 if verification.success else 0.6)
        )
        contributors = [
            CausalContributor(
                kind=CausalContributorKind.SELF,
                causal_share=self_share,
                confidence=0.9,
                controllability=0.8,
                foreseeability=0.7,
                responsibility_share=self_share,
            ),
            CausalContributor(
                kind=CausalContributorKind.ENVIRONMENT,
                causal_share=environment_share,
                confidence=0.8,
                controllability=0.1,
                foreseeability=0.5,
                responsibility_share=0.0,
            ),
            *(
                [
                    CausalContributor(
                        kind=CausalContributorKind.OTHER,
                        contributor_ref=participant_ids[0],
                        causal_share=0.2,
                        confidence=0.7,
                        controllability=0.3,
                        foreseeability=0.6,
                        responsibility_share=0.1,
                    )
                ]
                if participant_ids
                else []
            ),
            CausalContributor(
                kind=CausalContributorKind.CHANCE,
                causal_share=0.1,
                confidence=0.5,
                controllability=0.0,
                foreseeability=0.1,
                responsibility_share=0.0,
            ),
        ]
        attribution = self.agency_attribution_store.create(
            decision_id=intent.provenance.decision_id,
            action_intent_id=intent.intent_id,
            execution_receipt_id=receipt.receipt_id,
            observation_id=observation.observation_id,
            outcome_ref=f"decision:{intent.provenance.decision_id}:outcome",
            contributors=tuple(contributors),
            intended=verification.success,
            uncertainty=0.2,
            evidence_refs=(
                f"decision:{intent.provenance.decision_id}",
                f"action-intent:{intent.intent_id}@{intent.revision}",
                f"execution-receipt:{receipt.receipt_id}",
                f"observation:{observation.observation_id}",
                f"outcome-verification:{verification.verification_id}",
            ),
            reason_codes=("autonomous_structured_outcome_verification",),
        )
        self._apply_attribution(attribution)
        source_experience = (
            None
            if decision.context_id is None
            else self.experience_store.latest_for_context(decision.context_id)
        )
        if source_experience is not None:
            belief = self.propose_belief_from_experience(
                source_experience.experience_id,
                proposition=Proposition.create(
                    "A verified autonomous action produced an observed outcome",
                    subject=f"action:{intent.intent_id}",
                    predicate="produced",
                    object=f"observation:{observation.observation_id}",
                ),
                source_trust=0.8,
                confidence=0.7,
                context_scope=(decision.context_id,),
                belief_id=f"action-outcome:{decision.decision_id}",
            )
            self.resolve_belief(
                belief.belief_id,
                accept=True,
                confidence=0.85,
                epistemic_status=EpistemicStatus.PROBABLE,
                reason_code="verified_action_observation",
                evidence_refs=(
                    f"outcome-verification:{verification.verification_id}",
                    f"observation:{observation.observation_id}",
                ),
            )
        return attribution

    def revise_agency_attribution(
        self,
        attribution_id: str,
        *,
        expected_revision: int,
        contributors: tuple[CausalContributor, ...],
        intended: bool,
        uncertainty: float,
        evidence_refs: tuple[str, ...],
        reason_code: str,
    ) -> AgencyAttribution:
        if current_agent_event() is None:
            raise RuntimeError("Agency attribution revisions require AgentRuntime")
        attribution = self.agency_attribution_store.revise(
            attribution_id,
            expected_revision=expected_revision,
            contributors=contributors,
            intended=intended,
            uncertainty=uncertainty,
            evidence_refs=evidence_refs,
            reason_code=reason_code,
        )
        self._apply_attribution(attribution)
        return attribution

    def simulate_counterfactual(self, attribution_id: str) -> CounterfactualSimulation:
        if current_agent_event() is None:
            raise RuntimeError("Counterfactual simulation requires AgentRuntime")
        attribution = self.agency_attribution_store.history(attribution_id)[-1]
        decision = self.decision_store.get(attribution.decision_id)
        outcome = decision.actual_outcome
        if outcome is None:
            raise ValueError("Counterfactual simulation requires an observed outcome")
        alternatives = tuple(
            AlternativeOutcome(
                candidate_id=item.candidate.candidate_id,
                candidate_type=item.candidate.candidate_type.value,
                plausible_utility=max(-1.0, min(1.0, item.predicted_utility)),
                confidence=min(
                    0.6,
                    (1.0 - item.candidate.uncertainty)
                    * (1.0 - attribution.uncertainty)
                    * 0.6,
                ),
                evidence_status=EvidenceStatus.SPECULATIVE,
                assumption_codes=("decision_time_prediction_held",),
                evidence_refs=(
                    f"decision:{decision.decision_id}:candidate:{item.candidate.candidate_id}",
                ),
            )
            for item in decision.considered_candidates
            if item.eligible
            and item.candidate.candidate_id != decision.selected_candidate_id
            and item.candidate.predicted_outcomes
        )
        if not alternatives:
            raise ValueError("Decision has no valid counterfactual alternatives")
        best = max(
            alternatives,
            key=lambda item: (item.plausible_utility, item.candidate_id),
        )
        gap = best.plausible_utility - outcome.utility
        confidence = max(item.confidence for item in alternatives)
        if gap > 0.05:
            signal = (
                CounterfactualSignal.REGRET
                if outcome.utility < 0.0
                else CounterfactualSignal.MISSED_OPPORTUNITY
            )
        elif gap < -0.05:
            signal = CounterfactualSignal.RELIEF
        else:
            signal = CounterfactualSignal.NONE
        magnitude = (
            0.0
            if signal == CounterfactualSignal.NONE
            else min(0.5, abs(gap) * confidence)
        )
        evidence_refs = tuple(
            item
            for item in (
                None
                if outcome.observed_event_id is None
                else f"event:{outcome.observed_event_id}",
                attribution.reference,
                attribution.outcome_ref,
            )
            if item is not None
        )
        existing = self.counterfactual_store.current_for_decision(decision.decision_id)
        if existing is None:
            simulation = self.counterfactual_store.create(
                decision_id=decision.decision_id,
                selected_candidate_id=decision.selected_candidate_id,
                action_intent_id=attribution.action_intent_id,
                execution_receipt_id=attribution.execution_receipt_id,
                outcome_ref=attribution.outcome_ref,
                agency_attribution_id=attribution.attribution_id,
                agency_attribution_revision=attribution.revision,
                observed_utility=outcome.utility,
                alternatives=alternatives,
                signal=signal,
                signal_magnitude=magnitude,
                confidence=confidence,
                evidence_refs=evidence_refs,
                reason_codes=("bounded_decision_alternative_comparison",),
            )
        elif existing.agency_attribution_revision == attribution.revision:
            return existing
        else:
            simulation = self.counterfactual_store.revise(
                existing.simulation_id,
                expected_revision=existing.revision,
                agency_attribution_revision=attribution.revision,
                alternatives=alternatives,
                signal=signal,
                signal_magnitude=magnitude,
                confidence=confidence,
                evidence_refs=(attribution.reference,),
                reason_code="agency_attribution_revised",
            )
        self._apply_counterfactual(simulation)
        return simulation

    def revise_counterfactual(
        self,
        simulation_id: str,
        *,
        expected_revision: int,
        agency_attribution_revision: int,
        alternatives: tuple[AlternativeOutcome, ...],
        signal: CounterfactualSignal,
        signal_magnitude: float,
        confidence: float,
        evidence_refs: tuple[str, ...],
        reason_code: str,
    ) -> CounterfactualSimulation:
        if current_agent_event() is None:
            raise RuntimeError("Counterfactual revisions require AgentRuntime")
        simulation = self.counterfactual_store.revise(
            simulation_id,
            expected_revision=expected_revision,
            agency_attribution_revision=agency_attribution_revision,
            alternatives=alternatives,
            signal=signal,
            signal_magnitude=signal_magnitude,
            confidence=confidence,
            evidence_refs=evidence_refs,
            reason_code=reason_code,
        )
        self._apply_counterfactual(simulation)
        return simulation

    def _apply_counterfactual(self, simulation: CounterfactualSimulation) -> None:
        decision = self.decision_store.get(simulation.decision_id)
        selected = next(
            item.candidate
            for item in decision.considered_candidates
            if item.candidate.candidate_id == decision.selected_candidate_id
        )
        history = self.counterfactual_store.history(simulation.simulation_id)
        previous = history[-2] if simulation.revision > 1 else None

        def comparison(
            record: CounterfactualSimulation,
        ) -> tuple[AlternativeOutcome, ActionCandidate, float, float]:
            best_outcome = max(
                record.alternatives,
                key=lambda item: (item.plausible_utility, item.candidate_id),
            )
            candidate = next(
                item.candidate
                for item in decision.considered_candidates
                if item.candidate.candidate_id == best_outcome.candidate_id
            )
            direction = -1.0 if record.signal == CounterfactualSignal.RELIEF else 1.0
            if record.signal == CounterfactualSignal.NONE:
                direction = 0.0
            amount = min(0.05, record.signal_magnitude * record.confidence * 0.2)
            return best_outcome, candidate, direction, amount

        best, alternative, preference, learning = comparison(simulation)
        old_comparison = None if previous is None else comparison(previous)
        selected_subject = f"candidate-type:{selected.candidate_type.value}"

        def policy_desired(
            values: tuple[AlternativeOutcome, ActionCandidate, float, float] | None,
        ) -> dict[str, float]:
            if values is None:
                return {}
            _, candidate, direction, amount = values
            desired = {selected_subject: -direction * amount}
            subject = f"candidate-type:{candidate.candidate_type.value}"
            desired[subject] = desired.get(subject, 0.0) + direction * amount
            return desired

        desired_policy = policy_desired((best, alternative, preference, learning))
        old_policy = policy_desired(old_comparison)
        for subject in set(desired_policy) | set(old_policy) or {selected_subject}:
            self.counterfactual_store.record_projection(
                simulation,
                CounterfactualTarget.DECISION_CALIBRATION,
                subject_ref=subject,
                applied_delta=desired_policy.get(subject, 0.0)
                - old_policy.get(subject, 0.0),
                evidence_refs=(simulation.outcome_ref, best.evidence_refs[0]),
            )

        def value_desired(
            record: CounterfactualSimulation,
            values: tuple[AlternativeOutcome, ActionCandidate, float, float],
        ) -> dict[str, float]:
            _, candidate, direction, _ = values
            return {
                value_id: direction
                * (
                    candidate.value_effects.get(value_id, 0.0)
                    - selected.value_effects.get(value_id, 0.0)
                )
                * min(0.2, record.signal_magnitude)
                for value_id in set(selected.value_effects)
                | set(candidate.value_effects)
            }

        desired_values = value_desired(
            simulation, (best, alternative, preference, learning)
        )
        old_values = (
            {}
            if previous is None or old_comparison is None
            else value_desired(previous, old_comparison)
        )
        impacts = {
            value_id: desired_values.get(value_id, 0.0) - old_values.get(value_id, 0.0)
            for value_id in set(desired_values) | set(old_values)
        }
        desired_affect = (
            -preference * simulation.signal_magnitude * simulation.confidence
        )
        old_affect = (
            0.0
            if previous is None or old_comparison is None
            else -old_comparison[2] * previous.signal_magnitude * previous.confidence
        )
        affect_correction = desired_affect - old_affect
        appraisal = AppraisalResult(
            novelty=None,
            goal_progress=max(-1.0, min(1.0, affect_correction)),
            threat=max(0.0, -affect_correction),
            controllability=0.0,
            certainty=simulation.confidence,
            social_relevance=0.0,
            effort_cost=0.0,
            novelty_valid=False,
            reasons=("bounded_counterfactual", simulation.reference),
        )
        updates = self.apply_value_impacts(
            appraisal,
            impacts,
            kind=ValueUpdateKind.REFLECTION,
            source="counterfactual.simulation",
            proposal_id=simulation.reference,
            decision_id=decision.decision_id,
            context_id=decision.context_id,
        )
        for update in updates:
            self.counterfactual_store.record_projection(
                simulation,
                CounterfactualTarget.VALUE,
                subject_ref=f"value:{update.value_id}",
                applied_delta=update.applied_delta,
                evidence_refs=update.evidence_ids or (simulation.outcome_ref,),
            )
        if not updates:
            self.counterfactual_store.record_projection(
                simulation,
                CounterfactualTarget.VALUE,
                subject_ref="value:none",
                applied_delta=0.0,
                evidence_refs=(simulation.outcome_ref,),
            )
        if simulation.signal == CounterfactualSignal.NONE:
            motivation_ref = "motivation:none"
        else:
            motivation = self.motivation_dynamics.observe_structured_signal(
                MotivationKind.DESIRE if preference >= 0 else MotivationKind.AVERSION,
                MotivationSource.LEARNING,
                f"decision:{decision.decision_id}:candidate:{best.candidate_id}",
                signal=min(0.25, simulation.signal_magnitude),
                uncertainty=1.0 - simulation.confidence,
                source_refs=(simulation.reference,),
                value_ids=tuple(impacts),
            )
            motivation_ref = f"motivation:{motivation.motivation_id}"
        self.counterfactual_store.record_projection(
            simulation,
            CounterfactualTarget.MOTIVATION,
            subject_ref=motivation_ref,
            applied_delta=0.0,
            evidence_refs=(simulation.reference,),
        )

        def alternative_desired(
            values: tuple[AlternativeOutcome, ActionCandidate, float, float] | None,
            *,
            prefix: str,
            scale: float,
        ) -> dict[str, float]:
            if values is None:
                return {}
            _, candidate, direction, amount = values
            return {
                f"{prefix}:{candidate.candidate_type.value}": direction * amount * scale
            }

        for target, prefix, scale in (
            (CounterfactualTarget.PLAN_STRATEGY, "plan-strategy", 1.0),
            (CounterfactualTarget.METACOGNITION, "candidate-type", 0.5),
        ):
            desired = alternative_desired(
                (best, alternative, preference, learning), prefix=prefix, scale=scale
            )
            old_desired = alternative_desired(
                old_comparison, prefix=prefix, scale=scale
            )
            for subject in set(desired) | set(old_desired) or {f"{prefix}:none"}:
                self.counterfactual_store.record_projection(
                    simulation,
                    target,
                    subject_ref=subject,
                    applied_delta=desired.get(subject, 0.0)
                    - old_desired.get(subject, 0.0),
                    evidence_refs=(best.evidence_refs[0],),
                )
        previous_valence = self.emotion_engine.state.valence
        emotion = self.emotion_engine.update_from_appraisal(appraisal, max_delta=0.03)
        self.counterfactual_store.record_projection(
            simulation,
            CounterfactualTarget.EMOTION,
            subject_ref="emotion:valence",
            applied_delta=emotion.state.valence - previous_valence,
            evidence_refs=(simulation.outcome_ref,),
        )
        self._persist_motivation_state()
        self._persist_metacognition_state()

    def _apply_attribution(self, attribution: AgencyAttribution) -> None:
        decision = self.decision_store.get(attribution.decision_id)
        outcome = decision.actual_outcome
        if outcome is None:
            raise ValueError("Agency attribution requires an outcome")
        attribution_ref = attribution.reference
        self_contribution = attribution.contribution(CausalContributorKind.SELF)
        weighted_controllability = sum(
            item.causal_share * item.controllability
            for item in attribution.contributors
        )
        has_other = any(
            item.kind == CausalContributorKind.OTHER
            for item in attribution.contributors
        )
        appraisal = AppraisalResult(
            novelty=None,
            goal_progress=outcome.utility,
            threat=max(0.0, -outcome.utility) * (1.0 - weighted_controllability),
            controllability=weighted_controllability,
            certainty=1.0 - attribution.uncertainty,
            social_relevance=1.0 if has_other else 0.0,
            effort_cost=0.0,
            novelty_valid=False,
            reasons=("agency_attribution", attribution_ref),
        )
        selected = next(
            item.candidate
            for item in decision.considered_candidates
            if item.candidate.candidate_id == decision.selected_candidate_id
        )
        impacts = {
            value_id: effect * outcome.utility * self_contribution
            for value_id, effect in selected.value_effects.items()
        }
        value_updates = self.apply_value_impacts(
            appraisal,
            impacts,
            kind=ValueUpdateKind.OUTCOME,
            source="agency.attribution",
            proposal_id=attribution_ref,
            decision_id=decision.decision_id,
            context_id=decision.context_id,
            experience_ids=tuple(
                item.experience_id
                for item in self.experience_store.list_records()
                if f"decision:{decision.decision_id}"
                in item.result_refs.get("decision", ())
            ),
        )
        for experience in self.experience_store.list_records():
            if f"decision:{decision.decision_id}" not in experience.result_refs.get(
                "decision", ()
            ):
                continue
            for update in value_updates:
                self.link_experience_result(
                    experience.experience_id,
                    kind="value",
                    reference=f"value:{update.value_id}@{update.after['revision']}",
                    evidence_refs=update.evidence_ids,
                )
        self.agency_attribution_store.record_projection(
            attribution,
            AttributionTarget.VALUE,
            applied_delta=sum(item.applied_delta for item in value_updates),
            evidence_refs=tuple(
                evidence_id
                for item in value_updates
                for evidence_id in item.evidence_ids
            )
            or (attribution.outcome_ref,),
        )
        before_assessment = decision.metacognition_post_assessment_id
        decision = self._apply_metacognitive_outcome(
            decision,
            attribution_ref=attribution_ref,
            self_contribution=self_contribution,
            controllability=weighted_controllability,
        )
        assessment_ref = (
            decision.metacognition_post_assessment_id
            if decision.metacognition_post_assessment_id is not None
            and decision.metacognition_post_assessment_id != before_assessment
            else attribution.outcome_ref
        )
        self.agency_attribution_store.record_projection(
            attribution,
            AttributionTarget.METACOGNITION,
            applied_delta=0.0,
            evidence_refs=(assessment_ref,),
        )
        link_id = (
            f"agency-continuity:{attribution.attribution_id}:{attribution.revision}"
        )
        if link_id not in self.narrative_self.continuity_links:
            self.narrative_self.link_continuity(
                f"decision:{decision.decision_id}",
                attribution_ref,
                relation_code="causal_attribution",
                evidence_refs=(attribution.outcome_ref,),
                confidence=1.0 - attribution.uncertainty,
                link_id=link_id,
            )
        self.agency_attribution_store.record_projection(
            attribution,
            AttributionTarget.NARRATIVE_SELF,
            applied_delta=0.0,
            evidence_refs=(link_id,),
        )
        relationship_refs: list[str] = []
        for contributor in attribution.contributors:
            if (
                contributor.kind != CausalContributorKind.OTHER
                or contributor.contributor_ref is None
            ):
                continue
            relationship = self.relationship_store.for_interlocutor(
                contributor.contributor_ref
            )
            if relationship is None:
                continue
            updated = self.relationship_store.correct(
                relationship.relationship_id,
                reason="agency_attribution_evidence",
                evidence_refs=(attribution_ref,),
                uncertainty=max(
                    0.0,
                    relationship.uncertainty
                    - min(
                        0.05, contributor.causal_share * contributor.confidence * 0.05
                    ),
                ),
            )
            relationship_refs.append(
                f"relationship:{updated.relationship_id}@{updated.revision}"
            )
        self.agency_attribution_store.record_projection(
            attribution,
            AttributionTarget.RELATIONSHIP,
            applied_delta=0.0,
            evidence_refs=tuple(relationship_refs) or (attribution_ref,),
        )
        previous_valence = self.emotion_engine.state.valence
        emotion = self.emotion_engine.update_from_appraisal(appraisal, max_delta=0.05)
        self.agency_attribution_store.record_projection(
            attribution,
            AttributionTarget.EMOTION_APPRAISAL,
            applied_delta=emotion.state.valence - previous_valence,
            evidence_refs=(attribution.outcome_ref,),
        )
        motivation = self.motivation_dynamics.observe_structured_signal(
            MotivationKind.DRIVE if not outcome.success else MotivationKind.DESIRE,
            MotivationSource.DELIBERATION,
            f"decision:{decision.decision_id}",
            signal=min(0.5, abs(outcome.utility) * (0.25 + self_contribution)),
            uncertainty=attribution.uncertainty,
            source_refs=(attribution_ref,),
            value_ids=tuple(selected.value_effects),
        )
        self.agency_attribution_store.record_projection(
            attribution,
            AttributionTarget.MOTIVATION,
            applied_delta=0.0,
            evidence_refs=(
                f"motivation:{motivation.motivation_id}@{motivation.revision}",
            ),
        )
        self._persist_decision_state()
        self._persist_metacognition_state()
        self._persist_self_model_state()
        self._persist_narrative_self_state()
        self._persist_motivation_state()

    def record_decision_compensation(
        self, decision_id: str, *, receipt_id: str
    ) -> DecisionRecord:
        event = current_agent_event()
        record = self.decision_store.record_compensation(
            decision_id,
            receipt_id=receipt_id,
            observed_event_id=None if event is None else event.event_id,
            observed_event_sequence=None
            if event is None
            else event.processing_sequence,
        )
        self._persist_decision_state()
        return record

    def _metacognitive_candidate_scores(
        self, candidates: tuple[ActionCandidate, ...], recommended_action: ActionType
    ) -> dict[str, dict[str, float]]:
        scores: dict[str, dict[str, float]] = {
            candidate.candidate_id: {} for candidate in candidates
        }
        for candidate in candidates:
            contribution = 0.0
            if candidate.candidate_type == recommended_action:
                contribution = 0.75
            elif recommended_action.value in {
                "defer",
                "request_information",
                "delegate",
                "observe",
            } and candidate.candidate_type.value in {"respond", "internal"}:
                contribution = -0.75
            scores.setdefault(candidate.candidate_id, {})[
                "metacognition:recommended_action"
            ] = contribution
        return scores

    def _calibrated_candidate_scores(
        self, candidates: tuple[ActionCandidate, ...], recommended_action: ActionType
    ) -> dict[str, dict[str, float]]:
        scores = self._metacognitive_candidate_scores(candidates, recommended_action)
        for candidate in candidates:
            subject = f"candidate-type:{candidate.candidate_type.value}"
            decision_delta = self.counterfactual_store.calibration(
                CounterfactualTarget.DECISION_CALIBRATION, subject
            )
            metacognitive_delta = self.counterfactual_store.calibration(
                CounterfactualTarget.METACOGNITION, subject
            )
            if decision_delta:
                scores[candidate.candidate_id][
                    "counterfactual:decision_calibration"
                ] = decision_delta
            if metacognitive_delta:
                scores[candidate.candidate_id]["counterfactual:metacognition"] = (
                    metacognitive_delta
                )
            if candidate.plan_id is not None:
                strategy_delta = self.counterfactual_store.calibration(
                    CounterfactualTarget.PLAN_STRATEGY,
                    f"plan-strategy:{candidate.candidate_type.value}",
                )
                if strategy_delta:
                    scores[candidate.candidate_id]["counterfactual:plan_strategy"] = (
                        strategy_delta
                    )
        return scores

    def _current_cognitive_quality(self) -> CognitiveQuality:
        emotion = self.emotion_engine.state
        return self.metacognition.current_quality(
            cognitive_load=min(
                1.0, len(self.working_memory.items) / self.working_memory.item_capacity
            ),
            attention_saturation=min(
                1.0,
                len(self.attention_system.focus.candidate_ids)
                / self.attention_system.capacity,
            ),
            emotion_valence=emotion.valence,
            emotion_arousal=emotion.arousal,
            provenance_refs=(
                f"working-memory:occupancy:{len(self.working_memory.items)}",
                f"attention-focus@{self.attention_system.focus.revision}",
                "emotion:current",
            ),
        )

    def _apply_metacognitive_outcome(
        self,
        record: DecisionRecord,
        *,
        attribution_ref: str | None = None,
        self_contribution: float = 1.0,
        controllability: float = 1.0,
    ) -> DecisionRecord:
        if record.metacognition_pre_assessment_id is None:
            return record
        selected = next(
            item.candidate
            for item in record.considered_candidates
            if item.candidate.candidate_id == record.selected_candidate_id
        )
        capability_ids = string_values(selected.parameters.get("capability_ids"))
        if (
            record.actual_outcome is not None
            and not record.actual_outcome.success
            and self_contribution >= 0.5
        ):
            for capability_id in capability_ids:
                current = self.self_model.state.capabilities.get(capability_id)
                self.self_model.update_capability_from_decision(
                    capability_id,
                    capability_id if current is None else current.description,
                    record,
                    tags=string_values(selected.parameters.get("topic_tags")),
                )
        assessment = self.metacognition.assess_post(
            record,
            self_model_revision=self.self_model.state.revision,
            cognitive_quality=self._current_cognitive_quality(),
            attribution_ref=attribution_ref,
            self_contribution=self_contribution,
            controllability=controllability,
        )
        record = self.decision_store.link_post_assessment(
            record.decision_id, assessment.assessment_id
        )
        self._sync_metacognitive_narrative()
        self._persist_self_model_state()
        self._persist_narrative_self_state()
        return record

    def _sync_metacognitive_narrative(self) -> None:
        claim_prefix = "narrative:metacognitive-hypothesis:"
        active_claim_ids = {
            f"narrative:{item.hypothesis_id}"
            for item in self.metacognition.hypotheses.values()
        }
        for claim in tuple(self.narrative_self.claims.values()):
            if (
                claim.claim_id.startswith(claim_prefix)
                and claim.claim_id not in active_claim_ids
                and claim.status != IdentityClaimStatus.REVISED
            ):
                self.narrative_self.revise_claim(
                    claim.claim_id,
                    confidence=min(claim.confidence, 0.49),
                    reason_code="metacognitive_hypothesis_retired",
                    counterevidence_refs=(
                        "metacognition:outcome-withdrawn-or-revised",
                    ),
                    status=IdentityClaimStatus.REVISED,
                )
        for hypothesis in self.metacognition.hypotheses.values():
            claim_id = f"narrative:{hypothesis.hypothesis_id}"
            current = self.narrative_self.claims.get(claim_id)
            if current is None:
                self.narrative_self.propose_claim(
                    kind=IdentityClaimKind.LIMITATION,
                    statement=f"Recurring {hypothesis.hypothesis_code} in {hypothesis.scope_id}",
                    polarity=-1,
                    theme_codes=(hypothesis.scope_id, hypothesis.hypothesis_code),
                    confidence=hypothesis.confidence,
                    stability=0.4,
                    evidence_refs=hypothesis.evidence_refs,
                    related_decision_refs=tuple(
                        ref.split(":", 2)[1] for ref in hypothesis.evidence_refs
                    ),
                    claim_id=claim_id,
                )
            elif set(hypothesis.evidence_refs) - set(current.evidence_refs):
                self.narrative_self.revise_claim(
                    claim_id,
                    confidence=hypothesis.confidence,
                    reason_code="metacognitive_outcome_update",
                    evidence_refs=tuple(
                        ref
                        for ref in hypothesis.evidence_refs
                        if ref not in current.evidence_refs
                    ),
                    status=IdentityClaimStatus.HYPOTHESIS,
                )
