from dataclasses import asdict, replace
from datetime import UTC, datetime
from uuid import uuid4

from kagya.agency import (
    AGENCY_ATTRIBUTION_STATE_KEY,
    AgencyAttribution,
    AgencyAttributionStore,
)
from kagya.counterfactual import (
    COUNTERFACTUAL_STATE_KEY,
    CounterfactualSimulation,
    CounterfactualStore,
)
from kagya.decision import (
    ActionCandidate,
    DecisionDatasetGenerator,
    DecisionDatasetRecord,
    DecisionRecord,
    DecisionStatus,
    DecisionStore,
)
from kagya.feedback import (
    FeedbackPropagation,
    FeedbackProvenance,
    FeedbackRecord,
    FeedbackRevision,
    FeedbackSignal,
    FeedbackStatus,
    FeedbackTarget,
    FeedbackTargetType,
    TrainingDisposition,
    ValueEvidenceProposal,
    feedback_fingerprint,
    normalize_signals,
)
from kagya.memory import MemoryLifecycleStatus
from kagya.memory import ValidationStatus
from kagya.motivation import (
    GoalStatus,
)
from kagya.planning import (
    EvidenceReference,
    Plan,
    PlanCandidate,
    PlanStatus,
    PlanStore,
    StepStatus,
)
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.coordinators._shared import RuntimeDomainMixin, string_values

from typing import Callable


class PlanDecisionCoordinator(RuntimeDomainMixin):
    def __init__(
        self,
        plans: PlanStore,
        decisions: DecisionStore,
        *,
        persist: Callable[[], None],
    ) -> None:
        self._plans = plans
        self._decisions = decisions
        self._persist = persist

    def list_decisions(
        self, status: DecisionStatus | None = None
    ) -> list[DecisionRecord]:
        return self._decisions.list_records(status)

    def create_plan(self, candidate: PlanCandidate, *, actor_id: str) -> Plan:
        if not hasattr(self, "goal_manager"):
            plan = self._plans.create(candidate, actor_id=actor_id)
            self._persist()
            return plan
        goal = self.goal_manager.get(candidate.goal_id)
        if goal.status in {
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        }:
            raise ValueError("Terminal Goal cannot receive a Plan")
        plan = self.plan_decision_coordinator.create_plan(candidate, actor_id=actor_id)
        self._sync_plan_working_memory()
        return plan

    def activate_plan(self, plan_id: str) -> Plan:
        plan = self.plan_store.get(plan_id)
        if self.goal_manager.get(plan.goal_id).status != GoalStatus.ACTIVE:
            raise ValueError("Plan activation requires an active Goal")
        event = current_agent_event()
        plan = self.plan_store.activate(
            plan_id,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def revise_plan(
        self,
        plan_id: str,
        candidate: PlanCandidate,
        *,
        expected_revision: int,
        reason_code: str,
        actor_id: str,
        evidence_refs: tuple[str, ...] = (),
    ) -> Plan:
        event = current_agent_event()
        plan = self.plan_store.revise(
            plan_id,
            candidate,
            expected_revision=expected_revision,
            reason_code=reason_code,
            actor_id=actor_id,
            evidence_refs=evidence_refs,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def start_plan_step(self, plan_id: str, step_id: str) -> Plan:
        self._require_active_plan_goal(plan_id)
        event = current_agent_event()
        plan = self.plan_store.start_step(
            plan_id,
            step_id,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def complete_plan_step(
        self,
        plan_id: str,
        step_id: str,
        evidence: tuple[EvidenceReference, ...],
    ) -> Plan:
        self._require_active_plan_goal(plan_id)
        event = current_agent_event()
        plan = self.plan_store.complete_step(
            plan_id,
            step_id,
            evidence,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        if plan.status == PlanStatus.COMPLETED:
            goal = self.goal_manager.get(plan.goal_id)
            if goal.status != GoalStatus.ACTIVE:
                raise ValueError("Completed Plan is inconsistent with Goal lifecycle")
            self.transition_goal(
                plan.goal_id,
                GoalStatus.COMPLETED,
                reason="plan_success_condition_verified",
                outcome=f"plan:{plan.plan_id}@{plan.revision}",
            )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def fail_plan_step(
        self,
        plan_id: str,
        step_id: str,
        *,
        reason_code: str,
        evidence: tuple[EvidenceReference, ...] = (),
    ) -> Plan:
        self._require_active_plan_goal(plan_id)
        event = current_agent_event()
        plan = self.plan_store.fail_step(
            plan_id,
            step_id,
            reason_code=reason_code,
            evidence=evidence,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def retry_plan_step(self, plan_id: str, step_id: str) -> Plan:
        return self.start_plan_step(plan_id, step_id)

    def timeout_plan_step(self, plan_id: str, step_id: str) -> Plan:
        event = current_agent_event()
        reference = (
            f"event:{event.event_id}"
            if event is not None
            else f"step:{step_id}:timeout"
        )
        return self.fail_plan_step(
            plan_id,
            step_id,
            reason_code="step_timeout",
            evidence=(
                EvidenceReference(
                    reference=reference,
                    evidence_type="failure",
                    observation_code="step_timeout",
                ),
            ),
        )

    def abandon_plan(
        self,
        plan_id: str,
        evidence: tuple[EvidenceReference, ...],
    ) -> Plan:
        event = current_agent_event()
        plan = self.plan_store.abandon(
            plan_id,
            evidence,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_motivation_state()
        self._sync_plan_working_memory()
        return plan

    def restore_decision_state(self) -> None:
        payload = self.persistent_state.extensions.get("decision_records", [])
        self.decision_store.restore(payload if isinstance(payload, list) else [])
        self._persist_decision_state()

    def restore_agency_attribution_state(self) -> None:
        self.agency_attribution_store = self._new_agency_attribution_store()

    def restore_counterfactual_state(self) -> None:
        self.counterfactual_store = CounterfactualStore(
            load=lambda: self.persistent_state.extensions.get(COUNTERFACTUAL_STATE_KEY),
            save=lambda payload: self.persistent_state.extensions.__setitem__(
                COUNTERFACTUAL_STATE_KEY, payload
            ),
            validate_chain=self._validate_counterfactual_chain,
        )

    def _new_agency_attribution_store(self) -> AgencyAttributionStore:
        return AgencyAttributionStore(
            load=lambda: self.persistent_state.extensions.get(
                AGENCY_ATTRIBUTION_STATE_KEY
            ),
            save=lambda payload: self.persistent_state.extensions.__setitem__(
                AGENCY_ATTRIBUTION_STATE_KEY, payload
            ),
            validate_chain=self._validate_attribution_chain,
        )

    def _validate_attribution_chain(self, attribution: AgencyAttribution) -> None:
        decision = self.decision_store.get(attribution.decision_id)
        if decision.actual_outcome is None:
            raise ValueError("agency attribution requires a resolved decision outcome")
        if attribution.outcome_ref != f"decision:{decision.decision_id}:outcome":
            raise ValueError("agency attribution outcome provenance does not match")
        execution = self._action_execution
        if execution is None:
            return
        intent = execution.get_intent(attribution.action_intent_id)
        receipt = execution.get_receipt(attribution.execution_receipt_id)
        observation = execution.get_observation(attribution.observation_id)
        if (
            intent.provenance.decision_id != attribution.decision_id
            or receipt.intent_id != intent.intent_id
            or receipt.decision_id != attribution.decision_id
            or receipt.observation_id != observation.observation_id
            or observation.intent_id != intent.intent_id
            or observation.receipt_id != receipt.receipt_id
        ):
            raise ValueError("agency attribution causal provenance does not match")

    def _validate_counterfactual_chain(
        self, simulation: CounterfactualSimulation
    ) -> None:
        decision = self.decision_store.get(simulation.decision_id)
        if decision.actual_outcome is None:
            raise ValueError("counterfactual simulation requires an observed outcome")
        if (
            simulation.selected_candidate_id != decision.selected_candidate_id
            or simulation.observed_utility != decision.actual_outcome.utility
            or simulation.outcome_ref != f"decision:{decision.decision_id}:outcome"
        ):
            raise ValueError("counterfactual simulation cannot replace observed facts")
        considered = {
            item.candidate.candidate_id: item for item in decision.considered_candidates
        }
        if any(
            alternative.candidate_id not in considered
            or not considered[alternative.candidate_id].eligible
            or alternative.candidate_type
            != considered[alternative.candidate_id].candidate.candidate_type.value
            for alternative in simulation.alternatives
        ):
            raise ValueError(
                "counterfactual alternative was not an eligible Decision option"
            )
        attribution = next(
            (
                item
                for item in self.agency_attribution_store.history(
                    simulation.agency_attribution_id
                )
                if item.revision == simulation.agency_attribution_revision
            ),
            None,
        )
        if (
            attribution is None
            or attribution.decision_id != simulation.decision_id
            or attribution.action_intent_id != simulation.action_intent_id
            or attribution.execution_receipt_id != simulation.execution_receipt_id
            or attribution.outcome_ref != simulation.outcome_ref
        ):
            raise ValueError("counterfactual attribution provenance does not match")

    def restore_feedback_state(self) -> None:
        payload = self.persistent_state.extensions.get("feedback")
        self.feedback_store.restore(payload if isinstance(payload, dict) else None)
        self._persist_feedback_state()

    def _persist_feedback_state(self) -> None:
        self.persistent_state.extensions["feedback"] = self.feedback_store.to_json()

    def restore_metacognition_state(self) -> None:
        self.metacognition.restore(
            self.persistent_state.extensions.get("metacognition")
        )
        self._persist_metacognition_state()

    def _persist_metacognition_state(self) -> None:
        self.persistent_state.extensions["metacognition"] = self.metacognition.to_json()

    def submit_feedback(
        self,
        *,
        target: FeedbackTarget,
        signals: tuple[FeedbackSignal, ...],
        idempotency_key: str,
        actor_type: str,
        actor_id: str | None,
        source: str,
        correction: str | None = None,
        expected_answer: str | None = None,
        feedback_id: str | None = None,
    ) -> FeedbackRecord:
        normalized = normalize_signals(signals)
        self._validate_feedback_content(normalized, correction, expected_answer)
        fingerprint = feedback_fingerprint(
            "create",
            {
                "feedback_id": feedback_id,
                "target": asdict(target),
                "signals": [item.value for item in normalized],
                "correction": correction,
                "expected_answer": expected_answer,
                "actor_type": actor_type,
            },
        )
        existing = self.feedback_store.idempotent_result(idempotency_key, fingerprint)
        if existing is not None:
            return existing
        identifier = feedback_id or f"feedback-{uuid4()}"
        if identifier in self.feedback_store.records:
            raise ValueError(f"Feedback already exists: {identifier}")
        self._validate_feedback_target(target)
        revision = 1
        propagation, correction_id, expected_id = self._propagate_feedback(
            identifier,
            revision,
            target,
            normalized,
            correction=correction,
            expected_answer=expected_answer,
        )
        record = self.feedback_store.create(
            signals=normalized,
            target=target,
            provenance=self._feedback_provenance(actor_type, actor_id, source),
            correction_memory_id=correction_id,
            expected_answer_memory_id=expected_id,
            propagation=propagation,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
            feedback_id=identifier,
        )
        self._persist_feedback_state()
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def revise_feedback(
        self,
        feedback_id: str,
        *,
        expected_revision: int,
        signals: tuple[FeedbackSignal, ...],
        idempotency_key: str,
        actor_type: str,
        actor_id: str | None,
        source: str,
        correction: str | None = None,
        expected_answer: str | None = None,
    ) -> FeedbackRecord:
        normalized = normalize_signals(signals)
        self._validate_feedback_content(normalized, correction, expected_answer)
        fingerprint = feedback_fingerprint(
            "revise",
            {
                "feedback_id": feedback_id,
                "expected_revision": expected_revision,
                "signals": [item.value for item in normalized],
                "correction": correction,
                "expected_answer": expected_answer,
                "actor_type": actor_type,
            },
        )
        existing = self.feedback_store.idempotent_result(idempotency_key, fingerprint)
        if existing is not None:
            return existing
        current = self.feedback_store.get(feedback_id)
        if current.current_revision != expected_revision:
            raise ValueError(
                f"Feedback revision conflict: expected {expected_revision}, "
                f"current {current.current_revision}"
            )
        self._withdraw_feedback_effects(current.current)
        propagation, correction_id, expected_id = self._propagate_feedback(
            feedback_id,
            expected_revision + 1,
            current.current.target,
            normalized,
            correction=correction,
            expected_answer=expected_answer,
        )
        record = self.feedback_store.revise(
            feedback_id,
            expected_revision=expected_revision,
            signals=normalized,
            target=current.current.target,
            provenance=self._feedback_provenance(actor_type, actor_id, source),
            correction_memory_id=correction_id,
            expected_answer_memory_id=expected_id,
            propagation=propagation,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
        )
        self._persist_feedback_state()
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def withdraw_feedback(
        self,
        feedback_id: str,
        *,
        expected_revision: int,
        idempotency_key: str,
        actor_type: str,
        actor_id: str | None,
        source: str,
    ) -> FeedbackRecord:
        fingerprint = feedback_fingerprint(
            "withdraw",
            {
                "feedback_id": feedback_id,
                "expected_revision": expected_revision,
                "actor_type": actor_type,
            },
        )
        existing = self.feedback_store.idempotent_result(idempotency_key, fingerprint)
        if existing is not None:
            return existing
        current = self.feedback_store.get(feedback_id)
        if current.current_revision != expected_revision:
            raise ValueError(
                f"Feedback revision conflict: expected {expected_revision}, "
                f"current {current.current_revision}"
            )
        self._withdraw_feedback_effects(current.current)
        propagation = FeedbackPropagation(
            memory_id=current.current.propagation.memory_id,
            correction_memory_id=None,
            memory_before=current.current.propagation.memory_after,
            memory_after=current.current.propagation.memory_before,
            decision_id=current.current.propagation.decision_id,
            decision_outcome_applied=False,
            prediction_error=None,
            value_evidence=None,
            training_disposition=TrainingDisposition.INCLUDE,
            exclusion_refs=(),
            reason_codes=("feedback_withdrawn",),
        )
        record = self.feedback_store.revise(
            feedback_id,
            expected_revision=expected_revision,
            signals=(),
            target=current.current.target,
            provenance=self._feedback_provenance(actor_type, actor_id, source),
            correction_memory_id=None,
            expected_answer_memory_id=None,
            propagation=propagation,
            idempotency_key=idempotency_key,
            fingerprint=fingerprint,
            status=FeedbackStatus.WITHDRAWN,
        )
        self._persist_feedback_state()
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def _propagate_feedback(
        self,
        feedback_id: str,
        revision: int,
        target: FeedbackTarget,
        signals: tuple[FeedbackSignal, ...],
        *,
        correction: str | None,
        expected_answer: str | None,
    ) -> tuple[FeedbackPropagation, str | None, str | None]:
        event = current_agent_event()
        memory_id = self._feedback_memory_id(target)
        before: dict[str, str] = {}
        after: dict[str, str] = {}
        correction_id: str | None = None
        expected_id: str | None = None
        exclude = bool(
            set(signals)
            & {
                FeedbackSignal.BAD,
                FeedbackSignal.FACTUAL_ERROR,
                FeedbackSignal.STYLE_PROBLEM,
                FeedbackSignal.UNSAFE_BEHAVIOR,
                FeedbackSignal.DO_NOT_REMEMBER,
                FeedbackSignal.EXCLUDE_FROM_TRAINING,
            }
        )
        if memory_id is not None:
            memory = self.memory_system.get_episodic(memory_id)
            if memory is None:
                raise ValueError(f"Unknown episodic memory: {memory_id}")
            before = {
                "validation_status": memory.validation_status.value,
                "lifecycle_status": memory.lifecycle_status.value,
                "training_included": str(memory.training_included).lower(),
            }
            validation = memory.validation_status
            lifecycle = memory.lifecycle_status
            if FeedbackSignal.DO_NOT_REMEMBER in signals:
                validation = ValidationStatus.REJECTED
                lifecycle = MemoryLifecycleStatus.REJECTED
            elif (
                FeedbackSignal.CORRECTION in signals
                or FeedbackSignal.EXPECTED_ANSWER in signals
                or FeedbackSignal.FACTUAL_ERROR in signals
            ):
                validation = ValidationStatus.DISPUTED
                lifecycle = (
                    MemoryLifecycleStatus.CORRECTED
                    if correction
                    else MemoryLifecycleStatus.QUARANTINED
                )
            elif set(signals) & {
                FeedbackSignal.BAD,
                FeedbackSignal.STYLE_PROBLEM,
                FeedbackSignal.UNSAFE_BEHAVIOR,
            }:
                validation = ValidationStatus.DISPUTED
                lifecycle = MemoryLifecycleStatus.QUARANTINED
            elif FeedbackSignal.REMEMBER in signals:
                lifecycle = MemoryLifecycleStatus.ACTIVE
            if correction is not None:
                correction_id = self.memory_system.save_feedback_correction(
                    memory_id,
                    correction,
                    feedback_id=feedback_id,
                    kind="correction",
                )
            if expected_answer is not None:
                expected_id = self.memory_system.save_feedback_correction(
                    memory_id,
                    expected_answer,
                    feedback_id=feedback_id,
                    kind="expected_answer",
                )
            updated = self.memory_system.apply_feedback_policy(
                memory_id,
                validation_status=validation,
                lifecycle_status=lifecycle,
                training_included=not exclude,
                feedback_id=feedback_id,
            )
            if FeedbackSignal.DO_NOT_REMEMBER in signals:
                for linked_id in (correction_id, expected_id):
                    if linked_id is not None:
                        self.memory_system.apply_feedback_policy(
                            linked_id,
                            validation_status=ValidationStatus.REJECTED,
                            lifecycle_status=MemoryLifecycleStatus.REJECTED,
                            training_included=False,
                            feedback_id=feedback_id,
                        )
            after = {
                "validation_status": updated.validation_status.value,
                "lifecycle_status": updated.lifecycle_status.value,
                "training_included": str(updated.training_included).lower(),
            }
        decision_id = target.decision_id or (
            target.target_id
            if target.target_type == FeedbackTargetType.DECISION
            else None
        )
        prediction_error: float | None = None
        outcome_applied = False
        if decision_id is not None and set(signals) & {
            FeedbackSignal.GOOD,
            FeedbackSignal.BAD,
            FeedbackSignal.FACTUAL_ERROR,
            FeedbackSignal.STYLE_PROBLEM,
            FeedbackSignal.UNSAFE_BEHAVIOR,
        }:
            utility = self._feedback_utility(signals)
            decision = self.decision_store.record_feedback_outcome(
                decision_id,
                utility=utility,
                success=utility > 0.0,
                feedback_id=feedback_id,
                feedback_revision=revision,
                observed_event_id=None if event is None else event.event_id,
                observed_event_sequence=None
                if event is None
                else event.processing_sequence,
            )
            decision = self._apply_metacognitive_outcome(decision)
            prediction_error = decision.prediction_error
            outcome_applied = True
        if decision_id is not None:
            self.decision_store.set_training_policy(
                decision_id, included=not exclude, feedback_id=feedback_id
            )
        direction = "supporting" if self._feedback_utility(signals) > 0 else "opposing"
        value_impacts: dict[str, float] = {}
        if decision_id is not None:
            decision = self.decision_store.get(decision_id)
            selected = next(
                item.candidate
                for item in decision.considered_candidates
                if item.candidate.candidate_id == decision.selected_candidate_id
            )
            utility = self._feedback_utility(signals)
            value_impacts = {
                value_id: max(-1.0, min(1.0, effect * utility))
                for value_id, effect in selected.value_effects.items()
            }
        value_proposal = (
            None
            if not set(signals)
            & {
                FeedbackSignal.GOOD,
                FeedbackSignal.BAD,
                FeedbackSignal.FACTUAL_ERROR,
                FeedbackSignal.STYLE_PROBLEM,
                FeedbackSignal.UNSAFE_BEHAVIOR,
            }
            else ValueEvidenceProposal(
                proposal_id=f"feedback:{feedback_id}:{revision}",
                direction=direction,
                strength=abs(self._feedback_utility(signals)),
                reason_codes=tuple(item.value for item in signals),
                target_refs=(f"{target.target_type.value}:{target.target_id}",),
                value_impacts=value_impacts,
            )
        )
        propagation = FeedbackPropagation(
            memory_id=memory_id,
            correction_memory_id=correction_id or expected_id,
            memory_before=before,
            memory_after=after,
            decision_id=decision_id,
            decision_outcome_applied=outcome_applied,
            prediction_error=prediction_error,
            value_evidence=value_proposal,
            training_disposition=(
                TrainingDisposition.EXCLUDE if exclude else TrainingDisposition.INCLUDE
            ),
            exclusion_refs=((feedback_id,) if exclude else ()),
            reason_codes=tuple(item.value for item in signals),
        )
        self.memory_system.reevaluate_semantics_for_feedback(
            feedback_id, rejected=False
        )
        return propagation, correction_id, expected_id

    def _withdraw_feedback_effects(self, revision: FeedbackRevision) -> None:
        propagation = revision.propagation
        self.memory_system.reevaluate_semantics_for_feedback(
            self._feedback_id_for_revision(revision), rejected=True
        )
        if propagation.memory_id is not None and propagation.memory_before:
            before = propagation.memory_before
            self.memory_system.apply_feedback_policy(
                propagation.memory_id,
                validation_status=ValidationStatus(before["validation_status"]),
                lifecycle_status=MemoryLifecycleStatus(before["lifecycle_status"]),
                training_included=before.get("training_included", "true") == "true",
                feedback_id=self._feedback_id_for_revision(revision),
            )
            for correction_id in (
                revision.correction_memory_id,
                revision.expected_answer_memory_id,
            ):
                if correction_id is not None:
                    self.memory_system.withdraw_feedback_correction(
                        propagation.memory_id, correction_id
                    )
        if propagation.decision_id is not None:
            feedback_id = self._feedback_id_for_revision(revision)
            self.decision_store.withdraw_feedback_outcome(
                propagation.decision_id, feedback_id=feedback_id
            )
            self.decision_store.clear_post_assessment(propagation.decision_id)
            self.metacognition.withdraw_outcome(propagation.decision_id)
            self._sync_metacognitive_narrative()
            self.decision_store.set_training_policy(
                propagation.decision_id, included=True, feedback_id=feedback_id
            )

    def _feedback_id_for_revision(self, revision: FeedbackRevision) -> str:
        for record in self.feedback_store.records.values():
            if revision in record.revisions:
                return record.feedback_id
        raise ValueError("Feedback revision is not attached to a record")

    def _validate_feedback_target(self, target: FeedbackTarget) -> None:
        memory_id = self._feedback_memory_id(target)
        if memory_id is not None:
            memory = self.memory_system.get_episodic(memory_id)
            if memory is None:
                raise ValueError(f"Unknown episodic memory: {memory_id}")
            if target.context_id is not None and memory.context_id != target.context_id:
                raise ValueError("Feedback context does not own the target episode")
            if (
                target.experience_id is not None
                and memory.experience_id != target.experience_id
            ):
                raise ValueError("Feedback experience does not own the target episode")
        if target.target_type == FeedbackTargetType.DECISION:
            self.decision_store.get(target.target_id)
        if target.decision_id is not None:
            self.decision_store.get(target.decision_id)
        if target.target_type == FeedbackTargetType.CONTEXT:
            if self.context_registry.get(target.target_id) is None:
                raise ValueError(f"Unknown context: {target.target_id}")

    @staticmethod
    def _feedback_memory_id(target: FeedbackTarget) -> str | None:
        if target.episode_id is not None:
            return target.episode_id
        if target.target_type in {
            FeedbackTargetType.RESPONSE,
            FeedbackTargetType.EPISODE,
            FeedbackTargetType.MEMORY,
        }:
            return target.target_id
        return None

    @staticmethod
    def _validate_feedback_content(
        signals: tuple[FeedbackSignal, ...],
        correction: str | None,
        expected_answer: str | None,
    ) -> None:
        if FeedbackSignal.CORRECTION in signals and not correction:
            raise ValueError("correction signal requires correction content")
        if FeedbackSignal.EXPECTED_ANSWER in signals and not expected_answer:
            raise ValueError("expected_answer signal requires expected answer content")
        if correction is not None and FeedbackSignal.CORRECTION not in signals:
            raise ValueError("Correction content requires the correction signal")
        if (
            expected_answer is not None
            and FeedbackSignal.EXPECTED_ANSWER not in signals
        ):
            raise ValueError(
                "Expected answer content requires the expected_answer signal"
            )

    @staticmethod
    def _feedback_utility(signals: tuple[FeedbackSignal, ...]) -> float:
        if FeedbackSignal.UNSAFE_BEHAVIOR in signals:
            return -1.0
        if FeedbackSignal.BAD in signals or FeedbackSignal.FACTUAL_ERROR in signals:
            return -0.75
        if FeedbackSignal.STYLE_PROBLEM in signals:
            return -0.4
        if FeedbackSignal.GOOD in signals:
            return 1.0
        return 0.0

    @staticmethod
    def _feedback_provenance(
        actor_type: str, actor_id: str | None, source: str
    ) -> FeedbackProvenance:
        event = current_agent_event()
        return FeedbackProvenance(
            actor_type=actor_type,
            actor_id=actor_id,
            source=source,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
            submitted_at=datetime.now(UTC).isoformat(),
        )

    def _persist_decision_state(self) -> None:
        self.persistent_state.extensions["decision_records"] = (
            self.decision_store.to_json()
        )

    def create_decision(
        self,
        candidates: list[ActionCandidate],
        *,
        context_id: str | None = None,
        satisfied_prerequisites: set[str] | None = None,
        decision_id: str | None = None,
        boundary_assessment_id: str | None = None,
    ) -> DecisionRecord:
        event = current_agent_event()
        boundary_assessment = (
            None
            if boundary_assessment_id is None
            else self.identity_boundary_store.get_assessment(boundary_assessment_id)
        )
        for candidate in candidates:
            self.plan_store.validate_candidate(candidate)
        if context_id is not None and self.context_registry.get(context_id) is None:
            raise ValueError(f"Unknown context: {context_id}")
        completed_goals = {
            goal.goal_id
            for goal in self.goal_manager.goals.values()
            if goal.status == GoalStatus.COMPLETED
        }
        completed_steps = {
            f"step:{plan.plan_id}:{plan.revision}:{state.step_id}:completed"
            for plan in self.plan_store.list_plans()
            for state in plan.step_states
            if state.status == StepStatus.COMPLETED
        }
        satisfied_prerequisites = set(satisfied_prerequisites or ()) | completed_steps
        emotion = self.emotion_engine.state
        source_experience = (
            None
            if context_id is None
            else self.experience_store.latest_for_context(context_id)
        )
        active_beliefs = self.belief_store.active(context_id=context_id)
        narrative_claim_ids = tuple(
            dict.fromkeys(
                claim_id
                for candidate in candidates
                for claim_id in self.narrative_self.select_relevant(
                    theme_codes=string_values(candidate.parameters.get("topic_tags"))
                    + string_values(candidate.parameters.get("theme_codes")),
                    capability_ids=string_values(
                        candidate.parameters.get("capability_ids")
                    ),
                ).claim_ids
            )
        )
        decision_context = (
            None if context_id is None else self.context_registry.get(context_id)
        )
        relationship_influence = self.relationship_store.influence(
            () if decision_context is None else decision_context.participant_ids
        )
        relationship_candidates = [
            replace(
                candidate,
                appraisal_contributions={
                    **candidate.appraisal_contributions,
                    "relationship_caution": -relationship_influence.threat
                    * candidate.estimated_risk,
                    "relationship_reciprocity": (
                        relationship_influence.expected_reciprocity - 0.5
                    )
                    * (1.0 - candidate.estimated_risk)
                    if candidate.candidate_type.value
                    in {"respond", "request_information"}
                    else 0.0,
                },
            )
            for candidate in candidates
        ]
        identifier = decision_id or str(uuid4())
        narrative_refs = tuple(
            f"identity-claim:{claim_id}@{self.narrative_self.get_claim(claim_id).revision}"
            for claim_id in narrative_claim_ids
        )
        pre_assessment = self.metacognition.assess_pre(
            identifier,
            relationship_candidates,
            self_model=self.self_model.state,
            narrative_self_refs=narrative_refs,
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
            quality_provenance_refs=(
                f"working-memory:occupancy:{len(self.working_memory.items)}",
                f"attention-focus@{self.attention_system.focus.revision}",
                "emotion:current",
            ),
        )
        record = self.decision_store.create(
            relationship_candidates,
            triggering_event_id=None if event is None else event.event_id,
            triggering_event_sequence=None
            if event is None
            else event.processing_sequence,
            context_id=context_id,
            active_goal_ids=tuple(
                goal.goal_id for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)
            ),
            value_revision_refs={
                value.value_id: value.revision
                for value in self.value_system.active_values()
            },
            emotion_snapshot={
                "valence": emotion.valence,
                "arousal": emotion.arousal,
                "optimal_loss": emotion.optimal_loss,
            },
            adapter_id=self.adapter_id,
            adapter_hash=self.adapter_hash,
            activation_sequence=self.activation_sequence,
            identity_origin_refs={
                **{
                    f"goal:{goal.goal_id}": goal.identity_origin.origin_id
                    for goal in self.goal_manager.list_goals(GoalStatus.ACTIVE)
                },
                **{
                    f"value:{value.value_id}": value.origin_provenance.origin_id
                    for value in self.value_system.active_values()
                    if value.origin_provenance is not None
                },
                **{
                    f"belief:{belief.belief_id}": belief.identity_origin.origin_id
                    for belief in active_beliefs
                },
            },
            experience_refs=(
                () if source_experience is None else (source_experience.experience_id,)
            ),
            belief_revision_refs={
                belief.belief_id: belief.revision for belief in active_beliefs
            },
            narrative_self_refs=narrative_refs,
            metacognition_pre_assessment_id=pre_assessment.assessment_id,
            boundary_assessment_id=boundary_assessment_id,
            boundary_assessment_revision=(
                None if boundary_assessment is None else boundary_assessment.revision
            ),
            boundary_assessment_digest=(
                None
                if boundary_assessment is None
                else self.identity_boundary_store.assessment_digest(
                    boundary_assessment.assessment_id
                )
            ),
            boundary_recommendation=(
                None
                if boundary_assessment is None
                else boundary_assessment.recommendation.value
            ),
            satisfied_prerequisites=completed_goals
            | {
                f"capability:{capability.capability_id}"
                for capability in self.self_model.state.capabilities.values()
                if capability.confidence >= 0.5
            }
            | (satisfied_prerequisites or set()),
            value_evaluator=lambda options: self._decision_value_evaluator(
                options, context_id=context_id
            ),
            self_model_evaluator=self.self_model.evaluate_candidates,
            metacognition_evaluator=lambda values: self._calibrated_candidate_scores(
                values, pre_assessment.recommended_action
            ),
            decision_id=identifier,
        )
        decision_scores = self.value_system.evaluate(
            {
                candidate.candidate_id: candidate.value_effects
                for candidate in candidates
                if candidate.value_effects
            },
            context_id=context_id,
        )
        tradeoffs = self.value_system.record_tradeoffs(
            decision_scores,
            context_id=context_id,
            decision_id=record.decision_id,
        )
        if tradeoffs:
            record = self.decision_store.link_value_tradeoffs(
                record.decision_id, tuple(item.tradeoff_id for item in tradeoffs)
            )
        if source_experience is not None:
            self.link_experience_result(
                source_experience.experience_id,
                kind="decision",
                reference=f"decision:{record.decision_id}",
                evidence_refs=(f"decision:{record.decision_id}",),
            )
        self._sync_self_model_working_memory(candidates)
        self._persist_decision_state()
        self._persist_metacognition_state()
        return record

    def decision_dataset(self) -> list[DecisionDatasetRecord]:
        return DecisionDatasetGenerator().generate(
            self.decision_store.list_records(DecisionStatus.RESOLVED)
        )

    def _decision_value_evaluator(
        self,
        options: dict[str, dict[str, float]],
        *,
        context_id: str | None = None,
    ) -> dict[str, dict[str, float]]:
        if not options:
            return {}
        return {
            score.option_id: {
                contribution.value_id: contribution.contribution
                for contribution in score.contributions
            }
            for score in self.value_system.evaluate(options, context_id=context_id)
        }
