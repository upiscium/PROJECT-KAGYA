from kagya.decision import (
    ActionCandidate,
)
from kagya.identity import (
    AutobiographicalEpisode,
    ContinuityLink,
    EndorsementStatus,
    EpistemicUncertainty,
    IdentityRevisionProposal,
    KnownLimitation,
    IdentityClaim,
    IdentityClaimKind,
    IdentityClaimStatus,
    FutureSelfProjection,
    NarrativeChapter,
    NarrativeSelf,
    OriginActor,
    OriginInputKind,
    SelfModel,
    SelfModelState,
    new_identity_origin,
    BoundaryAssessmentInput,
    IdentityBoundaryAssessment,
    SocialPressureMetadata,
    SocialPressureSignal,
)
from kagya.motivation import (
    ACCEPTED_COMMITMENT_STATUSES,
    CommitmentStatus,
    GoalStatus,
)
from kagya.runtime.agent_runtime import current_agent_event
from kagya.runtime.working_memory import (
    RetentionReason,
    WorkingMemoryKind,
    working_memory_item,
)
from kagya.runtime.coordinators._shared import RuntimeDomainMixin, string_values

from typing import Callable


class IdentityNarrativeCoordinator(RuntimeDomainMixin):
    def __init__(
        self,
        self_model: SelfModel,
        narrative: NarrativeSelf,
        *,
        persist_self_model: Callable[[], None],
        persist_narrative: Callable[[], None],
    ) -> None:
        self._self_model = self_model
        self._narrative = narrative
        self._persist_self_model = persist_self_model
        self._persist_narrative = persist_narrative

    def persist_identity(self) -> None:
        self._persist_self_model()
        self._persist_narrative()

    def restore_self_model_state(self) -> None:
        self.self_model.restore(self.persistent_state.self_model)
        self._sync_self_references()
        self._persist_self_model_state()

    def _persist_self_model_state(self) -> None:
        self.persistent_state.self_model = self.self_model.to_json()

    def restore_identity_boundary_state(self) -> None:
        self.identity_boundary_store.restore(
            self.persistent_state.identity_extensions.get("identity_boundary")
        )
        self._persist_identity_boundary_state()

    def _persist_identity_boundary_state(self) -> None:
        self.persistent_state.identity_extensions["identity_boundary"] = (
            self.identity_boundary_store.to_json()
        )

    def record_social_pressure(
        self, metadata: SocialPressureMetadata, *, context_id: str | None = None
    ) -> SocialPressureSignal:
        event = current_agent_event()
        if event is None or event.processing_sequence is None:
            raise RuntimeError("social pressure mutation requires AgentRuntime")
        signal = self.identity_boundary_store.add_pressure(
            metadata,
            context_id=context_id,
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
        )
        self._persist_identity_boundary_state()
        return signal

    def assess_identity_boundary(
        self, inputs: BoundaryAssessmentInput
    ) -> IdentityBoundaryAssessment:
        event = current_agent_event()
        if event is None or event.processing_sequence is None:
            raise RuntimeError("identity-boundary assessment requires AgentRuntime")
        values = {item.value_id: item for item in self.value_system.active_values()}
        requested_values = set(inputs.self_endorsed_value_refs)
        if requested_values - values.keys():
            raise ValueError("assessment references an inactive or unreviewed Value")
        authorized_values = {
            key
            for key in requested_values
            if values[key].origin_provenance is not None
            and (
                values[key].origin_provenance.actor == OriginActor.SELF
                or (
                    values[key].origin_provenance.actor == OriginActor.SYSTEM
                    and values[key].origin_provenance.endorsed_by_event_id is not None
                    and values[key].origin_provenance.endorsement_ref
                    == "subject_endorsement"
                )
            )
        }
        if authorized_values != requested_values:
            inputs = inputs.model_copy(
                update={"self_endorsed_value_refs": tuple(sorted(authorized_values))}
            )
        goals = {item.goal_id: item for item in self.goal_manager.list_goals()}
        commitments = {
            item.commitment_id: item
            for item in self.commitment_store.list_commitments()
        }
        relationships = {
            item.relationship_id: item
            for item in self.relationship_store.list_relationships()
        }
        requested_goals = set(inputs.self_endorsed_goal_refs)
        requested_commitments = set(inputs.self_endorsed_commitment_refs)
        requested_relationships = set(inputs.relationship_refs)
        if (
            inputs.context_id is not None
            and self.context_registry.get(inputs.context_id) is None
        ):
            raise ValueError("assessment references an unknown Context")
        if any(
            key not in goals
            or goals[key].status != GoalStatus.ACTIVE
            or goals[key].identity_origin.endorsement != EndorsementStatus.ENDORSED
            for key in requested_goals
        ):
            raise ValueError("assessment references an inactive or unendorsed Goal")
        if any(
            key not in commitments
            or commitments[key].status not in ACCEPTED_COMMITMENT_STATUSES
            or commitments[key].identity_origin.endorsement
            != EndorsementStatus.ENDORSED
            or not (
                commitments[key].identity_origin.actor == OriginActor.SELF
                or (
                    commitments[key].identity_origin.actor == OriginActor.SYSTEM
                    and commitments[key].identity_origin.endorsed_by_event_id
                    is not None
                    and commitments[key].identity_origin.endorsement_ref
                    == "subject_endorsement"
                )
            )
            for key in requested_commitments
        ):
            raise ValueError(
                "assessment references an inactive or unendorsed Commitment"
            )
        if requested_relationships - relationships.keys():
            raise ValueError("assessment references an unknown Relationship")
        known_protected_refs = {
            *(f"value:{item.value_id}@{item.revision}" for item in values.values()),
            *(
                f"commitment:{item.commitment_id}@{len(item.transitions)}"
                for item in commitments.values()
                if item.status in ACCEPTED_COMMITMENT_STATUSES
            ),
            *(
                reference
                for signal in self.identity_boundary_store.signals
                if signal.signal_id in inputs.pressure_signal_ids
                for reference in signal.evidence_refs
            ),
        }
        if set(inputs.protected_state_conflict_refs) - known_protected_refs:
            raise ValueError(
                "protected-state conflict must reference active state or typed runtime evidence"
            )
        for reference in inputs.other_welfare_evidence_refs:
            if not reference.startswith("experience:"):
                raise ValueError(
                    "other-welfare evidence must reference a structured Experience"
                )
            experience = self.experience_store.get(
                reference.removeprefix("experience:")
            )
            if (
                "other_welfare_reviewed" not in experience.interpretation_codes
                or not any(
                    revision.event_id is not None
                    and revision.event_sequence is not None
                    and revision.evidence_refs
                    and revision.reason_code == "other_welfare_reviewed"
                    for revision in experience.revisions
                )
            ):
                raise ValueError(
                    "other-welfare Experience requires a reviewed typed interpretation"
                )
        assessment = self.identity_boundary_store.assess(
            inputs,
            event_id=event.event_id,
            event_sequence=event.processing_sequence,
            value_revision_refs={key: values[key].revision for key in requested_values},
            goal_revision_refs={
                key: len(goals[key].transitions) for key in requested_goals
            },
            commitment_revision_refs={
                key: len(commitments[key].transitions) for key in requested_commitments
            },
            relationship_revision_refs={
                key: relationships[key].revision for key in requested_relationships
            },
            adapter_id=self.adapter_id,
            adapter_hash=self.adapter_hash,
            activation_sequence=self.activation_sequence,
        )
        self._persist_identity_boundary_state()
        return assessment

    def validate_identity_boundary_assessment(
        self,
        assessment: IdentityBoundaryAssessment,
        *,
        action_ref: str,
        context_id: str | None,
    ) -> None:
        if assessment.action_ref != action_ref:
            raise ValueError("Boundary assessment action binding is invalid")
        if assessment.context_id != context_id:
            raise ValueError("Boundary assessment context binding is invalid")
        if (
            assessment.adapter_id != self.adapter_id
            or assessment.adapter_hash != self.adapter_hash
            or assessment.activation_sequence != self.activation_sequence
        ):
            raise ValueError(
                "Boundary assessment adapter activation binding is invalid"
            )
        current_values = {
            item.value_id: item.revision for item in self.value_system.active_values()
        }
        current_goals = {
            item.goal_id: len(item.transitions)
            for item in self.goal_manager.list_goals()
            if item.status == GoalStatus.ACTIVE
        }
        current_commitments = {
            item.commitment_id: len(item.transitions)
            for item in self.commitment_store.list_commitments()
            if item.status in ACCEPTED_COMMITMENT_STATUSES
        }
        current_relationships = {
            item.relationship_id: item.revision
            for item in self.relationship_store.list_relationships()
        }
        if (
            any(
                current_values.get(key) != revision
                for key, revision in assessment.value_revision_refs.items()
            )
            or any(
                current_goals.get(key) != revision
                for key, revision in assessment.goal_revision_refs.items()
            )
            or any(
                current_commitments.get(key) != revision
                for key, revision in assessment.commitment_revision_refs.items()
            )
            or any(
                current_relationships.get(key) != revision
                for key, revision in assessment.relationship_revision_refs.items()
            )
        ):
            raise ValueError("Boundary assessment references stale subject state")

    def attach_identity_boundary_probe(self, assessment_id, probe):
        assessment = self.identity_boundary_store.attach_probe(assessment_id, probe)
        self._persist_identity_boundary_state()
        return assessment

    def update_capability_from_decision(
        self,
        capability_id: str,
        description: str,
        decision_id: str,
        *,
        tags: tuple[str, ...] = (),
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.update_capability_from_decision(
            capability_id,
            description,
            self.decision_store.get(decision_id),
            tags=tags,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def manual_correct_capability(
        self,
        capability_id: str,
        description: str,
        confidence: float,
        *,
        reason: str,
        tags: tuple[str, ...] = (),
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.manual_correct_capability(
            capability_id,
            description,
            confidence,
            reason=reason,
            tags=tags,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def add_self_limitation(
        self, limitation: KnownLimitation, *, reason: str
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.add_limitation(
            limitation,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def add_self_uncertainty(
        self, uncertainty: EpistemicUncertainty, *, reason: str
    ) -> SelfModelState:
        event = current_agent_event()
        self.self_model.add_uncertainty(
            uncertainty,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return self.self_model.state

    def propose_identity_revision(
        self,
        *,
        proposed_summary: str | None,
        proposed_traits: dict[str, float],
        evidence_refs: tuple[str, ...],
        source: str,
        origin_actor: OriginActor | None = None,
        origin_input_kind: OriginInputKind | None = None,
        proposal_id: str | None = None,
    ) -> IdentityRevisionProposal:
        event = current_agent_event()
        proposal = self.self_model.propose_identity_revision(
            proposed_summary=proposed_summary,
            proposed_traits=proposed_traits,
            evidence_refs=evidence_refs,
            source=source,
            identity_origin=(
                None
                if origin_actor is None
                else new_identity_origin(
                    origin_actor,
                    origin_input_kind or OriginInputKind.SUGGESTION,
                    source_ref="self_model_proposal",
                    event_id=None if event is None else event.event_id,
                    event_sequence=None if event is None else event.processing_sequence,
                )
            ),
            proposal_id=proposal_id,
        )
        self._persist_self_model_state()
        return proposal

    def resolve_identity_revision(
        self, proposal_id: str, *, apply: bool, reason: str
    ) -> IdentityRevisionProposal:
        event = current_agent_event()
        proposal = self.self_model.resolve_identity_revision(
            proposal_id,
            apply=apply,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return proposal

    def rollback_self_model(
        self, target_revision: int, *, reason: str
    ) -> SelfModelState:
        event = current_agent_event()
        state = self.self_model.rollback(
            target_revision,
            reason=reason,
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )
        self._persist_self_model_state()
        return state

    def form_autobiographical_episode(
        self, experience_id: str
    ) -> AutobiographicalEpisode:
        experience = self.experience_store.get(experience_id)
        episode = self.narrative_self.observe_experience(experience)
        if episode is None:
            raise ValueError(
                "Experience does not meet autobiographical importance threshold"
            )
        self.link_experience_result(
            experience_id,
            kind="narrative",
            reference=f"narrative:{episode.episode_id}",
            evidence_refs=(f"experience:{experience_id}",),
        )
        self._sync_self_references()
        self._persist_self_model_state()
        self._persist_narrative_self_state()
        return episode

    def create_narrative_chapter(
        self,
        *,
        title: str,
        theme_codes: tuple[str, ...],
        episode_ids: tuple[str, ...],
        chapter_id: str | None = None,
    ) -> NarrativeChapter:
        chapter = self.narrative_self.create_chapter(
            title=title,
            theme_codes=theme_codes,
            episode_ids=episode_ids,
            chapter_id=chapter_id,
        )
        self._persist_narrative_self_state()
        return chapter

    def propose_narrative_claim(
        self,
        *,
        kind: IdentityClaimKind,
        statement: str,
        polarity: int,
        theme_codes: tuple[str, ...],
        confidence: float,
        stability: float,
        evidence_refs: tuple[str, ...],
        related_experience_ids: tuple[str, ...] = (),
        related_value_refs: tuple[str, ...] = (),
        related_goal_refs: tuple[str, ...] = (),
        related_decision_refs: tuple[str, ...] = (),
        claim_id: str | None = None,
    ) -> IdentityClaim:
        claim = self.narrative_self.propose_claim(
            kind=kind,
            statement=statement,
            polarity=polarity,
            theme_codes=theme_codes,
            confidence=confidence,
            stability=stability,
            evidence_refs=evidence_refs,
            related_experience_ids=related_experience_ids,
            related_value_refs=related_value_refs,
            related_goal_refs=related_goal_refs,
            related_decision_refs=related_decision_refs,
            claim_id=claim_id,
        )
        self._persist_narrative_self_state()
        return claim

    def revise_narrative_claim(
        self,
        claim_id: str,
        *,
        confidence: float,
        reason_code: str,
        evidence_refs: tuple[str, ...] = (),
        counterevidence_refs: tuple[str, ...] = (),
        status: IdentityClaimStatus | None = None,
    ) -> IdentityClaim:
        claim = self.narrative_self.revise_claim(
            claim_id,
            confidence=confidence,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            counterevidence_refs=counterevidence_refs,
            status=status,
        )
        self._persist_narrative_self_state()
        return claim

    def set_future_self_projection(
        self,
        *,
        description: str,
        theme_codes: tuple[str, ...],
        desired_level: float,
        current_level: float,
        evidence_refs: tuple[str, ...],
        projection_id: str | None = None,
    ) -> FutureSelfProjection:
        projection = self.narrative_self.set_future_self(
            description=description,
            theme_codes=theme_codes,
            desired_level=desired_level,
            current_level=current_level,
            evidence_refs=evidence_refs,
            projection_id=projection_id,
        )
        motivation = self.motivation_dynamics.observe_future_self_gap(
            projection.projection_id,
            gap=projection.gap,
            uncertainty=1.0 - min(1.0, len(evidence_refs) * 0.25),
        )
        if motivation is not None:
            projection = self.narrative_self.link_future_motivation(
                projection.projection_id, motivation.motivation_id
            )
        self._persist_narrative_self_state()
        self._persist_motivation_state()
        return projection

    def create_narrative_continuity_link(
        self,
        earlier_ref: str,
        later_ref: str,
        *,
        relation_code: str,
        evidence_refs: tuple[str, ...],
        confidence: float,
        link_id: str | None = None,
    ) -> ContinuityLink:
        link = self.narrative_self.link_continuity(
            earlier_ref,
            later_ref,
            relation_code=relation_code,
            evidence_refs=evidence_refs,
            confidence=confidence,
            link_id=link_id,
        )
        self._persist_narrative_self_state()
        return link

    def _sync_self_references(self) -> None:
        event = current_agent_event()
        self.self_model.sync_references(
            commitment_refs=(
                commitment.commitment_id
                for commitment in self.commitment_store.commitments.values()
                if commitment.status != CommitmentStatus.PROPOSED
            ),
            value_revision_refs={
                value.value_id: value.revision
                for value in self.value_system.active_values()
            },
            autobiographical_summary_refs=(
                f"narrative:{episode.episode_id}"
                for episode in self.narrative_self.episodes.values()
            ),
            evidence_refs=tuple(
                f"narrative:{episode.episode_id}"
                for episode in self.narrative_self.episodes.values()
            ),
            event_id=None if event is None else event.event_id,
            event_sequence=None if event is None else event.processing_sequence,
        )

    def _sync_self_model_working_memory(
        self, candidates: list[ActionCandidate]
    ) -> None:
        for item in tuple(self.working_memory.items):
            if item.kind == WorkingMemoryKind.SELF_MODEL:
                self.working_memory.forget(item.item_id)
        for candidate in candidates:
            selection = self.self_model.select_relevant(candidate)
            narrative = self.narrative_self.select_relevant(
                theme_codes=string_values(candidate.parameters.get("topic_tags"))
                + string_values(candidate.parameters.get("theme_codes")),
                capability_ids=string_values(
                    candidate.parameters.get("capability_ids")
                ),
            )
            for index, rendered in enumerate(
                (*selection.rendered_items, *narrative.rendered_items)
            ):
                self.working_memory.admit(
                    working_memory_item(
                        item_id=f"self:{candidate.candidate_id}:{index}",
                        kind=WorkingMemoryKind.SELF_MODEL,
                        content=rendered,
                        activation=0.85,
                        salience=0.8,
                        retention_reason=RetentionReason.RELEVANT_SELF_MODEL,
                        source="runtime.self_model",
                    )
                )
