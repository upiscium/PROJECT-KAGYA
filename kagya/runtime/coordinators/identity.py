from kagya.decision import (
    ActionCandidate,
)
from kagya.identity import (
    AutobiographicalEpisode,
    ContinuityLink,
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
)
from kagya.motivation import (
    CommitmentStatus,
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
                for value in self.value_system.list_values()
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
