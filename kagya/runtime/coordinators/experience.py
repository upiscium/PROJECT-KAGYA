"""Experience integration without store-to-store dependencies."""

from dataclasses import dataclass
from typing import Callable

from kagya.experience import ExperienceRecord, ExperienceStore
from kagya.identity import AutobiographicalEpisode, NarrativeSelf
from kagya.motivation import MotivationDynamics
from kagya.relationship import RelationshipState, RelationshipStore


@dataclass(frozen=True)
class ExperienceIntegrationResult:
    experience: ExperienceRecord
    relationships: tuple[RelationshipState, ...]
    narrative_episode: AutobiographicalEpisode | None


class ExperienceIntegrationCoordinator:
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
