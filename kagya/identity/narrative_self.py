"""Structured autobiographical continuity and revisable identity claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import TYPE_CHECKING, Any, Iterable
from uuid import uuid4

if TYPE_CHECKING:
    from kagya.experience import ExperienceRecord


class IdentityClaimKind(StrEnum):
    IDENTITY = "identity"
    TRAIT = "trait"
    ROLE = "role"
    RELATIONSHIP = "relationship"
    CAPABILITY = "capability"
    LIMITATION = "limitation"


class IdentityClaimStatus(StrEnum):
    HYPOTHESIS = "hypothesis"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    REVISED = "revised"


@dataclass(frozen=True)
class AutobiographicalEpisode:
    episode_id: str
    title: str
    experience_ids: tuple[str, ...]
    theme_codes: tuple[str, ...]
    related_value_refs: tuple[str, ...]
    related_goal_refs: tuple[str, ...]
    related_decision_refs: tuple[str, ...]
    related_commitment_refs: tuple[str, ...]
    salience: float
    turning_point: bool
    turning_point_codes: tuple[str, ...]
    unresolved_tension: float
    occurred_at: str
    created_at: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.episode_id or not self.title:
            raise ValueError("invalid autobiographical episode")
        _unit(self.salience, "episode salience")
        _unit(self.unresolved_tension, "episode unresolved tension")
        if not self.experience_ids:
            raise ValueError("autobiographical episode requires experience evidence")


@dataclass(frozen=True)
class NarrativeChapter:
    chapter_id: str
    title: str
    theme_codes: tuple[str, ...]
    episode_ids: tuple[str, ...]
    start_at: str
    end_at: str
    created_at: str
    revision: int = 0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.chapter_id or not self.title:
            raise ValueError("invalid narrative chapter")
        if len(self.episode_ids) < 2:
            raise ValueError("narrative chapter requires multiple episodes")


@dataclass(frozen=True)
class IdentityClaimRevision:
    revision_id: str
    from_revision: int
    to_revision: int
    reason_code: str
    evidence_refs: tuple[str, ...]
    counterevidence_refs: tuple[str, ...]
    previous_status: IdentityClaimStatus
    created_at: str


@dataclass(frozen=True)
class IdentityClaim:
    claim_id: str
    kind: IdentityClaimKind
    statement: str
    polarity: int
    theme_codes: tuple[str, ...]
    confidence: float
    stability: float
    evidence_refs: tuple[str, ...]
    counterevidence_refs: tuple[str, ...]
    conflicting_claim_ids: tuple[str, ...]
    related_experience_ids: tuple[str, ...]
    related_value_refs: tuple[str, ...]
    related_goal_refs: tuple[str, ...]
    related_decision_refs: tuple[str, ...]
    status: IdentityClaimStatus
    created_at: str
    updated_at: str
    revision: int = 0
    revisions: tuple[IdentityClaimRevision, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1 or not self.claim_id or not self.statement:
            raise ValueError("invalid identity claim")
        if self.polarity not in {-1, 1}:
            raise ValueError("identity claim polarity must be -1 or 1")
        _unit(self.confidence, "identity claim confidence")
        _unit(self.stability, "identity claim stability")
        if not self.evidence_refs:
            raise ValueError("identity claim requires evidence")
        if _contains_private_key(asdict(self)):
            raise ValueError("identity claim contains private reasoning")


@dataclass(frozen=True)
class ContinuityLink:
    link_id: str
    earlier_ref: str
    later_ref: str
    relation_code: str
    evidence_refs: tuple[str, ...]
    confidence: float
    created_at: str

    def __post_init__(self) -> None:
        if not all(
            (self.link_id, self.earlier_ref, self.later_ref, self.relation_code)
        ):
            raise ValueError("continuity link fields must not be empty")
        if self.earlier_ref == self.later_ref or not self.evidence_refs:
            raise ValueError("continuity link requires distinct, evidenced references")
        _unit(self.confidence, "continuity confidence")


@dataclass(frozen=True)
class UnresolvedSelfConflict:
    conflict_id: str
    claim_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    description: str
    created_at: str
    resolved_at: str | None = None


@dataclass(frozen=True)
class FutureSelfProjection:
    projection_id: str
    description: str
    theme_codes: tuple[str, ...]
    desired_level: float
    current_level: float
    gap: float
    evidence_refs: tuple[str, ...]
    related_motivation_ids: tuple[str, ...]
    created_at: str
    updated_at: str

    def __post_init__(self) -> None:
        if not self.projection_id or not self.description or not self.evidence_refs:
            raise ValueError("future-self projection requires identity and evidence")
        _unit(self.desired_level, "future-self desired level")
        _unit(self.current_level, "future-self current level")
        _unit(self.gap, "future-self gap")


@dataclass(frozen=True)
class NarrativeSelection:
    claim_ids: tuple[str, ...]
    episode_ids: tuple[str, ...]
    rendered_items: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeCommitmentEvent:
    event_id: str
    commitment_ref: str
    kind: str
    description: str
    evidence_refs: tuple[str, ...]
    relationship_refs: tuple[str, ...]
    created_at: str

    def __post_init__(self) -> None:
        if self.kind not in {"breach", "repair"}:
            raise ValueError("Narrative commitment event must be breach or repair")
        if not all((self.event_id, self.commitment_ref, self.description)):
            raise ValueError("Narrative commitment event fields must not be empty")
        if not self.evidence_refs:
            raise ValueError("Narrative commitment event requires evidence")


class NarrativeSelf:
    """Durable, inspectable autobiography independent of generated prose."""

    SCHEMA_VERSION = 1

    def __init__(self, *, episode_threshold: float = 0.6) -> None:
        _unit(episode_threshold, "episode threshold")
        self.episode_threshold = episode_threshold
        self.episodes: dict[str, AutobiographicalEpisode] = {}
        self.chapters: dict[str, NarrativeChapter] = {}
        self.claims: dict[str, IdentityClaim] = {}
        self.continuity_links: dict[str, ContinuityLink] = {}
        self.conflicts: dict[str, UnresolvedSelfConflict] = {}
        self.future_self: dict[str, FutureSelfProjection] = {}
        self.commitment_events: dict[str, NarrativeCommitmentEvent] = {}
        self._experience_index: dict[str, str] = {}

    def observe_experience(
        self, experience: ExperienceRecord
    ) -> AutobiographicalEpisode | None:
        existing_id = self._experience_index.get(experience.experience_id)
        if existing_id is not None:
            current = self.episodes[existing_id]
            existing_turning_codes = list(current.turning_point_codes)
            if (
                experience.result_refs.get("value")
                and "value_change" not in existing_turning_codes
            ):
                existing_turning_codes.append("value_change")
            updated = replace(
                current,
                related_value_refs=tuple(
                    dict.fromkeys(
                        (
                            *current.related_value_refs,
                            *experience.result_refs.get("value", ()),
                        )
                    )
                ),
                related_goal_refs=tuple(
                    dict.fromkeys(
                        (
                            *current.related_goal_refs,
                            *experience.result_refs.get("goal", ()),
                        )
                    )
                ),
                related_decision_refs=tuple(
                    dict.fromkeys(
                        (
                            *current.related_decision_refs,
                            *experience.result_refs.get("decision", ()),
                        )
                    )
                ),
                turning_point=bool(existing_turning_codes),
                turning_point_codes=tuple(existing_turning_codes),
            )
            self.episodes[existing_id] = updated
            return updated
        if experience.autobiographical_importance < self.episode_threshold:
            return None
        value_refs = tuple(experience.result_refs.get("value", ())) or tuple(
            f"value:{value_id}@{revision}"
            for value_id, revision in sorted(experience.value_revision_refs.items())
        )
        goal_refs = tuple(
            dict.fromkeys(
                (*experience.active_goal_refs, *experience.result_refs.get("goal", ()))
            )
        )
        decision_refs = tuple(experience.result_refs.get("decision", ()))
        turning_codes: list[str] = []
        if experience.appraisal.goal_progress <= -0.6:
            turning_codes.append("significant_failure")
        if experience.appraisal.goal_progress >= 0.6:
            turning_codes.append("significant_success")
        if experience.result_refs.get("value"):
            turning_codes.append("value_change")
        now = _now()
        themes = tuple(
            dict.fromkeys(
                (*experience.situation_codes, *experience.interpretation_codes)
            )
        )
        episode = AutobiographicalEpisode(
            episode_id=f"autobiographical-episode-{uuid4()}",
            title=f"Experience in {experience.context_id}",
            experience_ids=(experience.experience_id,),
            theme_codes=themes,
            related_value_refs=value_refs,
            related_goal_refs=goal_refs,
            related_decision_refs=decision_refs,
            related_commitment_refs=(),
            salience=experience.autobiographical_importance,
            turning_point=bool(turning_codes),
            turning_point_codes=tuple(turning_codes),
            unresolved_tension=experience.unresolved_tension,
            occurred_at=experience.created_at,
            created_at=now,
        )
        self.episodes[episode.episode_id] = episode
        self._experience_index[experience.experience_id] = episode.episode_id
        self._integrate_chapters(episode)
        return episode

    def create_chapter(
        self,
        *,
        title: str,
        theme_codes: tuple[str, ...],
        episode_ids: tuple[str, ...],
        chapter_id: str | None = None,
    ) -> NarrativeChapter:
        identifiers = tuple(dict.fromkeys(episode_ids))
        episodes = [self.get_episode(item) for item in identifiers]
        identifier = chapter_id or f"chapter-{uuid4()}"
        if identifier in self.chapters:
            raise ValueError(f"Narrative chapter already exists: {identifier}")
        chapter = NarrativeChapter(
            chapter_id=identifier,
            title=title,
            theme_codes=tuple(dict.fromkeys(theme_codes)),
            episode_ids=identifiers,
            start_at=min(item.occurred_at for item in episodes),
            end_at=max(item.occurred_at for item in episodes),
            created_at=_now(),
        )
        self.chapters[identifier] = chapter
        return chapter

    def propose_claim(
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
        for experience_id in related_experience_ids:
            if experience_id not in self._experience_index:
                raise ValueError(f"Experience is not autobiographical: {experience_id}")
        identifier = claim_id or f"identity-claim-{uuid4()}"
        if identifier in self.claims:
            raise ValueError(f"Identity claim already exists: {identifier}")
        distinct_episode_evidence = set(related_experience_ids)
        status = IdentityClaimStatus.SUPPORTED
        bounded_confidence = confidence
        if kind == IdentityClaimKind.TRAIT and len(distinct_episode_evidence) < 2:
            status = IdentityClaimStatus.HYPOTHESIS
            bounded_confidence = min(confidence, 0.49)
        conflicts = tuple(
            item.claim_id
            for item in self.claims.values()
            if set(item.theme_codes).intersection(theme_codes)
            and item.polarity != polarity
            and item.status != IdentityClaimStatus.REVISED
        )
        now = _now()
        claim = IdentityClaim(
            claim_id=identifier,
            kind=kind,
            statement=statement,
            polarity=polarity,
            theme_codes=tuple(dict.fromkeys(theme_codes)),
            confidence=bounded_confidence,
            stability=stability,
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
            counterevidence_refs=(),
            conflicting_claim_ids=conflicts,
            related_experience_ids=tuple(dict.fromkeys(related_experience_ids)),
            related_value_refs=tuple(dict.fromkeys(related_value_refs)),
            related_goal_refs=tuple(dict.fromkeys(related_goal_refs)),
            related_decision_refs=tuple(dict.fromkeys(related_decision_refs)),
            status=IdentityClaimStatus.CONTESTED if conflicts else status,
            created_at=now,
            updated_at=now,
        )
        self.claims[identifier] = claim
        if conflicts:
            self._register_conflict(claim, conflicts)
        return self.claims[identifier]

    def revise_claim(
        self,
        claim_id: str,
        *,
        confidence: float,
        reason_code: str,
        evidence_refs: tuple[str, ...] = (),
        counterevidence_refs: tuple[str, ...] = (),
        status: IdentityClaimStatus | None = None,
    ) -> IdentityClaim:
        current = self.get_claim(claim_id)
        if not reason_code or (not evidence_refs and not counterevidence_refs):
            raise ValueError("claim revision requires reason and evidence")
        resolved_status = status or current.status
        if counterevidence_refs and status is None:
            resolved_status = IdentityClaimStatus.CONTESTED
        now = _now()
        revision = IdentityClaimRevision(
            revision_id=f"identity-claim-revision-{uuid4()}",
            from_revision=current.revision,
            to_revision=current.revision + 1,
            reason_code=reason_code,
            evidence_refs=evidence_refs,
            counterevidence_refs=counterevidence_refs,
            previous_status=current.status,
            created_at=now,
        )
        updated = replace(
            current,
            confidence=confidence,
            evidence_refs=tuple(
                dict.fromkeys((*current.evidence_refs, *evidence_refs))
            ),
            counterevidence_refs=tuple(
                dict.fromkeys((*current.counterevidence_refs, *counterevidence_refs))
            ),
            status=resolved_status,
            updated_at=now,
            revision=revision.to_revision,
            revisions=(*current.revisions, revision),
        )
        self.claims[claim_id] = updated
        return updated

    def link_continuity(
        self,
        earlier_ref: str,
        later_ref: str,
        *,
        relation_code: str,
        evidence_refs: tuple[str, ...],
        confidence: float,
        link_id: str | None = None,
    ) -> ContinuityLink:
        identifier = link_id or f"continuity-link-{uuid4()}"
        link = ContinuityLink(
            link_id=identifier,
            earlier_ref=earlier_ref,
            later_ref=later_ref,
            relation_code=relation_code,
            evidence_refs=evidence_refs,
            confidence=confidence,
            created_at=_now(),
        )
        if identifier in self.continuity_links:
            raise ValueError(f"Continuity link already exists: {identifier}")
        self.continuity_links[identifier] = link
        return link

    def set_future_self(
        self,
        *,
        description: str,
        theme_codes: tuple[str, ...],
        desired_level: float,
        current_level: float,
        evidence_refs: tuple[str, ...],
        projection_id: str | None = None,
    ) -> FutureSelfProjection:
        identifier = projection_id or f"future-self-{uuid4()}"
        now = _now()
        current = self.future_self.get(identifier)
        projection = FutureSelfProjection(
            projection_id=identifier,
            description=description,
            theme_codes=tuple(dict.fromkeys(theme_codes)),
            desired_level=desired_level,
            current_level=current_level,
            gap=max(0.0, desired_level - current_level),
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
            related_motivation_ids=()
            if current is None
            else current.related_motivation_ids,
            created_at=now if current is None else current.created_at,
            updated_at=now,
        )
        self.future_self[identifier] = projection
        return projection

    def link_future_motivation(
        self, projection_id: str, motivation_id: str
    ) -> FutureSelfProjection:
        current = self.future_self.get(projection_id)
        if current is None:
            raise ValueError(f"Unknown future-self projection: {projection_id}")
        updated = replace(
            current,
            related_motivation_ids=tuple(
                dict.fromkeys((*current.related_motivation_ids, motivation_id))
            ),
            updated_at=_now(),
        )
        self.future_self[projection_id] = updated
        return updated

    def select_relevant(
        self, *, theme_codes: Iterable[str] = (), capability_ids: Iterable[str] = ()
    ) -> NarrativeSelection:
        themes = set(theme_codes) | set(capability_ids)
        claims = tuple(
            claim
            for claim in self.claims.values()
            if themes.intersection(claim.theme_codes)
            and claim.status != IdentityClaimStatus.REVISED
        )
        episode_ids = tuple(
            dict.fromkeys(
                self._experience_index[item]
                for claim in claims
                for item in claim.related_experience_ids
                if item in self._experience_index
            )
        )
        return NarrativeSelection(
            claim_ids=tuple(claim.claim_id for claim in claims),
            episode_ids=episode_ids,
            rendered_items=tuple(
                f"Identity {claim.kind.value} ({claim.status.value}, confidence={claim.confidence:.3f}): {claim.statement}"
                for claim in claims
            ),
        )

    def get_episode(self, episode_id: str) -> AutobiographicalEpisode:
        try:
            return self.episodes[episode_id]
        except KeyError as exc:
            raise ValueError(f"Unknown autobiographical episode: {episode_id}") from exc

    def get_claim(self, claim_id: str) -> IdentityClaim:
        try:
            return self.claims[claim_id]
        except KeyError as exc:
            raise ValueError(f"Unknown identity claim: {claim_id}") from exc

    def record_commitment_event(
        self,
        commitment_ref: str,
        *,
        kind: str,
        description: str,
        evidence_refs: tuple[str, ...],
        relationship_refs: tuple[str, ...] = (),
    ) -> NarrativeCommitmentEvent:
        event = NarrativeCommitmentEvent(
            event_id=f"narrative-commitment-{uuid4()}",
            commitment_ref=commitment_ref,
            kind=kind,
            description=description,
            evidence_refs=tuple(dict.fromkeys(evidence_refs)),
            relationship_refs=tuple(dict.fromkeys(relationship_refs)),
            created_at=_now(),
        )
        self.commitment_events[event.event_id] = event
        return event

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "episode_threshold": self.episode_threshold,
            "episodes": [asdict(item) for item in self.episodes.values()],
            "chapters": [asdict(item) for item in self.chapters.values()],
            "identity_claims": [asdict(item) for item in self.claims.values()],
            "continuity_links": [
                asdict(item) for item in self.continuity_links.values()
            ],
            "unresolved_conflicts": [asdict(item) for item in self.conflicts.values()],
            "future_self": [asdict(item) for item in self.future_self.values()],
            "commitment_events": [
                asdict(item) for item in self.commitment_events.values()
            ],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.episodes = {}
            self.chapters = {}
            self.claims = {}
            self.continuity_links = {}
            self.conflicts = {}
            self.future_self = {}
            self.commitment_events = {}
            self._experience_index = {}
            return
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported narrative-self schema version")
        threshold = float(payload.get("episode_threshold", self.episode_threshold))
        _unit(threshold, "episode threshold")
        episodes = [_episode_from_json(item) for item in payload.get("episodes", [])]
        chapters = [_chapter_from_json(item) for item in payload.get("chapters", [])]
        claims = [_claim_from_json(item) for item in payload.get("identity_claims", [])]
        links = [_link_from_json(item) for item in payload.get("continuity_links", [])]
        conflicts = [
            _conflict_from_json(item)
            for item in payload.get("unresolved_conflicts", [])
        ]
        projections = [
            _future_from_json(item) for item in payload.get("future_self", [])
        ]
        commitment_events = [
            NarrativeCommitmentEvent(
                **{
                    **item,
                    "evidence_refs": tuple(item.get("evidence_refs", ())),
                    "relationship_refs": tuple(item.get("relationship_refs", ())),
                }
            )
            for item in payload.get("commitment_events", [])
        ]
        _require_unique(episodes, "episode_id")
        _require_unique(chapters, "chapter_id")
        _require_unique(claims, "claim_id")
        self.episode_threshold = threshold
        self.episodes = {item.episode_id: item for item in episodes}
        self.chapters = {item.chapter_id: item for item in chapters}
        self.claims = {item.claim_id: item for item in claims}
        self.continuity_links = {item.link_id: item for item in links}
        self.conflicts = {item.conflict_id: item for item in conflicts}
        self.future_self = {item.projection_id: item for item in projections}
        self.commitment_events = {item.event_id: item for item in commitment_events}
        self._experience_index = {
            experience_id: episode.episode_id
            for episode in episodes
            for experience_id in episode.experience_ids
        }

    def _integrate_chapters(self, episode: AutobiographicalEpisode) -> None:
        best_theme = next(iter(episode.theme_codes), None)
        if best_theme is None:
            return
        chapter = next(
            (item for item in self.chapters.values() if best_theme in item.theme_codes),
            None,
        )
        if chapter is not None:
            episodes = (*chapter.episode_ids, episode.episode_id)
            values = [self.get_episode(item) for item in episodes]
            self.chapters[chapter.chapter_id] = replace(
                chapter,
                episode_ids=episodes,
                start_at=min(item.occurred_at for item in values),
                end_at=max(item.occurred_at for item in values),
                revision=chapter.revision + 1,
            )
            return
        peers = [
            item
            for item in self.episodes.values()
            if item.episode_id != episode.episode_id and best_theme in item.theme_codes
        ]
        if peers:
            self.create_chapter(
                title=f"Theme: {best_theme}",
                theme_codes=(best_theme,),
                episode_ids=(peers[-1].episode_id, episode.episode_id),
            )

    def _register_conflict(
        self, claim: IdentityClaim, conflicting_ids: tuple[str, ...]
    ) -> None:
        all_claim_ids = tuple(sorted((claim.claim_id, *conflicting_ids)))
        for conflicting_id in conflicting_ids:
            current = self.claims[conflicting_id]
            if current.status != IdentityClaimStatus.CONTESTED:
                self.claims[conflicting_id] = replace(
                    current,
                    status=IdentityClaimStatus.CONTESTED,
                    conflicting_claim_ids=tuple(
                        dict.fromkeys((*current.conflicting_claim_ids, claim.claim_id))
                    ),
                    updated_at=_now(),
                )
        conflict = UnresolvedSelfConflict(
            conflict_id=f"self-conflict-{uuid4()}",
            claim_ids=all_claim_ids,
            evidence_refs=tuple(
                dict.fromkeys(
                    reference
                    for claim_id in all_claim_ids
                    for reference in self.claims[claim_id].evidence_refs
                )
            ),
            description="Conflicting identity claims remain unresolved",
            created_at=_now(),
        )
        self.conflicts[conflict.conflict_id] = conflict


def _episode_from_json(payload: dict[str, Any]) -> AutobiographicalEpisode:
    data = dict(payload)
    for name in (
        "experience_ids",
        "theme_codes",
        "related_value_refs",
        "related_goal_refs",
        "related_decision_refs",
        "related_commitment_refs",
        "turning_point_codes",
    ):
        data[name] = tuple(data.get(name, ()))
    return AutobiographicalEpisode(**data)


def _chapter_from_json(payload: dict[str, Any]) -> NarrativeChapter:
    data = dict(payload)
    data["theme_codes"] = tuple(data.get("theme_codes", ()))
    data["episode_ids"] = tuple(data.get("episode_ids", ()))
    return NarrativeChapter(**data)


def _claim_from_json(payload: dict[str, Any]) -> IdentityClaim:
    data = dict(payload)
    data["kind"] = IdentityClaimKind(data["kind"])
    data["status"] = IdentityClaimStatus(data["status"])
    for name in (
        "theme_codes",
        "evidence_refs",
        "counterevidence_refs",
        "conflicting_claim_ids",
        "related_experience_ids",
        "related_value_refs",
        "related_goal_refs",
        "related_decision_refs",
    ):
        data[name] = tuple(data.get(name, ()))
    data["revisions"] = tuple(
        IdentityClaimRevision(
            **{
                **item,
                "evidence_refs": tuple(item.get("evidence_refs", ())),
                "counterevidence_refs": tuple(item.get("counterevidence_refs", ())),
                "previous_status": IdentityClaimStatus(item["previous_status"]),
            }
        )
        for item in data.get("revisions", ())
    )
    return IdentityClaim(**data)


def _link_from_json(payload: dict[str, Any]) -> ContinuityLink:
    return ContinuityLink(
        **{**payload, "evidence_refs": tuple(payload.get("evidence_refs", ()))}
    )


def _conflict_from_json(payload: dict[str, Any]) -> UnresolvedSelfConflict:
    return UnresolvedSelfConflict(
        **{
            **payload,
            "claim_ids": tuple(payload.get("claim_ids", ())),
            "evidence_refs": tuple(payload.get("evidence_refs", ())),
        }
    )


def _future_from_json(payload: dict[str, Any]) -> FutureSelfProjection:
    data = dict(payload)
    for name in ("theme_codes", "evidence_refs", "related_motivation_ids"):
        data[name] = tuple(data.get(name, ()))
    return FutureSelfProjection(**data)


def _require_unique(values: list[Any], attribute: str) -> None:
    identifiers = [getattr(item, attribute) for item in values]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Narrative {attribute} values must be unique")


def _unit(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")


def _contains_private_key(value: Any) -> bool:
    private = {
        "hiddenthought",
        "prompt",
        "rawprompt",
        "reasoning",
        "chainofthought",
    }
    if isinstance(value, dict):
        return any(
            "".join(character for character in str(key).lower() if character.isalnum())
            in private
            or _contains_private_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_private_key(item) for item in value)
    return False


def _now() -> str:
    return datetime.now(UTC).isoformat()
