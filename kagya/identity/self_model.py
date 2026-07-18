"""Evidence-bound capabilities and revision-controlled identity state."""

from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
import math
from typing import Any, Iterable
from uuid import uuid4

from kagya.decision import ActionCandidate, DecisionRecord, DecisionStatus
from kagya.identity.origin import (
    IdentityOrigin,
    OriginActor,
    OriginInputKind,
    identity_origin_from_json,
    new_identity_origin,
)


class ProposalStatus(StrEnum):
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"


@dataclass(frozen=True)
class CapabilityEvidence:
    evidence_id: str
    evidence_type: str
    source_id: str
    success: bool | None
    utility: float | None
    created_at: str

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.evidence_type or not self.source_id:
            raise ValueError("Capability evidence identifiers must not be empty")
        if self.utility is not None:
            _signed(self.utility, "evidence utility")


@dataclass(frozen=True)
class Capability:
    capability_id: str
    description: str
    confidence: float
    stability: float
    tags: tuple[str, ...]
    evidence: tuple[CapabilityEvidence, ...]
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.capability_id or not self.description:
            raise ValueError("Capability ID and description must not be empty")
        _unit(self.confidence, "capability confidence")
        _unit(self.stability, "capability stability")


@dataclass(frozen=True)
class KnownLimitation:
    limitation_id: str
    description: str
    confidence: float
    capability_ids: tuple[str, ...]
    tags: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.limitation_id or not self.description:
            raise ValueError("Limitation ID and description must not be empty")
        _unit(self.confidence, "limitation confidence")


@dataclass(frozen=True)
class EpistemicUncertainty:
    uncertainty_id: str
    description: str
    confidence: float
    tags: tuple[str, ...]
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.uncertainty_id or not self.description:
            raise ValueError("Uncertainty ID and description must not be empty")
        _unit(self.confidence, "uncertainty confidence")


@dataclass(frozen=True)
class IdentityRevisionProposal:
    proposal_id: str
    proposed_summary: str | None
    proposed_traits: dict[str, float]
    evidence_refs: tuple[str, ...]
    source: str
    identity_origin: IdentityOrigin
    contradictions: tuple[str, ...]
    status: ProposalStatus
    created_at: str
    resolved_at: str | None = None


@dataclass(frozen=True)
class SelfModelUpdateRecord:
    update_id: str
    operation: str
    reason: str
    evidence_refs: tuple[str, ...]
    before: dict[str, Any]
    after: dict[str, Any]
    revision: int
    event_id: str | None
    event_sequence: int | None
    created_at: str
    rollback_target_revision: int | None = None


@dataclass(frozen=True)
class SelfModelState:
    identity_summary: str
    traits: dict[str, float]
    capabilities: dict[str, Capability]
    known_limitations: dict[str, KnownLimitation]
    epistemic_uncertainties: dict[str, EpistemicUncertainty]
    roles: tuple[str, ...]
    commitment_refs: tuple[str, ...]
    autobiographical_summary_refs: tuple[str, ...]
    revision: int
    updated_at: str
    value_revision_refs: dict[str, int] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if not self.identity_summary:
            raise ValueError("Identity summary must not be empty")
        if self.schema_version not in {1, 2}:
            raise ValueError(
                f"Unsupported self-model schema version: {self.schema_version}"
            )
        for name, value in self.traits.items():
            if not name:
                raise ValueError("Trait names must not be empty")
            _signed(value, "trait")


@dataclass(frozen=True)
class SelfModelSelection:
    capability_ids: tuple[str, ...]
    limitation_ids: tuple[str, ...]
    uncertainty_ids: tuple[str, ...]
    contributions: dict[str, float]
    rendered_items: tuple[str, ...]


class SelfModel:
    def __init__(
        self,
        state: SelfModelState | None = None,
        *,
        max_capability_update: float = 0.1,
        max_trait_update: float = 0.1,
    ) -> None:
        _unit(max_capability_update, "max capability update")
        _unit(max_trait_update, "max trait update")
        self.max_capability_update = max_capability_update
        self.max_trait_update = max_trait_update
        self._default_state = state or SelfModelState(
            identity_summary="PROJECT-KAGYA persistent subject",
            traits={},
            capabilities={},
            known_limitations={},
            epistemic_uncertainties={},
            roles=("private_local_assistant",),
            commitment_refs=(),
            autobiographical_summary_refs=(),
            revision=0,
            updated_at=_now(),
        )
        self.state = self._default_state
        self.history: list[SelfModelUpdateRecord] = []
        self.proposals: dict[str, IdentityRevisionProposal] = {}

    def update_capability_from_decision(
        self,
        capability_id: str,
        description: str,
        decision: DecisionRecord,
        *,
        tags: tuple[str, ...] = (),
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Capability:
        if (
            decision.status != DecisionStatus.RESOLVED
            or decision.actual_outcome is None
        ):
            raise ValueError("Capability updates require a resolved DecisionRecord")
        selected = next(
            item.candidate
            for item in decision.considered_candidates
            if item.candidate.candidate_id == decision.selected_candidate_id
        )
        declared_capabilities = _string_tuple(selected.parameters.get("capability_ids"))
        if capability_id not in declared_capabilities:
            raise ValueError(
                "Selected action did not declare the capability being updated"
            )
        evidence_id = f"decision:{decision.decision_id}"
        current = self.state.capabilities.get(
            capability_id,
            Capability(
                capability_id=capability_id,
                description=description,
                confidence=0.5,
                stability=0.7,
                tags=tags,
                evidence=(),
            ),
        )
        if any(item.evidence_id == evidence_id for item in current.evidence):
            return current
        outcome = decision.actual_outcome
        direction = 1.0 if outcome.success else -1.0
        strength = max(0.2, abs(outcome.utility))
        cap = self.max_capability_update * (1.0 - current.stability)
        delta = direction * min(cap, strength * cap)
        evidence = CapabilityEvidence(
            evidence_id=evidence_id,
            evidence_type="decision_outcome",
            source_id=decision.decision_id,
            success=outcome.success,
            utility=outcome.utility,
            created_at=_now(),
        )
        updated = replace(
            current,
            description=description,
            confidence=max(0.0, min(1.0, current.confidence + delta)),
            tags=tuple(sorted(set((*current.tags, *tags)))),
            evidence=(*current.evidence, evidence),
            revision=current.revision + 1,
        )
        self._replace_capability(
            updated,
            operation="capability_evidence_update",
            reason="resolved_decision_outcome",
            evidence_refs=(evidence_id,),
            event_id=event_id,
            event_sequence=event_sequence,
        )
        return updated

    def manual_correct_capability(
        self,
        capability_id: str,
        description: str,
        confidence: float,
        *,
        reason: str,
        tags: tuple[str, ...] = (),
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> Capability:
        _unit(confidence, "capability confidence")
        if not reason:
            raise ValueError("Manual correction reason must not be empty")
        current = self.state.capabilities.get(
            capability_id,
            Capability(
                capability_id=capability_id,
                description=description,
                confidence=0.5,
                stability=0.7,
                tags=tags,
                evidence=(),
            ),
        )
        evidence = CapabilityEvidence(
            evidence_id=f"manual:{uuid4()}",
            evidence_type="manual_correction",
            source_id=reason,
            success=None,
            utility=None,
            created_at=_now(),
        )
        updated = replace(
            current,
            description=description,
            confidence=confidence,
            tags=tuple(sorted(set((*current.tags, *tags)))),
            evidence=(*current.evidence, evidence),
            revision=current.revision + 1,
        )
        self._replace_capability(
            updated,
            operation="manual_capability_correction",
            reason=reason,
            evidence_refs=(evidence.evidence_id,),
            event_id=event_id,
            event_sequence=event_sequence,
        )
        return updated

    def add_limitation(
        self,
        limitation: KnownLimitation,
        *,
        reason: str,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> KnownLimitation:
        before = self._snapshot()
        limitations = dict(self.state.known_limitations)
        limitations[limitation.limitation_id] = limitation
        self.state = replace(
            self.state,
            known_limitations=limitations,
            revision=self.state.revision + 1,
            updated_at=_now(),
        )
        self._record(
            "limitation_update",
            reason,
            limitation.evidence_refs,
            before,
            event_id,
            event_sequence,
        )
        return limitation

    def add_uncertainty(
        self,
        uncertainty: EpistemicUncertainty,
        *,
        reason: str,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> EpistemicUncertainty:
        before = self._snapshot()
        uncertainties = dict(self.state.epistemic_uncertainties)
        uncertainties[uncertainty.uncertainty_id] = uncertainty
        self.state = replace(
            self.state,
            epistemic_uncertainties=uncertainties,
            revision=self.state.revision + 1,
            updated_at=_now(),
        )
        self._record(
            "uncertainty_update",
            reason,
            uncertainty.evidence_refs,
            before,
            event_id,
            event_sequence,
        )
        return uncertainty

    def propose_identity_revision(
        self,
        *,
        proposed_summary: str | None,
        proposed_traits: dict[str, float],
        evidence_refs: tuple[str, ...],
        source: str,
        identity_origin: IdentityOrigin | None = None,
        proposal_id: str | None = None,
    ) -> IdentityRevisionProposal:
        if proposed_summary is None and not proposed_traits:
            raise ValueError("Identity revision proposal must contain a change")
        if not source:
            raise ValueError("Identity revision source must not be empty")
        for value in proposed_traits.values():
            _signed(value, "trait")
        if any(not name for name in proposed_traits):
            raise ValueError("Trait names must not be empty")
        identifier = proposal_id or str(uuid4())
        if identifier in self.proposals:
            raise ValueError(f"Identity proposal already exists: {identifier}")
        contradictions: list[str] = []
        if (
            proposed_summary is not None
            and proposed_summary != self.state.identity_summary
        ):
            contradictions.append("identity_summary_changed")
        for name, value in proposed_traits.items():
            current = self.state.traits.get(name)
            if current is not None and abs(value - current) > self.max_trait_update:
                contradictions.append(f"trait_change_exceeds_inertia:{name}")
            if current is not None and current * value < 0:
                contradictions.append(f"trait_direction_conflict:{name}")
        proposal = IdentityRevisionProposal(
            proposal_id=identifier,
            proposed_summary=proposed_summary,
            proposed_traits=dict(proposed_traits),
            evidence_refs=evidence_refs,
            source=source,
            identity_origin=identity_origin
            or new_identity_origin(
                OriginActor.MODEL_INFERENCE,
                OriginInputKind.SUGGESTION,
                source_ref="self_model_proposal",
                confidence=0.5,
            ),
            contradictions=tuple(contradictions),
            status=ProposalStatus.PENDING,
            created_at=_now(),
        )
        self.proposals[identifier] = proposal
        return proposal

    def resolve_identity_revision(
        self,
        proposal_id: str,
        *,
        apply: bool,
        reason: str,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> IdentityRevisionProposal:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            raise ValueError(f"Unknown identity proposal: {proposal_id}")
        if proposal.status != ProposalStatus.PENDING:
            raise ValueError("Identity proposal is already resolved")
        if not reason:
            raise ValueError("Identity proposal resolution reason must not be empty")
        if apply:
            before = self._snapshot()
            traits = dict(self.state.traits)
            for name, requested in proposal.proposed_traits.items():
                current = traits.get(name, 0.0)
                delta = max(
                    -self.max_trait_update,
                    min(self.max_trait_update, requested - current),
                )
                traits[name] = max(-1.0, min(1.0, current + delta))
            self.state = replace(
                self.state,
                identity_summary=proposal.proposed_summary
                or self.state.identity_summary,
                traits=traits,
                revision=self.state.revision + 1,
                updated_at=_now(),
            )
            self._record(
                "identity_revision",
                reason,
                proposal.evidence_refs,
                before,
                event_id,
                event_sequence,
            )
        resolved = replace(
            proposal,
            status=ProposalStatus.APPLIED if apply else ProposalStatus.REJECTED,
            identity_origin=(
                proposal.identity_origin.endorse(
                    "identity_revision_approved",
                    event_id=event_id,
                    event_sequence=event_sequence,
                )
                if apply
                else proposal.identity_origin.reject(
                    "identity_revision_rejected",
                    event_id=event_id,
                    event_sequence=event_sequence,
                )
            ),
            resolved_at=_now(),
        )
        self.proposals[proposal_id] = resolved
        return resolved

    def sync_references(
        self,
        *,
        commitment_refs: Iterable[str],
        autobiographical_summary_refs: Iterable[str] | None = None,
        value_revision_refs: dict[str, int] | None = None,
    ) -> None:
        commitments = tuple(sorted(set(commitment_refs)))
        autobiography = (
            self.state.autobiographical_summary_refs
            if autobiographical_summary_refs is None
            else tuple(sorted(set(autobiographical_summary_refs)))
        )
        values = (
            self.state.value_revision_refs
            if value_revision_refs is None
            else dict(value_revision_refs)
        )
        if (
            commitments == self.state.commitment_refs
            and autobiography == self.state.autobiographical_summary_refs
            and values == self.state.value_revision_refs
        ):
            return
        self.state = replace(
            self.state,
            commitment_refs=commitments,
            autobiographical_summary_refs=autobiography,
            value_revision_refs=values,
            updated_at=_now(),
        )

    def select_relevant(self, candidate: ActionCandidate) -> SelfModelSelection:
        capability_ids = _string_tuple(candidate.parameters.get("capability_ids"))
        tags = set(_string_tuple(candidate.parameters.get("topic_tags")))
        capabilities = tuple(
            item
            for item in self.state.capabilities.values()
            if item.capability_id in capability_ids or tags.intersection(item.tags)
        )
        limitations = tuple(
            item
            for item in self.state.known_limitations.values()
            if set(item.capability_ids).intersection(capability_ids)
            or tags.intersection(item.tags)
        )
        uncertainties = tuple(
            item
            for item in self.state.epistemic_uncertainties.values()
            if tags.intersection(item.tags)
        )
        contributions = {
            **{
                f"capability:{item.capability_id}": 0.5 * (item.confidence - 0.5)
                for item in capabilities
            },
            **{
                f"limitation:{item.limitation_id}": -0.5 * item.confidence
                for item in limitations
            },
            **{
                f"uncertainty:{item.uncertainty_id}": -0.4 * item.confidence
                for item in uncertainties
            },
        }
        rendered = tuple(
            [
                f"Capability {item.description}: confidence={item.confidence:.3f}"
                for item in capabilities
            ]
            + [
                f"Known limitation {item.description}: confidence={item.confidence:.3f}"
                for item in limitations
            ]
            + [
                f"Known unknown {item.description}: confidence={item.confidence:.3f}"
                for item in uncertainties
            ]
        )
        return SelfModelSelection(
            capability_ids=tuple(item.capability_id for item in capabilities),
            limitation_ids=tuple(item.limitation_id for item in limitations),
            uncertainty_ids=tuple(item.uncertainty_id for item in uncertainties),
            contributions=contributions,
            rendered_items=rendered,
        )

    def evaluate_candidates(
        self, candidates: Iterable[ActionCandidate]
    ) -> dict[str, dict[str, float]]:
        return {
            candidate.candidate_id: self.select_relevant(candidate).contributions
            for candidate in candidates
        }

    def rollback(
        self,
        target_revision: int,
        *,
        reason: str,
        event_id: str | None = None,
        event_sequence: int | None = None,
    ) -> SelfModelState:
        snapshots = [
            record.before
            for record in self.history
            if int(record.before.get("revision", -1)) == target_revision
        ]
        snapshots.extend(
            record.after
            for record in self.history
            if int(record.after.get("revision", -1)) == target_revision
        )
        if not snapshots:
            raise ValueError(f"Unknown self-model revision: {target_revision}")
        before = self._snapshot()
        restored = _state_from_json(snapshots[-1])
        self.state = replace(
            restored,
            revision=before["revision"] + 1,
            updated_at=_now(),
        )
        self._record(
            "rollback",
            reason,
            (),
            before,
            event_id,
            event_sequence,
            rollback_target_revision=target_revision,
        )
        return self.state

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "state": asdict(self.state),
            "history": [asdict(item) for item in self.history],
            "proposals": [asdict(item) for item in self.proposals.values()],
        }

    def restore(self, payload: object) -> None:
        if not isinstance(payload, dict) or not payload:
            self.state = self._default_state
            self.history = []
            self.proposals = {}
            return
        if payload.get("schema_version") not in {1, 2} or not isinstance(
            payload.get("state"), dict
        ):
            self._restore_legacy(payload)
            return
        self.state = _state_from_json(payload["state"])
        self.history = [
            SelfModelUpdateRecord(**item)
            for item in payload.get("history", [])
            if isinstance(item, dict)
        ]
        proposals = [
            _proposal_from_json(item)
            for item in payload.get("proposals", [])
            if isinstance(item, dict)
        ]
        self.proposals = {item.proposal_id: item for item in proposals}

    def _replace_capability(
        self,
        capability: Capability,
        *,
        operation: str,
        reason: str,
        evidence_refs: tuple[str, ...],
        event_id: str | None,
        event_sequence: int | None,
    ) -> None:
        before = self._snapshot()
        capabilities = dict(self.state.capabilities)
        capabilities[capability.capability_id] = capability
        self.state = replace(
            self.state,
            capabilities=capabilities,
            revision=self.state.revision + 1,
            updated_at=_now(),
        )
        self._record(
            operation,
            reason,
            evidence_refs,
            before,
            event_id,
            event_sequence,
        )

    def _record(
        self,
        operation: str,
        reason: str,
        evidence_refs: tuple[str, ...],
        before: dict[str, Any],
        event_id: str | None,
        event_sequence: int | None,
        *,
        rollback_target_revision: int | None = None,
    ) -> None:
        self.history.append(
            SelfModelUpdateRecord(
                update_id=str(uuid4()),
                operation=operation,
                reason=reason,
                evidence_refs=evidence_refs,
                before=before,
                after=self._snapshot(),
                revision=self.state.revision,
                event_id=event_id,
                event_sequence=event_sequence,
                created_at=_now(),
                rollback_target_revision=rollback_target_revision,
            )
        )

    def _snapshot(self) -> dict[str, Any]:
        return asdict(self.state)

    def _restore_legacy(self, payload: dict[str, Any]) -> None:
        summary = payload.get("identity_summary")
        traits = {
            key: float(value)
            for key, value in payload.items()
            if key != "identity_summary"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and -1.0 <= value <= 1.0
        }
        self.state = replace(
            self._default_state,
            identity_summary=summary
            if isinstance(summary, str) and summary
            else self._default_state.identity_summary,
            traits=traits,
            updated_at=_now(),
        )
        self.history = []
        self.proposals = {}


def _state_from_json(payload: dict[str, Any]) -> SelfModelState:
    data = dict(payload)
    data["capabilities"] = {
        key: _capability_from_json(value)
        for key, value in data.get("capabilities", {}).items()
    }
    data["known_limitations"] = {
        key: KnownLimitation(
            **{
                **value,
                "capability_ids": tuple(value.get("capability_ids", ())),
                "tags": tuple(value.get("tags", ())),
                "evidence_refs": tuple(value.get("evidence_refs", ())),
            }
        )
        for key, value in data.get("known_limitations", {}).items()
    }
    data["epistemic_uncertainties"] = {
        key: EpistemicUncertainty(
            **{
                **value,
                "tags": tuple(value.get("tags", ())),
                "evidence_refs": tuple(value.get("evidence_refs", ())),
            }
        )
        for key, value in data.get("epistemic_uncertainties", {}).items()
    }
    for field_name in (
        "roles",
        "commitment_refs",
        "autobiographical_summary_refs",
    ):
        data[field_name] = tuple(data.get(field_name, ()))
    data.setdefault("value_revision_refs", {})
    data["schema_version"] = 2
    return SelfModelState(**data)


def _capability_from_json(payload: dict[str, Any]) -> Capability:
    data = dict(payload)
    data["tags"] = tuple(data.get("tags", ()))
    data["evidence"] = tuple(
        CapabilityEvidence(**item) for item in data.get("evidence", ())
    )
    return Capability(**data)


def _proposal_from_json(payload: dict[str, Any]) -> IdentityRevisionProposal:
    data = dict(payload)
    data["evidence_refs"] = tuple(data.get("evidence_refs", ()))
    data["contradictions"] = tuple(data.get("contradictions", ()))
    data["status"] = ProposalStatus(data["status"])
    data["identity_origin"] = identity_origin_from_json(
        data.get("identity_origin"), fallback_source="legacy_identity_proposal"
    )
    return IdentityRevisionProposal(**data)


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def _unit(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between zero and one")


def _signed(value: float, name: str) -> None:
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and between minus one and one")


def _now() -> str:
    return datetime.now(UTC).isoformat()
