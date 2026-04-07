from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol, Sequence
from uuid import uuid4


@dataclass(slots=True)
class EpisodicRecord:
    id: str
    user_input: str
    response: str
    valence: float
    arousal: float
    timestamp: str
    source: str = "chat"
    confidence: float = 1.0


@dataclass(slots=True)
class SemanticRecord:
    id: str
    fact: str
    source_ids: list[str]
    confidence: float
    status: str
    timestamp: str


@dataclass(slots=True)
class ConsolidationResult:
    migrated: int = 0
    pending: int = 0
    failed: int = 0


class MemoryCollection(Protocol):
    def add(self, record: EpisodicRecord | SemanticRecord) -> None: ...

    def search(
        self, query: str, top_k: int
    ) -> list[EpisodicRecord | SemanticRecord]: ...

    def delete(self, record_id: str) -> None: ...

    def all(self) -> list[EpisodicRecord | SemanticRecord]: ...


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _tokenize(text: str) -> set[str]:
    return {token for token in text.lower().replace("\n", " ").split(" ") if token}


def _score_match(query: str, text: str) -> float:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 0.0
    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _priority_label(confidence: float, index: int) -> str:
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.4:
        return "medium"
    if index == 0:
        return "high"
    return "low"


def _normalize_status(status: Any) -> str:
    value = str(status).strip().lower()
    if value in {"confirmed", "tentative", "conflicted"}:
        return value
    return "tentative"


class InMemoryMemoryCollection:
    def __init__(self) -> None:
        self._records: list[EpisodicRecord | SemanticRecord] = []

    def add(self, record: EpisodicRecord | SemanticRecord) -> None:
        self._records.append(record)

    def search(self, query: str, top_k: int) -> list[EpisodicRecord | SemanticRecord]:
        scored = sorted(
            (
                (_score_match(query, _record_text(record)), index, record)
                for index, record in enumerate(self._records)
            ),
            key=lambda item: (-item[0], item[1]),
        )
        results = [record for score, _, record in scored if score > 0.0]
        return results[:top_k]

    def delete(self, record_id: str) -> None:
        self._records = [
            record for record in self._records if _record_id(record) != record_id
        ]

    def all(self) -> list[EpisodicRecord | SemanticRecord]:
        return list(self._records)


def _record_id(record: EpisodicRecord | SemanticRecord) -> str:
    return record.id


def _record_text(record: EpisodicRecord | SemanticRecord) -> str:
    if isinstance(record, EpisodicRecord):
        return " ".join([record.user_input, record.response])
    return record.fact


def _record_timestamp(record: EpisodicRecord | SemanticRecord) -> str:
    return record.timestamp


def _record_confidence(record: EpisodicRecord | SemanticRecord) -> float:
    return record.confidence


def _format_empty_block(title: str) -> str:
    return f"{title}\n- なし"


class DualMemorySystem:
    def __init__(
        self,
        hippocampus: MemoryCollection | None = None,
        cortex: MemoryCollection | None = None,
    ) -> None:
        self.hippocampus = (
            hippocampus if hippocampus is not None else InMemoryMemoryCollection()
        )
        self.cortex = cortex if cortex is not None else InMemoryMemoryCollection()

    def save_episodic(
        self,
        user_input: str,
        response: str,
        valence: float,
        arousal: float,
    ) -> EpisodicRecord:
        record = EpisodicRecord(
            id=str(uuid4()),
            user_input=user_input,
            response=response,
            valence=_clamp(valence, -1.0, 1.0),
            arousal=_clamp(arousal, 0.0, 1.0),
            timestamp=_utc_timestamp(),
            source="chat",
            confidence=1.0,
        )
        self.hippocampus.add(record)
        return record

    def retrieve_context(self, query: str, top_k: int = 3) -> str:
        recent = self.hippocampus.search(query, top_k)
        semantic = self.cortex.search(query, top_k)
        return "\n\n".join(
            [
                self._format_recent_memory(recent),
                self._format_semantic_memory(semantic),
                "Priority Notes\n- DB1 first for recency\n- DB2 only for confirmed facts",
            ]
        )

    def consolidate_to_semantic(self, llm_pipeline: Any) -> ConsolidationResult:
        episodes = self.hippocampus.all()
        if not episodes:
            return ConsolidationResult()

        result = ConsolidationResult()
        candidates = self._call_pipeline(llm_pipeline, episodes)
        for candidate in candidates:
            status = _normalize_status(candidate.get("status"))
            if status == "confirmed":
                semantic = SemanticRecord(
                    id=str(candidate.get("id", uuid4())),
                    fact=str(candidate.get("fact", "")),
                    source_ids=[
                        str(source_id) for source_id in candidate.get("source_ids", [])
                    ],
                    confidence=float(candidate.get("confidence", 0.0)),
                    status=status,
                    timestamp=str(candidate.get("timestamp", _utc_timestamp())),
                )
                self.cortex.add(semantic)
                for source_id in semantic.source_ids:
                    self.hippocampus.delete(source_id)
                result.migrated += 1
            elif status == "tentative":
                result.pending += 1
            else:
                result.failed += 1

        return result

    def _format_recent_memory(
        self, records: Sequence[EpisodicRecord | SemanticRecord]
    ) -> str:
        if not records:
            return _format_empty_block("Recent Memory (DB1)")
        lines = ["Recent Memory (DB1)"]
        for index, record in enumerate(records):
            if isinstance(record, EpisodicRecord):
                confidence = record.confidence
                content = record.user_input
            else:
                confidence = record.confidence
                content = record.fact
            lines.append(
                f"- [{_priority_label(confidence, index)}] {_record_timestamp(record)} {content}"
            )
        return "\n".join(lines)

    def _format_semantic_memory(
        self, records: Sequence[EpisodicRecord | SemanticRecord]
    ) -> str:
        if not records:
            return _format_empty_block("Semantic Memory (DB2)")
        lines = ["Semantic Memory (DB2)"]
        for record in records:
            status = (
                record.status if isinstance(record, SemanticRecord) else "tentative"
            )
            content = (
                record.fact
                if isinstance(record, SemanticRecord)
                else _record_text(record)
            )
            lines.append(f"- [{status}] {content}")
        return "\n".join(lines)

    def _call_pipeline(
        self,
        llm_pipeline: Any,
        episodes: Sequence[EpisodicRecord | SemanticRecord],
    ) -> list[dict[str, Any]]:
        raw = llm_pipeline(episodes)
        if raw is None:
            return []
        if isinstance(raw, dict):
            return [raw]
        return [dict(item) for item in raw]
