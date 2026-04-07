from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass
class MemoryRecord:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class _Collection:
    def __init__(self) -> None:
        self.records: dict[str, MemoryRecord] = {}

    def add(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
    ) -> None:
        for record_id, document, metadata in zip(
            ids, documents, metadatas, strict=True
        ):
            self.records[record_id] = MemoryRecord(
                id=record_id, text=document, metadata=metadata
            )

    def query(self, text: str, top_k: int) -> list[MemoryRecord]:
        terms = {token.lower() for token in text.split() if token}
        scored: list[tuple[int, MemoryRecord]] = []
        for record in self.records.values():
            record_terms = set(record.text.lower().split())
            score = len(terms & record_terms)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [record for _, record in scored[:top_k]]

    def delete(self, ids: list[str]) -> None:
        for record_id in ids:
            self.records.pop(record_id, None)


class DualMemorySystem:
    def __init__(self, top_k: int = 3) -> None:
        self.top_k = top_k
        self.hippocampus = _Collection()
        self.cortex = _Collection()

    def save_episodic(
        self, user_input: str, response: str, valence: float, arousal: float
    ) -> str:
        record_id = str(uuid4())
        text = f"User: {user_input}\nAssistant: {response}"
        metadata = {"type": "episodic", "valence": valence, "arousal": arousal}
        self.hippocampus.add([record_id], [text], [metadata])
        return record_id

    def retrieve_context(self, query: str) -> str:
        episodic = self.hippocampus.query(query, self.top_k)
        semantic = self.cortex.query(query, self.top_k)
        return self._format_context(episodic, semantic)

    def consolidate_to_semantic(self, llm_pipeline: Any) -> list[str]:
        moved_ids: list[str] = []
        for record in list(self.hippocampus.records.values()):
            prompt = record.text
            extracted = llm_pipeline(prompt)
            semantic_text = self._normalize_extraction(extracted)
            semantic_id = str(uuid4())
            self.cortex.add(
                [semantic_id],
                [semantic_text],
                [{"type": "semantic", **record.metadata}],
            )
            moved_ids.append(record.id)
        if moved_ids:
            self.hippocampus.delete(moved_ids)
        return moved_ids

    def _format_context(
        self, episodic: list[MemoryRecord], semantic: list[MemoryRecord]
    ) -> str:
        lines: list[str] = []
        if episodic:
            lines.append("[Episodic Memory]")
            lines.extend(self._format_record(record) for record in episodic)
        if semantic:
            lines.append("[Semantic Memory]")
            lines.extend(self._format_record(record) for record in semantic)
        return "\n".join(lines) if lines else "[No Relevant Memory]"

    def _format_record(self, record: MemoryRecord) -> str:
        metadata = ", ".join(
            f"{key}={value}" for key, value in sorted(record.metadata.items())
        )
        return f"- {record.text} ({metadata})" if metadata else f"- {record.text}"

    def _normalize_extraction(self, extracted: Any) -> str:
        if isinstance(extracted, str):
            return extracted.strip()
        if isinstance(extracted, dict):
            if "facts" in extracted:
                return str(extracted["facts"]).strip()
            if "text" in extracted:
                return str(extracted["text"]).strip()
        return str(extracted).strip()
