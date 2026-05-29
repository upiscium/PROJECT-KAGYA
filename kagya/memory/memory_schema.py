"""Data model for logical memory records."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryRecordType(StrEnum):
    EPISODIC_LOG = "episodic_log"
    THOUGHT_LOG = "thought_log"
    EXTRACTED_FACT = "extracted_fact"
    SEMANTIC_MEMORY = "semantic_memory"
    EVALUATION_LOG = "evaluation_log"


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    id: str
    user_input: str
    response: str
    hidden_thought: str = ""
    loss: float = 0.0
    emotion_valence: float = 0.0
    emotion_arousal: float = 0.0
    record_type: MemoryRecordType = MemoryRecordType.EPISODIC_LOG
    archived: bool = False
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticMemoryRecord:
    id: str
    text: str
    source_episode_ids: list[str] = field(default_factory=list)
    record_type: MemoryRecordType = MemoryRecordType.SEMANTIC_MEMORY
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryContext:
    db1_results: list[EpisodicMemoryRecord] = field(default_factory=list)
    db2_results: list[SemanticMemoryRecord] = field(default_factory=list)
