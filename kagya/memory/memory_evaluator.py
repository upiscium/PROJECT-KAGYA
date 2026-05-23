"""Memory selection helpers for consolidation."""

from kagya.memory.memory_schema import EpisodicMemoryRecord


class MemoryEvaluator:
    """Select DB1 episodes that are useful for semantic consolidation."""

    def __init__(self, min_arousal: float = 0.0) -> None:
        self.min_arousal = min_arousal

    def should_consolidate(self, record: EpisodicMemoryRecord) -> bool:
        return not record.archived and record.emotion_arousal >= self.min_arousal
