"""Memory selection helpers for consolidation."""

from kagya.memory.memory_schema import EpisodicMemoryRecord


class MemoryEvaluator:
    """Select DB1 episodes that are useful for semantic consolidation."""

    def __init__(
        self,
        min_arousal: float = 0.55,
        min_subjective_salience: float | None = None,
    ) -> None:
        if not 0.0 < min_arousal <= 1.0:
            raise ValueError("minimum consolidation arousal must be in (0, 1]")
        self.min_arousal = min_arousal
        self.min_subjective_salience = (
            min_arousal if min_subjective_salience is None else min_subjective_salience
        )
        if not 0.0 < self.min_subjective_salience <= 1.0:
            raise ValueError("minimum consolidation salience must be in (0, 1]")

    def should_consolidate(self, record: EpisodicMemoryRecord) -> bool:
        return not record.archived and (
            record.emotion_arousal >= self.min_arousal
            or record.subjective_salience >= self.min_subjective_salience
        )
