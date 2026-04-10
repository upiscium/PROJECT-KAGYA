from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class DriftSnapshot:
    state: dict[str, Any]


@dataclass(slots=True)
class DriftReport:
    output_tone_shift: float
    output_length_shift: float
    vocabulary_shift: float
    known_sample_shift: float
    thought_shift: float
    iterations: int


class DriftAwareModelProtocol(Protocol):
    def snapshot_state(self) -> dict[str, Any]: ...

    def restore_state(self, snapshot: dict[str, Any]) -> None: ...


class DriftController:
    def __init__(
        self,
        tone_threshold: float = 0.2,
        length_threshold: float = 0.25,
        vocabulary_threshold: float = 0.3,
        known_sample_threshold: float = 0.35,
        thought_threshold: float = 0.25,
        iteration_penalty: float = 0.05,
    ) -> None:
        self.tone_threshold = tone_threshold
        self.length_threshold = length_threshold
        self.vocabulary_threshold = vocabulary_threshold
        self.known_sample_threshold = known_sample_threshold
        self.thought_threshold = thought_threshold
        self.iteration_penalty = iteration_penalty

    def snapshot(self, model: DriftAwareModelProtocol) -> DriftSnapshot:
        return DriftSnapshot(state=model.snapshot_state())

    def measure_drift(self, before: DriftSnapshot, after: DriftSnapshot) -> DriftReport:
        return DriftReport(
            output_tone_shift=self._numeric_delta(
                before.state.get("tone"), after.state.get("tone")
            ),
            output_length_shift=self._numeric_delta(
                before.state.get("length"), after.state.get("length")
            ),
            vocabulary_shift=self._numeric_delta(
                before.state.get("vocabulary"), after.state.get("vocabulary")
            ),
            known_sample_shift=self._numeric_delta(
                before.state.get("known_sample"), after.state.get("known_sample")
            ),
            thought_shift=self._numeric_delta(
                before.state.get("thought"), after.state.get("thought")
            ),
            iterations=int(after.state.get("iterations", 0)),
        )

    def should_accept_update(self, drift_report: DriftReport) -> bool:
        if drift_report.iterations > 0:
            iteration_limit = self.iteration_penalty * drift_report.iterations
        else:
            iteration_limit = 0.0

        tone_limit = max(0.0, self.tone_threshold - iteration_limit)
        length_limit = max(0.0, self.length_threshold - iteration_limit)
        vocabulary_limit = max(0.0, self.vocabulary_threshold - iteration_limit)
        known_sample_limit = max(0.0, self.known_sample_threshold - iteration_limit)
        thought_limit = max(0.0, self.thought_threshold - iteration_limit)

        return all(
            [
                drift_report.output_tone_shift <= tone_limit,
                drift_report.output_length_shift <= length_limit,
                drift_report.vocabulary_shift <= vocabulary_limit,
                drift_report.known_sample_shift <= known_sample_limit,
                drift_report.thought_shift <= thought_limit,
            ]
        )

    def rollback(self, model: DriftAwareModelProtocol, snapshot: DriftSnapshot) -> None:
        model.restore_state(snapshot.state)

    @staticmethod
    def _numeric_delta(before: Any, after: Any) -> float:
        if not DriftController._is_number(before) or not DriftController._is_number(
            after
        ):
            return 1.0
        return abs(float(after) - float(before))

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
