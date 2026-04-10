from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(slots=True)
class EvaluatedExample:
    input: str
    thought: str
    output: str
    source_ids: list[str]
    confidence: float
    status: str


@dataclass(slots=True)
class DataQualityReport:
    total_examples: int
    valid_examples: int
    invalid_examples: int
    empty_thoughts: int
    confirmed_examples: int
    low_confidence_examples: int
    duplicate_examples: int
    conflict_examples: int
    warnings: list[str]


class EvaluatedExampleProtocol(Protocol):
    input: str
    thought: str
    output: str
    source_ids: list[str]
    confidence: float
    status: str


class DataQualityEvaluator:
    def __init__(self, min_confidence: float = 0.5) -> None:
        self.min_confidence = min_confidence

    def evaluate(
        self, dataset: Sequence[EvaluatedExampleProtocol]
    ) -> DataQualityReport:
        valid_examples = 0
        invalid_examples = 0
        empty_thoughts = 0
        confirmed_examples = 0
        low_confidence_examples = 0
        conflict_examples = 0
        duplicate_examples = 0
        warnings: list[str] = []
        seen_signatures: set[tuple[str, str, str]] = set()

        for example in dataset:
            if not self._is_shape_valid(example):
                invalid_examples += 1
                continue

            if not example.thought.strip():
                empty_thoughts += 1
                invalid_examples += 1
                continue

            signature = (
                example.input.strip(),
                example.thought.strip(),
                example.output.strip(),
            )
            if signature in seen_signatures:
                duplicate_examples += 1
            else:
                seen_signatures.add(signature)

            if example.status == "confirmed":
                confirmed_examples += 1
            if example.status == "conflicted":
                conflict_examples += 1
            if example.confidence < self.min_confidence:
                low_confidence_examples += 1
                invalid_examples += 1
                continue

            valid_examples += 1

        total_examples = len(dataset)

        if confirmed_examples < total_examples:
            warnings.append("dataset contains non-confirmed examples")
        if low_confidence_examples > 0:
            warnings.append("dataset contains low-confidence examples")
        if conflict_examples > 0:
            warnings.append("dataset contains conflicted examples")
        if duplicate_examples > 0:
            warnings.append("dataset contains duplicate examples")
        if empty_thoughts > 0:
            warnings.append("dataset contains empty thoughts")

        return DataQualityReport(
            total_examples=total_examples,
            valid_examples=valid_examples,
            invalid_examples=invalid_examples,
            empty_thoughts=empty_thoughts,
            confirmed_examples=confirmed_examples,
            low_confidence_examples=low_confidence_examples,
            duplicate_examples=duplicate_examples,
            conflict_examples=conflict_examples,
            warnings=warnings,
        )

    def filter_dataset(
        self, dataset: Sequence[EvaluatedExampleProtocol]
    ) -> list[EvaluatedExampleProtocol]:
        return [
            example
            for example in dataset
            if self._is_shape_valid(example)
            and example.thought.strip()
            and example.status == "confirmed"
            and example.confidence >= self.min_confidence
        ]

    def summarize_issues(
        self, dataset: Sequence[EvaluatedExampleProtocol]
    ) -> dict[str, int]:
        report = self.evaluate(dataset)
        return {
            "total_examples": report.total_examples,
            "invalid_examples": report.invalid_examples,
            "empty_thoughts": report.empty_thoughts,
            "low_confidence_examples": report.low_confidence_examples,
            "duplicate_examples": report.duplicate_examples,
            "conflict_examples": report.conflict_examples,
        }

    def _is_shape_valid(self, example: EvaluatedExampleProtocol) -> bool:
        return (
            isinstance(example.input, str)
            and isinstance(example.thought, str)
            and isinstance(example.output, str)
            and isinstance(example.status, str)
            and isinstance(example.source_ids, list)
            and all(isinstance(source_id, str) for source_id in example.source_ids)
            and self._is_number(example.confidence)
        )

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
