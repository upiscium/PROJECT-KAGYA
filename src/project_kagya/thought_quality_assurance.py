from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(slots=True)
class ThoughtExample:
    input: str
    thought: str
    output: str
    source_ids: list[str]
    confidence: float
    status: str


@dataclass(slots=True)
class ThoughtValidationReport:
    valid: bool
    score: float
    reasons: list[str]


class ThoughtExampleProtocol(Protocol):
    input: str
    thought: str
    output: str
    source_ids: list[str]
    confidence: float
    status: str


class ThoughtQualityAssurer:
    def __init__(self, min_score: float = 0.6, max_thought_chars: int = 400) -> None:
        self.min_score = min_score
        self.max_thought_chars = max_thought_chars

    def validate_example(
        self, example: ThoughtExampleProtocol
    ) -> ThoughtValidationReport:
        reasons: list[str] = []

        if not self._is_example_shape_valid(example):
            return ThoughtValidationReport(False, 0.0, ["invalid example shape"])

        thought = example.thought.strip()
        if not thought:
            reasons.append("thought is empty")

        if len(thought) > self.max_thought_chars:
            reasons.append("thought is too long")

        score = self.score_thought(thought)

        if score < self.min_score:
            reasons.append("thought score below threshold")

        if not self._has_alignment(example.input, thought, example.output):
            reasons.append("thought is not aligned with input/output")

        valid = not reasons
        return ThoughtValidationReport(valid=valid, score=score, reasons=reasons)

    def filter_examples(
        self, examples: Sequence[ThoughtExampleProtocol]
    ) -> list[ThoughtExampleProtocol]:
        return [example for example in examples if self.validate_example(example).valid]

    def score_thought(self, thought: str) -> float:
        normalized = thought.strip()
        if not normalized:
            return 0.0

        tokens = self._tokenize(normalized)
        if not tokens:
            return 0.0

        length_score = self._length_score(len(normalized))
        repetition_score = len(set(tokens)) / len(tokens)
        newline_penalty = 1.0 - min(normalized.count("\n") * 0.1, 0.3)

        score = (
            (length_score * 0.45) + (repetition_score * 0.4) + (newline_penalty * 0.15)
        )
        return max(0.0, min(1.0, score))

    def _is_example_shape_valid(self, example: ThoughtExampleProtocol) -> bool:
        return (
            all(
                isinstance(value, str)
                for value in (
                    example.input,
                    example.thought,
                    example.output,
                    example.status,
                )
            )
            and isinstance(example.source_ids, list)
            and all(isinstance(source_id, str) for source_id in example.source_ids)
            and self._is_number(example.confidence)
        )

    def _has_alignment(self, input_text: str, thought: str, output_text: str) -> bool:
        thought_tokens = set(self._tokenize(thought))
        input_tokens = set(self._tokenize(input_text))
        output_tokens = set(self._tokenize(output_text))

        if not thought_tokens:
            return False
        if not input_tokens and not output_tokens:
            return True
        return bool(thought_tokens & (input_tokens | output_tokens))

    def _length_score(self, length: int) -> float:
        if length <= 160:
            return 1.0
        if length <= 320:
            return 0.8
        if length <= 480:
            return 0.5
        return 0.2

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in text.lower().split() if token]

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
