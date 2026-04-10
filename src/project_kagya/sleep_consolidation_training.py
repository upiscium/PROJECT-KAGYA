from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(slots=True)
class TrainingExample:
    input: str
    thought: str
    output: str
    source_ids: list[str]
    confidence: float
    status: str


@dataclass(slots=True)
class TrainingSummary:
    total_lines: int
    valid_examples: int
    trained_examples: int
    adapter_path: str | None


class TrainerProtocol(Protocol):
    def train(
        self, examples: Sequence[TrainingExample], model: Any, output_dir: Path
    ) -> Any: ...

    def save_adapter(self, output_dir: Path) -> Path: ...

    def load_adapter(self, base_model: Any, adapter_dir: Path) -> Any: ...


class SleepConsolidationTrainingPipeline:
    def __init__(
        self,
        trainer: TrainerProtocol,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.trainer = trainer
        self.confidence_threshold = confidence_threshold

    def train(
        self, dataset_path: str | Path, model: Any, output_dir: str | Path
    ) -> TrainingSummary:
        dataset_file = Path(dataset_path)
        output_path = Path(output_dir)
        lines = self._read_lines(dataset_file)
        examples = [
            example
            for example in (self._parse_line(line) for line in lines)
            if example is not None
        ]
        valid_examples = [
            example for example in examples if self._is_trainable(example)
        ]

        if valid_examples:
            self.trainer.train(valid_examples, model, output_path)

        adapter_path: str | None = None
        if valid_examples:
            adapter_path = str(self.trainer.save_adapter(output_path))
        else:
            output_path.mkdir(parents=True, exist_ok=True)

        return TrainingSummary(
            total_lines=len(lines),
            valid_examples=len(valid_examples),
            trained_examples=len(valid_examples),
            adapter_path=adapter_path,
        )

    def save_adapter(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir)
        return self.trainer.save_adapter(output_path)

    def load_adapter(self, base_model: Any, adapter_dir: str | Path) -> Any:
        return self.trainer.load_adapter(base_model, Path(adapter_dir))

    def _read_lines(self, dataset_path: Path) -> list[str]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset not found: {dataset_path}")
        return [
            line
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def _parse_line(self, line: str) -> TrainingExample | None:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        input_text = payload.get("input")
        thought = payload.get("thought")
        output = payload.get("output")
        source_ids = payload.get("source_ids")
        confidence = payload.get("confidence")
        status = payload.get("status")

        if (
            not isinstance(input_text, str)
            or not isinstance(thought, str)
            or not isinstance(output, str)
        ):
            return None
        if not isinstance(source_ids, list) or not all(
            isinstance(source_id, str) for source_id in source_ids
        ):
            return None
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return None
        if not isinstance(status, str):
            return None

        return TrainingExample(
            input=input_text,
            thought=thought,
            output=output,
            source_ids=list(source_ids),
            confidence=float(confidence),
            status=status,
        )

    def _is_trainable(self, example: TrainingExample) -> bool:
        return (
            example.status == "confirmed"
            and bool(example.thought.strip())
            and example.confidence >= self.confidence_threshold
            and bool(example.input.strip())
            and bool(example.output.strip())
        )
