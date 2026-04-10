from __future__ import annotations

import json
from pathlib import Path

from project_kagya.sleep_consolidation_training import (
    SleepConsolidationTrainingPipeline,
    TrainingExample,
)


class DummyTrainer:
    def __init__(self) -> None:
        self.trained_examples: list[TrainingExample] = []
        self.saved_to: Path | None = None
        self.loaded_from: Path | None = None

    def train(
        self, examples: list[TrainingExample], model: object, output_dir: Path
    ) -> object:
        self.trained_examples = list(examples)
        return object()

    def save_adapter(self, output_dir: Path) -> Path:
        self.saved_to = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "adapter"

    def load_adapter(self, base_model: object, adapter_dir: Path) -> object:
        self.loaded_from = adapter_dir
        return base_model


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_train_reads_jsonl_filters_invalid_rows_and_saves_adapter(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(
        dataset,
        [
            {
                "input": "hello",
                "thought": "think",
                "output": "world",
                "source_ids": ["a"],
                "confidence": 0.9,
                "status": "confirmed",
            },
            {
                "input": "bad",
                "thought": "",
                "output": "skip",
                "source_ids": ["b"],
                "confidence": 0.9,
                "status": "confirmed",
            },
            {
                "input": "low",
                "thought": "think",
                "output": "skip",
                "source_ids": ["c"],
                "confidence": 0.1,
                "status": "confirmed",
            },
        ],
    )

    trainer = DummyTrainer()
    pipeline = SleepConsolidationTrainingPipeline(trainer)

    summary = pipeline.train(dataset, model=object(), output_dir=tmp_path / "out")

    assert summary.total_lines == 3
    assert summary.valid_examples == 1
    assert summary.trained_examples == 1
    assert summary.adapter_path == str(tmp_path / "out" / "adapter")
    assert len(trainer.trained_examples) == 1
    assert trainer.saved_to == tmp_path / "out"


def test_load_adapter_delegates_to_trainer(tmp_path: Path) -> None:
    trainer = DummyTrainer()
    pipeline = SleepConsolidationTrainingPipeline(trainer)

    model = object()
    restored = pipeline.load_adapter(model, tmp_path / "adapter")

    assert restored is model
    assert trainer.loaded_from == tmp_path / "adapter"


def test_train_handles_empty_dataset(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("", encoding="utf-8")

    trainer = DummyTrainer()
    pipeline = SleepConsolidationTrainingPipeline(trainer)

    summary = pipeline.train(dataset, model=object(), output_dir=tmp_path / "out")

    assert summary.total_lines == 0
    assert summary.valid_examples == 0
    assert summary.trained_examples == 0
    assert summary.adapter_path is None
    assert trainer.trained_examples == []
