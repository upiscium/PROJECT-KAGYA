from __future__ import annotations

from pathlib import Path

from project_kagya.qlora_training import QLoRATrainer, QLoRATrainingSummary


class DummyModel:
    def __init__(self) -> None:
        self.trained_on: list[str] = []
        self.saved_to: str | None = None

    def train_on_texts(self, prompts: list[str]) -> None:
        self.trained_on = list(prompts)

    def save_pretrained(self, path: str) -> None:
        self.saved_to = path


class DummyExample:
    def __init__(
        self, input: str, thought: str, output: str, confidence: float = 0.9
    ) -> None:
        self.input = input
        self.thought = thought
        self.output = output
        self.source_ids = ["1"]
        self.confidence = confidence
        self.status = "confirmed"


def test_train_formats_prompts_and_saves_adapter(tmp_path: Path) -> None:
    model = DummyModel()
    trainer = QLoRATrainer()

    summary = trainer.train(
        [DummyExample("hello", "think hello", "world")],
        model,
        tmp_path / "out",
    )

    assert isinstance(summary, QLoRATrainingSummary)
    assert summary.total_examples == 1
    assert summary.trained_examples == 1
    assert summary.adapter_path == str(tmp_path / "out")
    assert model.trained_on
    assert "<think>" in model.trained_on[0]


def test_train_returns_empty_summary_for_no_valid_examples(tmp_path: Path) -> None:
    model = DummyModel()
    trainer = QLoRATrainer()

    summary = trainer.train(
        [DummyExample("hello", "", "world")],
        model,
        tmp_path / "out",
    )

    assert summary.total_examples == 1
    assert summary.trained_examples == 0
    assert summary.adapter_path is None


def test_load_adapter_delegates_to_backend(tmp_path: Path) -> None:
    model = DummyModel()
    trainer = QLoRATrainer()

    restored = trainer.load_adapter(model, tmp_path / "adapter")

    assert restored is model
