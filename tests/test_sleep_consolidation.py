from __future__ import annotations

import json
from dataclasses import dataclass

from project_kagya.sleep_consolidation import DreamSample, SleepCycleManager


class FakeCollection:
    def __init__(self) -> None:
        self.records = {
            "ids": ["e1", "e2", "e3"],
            "documents": ["low", "high arousal", "high valence"],
            "metadatas": [
                {"valence": 0.1, "arousal": 0.2},
                {"valence": 0.2, "arousal": 0.8},
                {"valence": 0.7, "arousal": 0.3},
            ],
        }

    def get(self) -> dict[str, list[object]]:
        return self.records

    def delete(self, *, ids: list[str]) -> None:
        del ids


@dataclass
class FakeMemorySystem:
    hippocampus: FakeCollection


def test_triage_high_emotion_episodes() -> None:
    manager = SleepCycleManager(memory_system=FakeMemorySystem(FakeCollection()))

    selected = manager.triage_high_emotion_episodes()

    assert [item["id"] for item in selected] == ["e2", "e3"]


def test_generate_dream_dataset_writes_jsonl(tmp_path) -> None:
    manager = SleepCycleManager(memory_system=FakeMemorySystem(FakeCollection()))

    class Pipeline:
        def __call__(self, prompt: str) -> dict[str, str]:
            assert "Generate an ideal thought process" in prompt
            return {"thought": "plan", "output": "reply"}

    output = manager.generate_dream_dataset(
        Pipeline(),
        episodes=[{"document": "high arousal", "metadata": {"arousal": 0.8}}],
        output_path=tmp_path / "dream_dataset.jsonl",
    )

    assert output.exists()
    record = json.loads(output.read_text(encoding="utf-8").strip())
    assert record == {"input": "high arousal", "thought": "plan", "output": "reply"}


def test_build_sft_text_matches_spec() -> None:
    manager = SleepCycleManager(memory_system=FakeMemorySystem(FakeCollection()))
    sample = DreamSample(input_text="hello", thought="plan", output="reply")

    assert manager.build_sft_text(sample) == (
        "ユーザー: hello\n私: <think>\nplan\n</think>\nreply<eos>"
    )


def test_train_qlora_saves_adapter(tmp_path) -> None:
    manager = SleepCycleManager(memory_system=FakeMemorySystem(FakeCollection()))
    dataset = tmp_path / "dream_dataset.jsonl"
    dataset.write_text(
        json.dumps({"input": "hello", "thought": "plan", "output": "reply"}) + "\n",
        encoding="utf-8",
    )

    class Model:
        def __init__(self) -> None:
            self.saved_path: str | None = None

        def save_pretrained(self, path: str) -> None:
            self.saved_path = path

    class Trainer:
        def __init__(self, model: Model, train_dataset: list[str]) -> None:
            self.model = model
            self.train_dataset = train_dataset

        def train(self) -> str:
            assert self.train_dataset == [
                "ユーザー: hello\n私: <think>\nplan\n</think>\nreply<eos>"
            ]
            return "trained"

    result = manager.train_qlora(
        model=Model(),
        dataset_path=dataset,
        output_dir=tmp_path / "adapter",
        trainer_factory=lambda model, train_dataset: Trainer(model, train_dataset),
    )

    assert result == "trained"
