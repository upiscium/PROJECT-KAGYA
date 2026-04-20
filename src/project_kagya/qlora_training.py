from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(slots=True)
class QLoRATrainingSummary:
    total_examples: int
    trained_examples: int
    adapter_path: str | None


class TrainingExampleProtocol(Protocol):
    input: str
    thought: str
    output: str
    source_ids: list[str]
    confidence: float
    status: str


class QLoRABackendProtocol(Protocol):
    def train(
        self,
        prompts: Sequence[str],
        model: Any,
        output_dir: Path,
        tokenizer: Any | None,
    ) -> Any: ...

    def save_adapter(self, artifact: Any, output_dir: Path) -> Path: ...

    def load_adapter(self, base_model: Any, adapter_dir: Path) -> Any: ...


class _FallbackQLoRABackend:
    def train(
        self,
        prompts: Sequence[str],
        model: Any,
        output_dir: Path,
        tokenizer: Any | None,
    ) -> Any:
        dataset = list(prompts)
        if hasattr(model, "train_on_texts"):
            model.train_on_texts(dataset)
            return model
        return {
            "prompts": dataset,
            "output_dir": str(output_dir),
            "tokenizer": tokenizer,
        }

    def save_adapter(self, artifact: Any, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(artifact, "save_pretrained"):
            artifact.save_pretrained(str(output_dir))
        elif hasattr(artifact, "model") and hasattr(artifact.model, "save_pretrained"):
            artifact.model.save_pretrained(str(output_dir))
        return output_dir

    def load_adapter(self, base_model: Any, adapter_dir: Path) -> Any:
        if hasattr(base_model, "load_adapter"):
            return base_model.load_adapter(str(adapter_dir))
        return base_model


class QLoRATrainer:
    def __init__(
        self,
        tokenizer: Any | None = None,
        backend: QLoRABackendProtocol | None = None,
        confidence_threshold: float = 0.5,
    ) -> None:
        self.tokenizer = tokenizer
        self.backend = backend if backend is not None else _FallbackQLoRABackend()
        self.confidence_threshold = confidence_threshold
        self._trained_artifact: Any | None = None

    def prepare_model_for_training(self, model: Any) -> Any:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError:
            return model
        if not hasattr(model, "named_parameters"):
            return model
        return prepare_model_for_kbit_training(model)

    def train(
        self,
        examples: Sequence[TrainingExampleProtocol],
        model: Any,
        output_dir: str | Path,
    ) -> QLoRATrainingSummary:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        trainable_examples = [
            example for example in examples if self._is_trainable(example)
        ]
        if not trainable_examples:
            self._trained_artifact = None
            return QLoRATrainingSummary(
                total_examples=len(examples),
                trained_examples=0,
                adapter_path=None,
            )

        prepared_model = self.prepare_model_for_training(model)
        prompts = [self._format_prompt(example) for example in trainable_examples]
        self._trained_artifact = self.backend.train(
            prompts,
            prepared_model,
            output_path,
            self.tokenizer,
        )
        adapter_path = self.backend.save_adapter(self._trained_artifact, output_path)

        return QLoRATrainingSummary(
            total_examples=len(examples),
            trained_examples=len(trainable_examples),
            adapter_path=str(adapter_path),
        )

    def save_adapter(self, output_dir: str | Path) -> Path:
        if self._trained_artifact is None:
            raise RuntimeError("no trained artifact available")
        return self.backend.save_adapter(self._trained_artifact, Path(output_dir))

    def load_adapter(self, base_model: Any, adapter_dir: str | Path) -> Any:
        return self.backend.load_adapter(base_model, Path(adapter_dir))

    def _is_trainable(self, example: TrainingExampleProtocol) -> bool:
        return (
            isinstance(example.input, str)
            and isinstance(example.thought, str)
            and isinstance(example.output, str)
            and isinstance(example.status, str)
            and isinstance(example.source_ids, list)
            and all(isinstance(source_id, str) for source_id in example.source_ids)
            and self._is_number(example.confidence)
            and example.status == "confirmed"
            and bool(example.thought.strip())
            and example.confidence >= self.confidence_threshold
            and bool(example.input.strip())
            and bool(example.output.strip())
        )

    @staticmethod
    def _format_prompt(example: TrainingExampleProtocol) -> str:
        return (
            f"ユーザー: {example.input}\n"
            f"私: <think>\n"
            f"{example.thought}\n"
            f"</think>\n"
            f"{example.output}<eos>"
        )

    @staticmethod
    def _is_number(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
