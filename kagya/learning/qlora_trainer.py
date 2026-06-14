"""QLoRA training entry points."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from kagya.config import Settings
from kagya.learning.dream_dataset_generator import DreamDatasetRecord, format_training_text


@dataclass(frozen=True)
class QloraTrainingResult:
    adapter_id: str
    adapter_path: Path
    dataset_path: Path
    dataset_hash: str
    dry_run: bool
    training_records: int


class QloraTrainer:
    """Validate dream datasets and optionally run QLoRA training."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def train(self, dataset_path: Path) -> QloraTrainingResult:
        records = self._load_dataset(dataset_path)
        dataset_hash = _hash_file(dataset_path)
        adapter_id = f"adapter-{uuid4()}"
        adapter_path = self.settings.qlora.output_dir / adapter_id
        adapter_path.mkdir(parents=True, exist_ok=True)
        if self.settings.qlora.dry_run:
            self._write_dry_run_manifest(adapter_path, dataset_path, dataset_hash, records)
            return QloraTrainingResult(
                adapter_id=adapter_id,
                adapter_path=adapter_path,
                dataset_path=dataset_path,
                dataset_hash=dataset_hash,
                dry_run=True,
                training_records=len(records),
            )
        self._run_training(adapter_path, dataset_path, dataset_hash, records)
        return QloraTrainingResult(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            dry_run=False,
            training_records=len(records),
        )

    def _load_dataset(self, dataset_path: Path) -> list[DreamDatasetRecord]:
        if not dataset_path.exists():
            raise ValueError(f"Dream dataset does not exist: {dataset_path}")
        records: list[DreamDatasetRecord] = []
        with dataset_path.open("r", encoding="utf-8") as dataset_file:
            for line_number, line in enumerate(dataset_file, start=1):
                if not line.strip():
                    continue
                data = json.loads(line)
                try:
                    record = DreamDatasetRecord(
                        input=str(data["input"]),
                        thought=str(data["thought"]),
                        output=str(data["output"]),
                    )
                except KeyError as exc:
                    raise ValueError(f"Invalid dream dataset record at line {line_number}") from exc
                format_training_text(record)
                records.append(record)
        if not records:
            raise ValueError("Dream dataset is empty")
        return records

    def _write_dry_run_manifest(
        self,
        adapter_path: Path,
        dataset_path: Path,
        dataset_hash: str,
        records: list[DreamDatasetRecord],
    ) -> None:
        manifest = self._training_manifest(True, dataset_path, dataset_hash, records)
        with (adapter_path / "dry_run_manifest.json").open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

    def _run_training(
        self,
        adapter_path: Path,
        dataset_path: Path,
        dataset_hash: str,
        records: list[DreamDatasetRecord],
    ) -> None:
        deps = self._load_training_dependencies()
        processor = deps["AutoProcessor"].from_pretrained(self.settings.model.primary_id)
        model = deps["AutoModelForImageTextToText"].from_pretrained(
            self.settings.model.primary_id,
            **self._model_load_kwargs(deps["BitsAndBytesConfig"]),
        )
        model = deps["prepare_model_for_kbit_training"](model)
        dataset = deps["Dataset"].from_list(
            [{"text": format_training_text(record)} for record in records]
        )
        peft_config = deps["LoraConfig"](
            r=self.settings.qlora.r,
            lora_alpha=self.settings.qlora.lora_alpha,
            lora_dropout=self.settings.qlora.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        training_args = deps["SFTConfig"](
            output_dir=str(adapter_path),
            learning_rate=self.settings.qlora.learning_rate,
            num_train_epochs=self.settings.qlora.num_train_epochs,
            max_steps=self.settings.qlora.max_steps,
            per_device_train_batch_size=1,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )
        trainer = self._build_trainer(deps, model, processor, dataset, peft_config, training_args)
        trainer.train()
        trainer.save_model(str(adapter_path))
        self._write_training_manifest(adapter_path, dataset_path, dataset_hash, records)

    def _load_training_dependencies(self) -> dict[str, Any]:
        try:
            from datasets import Dataset
            from peft import LoraConfig, prepare_model_for_kbit_training
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:
            raise RuntimeError("Non-dry-run QLoRA training dependencies are not installed") from exc
        return {
            "AutoModelForImageTextToText": AutoModelForImageTextToText,
            "AutoProcessor": AutoProcessor,
            "BitsAndBytesConfig": BitsAndBytesConfig,
            "Dataset": Dataset,
            "LoraConfig": LoraConfig,
            "SFTConfig": SFTConfig,
            "SFTTrainer": SFTTrainer,
            "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        }

    def _model_load_kwargs(self, bits_and_bytes_config: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.settings.model.device == "auto":
            kwargs["device_map"] = "auto"
        if self.settings.model.dtype != "auto":
            import torch

            kwargs["torch_dtype"] = getattr(torch, self.settings.model.dtype)
        if self.settings.model.load_in_4bit:
            import torch

            kwargs["quantization_config"] = bits_and_bytes_config(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        return kwargs

    def _build_trainer(
        self,
        deps: dict[str, Any],
        model: Any,
        processor: Any,
        dataset: Any,
        peft_config: Any,
        training_args: Any,
    ) -> Any:
        trainer_cls = deps["SFTTrainer"]
        try:
            return trainer_cls(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=peft_config,
                processing_class=processor,
            )
        except TypeError:
            return trainer_cls(
                model=model,
                args=training_args,
                train_dataset=dataset,
                peft_config=peft_config,
                tokenizer=processor,
            )

    def _write_training_manifest(
        self,
        adapter_path: Path,
        dataset_path: Path,
        dataset_hash: str,
        records: list[DreamDatasetRecord],
    ) -> None:
        manifest = self._training_manifest(False, dataset_path, dataset_hash, records)
        with (adapter_path / "training_manifest.json").open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

    def _training_manifest(
        self,
        dry_run: bool,
        dataset_path: Path,
        dataset_hash: str,
        records: list[DreamDatasetRecord],
    ) -> dict[str, Any]:
        return {
            "dry_run": dry_run,
            "base_model": self.settings.model.primary_id,
            "dataset_path": str(dataset_path),
            "dataset_hash": dataset_hash,
            "training_records": len(records),
            "qlora": {
                "r": self.settings.qlora.r,
                "alpha": self.settings.qlora.alpha,
                "lora_alpha": self.settings.qlora.lora_alpha,
                "dropout": self.settings.qlora.dropout,
                "lora_dropout": self.settings.qlora.lora_dropout,
                "learning_rate": self.settings.qlora.learning_rate,
                "num_train_epochs": self.settings.qlora.num_train_epochs,
                "max_steps": self.settings.qlora.max_steps,
                "quantization": "nf4",
                "compute_dtype": "bfloat16",
            },
        }


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as target_file:
        for chunk in iter(lambda: target_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
