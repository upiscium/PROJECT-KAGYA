"""QLoRA training entry points."""

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterRegistry
from kagya.learning.dream_dataset_generator import (
    DreamDatasetRecord,
    format_training_text,
)


class QloraTrainingError(RuntimeError):
    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category


@dataclass(frozen=True)
class QloraTrainingResult:
    adapter_id: str
    adapter_path: Path
    dataset_path: Path
    dataset_hash: str
    dry_run: bool
    training_records: int
    artifact_path: Path | None = None
    parent_adapter_id: str | None = None
    parent_adapter_hash: str | None = None
    adapter_hash: str | None = None


class QloraTrainer:
    """Validate dream datasets and optionally run QLoRA training."""

    def __init__(
        self, settings: Settings, registry: AdapterRegistry | None = None
    ) -> None:
        self.settings = settings
        self.registry = registry

    def train_bundle(self, bundle_path: Path) -> QloraTrainingResult:
        from kagya.training.artifacts import TrainingArtifactContract

        manifest = TrainingArtifactContract().validate_bundle(
            bundle_path,
            expected_model_id=self.settings.model.primary_id,
            expected_model_revision=self.settings.model.revision,
            expected_processor_revision=self.settings.model.processor_revision,
        )
        if manifest.chat_template_version != "gemma-v1":
            raise QloraTrainingError(
                "chat_template_mismatch", manifest.chat_template_version
            )
        if manifest.dataset_format_version != "dream-v2":
            raise QloraTrainingError(
                "dataset_version_mismatch", manifest.dataset_format_version
            )
        if manifest.parent_adapter_id is not None:
            if self.registry is not None:
                self.registry.validate_continuation(
                    adapter_id=f"training:{manifest.job_id}",
                    base_model=manifest.base_model_id,
                    base_model_revision=manifest.base_model_revision,
                    parent_adapter_id=manifest.parent_adapter_id,
                    parent_adapter_hash=manifest.parent_adapter_hash,
                )
            if not self.settings.qlora.dry_run:
                raise QloraTrainingError(
                    "hardware_validation_required",
                    "continued adapter loading remains gated by hardware validation issue #98",
                )
        result = self.train(bundle_path / manifest.dataset_path)
        return QloraTrainingResult(
            **{
                **result.__dict__,
                "parent_adapter_id": manifest.parent_adapter_id,
                "parent_adapter_hash": manifest.parent_adapter_hash,
            }
        )

    def train(self, dataset_path: Path) -> QloraTrainingResult:
        records = self._load_dataset(dataset_path)
        dataset_hash = _hash_file(dataset_path)
        adapter_id = f"adapter-{uuid4()}"
        adapter_path = self.settings.qlora.output_dir / adapter_id
        staging_path = self.settings.qlora.output_dir / f".{adapter_id}.tmp"
        self.settings.qlora.output_dir.mkdir(parents=True, exist_ok=True)
        staging_path.mkdir()
        try:
            if self.settings.qlora.dry_run:
                self._write_dry_run_manifest(
                    staging_path, dataset_path, dataset_hash, records
                )
            else:
                self._run_training(staging_path, dataset_path, dataset_hash, records)
            if not any(staging_path.iterdir()):
                raise QloraTrainingError("empty_adapter", "trainer produced no files")
            os.rename(staging_path, adapter_path)
        except Exception:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise
        return QloraTrainingResult(
            adapter_id=adapter_id,
            adapter_path=adapter_path,
            dataset_path=dataset_path,
            dataset_hash=dataset_hash,
            dry_run=self.settings.qlora.dry_run,
            training_records=len(records),
            adapter_hash=_hash_directory(adapter_path),
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
                    raise ValueError(
                        f"Invalid dream dataset record at line {line_number}"
                    ) from exc
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
        with (adapter_path / "dry_run_manifest.json").open(
            "w", encoding="utf-8"
        ) as manifest_file:
            json.dump(manifest, manifest_file, indent=2)

    def _run_training(
        self,
        adapter_path: Path,
        dataset_path: Path,
        dataset_hash: str,
        records: list[DreamDatasetRecord],
    ) -> None:
        deps = self._load_training_dependencies()
        processor = deps["AutoProcessor"].from_pretrained(
            self.settings.model.primary_id,
            revision=self.settings.model.processor_revision,
        )
        if not getattr(processor, "chat_template", None):
            raise QloraTrainingError(
                "chat_template_mismatch", "processor has no configured chat template"
            )
        if getattr(processor, "tokenizer", None) is None:
            raise QloraTrainingError("tokenizer_mismatch", "processor has no tokenizer")
        model = deps["AutoModelForImageTextToText"].from_pretrained(
            self.settings.model.primary_id,
            revision=self.settings.model.revision,
            **self._model_load_kwargs(deps["BitsAndBytesConfig"]),
        )
        self._validate_target_modules(model)
        model = deps["prepare_model_for_kbit_training"](
            model,
            use_gradient_checkpointing=self.settings.qlora.gradient_checkpointing,
        )
        dataset = deps["Dataset"].from_list(
            [{"text": format_training_text(record)} for record in records]
        )
        peft_config = deps["LoraConfig"](
            r=self.settings.qlora.r,
            lora_alpha=self.settings.qlora.lora_alpha,
            lora_dropout=self.settings.qlora.lora_dropout,
            target_modules=self.settings.qlora.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        training_args = deps["SFTConfig"](
            output_dir=str(adapter_path),
            learning_rate=self.settings.qlora.learning_rate,
            num_train_epochs=self.settings.qlora.num_train_epochs,
            max_steps=self.settings.qlora.max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=self.settings.qlora.gradient_accumulation_steps,
            gradient_checkpointing=self.settings.qlora.gradient_checkpointing,
            max_length=self.settings.qlora.max_sequence_length,
            optim=self.settings.qlora.optimizer,
            bf16=True,
            seed=self.settings.qlora.seed,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
        )
        trainer = self._build_trainer(
            deps, model, processor, dataset, peft_config, training_args
        )
        try:
            output = trainer.train(resume_from_checkpoint=False)
        except Exception as exc:
            if (
                exc.__class__.__name__ == "OutOfMemoryError"
                or "out of memory" in str(exc).lower()
            ):
                raise QloraTrainingError("cuda_oom", str(exc)) from exc
            raise QloraTrainingError("training_failed", str(exc)) from exc
        metrics = getattr(output, "metrics", {}) if output is not None else {}
        if any(
            isinstance(value, float) and not math.isfinite(value)
            for value in metrics.values()
        ):
            raise QloraTrainingError(
                "non_finite_metrics", "training returned NaN or infinity"
            )
        trainer.save_model(str(adapter_path))
        self._write_training_manifest(
            adapter_path, dataset_path, dataset_hash, records, metrics=metrics
        )

    def _validate_target_modules(self, model: Any) -> None:
        named_modules = getattr(model, "named_modules", None)
        if not callable(named_modules):
            raise QloraTrainingError(
                "target_module_mismatch", "model exposes no named modules"
            )
        available = {name.rsplit(".", 1)[-1] for name, _module in named_modules()}
        missing = set(self.settings.qlora.target_modules) - available
        if missing:
            raise QloraTrainingError(
                "target_module_mismatch",
                "missing modules: " + ", ".join(sorted(missing)),
            )

    def _load_training_dependencies(self) -> dict[str, Any]:
        try:
            from datasets import Dataset  # type: ignore[import-untyped]
            from peft import LoraConfig, prepare_model_for_kbit_training
            from transformers import (
                AutoModelForImageTextToText,
                AutoProcessor,
                BitsAndBytesConfig,
            )
            from trl import SFTConfig, SFTTrainer
        except ImportError as exc:
            raise RuntimeError(
                "Non-dry-run QLoRA training dependencies are not installed"
            ) from exc
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
        *,
        metrics: dict[str, Any],
    ) -> None:
        manifest = self._training_manifest(False, dataset_path, dataset_hash, records)
        manifest["metrics"] = metrics
        manifest["environment"] = self._environment_metadata()
        with (adapter_path / "training_manifest.json").open(
            "w", encoding="utf-8"
        ) as manifest_file:
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
            "base_model_revision": self.settings.model.revision,
            "processor_revision": self.settings.model.processor_revision,
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
                "gradient_checkpointing": self.settings.qlora.gradient_checkpointing,
                "gradient_accumulation_steps": self.settings.qlora.gradient_accumulation_steps,
                "max_sequence_length": self.settings.qlora.max_sequence_length,
                "optimizer": self.settings.qlora.optimizer,
                "seed": self.settings.qlora.seed,
                "target_modules": self.settings.qlora.target_modules,
                "resume_policy": self.settings.qlora.resume_policy,
                "quantization": "nf4",
                "compute_dtype": "bfloat16",
            },
        }

    @staticmethod
    def _environment_metadata() -> dict[str, Any]:
        versions: dict[str, str | None] = {}
        for package in ("torch", "transformers", "peft", "trl", "bitsandbytes"):
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        try:
            import torch

            cuda = torch.version.cuda
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except ImportError:
            cuda = None
            gpu = None
        return {"packages": versions, "cuda": cuda, "gpu": gpu}


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as target_file:
        for chunk in iter(lambda: target_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _hash_directory(path: Path) -> str:
    hasher = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            hasher.update(item.relative_to(path).as_posix().encode())
            hasher.update(item.read_bytes())
    return hasher.hexdigest()
