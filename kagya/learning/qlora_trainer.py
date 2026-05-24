"""QLoRA training entry points."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
        raise NotImplementedError("Minimal non-dry-run QLoRA training is not implemented yet")

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
        manifest = {
            "dry_run": True,
            "base_model": self.settings.model.primary_id,
            "dataset_path": str(dataset_path),
            "dataset_hash": dataset_hash,
            "training_records": len(records),
            "qlora": {
                "r": self.settings.qlora.r,
                "lora_alpha": self.settings.qlora.lora_alpha,
                "lora_dropout": self.settings.qlora.lora_dropout,
                "learning_rate": self.settings.qlora.learning_rate,
                "num_train_epochs": self.settings.qlora.num_train_epochs,
                "quantization": "nf4",
                "compute_dtype": "bfloat16",
            },
        }
        with (adapter_path / "dry_run_manifest.json").open("w", encoding="utf-8") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)


def _hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as target_file:
        for chunk in iter(lambda: target_file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
