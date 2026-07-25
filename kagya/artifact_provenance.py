"""Canonical model and adapter artifact provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


IMMUTABLE_REVISION = re.compile(r"[0-9a-f]{40}")


class ProvenanceFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    filename: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ModelArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    model_id: str
    requested_revision: str
    resolved_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    processor_requested_revision: str
    processor_resolved_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    metadata_files: tuple[ProvenanceFile, ...]
    processor_files: tuple[ProvenanceFile, ...]
    weight_files: tuple[ProvenanceFile, ...]
    quantization_files: tuple[ProvenanceFile, ...]

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


class AdapterArtifactManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal[1] = 1
    adapter_config: ProvenanceFile
    weight_files: tuple[ProvenanceFile, ...]
    peft_type: str
    base_model_name: str
    base_model_revision: str
    target_modules: tuple[str, ...]
    rank: int | None = None
    alpha: float | None = None
    quantization_compatible: bool

    @property
    def sha256(self) -> str:
        return _canonical_hash(self.model_dump(mode="json"))


_MODEL_METADATA = {
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model_index.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "tokenizer.model",
    "chat_template.jinja",
}

_WEIGHT_EXTENSIONS = (".safetensors", ".bin", ".pt", ".pth")


def require_immutable_revision(value: str, label: str) -> str:
    if not IMMUTABLE_REVISION.fullmatch(value):
        raise ValueError(f"{label} must be an exact immutable 40-hex commit")
    return value


def build_model_artifact_manifest(
    snapshot: Path,
    *,
    processor_snapshot: Path | None = None,
    model_id: str,
    requested_revision: str,
    resolved_revision: str,
    processor_requested_revision: str,
    processor_resolved_revision: str,
) -> ModelArtifactManifest:
    require_immutable_revision(resolved_revision, "resolved model revision")
    require_immutable_revision(
        processor_resolved_revision, "resolved processor revision"
    )
    metadata: list[ProvenanceFile] = []
    processor_files: list[ProvenanceFile] = []
    weights: list[ProvenanceFile] = []
    quantization: list[ProvenanceFile] = []
    for path in _regular_files(snapshot):
        relative = path.relative_to(snapshot).as_posix()
        record = _file_record(path, relative)
        lowered = path.name.casefold()
        if "quant" in lowered or "gptq" in lowered or "awq" in lowered:
            quantization.append(record)
        elif lowered.endswith(_WEIGHT_EXTENSIONS):
            weights.append(record)
        else:
            metadata.append(record)
    processor_root = (processor_snapshot or snapshot).resolve()
    for path in _regular_files(processor_root):
        lowered = path.name.casefold()
        if not lowered.endswith(_WEIGHT_EXTENSIONS):
            processor_files.append(
                _file_record(path, path.relative_to(processor_root).as_posix())
            )
    if not weights:
        raise ValueError("Model snapshot contains no weight files")
    return ModelArtifactManifest(
        model_id=model_id,
        requested_revision=requested_revision,
        resolved_revision=resolved_revision,
        processor_requested_revision=processor_requested_revision,
        processor_resolved_revision=processor_resolved_revision,
        metadata_files=tuple(metadata),
        processor_files=tuple(processor_files),
        weight_files=tuple(weights),
        quantization_files=tuple(quantization),
    )


def build_adapter_artifact_manifest(
    path: Path,
    *,
    base_model_name: str | None = None,
    base_model_revision: str | None = None,
) -> AdapterArtifactManifest:
    config_path = path / "adapter_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Adapter config is missing or invalid") from exc
    if not isinstance(config, dict):
        raise ValueError("Adapter config is invalid")
    weights = tuple(
        _file_record(item, item.relative_to(path).as_posix())
        for item in sorted(path.rglob("*"))
        if item.is_file()
        and not item.is_symlink()
        and item.name != "adapter_config.json"
        and item.name.casefold().endswith((".safetensors", ".bin", ".pt", ".pth"))
    )
    target = config.get("target_modules", ())
    modules = (
        tuple(sorted(str(item) for item in target))
        if isinstance(target, (list, tuple, set))
        else ()
    )
    return AdapterArtifactManifest(
        adapter_config=_file_record(config_path, "adapter_config.json"),
        weight_files=weights,
        peft_type=str(config.get("peft_type", config.get("task_type", "unknown"))),
        base_model_name=str(
            config.get("base_model_name_or_path", base_model_name or "unknown")
        ),
        base_model_revision=str(
            config.get(
                "revision",
                config.get("base_model_revision", base_model_revision or "unknown"),
            )
        ),
        target_modules=modules,
        rank=_optional_int(config.get("r")),
        alpha=_optional_float(config.get("lora_alpha")),
        quantization_compatible=bool(config.get("quantization_compatible", True)),
    )


def verify_attached_adapter_config(
    attached: Any, expected: AdapterArtifactManifest
) -> None:
    configs = getattr(attached, "peft_config", None)
    if not isinstance(configs, dict) or not configs:
        raise RuntimeError("Loaded adapter has no attached PEFT config")
    actual = next(iter(configs.values()))
    actual_target = tuple(
        sorted(str(item) for item in (getattr(actual, "target_modules", ()) or ()))
    )
    comparisons = (
        (
            str(getattr(actual, "peft_type", "")).split(".")[-1].casefold(),
            expected.peft_type.casefold(),
        ),
        (
            str(getattr(actual, "base_model_name_or_path", "unknown")),
            expected.base_model_name,
        ),
        (str(getattr(actual, "revision", "unknown")), expected.base_model_revision),
        (actual_target, expected.target_modules),
        (_optional_int(getattr(actual, "r", None)), expected.rank),
        (_optional_float(getattr(actual, "lora_alpha", None)), expected.alpha),
    )
    if any(
        actual_value != expected_value for actual_value, expected_value in comparisons
    ):
        raise RuntimeError("Attached PEFT config does not match adapter manifest")


def _regular_files(root: Path) -> list[Path]:
    return sorted(item for item in root.rglob("*") if item.is_file())


def _file_record(path: Path, relative: str) -> ProvenanceFile:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Artifact manifest path is unsafe")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return ProvenanceFile(filename=relative, size=size, sha256=digest.hexdigest())


def _canonical_hash(value: dict[str, Any]) -> str:
    content = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
