"""Production non-dry-run QLoRA readiness checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Callable

from kagya.config import Settings, get_settings, load_settings


DependencyChecker = Callable[[str], bool]
CudaChecker = Callable[[], bool]


@dataclass(frozen=True)
class QloraRequirementReport:
    ready: bool
    failures: list[str]
    warnings: list[str]


def check_qlora_production_readiness(
    settings: Settings,
    *,
    dependency_available: DependencyChecker | None = None,
    cuda_available: CudaChecker | None = None,
) -> QloraRequirementReport:
    """Validate production QLoRA boundaries without starting training."""

    dependency_available = dependency_available or _dependency_available
    cuda_available = cuda_available or _cuda_available
    failures: list[str] = []
    warnings: list[str] = []

    if settings.qlora.dry_run:
        failures.append("qlora.dry_run must be false for production training")
    if settings.model.provider.lower() != "transformers":
        failures.append("model.provider must be transformers")
    if not settings.model.load_in_4bit:
        failures.append("model.load_in_4bit must be true for the supported QLoRA path")
    if not settings.adapter_registry.manual_approval_required:
        failures.append("adapter_registry.manual_approval_required must remain true")
    if not settings.adapter_registry.eval_sets:
        failures.append("adapter_registry.eval_sets must include at least one eval set")
    for eval_set in settings.adapter_registry.eval_sets:
        if not eval_set.exists():
            failures.append(f"configured eval set does not exist: {eval_set}")
    missing_deps = [
        name
        for name in ("datasets", "peft", "torch", "transformers", "trl")
        if not dependency_available(name)
    ]
    if missing_deps:
        failures.append("missing training dependencies: " + ", ".join(missing_deps))
    if not cuda_available():
        failures.append("CUDA must be available for the production QLoRA path")
    if settings.qlora.output_dir == settings.memory.persist_directory:
        failures.append("qlora.output_dir must not point at the Chroma memory directory")

    warnings.append("training remains opt-in; dry-run mode is still the default config")
    warnings.append("trained adapters must pass evaluation history before manual approval")
    return QloraRequirementReport(ready=not failures, failures=failures, warnings=warnings)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check production non-dry-run QLoRA prerequisites without training."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    args = parser.parse_args(argv)

    settings = load_settings(args.config) if args.config is not None else get_settings()
    report = check_qlora_production_readiness(settings)
    for warning in report.warnings:
        print(f"[WARN] {warning}")
    if report.ready:
        print("[OK] Production QLoRA readiness check passed.")
        return 0
    for failure in report.failures:
        print(f"[FAIL] {failure}", file=sys.stderr)
    return 1


def _dependency_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


if __name__ == "__main__":
    raise SystemExit(main())
