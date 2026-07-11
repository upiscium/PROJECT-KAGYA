"""Opt-in real Transformers provider smoke checks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Callable, Protocol

from kagya.config import Settings, get_settings, load_settings


class ProviderFactory(Protocol):
    def __call__(self, settings: Settings) -> Any:
        ...


@dataclass(frozen=True)
class SmokeStep:
    name: str
    ok: bool
    message: str


class TransformersSmokeError(RuntimeError):
    """Structured smoke failure with an operator-facing category."""

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(f"{category}: {detail}")
        self.category = category
        self.detail = detail


def run_transformers_smoke(
    settings: Settings,
    *,
    prompt: str = "PROJECT-KAGYA real-model smoke test. Reply with one short sentence.",
    check_fallback: bool = False,
    provider_factory: ProviderFactory | None = None,
) -> list[SmokeStep]:
    """Load configured real models and run minimal generation checks."""

    if settings.model.provider.lower() != "transformers":
        raise TransformersSmokeError(
            "configuration",
            "model.provider must be 'transformers' for this smoke test",
        )

    factory = provider_factory or _load_transformers_provider
    provider = factory(settings)
    steps: list[SmokeStep] = []

    _run_step(
        steps,
        "primary_load",
        lambda: _load_primary(provider),
        failure_category="model_load_failure",
    )
    primary_text = _run_step(
        steps,
        "primary_generate",
        lambda: _generate_primary(provider, prompt),
        failure_category="generation_failure",
    )
    _ensure_visible_text(primary_text, "generation_failure")

    if check_fallback:
        _run_step(
            steps,
            "fallback_load",
            lambda: _load_fallback(provider),
            failure_category="fallback_failure",
        )
        fallback_text = _run_step(
            steps,
            "fallback_generate",
            lambda: str(provider.generate_fallback(prompt)),
            failure_category="fallback_failure",
        )
        _ensure_visible_text(fallback_text, "fallback_failure")

    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Opt-in smoke test for the real Transformers provider."
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--prompt", default="PROJECT-KAGYA real-model smoke test. Reply with one short sentence.")
    parser.add_argument(
        "--check-fallback",
        action="store_true",
        help="Also load and generate with model.fallback_id.",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config) if args.config is not None else get_settings()
    try:
        steps = run_transformers_smoke(
            settings, prompt=args.prompt, check_fallback=args.check_fallback
        )
    except TransformersSmokeError as exc:
        print(f"[FAIL] {exc.category}: {exc.detail}", file=sys.stderr)
        return 1

    for step in steps:
        marker = "OK" if step.ok else "FAIL"
        print(f"[{marker}] {step.name}: {step.message}")
    print("Transformers smoke check passed.")
    return 0


def _load_transformers_provider(settings: Settings) -> Any:
    try:
        from kagya.models.transformers_provider import TransformersProvider
    except (ImportError, ModuleNotFoundError) as exc:
        raise TransformersSmokeError(
            "missing_dependency",
            f"Transformers provider dependencies are unavailable: {exc}",
        ) from exc
    return TransformersProvider(settings)


def _run_step(
    steps: list[SmokeStep],
    name: str,
    operation: Callable[[], Any],
    *,
    failure_category: str,
) -> Any:
    try:
        result = operation()
    except TransformersSmokeError:
        raise
    except Exception as exc:
        steps.append(SmokeStep(name=name, ok=False, message=str(exc)))
        raise TransformersSmokeError(failure_category, f"{name} failed: {exc}") from exc
    steps.append(SmokeStep(name=name, ok=True, message="completed"))
    return result


def _load_primary(provider: Any) -> None:
    provider._get_primary_processor()
    provider._get_primary_model()


def _load_fallback(provider: Any) -> None:
    provider._get_fallback_processor()
    provider._get_fallback_model()


def _generate_primary(provider: Any, prompt: str) -> str:
    return str(provider._generate_with(prompt, provider._get_primary_model(), provider._get_primary_processor()))


def _ensure_visible_text(text: str, category: str) -> None:
    if not text.strip():
        raise TransformersSmokeError(category, "model generated empty text")


if __name__ == "__main__":
    raise SystemExit(main())
