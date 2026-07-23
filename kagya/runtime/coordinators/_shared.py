"""Small helpers shared by coordinator implementations."""

from typing import Any

from kagya.models import ModelProvider


class RuntimeDomainMixin:
    """Provides access to dependencies supplied by the main-loop host."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(name)


def generate_fallback(provider: ModelProvider, prompt: str) -> str:
    generate = getattr(provider, "generate_fallback", None)
    if not callable(generate):
        raise RuntimeError("Model generation failed and no fallback model is available")
    return str(generate(prompt))


def string_values(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str) and item)


def unit_target(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between zero and one")
    return result


def fallback_used(provider: ModelProvider) -> bool:
    return bool(getattr(provider, "last_fallback_used", False))
