"""Configuration compatibility policy helpers."""

from dataclasses import dataclass
from pathlib import Path

from kagya.config.schema import Settings


@dataclass(frozen=True)
class CompatibilityField:
    field: str
    status: str
    canonical_field: str | None = None


COMPATIBILITY_FIELDS = (
    CompatibilityField(
        field="qlora.alpha",
        status="legacy_alias",
        canonical_field="qlora.lora_alpha",
    ),
    CompatibilityField(
        field="qlora.dropout",
        status="legacy_alias",
        canonical_field="qlora.lora_dropout",
    ),
    CompatibilityField(
        field="adapter_registry.manual_approval_required",
        status="reserved",
    ),
    CompatibilityField(
        field="adapter_registry.allowed_states",
        status="reserved_contract",
    ),
)


def compatibility_report(settings: Settings) -> list[str]:
    """Return operator-readable compatibility notes for loaded settings."""

    notes: list[str] = []
    if settings.qlora.alpha != settings.qlora.lora_alpha:
        notes.append(
            "qlora.alpha differs from qlora.lora_alpha; lora_alpha is training-facing"
        )
    if settings.qlora.dropout != settings.qlora.lora_dropout:
        notes.append(
            "qlora.dropout differs from qlora.lora_dropout; lora_dropout is training-facing"
        )
    if not settings.adapter_registry.manual_approval_required:
        notes.append(
            "adapter_registry.manual_approval_required=false is reserved; manual approval is still enforced"
        )
    return notes


def documented_compatibility_fields(readme_path: Path) -> set[str]:
    text = readme_path.read_text(encoding="utf-8")
    return {field.field for field in COMPATIBILITY_FIELDS if field.field in text}
