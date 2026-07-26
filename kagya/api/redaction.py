"""Shared helpers for removing private runtime fields from API payloads."""

from typing import Any


PRIVATE_FIELD_KEYS = frozenset(
    {
        "hiddenthought",
        "prompt",
        "rawprompt",
        "retrievedmemory",
        "requesthmackey",
        "requestcounts",
    }
)
REDACTED_VALUE = "[redacted]"


def redact_private_fields(
    value: Any, *, private_keys: frozenset[str] = PRIVATE_FIELD_KEYS
) -> Any:
    """Return a copy of a nested payload with private fields redacted."""

    if isinstance(value, dict):
        return {
            key: REDACTED_VALUE
            if "".join(
                character for character in str(key).lower() if character.isalnum()
            )
            in private_keys
            else redact_private_fields(item, private_keys=private_keys)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            redact_private_fields(item, private_keys=private_keys) for item in value
        ]
    return value
