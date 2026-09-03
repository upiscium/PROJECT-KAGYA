"""Privacy-boundary helpers for data that may cross durable/public boundaries."""

from typing import Any


PRIVATE_FIELD_KEYS = frozenset(
    {
        "hiddenthought",
        "thought",
        "reasoning",
        "chainofthought",
        "prompt",
        "rawprompt",
        "rawgeneration",
        "retrievedmemory",
        "privatestate",
    }
)


def normalize_private_key(key: object) -> str:
    """Normalize field names so spelling style cannot bypass private-key checks."""

    return "".join(character for character in str(key).casefold() if character.isalnum())


def contains_private_fields(value: Any) -> bool:
    """Return whether a nested mapping/sequence contains a private field name."""

    if isinstance(value, dict):
        return any(
            normalize_private_key(key) in PRIVATE_FIELD_KEYS
            or contains_private_fields(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_private_fields(item) for item in value)
    return False


def reject_private_fields(value: Any, *, context: str) -> None:
    """Fail closed when private fields are presented to a durable/public boundary."""

    if contains_private_fields(value):
        raise ValueError(f"{context} cannot contain private model fields")


def scrub_private_fields(value: Any) -> Any:
    """Return a copy with private-keyed values removed for one-way legacy migration."""

    if isinstance(value, dict):
        return {
            key: scrub_private_fields(item)
            for key, item in value.items()
            if normalize_private_key(key) not in PRIVATE_FIELD_KEYS
        }
    if isinstance(value, list):
        return [scrub_private_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(scrub_private_fields(item) for item in value)
    return value
