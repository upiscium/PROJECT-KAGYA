"""Persistence helpers for coordinator-owned domain state."""

from typing import Any, Callable


class PersistenceCoordinator:
    """Copies store state to and from the existing persistent schema."""

    @staticmethod
    def restore(
        container: dict[str, Any],
        key: str,
        restore: Callable[[Any], None],
        persist: Callable[[], None],
    ) -> None:
        restore(container.get(key))
        persist()

    @staticmethod
    def persist(
        container: dict[str, Any], key: str, serialize: Callable[[], Any]
    ) -> None:
        container[key] = serialize()
