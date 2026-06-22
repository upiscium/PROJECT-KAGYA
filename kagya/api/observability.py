"""In-memory runtime event log for operator observability."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import count
from typing import Any

from kagya.api.redaction import redact_private_fields


_ids = count(1)


@dataclass(frozen=True)
class RuntimeEvent:
    id: int
    timestamp: str
    category: str
    event_type: str
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeEventLog:
    """Small bounded event buffer for recent operator-visible runtime events."""

    def __init__(self, max_events: int = 100) -> None:
        self._events: deque[RuntimeEvent] = deque(maxlen=max_events)

    def record(
        self,
        *,
        category: str,
        event_type: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            id=next(_ids),
            timestamp=datetime.now(UTC).isoformat(),
            category=category,
            event_type=event_type,
            message=message,
            metadata=redact_private_fields(metadata or {}),
        )
        self._events.append(event)
        return event

    def recent(self, limit: int = 50) -> list[RuntimeEvent]:
        return list(self._events)[-limit:]
