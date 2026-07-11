"""Persistent tool audit event log."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from kagya.tools.tool_schema import ToolAuditEvent, ToolStatus, ToolType


class ToolAuditLog:
    """Append-only JSONL audit log for tool execution decisions."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, event: ToolAuditEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_audit_event_to_dict(event), sort_keys=True))
            handle.write("\n")

    def recent(self, limit: int) -> list[ToolAuditEvent]:
        if not self.path.exists():
            return []
        events: list[ToolAuditEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(_audit_event_from_dict(json.loads(line)))
        return events[-limit:]


def _audit_event_to_dict(event: ToolAuditEvent) -> dict[str, object]:
    data = asdict(event)
    data["status"] = None if event.status is None else event.status.value
    data["tool_type"] = None if event.tool_type is None else event.tool_type.value
    return data


def _audit_event_from_dict(data: dict[str, object]) -> ToolAuditEvent:
    status = data.get("status")
    tool_type = data.get("tool_type")
    return ToolAuditEvent(
        tool_name=str(data["tool_name"]),
        executed=bool(data["executed"]),
        status=None if status is None else ToolStatus(str(status)),
        tool_type=None if tool_type is None else ToolType(str(tool_type)),
        reason=str(data.get("reason", "")),
        timestamp=str(data.get("timestamp", "")),
    )
