"""Tool schema types for future safe tool support."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ToolStatus(StrEnum):
    DECLARED = "declared"
    GENERATED_PENDING_APPROVAL = "generated_pending_approval"
    APPROVED = "approved"
    DISABLED = "disabled"


class ToolType(StrEnum):
    METADATA = "metadata"
    TEXT_TEMPLATE = "text_template"
    METADATA_LOOKUP = "metadata_lookup"
    SHELL = "shell"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    tool_type: ToolType = ToolType.METADATA
    output_template: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    status: ToolStatus = ToolStatus.DECLARED
    human_approved: bool = False
    generated: bool = False


@dataclass(frozen=True)
class ToolExecutionRequest:
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    requires_human_approval: bool = True


@dataclass(frozen=True)
class ToolExecutionResult:
    tool_name: str
    executed: bool
    output: str | None = None
    blocked_reason: str | None = None


@dataclass(frozen=True)
class ToolAuditEvent:
    tool_name: str
    executed: bool
    status: ToolStatus | None
    tool_type: ToolType | None
    reason: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
