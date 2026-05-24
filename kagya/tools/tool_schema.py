"""Tool schema types for future safe tool support."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ToolStatus(StrEnum):
    DECLARED = "declared"
    GENERATED_PENDING_APPROVAL = "generated_pending_approval"
    APPROVED = "approved"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
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
