"""Tool executor for approved declarative side-effect-free tools."""

import json

from kagya.tools.tool_registry import ToolRegistry
from kagya.tools.tool_sandbox import ToolSandbox
from kagya.tools.tool_audit import ToolAuditLog
from kagya.tools.tool_schema import (
    ToolAuditEvent,
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutionResult,
    ToolStatus,
    ToolType,
)


class ToolExecutionBlocked(RuntimeError):
    """Raised when a tool execution request is blocked by v1.0 safety policy."""


class ToolExecutor:
    """Executor for approved declarative safe tool milestones."""

    def __init__(
        self,
        registry: ToolRegistry,
        sandbox: ToolSandbox | None = None,
        audit_log_store: ToolAuditLog | None = None,
    ) -> None:
        self.registry = registry
        self.sandbox = sandbox or ToolSandbox()
        self.audit_log: list[ToolAuditEvent] = []
        self.audit_log_store = audit_log_store

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        tool = self.registry.lookup(request.tool_name)
        if tool is None:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason="Tool is not registered",
            )
            self._audit(request.tool_name, False, None, None, result.blocked_reason)
            return result
        try:
            self.sandbox.validate(tool)
        except PermissionError as exc:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason=str(exc),
            )
            self._audit(
                tool.name, False, tool.status, tool.tool_type, result.blocked_reason
            )
            return result
        if tool.status != ToolStatus.APPROVED:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason="Tool must be approved before execution",
            )
            self._audit(
                tool.name, False, tool.status, tool.tool_type, result.blocked_reason
            )
            return result
        if tool.tool_type == ToolType.TEXT_TEMPLATE:
            return self._execute_text_template(tool, request)
        if tool.tool_type == ToolType.METADATA_LOOKUP:
            return self._execute_metadata_lookup(tool, request)
        result = ToolExecutionResult(
            tool_name=request.tool_name,
            executed=False,
            blocked_reason="Tool type is not executable in the safe milestone",
        )
        self._audit(
            tool.name, False, tool.status, tool.tool_type, result.blocked_reason
        )
        return result

    def _execute_text_template(
        self, tool: ToolDefinition, request: ToolExecutionRequest
    ) -> ToolExecutionResult:
        try:
            output = tool.output_template.format_map(_SafeFormatMap(request.arguments))
        except Exception as exc:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason=f"Tool template rendering failed: {exc}",
            )
            self._audit(
                tool.name, False, tool.status, tool.tool_type, result.blocked_reason
            )
            return result
        result = ToolExecutionResult(
            tool_name=request.tool_name, executed=True, output=output
        )
        self._audit(
            tool.name, True, tool.status, tool.tool_type, "executed text_template"
        )
        return result

    def _execute_metadata_lookup(
        self, tool: ToolDefinition, request: ToolExecutionRequest
    ) -> ToolExecutionResult:
        key = request.arguments.get("key")
        if not isinstance(key, str):
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason="Metadata lookup requires a string key argument",
            )
            self._audit(
                tool.name, False, tool.status, tool.tool_type, result.blocked_reason
            )
            return result
        if key not in tool.metadata:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason="Metadata key is not available",
            )
            self._audit(
                tool.name, False, tool.status, tool.tool_type, result.blocked_reason
            )
            return result
        try:
            output = _stringify_metadata_value(tool.metadata[key])
        except TypeError as exc:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason=f"Metadata value is not serializable: {exc}",
            )
            self._audit(
                tool.name, False, tool.status, tool.tool_type, result.blocked_reason
            )
            return result
        result = ToolExecutionResult(
            tool_name=request.tool_name, executed=True, output=output
        )
        self._audit(
            tool.name, True, tool.status, tool.tool_type, "executed metadata_lookup"
        )
        return result

    def _audit(
        self,
        tool_name: str,
        executed: bool,
        status: ToolStatus | None,
        tool_type: ToolType | None,
        reason: str | None,
    ) -> None:
        event = ToolAuditEvent(
            tool_name=tool_name,
            executed=executed,
            status=status,
            tool_type=tool_type,
            reason=reason or "",
        )
        self.audit_log.append(event)
        if self.audit_log_store is not None:
            self.audit_log_store.append(event)


class _SafeFormatMap(dict[str, object]):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"Missing tool argument: {key}")


def _stringify_metadata_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
