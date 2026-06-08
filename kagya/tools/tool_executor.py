"""Tool executor skeleton that intentionally executes nothing."""

from kagya.tools.tool_registry import ToolRegistry
from kagya.tools.tool_sandbox import ToolSandbox
from kagya.tools.tool_schema import ToolAuditEvent, ToolExecutionRequest, ToolExecutionResult, ToolStatus, ToolType


class ToolExecutionBlocked(RuntimeError):
    """Raised when a tool execution request is blocked by v1.0 safety policy."""


class ToolExecutor:
    """Executor interface that blocks all execution until sandboxing is implemented."""

    def __init__(self, registry: ToolRegistry, sandbox: ToolSandbox | None = None) -> None:
        self.registry = registry
        self.sandbox = sandbox or ToolSandbox()
        self.audit_log: list[ToolAuditEvent] = []

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
            self._audit(tool.name, False, tool.status, tool.tool_type, result.blocked_reason)
            return result
        if tool.status != ToolStatus.APPROVED:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason="Tool must be approved before execution",
            )
            self._audit(tool.name, False, tool.status, tool.tool_type, result.blocked_reason)
            return result
        if tool.tool_type != ToolType.TEXT_TEMPLATE:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason="Tool type is not executable in the safe milestone",
            )
            self._audit(tool.name, False, tool.status, tool.tool_type, result.blocked_reason)
            return result
        try:
            output = tool.output_template.format_map(_SafeFormatMap(request.arguments))
        except Exception as exc:
            result = ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason=f"Tool template rendering failed: {exc}",
            )
            self._audit(tool.name, False, tool.status, tool.tool_type, result.blocked_reason)
            return result
        result = ToolExecutionResult(tool_name=request.tool_name, executed=True, output=output)
        self._audit(tool.name, True, tool.status, tool.tool_type, "executed text_template")
        return result

    def _audit(
        self,
        tool_name: str,
        executed: bool,
        status: ToolStatus | None,
        tool_type: ToolType | None,
        reason: str | None,
    ) -> None:
        self.audit_log.append(
            ToolAuditEvent(
                tool_name=tool_name,
                executed=executed,
                status=status,
                tool_type=tool_type,
                reason=reason or "",
            )
        )


class _SafeFormatMap(dict[str, object]):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"Missing tool argument: {key}")
