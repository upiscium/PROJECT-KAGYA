"""Tool executor skeleton that intentionally executes nothing."""

from kagya.tools.tool_registry import ToolRegistry
from kagya.tools.tool_sandbox import ToolSandbox
from kagya.tools.tool_schema import ToolExecutionRequest, ToolExecutionResult


class ToolExecutionBlocked(RuntimeError):
    """Raised when a tool execution request is blocked by v1.0 safety policy."""


class ToolExecutor:
    """Executor interface that blocks all execution until sandboxing is implemented."""

    def __init__(self, registry: ToolRegistry, sandbox: ToolSandbox | None = None) -> None:
        self.registry = registry
        self.sandbox = sandbox or ToolSandbox()

    def execute(self, request: ToolExecutionRequest) -> ToolExecutionResult:
        tool = self.registry.lookup(request.tool_name)
        if tool is None:
            return ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason="Tool is not registered",
            )
        try:
            self.sandbox.validate(tool)
        except PermissionError as exc:
            return ToolExecutionResult(
                tool_name=request.tool_name,
                executed=False,
                blocked_reason=str(exc),
            )
        return ToolExecutionResult(
            tool_name=request.tool_name,
            executed=False,
            blocked_reason="Tool execution is disabled in v1.0",
        )
