"""Sandbox policy skeleton for future tool execution."""

from dataclasses import dataclass

from kagya.tools.tool_schema import ToolDefinition


@dataclass(frozen=True)
class ToolSandboxPolicy:
    allow_shell: bool = False
    require_human_approval: bool = True
    allow_generated_code_execution: bool = False


class ToolSandbox:
    """Policy checker that blocks unsafe execution in v1.0."""

    def __init__(self, policy: ToolSandboxPolicy | None = None) -> None:
        self.policy = policy or ToolSandboxPolicy()

    def validate(self, tool: ToolDefinition) -> None:
        if tool.generated and not self.policy.allow_generated_code_execution:
            raise PermissionError("Generated tool code execution is disabled in v1.0")
        if self.policy.require_human_approval and not tool.human_approved:
            raise PermissionError("Tool execution requires human approval")
