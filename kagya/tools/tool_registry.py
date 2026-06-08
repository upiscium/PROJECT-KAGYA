"""In-memory tool registry skeleton."""

from kagya.tools.tool_schema import ToolDefinition, ToolStatus


class ToolRegistry:
    """Registry that never auto-registers generated tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register_declared(self, tool: ToolDefinition) -> ToolDefinition:
        if tool.generated and not tool.human_approved:
            raise ValueError("Generated tools require human approval before registration")
        if tool.status == ToolStatus.GENERATED_PENDING_APPROVAL:
            raise ValueError("Pending generated tools cannot be registered")
        self._tools[tool.name] = tool
        return tool

    def propose_generated(self, tool: ToolDefinition) -> ToolDefinition:
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            tool_type=tool.tool_type,
            output_template=tool.output_template,
            status=ToolStatus.GENERATED_PENDING_APPROVAL,
            human_approved=False,
            generated=True,
        )

    def approve_generated(self, proposal: ToolDefinition) -> ToolDefinition:
        if not proposal.generated:
            raise ValueError("Only generated tool proposals use approval flow")
        approved = ToolDefinition(
            name=proposal.name,
            description=proposal.description,
            input_schema=proposal.input_schema,
            tool_type=proposal.tool_type,
            output_template=proposal.output_template,
            status=ToolStatus.APPROVED,
            human_approved=True,
            generated=True,
        )
        self._tools[approved.name] = approved
        return approved

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def lookup(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)
