"""Generated tool proposal skeleton."""

from dataclasses import dataclass

from kagya.tools.tool_schema import ToolDefinition, ToolStatus


@dataclass(frozen=True)
class GeneratedToolProposal:
    tool: ToolDefinition
    generated_code: str
    requires_human_approval: bool = True


class ToolGenerator:
    """Generator interface that proposes tools but never registers or executes them."""

    def propose(self, name: str, description: str, generated_code: str) -> GeneratedToolProposal:
        return GeneratedToolProposal(
            tool=ToolDefinition(
                name=name,
                description=description,
                status=ToolStatus.GENERATED_PENDING_APPROVAL,
                human_approved=False,
                generated=True,
            ),
            generated_code=generated_code,
        )
