"""Tool registry with optional JSON persistence."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from kagya.tools.tool_schema import ToolDefinition, ToolStatus, ToolType


class ToolRegistry:
    """Registry that never auto-registers generated tools."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self._tools: dict[str, ToolDefinition] = {}
        if self.path is not None:
            self._load()

    def register_declared(self, tool: ToolDefinition) -> ToolDefinition:
        if tool.generated and not tool.human_approved:
            raise ValueError(
                "Generated tools require human approval before registration"
            )
        if tool.status == ToolStatus.GENERATED_PENDING_APPROVAL:
            raise ValueError("Pending generated tools cannot be registered")
        self._tools[tool.name] = tool
        self._save()
        return tool

    def propose_generated(self, tool: ToolDefinition) -> ToolDefinition:
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=tool.input_schema,
            tool_type=tool.tool_type,
            output_template=tool.output_template,
            metadata=tool.metadata,
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
            metadata=proposal.metadata,
            status=ToolStatus.APPROVED,
            human_approved=True,
            generated=True,
        )
        self._tools[approved.name] = approved
        self._save()
        return approved

    def list(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def lookup(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        data = json.loads(self.path.read_text(encoding="utf-8"))
        tools = data.get("tools", []) if isinstance(data, dict) else []
        for item in tools:
            tool = _tool_from_dict(item)
            self._tools[tool.name] = tool

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "tools": [_tool_to_dict(tool) for tool in self.list()],
        }
        self.path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def _tool_to_dict(tool: ToolDefinition) -> dict[str, object]:
    data = asdict(tool)
    data["tool_type"] = tool.tool_type.value
    data["status"] = tool.status.value
    return data


def _tool_from_dict(data: dict[str, object]) -> ToolDefinition:
    return ToolDefinition(
        name=str(data["name"]),
        description=str(data["description"]),
        input_schema=_dict_value(data, "input_schema"),
        tool_type=ToolType(str(data.get("tool_type", ToolType.METADATA.value))),
        output_template=str(data.get("output_template", "")),
        metadata=_dict_value(data, "metadata"),
        status=ToolStatus(str(data.get("status", ToolStatus.DECLARED.value))),
        human_approved=bool(data.get("human_approved", False)),
        generated=bool(data.get("generated", False)),
    )


def _dict_value(data: dict[str, object], key: str) -> dict[str, object]:
    value = data.get(key)
    return dict(value) if isinstance(value, dict) else {}
