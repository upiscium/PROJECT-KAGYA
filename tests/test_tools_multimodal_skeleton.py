from pathlib import Path
import os

from fastapi.testclient import TestClient

from kagya.api.server import create_app
from kagya.config import Settings, load_settings
from kagya.learning import AdapterRegistry
from kagya.memory import DualMemorySystem
from kagya.models import DummyProvider
from kagya.tools import (
    ToolDefinition,
    ToolExecutionRequest,
    ToolExecutor,
    ToolGenerator,
    ToolRegistry,
    ToolStatus,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
ADMIN_TOKEN = "test-admin-token"


def test_chat_request_accepts_empty_attachments(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/chat",
        json={"text": "hello", "attachments": [], "debug": False},
    )

    assert response.status_code == 200


def test_chat_request_accepts_non_empty_attachments(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/chat",
        json={
            "text": "hello",
            "attachments": [{"type": "image", "url": "file:///tmp/image.png"}],
            "debug": False,
        },
    )

    assert response.status_code == 200


def test_debug_chat_accepts_non_empty_attachments(tmp_path: Path) -> None:
    response = _client(tmp_path).post(
        "/api/chat/debug",
        headers={"X-KAGYA-Admin-Token": ADMIN_TOKEN},
        json={
            "text": "hello",
            "attachments": [{"type": "image", "url": "file:///tmp/image.png"}],
            "debug": True,
        },
    )

    assert response.status_code == 200


def test_tool_executor_skeleton_does_not_execute_registered_tools() -> None:
    registry = ToolRegistry()
    registry.register_declared(
        ToolDefinition(
            name="safe_lookup",
            description="future lookup tool",
            human_approved=True,
            status=ToolStatus.APPROVED,
        )
    )

    result = ToolExecutor(registry).execute(ToolExecutionRequest(tool_name="safe_lookup"))

    assert result.executed is False
    assert result.blocked_reason == "Tool execution is disabled in v1.0"


def test_tool_executor_blocks_unknown_tools() -> None:
    result = ToolExecutor(ToolRegistry()).execute(ToolExecutionRequest(tool_name="missing"))

    assert result.executed is False
    assert result.blocked_reason == "Tool is not registered"


def test_tool_registry_does_not_auto_register_generated_tools() -> None:
    registry = ToolRegistry()
    proposal = ToolGenerator().propose(
        name="generated_shell",
        description="would run generated code later",
        generated_code="import os; os.system('date')",
    )

    assert proposal.requires_human_approval is True
    assert proposal.tool.status == ToolStatus.GENERATED_PENDING_APPROVAL
    assert registry.lookup("generated_shell") is None


def test_tool_registry_rejects_unapproved_generated_registration() -> None:
    registry = ToolRegistry()
    proposal = ToolGenerator().propose("generated", "pending", "print('no')")

    try:
        registry.register_declared(proposal.tool)
    except ValueError as exc:
        assert "human approval" in str(exc) or "Pending generated" in str(exc)
    else:
        raise AssertionError("Generated tool registration should require human approval")


def _client(tmp_path: Path) -> TestClient:
    settings = _settings(tmp_path)
    os.environ["KAGYA_TEST_ADMIN_TOKEN"] = ADMIN_TOKEN
    app = create_app(settings)
    app.state.model_provider = DummyProvider()
    app.state.memory_system = DualMemorySystem(settings)
    app.state.adapter_registry = AdapterRegistry(settings)
    return TestClient(app)


def _settings(tmp_path: Path) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "memory": settings.memory.model_copy(
                update={
                    "persist_directory": tmp_path / "chroma",
                    "db1_collection": "hippocampus_tools_test",
                    "db2_collection": "cortex_tools_test",
                }
            ),
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": [],
                }
            ),
            "api": settings.api.model_copy(update={"admin_token_env": "KAGYA_TEST_ADMIN_TOKEN"}),
        }
    )
