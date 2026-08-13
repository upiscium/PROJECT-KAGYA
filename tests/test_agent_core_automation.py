from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import ModuleType

import pytest


def _agent_core() -> ModuleType:
    path = Path(__file__).parents[1] / ".automation" / "bin" / "agent_core.py"
    spec = spec_from_file_location("agent_core_under_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _task_lifecycle() -> ModuleType:
    path = Path(__file__).parents[1] / ".automation" / "bin" / "task_lifecycle.py"
    spec = spec_from_file_location("task_lifecycle_under_test", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_state(
    root: Path,
    *,
    task: str,
    branch: str,
    worktree: Path,
    base: str = "main",
) -> None:
    state = root / ".task-state" / "task.md"
    state.parent.mkdir()
    state.write_text(
        "\n".join(
            (
                f"- Task ID: {task}",
                f"- Branch: {branch}",
                f"- Worktree: {worktree}",
                f"- Base branch: {base}",
            )
        ),
        encoding="utf-8",
    )


def test_task_publication_requires_exact_task_state_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _agent_core()
    branch = "task/issue-55-agent-template"
    _write_state(tmp_path, task="issue-55", branch=branch, worktree=tmp_path)
    monkeypatch.setattr(module, "current_branch", lambda _root: branch)
    monkeypatch.setattr(module, "default_branch", lambda _root: "main")

    assert module.ensure_task_branch(tmp_path, "issue-55") == branch


@pytest.mark.parametrize(
    ("requested", "state_task", "state_branch"),
    [
        ("55", "issue-55", "task/issue-55-agent-template"),
        ("issue-55", "other", "task/issue-55-agent-template"),
        ("issue-55", "issue-55", "task/issue-55-other"),
    ],
)
def test_task_publication_rejects_rebound_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    state_task: str,
    state_branch: str,
) -> None:
    module = _agent_core()
    branch = "task/issue-55-agent-template"
    _write_state(tmp_path, task=state_task, branch=state_branch, worktree=tmp_path)
    monkeypatch.setattr(module, "current_branch", lambda _root: branch)
    monkeypatch.setattr(module, "default_branch", lambda _root: "main")

    with pytest.raises(module.AutomationError):
        module.ensure_task_branch(tmp_path, requested)


def test_task_publication_rejects_rebound_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _agent_core()
    branch = "task/issue-55-agent-template"
    _write_state(
        tmp_path,
        task="issue-55",
        branch=branch,
        worktree=tmp_path / "sibling",
    )
    monkeypatch.setattr(module, "current_branch", lambda _root: branch)
    monkeypatch.setattr(module, "default_branch", lambda _root: "main")

    with pytest.raises(module.AutomationError):
        module.ensure_task_branch(tmp_path, "issue-55")


def test_packed_batch_rejects_shell_metacharacters_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _task_lifecycle()
    monkeypatch.setattr(
        module,
        "require_main_worktree",
        lambda _root: pytest.fail("invalid Task ID reached lifecycle execution"),
    )

    with pytest.raises(module.LifecycleError, match="invalid Task ID"):
        module.batch_plan(tmp_path, "task-one task-two; touch-marker".split())


def test_automation_source_uses_just_shell_quoting() -> None:
    recipe = (
        Path(__file__).parents[1] / ".automation" / "just" / "automation.just"
    ).read_text(encoding="utf-8")
    assert "--source '{{source}}'" in recipe


def test_publication_uses_allowed_task_state_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _agent_core()
    _write_state(
        tmp_path,
        task="issue-55",
        branch="task/issue-55-agent-template",
        worktree=tmp_path,
        base="develop",
    )
    monkeypatch.setattr(
        module,
        "policy",
        lambda _root: {"publication": {"base_branches": ["main", "develop"]}},
    )

    assert module.publication_base(tmp_path, "issue-55") == "develop"


def test_automation_maintenance_does_not_relax_secret_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _agent_core()
    monkeypatch.setattr(
        module,
        "policy",
        lambda _root: {
            "paths": {
                "automation_core": [".automation/**"],
                "secret_patterns": [".env", "secret"],
            }
        },
    )
    module.reject_unsafe_paths(
        tmp_path, [".automation/VERSION"], allow_automation_core=True
    )
    with pytest.raises(module.AutomationError, match="potential secret"):
        module.reject_unsafe_paths(
            tmp_path,
            [".automation/VERSION", ".env"],
            allow_automation_core=True,
        )
