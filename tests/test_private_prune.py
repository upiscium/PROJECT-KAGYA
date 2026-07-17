import json
import os
from pathlib import Path
import subprocess
import time


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "private-prune.sh"


def test_prune_preserves_active_rollback_recent_and_external_adapters(
    tmp_path: Path,
) -> None:
    app_dir, paths = _runtime_tree(tmp_path, include_history=True)
    environment = {
        **os.environ,
        "KAGYA_APP_DIR": str(app_dir),
        "KAGYA_RETENTION_DAYS": "30",
    }

    dry_run = subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert str(paths["expired"]) in dry_run.stdout
    assert str(paths["rollback"]) not in dry_run.stdout
    assert str(paths["recent"]) not in dry_run.stdout
    assert str(paths["active"]) not in dry_run.stdout
    assert "outside the managed directory" in dry_run.stderr
    assert all(path.exists() for path in paths.values())

    subprocess.run(
        ["bash", str(SCRIPT), "--apply", "--confirm", "PRUNE"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert not paths["expired"].exists()
    assert paths["rollback"].exists()
    assert paths["recent"].exists()
    assert paths["active"].exists()
    assert paths["external"].exists()


def test_prune_protects_all_archived_adapters_without_activation_history(
    tmp_path: Path,
) -> None:
    app_dir, paths = _runtime_tree(tmp_path, include_history=False)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--apply", "--confirm", "PRUNE", "--days", "30"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "KAGYA_APP_DIR": str(app_dir)},
    )

    assert "activation history is missing" in result.stderr
    assert all(path.exists() for path in paths.values())


def test_prune_preserves_rollback_target_when_runtime_is_back_on_base(
    tmp_path: Path,
) -> None:
    app_dir, paths = _runtime_tree(tmp_path, include_history=True)
    registry_path = app_dir / ".kagya" / "adapter_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in registry["adapters"]:
        if entry["adapter_id"] == "active":
            entry["status"] = "archived"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    history_path = app_dir / ".kagya" / "adapter_registry_activations.json"
    history_path.write_text(
        json.dumps(
            {
                "activations": [
                    {
                        "action": "activate",
                        "adapter_id": "active",
                        "previous_adapter_id": None,
                    },
                    {
                        "action": "rollback",
                        "adapter_id": None,
                        "previous_adapter_id": "active",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        ["bash", str(SCRIPT), "--apply", "--confirm", "PRUNE", "--days", "30"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "KAGYA_APP_DIR": str(app_dir)},
    )

    assert paths["active"].exists()
    assert not paths["expired"].exists()


def _runtime_tree(
    tmp_path: Path, *, include_history: bool
) -> tuple[Path, dict[str, Path]]:
    app_dir = tmp_path / "app"
    kagya_dir = app_dir / ".kagya"
    adapters_dir = kagya_dir / "adapters"
    adapters_dir.mkdir(parents=True)
    paths = {
        "active": adapters_dir / "active",
        "rollback": adapters_dir / "rollback",
        "expired": adapters_dir / "expired",
        "recent": adapters_dir / "recent",
        "external": tmp_path / "external",
    }
    for path in paths.values():
        path.mkdir()
    old = time.time() - 31 * 86400
    for key in ("active", "rollback", "expired", "external"):
        os.utime(paths[key], (old, old))
    entries = [
        _entry("active", paths["active"], "active"),
        _entry("rollback", paths["rollback"], "archived"),
        _entry("expired", paths["expired"], "archived"),
        _entry("recent", paths["recent"], "archived"),
        _entry("external", paths["external"], "archived"),
    ]
    (kagya_dir / "adapter_registry.json").write_text(
        json.dumps({"adapters": entries}), encoding="utf-8"
    )
    if include_history:
        (kagya_dir / "adapter_registry_activations.json").write_text(
            json.dumps(
                {
                    "activations": [
                        {
                            "action": "activate",
                            "adapter_id": "active",
                            "previous_adapter_id": "rollback",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
    return app_dir, paths


def _entry(adapter_id: str, path: Path, status: str) -> dict[str, str]:
    return {
        "adapter_id": adapter_id,
        "path": str(path),
        "status": status,
    }
