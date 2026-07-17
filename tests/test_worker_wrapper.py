import os
from pathlib import Path
import subprocess


WRAPPER = (
    Path(__file__).resolve().parents[1] / "deploy" / "bin" / "kagya-worker-remote"
)


def test_worker_wrapper_loads_env_and_forwards_worker_arguments(tmp_path: Path) -> None:
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    config = tmp_path / "worker.yaml"
    config.write_text("deployment: worker\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    uv = bin_dir / "uv"
    uv.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$KAGYA_CONFIG_PATH\" \"$CUDA_VISIBLE_DEVICES\" \"$*\"\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    env_file = tmp_path / "worker.env"
    env_file.write_text(
        "\n".join(
            (
                f"KAGYA_APP_DIR={app_dir}",
                f"KAGYA_CONFIG_PATH={config}",
                "CUDA_VISIBLE_DEVICES=1",
                "KAGYA_WORKER_NIX_DEVELOP=0",
            )
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(WRAPPER), "status", "--job-id", "job-1"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "KAGYA_WORKER_ENV_FILE": str(env_file),
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
        },
    )

    assert result.stdout.splitlines() == [
        str(config),
        "1",
        "run kagya-worker status --job-id job-1",
    ]


def test_worker_wrapper_rejects_relative_runtime_paths(tmp_path: Path) -> None:
    env_file = tmp_path / "worker.env"
    env_file.write_text(
        "KAGYA_APP_DIR=relative\nKAGYA_CONFIG_PATH=relative.yaml\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(WRAPPER), "health"],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "KAGYA_WORKER_ENV_FILE": str(env_file)},
    )

    assert result.returncode == 2
    assert "must be absolute paths" in result.stderr
