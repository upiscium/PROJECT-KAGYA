from __future__ import annotations

import sys

from project_kagya.cli import main


def test_cli_runs_embodied_emotion_demo(capsys) -> None:
    sys_argv = sys.argv
    sys.argv = ["project-kagya"]
    try:
        exit_code = main()
    finally:
        sys.argv = sys_argv

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "body_state:" in captured.out
    assert "emotion_state:" in captured.out


def test_cli_runs_demo_with_explicit_flag(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["project-kagya", "--demo"])

    assert main() == 0
