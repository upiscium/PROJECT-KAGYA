from pathlib import Path

from project_kagya.cli import main as cli_main
from project_kagya.settings import (
    AppSettings,
    LoggingSettings,
    PathSettings,
    RuntimeSettings,
)


class DummyRuntime:
    def handle_turn(self, user_input: str) -> str:
        return f"response:{user_input}"


def test_cli_uses_settings_file(monkeypatch, tmp_path: Path, capsys) -> None:
    settings_path = tmp_path / "settings.toml"
    settings_path.write_text("[runtime]\n", encoding="utf-8")

    settings = AppSettings(
        runtime=RuntimeSettings(input_text="hello from settings"),
        logging=LoggingSettings(file_path="run.log"),
        paths=PathSettings(log_dir="logs", sleep_dir="sleep"),
        source_path=settings_path.resolve(),
    )

    monkeypatch.setattr("project_kagya.cli.load_settings", lambda _: settings)
    monkeypatch.setattr(
        "project_kagya.cli.load_runtime_from_settings", lambda _: DummyRuntime()
    )
    monkeypatch.setattr("sys.argv", ["project-kagya", "--settings", str(settings_path)])

    cli_main()

    captured = capsys.readouterr()
    assert "response:hello from settings" in captured.out
    assert (tmp_path / "logs" / "run.log").read_text(encoding="utf-8").strip() == (
        "response:hello from settings"
    )
