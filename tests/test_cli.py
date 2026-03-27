from __future__ import annotations

from project_kagya import cli


def test_build_parser_has_expected_defaults() -> None:
    parser = cli.build_parser()
    args = parser.parse_args(["--input", "hello"])

    assert args.model_name == "Qwen/Qwen3.5-9B-Instruct"
    assert args.adapter_path == "./kagya_subjective_adapter"
    assert args.user_input == "hello"
    assert args.valence == 0.0
    assert args.arousal == 0.0


def test_run_invokes_runtime(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str, str]] = []

    def fake_load_runtime(*, model_name: str, adapter_path: str):
        calls.append((model_name, adapter_path, "load"))
        return "runtime"

    def fake_chat_once(runtime, user_input: str, valence: float, arousal: float) -> str:
        calls.append((runtime, user_input, str(valence)))
        return "reply"

    monkeypatch.setattr(cli, "load_runtime", fake_load_runtime)
    monkeypatch.setattr(cli, "chat_once", fake_chat_once)

    code = cli.run(["--input", "hello", "--valence", "0.2", "--arousal", "0.7"])

    out = capsys.readouterr().out
    assert code == 0
    assert out.strip() == "reply"
    assert calls == [
        (
            "Qwen/Qwen3.5-9B-Instruct",
            "./kagya_subjective_adapter",
            "load",
        ),
        ("runtime", "hello", "0.2"),
    ]
