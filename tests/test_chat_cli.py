from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from pathlib import Path

import torch

from project_kagya.chat import (
    Attachment,
    ChatTurn,
    build_messages,
    build_user_content,
    parse_attachment_command,
    render_prompt,
    trim_generated_text,
)


def test_build_messages_includes_system_and_history() -> None:
    messages = build_messages(
        "be helpful",
        [
            ChatTurn(role="user", content="hi"),
            ChatTurn(role="assistant", content="hello"),
        ],
        "how are you?",
    )

    assert messages == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "how are you?"},
    ]


def test_build_user_content_preserves_attachment_order() -> None:
    attachments = [
        Attachment(path=Path("/tmp/one.png"), media_type="image", name="one.png"),
        Attachment(path=Path("/tmp/two.wav"), media_type="audio", name="two.wav"),
    ]

    content = build_user_content("describe these", attachments)

    assert content == [
        {"type": "image", "url": "/tmp/one.png"},
        {"type": "audio", "audio": "/tmp/two.wav"},
        {"type": "text", "text": "describe these"},
    ]


def test_parse_attachment_command_supports_multiple_paths(tmp_path) -> None:
    image = tmp_path / "image.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"x")
    audio.write_bytes(b"x")

    attachments = parse_attachment_command(f":attach {image} {audio}")

    assert [attachment.media_type for attachment in attachments] == ["image", "audio"]
    assert [attachment.path for attachment in attachments] == [
        image.resolve(),
        audio.resolve(),
    ]


def test_render_prompt_falls_back_to_plain_text() -> None:
    tokenizer = SimpleNamespace()
    prompt = render_prompt(tokenizer, [{"role": "user", "content": "hi"}], "plain")

    assert prompt == "User: hi\n\nAssistant:"


def test_render_prompt_uses_chat_template_when_available() -> None:
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return f"templated:{len(messages)}:{tokenize}:{add_generation_prompt}"

    prompt = render_prompt(Tokenizer(), [{"role": "user", "content": "hi"}], "auto")

    assert prompt == "templated:1:False:True"


def test_render_prompt_uses_gemma_turn_tokens_when_available() -> None:
    tokenizer = SimpleNamespace(sot_token="<|turn>", eot_token="<turn|>")

    prompt = render_prompt(
        tokenizer,
        [
            {"role": "system", "content": "be helpful"},
            {"role": "user", "content": "hi"},
        ],
        "auto",
    )

    assert (
        prompt
        == "<|turn>system\nbe helpful<turn|>\n<|turn>user\nhi<turn|>\n<|turn>assistant\n"
    )


def test_trim_generated_text_removes_prompt_prefix() -> None:
    assert trim_generated_text("prompt", "prompt answer") == "answer"


def test_trim_generated_text_discards_gemma_control_tokens() -> None:
    assert trim_generated_text("prompt", "<unused56><eos>") == ""


def test_main_handles_attachment_commands_and_submission(
    monkeypatch, tmp_path, capsys
) -> None:
    image = tmp_path / "image.png"
    audio = tmp_path / "voice.wav"
    image.write_bytes(b"png")
    audio.write_bytes(b"wav")

    calls: dict[str, object] = {}

    class BatchFeature(dict):
        def to(self, device):
            calls["device"] = device
            return self

    class FakeProcessor:
        chat_template = "template"

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ):
            calls["messages"] = messages
            return BatchFeature({"input_ids": torch.tensor([[1, 2, 3]])})

        def batch_decode(self, outputs, skip_special_tokens=False):
            calls["decoded_outputs"] = outputs
            return ["assistant reply"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            calls["generate_kwargs"] = kwargs
            return torch.tensor([[1, 2, 3, 4, 5]])

    def fake_load_model_and_processor(args):
        calls["args"] = args
        return FakeModel(), FakeProcessor(), "transformers"

    inputs: Iterator[str] = iter(
        [f":attach {image} {audio}", "describe these files", "/exit"]
    )

    monkeypatch.setattr(
        "project_kagya.chat._load_model_and_processor", fake_load_model_and_processor
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.argv", ["project-kagya-chat"])

    from project_kagya import chat

    chat.main()

    out = capsys.readouterr().out
    assert f"attached image: {image.resolve()}" in out
    assert f"attached audio: {audio.resolve()}" in out
    assert "assistant> assistant reply" in out
    assert calls["messages"][0]["role"] == "user"
    assert calls["messages"][0]["content"] == [
        {"type": "image", "url": str(image.resolve())},
        {"type": "audio", "audio": str(audio.resolve())},
        {"type": "text", "text": "describe these files"},
    ]


def test_main_lists_and_clears_pending_attachments(
    monkeypatch, tmp_path, capsys
) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"png")

    class FakeProcessor:
        def batch_decode(self, outputs, skip_special_tokens=False):
            return ["assistant reply"]

    class FakeModel:
        device = "cuda:0"

        def generate(self, **kwargs):
            raise AssertionError("generate should not be called")

    def fake_load_model_and_processor(args):
        return FakeModel(), FakeProcessor(), "transformers"

    inputs = iter(
        [
            f":attach {image}",
            ":list-attachments",
            ":clear-attachments",
            ":list-attachments",
            "/exit",
        ]
    )

    monkeypatch.setattr(
        "project_kagya.chat._load_model_and_processor", fake_load_model_and_processor
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    monkeypatch.setattr("sys.argv", ["project-kagya-chat"])

    from project_kagya import chat

    chat.main()

    out = capsys.readouterr().out
    assert f"attached image: {image.resolve()}" in out
    assert f"- image: {image.resolve()}" in out
    assert "attachments cleared" in out
    assert out.count("no attachments") == 1
