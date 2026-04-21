from __future__ import annotations

from types import SimpleNamespace

from project_kagya.chat import (
    ChatTurn,
    build_messages,
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


def test_render_prompt_falls_back_to_plain_text() -> None:
    tokenizer = SimpleNamespace()
    prompt = render_prompt(tokenizer, [{"role": "user", "content": "hi"}], "plain")

    assert prompt == "User: hi\n\nAssistant:"


def test_render_prompt_uses_chat_template_when_available() -> None:
    class Tokenizer:
        def apply_chat_template(
            self, messages, tokenize=False, add_generation_prompt=True
        ):
            return f"templated:{len(messages)}:{tokenize}:{add_generation_prompt}"

    prompt = render_prompt(Tokenizer(), [{"role": "user", "content": "hi"}], "auto")

    assert prompt == "templated:1:False:True"


def test_trim_generated_text_removes_prompt_prefix() -> None:
    assert trim_generated_text("prompt", "prompt answer") == "answer"
