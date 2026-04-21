from __future__ import annotations

from types import SimpleNamespace

import pytest

from project_kagya.qlora_train import (
    _prepare_dataset,
    format_alpaca_record,
    format_chat_record,
    format_plain_record,
)


def test_format_plain_record_uses_text_field() -> None:
    assert format_plain_record({"text": " hello "}, "text") == "hello"


def test_format_alpaca_record_renders_prompt_block() -> None:
    text = format_alpaca_record(
        {"instruction": "Write a haiku", "input": "", "output": "blue sky"},
        "instruction",
        "input",
        "output",
    )

    assert text == "Instruction:\nWrite a haiku\n\nResponse:\nblue sky"


def test_format_chat_record_renders_message_sequence() -> None:
    text = format_chat_record(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        "messages",
    )

    assert text == "user: hi\nassistant: hello"


def test_prepare_dataset_infers_plain_schema() -> None:
    class Split:
        column_names = ["text"]

        def map(self, fn, remove_columns):
            assert remove_columns == ["text"]
            return [fn({"text": "hello"})]

    dataset = {"train": Split()}
    args = SimpleNamespace(
        schema="auto",
        text_field="text",
        instruction_field="instruction",
        input_field="input",
        output_field="output",
        messages_field="messages",
    )

    prepared, schema = _prepare_dataset(dataset, args)

    assert schema == "plain"
    assert prepared["train"] == [{"text": "hello"}]


def test_format_plain_record_rejects_empty_text() -> None:
    with pytest.raises(ValueError):
        format_plain_record({"text": ""}, "text")
