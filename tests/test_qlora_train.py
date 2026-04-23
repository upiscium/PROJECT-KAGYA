from __future__ import annotations

import sys
from pathlib import Path
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


def test_main_wires_training_pipeline(monkeypatch, tmp_path) -> None:
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "valid.jsonl"
    train_file.write_text('{"text": "hello"}\n', encoding="utf-8")
    validation_file.write_text('{"text": "world"}\n', encoding="utf-8")

    calls: dict[str, object] = {}

    class Split:
        column_names = ["text"]

        def __init__(self, rows):
            self.rows = rows

        def map(self, fn, remove_columns):
            calls["remove_columns"] = remove_columns
            return [fn(row) for row in self.rows]

    class FakeDataset(dict):
        pass

    def load_dataset(dataset_name, **kwargs):
        calls["dataset_name"] = dataset_name
        calls["data_files"] = kwargs["data_files"]
        return FakeDataset(
            {
                "train": Split([{"text": "hello"}]),
                "validation": Split([{"text": "world"}]),
            }
        )

    class FakeTrainingArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def __init__(self):
            self.saved_dir = None

        def save_pretrained(self, output_dir):
            self.saved_dir = output_dir

    class FakeModel:
        def __init__(self):
            self.device = "cuda:0"

    class FakeFastLanguageModel:
        @staticmethod
        def from_pretrained(**kwargs):
            calls["from_pretrained"] = kwargs
            return FakeModel(), FakeTokenizer()

        @staticmethod
        def get_peft_model(model, **kwargs):
            calls["peft"] = kwargs
            return model

    class FakeTrainer:
        def __init__(self, **kwargs):
            calls["trainer_kwargs"] = kwargs

        def train(self):
            calls["train_called"] = True

        def save_model(self, output_dir):
            calls["save_model"] = output_dir

    datasets_mod = SimpleNamespace(load_dataset=load_dataset)
    transformers_mod = SimpleNamespace(TrainingArguments=FakeTrainingArguments)
    trl_mod = SimpleNamespace(SFTTrainer=FakeTrainer)
    unsloth_mod = SimpleNamespace(FastLanguageModel=FakeFastLanguageModel)

    monkeypatch.setitem(sys.modules, "datasets", datasets_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)
    monkeypatch.setitem(sys.modules, "trl", trl_mod)
    monkeypatch.setitem(sys.modules, "unsloth", unsloth_mod)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "project-kagya-qlora",
            "--train-file",
            str(train_file),
            "--validation-file",
            str(validation_file),
            "--output-dir",
            str(tmp_path / "out"),
        ],
    )

    from project_kagya import qlora_train

    qlora_train.main()

    assert calls["dataset_name"] == "json"
    assert calls["data_files"] == {
        "train": str(train_file),
        "validation": str(validation_file),
    }
    assert calls["remove_columns"] == ["text"]
    assert calls["trainer_kwargs"]["processing_class"] is not None
    assert calls["trainer_kwargs"]["formatting_func"]({"text": "hello"}) == "hello"
    assert calls["from_pretrained"]["model_name"] == "google/gemma-4-E4B"
    assert calls["train_called"] is True
    assert calls["save_model"] == str(tmp_path / "out")
