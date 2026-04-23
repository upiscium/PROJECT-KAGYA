"""QLoRA fine-tuning for Gemma using Unsloth."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

SchemaKind = Literal["auto", "plain", "alpaca", "chat"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Google Gemma with QLoRA via Unsloth."
    )
    parser.add_argument("--model-name", default="google/gemma-4-E4B")
    parser.add_argument("--dataset", default="json")
    parser.add_argument(
        "--schema", choices=["auto", "plain", "alpaca", "chat"], default="auto"
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--messages-field", default="messages")
    parser.add_argument(
        "--train-file", required=True, help="Path to training data JSON/JSONL."
    )
    parser.add_argument(
        "--validation-file", default=None, help="Optional validation JSON/JSONL file."
    )
    parser.add_argument("--output-dir", default="./outputs/gemma-4-e4b-qlora")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _validate_file(path: str | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return str(resolved)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def format_plain_record(record: Mapping[str, Any], text_field: str) -> str:
    text = _clean_text(record.get(text_field))
    if not text:
        raise ValueError(
            f"Missing or empty '{text_field}' field in plain-text example."
        )
    return text


def format_alpaca_record(
    record: Mapping[str, Any],
    instruction_field: str,
    input_field: str,
    output_field: str,
) -> str:
    instruction = _clean_text(record.get(instruction_field))
    input_text = _clean_text(record.get(input_field))
    output = _clean_text(record.get(output_field))

    if not instruction:
        raise ValueError(
            f"Missing or empty '{instruction_field}' field in Alpaca example."
        )
    if not output:
        raise ValueError(f"Missing or empty '{output_field}' field in Alpaca example.")

    parts = [f"Instruction:\n{instruction}"]
    if input_text:
        parts.append(f"Input:\n{input_text}")
    parts.append(f"Response:\n{output}")
    return "\n\n".join(parts)


def format_chat_record(record: Mapping[str, Any], messages_field: str) -> str:
    messages = record.get(messages_field)
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError(
            f"Missing or invalid '{messages_field}' field in chat example."
        )

    lines: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError(
                "Chat messages must be mappings with role and content fields."
            )

        role = _clean_text(
            message.get("role") or message.get("from") or message.get("speaker")
        )
        content = _clean_text(
            message.get("content") or message.get("value") or message.get("text")
        )
        if not role or not content:
            raise ValueError("Chat messages must include both role and content.")
        lines.append(f"{role}: {content}")

    if not lines:
        raise ValueError(f"No messages found in '{messages_field}' field.")
    return "\n".join(lines)


def _resolve_schema(columns: Sequence[str], args: argparse.Namespace) -> SchemaKind:
    explicit_schema = args.schema
    if explicit_schema != "auto":
        return explicit_schema

    column_set = set(columns)
    if args.text_field in column_set:
        return "plain"
    if {args.instruction_field, args.output_field}.issubset(column_set):
        return "alpaca"
    if args.messages_field in column_set:
        return "chat"
    raise ValueError(
        "Unable to infer dataset schema. Provide --schema, --text-field, or structured field names."
    )


def _validate_explicit_schema(columns: Sequence[str], args: argparse.Namespace) -> None:
    column_set = set(columns)
    if args.schema == "plain" and args.text_field not in column_set:
        raise ValueError(f"Dataset is missing required field '{args.text_field}'.")
    if args.schema == "alpaca":
        missing = [
            field
            for field in (args.instruction_field, args.output_field)
            if field not in column_set
        ]
        if missing:
            raise ValueError(
                f"Dataset is missing required Alpaca fields: {', '.join(missing)}."
            )
    if args.schema == "chat" and args.messages_field not in column_set:
        raise ValueError(f"Dataset is missing required field '{args.messages_field}'.")


def _format_record(
    record: Mapping[str, Any], schema: SchemaKind, args: argparse.Namespace
) -> str:
    if schema == "plain":
        return format_plain_record(record, args.text_field)
    if schema == "alpaca":
        return format_alpaca_record(
            record, args.instruction_field, args.input_field, args.output_field
        )
    if schema == "chat":
        return format_chat_record(record, args.messages_field)
    raise ValueError(f"Unsupported schema: {schema}")


def _prepare_split(split: Any, schema: SchemaKind, args: argparse.Namespace):
    _validate_explicit_schema(split.column_names, args)

    def to_text(record: Mapping[str, Any]) -> dict[str, str]:
        return {"text": _format_record(record, schema, args)}

    return split.map(to_text, remove_columns=split.column_names)


def _prepare_dataset(dataset: Any, args: argparse.Namespace):
    train_split = dataset["train"]
    schema = _resolve_schema(train_split.column_names, args)
    prepared_splits = {
        split_name: _prepare_split(split, schema, args)
        for split_name, split in dataset.items()
    }
    return prepared_splits, schema


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    train_file = _validate_file(args.train_file)
    validation_file = _validate_file(args.validation_file)

    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    dataset_kwargs = {"data_files": {"train": train_file}}
    if validation_file:
        dataset_kwargs["data_files"]["validation"] = validation_file

    dataset = load_dataset(args.dataset, **dataset_kwargs)
    prepared_dataset, _schema = _prepare_dataset(dataset, args)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        evaluation_strategy="steps" if validation_file else "no",
        fp16=True,
        bf16=False,
        seed=args.seed,
        report_to="none",
        optim="paged_adamw_8bit",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset.get("validation"),
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
