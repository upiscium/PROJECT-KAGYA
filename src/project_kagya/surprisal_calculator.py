from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import importlib
from typing import Any


@dataclass(frozen=True)
class SurpriseInputs:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def build_surprisal_inputs(
    context_text: str, new_text: str, tokenizer: Any
) -> SurpriseInputs:
    context_ids = list(tokenizer.encode(context_text, add_special_tokens=False))
    new_ids = list(tokenizer.encode(new_text, add_special_tokens=False))
    input_ids = context_ids + new_ids
    labels = [-100] * len(context_ids) + new_ids.copy()
    attention_mask = [1] * len(input_ids)
    return SurpriseInputs(
        input_ids=input_ids, labels=labels, attention_mask=attention_mask
    )


def compute_surprisal_loss(
    model: Any, tokenizer: Any, context_text: str, new_text: str
) -> float:
    try:  # pragma: no cover - optional dependency at runtime
        torch_lib = importlib.import_module("torch")
    except ModuleNotFoundError:  # pragma: no cover - torch is optional in tests
        torch_lib = None

    inputs = build_surprisal_inputs(context_text, new_text, tokenizer)
    encoded = {
        "input_ids": inputs.input_ids,
        "labels": inputs.labels,
        "attention_mask": inputs.attention_mask,
    }
    context: Any = torch_lib.no_grad() if torch_lib is not None else nullcontext()
    with context:
        output = model(**encoded)
    loss = getattr(output, "loss", output)
    if hasattr(loss, "item"):
        return float(loss.item())
    return float(loss)
