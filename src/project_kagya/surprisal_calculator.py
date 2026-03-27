from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SurprisalResult:
    loss: float
    labels: list[int]


class SurprisalCalculator:
    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def calculate(self, context_text: str, input_text: str) -> SurprisalResult:
        full_text = f"{context_text}{input_text}"
        context_ids = self._tokenize_ids(context_text)
        full_ids = self._tokenize_ids(full_text)
        labels = [-100] * len(context_ids) + list(full_ids[len(context_ids) :])

        model_inputs = self._prepare_model_inputs(full_ids, labels)
        loss = self._run_model(model_inputs)
        return SurprisalResult(loss=loss, labels=labels)

    def _tokenize_ids(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], Sequence):
            return list(input_ids[0])
        return list(input_ids)

    def _prepare_model_inputs(
        self, input_ids: list[int], labels: list[int]
    ) -> dict[str, Any]:
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

    def _run_model(self, model_inputs: dict[str, Any]) -> float:
        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            torch = None

        no_grad = torch.no_grad() if torch is not None else nullcontext()
        with no_grad:
            output = self.model(**model_inputs)

        if isinstance(output, dict):
            loss = output["loss"]
        else:
            loss = output.loss

        return float(loss)
