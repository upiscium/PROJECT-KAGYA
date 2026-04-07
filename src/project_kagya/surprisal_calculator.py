from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(slots=True)
class SurprisalInputs:
    text: str
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class ModelOutputProtocol(Protocol):
    loss: torch.Tensor


class ModelProtocol(Protocol):
    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> ModelOutputProtocol: ...


class SurprisalCalculator:
    def __init__(self, tokenizer: TokenizerProtocol, model: ModelProtocol) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def build_inputs(self, context_text: str, new_input: str) -> SurprisalInputs:
        if not isinstance(context_text, str) or not isinstance(new_input, str):
            raise TypeError("context_text and new_input must be strings")
        if new_input == "":
            raise ValueError("new_input must not be empty")

        context_ids = self.tokenizer.encode(context_text, add_special_tokens=False)
        new_input_ids = self.tokenizer.encode(new_input, add_special_tokens=False)
        input_ids = [*context_ids, *new_input_ids]
        labels = [-100] * len(context_ids) + new_input_ids
        attention_mask = [1] * len(input_ids)
        text = new_input if not context_text else f"{context_text}\n{new_input}"
        return SurprisalInputs(
            text=text,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

    def compute_surprisal(self, loss_inputs: SurprisalInputs) -> float:
        if not isinstance(loss_inputs, SurprisalInputs):
            raise TypeError("loss_inputs must be SurprisalInputs")

        input_ids = torch.tensor([loss_inputs.input_ids], dtype=torch.long)
        attention_mask = torch.tensor([loss_inputs.attention_mask], dtype=torch.long)
        labels = torch.tensor([loss_inputs.labels], dtype=torch.long)

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return float(output.loss.item())
