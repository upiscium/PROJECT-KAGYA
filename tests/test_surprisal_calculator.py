from __future__ import annotations

import torch

from project_kagya.surprisal_calculator import SurprisalCalculator, SurprisalInputs


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(token) for token in text.split() if token]


class DummyOutput:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss


class DummyModel:
    def __init__(self) -> None:
        self.last_input_ids: torch.Tensor | None = None
        self.last_attention_mask: torch.Tensor | None = None
        self.last_labels: torch.Tensor | None = None
        self.seen_grad_enabled: bool | None = None

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> DummyOutput:
        self.last_input_ids = input_ids
        self.last_attention_mask = attention_mask
        self.last_labels = labels
        self.seen_grad_enabled = torch.is_grad_enabled()
        return DummyOutput(torch.tensor(2.5))


def test_build_inputs_masks_context_tokens() -> None:
    calculator = SurprisalCalculator(DummyTokenizer(), DummyModel())

    inputs = calculator.build_inputs("hello world", "new input")

    assert inputs.text == "hello world\nnew input"
    assert inputs.input_ids == [5, 5, 3, 5]
    assert inputs.labels == [-100, -100, 3, 5]
    assert inputs.attention_mask == [1, 1, 1, 1]


def test_build_inputs_allows_empty_context_text() -> None:
    calculator = SurprisalCalculator(DummyTokenizer(), DummyModel())

    inputs = calculator.build_inputs("", "new input")

    assert inputs.text == "new input"
    assert inputs.input_ids == [3, 5]
    assert inputs.labels == [3, 5]


def test_build_inputs_rejects_empty_new_input() -> None:
    calculator = SurprisalCalculator(DummyTokenizer(), DummyModel())

    try:
        calculator.build_inputs("context", "")
    except ValueError as exc:
        assert str(exc) == "new_input must not be empty"
    else:
        raise AssertionError("ValueError was not raised")


def test_compute_surprisal_uses_no_grad_and_returns_loss() -> None:
    model = DummyModel()
    calculator = SurprisalCalculator(DummyTokenizer(), model)
    inputs = SurprisalInputs(
        text="context\nnew",
        input_ids=[7, 3],
        labels=[-100, 3],
        attention_mask=[1, 1],
    )

    surprisal = calculator.compute_surprisal(inputs)

    assert surprisal == 2.5
    assert model.seen_grad_enabled is False
    assert model.last_input_ids is not None
    assert model.last_labels is not None
    assert model.last_attention_mask is not None
    assert model.last_input_ids.tolist() == [[7, 3]]
    assert model.last_labels.tolist() == [[-100, 3]]
    assert model.last_attention_mask.tolist() == [[1, 1]]
