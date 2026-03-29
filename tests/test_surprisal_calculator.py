from project_kagya.surprisal_calculator import (
    build_surprisal_inputs,
    compute_surprisal_loss,
)

from typing import cast


class DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [len(word) for word in text.split()]


class DummyModel:
    def __init__(self) -> None:
        self.called_with = None

    def __call__(self, **kwargs: object) -> object:
        self.called_with = kwargs
        labels = cast(list[int], kwargs["labels"])
        return type(
            "Output", (), {"loss": sum(value for value in labels if value != -100)}
        )()


def test_build_surprisal_inputs_masks_context() -> None:
    tokenizer = DummyTokenizer()
    inputs = build_surprisal_inputs("old context", "new text", tokenizer)

    assert inputs.labels[:2] == [-100, -100]
    assert inputs.labels[-2:] == [3, 4]


def test_compute_surprisal_loss_uses_model_output() -> None:
    tokenizer = DummyTokenizer()
    model = DummyModel()

    loss = compute_surprisal_loss(model, tokenizer, "old context", "new text")

    assert loss == 7.0
    assert model.called_with is not None
