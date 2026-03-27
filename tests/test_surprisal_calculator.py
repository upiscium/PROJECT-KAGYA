from __future__ import annotations

from dataclasses import dataclass

from project_kagya import SurprisalCalculator


class DummyTokenizer:
    def __call__(
        self, text: str, add_special_tokens: bool = False
    ) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


@dataclass
class DummyOutput:
    loss: float


class DummyModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, list[int]]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return DummyOutput(loss=1.25)


def test_calculate_masks_context_tokens() -> None:
    model = DummyModel()
    calculator = SurprisalCalculator(model=model, tokenizer=DummyTokenizer())

    result = calculator.calculate("abc", "de")

    assert result.loss == 1.25
    assert result.labels == [-100, -100, -100, 100, 101]
    assert model.calls[0]["labels"] == [-100, -100, -100, 100, 101]
