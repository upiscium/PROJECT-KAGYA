from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from kagya.config import load_settings
from kagya.models.dummy_provider import DummyProvider
from kagya.models.model_loader import load_model_provider
from kagya.models.transformers_provider import (
    TransformersProvider,
    is_registry_approved_adapter,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class FakeProcessor:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def __call__(self, *, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        self.texts.append(text)
        token_ids = list(range(1, len(text.split()) + 1)) or [0]
        return {"input_ids": torch.tensor([token_ids])}

    def decode(self, output_ids: torch.Tensor, skip_special_tokens: bool) -> str:
        return " ".join(str(token_id) for token_id in output_ids.tolist())


class FakeChatTemplateProcessor(FakeProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.messages_seen: list[dict[str, str]] | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.messages_seen = messages
        suffix = "<start_of_turn>model\n" if add_generation_prompt else ""
        return f"<start_of_turn>user\n{messages[0]['content']}<end_of_turn>\n{suffix}"


class FakeModel:
    def __init__(self) -> None:
        self.eval_called = False
        self.labels_seen: torch.Tensor | None = None
        self.generate_kwargs: dict | None = None

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs) -> SimpleNamespace:
        self.labels_seen = kwargs["labels"]
        return SimpleNamespace(loss=torch.tensor(2.5))

    def generate(self, **kwargs) -> torch.Tensor:
        self.generate_kwargs = kwargs
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_dummy_provider_is_deterministic() -> None:
    provider = DummyProvider()

    assert provider.generate("hello") == provider.response_text
    assert provider.calculate_loss("context", "target") == provider.loss_value


def test_model_loader_defaults_to_dummy_provider() -> None:
    provider = load_model_provider(load_settings(CONFIG_PATH))

    assert isinstance(provider, DummyProvider)


def test_transformers_provider_rejects_empty_target_text() -> None:
    provider = TransformersProvider(
        load_settings(CONFIG_PATH),
        model=FakeModel(),
        processor=FakeProcessor(),
    )

    with pytest.raises(ValueError, match="target_text"):
        provider.calculate_loss("context", "")


def test_transformers_loss_masks_context_tokens() -> None:
    fake_model = FakeModel()
    provider = TransformersProvider(
        load_settings(CONFIG_PATH),
        model=fake_model,
        processor=FakeProcessor(),
    )

    loss = provider.calculate_loss("one two", " three four")

    assert loss == 2.5
    assert fake_model.eval_called
    assert fake_model.labels_seen is not None
    assert fake_model.labels_seen.tolist() == [[-100, -100, 3, 4]]


def test_transformers_provider_loads_configured_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: dict[str, object] = {}

    def fake_processor_from_pretrained(model_id: str) -> FakeProcessor:
        loaded["processor_model_id"] = model_id
        return FakeProcessor()

    def fake_model_from_pretrained(model_id: str, **kwargs) -> FakeModel:
        loaded["model_model_id"] = model_id
        loaded["model_kwargs"] = kwargs
        return FakeModel()

    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoModelForImageTextToText.from_pretrained",
        fake_model_from_pretrained,
    )

    settings = load_settings(CONFIG_PATH)
    TransformersProvider(settings)

    assert loaded["processor_model_id"] == settings.model.primary_id
    assert loaded["model_model_id"] == settings.model.primary_id
    assert "quantization_config" in loaded["model_kwargs"]


def test_transformers_generate_decodes_only_new_tokens() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(settings, model=fake_model, processor=FakeProcessor())

    generated = provider.generate("prompt has three")

    assert generated == "4 5"


def test_transformers_generate_omits_sampling_kwargs_when_not_sampling() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={"generation": settings.generation.model_copy(update={"do_sample": False})}
    )
    provider = TransformersProvider(settings, model=fake_model, processor=FakeProcessor())

    provider.generate("hello")

    assert fake_model.generate_kwargs is not None
    assert "temperature" not in fake_model.generate_kwargs
    assert "top_p" not in fake_model.generate_kwargs
    assert fake_model.generate_kwargs["do_sample"] is False


def test_transformers_generate_includes_sampling_kwargs_when_sampling() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={"generation": settings.generation.model_copy(update={"do_sample": True})}
    )
    provider = TransformersProvider(settings, model=fake_model, processor=FakeProcessor())

    provider.generate("hello")

    assert fake_model.generate_kwargs is not None
    assert fake_model.generate_kwargs["temperature"] == settings.generation.temperature
    assert fake_model.generate_kwargs["top_p"] == settings.generation.top_p
    assert fake_model.generate_kwargs["do_sample"] is True


def test_transformers_generate_uses_processor_chat_template_when_available() -> None:
    fake_model = FakeModel()
    processor = FakeChatTemplateProcessor()
    provider = TransformersProvider(load_settings(CONFIG_PATH), model=fake_model, processor=processor)

    provider.generate("Context: private runtime\nUser: hello\nAssistant:")

    assert processor.messages_seen == [{"role": "user", "content": "Context: private runtime\nUser: hello"}]
    assert processor.texts[0].startswith("<start_of_turn>user\n")
    assert processor.texts[0].endswith("<start_of_turn>model\n")
    assert "Assistant:" not in processor.texts[0]


def test_transformers_generate_falls_back_when_chat_template_is_unavailable() -> None:
    processor = FakeProcessor()
    provider = TransformersProvider(load_settings(CONFIG_PATH), model=FakeModel(), processor=processor)

    provider.generate("plain prompt")

    assert processor.texts == ["plain prompt"]


def test_adapter_paths_must_be_approved_by_registry(tmp_path: Path) -> None:
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps({"adapters": [{"path": str(adapter_path), "state": "approved"}]}),
        encoding="utf-8",
    )
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={"path": registry_path}
            )
        }
    )

    assert is_registry_approved_adapter(settings, adapter_path)
    assert not is_registry_approved_adapter(settings, tmp_path / "unregistered")
