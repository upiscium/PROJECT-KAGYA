from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch
from PIL import Image

from kagya.config import load_settings
from kagya.models.dummy_provider import DummyProvider
from kagya.models.model_loader import load_model_provider
from kagya.models.transformers_provider import (
    TransformersProvider,
    is_registry_approved_adapter,
)
from kagya.models.transformers_smoke import (
    TransformersSmokeError,
    run_transformers_smoke,
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


class FakeImageChatTemplateProcessor(FakeChatTemplateProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.images_seen: list[object] | None = None

    def __call__(
        self,
        *,
        text: str,
        return_tensors: str,
        images: list[object] | None = None,
    ) -> dict[str, torch.Tensor]:
        self.texts.append(text)
        self.images_seen = images
        return {"input_ids": torch.tensor([[1, 2]])}


class MissingChatTemplateProcessor(FakeProcessor):
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        raise ValueError(
            "Cannot use apply_chat_template because this processor does not have a chat template."
        )


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


class FailingGenerateModel(FakeModel):
    def generate(self, **kwargs) -> torch.Tensor:
        self.generate_kwargs = kwargs
        raise RuntimeError("primary generation failed")


class FakeSmokeProvider:
    def __init__(self, *, fail_step: str | None = None, empty_generation: bool = False) -> None:
        self.fail_step = fail_step
        self.empty_generation = empty_generation
        self.primary_loaded = False
        self.fallback_loaded = False

    def _get_primary_processor(self) -> object:
        if self.fail_step == "primary_load":
            raise RuntimeError("primary load failed")
        return object()

    def _get_primary_model(self) -> object:
        if self.fail_step == "primary_load":
            raise RuntimeError("primary load failed")
        self.primary_loaded = True
        return object()

    def _get_fallback_processor(self) -> object:
        if self.fail_step == "fallback_load":
            raise RuntimeError("fallback load failed")
        return object()

    def _get_fallback_model(self) -> object:
        if self.fail_step == "fallback_load":
            raise RuntimeError("fallback load failed")
        self.fallback_loaded = True
        return object()

    def _generate_with(self, prompt: str, model: object, processor: object) -> str:
        if self.fail_step == "primary_generate":
            raise RuntimeError("primary generation failed")
        return "" if self.empty_generation else f"generated: {prompt}"

    def generate_fallback(self, prompt: str) -> str:
        if self.fail_step == "fallback_generate":
            raise RuntimeError("fallback generation failed")
        return f"fallback: {prompt}"


def test_dummy_provider_is_deterministic() -> None:
    provider = DummyProvider()

    assert provider.generate("hello") == provider.response_text
    assert provider.calculate_loss("context", "target") == provider.loss_value


def test_model_loader_defaults_to_dummy_provider() -> None:
    provider = load_model_provider(load_settings(CONFIG_PATH))

    assert isinstance(provider, DummyProvider)


def test_transformers_smoke_requires_transformers_provider() -> None:
    with pytest.raises(TransformersSmokeError) as exc_info:
        run_transformers_smoke(load_settings(CONFIG_PATH))

    assert exc_info.value.category == "configuration"


def test_transformers_smoke_runs_primary_and_fallback_checks() -> None:
    settings = _transformers_settings()

    steps = run_transformers_smoke(
        settings,
        prompt="hello",
        check_fallback=True,
        provider_factory=lambda settings: FakeSmokeProvider(),
    )

    assert [step.name for step in steps] == [
        "primary_load",
        "primary_generate",
        "fallback_load",
        "fallback_generate",
    ]
    assert all(step.ok for step in steps)


def test_transformers_smoke_distinguishes_generation_failure() -> None:
    settings = _transformers_settings()

    with pytest.raises(TransformersSmokeError) as exc_info:
        run_transformers_smoke(
            settings,
            provider_factory=lambda settings: FakeSmokeProvider(
                fail_step="primary_generate"
            ),
        )

    assert exc_info.value.category == "generation_failure"


def test_transformers_smoke_distinguishes_fallback_failure() -> None:
    settings = _transformers_settings()

    with pytest.raises(TransformersSmokeError) as exc_info:
        run_transformers_smoke(
            settings,
            check_fallback=True,
            provider_factory=lambda settings: FakeSmokeProvider(
                fail_step="fallback_generate"
            ),
        )

    assert exc_info.value.category == "fallback_failure"


def test_transformers_provider_rejects_empty_target_text() -> None:
    provider = TransformersProvider(
        load_settings(CONFIG_PATH),
        model=FakeModel(),
        processor=FakeProcessor(),
    )

    with pytest.raises(ValueError, match="target_text"):
        provider.calculate_loss("context", "")


def _transformers_settings():
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={"model": settings.model.model_copy(update={"provider": "transformers"})}
    )


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


def test_transformers_provider_loads_configured_model_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    TransformersProvider(settings).generate("hello")

    assert loaded["processor_model_id"] == settings.model.primary_id
    assert loaded["model_model_id"] == settings.model.primary_id
    assert "quantization_config" in loaded["model_kwargs"]


def test_transformers_provider_uses_fallback_when_primary_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_model_ids: list[str] = []

    def fake_processor_from_pretrained(model_id: str) -> FakeProcessor:
        return FakeProcessor()

    def fake_model_from_pretrained(model_id: str, **kwargs) -> FakeModel:
        loaded_model_ids.append(model_id)
        if model_id == load_settings(CONFIG_PATH).model.primary_id:
            raise RuntimeError("primary load failed")
        return FakeModel()

    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoModelForImageTextToText.from_pretrained",
        fake_model_from_pretrained,
    )
    provider = TransformersProvider(load_settings(CONFIG_PATH))

    generated = provider.generate("hello")

    assert generated == "2 3 4 5"
    assert loaded_model_ids == [
        load_settings(CONFIG_PATH).model.primary_id,
        load_settings(CONFIG_PATH).model.fallback_id,
    ]
    assert provider.last_model_id == load_settings(CONFIG_PATH).model.fallback_id
    assert provider.last_fallback_used is True


def test_transformers_provider_uses_fallback_when_primary_generation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_model_ids: list[str] = []

    def fake_processor_from_pretrained(model_id: str) -> FakeProcessor:
        return FakeProcessor()

    def fake_model_from_pretrained(model_id: str, **kwargs) -> FakeModel:
        loaded_model_ids.append(model_id)
        if model_id == load_settings(CONFIG_PATH).model.primary_id:
            return FailingGenerateModel()
        return FakeModel()

    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoModelForImageTextToText.from_pretrained",
        fake_model_from_pretrained,
    )
    provider = TransformersProvider(load_settings(CONFIG_PATH))

    generated = provider.generate("hello")

    assert generated == "2 3 4 5"
    assert loaded_model_ids == [
        load_settings(CONFIG_PATH).model.primary_id,
        load_settings(CONFIG_PATH).model.fallback_id,
    ]
    assert provider.last_fallback_used is True


def test_transformers_provider_retries_primary_on_each_request_after_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = load_settings(CONFIG_PATH)
    primary_attempts = 0

    def fake_processor_from_pretrained(model_id: str) -> FakeProcessor:
        return FakeProcessor()

    def fake_model_from_pretrained(model_id: str, **kwargs) -> FakeModel:
        nonlocal primary_attempts
        if model_id == settings.model.primary_id:
            primary_attempts += 1
            if primary_attempts == 1:
                raise RuntimeError("primary load failed once")
        return FakeModel()

    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr(
        "kagya.models.transformers_provider.AutoModelForImageTextToText.from_pretrained",
        fake_model_from_pretrained,
    )
    provider = TransformersProvider(settings)

    provider.generate("first")
    provider.generate("second")

    assert primary_attempts == 2
    assert provider.last_model_id == settings.model.primary_id
    assert provider.last_fallback_used is False


def test_transformers_generate_decodes_only_new_tokens() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    provider = TransformersProvider(
        settings, model=fake_model, processor=FakeProcessor()
    )

    generated = provider.generate("prompt has three")

    assert generated == "4 5"


def test_transformers_generate_omits_sampling_kwargs_when_not_sampling() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "generation": settings.generation.model_copy(update={"do_sample": False})
        }
    )
    provider = TransformersProvider(
        settings, model=fake_model, processor=FakeProcessor()
    )

    provider.generate("hello")

    assert fake_model.generate_kwargs is not None
    assert "temperature" not in fake_model.generate_kwargs
    assert "top_p" not in fake_model.generate_kwargs
    assert fake_model.generate_kwargs["do_sample"] is False


def test_transformers_generate_includes_sampling_kwargs_when_sampling() -> None:
    fake_model = FakeModel()
    settings = load_settings(CONFIG_PATH)
    settings = settings.model_copy(
        update={
            "generation": settings.generation.model_copy(update={"do_sample": True})
        }
    )
    provider = TransformersProvider(
        settings, model=fake_model, processor=FakeProcessor()
    )

    provider.generate("hello")

    assert fake_model.generate_kwargs is not None
    assert fake_model.generate_kwargs["temperature"] == settings.generation.temperature
    assert fake_model.generate_kwargs["top_p"] == settings.generation.top_p
    assert fake_model.generate_kwargs["do_sample"] is True


def test_transformers_generate_uses_processor_chat_template_when_available() -> None:
    fake_model = FakeModel()
    processor = FakeChatTemplateProcessor()
    provider = TransformersProvider(
        load_settings(CONFIG_PATH), model=fake_model, processor=processor
    )

    provider.generate("Context: private runtime\nUser: hello\nAssistant:")

    assert processor.messages_seen == [
        {"role": "user", "content": "Context: private runtime\nUser: hello"}
    ]
    assert processor.texts[0].startswith("<start_of_turn>user\n")
    assert processor.texts[0].endswith("<start_of_turn>model\n")
    assert "Assistant:" not in processor.texts[0]


def test_transformers_generate_with_image_attachment_uses_image_inputs(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (1, 1), color="white").save(image_path)
    fake_model = FakeModel()
    processor = FakeImageChatTemplateProcessor()
    provider = TransformersProvider(
        load_settings(CONFIG_PATH), model=fake_model, processor=processor
    )

    generated = provider.generate_with_attachments(
        "describe image",
        [
            {
                "type": "image",
                "url": image_path.as_uri(),
                "name": "sample.png",
                "content_type": "image/png",
            }
        ],
    )

    assert generated == "3 4 5"
    assert processor.images_seen is not None
    assert len(processor.images_seen) == 1
    assert processor.messages_seen is not None
    assert processor.messages_seen[0]["content"] == [
        {"type": "text", "text": "describe image"},
        {"type": "image"},
    ]


def test_transformers_rejects_invalid_image_attachment(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.txt"
    image_path.write_text("not an image", encoding="utf-8")
    provider = TransformersProvider(
        load_settings(CONFIG_PATH), model=FakeModel(), processor=FakeImageChatTemplateProcessor()
    )

    with pytest.raises(ValueError, match="Unsupported image content type"):
        provider.generate_with_attachments(
            "describe image",
            [
                {
                    "type": "image",
                    "url": image_path.as_uri(),
                    "name": "sample.txt",
                    "content_type": "text/plain",
                }
            ],
        )


def test_transformers_generate_falls_back_when_chat_template_is_unavailable() -> None:
    processor = FakeProcessor()
    provider = TransformersProvider(
        load_settings(CONFIG_PATH), model=FakeModel(), processor=processor
    )

    provider.generate("plain prompt")

    assert processor.texts == ["plain prompt"]


def test_transformers_generate_falls_back_when_processor_has_no_chat_template() -> None:
    processor = MissingChatTemplateProcessor()
    provider = TransformersProvider(
        load_settings(CONFIG_PATH), model=FakeModel(), processor=processor
    )

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
