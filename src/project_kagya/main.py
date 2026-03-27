from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .conscious_agent import ConsciousAgent
from .dual_memory_system import DualMemorySystem
from .sleep_consolidation import SleepCycleManager


@dataclass(slots=True)
class ChatRuntime:
    model: Any
    tokenizer: Any
    memory_system: DualMemorySystem
    agent: ConsciousAgent
    sleep_manager: SleepCycleManager


def load_runtime(
    model_name: str = "Qwen/Qwen3.5-9B-Instruct",
    adapter_path: str = "./kagya_subjective_adapter",
) -> ChatRuntime:
    tokenizer, model = _load_base_model(model_name)
    model = _attach_adapter_if_present(model, adapter_path)
    memory_system = DualMemorySystem()
    agent = ConsciousAgent(
        memory_system=memory_system, llm_pipeline=_build_pipeline(model, tokenizer)
    )
    sleep_manager = SleepCycleManager(memory_system=memory_system)
    return ChatRuntime(
        model=model,
        tokenizer=tokenizer,
        memory_system=memory_system,
        agent=agent,
        sleep_manager=sleep_manager,
    )


def chat_once(
    runtime: ChatRuntime, user_input: str, valence: float, arousal: float
) -> str:
    return runtime.agent.generate(user_input, valence, arousal)


def _load_base_model(model_name: str) -> tuple[Any, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
    )
    return tokenizer, model


def _attach_adapter_if_present(model: Any, adapter_path: str) -> Any:
    path = Path(adapter_path)
    if not path.exists():
        return model

    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(path))


def _build_pipeline(model: Any, tokenizer: Any):
    def pipeline(payload: dict[str, str]) -> dict[str, str]:
        prompt = (
            f"{payload['system_prompt']}\n\nUser: {payload['user_prompt']}\nAssistant:"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        generated = model.generate(**inputs, max_new_tokens=256)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        return {"text": text}

    return pipeline
