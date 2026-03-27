from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class EpisodicMemoryProtocol(Protocol):
    def get(self) -> dict[str, list[Any]]: ...

    def delete(self, *, ids: list[str]) -> None: ...


@dataclass(slots=True)
class DreamSample:
    input_text: str
    thought: str
    output: str


class SleepCycleManager:
    def __init__(
        self,
        memory_system: Any,
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    ) -> None:
        self.memory_system = memory_system
        self.model_name = model_name

    def triage_high_emotion_episodes(self) -> list[dict[str, Any]]:
        records = self.memory_system.hippocampus.get()
        ids = list(records.get("ids", []))
        documents = list(records.get("documents", []))
        metadatas = list(records.get("metadatas", []))

        selected: list[dict[str, Any]] = []
        for record_id, document, metadata in zip(
            ids, documents, metadatas, strict=False
        ):
            metadata = dict(metadata)
            valence = float(metadata.get("valence", 0.0))
            arousal = float(metadata.get("arousal", 0.0))
            if arousal > 0.7 or abs(valence) > 0.6:
                selected.append(
                    {
                        "id": record_id,
                        "document": str(document),
                        "metadata": metadata,
                    }
                )
        return selected

    def generate_dream_dataset(
        self,
        llm_pipeline: Any,
        episodes: list[dict[str, Any]] | None = None,
        output_path: str | Path = "dream_dataset.jsonl",
    ) -> Path:
        records = (
            episodes if episodes is not None else self.triage_high_emotion_episodes()
        )
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        samples: list[DreamSample] = []
        for episode in records:
            prompt = self._build_dream_prompt(episode)
            result = self._invoke_pipeline(llm_pipeline, prompt)
            sample = self._extract_sample(episode, result)
            samples.append(sample)

        with path.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(
                    json.dumps(
                        {
                            "input": sample.input_text,
                            "thought": sample.thought,
                            "output": sample.output,
                        },
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")

        return path

    def build_sft_text(self, sample: DreamSample) -> str:
        return (
            f"ユーザー: {sample.input_text}\n"
            f"私: <think>\n{sample.thought}\n</think>\n"
            f"{sample.output}<eos>"
        )

    def train_qlora(
        self,
        model: Any,
        dataset_path: str | Path,
        output_dir: str | Path = "./kagya_subjective_adapter",
        trainer_factory: Any | None = None,
    ) -> Any:
        path = Path(dataset_path)
        data = path.read_text(encoding="utf-8").splitlines()
        samples = [json.loads(line) for line in data if line.strip()]

        formatted_samples = [
            self.build_sft_text(
                DreamSample(
                    input_text=item["input"],
                    thought=item["thought"],
                    output=item["output"],
                )
            )
            for item in samples
        ]

        if trainer_factory is None:
            trainer_factory = self._default_trainer_factory

        trainer = trainer_factory(model=model, train_dataset=formatted_samples)
        result = trainer.train()
        self._save_adapter(model, output_dir)
        return result

    def _build_dream_prompt(self, episode: dict[str, Any]) -> str:
        return (
            "Generate an ideal thought process and response for sleep consolidation.\n"
            f"Episode: {episode['document']}\n"
            f"Metadata: {episode['metadata']}\n"
            "Return JSON with keys thought and output."
        )

    def _invoke_pipeline(self, llm_pipeline: Any, prompt: str) -> Any:
        if hasattr(llm_pipeline, "invoke"):
            return llm_pipeline.invoke(prompt)
        if callable(llm_pipeline):
            return llm_pipeline(prompt)
        raise TypeError("llm_pipeline must be callable")

    def _extract_sample(self, episode: dict[str, Any], result: Any) -> DreamSample:
        payload = self._normalize_payload(result)
        return DreamSample(
            input_text=str(episode["document"]),
            thought=str(payload["thought"]),
            output=str(payload["output"]),
        )

    def _normalize_payload(self, result: Any) -> dict[str, Any]:
        if isinstance(result, dict):
            if "thought" in result and "output" in result:
                return result
            if "text" in result:
                return self._parse_text_payload(str(result["text"]))
        if hasattr(result, "content"):
            return self._parse_text_payload(str(result.content))
        if isinstance(result, str):
            return self._parse_text_payload(result)
        return self._parse_text_payload(str(result))

    def _parse_text_payload(self, text: str) -> dict[str, str]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "thought" in parsed and "output" in parsed:
                return {
                    "thought": str(parsed["thought"]),
                    "output": str(parsed["output"]),
                }
        except json.JSONDecodeError:
            pass

        return {"thought": "Reflective consolidation.", "output": text}

    def _save_adapter(self, model: Any, output_dir: str | Path) -> None:
        save_pretrained = getattr(model, "save_pretrained", None)
        if save_pretrained is None:
            raise TypeError("model must support save_pretrained")
        save_pretrained(str(output_dir))

    def _default_trainer_factory(self, *, model: Any, train_dataset: list[str]) -> Any:
        from peft import LoraConfig, prepare_model_for_kbit_training
        from trl import SFTTrainer

        del LoraConfig, prepare_model_for_kbit_training
        return SFTTrainer(model=model, train_dataset=train_dataset)
