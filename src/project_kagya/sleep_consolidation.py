from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .settings import SleepSettings


@dataclass(frozen=True)
class SleepEpisode:
    input: str
    thought: str
    output: str
    valence: float
    arousal: float


class SleepCycleManager:
    def __init__(
        self,
        episodes: Iterable[SleepEpisode] | None = None,
        settings: SleepSettings | None = None,
    ) -> None:
        self.episodes = list(episodes or [])
        self.settings = settings or SleepSettings()

    def triage_high_emotion(self) -> list[SleepEpisode]:
        return [
            episode
            for episode in self.episodes
            if episode.arousal > self.settings.high_arousal_threshold
            or abs(episode.valence) > self.settings.high_valence_threshold
        ]

    def generate_dream_dataset(
        self, llm_pipeline: Any, output_path: str | Path
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        high_emotion = self.triage_high_emotion()
        with output.open("w", encoding="utf-8") as handle:
            for episode in high_emotion:
                payload = llm_pipeline(episode.input)
                thought, response = self._normalize_payload(payload, episode)
                handle.write(
                    json.dumps(
                        {
                            "input": episode.input,
                            "thought": thought,
                            "output": response,
                        },
                        ensure_ascii=False,
                    )
                )
                handle.write("\n")
        return output

    def export_dream_dataset(self, llm_pipeline: Any) -> Path:
        return self.generate_dream_dataset(
            llm_pipeline, self.settings.dream_dataset_path
        )

    def _normalize_payload(
        self, payload: Any, episode: SleepEpisode
    ) -> tuple[str, str]:
        if isinstance(payload, dict):
            thought = str(payload.get("thought", episode.thought)).strip()
            output = str(payload.get("output", episode.output)).strip()
            return thought, output
        if isinstance(payload, str):
            return episode.thought, payload.strip()
        return episode.thought, episode.output
