"""Dream dataset generation for sleep-time learning."""

from dataclasses import dataclass
import json
from pathlib import Path

from kagya.memory import EpisodicMemoryRecord


@dataclass(frozen=True)
class DreamDatasetRecord:
    input: str
    output: str

    def to_json(self) -> dict[str, str]:
        return {"input": self.input, "output": self.output}


class DreamDatasetGenerator:
    """Write visible high-emotion episodes as JSONL training examples."""

    def generate(
        self,
        episodes: list[EpisodicMemoryRecord],
        dataset_path: Path,
    ) -> list[DreamDatasetRecord]:
        records = [
            DreamDatasetRecord(input=episode.user_input, output=episode.response)
            for episode in episodes
        ]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("w", encoding="utf-8") as dataset_file:
            for record in records:
                dataset_file.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        return records


def format_training_text(record: DreamDatasetRecord) -> str:
    return f"ユーザー: {record.input}\n私: {record.output}<eos>"
