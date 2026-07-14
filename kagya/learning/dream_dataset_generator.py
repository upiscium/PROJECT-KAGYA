"""Dream dataset generation for sleep-time learning."""

from dataclasses import dataclass
import json
from pathlib import Path

from kagya.memory import EpisodicMemoryRecord


@dataclass(frozen=True)
class DreamDatasetRecord:
    input: str
    thought: str
    output: str
    schema_version: int = 2
    source_id: str = ""
    validation_status: str = "verified"

    def to_json(self) -> dict[str, str | int]:
        return {
            "schema_version": self.schema_version,
            "source_kind": "verified_episode",
            "source_id": self.source_id,
            "validation_status": self.validation_status,
            "input": self.input,
            "thought": self.thought,
            "output": self.output,
        }


class DreamDatasetGenerator:
    """Write high-emotion episodes as JSONL dream training examples."""

    def generate(
        self,
        episodes: list[EpisodicMemoryRecord],
        dataset_path: Path,
    ) -> list[DreamDatasetRecord]:
        records = [
            DreamDatasetRecord(
                input=episode.user_input,
                thought="",
                output=episode.response,
                source_id=episode.id,
                validation_status=episode.validation_status.value,
            )
            for episode in episodes
        ]
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with dataset_path.open("x", encoding="utf-8") as dataset_file:
            for record in records:
                dataset_file.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        return records


def format_training_text(record: DreamDatasetRecord) -> str:
    return (
        f"ユーザー: {record.input}\n"
        "私: <think>\n"
        f"{record.thought}\n"
        "</think>\n"
        f"{record.output}<eos>"
    )
