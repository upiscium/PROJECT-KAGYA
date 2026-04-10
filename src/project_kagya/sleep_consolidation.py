from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence


@dataclass(slots=True)
class EpisodicEntry:
    id: str
    input: str
    response: str
    valence: float
    arousal: float
    timestamp: str
    source: str = "chat"
    confidence: float = 1.0


@dataclass(slots=True)
class SemanticCandidate:
    id: str
    fact: str
    source_ids: list[str]
    confidence: float
    status: str
    timestamp: str


@dataclass(slots=True)
class DreamExample:
    input: str
    thought: str
    output: str
    source_ids: list[str]
    confidence: float
    status: str


@dataclass(slots=True)
class SleepConsolidationResult:
    selected: int
    confirmed: int
    generated: int
    written_lines: int


class SemanticExtractorProtocol(Protocol):
    def extract(
        self, episodes: Sequence[EpisodicEntry]
    ) -> Sequence[SemanticCandidate]: ...


class DreamGeneratorProtocol(Protocol):
    def generate(self, candidate: SemanticCandidate) -> DreamExample: ...


class SleepCycleManager:
    def __init__(
        self,
        extractor: SemanticExtractorProtocol,
        dream_generator: DreamGeneratorProtocol,
        output_path: str | Path,
        confidence_threshold: float = 0.5,
        max_chars: int = 200,
    ) -> None:
        self.extractor = extractor
        self.dream_generator = dream_generator
        self.output_path = Path(output_path)
        self.confidence_threshold = confidence_threshold
        self.max_chars = max_chars

    def triage_episodes(self, episodes: Sequence[EpisodicEntry]) -> list[EpisodicEntry]:
        return [entry for entry in episodes if self._is_selected(entry)]

    def extract_semantic_candidates(
        self, episodes: Sequence[EpisodicEntry]
    ) -> list[SemanticCandidate]:
        candidates = list(self.extractor.extract(episodes))
        return [candidate for candidate in candidates if self._is_confirmed(candidate)]

    def generate_dream_dataset(
        self, candidates: Sequence[SemanticCandidate]
    ) -> list[DreamExample]:
        examples: list[DreamExample] = []
        for candidate in candidates:
            if not self._is_confirmed(candidate):
                continue
            example = self.dream_generator.generate(candidate)
            if self._is_valid_dream_example(example):
                examples.extend(self._split_example(example))

        self._write_jsonl(examples)
        return examples

    def save_adapter(self, adapter_dir: str | Path) -> Path:
        path = Path(adapter_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def run(self, episodes: Sequence[EpisodicEntry]) -> SleepConsolidationResult:
        selected = self.triage_episodes(episodes)
        confirmed = self.extract_semantic_candidates(selected)
        dreams = self.generate_dream_dataset(confirmed)
        written_lines = len(self._read_jsonl())
        return SleepConsolidationResult(
            selected=len(selected),
            confirmed=len(confirmed),
            generated=len(dreams),
            written_lines=written_lines,
        )

    def _is_selected(self, entry: EpisodicEntry) -> bool:
        if not self._is_numeric(entry.valence) or not self._is_numeric(entry.arousal):
            return False
        return entry.arousal > 0.7 or abs(entry.valence) > 0.6

    def _is_confirmed(self, candidate: SemanticCandidate) -> bool:
        return (
            candidate.status == "confirmed"
            and self._is_numeric(candidate.confidence)
            and candidate.confidence >= self.confidence_threshold
            and bool(candidate.fact.strip())
            and candidate.status != "conflicted"
        )

    def _is_valid_dream_example(self, example: DreamExample) -> bool:
        return (
            bool(example.input.strip())
            and bool(example.thought.strip())
            and bool(example.output.strip())
        )

    def _split_example(self, example: DreamExample) -> list[DreamExample]:
        if (
            len(example.input) <= self.max_chars
            and len(example.thought) <= self.max_chars
            and len(example.output) <= self.max_chars
        ):
            return [example]

        chunks: list[DreamExample] = []
        input_chunks = self._chunk_text(example.input)
        thought_chunks = self._chunk_text(example.thought)
        output_chunks = self._chunk_text(example.output)
        max_len = max(len(input_chunks), len(thought_chunks), len(output_chunks))
        for index in range(max_len):
            chunks.append(
                DreamExample(
                    input=input_chunks[index] if index < len(input_chunks) else "",
                    thought=thought_chunks[index]
                    if index < len(thought_chunks)
                    else "",
                    output=output_chunks[index] if index < len(output_chunks) else "",
                    source_ids=list(example.source_ids),
                    confidence=example.confidence,
                    status=example.status,
                )
            )
        return [chunk for chunk in chunks if self._is_valid_dream_example(chunk)]

    def _chunk_text(self, text: str) -> list[str]:
        if len(text) <= self.max_chars:
            return [text]
        return [
            text[index : index + self.max_chars]
            for index in range(0, len(text), self.max_chars)
        ]

    def _write_jsonl(self, examples: Sequence[DreamExample]) -> None:
        lines = [self._to_json_line(example) for example in examples]
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("\n".join(lines), encoding="utf-8")

    def _read_jsonl(self) -> list[str]:
        if not self.output_path.exists():
            return []
        return [
            line
            for line in self.output_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def _to_json_line(self, example: DreamExample) -> str:
        source_ids = ", ".join(f'"{source_id}"' for source_id in example.source_ids)
        return (
            f'{{"input": "{self._escape(example.input)}", '
            f'"thought": "{self._escape(example.thought)}", '
            f'"output": "{self._escape(example.output)}", '
            f'"source_ids": [{source_ids}], '
            f'"confidence": {example.confidence}, '
            f'"status": "{self._escape(example.status)}"}}'
        )

    @staticmethod
    def _escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def _is_numeric(value: object) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
