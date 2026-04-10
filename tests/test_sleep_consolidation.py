from __future__ import annotations

from pathlib import Path

from project_kagya.sleep_consolidation import (
    DreamExample,
    EpisodicEntry,
    SemanticCandidate,
    SleepCycleManager,
)


class DummyExtractor:
    def extract(self, episodes: list[EpisodicEntry]) -> list[SemanticCandidate]:
        return [
            SemanticCandidate(
                id=episode.id,
                fact=episode.response,
                source_ids=[episode.id],
                confidence=0.9,
                status="confirmed",
                timestamp=episode.timestamp,
            )
            for episode in episodes
        ]


class DummyDreamGenerator:
    def generate(self, candidate: SemanticCandidate) -> DreamExample:
        return DreamExample(
            input=candidate.fact,
            thought=f"think about {candidate.fact}",
            output=f"reply about {candidate.fact}",
            source_ids=list(candidate.source_ids),
            confidence=candidate.confidence,
            status=candidate.status,
        )


def test_triage_episodes_selects_high_arousal_or_valence() -> None:
    manager = SleepCycleManager(
        DummyExtractor(), DummyDreamGenerator(), Path("/tmp/out.jsonl")
    )

    selected = manager.triage_episodes(
        [
            EpisodicEntry("1", "a", "b", 0.1, 0.2, "t"),
            EpisodicEntry("2", "a", "b", 0.7, 0.2, "t"),
            EpisodicEntry("3", "a", "b", 0.1, 0.8, "t"),
        ]
    )

    assert [entry.id for entry in selected] == ["2", "3"]


def test_extract_semantic_candidates_keeps_only_confirmed_high_confidence() -> None:
    class MixedExtractor:
        def extract(self, episodes: list[EpisodicEntry]) -> list[SemanticCandidate]:
            return [
                SemanticCandidate("1", "fact", ["1"], 0.9, "confirmed", "t"),
                SemanticCandidate("2", "fact", ["2"], 0.4, "tentative", "t"),
                SemanticCandidate("3", "fact", ["3"], 0.9, "conflicted", "t"),
            ]

    manager = SleepCycleManager(
        MixedExtractor(), DummyDreamGenerator(), Path("/tmp/out.jsonl")
    )

    confirmed = manager.extract_semantic_candidates(
        [EpisodicEntry("1", "a", "b", 0.7, 0.8, "t")]
    )

    assert [candidate.id for candidate in confirmed] == ["1"]


def test_generate_dream_dataset_writes_jsonl_and_splits_long_text(
    tmp_path: Path,
) -> None:
    manager = SleepCycleManager(
        DummyExtractor(),
        DummyDreamGenerator(),
        tmp_path / "dream.jsonl",
        max_chars=12,
    )

    examples = manager.generate_dream_dataset(
        [
            SemanticCandidate(
                id="1",
                fact="very long fact for splitting",
                source_ids=["s1"],
                confidence=0.9,
                status="confirmed",
                timestamp="t",
            )
        ]
    )

    assert examples
    assert (tmp_path / "dream.jsonl").exists()
    lines = (tmp_path / "dream.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(examples)
    assert all('"status": "confirmed"' in line for line in lines)


def test_run_returns_summary_for_empty_input(tmp_path: Path) -> None:
    manager = SleepCycleManager(
        DummyExtractor(), DummyDreamGenerator(), tmp_path / "dream.jsonl"
    )

    result = manager.run([])

    assert result.selected == 0
    assert result.confirmed == 0
    assert result.generated == 0
    assert result.written_lines == 0
