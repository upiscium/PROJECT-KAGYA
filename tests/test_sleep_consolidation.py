from pathlib import Path

from project_kagya.sleep_consolidation import SleepCycleManager, SleepEpisode
from project_kagya.settings import SleepSettings


def test_sleep_cycle_triage_and_dataset_generation(tmp_path: Path) -> None:
    manager = SleepCycleManager()
    manager.episodes = [
        SleepEpisode("a", "b", "c", valence=0.1, arousal=0.8),
        SleepEpisode("d", "e", "f", valence=0.2, arousal=0.1),
        SleepEpisode("g", "h", "i", valence=-0.7, arousal=0.2),
    ]
    manager.settings = SleepSettings(
        high_arousal_threshold=0.75, high_valence_threshold=0.65
    )

    selected = manager.triage_high_emotion()
    assert len(selected) == 2

    output = manager.generate_dream_dataset(
        lambda prompt: {"thought": "dream", "output": "reply"},
        tmp_path / "dream_dataset.jsonl",
    )
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert '"input": "a"' in lines[0]
