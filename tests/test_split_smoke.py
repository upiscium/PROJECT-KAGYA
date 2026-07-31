import json
from pathlib import Path

import pytest

from kagya.config import load_settings
from kagya.training.split_smoke import CONFIRMATION, _bundle_and_job, main


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def test_split_smoke_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=CONFIRMATION):
        main(["--work-dir", str(tmp_path), "--confirm", "wrong"])


def test_split_smoke_bundle_uses_valid_thought_free_dataset(tmp_path: Path) -> None:
    bundle, _ = _bundle_and_job(load_settings(CONFIG_PATH), tmp_path)

    record = json.loads((bundle / "dataset.jsonl").read_text(encoding="utf-8"))

    assert record["thought"] == ""
