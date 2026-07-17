from pathlib import Path

import pytest

from kagya.training.split_smoke import CONFIRMATION, main


def test_split_smoke_requires_explicit_confirmation(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match=CONFIRMATION):
        main(["--work-dir", str(tmp_path), "--confirm", "wrong"])
