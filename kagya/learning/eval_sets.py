"""Evaluation set loading for adapter evaluation."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    prompt: str
    expected: str = ""


@dataclass(frozen=True)
class EvalSet:
    path: Path
    cases: list[EvalCase]


def load_eval_sets(paths: list[Path]) -> list[EvalSet]:
    """Load JSON eval sets, ignoring missing files for bootstrap environments."""

    eval_sets: list[EvalSet] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as eval_file:
            data = json.load(eval_file)
        eval_sets.append(EvalSet(path=path, cases=_cases_from_json(data)))
    return eval_sets


def _cases_from_json(data: Any) -> list[EvalCase]:
    raw_cases = data.get("cases", data) if isinstance(data, dict) else data
    if not isinstance(raw_cases, list):
        return []
    cases: list[EvalCase] = []
    for item in raw_cases:
        if not isinstance(item, dict):
            continue
        cases.append(
            EvalCase(
                prompt=str(item.get("prompt", "")),
                expected=str(item.get("expected", "")),
            )
        )
    return cases
