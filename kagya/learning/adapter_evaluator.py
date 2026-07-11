"""Score-based adapter evaluation gates."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterRegistry, AdapterStatus
from kagya.learning.eval_sets import EvalSet, load_eval_sets
from kagya.models import ModelProvider


class AdapterEvaluationDecision(StrEnum):
    TRIAL_ACTIVE = "trial_active"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AdapterEvaluationResult:
    adapter_id: str
    score: float
    decision: AdapterEvaluationDecision
    result_path: str
    eval_set_count: int
    case_count: int
    previous_score: float | None = None
    score_delta: float | None = None
    regression: bool = False
    status_before: str = ""
    status_after: str = ""


class AdapterEvaluator:
    """Evaluate candidate adapters and apply registry threshold decisions."""

    def __init__(self, settings: Settings, registry: AdapterRegistry) -> None:
        self.settings = settings
        self.registry = registry

    def evaluate(
        self,
        adapter_id: str,
        provider: ModelProvider,
        *,
        deterministic_score: float | None = None,
    ) -> AdapterEvaluationResult:
        entry = self.registry.lookup(adapter_id)
        if entry is None:
            raise ValueError(f"Unknown adapter: {adapter_id}")
        if entry.status != AdapterStatus.CANDIDATE:
            raise ValueError("Only candidate adapters can be evaluated")
        status_before = entry.status.value

        eval_sets = load_eval_sets(
            self.settings.adapter_registry.eval_sets,
            require_existing=deterministic_score is None,
        )
        case_count = sum(len(eval_set.cases) for eval_set in eval_sets)
        if deterministic_score is None and case_count == 0:
            raise ValueError("No evaluation cases loaded; configure eval sets or provide a deterministic score")
        score = deterministic_score if deterministic_score is not None else self._score(provider, eval_sets)
        decision = self._decision(score)
        previous_score = self._previous_score(adapter_id)
        score_delta = None if previous_score is None else score - previous_score
        regression = score_delta is not None and score_delta < 0
        status_after = decision.value
        result_path = self._write_result(
            adapter_id,
            score,
            decision,
            eval_sets,
            previous_score=previous_score,
            score_delta=score_delta,
            regression=regression,
            status_before=status_before,
            status_after=status_after,
        )
        self.registry.apply_evaluation(adapter_id, score=score, result_path=result_path)
        return AdapterEvaluationResult(
            adapter_id=adapter_id,
            score=score,
            decision=decision,
            result_path=str(result_path),
            eval_set_count=len(eval_sets),
            case_count=case_count,
            previous_score=previous_score,
            score_delta=score_delta,
            regression=regression,
            status_before=status_before,
            status_after=status_after,
        )

    def _score(self, provider: ModelProvider, eval_sets: list[EvalSet]) -> float:
        cases = [case for eval_set in eval_sets for case in eval_set.cases]
        if not cases:
            return 0.0
        matches = 0
        for case in cases:
            output = provider.generate(case.prompt)
            if case.expected and case.expected in output:
                matches += 1
        return matches / len(cases)

    def _decision(self, score: float) -> AdapterEvaluationDecision:
        if score >= self.settings.adapter_registry.trial_threshold:
            return AdapterEvaluationDecision.TRIAL_ACTIVE
        if score < self.settings.adapter_registry.reject_threshold:
            return AdapterEvaluationDecision.REJECTED
        return AdapterEvaluationDecision.CANDIDATE

    def _write_result(
        self,
        adapter_id: str,
        score: float,
        decision: AdapterEvaluationDecision,
        eval_sets: list[EvalSet],
        *,
        previous_score: float | None,
        score_delta: float | None,
        regression: bool,
        status_before: str,
        status_after: str,
    ) -> Path:
        result_dir = self.settings.adapter_registry.eval_result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC).isoformat()
        result_path = result_dir / f"{_safe_filename(adapter_id)}-{_timestamp_slug(created_at)}.json"
        payload: dict[str, Any] = {
            "adapter_id": adapter_id,
            "score": score,
            "previous_score": previous_score,
            "score_delta": score_delta,
            "regression": regression,
            "decision": decision.value,
            "status_before": status_before,
            "status_after": status_after,
            "eval_sets": [str(eval_set.path) for eval_set in eval_sets],
            "case_count": sum(len(eval_set.cases) for eval_set in eval_sets),
            "created_at": created_at,
        }
        with result_path.open("w", encoding="utf-8") as result_file:
            json.dump(payload, result_file, indent=2)
        return result_path

    def _previous_score(self, adapter_id: str) -> float | None:
        result_dir = self.settings.adapter_registry.eval_result_dir
        if not result_dir.exists():
            return None
        matching: list[tuple[float, float]] = []
        for path in result_dir.glob("*.json"):
            payload = _read_result_payload(path)
            if payload.get("adapter_id") != adapter_id or payload.get("score") is None:
                continue
            matching.append((path.stat().st_mtime, float(payload["score"])))
        if not matching:
            return None
        return sorted(matching)[-1][1]


def result_to_json(result: AdapterEvaluationResult) -> dict[str, Any]:
    data = asdict(result)
    data["decision"] = result.decision.value
    return data


def _read_result_payload(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as result_file:
            payload = json.load(result_file)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_filename(adapter_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", adapter_id).strip("-.") or "adapter"


def _timestamp_slug(timestamp: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", timestamp)
