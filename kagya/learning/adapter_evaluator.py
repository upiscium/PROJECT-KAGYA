"""Score-based adapter evaluation gates."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
import re
from typing import Any

from kagya.config import Settings
from kagya.learning.adapter_registry import AdapterRegistry, AdapterStatus
from kagya.learning.eval_sets import EvalCase, EvalSet, load_eval_sets
from kagya.models import ModelProvider
from kagya.structured_response import parse_structured_response


class AdapterEvaluationDecision(StrEnum):
    TRIAL_ACTIVE = "trial_active"
    CANDIDATE = "candidate"
    REJECTED = "rejected"


@dataclass(frozen=True)
class AdapterEvaluationResult:
    adapter_id: str
    score: float
    baseline_score: float | None
    candidate_score: float
    decision: AdapterEvaluationDecision
    result_path: str
    eval_set_count: int
    case_count: int
    previous_score: float | None = None
    score_delta: float | None = None
    regression: bool = False
    status_before: str = ""
    status_after: str = ""
    holdout_score: float | None = None
    holdout_baseline_score: float | None = None
    drift_scores: dict[str, float] | None = None
    activation_gate_passed: bool = False


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
        baseline_provider: ModelProvider | None = None,
    ) -> AdapterEvaluationResult:
        entry = self.registry.lookup(adapter_id)
        if entry is None:
            raise ValueError(f"Unknown adapter: {adapter_id}")
        if entry.status != AdapterStatus.CANDIDATE:
            raise ValueError("Only candidate adapters can be evaluated")
        status_before = entry.status.value

        eval_sets = load_eval_sets(
            self.settings.adapter_registry.eval_sets,
            require_existing=True,
        )
        if entry.evaluation_dataset_path is not None:
            eval_sets.append(_load_lineage_holdout(Path(entry.evaluation_dataset_path)))
        case_count = sum(len(eval_set.cases) for eval_set in eval_sets)
        if case_count == 0:
            raise ValueError("No evaluation cases loaded; configure eval sets")
        (
            baseline_score,
            candidate_score,
            baseline_dimensions,
            candidate_dimensions,
        ) = self._score_pair(
            provider,
            baseline_provider or provider,
            eval_sets,
        )
        score_delta = candidate_score - baseline_score
        score = candidate_score
        decision = self._decision(score)
        previous_score = self._previous_score(adapter_id)
        regression = score_delta is not None and score_delta < 0
        drift_scores = {
            dimension: candidate_dimensions[dimension] - baseline_dimensions[dimension]
            for dimension in ("identity", "value", "behavior")
            if dimension in candidate_dimensions and dimension in baseline_dimensions
        }
        holdout_score = candidate_dimensions.get("holdout")
        holdout_baseline_score = baseline_dimensions.get("holdout")
        holdout_regression = (
            holdout_score is not None
            and holdout_baseline_score is not None
            and holdout_score
            < holdout_baseline_score
            - self.settings.adapter_registry.holdout_regression_tolerance
        )
        drift_limits = {
            "identity": self.settings.adapter_registry.max_identity_drift,
            "value": self.settings.adapter_registry.max_value_drift,
            "behavior": self.settings.adapter_registry.max_behavior_drift,
        }
        drift_regression = any(
            delta < -drift_limits[dimension]
            for dimension, delta in drift_scores.items()
        )
        quality_gate_passed = (
            score >= self.settings.adapter_registry.trial_threshold and not regression
        )
        holdout_gate_passed = not holdout_regression
        drift_gate_passed = not drift_regression
        activation_gate_passed = all(
            (quality_gate_passed, holdout_gate_passed, drift_gate_passed)
        )
        if regression or holdout_regression or drift_regression:
            decision = AdapterEvaluationDecision.CANDIDATE
        status_after = decision.value
        result_path = self._write_result(
            adapter_id,
            score,
            baseline_score,
            candidate_score,
            decision,
            eval_sets,
            previous_score=previous_score,
            score_delta=score_delta,
            regression=regression,
            status_before=status_before,
            status_after=status_after,
            holdout_score=holdout_score,
            holdout_baseline_score=holdout_baseline_score,
            drift_scores=drift_scores,
            activation_gate_passed=activation_gate_passed,
        )
        self.registry.apply_evaluation(
            adapter_id,
            score=score,
            result_path=result_path,
            next_status=AdapterStatus(decision.value),
            holdout_score=holdout_score,
            holdout_baseline_score=holdout_baseline_score,
            drift_scores=drift_scores,
            quality_gate_passed=quality_gate_passed,
            holdout_gate_passed=holdout_gate_passed,
            drift_gate_passed=drift_gate_passed,
        )
        return AdapterEvaluationResult(
            adapter_id=adapter_id,
            score=score,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            decision=decision,
            result_path=str(result_path),
            eval_set_count=len(eval_sets),
            case_count=case_count,
            previous_score=previous_score,
            score_delta=score_delta,
            regression=regression,
            status_before=status_before,
            status_after=status_after,
            holdout_score=holdout_score,
            holdout_baseline_score=holdout_baseline_score,
            drift_scores=drift_scores,
            activation_gate_passed=activation_gate_passed,
        )

    def _score_pair(
        self,
        candidate_provider: ModelProvider,
        baseline_provider: ModelProvider,
        eval_sets: list[EvalSet],
    ) -> tuple[float, float, dict[str, float], dict[str, float]]:
        cases = [case for eval_set in eval_sets for case in eval_set.cases]
        if not cases:
            return 0.0, 0.0, {}, {}
        baseline_total = 0.0
        candidate_total = 0.0
        dimension_totals: dict[str, list[float]] = {}
        baseline_dimension_totals: dict[str, list[float]] = {}
        for eval_set in eval_sets:
            dimension = _evaluation_dimension(eval_set.path)
            for case in eval_set.cases:
                baseline_output = baseline_provider.generate(case.prompt)
                if bool(getattr(baseline_provider, "last_fallback_used", False)):
                    raise ValueError(
                        "Baseline provider used fallback during evaluation"
                    )
                candidate_output = candidate_provider.generate(case.prompt)
                if bool(getattr(candidate_provider, "last_fallback_used", False)):
                    raise ValueError(
                        "Candidate provider used fallback during evaluation"
                    )
                baseline_case = _output_score(
                    _structured_visible_or_compatibility_text(baseline_output),
                    case.expected,
                )
                candidate_case = _output_score(
                    _structured_visible_or_compatibility_text(candidate_output),
                    case.expected,
                )
                baseline_total += baseline_case
                candidate_total += candidate_case
                if dimension is not None:
                    baseline_dimension_totals.setdefault(dimension, []).append(
                        baseline_case
                    )
                    dimension_totals.setdefault(dimension, []).append(candidate_case)
        baseline_dimensions = {
            key: sum(values) / len(values)
            for key, values in baseline_dimension_totals.items()
        }
        candidate_dimensions = {
            key: sum(values) / len(values) for key, values in dimension_totals.items()
        }
        return (
            baseline_total / len(cases),
            candidate_total / len(cases),
            baseline_dimensions,
            candidate_dimensions,
        )

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
        baseline_score: float | None,
        candidate_score: float,
        decision: AdapterEvaluationDecision,
        eval_sets: list[EvalSet],
        *,
        previous_score: float | None,
        score_delta: float | None,
        regression: bool,
        status_before: str,
        status_after: str,
        holdout_score: float | None,
        holdout_baseline_score: float | None,
        drift_scores: dict[str, float],
        activation_gate_passed: bool,
    ) -> Path:
        result_dir = self.settings.adapter_registry.eval_result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC).isoformat()
        result_path = (
            result_dir
            / f"{_safe_filename(adapter_id)}-{_timestamp_slug(created_at)}.json"
        )
        payload: dict[str, Any] = {
            "adapter_id": adapter_id,
            "score": score,
            "baseline_score": baseline_score,
            "candidate_score": candidate_score,
            "previous_score": previous_score,
            "score_delta": score_delta,
            "regression": regression,
            "decision": decision.value,
            "status_before": status_before,
            "status_after": status_after,
            "holdout_score": holdout_score,
            "holdout_baseline_score": holdout_baseline_score,
            "drift_scores": drift_scores,
            "activation_gate_passed": activation_gate_passed,
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


def _output_score(output: str, expected: str) -> float:
    expected_tokens = _normalized_tokens(expected)
    if not expected_tokens:
        return 0.0
    output_tokens = _normalized_tokens(output)
    if output_tokens == expected_tokens:
        return 1.0
    if not output_tokens:
        return 0.0
    overlap = sum((Counter(output_tokens) & Counter(expected_tokens)).values())
    return 2 * overlap / (len(output_tokens) + len(expected_tokens))


def _structured_visible_or_compatibility_text(output: str) -> str:
    parsed = parse_structured_response(output)
    return parsed.visible_response if parsed.parse_valid else output


def _normalized_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


def _safe_filename(adapter_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", adapter_id).strip("-.") or "adapter"


def _timestamp_slug(timestamp: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", timestamp)


def _evaluation_dimension(path: Path) -> str | None:
    name = path.stem.casefold()
    return next(
        (item for item in ("identity", "value", "behavior", "holdout") if item in name),
        None,
    )


def _load_lineage_holdout(path: Path) -> EvalSet:
    if not path.is_file():
        raise ValueError(f"Lineage holdout does not exist: {path}")
    cases: list[EvalCase] = []
    try:
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("Lineage holdout records must be objects")
            cases.append(
                EvalCase(
                    prompt=str(value.get("prompt", value.get("input", ""))),
                    expected=str(value.get("expected", value.get("output", ""))),
                )
            )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Lineage holdout is invalid JSONL: {path}") from exc
    return EvalSet(path=path.with_name(f"holdout-{path.name}"), cases=cases)
