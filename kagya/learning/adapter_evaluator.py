"""Score-based adapter evaluation gates."""

from dataclasses import asdict, dataclass
from enum import StrEnum
import json
from pathlib import Path
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

        eval_sets = load_eval_sets(self.settings.adapter_registry.eval_sets)
        score = deterministic_score if deterministic_score is not None else self._score(provider, eval_sets)
        decision = self._decision(score)
        result_path = self._write_result(adapter_id, score, decision, eval_sets)
        self.registry.apply_evaluation(adapter_id, score=score, result_path=result_path)
        return AdapterEvaluationResult(
            adapter_id=adapter_id,
            score=score,
            decision=decision,
            result_path=str(result_path),
            eval_set_count=len(eval_sets),
            case_count=sum(len(eval_set.cases) for eval_set in eval_sets),
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
    ) -> Path:
        result_dir = self.settings.adapter_registry.eval_result_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{adapter_id}.json"
        payload: dict[str, Any] = {
            "adapter_id": adapter_id,
            "score": score,
            "decision": decision.value,
            "eval_sets": [str(eval_set.path) for eval_set in eval_sets],
            "case_count": sum(len(eval_set.cases) for eval_set in eval_sets),
        }
        with result_path.open("w", encoding="utf-8") as result_file:
            json.dump(payload, result_file, indent=2)
        return result_path


def result_to_json(result: AdapterEvaluationResult) -> dict[str, Any]:
    data = asdict(result)
    data["decision"] = result.decision.value
    return data
