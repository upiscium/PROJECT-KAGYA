import json
from pathlib import Path

import pytest

from kagya.config import Settings, load_settings
from kagya.learning import AdapterEvaluationDecision, AdapterEvaluator, AdapterRegistry, AdapterStatus
from kagya.models import DummyProvider


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


class MatchingProvider(DummyProvider):
    def generate(self, prompt: str) -> str:
        return f"answer for {prompt}"


def test_evaluator_promotes_candidate_to_trial_active_with_high_score(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path)
    evaluator = AdapterEvaluator(_settings_for_tmp_registry(tmp_path), registry)

    result = evaluator.evaluate("adapter-a", DummyProvider(), deterministic_score=0.9)

    assert result.decision == AdapterEvaluationDecision.TRIAL_ACTIVE
    assert registry.lookup("adapter-a").status == AdapterStatus.TRIAL_ACTIVE
    assert Path(result.result_path).exists()


def test_evaluator_rejects_candidate_with_low_score(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path)
    evaluator = AdapterEvaluator(_settings_for_tmp_registry(tmp_path), registry)

    result = evaluator.evaluate("adapter-a", DummyProvider(), deterministic_score=0.1)

    assert result.decision == AdapterEvaluationDecision.REJECTED
    assert registry.lookup("adapter-a").status == AdapterStatus.REJECTED


def test_evaluator_keeps_candidate_with_mid_score(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path)
    evaluator = AdapterEvaluator(_settings_for_tmp_registry(tmp_path), registry)

    result = evaluator.evaluate("adapter-a", DummyProvider(), deterministic_score=0.6)

    assert result.decision == AdapterEvaluationDecision.CANDIDATE
    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def test_evaluator_writes_history_and_score_comparison(tmp_path: Path) -> None:
    registry = _registry_with_candidate(tmp_path)
    evaluator = AdapterEvaluator(_settings_for_tmp_registry(tmp_path), registry)

    first = evaluator.evaluate("adapter-a", DummyProvider(), deterministic_score=0.7)
    second = evaluator.evaluate("adapter-a", DummyProvider(), deterministic_score=0.5)

    assert first.result_path != second.result_path
    result_files = sorted((tmp_path / "eval_results").glob("*.json"))
    assert len(result_files) == 2
    second_data = json.loads(Path(second.result_path).read_text(encoding="utf-8"))
    assert second.previous_score == 0.7
    assert second.score_delta == pytest.approx(-0.2)
    assert second.regression is True
    assert second_data["previous_score"] == 0.7
    assert second_data["regression"] is True
    assert second_data["status_before"] == "candidate"
    assert second_data["status_after"] == "candidate"


def test_evaluator_loads_eval_sets_and_writes_result_json(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "eval_set.json"
    eval_set_path.write_text(
        json.dumps({"cases": [{"prompt": "alpha", "expected": "alpha"}]}),
        encoding="utf-8",
    )
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    result = evaluator.evaluate("adapter-a", MatchingProvider())

    result_data = json.loads(Path(result.result_path).read_text(encoding="utf-8"))
    assert result.score == 1.0
    assert result.eval_set_count == 1
    assert result.case_count == 1
    assert result_data["decision"] == "trial_active"


def test_evaluator_fails_when_configured_eval_set_is_missing(tmp_path: Path) -> None:
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[tmp_path / "missing.json"])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    with pytest.raises(ValueError, match="Configured eval set does not exist"):
        evaluator.evaluate("adapter-a", DummyProvider())

    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def test_evaluator_allows_deterministic_score_when_eval_set_is_missing(tmp_path: Path) -> None:
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[tmp_path / "missing.json"])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    result = evaluator.evaluate("adapter-a", DummyProvider(), deterministic_score=0.9)

    assert result.decision == AdapterEvaluationDecision.TRIAL_ACTIVE
    assert registry.lookup("adapter-a").status == AdapterStatus.TRIAL_ACTIVE


def test_evaluator_fails_when_eval_sets_have_no_cases(tmp_path: Path) -> None:
    eval_set_path = tmp_path / "empty_eval_set.json"
    eval_set_path.write_text(json.dumps({"cases": []}), encoding="utf-8")
    settings = _settings_for_tmp_registry(tmp_path, eval_sets=[eval_set_path])
    registry = _registry_with_candidate(tmp_path, settings=settings)
    evaluator = AdapterEvaluator(settings, registry)

    with pytest.raises(ValueError, match="No evaluation cases loaded"):
        evaluator.evaluate("adapter-a", DummyProvider())

    assert registry.lookup("adapter-a").status == AdapterStatus.CANDIDATE


def _registry_with_candidate(tmp_path: Path, settings: Settings | None = None) -> AdapterRegistry:
    registry = AdapterRegistry(settings or _settings_for_tmp_registry(tmp_path))
    registry.register_candidate(
        adapter_id="adapter-a",
        adapter_path=tmp_path / "adapter-a",
        dataset_path=tmp_path / "dataset.jsonl",
        dataset_hash="hash-a",
    )
    return registry


def _settings_for_tmp_registry(
    tmp_path: Path,
    *,
    eval_sets: list[Path] | None = None,
) -> Settings:
    settings = load_settings(CONFIG_PATH)
    return settings.model_copy(
        update={
            "adapter_registry": settings.adapter_registry.model_copy(
                update={
                    "path": tmp_path / "adapter_registry.json",
                    "eval_result_dir": tmp_path / "eval_results",
                    "eval_sets": eval_sets or [],
                    "trial_threshold": 0.8,
                    "reject_threshold": 0.4,
                }
            )
        }
    )
