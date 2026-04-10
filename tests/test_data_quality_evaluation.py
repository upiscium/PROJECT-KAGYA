from __future__ import annotations

from project_kagya.data_quality_evaluation import DataQualityEvaluator, EvaluatedExample


def test_evaluate_counts_invalid_low_confidence_and_duplicates() -> None:
    evaluator = DataQualityEvaluator(min_confidence=0.5)
    dataset = [
        EvaluatedExample("a", "think tea", "tea", ["1"], 0.9, "confirmed"),
        EvaluatedExample("a", "think tea", "tea", ["1"], 0.9, "confirmed"),
        EvaluatedExample("b", "", "tea", ["2"], 0.9, "confirmed"),
        EvaluatedExample("c", "think tea", "tea", ["3"], 0.1, "confirmed"),
        EvaluatedExample("d", "think tea", "tea", ["4"], 0.9, "conflicted"),
    ]

    report = evaluator.evaluate(dataset)

    assert report.total_examples == 5
    assert report.valid_examples == 3
    assert report.invalid_examples == 2
    assert report.empty_thoughts == 1
    assert report.low_confidence_examples == 1
    assert report.duplicate_examples == 1
    assert report.conflict_examples == 1
    assert report.warnings


def test_filter_dataset_keeps_only_confirmed_high_confidence_examples() -> None:
    evaluator = DataQualityEvaluator(min_confidence=0.5)
    valid = EvaluatedExample("a", "think tea", "tea", ["1"], 0.9, "confirmed")
    invalid = EvaluatedExample("b", "", "tea", ["2"], 0.9, "confirmed")

    filtered = evaluator.filter_dataset([valid, invalid])

    assert filtered == [valid]


def test_summarize_issues_returns_counts() -> None:
    evaluator = DataQualityEvaluator(min_confidence=0.5)

    summary = evaluator.summarize_issues([])

    assert summary["total_examples"] == 0
    assert summary["invalid_examples"] == 0
