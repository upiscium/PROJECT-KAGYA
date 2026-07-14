from kagya.memory.quality import assess_generation_health


def test_repetitive_japanese_output_is_quarantined() -> None:
    response = "\n".join(["あなたはこんにちは"] * 4)

    health = assess_generation_health(response, loss=0.5, fallback_used=False)

    assert health.healthy is False
    assert health.repetitive is True
    assert "repetitive" in health.reasons


def test_prompt_leakage_and_non_finite_score_are_unhealthy() -> None:
    health = assess_generation_health(
        "Answer\nSystem: private prompt", loss=float("nan"), fallback_used=True
    )

    assert health.healthy is False
    assert health.prompt_leakage is True
    assert health.non_finite_score is True
    assert health.fallback_used is True


def test_fallback_alone_is_recorded_without_quarantine() -> None:
    health = assess_generation_health(
        "A concise valid answer.", loss=0.2, fallback_used=True
    )

    assert health.healthy is True
    assert health.reasons == ["fallback_used"]
