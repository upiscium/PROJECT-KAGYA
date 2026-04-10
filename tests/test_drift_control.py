from __future__ import annotations

from project_kagya.drift_control import DriftController


class DummyModel:
    def __init__(self) -> None:
        self.state = {
            "tone": 0.2,
            "length": 0.4,
            "vocabulary": 0.3,
            "known_sample": 0.25,
            "thought": 0.2,
            "iterations": 0,
        }

    def snapshot_state(self) -> dict[str, float]:
        return dict(self.state)

    def restore_state(self, snapshot: dict[str, float]) -> None:
        self.state = dict(snapshot)


def test_snapshot_and_rollback_restore_state() -> None:
    controller = DriftController()
    model = DummyModel()

    snapshot = controller.snapshot(model)
    model.state["tone"] = 0.9
    controller.rollback(model, snapshot)

    assert model.state["tone"] == 0.2


def test_should_accept_update_rejects_large_drift() -> None:
    controller = DriftController()
    before = controller.snapshot(DummyModel())
    after = controller.snapshot(DummyModel())
    after.state["tone"] = 0.9
    after.state["iterations"] = 10

    report = controller.measure_drift(before, after)

    assert controller.should_accept_update(report) is False


def test_should_accept_update_accepts_small_drift() -> None:
    controller = DriftController()
    before = controller.snapshot(DummyModel())
    after_model = DummyModel()
    after_model.state["tone"] = 0.25
    after_model.state["length"] = 0.42
    after = controller.snapshot(after_model)

    report = controller.measure_drift(before, after)

    assert controller.should_accept_update(report) is True


def test_initial_baseline_can_be_snapshot() -> None:
    controller = DriftController()
    model = DummyModel()

    snapshot = controller.snapshot(model)

    assert snapshot.state["tone"] == 0.2
