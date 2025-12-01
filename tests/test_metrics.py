# tests/test_metrics.py
from unittest.mock import patch

from green_agent.metrics import _parse_advice_by_time, compute_metrics
from green_agent.val_wrapper import TraceStep, ValResult


def test_parse_advice_regex(val_stdout_failure):
    advice = _parse_advice_by_time(val_stdout_failure)

    # In fixture: (pick-up b1) has unsatisfied precondition at time 1
    assert 1 in advice

    # In fixture: Set (hand-empty) to true
    # Logic: tuple is (atom, desired_boolean)
    assert ("hand-empty", True) in advice[1]
    assert ("on-table b1", True) in advice[1]


def test_compute_metrics_integration():
    """
    Test compute_metrics without calling subprocess,
    but by mocking run_val to return a specific ValResult.
    """
    mock_val_result = ValResult(
        ok=False,
        stdout="Plan failed",
        stderr="",
        failure_reason="test_reason",
        steps=[TraceStep(time=1, failed=True, action="(bad-action)")],
    )

    with patch("green_agent.metrics.run_val", return_value=mock_val_result):
        metrics = compute_metrics(domain="d", problem="p", plan_text="(bad-action)")

        assert metrics.valid is False
        assert metrics.first_failed_action == "(bad-action)"
        assert metrics.failure_reason == "test_reason"
