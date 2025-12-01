# tests/test_runner.py
from unittest.mock import MagicMock, patch

import pytest

from green_agent.config import EvalConfig
from green_agent.runner import evaluate_once


@pytest.fixture
def mock_purple_agent():
    agent = MagicMock()
    # Mock the LLM returning a plan string
    agent.generate_plan.return_value = "(mock-action)"
    return agent


def test_evaluate_once_flow(tmp_path, mock_purple_agent):
    """
    Full flow test mocking out:
    1. Purple Agent (LLM)
    2. Compute Metrics (VAL)
    """

    # 1. Setup Config
    cfg = EvalConfig(
        domain_path="domain.pddl",
        problem_path="problem.pddl",
        out_dir=str(tmp_path),
        purple_kind="mock",
        prompt_text="Solve this.",
        optimal_cost=10.0,
    )

    # 2. Mock metrics result
    mock_metrics_obj = MagicMock()
    mock_metrics_obj.valid = True
    mock_metrics_obj.cost_value = 12.0
    mock_metrics_obj.length = 5
    mock_metrics_obj.steps = []
    # These attributes are required for the report
    mock_metrics_obj.first_failure_at = None
    mock_metrics_obj.first_failed_action = None
    mock_metrics_obj.first_failure_reason = None
    mock_metrics_obj.first_failure_detail = None
    mock_metrics_obj.unsat_count = 0
    mock_metrics_obj.redundant_indices = []
    mock_metrics_obj.advice_count = 0
    mock_metrics_obj.advice_top_predicates = []
    mock_metrics_obj.val_stdout = ""
    mock_metrics_obj.val_stderr = ""
    mock_metrics_obj.failure_reason = None
    mock_metrics_obj.val_attempts = 1
    mock_metrics_obj.val_warning = None

    # 3. Patch dependencies
    with (
        patch("green_agent.runner.build_purple", return_value=mock_purple_agent),
        patch("green_agent.runner.compute_metrics", return_value=mock_metrics_obj),
    ):
        record = evaluate_once(cfg)

    # 4. Assertions
    assert record["valid"] is True
    assert record["cost_value"] == 12.0
    # Score = optimal / cost = 10 / 12 = 0.833...
    assert record["score"] == pytest.approx(0.833, 0.01)

    # Ensure files were "written" (runner writes to files)
    # Since we passed tmp_path as run_dir, runner creates a subdirectory.
    # We need to find where it wrote.
    # Actually, evaluate_once creates a timestamped dir unless run_dir is explicit.
    # In the test, we didn't set run_dir explicitly in the config above?
    # Wait, runner.py logic: "run_dir = cfg.run_dir or _make_run_dir..."
    # Let's verify the artifacts exist in the return dictionary
    assert "purple.plan" in str(record["norm_plan_path"])
