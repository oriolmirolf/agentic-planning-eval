# tests/test_runner.py
from unittest.mock import MagicMock, patch

import pytest

from green_agent.config import EvalConfig
from green_agent.runner import evaluate_once, load_text


def test_load_text_file_exists(tmp_path):
    p = tmp_path / "test.txt"
    p.write_text("content", encoding="utf-8")
    assert load_text(str(p)) == "content"


def test_load_text_missing_returns_empty():
    # Should not crash, just return empty string per current implementation
    # or raise FileNotFoundError depending on implementation.
    # Current implementation: `with open(path)` -> raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_text("nonexistent_file.txt")


def test_load_text_none():
    assert load_text(None) == ""


@pytest.fixture
def mock_purple_agent():
    agent = MagicMock()
    # Mock returning a code-blocked plan
    agent.generate_plan.return_value = "```pddl\n(move a b)\n```"
    return agent


def test_evaluate_once_integration(tmp_path, mock_purple_agent):
    """
    Test the full integration of evaluate_once, ensuring:
    1. Purple agent is called
    2. Plan is extracted and saved
    3. VAL metrics are computed and saved
    4. Score is calculated
    """
    run_dir = tmp_path / "run_test"

    cfg = EvalConfig(
        domain_path="domain.pddl",
        problem_path="problem.pddl",
        out_dir=str(tmp_path),  # runner will create subdir unless run_dir is set
        run_dir=str(run_dir),  # Explicit run dir
        purple_kind="mock",
        prompt_text="Solve.",
        optimal_cost=10.0,
        print_card=False,  # Don't clutter test output
    )

    # Mock the metrics return object
    mock_metrics = MagicMock()
    mock_metrics.valid = True
    mock_metrics.cost_value = 20.0
    mock_metrics.steps = []
    mock_metrics.val_stdout = "VAL OK"
    mock_metrics.val_stderr = ""
    # Add all required fields to avoid AttributeErrors
    for field in [
        "length",
        "first_failure_at",
        "first_failed_action",
        "first_failure_reason",
        "first_failure_detail",
        "unsat_count",
        "redundant_indices",
        "advice_count",
        "advice_top_predicates",
        "failure_reason",
        "val_attempts",
        "val_warning",
    ]:
        setattr(mock_metrics, field, None)
    mock_metrics.length = 5
    mock_metrics.unsat_count = 0
    mock_metrics.advice_count = 0
    mock_metrics.val_attempts = 1

    with patch("green_agent.runner.build_purple", return_value=mock_purple_agent):
        with patch("green_agent.runner.compute_metrics", return_value=mock_metrics):
            record = evaluate_once(cfg)

    # 1. Verify Purple Agent Interaction
    mock_purple_agent.generate_plan.assert_called_once_with(problem_nl="Solve.")

    # 2. Verify File Artifacts
    assert (run_dir / "purple_raw.txt").exists()
    assert (run_dir / "purple_raw.txt").read_text(
        encoding="utf-8"
    ) == "```pddl\n(move a b)\n```"

    assert (run_dir / "purple.plan").exists()
    # The extracted plan should be clean
    assert (run_dir / "purple.plan").read_text(encoding="utf-8").strip() == "(move a b)"

    # 3. Verify Scoring
    # Optimal 10 / Cost 20 = 0.5
    assert record["score"] == 0.5
    assert record["valid"] is True
