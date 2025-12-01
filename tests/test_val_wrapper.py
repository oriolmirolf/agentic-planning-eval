# tests/test_val_wrapper.py
from unittest.mock import MagicMock, patch

import pytest

from green_agent.val_wrapper import run_val


# We mock shutil.which so the code thinks 'validate' is installed
@pytest.fixture(autouse=True)
def mock_val_binary():
    with (
        patch("shutil.which", return_value="/fake/path/to/validate"),
        patch("os.path.exists", return_value=True),
    ):
        yield


def test_run_val_success(val_stdout_success):
    with patch("subprocess.run") as mock_run:
        # Configure the mock to return success
        mock_run.return_value = MagicMock(
            returncode=0, stdout=val_stdout_success, stderr=""
        )

        result = run_val("domain.pddl", "problem.pddl", "(action)")

        assert result.ok is True
        assert result.value == 4.0
        assert len(result.plan_actions) == 2
        assert result.plan_actions[1] == "(pick-up b1)"


def test_run_val_failure(val_stdout_failure):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=val_stdout_failure,
            stderr="",
        )

        result = run_val("domain.pddl", "problem.pddl", "(action)")

        assert result.ok is False
        assert result.failure_reason == "precondition_unsatisfied"
        assert len(result.unsatisfied) > 0


def test_run_val_retry_logic():
    """Test that wrapper retries if output is empty."""
    with patch("subprocess.run") as mock_run:
        # First 2 calls return empty, 3rd returns success
        mock_run.side_effect = [
            MagicMock(stdout="", stderr=""),
            MagicMock(stdout="", stderr=""),
            MagicMock(stdout="Plan valid", stderr=""),
        ]

        # Speed up retry backoff for test
        with patch("time.sleep"):
            result = run_val("d", "p", "plan", retries=3)

        assert result.ok is True
        assert mock_run.call_count == 3
        assert result.attempts == 3
