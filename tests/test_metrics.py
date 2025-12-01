# tests/test_metrics.py
from unittest.mock import patch

from green_agent.metrics import compute_metrics
from green_agent.val_wrapper import ValResult

# ... (Previous regex tests remain) ...


def test_redundancy_checking_logic():
    """
    Test that compute_metrics correctly identifies redundant steps
    by analyzing the results of iterative run_val calls.
    """
    # Scenario: Plan has 3 steps.
    # 1. Full plan -> Valid
    # 2. Remove step 1 -> Invalid
    # 3. Remove step 2 -> Valid (This means step 2 was redundant!)
    # 4. Remove step 3 -> Invalid

    # We mock run_val. It will be called 4 times.
    # 1st call: The initial validation
    # 2nd, 3rd, 4th calls: The redundancy check loop (once per action)

    mock_results = [
        # 1. Initial Validation
        ValResult(ok=True, stdout="", stderr="", steps=[], value=10),
        # 2. Check removal of step 1
        ValResult(ok=False, stdout="", stderr="", steps=[]),
        # 3. Check removal of step 2 (SUCCESS -> Redundant)
        ValResult(ok=True, stdout="", stderr="", steps=[]),
        # 4. Check removal of step 3
        ValResult(ok=False, stdout="", stderr="", steps=[]),
    ]

    with patch("green_agent.metrics.run_val", side_effect=mock_results) as mock_run:
        metrics = compute_metrics(
            domain="d", problem="p", plan_text="(s1)\n(s2)\n(s3)", check_redundancy=True
        )

        assert metrics.valid is True
        # Step 2 (index 1 in 0-based list) should be marked.
        # Your code returns 1-based indices, so it should be [2].
        assert metrics.redundant_indices == [2]

        # Verify it was called 4 times
        assert mock_run.call_count == 4
