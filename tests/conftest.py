# tests/conftest.py
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "local_llm: mark test as requiring a local LLM server running."
    )


@pytest.fixture
def val_stdout_success():
    """Simulated STDOUT from VAL for a successful plan."""
    return """
    Checking plan: /tmp/tmpxyz.plan
    Plan size: 2
    1: (pick-up b1)
    2: (stack b1 b2)

    Plan valid
    Final value: 4.0
    """


@pytest.fixture
def val_stdout_failure():
    """Simulated STDOUT from VAL for a failed plan with advice."""
    return """
    Checking plan: /tmp/tmpxyz.plan
    Plan size: 2
    1: (pick-up b1)
    2: (stack b1 b2)

    Plan failed because of unsatisfied precondition

    Plan Repair Advice:
    (pick-up b1) has an unsatisfied precondition at time 1
    Set (hand-empty) to true
    Set (on-table b1) to true
    """
