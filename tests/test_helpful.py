# tests/test_helpful.py
from helpful.optimal_plans import _extract_plan_from_stdout


def test_extract_ff_standard_output():
    """Test parsing standard FF output format."""
    stdout = """
    ff: found legal plan as follows
    step    0: PICK-UP B
            1: STACK B A
    """
    plan = _extract_plan_from_stdout(stdout)
    expected = "(pick-up b)\n(stack b a)\n"
    assert plan.strip() == expected.strip()


def test_extract_ff_complex_args():
    """Test FF parsing with multiple arguments."""
    stdout = """
    step 0: MOVE-BLOCK A B C
    """
    plan = _extract_plan_from_stdout(stdout)
    assert plan.strip() == "(move-block a b c)"


def test_extract_fallback_to_standard_extractor():
    """
    If FF output looks like VAL/Standard PDDL (parentheses),
    it should use the standard extractor instead of the 'step' parser.
    """
    stdout = """
    Here is the plan:
    (pick-up a)
    (drop a)
    """
    plan = _extract_plan_from_stdout(stdout)
    assert "(pick-up a)" in plan
    assert "(drop a)" in plan


def test_extract_failure_empty():
    stdout = "No plan found."
    plan = _extract_plan_from_stdout(stdout)
    assert plan == ""
