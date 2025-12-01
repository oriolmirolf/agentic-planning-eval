# tests/test_plan_parser.py
from green_agent.plan_parser import extract_plan


def test_extract_clean_plan():
    raw = "(pick-up a)\n(drop a)"
    plan = extract_plan(raw)
    assert len(plan.steps) == 2
    assert plan.steps[0].text == "(pick-up a)"


def test_extract_markdown_block():
    raw = """
    Here is the plan:
    ```pddl
    (move a b)
    ```
    Hope it works.
    """
    plan = extract_plan(raw)
    assert len(plan.steps) == 1
    assert plan.steps[0].text == "(move a b)"


def test_extract_numbered_list_with_comments():
    raw = """
    1. (action-1 arg) ; This is a comment
    2. (action-2 arg)
    """
    plan = extract_plan(raw)
    assert len(plan.steps) == 2
    assert (
        plan.steps[0].text == "(action-1 arg)"
    )  # Comments should be stripped by parser if regex handles it,
    # or at least the parens captured correctly.
    assert plan.steps[1].text == "(action-2 arg)"


def test_case_insensitivity():
    raw = "(Pick-Up A)"
    plan = extract_plan(raw)
    assert plan.steps[0].text == "(Pick-Up A)"


def test_empty_input():
    plan = extract_plan("")
    assert len(plan.steps) == 0
    assert plan.to_val_plan_text() == ""
