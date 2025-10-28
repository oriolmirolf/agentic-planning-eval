from green_agent.plan_parser import extract_plan

def test_extract():
    raw = """Here is your plan:
    ```
    1: (pick-up a)
    2: (stack a b)
    ; cost = 10
    ```
    """
    ep = extract_plan(raw)
    assert len(ep.steps) == 2
    assert ep.steps[0].text == "(pick-up a)"
    assert ep.raw_cost == 10.0
