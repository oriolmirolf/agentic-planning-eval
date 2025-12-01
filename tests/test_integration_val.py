# tests/test_integration_val.py
import os
import shutil
import textwrap

import pytest
from dotenv import load_dotenv

from green_agent.val_wrapper import run_val

pytestmark = pytest.mark.integration

load_dotenv()


def resolve_val_binary():
    env_path = os.getenv("VAL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path
    return shutil.which("validate") or shutil.which("Validate")


VAL_BINARY = resolve_val_binary()


@pytest.fixture
def blocks_domain_text():
    return """(define (domain blocksworld)
  (:requirements :strips :typing)
  (:types block)
  (:predicates (on ?x - block ?y - block)
               (ontable ?x - block)
               (clear ?x - block)
               (handempty)
               (holding ?x - block)
  )
  (:action pick-up
             :parameters (?x - block)
             :precondition (and (clear ?x) (ontable ?x) (handempty))
             :effect (and (not (ontable ?x))
                          (not (clear ?x))
                          (not (handempty))
                          (holding ?x)))
  (:action put-down
             :parameters (?x - block)
             :precondition (holding ?x)
             :effect (and (not (holding ?x))
                          (clear ?x)
                          (handempty)
                          (ontable ?x)))
  (:action stack
             :parameters (?x - block ?y - block)
             :precondition (and (holding ?x) (clear ?y))
             :effect (and (not (holding ?x))
                          (not (clear ?y))
                          (clear ?x)
                          (handempty)
                          (on ?x ?y)))
  (:action unstack
             :parameters (?x - block ?y - block)
             :precondition (and (on ?x ?y) (clear ?x) (handempty))
             :effect (and (holding ?x)
                          (clear ?y)
                          (not (clear ?x))
                          (not (handempty))
                          (not (on ?x ?y))))
)"""


@pytest.fixture
def blocks_problem_text():
    return """(define (problem invert-stack)
  (:domain blocksworld)
  (:objects a b - block)
  (:init (on a b)
         (ontable b)
         (clear a)
         (handempty))
  (:goal (and (on b a) (ontable a)))
)"""


@pytest.fixture
def correct_plan_text():
    return textwrap.dedent("""
    (unstack a b)
    (put-down a)
    (pick-up b)
    (stack b a)
    """).strip()


@pytest.fixture
def incorrect_plan_text():
    return textwrap.dedent("""
    (unstack a b)
    (pick-up b)
    (stack b a)
    """).strip()


@pytest.fixture
def incomplete_plan_text():
    return textwrap.dedent("""
    (unstack a b)
    (put-down a)
    """).strip()


@pytest.fixture
def garbage_plan_text():
    return "This is not pddl and should cause a syntax error."


@pytest.fixture
def pddl_files(tmp_path, blocks_domain_text, blocks_problem_text):
    d_path = tmp_path / "domain.pddl"
    p_path = tmp_path / "problem.pddl"
    d_path.write_text(blocks_domain_text, encoding="utf-8")
    p_path.write_text(blocks_problem_text, encoding="utf-8")
    return str(d_path), str(p_path)


@pytest.mark.skipif(VAL_BINARY is None, reason="VAL binary not found")
def test_real_val_correct_plan(pddl_files, correct_plan_text):
    """Test a plan that fully satisfies the goal."""
    domain_path, problem_path = pddl_files
    result = run_val(domain_path, problem_path, correct_plan_text, val_path=VAL_BINARY)
    assert result.ok is True, f"Valid plan marked invalid.\n{result.stderr}"
    assert len(result.steps) == 4


@pytest.mark.skipif(VAL_BINARY is None, reason="VAL binary not found")
def test_real_val_incorrect_plan(pddl_files, incorrect_plan_text):
    """Test a plan that fails a precondition midway."""
    domain_path, problem_path = pddl_files
    result = run_val(
        domain_path, problem_path, incorrect_plan_text, val_path=VAL_BINARY
    )

    assert result.ok is False
    assert result.failure_reason == "precondition_unsatisfied"
    assert len(result.unsatisfied) > 0
    assert "pick-up" in result.unsatisfied[0].detail.lower()


@pytest.mark.skipif(VAL_BINARY is None, reason="VAL binary not found")
def test_real_val_goal_not_satisfied(pddl_files, incomplete_plan_text):
    """Test a plan that executes successfully but fails the goal state."""
    domain_path, problem_path = pddl_files
    result = run_val(
        domain_path, problem_path, incomplete_plan_text, val_path=VAL_BINARY
    )

    assert result.ok is False
    assert result.failure_reason == "goal_not_satisfied"
    assert len(result.steps) == 2
    assert len(result.unsatisfied) == 0


@pytest.mark.skipif(VAL_BINARY is None, reason="VAL binary not found")
def test_real_val_syntax_error(pddl_files, garbage_plan_text):
    """Test that garbage input is caught as a syntax error or invalid."""
    domain_path, problem_path = pddl_files
    result = run_val(domain_path, problem_path, garbage_plan_text, val_path=VAL_BINARY)

    assert result.ok is False
    assert result.failure_reason in [
        "syntax_error_or_invalid",
        "unknown_failure",
        "goal_not_satisfied",
    ]


@pytest.mark.skipif(VAL_BINARY is None, reason="VAL binary not found")
def test_real_val_trace_parsing(pddl_files, correct_plan_text):
    """
    Detailed check: Ensure we are correctly parsing the 'Adding' and 'Deleting'
    lines from VAL's verbose output.
    """
    domain_path, problem_path = pddl_files
    result = run_val(domain_path, problem_path, correct_plan_text, val_path=VAL_BINARY)

    assert result.ok is True

    step1 = result.steps[0]
    assert step1.action == "(unstack a b)"

    adds_flat = [a.lower() for a in step1.adds]
    assert "(holding a)" in adds_flat or "(holding a)" in adds_flat
    assert "(clear b)" in adds_flat or "(clear b)" in adds_flat

    dels_flat = [d.lower() for d in step1.deletes]
    assert "(on a b)" in dels_flat or "(on a b)" in dels_flat
    assert "(handempty)" in dels_flat or "(handempty)" in dels_flat
