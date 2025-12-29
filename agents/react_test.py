import unittest
import dspy
from dataclasses import dataclass
from unittest.mock import MagicMock

# =============================================================================
# 1. THE AGENT LOGIC (Fixed Order of Operations)
# =============================================================================

class AgentDeath(Exception):
    """Custom exception for immediate agent termination (Domain Failure)."""
    pass

class StrictReAct(dspy.ReAct):
    """
    A subclass of dspy.ReAct that:
    1. Stops immediately on AgentDeath (saving the error in trajectory first).
    2. Stops immediately after a 'submit' action is called.
    """
    def forward(self, **input_args):
        trajectory = {}
        # Support dspy versions where max_iters might be in kwargs or attribute
        max_iters = input_args.pop("max_iters", getattr(self, "max_iters", 5))

        for idx in range(max_iters):
            # 1. Thought & Tool Selection
            try:
                # Mocked internal call
                pred = self._call_with_potential_trajectory_truncation(
                    self.react, trajectory, **input_args
                )
            except ValueError:
                break

            # --- FIX: Record the thought/tool BEFORE checking for break conditions ---
            trajectory[f"thought_{idx}"] = pred.next_thought
            trajectory[f"tool_name_{idx}"] = pred.next_tool_name
            trajectory[f"tool_args_{idx}"] = pred.next_tool_args

            # 2. Check for "Submit" (Hard Stop)
            if pred.next_tool_name.lower() in ["submit", "finish", "submit_episode"]:
                try:
                    tool_fn = self.tools[pred.next_tool_name]
                    args = pred.next_tool_args or {}
                    observation = tool_fn(**args)
                    trajectory[f"observation_{idx}"] = observation
                except Exception as err:
                    trajectory[f"observation_{idx}"] = f"Error submitting: {err}"
                
                # MARKER for verification
                trajectory["stop_reason"] = "submit_break"
                break

            # 3. Standard Tool Execution
            try:
                tool_fn = self.tools[pred.next_tool_name]
                args = pred.next_tool_args or {}
                observation = tool_fn(**args)
                trajectory[f"observation_{idx}"] = observation
            
            except AgentDeath as e:
                # SOFT CRASH: Record and Stop
                trajectory[f"observation_{idx}"] = f"FATAL DOMAIN ERROR: {e}"
                trajectory["fatal_error_log"] = str(e)
                trajectory["stop_reason"] = "fatal_error"
                break
            
            except Exception as err:
                # SYNTAX ERROR: Record and Continue
                trajectory[f"observation_{idx}"] = f"Execution error in {pred.next_tool_name}: {err}"

        return dspy.Prediction(trajectory=trajectory)

class PlannerSig(dspy.Signature):
    """Mock Signature"""
    objective: str = dspy.InputField()
    final_plan: str = dspy.OutputField()

# =============================================================================
# 2. MOCK TOOLS
# =============================================================================

@dataclass
class MockEpisodeTools:
    def act(self, step_text: str) -> str:
        # SIMULATE BACKEND RESPONSES
        output = ""
        if "syntax_error" in step_text:
            output = "Executed: NO\nReason: Action 'stack' expects 2 args, got 1."
        elif "fatal_error" in step_text:
            output = "Executed: NO\nReason: Cannot pick up A, it is covered."
        else:
            output = "Executed: YES"

        # --- THE SMART LOGIC BEING TESTED ---
        if "Executed: NO" in output:
            lower_msg = output.lower()
            # Heuristic for recoverable errors
            is_syntax_error = any(x in lower_msg for x in [
                "arg", "expect", "found", "got", "parameter", "signature", "missing"
            ])
            
            if is_syntax_error:
                return output # Return string -> Agent continues
            else:
                raise AgentDeath(output) # Raise -> Agent stops
                
        return output

    def submit(self) -> str:
        return "Accepted: YES"

    def list_actions(self) -> str: return "act, submit"
    def get_task_overview(self) -> str: return "overview"

# =============================================================================
# 3. THE TESTS (Using MagicMock)
# =============================================================================

class TestStrictReActBehavior(unittest.TestCase):
    
    def setUp(self):
        self.mock_tools = MockEpisodeTools()
        self.tools_list = [
            self.mock_tools.act,
            self.mock_tools.submit,
            self.mock_tools.list_actions,
            self.mock_tools.get_task_overview
        ]
        
        # Initialize Agent with MagicMock for the internal predictor
        self.agent = StrictReAct(PlannerSig, tools=self.tools_list, max_iters=10)
        self.agent.react = MagicMock()

    def test_submit_hard_stop(self):
        """Test that calling 'submit' ends the loop immediately."""
        print("\n[TEST] Submit Hard Stop")
        
        # Scenario: Act -> Submit -> (Should Stop)
        self.agent.react.side_effect = [
            dspy.Prediction(next_thought="Step 1", next_tool_name="act", next_tool_args={"step_text": "move a"}),
            dspy.Prediction(next_thought="Done", next_tool_name="submit", next_tool_args={}),
            dspy.Prediction(next_thought="Bad", next_tool_name="act", next_tool_args={"step_text": "bad"}) # Should NOT be reached
        ]
        
        pred = self.agent(objective="test")
        traj = pred.trajectory

        self.assertEqual(traj.get("stop_reason"), "submit_break")
        self.assertIn("observation_1", traj)
        self.assertIn("thought_1", traj, "Failed to record thought for submit step!") # This was failing before
        self.assertNotIn("thought_2", traj)
        print("✅ PASS: Agent recorded submit step and stopped.")

    def test_fatal_error_soft_crash(self):
        """Test that a domain error (non-syntax) stops the loop."""
        print("\n[TEST] Fatal Error Soft Crash")
        
        # Scenario: Act (Fatal) -> (Should Stop)
        self.agent.react.side_effect = [
            dspy.Prediction(next_thought="Bad move", next_tool_name="act", next_tool_args={"step_text": "fatal_error"}),
            dspy.Prediction(next_thought="Next", next_tool_name="act", next_tool_args={"step_text": "move b"}) # Should NOT be reached
        ]
        
        pred = self.agent(objective="test")
        traj = pred.trajectory

        self.assertEqual(traj.get("stop_reason"), "fatal_error")
        self.assertIn("fatal_error_log", traj)
        self.assertIn("Cannot pick up A", traj["fatal_error_log"])
        self.assertNotIn("thought_1", traj)
        print("✅ PASS: Agent crashed and logged fatal error.")

    def test_syntax_error_recovery(self):
        """Test that a syntax error allows the agent to continue."""
        print("\n[TEST] Syntax Error Recovery")
        
        # Scenario: Act (Syntax Error) -> Submit
        self.agent.react.side_effect = [
            dspy.Prediction(next_thought="Typo", next_tool_name="act", next_tool_args={"step_text": "syntax_error"}),
            dspy.Prediction(next_thought="Fixing", next_tool_name="submit", next_tool_args={})
        ]

        pred = self.agent(objective="test")
        traj = pred.trajectory

        # Step 0: Syntax Error
        self.assertIn("observation_0", traj)
        self.assertIn("expects 2 args", traj["observation_0"])
        
        # Step 1: Submit (Proof it continued)
        self.assertIn("thought_1", traj)
        self.assertEqual(traj.get("stop_reason"), "submit_break")
        print("✅ PASS: Agent recovered from syntax error.")

if __name__ == "__main__":
    unittest.main()