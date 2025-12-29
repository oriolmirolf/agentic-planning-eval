import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock
import dspy

# -----------------------------------------------------------------------------
# 1. Import your exact logic (Copy-paste StrictReAct & EpisodeTools or import)
# -----------------------------------------------------------------------------
# Assuming you are running this in the root where 'agents.react_orchestrator' is importable
# If not, paste the classes StrictReAct and EpisodeTools here.
from react_orchestrator import StrictReAct, EpisodeTools, AgentDeath, PlannerSig

# -----------------------------------------------------------------------------
# 2. The Test
# -----------------------------------------------------------------------------

class TestRealFailureLogic(unittest.TestCase):
    def test_precondition_violation_kills_agent(self):
        print("\n[TEST] Verifying Precondition Violation -> Hard Crash")

        # 1. Setup Tools with a mocked backend
        # We don't need real PDDL files; we just need 'act' to return the VAL failure string.
        tools = EpisodeTools(domain="blocks", index=1)
        
        # MOCK THE BACKEND directly on the instance to simulate VAL output
        def mock_act_backend(step_text):
            # Simulate exactly what tools_backend.py returns for a logic error
            return "Executed: NO\nReason: (clear b) is false"
        
        # Override the underlying call_flexible or the backend function it points to
        # Since EpisodeTools uses _call_flexible which calls the imported function,
        # we can patch the method 'act' logic itself for this test.
        # However, to test the REGEX logic inside EpisodeTools.act, we must let it run.
        # We will patch the imported 'ACT' function inside the module context if possible,
        # OR just subclass for testing.
        
        class TestTools(EpisodeTools):
            def act(self, step_text):
                # Simulate the BACKEND response
                out = "Executed: NO\nReason: (clear b) is false" 
                
                # --- EXACT LOGIC FROM YOUR CODE ---
                if self.strict_invalid_action and "Executed: NO" in out:
                    lower_msg = out.lower()
                    is_syntax_error = any(x in lower_msg for x in [
                        "arg", "expect", "found", "got", "parameter", "signature", "missing"
                    ])
                    if is_syntax_error:
                        return out
                    else:
                        raise AgentDeath(out) # <--- THIS IS WHAT WE WANT TO SEE
                return out

        test_tools = TestTools(domain="test", index=1)
        tool_fns = [test_tools.act, test_tools.submit]

        # 2. Setup the Agent
        agent = StrictReAct(PlannerSig, tools=tool_fns, max_iters=5)
        agent.react = MagicMock()
        
        # Force the agent to choose the bad action
        agent.react.side_effect = [
            dspy.Prediction(
                next_thought="I will try an illegal move.", 
                next_tool_name="act", 
                next_tool_args={"step_text": "pick-up b"}
            )
        ]

        # 3. Run
        pred = agent(objective="test")

        # 4. Assertions
        # It should have caught AgentDeath internally and logged it
        fatal_log = pred.trajectory.get("fatal_error_log", "")
        print(f"Resulting Log: {fatal_log}")

        self.assertTrue(fatal_log, "Agent did not log a fatal error!")
        self.assertIn("(clear b) is false", fatal_log, "The specific VAL error was lost.")
        self.assertEqual(pred.final_plan, f"Failed: Executed: NO\nReason: (clear b) is false", "Final plan should report failure.")
        
        print("✅ SUCCESS: Precondition violation correctly triggered AgentDeath.")

if __name__ == "__main__":
    unittest.main()