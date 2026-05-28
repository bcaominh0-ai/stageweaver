from __future__ import annotations

import unittest

from memory.stageweaver_serializers import serialize_executor_trajectory_case


class ExecutorTrajectorySerializerTests(unittest.TestCase):
    def test_serializes_trajectory_rich_executor_case(self) -> None:
        text = serialize_executor_trajectory_case(
            {
                "stage": "EXEC_STEP",
                "agent_role": "executor",
                "reward": 1,
                "metadata": {
                    "executor_trajectory": {
                        "task_description": "Search for Ada Lovelace birthplace.",
                        "success_signal": "success",
                        "tool_calls": [
                            {
                                "tool_name": "search",
                                "arguments": {"query": "Ada Lovelace birthplace"},
                                "observation_summary": "The results state Ada Lovelace was born in London.",
                                "decision_rationale": "The result directly answers the subtask.",
                            }
                        ],
                        "final_output": "London",
                    }
                },
            }
        )

        self.assertIn("[EXECUTOR_CASE]", text)
        self.assertIn("[TOOL_TRACE]", text)
        self.assertIn("[TOOL_CALL]", text)
        self.assertIn("[OBSERVATION]", text)
        self.assertIn("[DECISION]", text)
        self.assertIn("[OUTPUT]", text)
        self.assertIn("London", text)

    def test_serializes_legacy_output_only_executor_case(self) -> None:
        text = serialize_executor_trajectory_case(
            {
                "stage": "EXEC_STEP",
                "agent_role": "executor",
                "state_text": "Find Ada Lovelace birthplace.",
                "target_text": "[RETURN] London",
            }
        )

        self.assertIn("[FORMAT_NOTE]", text)
        self.assertIn("Legacy executor memory", text)
        self.assertIn("[OUTPUT]", text)


if __name__ == "__main__":
    unittest.main()
