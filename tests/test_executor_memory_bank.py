from __future__ import annotations

import unittest

from memory.build_stageweaver_bank import build_tuples_from_traces


class ExecutorMemoryBankTests(unittest.TestCase):
    def test_executor_tuple_contains_trajectory_metadata(self) -> None:
        traces = [
            {
                "question": "Where was Ada Lovelace born?",
                "task_id": "trace-1",
                "connected_tools": ["search"],
                "cycles": [
                    {
                        "planner_output": '{"plan":[{"id":1,"description":"Search for Ada Lovelace birthplace."}]}',
                        "tasks": [
                            {
                                "task": {"id": 1, "description": "Search for Ada Lovelace birthplace."},
                                "tool_calls": [
                                    {
                                        "requested_name": "search",
                                        "resolved_name": "search",
                                        "arguments": {"query": "Ada Lovelace birthplace"},
                                        "result_preview": "Ada Lovelace was born in London.",
                                    }
                                ],
                                "result": "London",
                            }
                        ],
                    }
                ],
            }
        ]
        results = [{"question": "Where was Ada Lovelace born?", "correct": True, "data_source": "unit"}]

        tuples_ = build_tuples_from_traces(traces, result_rows=results, split_name="train")
        executor = [item for item in tuples_ if item.agent_role == "executor"][0]
        trajectory = executor.metadata["executor_trajectory"]

        self.assertTrue(executor.state_text.startswith("[EXECUTOR_QUERY]"))
        self.assertEqual(executor.current_state_text, executor.state_text)
        self.assertEqual(executor.metadata["raw_task_description"], "Search for Ada Lovelace birthplace.")
        self.assertIn("[EXECUTOR_CASE]", executor.metadata["executor_memory_text"])
        self.assertEqual(trajectory["task_description"], "Search for Ada Lovelace birthplace.")
        self.assertEqual(trajectory["final_output"], "London")
        self.assertFalse(trajectory["legacy_format"])
        self.assertEqual(trajectory["tool_calls"][0]["tool_name"], "search")
        self.assertIn("London", trajectory["tool_calls"][0]["observation_summary"])

    def test_missing_correct_does_not_default_to_positive(self) -> None:
        traces = [
            {
                "question": "Where was Ada Lovelace born?",
                "task_id": "trace-unknown",
                "cycles": [
                    {
                        "planner_output": '{"plan":[{"id":1,"description":"Search for Ada Lovelace birthplace."}]}',
                        "tasks": [
                            {
                                "task": {"id": 1, "description": "Search for Ada Lovelace birthplace."},
                                "tool_calls": [],
                                "result": "London",
                            }
                        ],
                    }
                ],
            }
        ]
        tuples_ = build_tuples_from_traces(
            traces,
            result_rows=[{"question": "Where was Ada Lovelace born?", "data_source": "unit"}],
            split_name="train",
        )
        executor = [item for item in tuples_ if item.agent_role == "executor"][0]

        self.assertEqual(executor.reward, 0)
        self.assertEqual(executor.metadata["executor_trajectory"]["success_signal"], "unknown")


if __name__ == "__main__":
    unittest.main()
