from __future__ import annotations

import unittest

from memory.stageweaver_schema import build_executor_current_state


class ExecutorStateBuilderTests(unittest.TestCase):
    def test_builds_observation_aware_state_with_truncation(self) -> None:
        state = build_executor_current_state(
            task_description="Find the birthplace of Ada Lovelace.",
            tool_history=[
                {
                    "resolved_name": "search",
                    "arguments": {"query": "Ada Lovelace birthplace"},
                    "result_preview": "Ada Lovelace was born in London." * 100,
                }
            ],
            latest_observation="Ada Lovelace was born in London." * 100,
            failed_calls=[{"error": "first query was too broad"}],
            repeated_calls=[],
            partial_result="London",
            max_chars=900,
            obs_max_chars=120,
            tool_history_k=2,
        )

        self.assertIn("[EXECUTOR_TASK]", state)
        self.assertIn("[CALLED_TOOLS]", state)
        self.assertIn("search", state)
        self.assertIn("[LATEST_OBSERVATION]", state)
        self.assertIn("[FAILURE_OR_AMBIGUITY]", state)
        self.assertIn("[PARTIAL_RESULT]", state)
        self.assertLessEqual(len(state), 900)

    def test_executor_state_does_not_include_need_next_instruction(self) -> None:
        state = build_executor_current_state(
            task_description="Find the nationality of a person.",
            tool_history=[],
            latest_observation="",
            failed_calls=[],
            repeated_calls=[],
            partial_result="",
        )

        self.assertNotIn("[NEED_NEXT]", state)
        self.assertNotIn("search again, crawl, extract, verify, or finish", state)
        self.assertIn("[EXECUTOR_TASK]", state)
        self.assertIn("[CALLED_TOOLS]", state)
        self.assertIn("[LATEST_OBSERVATION]", state)
        self.assertIn("[FAILURE_OR_AMBIGUITY]", state)
        self.assertIn("[PARTIAL_RESULT]", state)


if __name__ == "__main__":
    unittest.main()
