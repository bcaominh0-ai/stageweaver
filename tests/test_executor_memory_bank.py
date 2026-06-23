from __future__ import annotations

import unittest
from dataclasses import replace

from memory.build_stageweaver_bank import _failure_trace_prompt, build_tuples_from_traces, distill_failure_insight_tuples
from memory.stageweaver_schema import EXEC_STEP, PLAN_REVISE, retrieval_text
from memory.train_stageweaver_composer_sft import retrieve_role_memory_neighbors


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
        planner = [item for item in tuples_ if item.agent_role == "planner"][0]
        trajectory = executor.metadata["executor_trajectory"]
        serialized_planner = planner.to_dict(compact=True)
        serialized_executor = executor.to_dict(compact=True)

        self.assertEqual(planner.state_text, "Where was Ada Lovelace born?")
        self.assertEqual(planner.current_state_text, "Where was Ada Lovelace born?")
        self.assertEqual(planner.metadata["memory_type"], "success_case")
        self.assertEqual(planner.metadata["origin"], "success")
        self.assertEqual(planner.metadata["source_ids"], [planner.source_id])
        self.assertEqual(executor.state_text, "Search for Ada Lovelace birthplace.")
        self.assertEqual(executor.current_state_text, executor.state_text)
        self.assertEqual(executor.metadata["raw_task_description"], "Search for Ada Lovelace birthplace.")
        self.assertEqual(executor.metadata["memory_type"], "success_case")
        self.assertEqual(executor.metadata["origin"], "success")
        self.assertEqual(executor.metadata["source_ids"], [executor.source_id])
        self.assertIn("[EXECUTOR_CASE]", executor.metadata["executor_memory_text"])
        self.assertEqual(trajectory["task_description"], "Search for Ada Lovelace birthplace.")
        self.assertEqual(trajectory["final_output"], "London")
        self.assertFalse(trajectory["legacy_format"])
        self.assertEqual(trajectory["tool_calls"][0]["tool_name"], "search")
        self.assertIn("London", trajectory["tool_calls"][0]["observation_summary"])
        self.assertNotIn("available_tools", serialized_planner)
        self.assertNotIn("available_tools", serialized_executor)
        for key in ("episode_id", "query_id", "cycle_id", "task_id", "reward", "dataset", "split", "source_id", "retrieved_ids"):
            self.assertNotIn(key, serialized_planner)
            self.assertNotIn(key, serialized_executor)
        self.assertEqual(serialized_planner["metadata"], {"memory_type": "success_case", "trace_id": "trace-1"})
        self.assertEqual(serialized_executor["metadata"]["memory_type"], "success_case")
        self.assertIn("executor_memory_text", serialized_executor["metadata"])
        self.assertNotIn("source_type", serialized_planner["metadata"])
        self.assertNotIn("source_ids", serialized_planner["metadata"])

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

    def test_results_match_sample_index_and_retry_order(self) -> None:
        def trace(sample_index: int, question: str, suffix: str) -> dict:
            return {
                "question": question,
                "task_id": f"none-{sample_index}-{suffix}",
                "cycles": [{"planner_output": '{"plan":[{"id":1,"description":"Search."}]}', "tasks": []}],
            }

        traces = [
            trace(7, "Repeated question", "first"),
            trace(9, "Swapped question", "only"),
            trace(7, "Repeated question", "second"),
        ]
        results = [
            {"index": 9, "question": "Swapped question", "correct": True, "data_source": "unit"},
            {"index": 7, "question": "Repeated question", "correct": True, "data_source": "unit"},
            {"index": 7, "question": "Repeated question", "correct": False, "data_source": "unit"},
        ]

        tuples_ = build_tuples_from_traces(traces, result_rows=results, split_name="train")

        self.assertEqual([item.reward for item in tuples_], [1, 1, 0])
        self.assertEqual([item.dataset for item in tuples_], ["unit", "unit", "unit"])

    def test_retrieval_excludes_every_tuple_from_current_trace(self) -> None:
        traces = [
            {
                "question": "Current question",
                "task_id": "none-1-current",
                "cycles": [{"planner_output": '{"plan":[{"id":1,"description":"Search."}]}', "tasks": []}],
            },
            {
                "question": "Other question",
                "task_id": "none-2-other",
                "cycles": [{"planner_output": '{"plan":[{"id":1,"description":"Verify."}]}', "tasks": []}],
            },
        ]
        results = [
            {"index": 1, "question": "Current question", "correct": True},
            {"index": 2, "question": "Other question", "correct": True},
        ]
        current, other = build_tuples_from_traces(traces, result_rows=results, split_name="train")
        same_trace = replace(current, source_id="different-source-id")

        class FakeRetriever:
            def retrieve(self, _query: str, top_k: int) -> list[dict]:
                items = [same_trace.to_dict(), other.to_dict()]
                return [{"item": item} for item in items[:top_k]]

        neighbors = retrieve_role_memory_neighbors(current, FakeRetriever(), matched_k=2, oversample_k=2)

        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0]["metadata"]["trace_id"], "none-2-other")

    def test_failure_trace_distillation_creates_stage_attributed_insights(self) -> None:
        traces = [
            {
                "question": "Which city links Ada Lovelace and the Analytical Engine?",
                "task_id": "failed-trace-1",
                "cycles": [
                    {
                        "planner_output": '{"plan":[{"id":1,"description":"Search Ada Lovelace Analytical Engine relation."}]}',
                        "tasks": [
                            {
                                "task": {"id": 1, "description": "Search Ada Lovelace Analytical Engine relation."},
                                "tool_calls": [{"resolved_name": "search", "arguments": {"query": "Ada Engine"}, "result_preview": "Ambiguous results"}],
                                "result": "No clear evidence",
                            }
                        ],
                    },
                    {
                        "planner_output": '{"plan":[{"id":1,"description":"Verify the bridge entity before final answer."}]}',
                        "tasks": [],
                    },
                ],
            }
        ]
        results = [{"question": "Which city links Ada Lovelace and the Analytical Engine?", "correct": False, "data_source": "unit"}]

        insights = distill_failure_insight_tuples(
            traces,
            result_rows=results,
            split_name="train",
            llm_fn=lambda _prompt: """
            {
              "insights": [
                {
                  "agent_role": "planner",
                  "stage": "PLAN_REVISE",
                  "current_state": "should be ignored for planner",
                  "insight": "Do not revise to final before verifying the bridge entity."
                },
                {
                  "agent_role": "executor",
                  "stage": "EXEC_STEP",
                  "current_state": "Search Ada Lovelace Analytical Engine relation.",
                  "insight": "Avoid broad ambiguous searches; add the suspected relation and verify evidence."
                },
                {
                  "agent_role": "planner",
                  "stage": "EXEC_STEP",
                  "insight": "Invalid role-stage pair should be skipped."
                }
              ]
            }
            """,
        )

        self.assertEqual(len(insights), 2)
        planner = [item for item in insights if item.agent_role == "planner"][0]
        executor = [item for item in insights if item.agent_role == "executor"][0]
        self.assertEqual(planner.stage, PLAN_REVISE)
        self.assertEqual(planner.state_text, "Which city links Ada Lovelace and the Analytical Engine?")
        self.assertEqual(planner.metadata["memory_type"], "insight")
        self.assertEqual(planner.metadata["origin"], "failure")
        self.assertEqual(planner.metadata["source_ids"], ["failed-trace-1"])
        self.assertEqual(executor.stage, EXEC_STEP)
        self.assertEqual(retrieval_text(executor), "Search Ada Lovelace Analytical Engine relation.")
        self.assertIn("ambiguous searches", executor.target_text)

    def test_failure_insight_prompt_guides_error_attribution(self) -> None:
        prompt = _failure_trace_prompt(
            {
                "question": "Where should evidence be verified?",
                "task_id": "failed-trace",
                "cycles": [
                    {
                        "planner_output": '{"plan":[{"id":1,"description":"Search broad evidence."}]}',
                        "tasks": [
                            {
                                "task": {"id": 1, "description": "Search broad evidence."},
                                "tool_calls": [],
                                "result": "No evidence",
                            }
                        ],
                    }
                ],
            },
            "failed-trace",
        )
        self.assertIn("decisive failure was introduced", prompt)
        self.assertIn("earliest stage where changing behavior", prompt)
        self.assertIn("what a successful trajectory would have done differently", prompt)
        self.assertIn("causal difference between effective and ineffective behavior", prompt)
        self.assertIn("Say what to avoid and what to do instead", prompt)


if __name__ == "__main__":
    unittest.main()
