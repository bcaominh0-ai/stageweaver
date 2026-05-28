from __future__ import annotations

import hashlib
import unittest

from memory.stageweaver_schema import (
    EXEC_STEP,
    StageTuple,
    retrieval_text,
    require_explicit_executor_memory_jsonl,
    require_positive_executor_cases,
    serialize_role_conditioned_context,
    serialize_role_conditioned_source,
    stage_memory_retrieval_key,
)
from memory.stageweaver_serializers import render_positive_output_memory


class ExecutorDynamicMemoryRegressionTests(unittest.TestCase):
    def test_executor_retrieval_key_prefers_state_text(self) -> None:
        key = stage_memory_retrieval_key(
            {
                "agent_role": "executor",
                "question": "original question",
                "current_state_text": "old task-only state",
                "state_text": "[EXECUTOR_QUERY]\n[LATEST_OBSERVATION]\nnew observation-aware state",
            }
        )
        self.assertIn("new observation-aware state", key)
        self.assertNotIn("old task-only state", key)

    def test_memento_source_text_contains_current_state(self) -> None:
        source_text = serialize_role_conditioned_source(
            role="executor",
            current_state="[EXECUTOR_QUERY]\n[LATEST_OBSERVATION]\nobservation",
            retrieved_cases_text="[POSITIVE_EXECUTOR_CASE_1]\n...",
        )
        self.assertIn("[CURRENT_STATE]", source_text)
        self.assertIn("[LATEST_OBSERVATION]", source_text)
        self.assertIn("[RETRIEVED_POSITIVE_CASES]", source_text)

    def test_per_step_without_executor_cases_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires positive executor trajectory cases"):
            require_positive_executor_cases([], "empty-stage-bank.jsonl")

    def test_memento_per_step_requires_explicit_executor_memory_jsonl(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires explicit --executor_memory_jsonl"):
            require_explicit_executor_memory_jsonl("memento_text", "per_step", "")

    def test_executor_role_conditioned_context_uses_retrieval_text(self) -> None:
        item = StageTuple(
            stage=EXEC_STEP,
            episode_id="ep",
            query_id="q",
            cycle_id=0,
            task_id="task",
            state_text="[EXECUTOR_QUERY]\n[EXECUTOR_TASK]\nmaterialized state",
            target_text="[TOOL_CALL] search({})",
            reward=1,
            dataset="unit",
            split="train",
            source_id="src",
            agent_role="executor",
            current_state_text="task-only state",
        )
        context = serialize_role_conditioned_context(item, retrieved_cases_text="[POSITIVE_EXECUTOR_CASE_1]\n...")
        self.assertIn("materialized state", context)
        self.assertNotIn("task-only state", context)
        self.assertEqual(retrieval_text(item), item.state_text)

    def test_source_hash_changes_when_current_state_changes(self) -> None:
        retrieved = "[POSITIVE_EXECUTOR_CASE_1]\nconstant case"
        source_a = serialize_role_conditioned_source(
            role="executor",
            current_state="same task\nobservation A",
            retrieved_cases_text=retrieved,
        )
        source_b = serialize_role_conditioned_source(
            role="executor",
            current_state="same task\nobservation B",
            retrieved_cases_text=retrieved,
        )
        self.assertNotEqual(
            hashlib.sha256(source_a.encode("utf-8")).hexdigest(),
            hashlib.sha256(source_b.encode("utf-8")).hexdigest(),
        )

    def test_executor_case_not_wrapped_as_positive_output(self) -> None:
        rendered = render_positive_output_memory(
            [
                {
                    "stage": "EXEC_STEP",
                    "agent_role": "executor",
                    "reward": 1,
                    "metadata": {
                        "executor_trajectory": {
                            "task_description": "Search for Ada Lovelace birthplace.",
                            "success_signal": "success",
                            "tool_calls": [],
                            "final_output": "London",
                        }
                    },
                }
            ],
            budget_tokens=256,
        )["text"]
        self.assertIn("[POSITIVE_EXECUTOR_CASE_1]", rendered)
        self.assertIn("[EXECUTOR_CASE]", rendered)
        self.assertNotIn("[POSITIVE_OUTPUT_1]\n[EXECUTOR_CASE]", rendered)
        self.assertNotIn("[OUTPUT] [EXECUTOR_CASE]", rendered)


if __name__ == "__main__":
    unittest.main()
