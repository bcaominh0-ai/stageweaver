from __future__ import annotations

import hashlib
import unittest

from memory.stageweaver_schema import (
    EXEC_STEP,
    PLAN_INIT,
    PLAN_REVISE,
    StageTuple,
    is_role_memory_item,
    retrieval_text,
    require_explicit_executor_memory_jsonl,
    require_positive_executor_cases,
    role_memory_bucket_key,
    serialize_role_conditioned_context,
    serialize_role_conditioned_source,
    stage_memory_retrieval_key,
)
from memory.stageweaver_serializers import render_positive_output_memory


class ExecutorDynamicMemoryRegressionTests(unittest.TestCase):
    def test_executor_retrieval_key_uses_task_description(self) -> None:
        key = stage_memory_retrieval_key(
            {
                "agent_role": "executor",
                "question": "original question",
                "current_state_text": "old task-only state",
                "state_text": "[EXECUTOR_QUERY]\n[LATEST_OBSERVATION]\nnew observation-aware state",
                "metadata": {"raw_task_description": "Search for Ada Lovelace birthplace."},
            }
        )
        self.assertEqual(key, "Search for Ada Lovelace birthplace.")
        self.assertNotIn("new observation-aware state", key)
        self.assertNotIn("old task-only state", key)

    def test_memento_source_text_contains_current_state(self) -> None:
        source_text = serialize_role_conditioned_source(
            role="executor",
            stage="EXEC_STEP",
            current_state="Search for Ada Lovelace birthplace.",
            retrieved_cases_text="[SUCCESS_CASE_1]\n...",
        )
        self.assertIn("[CURRENT_STATE]", source_text)
        self.assertIn("[STAGE] EXEC_STEP", source_text)
        self.assertIn("[RETRIEVED_ROLE_MEMORY]", source_text)

    def test_per_step_without_executor_cases_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires positive executor trajectory cases"):
            require_positive_executor_cases([], "empty-stage-bank.jsonl")

    def test_memento_per_step_requires_explicit_executor_memory_jsonl(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires explicit --executor_memory_jsonl"):
            require_explicit_executor_memory_jsonl("memento_text", "per_step", "")

    def test_executor_role_conditioned_context_uses_short_retrieval_text(self) -> None:
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
            metadata={"raw_task_description": "task-only state"},
        )
        context = serialize_role_conditioned_context(item, retrieved_cases_text="[SUCCESS_CASE_1]\n...")
        self.assertIn("task-only state", context)
        self.assertNotIn("materialized state", context)
        self.assertEqual(retrieval_text(item), "task-only state")

    def test_planner_revise_retrieval_uses_question(self) -> None:
        item = StageTuple(
            stage=PLAN_REVISE,
            episode_id="ep",
            query_id="q",
            cycle_id=1,
            task_id="planner-cycle-1",
            state_text="previous planner output",
            target_text="revised plan",
            reward=1,
            dataset="unit",
            split="train",
            source_id="src-plan",
            question_text="original question",
            agent_role="planner",
            current_state_text="previous planner output",
        )
        self.assertEqual(retrieval_text(item), "original question")

    def test_legacy_positive_row_defaults_to_success_case(self) -> None:
        item = StageTuple(
            stage=PLAN_INIT,
            episode_id="ep",
            query_id="q",
            cycle_id=0,
            task_id="planner-cycle-0",
            state_text="original question",
            target_text="plan",
            reward=1,
            dataset="unit",
            split="train",
            source_id="legacy-positive",
            question_text="original question",
            agent_role="planner",
        )
        self.assertTrue(is_role_memory_item(item))

    def test_planner_stage_buckets_are_separate(self) -> None:
        init = StageTuple(
            stage=PLAN_INIT,
            episode_id="ep",
            query_id="q",
            cycle_id=0,
            task_id="planner-cycle-0",
            state_text="question",
            target_text="plan",
            reward=1,
            dataset="unit",
            split="train",
            source_id="init",
            question_text="question",
            agent_role="planner",
        )
        revise = StageTuple(
            stage=PLAN_REVISE,
            episode_id="ep",
            query_id="q",
            cycle_id=1,
            task_id="planner-cycle-1",
            state_text="question",
            target_text="revised plan",
            reward=1,
            dataset="unit",
            split="train",
            source_id="revise",
            question_text="question",
            agent_role="planner",
        )
        self.assertEqual(role_memory_bucket_key(init), (PLAN_INIT, "planner"))
        self.assertEqual(role_memory_bucket_key(revise), (PLAN_REVISE, "planner"))

    def test_source_hash_changes_when_current_state_changes(self) -> None:
        retrieved = "[SUCCESS_CASE_1]\nconstant case"
        source_a = serialize_role_conditioned_source(
            role="executor",
            stage="EXEC_STEP",
            current_state="same task\nobservation A",
            retrieved_cases_text=retrieved,
        )
        source_b = serialize_role_conditioned_source(
            role="executor",
            stage="EXEC_STEP",
            current_state="same task\nobservation B",
            retrieved_cases_text=retrieved,
        )
        self.assertNotEqual(
            hashlib.sha256(source_a.encode("utf-8")).hexdigest(),
            hashlib.sha256(source_b.encode("utf-8")).hexdigest(),
        )

    def test_role_memory_renders_executor_case_and_insight(self) -> None:
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
                },
                {
                    "stage": PLAN_INIT,
                    "agent_role": "planner",
                    "reward": 0,
                    "target_text": "Start multi-hop questions by identifying bridge entities.",
                    "metadata": {"memory_type": "insight", "origin": "failure"},
                },
            ],
            budget_tokens=256,
        )["text"]
        self.assertIn("[SUCCESS_CASE_1]", rendered)
        self.assertIn("[EXECUTOR_CASE]", rendered)
        self.assertIn("[INSIGHT_1]", rendered)
        self.assertIn("bridge entities", rendered)
        self.assertNotIn("[POSITIVE_OUTPUT_1]\n[EXECUTOR_CASE]", rendered)
        self.assertNotIn("[OUTPUT] [EXECUTOR_CASE]", rendered)


if __name__ == "__main__":
    unittest.main()
