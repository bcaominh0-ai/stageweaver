from __future__ import annotations

import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.memento_text_memory import load_memento_planner_cases
from client import stageweaver_runner

ACTIVE_PATHS = [
    ROOT / "client",
    ROOT / "memory",
    ROOT / "server",
    ROOT / "scripts",
]
ACTIVE_DOCS = [
    ROOT / "README.md",
    ROOT / "BUNDLE_INFO.txt",
    ROOT / "docs/CURRENT_PROTOCOL.md",
    ROOT / "docs/CLEANUP_MANIFEST.md",
]
WORKFLOW_DOCS = [
    ROOT / "README.md",
    ROOT / "BUNDLE_INFO.txt",
    ROOT / "docs/CURRENT_PROTOCOL.md",
]
WORKFLOW_SCRIPTS = [
    ROOT / "scripts/collect_seen_train_traces.sh",
    ROOT / "scripts/collect_seen_dev_traces.sh",
    ROOT / "scripts/build_stageweaver_bank.sh",
    ROOT / "scripts/train_append_sft.sh",
    ROOT / "scripts/eval_seen_dev_gate.sh",
    ROOT / "scripts/eval_ood_test.sh",
    ROOT / "scripts/linux_preflight.sh",
]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _active_python_text() -> str:
    parts: list[str] = []
    for base in ACTIVE_PATHS:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "archive" not in path.parts:
                parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _active_doc_text() -> str:
    parts: list[str] = []
    for path in ACTIVE_DOCS:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _workflow_doc_text() -> str:
    parts: list[str] = []
    for path in WORKFLOW_DOCS:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _workflow_script_text() -> str:
    parts: list[str] = []
    for path in WORKFLOW_SCRIPTS:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


class CurrentProtocolTests(unittest.TestCase):
    def test_runner_exposes_only_current_modes(self) -> None:
        runner = _read("client/stageweaver_runner.py")
        self.assertIn('ACTIVE_MEMORY_MODES = {"memento_text", "stageweaver", "none"}', runner)
        for obsolete in ("static_prefix", "generic_prefix", "legacy_prompt_bridge"):
            self.assertNotIn(obsolete, runner)

    def test_stageweaver_requires_both(self) -> None:
        runner = _read("client/stageweaver_runner.py")
        self.assertIn('choices=["both"]', runner)
        self.assertIn("Mainline StageWeaver requires --stage_mode both", runner)

    def test_stageweaver_rejects_non_append_interfaces(self) -> None:
        active = _active_python_text()
        self.assertNotIn("prepend", active)
        runner = _read("client/stageweaver_runner.py")
        self.assertIn('--latent_interface', runner)
        self.assertIn('choices=["append"]', runner)

    def test_default_tool_profile_is_six_tools(self) -> None:
        runner = _read("client/stageweaver_runner.py")
        match = re.search(r"ACTIVE_TOOL_SCRIPTS = \[(.*?)\]", runner, re.S)
        self.assertIsNotNone(match)
        tools = re.findall(r'"server" / "([^"]+)"', match.group(1))
        self.assertEqual(
            tools,
            [
                "code_agent.py",
                "documents_tool.py",
                "image_tool.py",
                "math_tool.py",
                "ai_crawl.py",
                "search_tool.py",
            ],
        )

    def test_video_tool_not_default_loaded(self) -> None:
        runner = _read("client/stageweaver_runner.py")
        self.assertNotIn("video_tool.py", runner)
        documents = _read("server/documents_tool.py")
        self.assertNotIn("video_tool", documents)

    def test_search_tool_uses_searxng_only(self) -> None:
        search_tool = _read("server/search_tool.py")
        ai_crawl = _read("server/ai_crawl.py")
        runner = _read("client/stageweaver_runner.py")
        self.assertIn("SEARXNG_HOST", search_tool)
        self.assertIn("/search", search_tool)
        self.assertNotIn("SEARCH_BACKEND", search_tool)
        self.assertNotIn("ALLOW_NETWORK_SEARCH", search_tool)
        self.assertNotIn("offline://", ai_crawl)
        self.assertNotIn("--offline", runner)

    def test_archive_not_imported_by_active_modules(self) -> None:
        active = _active_python_text()
        self.assertNotIn("archive.", active)
        self.assertNotIn("from archive", active)
        self.assertNotIn("import archive", active)

    def test_deepresearcher_splits_used_directly(self) -> None:
        runner = _read("client/stageweaver_runner.py")
        self.assertIn("deepresearcher_protocol", runner)
        self.assertIn("ood_test.jsonl", runner)
        self.assertTrue((ROOT / "data/deepresearcher_protocol/seen_train.jsonl").is_file())
        self.assertTrue((ROOT / "data/deepresearcher_protocol/seen_dev.jsonl").is_file())
        self.assertTrue((ROOT / "data/deepresearcher_protocol/ood_test.jsonl").is_file())
        shard_dir = ROOT / "data/deepresearcher_protocol/seen_train_shards"
        self.assertFalse(any(shard_dir.glob("*.jsonl")) if shard_dir.exists() else False)

    def test_no_historical_run_id_naming_in_active_files(self) -> None:
        active = _active_python_text() + "\n" + _workflow_doc_text() + "\n" + _workflow_script_text()
        prefixes = ["R" + suffix for suffix in ("040", "041", "042", "043", "044", "045")]
        run_wrappers = ["run_r" + suffix for suffix in ("040", "041", "043", "044", "045")]
        for obsolete in (*prefixes, *run_wrappers):
            self.assertNotIn(obsolete, active)

    def test_semantic_workflow_names_are_documented(self) -> None:
        docs = _active_doc_text()
        for stage_name in (
            "collect_seen_train_traces",
            "collect_seen_dev_traces",
            "build_stageweaver_bank",
            "train_append_sft",
            "eval_seen_dev_gate",
            "eval_ood_test",
        ):
            self.assertIn(stage_name, docs)

    def test_current_output_paths_and_memory_source_are_semantic(self) -> None:
        runner = _read("client/stageweaver_runner.py")
        protocol = _read("docs/CURRENT_PROTOCOL.md")
        self.assertIn("result\" / \"stageweaver\" / \"current\" / \"stage_bank\" / \"stage_bank_train.jsonl", runner)
        self.assertIn("result\" / \"stageweaver\" / \"current\" / \"eval_ood_test", runner)
        self.assertNotIn('PROJECT_ROOT / "memory" / "memory.jsonl"', runner)
        self.assertIn("There is no fallback to legacy `memory.jsonl`", protocol)

    def test_trace_bank_defaults_use_teacher_endpoint(self) -> None:
        argv = [
            "stageweaver_runner.py",
            "--memory_mode",
            "none",
            "--diagnostic_trace_bank",
        ]
        env = {
            "TEACHER_BASE_URL": "http://teacher.local/v1",
            "TEACHER_API_KEY": "teacher-key",
            "TEACHER_MODEL": "teacher-model",
        }
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, env, clear=False):
            args = stageweaver_runner.parse_args()
        self.assertEqual(args.openai_base_url, "http://teacher.local/v1")
        self.assertEqual(args.openai_api_key, "teacher-key")
        self.assertEqual(args.meta_model, "teacher-model")
        self.assertEqual(args.exec_model, "teacher-model")

    def test_trace_bank_explicit_agent_args_override_teacher_defaults(self) -> None:
        argv = [
            "stageweaver_runner.py",
            "--memory_mode",
            "none",
            "--diagnostic_trace_bank",
            "--openai_base_url",
            "http://manual.local/v1",
            "--meta_model",
            "manual-meta",
        ]
        env = {
            "TEACHER_BASE_URL": "http://teacher.local/v1",
            "TEACHER_API_KEY": "teacher-key",
            "TEACHER_MODEL": "teacher-model",
        }
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, env, clear=False):
            args = stageweaver_runner.parse_args()
        self.assertEqual(args.openai_base_url, "http://manual.local/v1")
        self.assertEqual(args.openai_api_key, "teacher-key")
        self.assertEqual(args.meta_model, "manual-meta")
        self.assertEqual(args.exec_model, "teacher-model")

    def test_generation_caps_include_executor_cap(self) -> None:
        argv = [
            "stageweaver_runner.py",
            "--memory_mode",
            "none",
            "--diagnostic_trace_bank",
            "--planner_max_new_tokens",
            "512",
            "--executor_max_new_tokens",
            "768",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = stageweaver_runner.parse_args()
        self.assertEqual(args.planner_max_new_tokens, 512)
        self.assertEqual(args.executor_max_new_tokens, 768)
        self.assertEqual(stageweaver_runner._generation_cap_tag(512, 768), "_pmax512_emax768")

    def test_semantic_workflow_scripts_exist(self) -> None:
        for path in WORKFLOW_SCRIPTS:
            self.assertTrue(path.is_file(), f"missing workflow script: {path}")

    def test_no_windows_absolute_paths_in_active_files(self) -> None:
        active = _active_python_text() + "\n" + _active_doc_text() + "\n" + _workflow_script_text()
        self.assertIsNone(re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", active))

    def test_memento_text_fails_clearly_if_stage_bank_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "stage_bank_train.jsonl"
            with self.assertRaisesRegex(FileNotFoundError, "Run build_stageweaver_bank first"):
                load_memento_planner_cases(missing_path)

    def test_memento_text_filters_executor_tuples(self) -> None:
        planner_row = {
            "stage": "PLAN_INIT",
            "agent_role": "planner",
            "state_text": "Question A",
            "question_text": "Question A",
            "target_text": '{"plan": [{"id": 1, "description": "Find A"}]}',
            "reward": 1,
        }
        executor_row = {
            "stage": "EXEC_STEP",
            "agent_role": "executor",
            "state_text": "Task A",
            "question_text": "Question A",
            "target_text": "[RETURN] value",
            "reward": 1,
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "stage_bank_train.jsonl"
            path.write_text(
                json.dumps(planner_row, ensure_ascii=False) + "\n" + json.dumps(executor_row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            rows = load_memento_planner_cases(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["agent_role"], "planner")

    def test_offline_search_interfaces_removed(self) -> None:
        search_tool = _read("server/search_tool.py")
        ai_crawl = _read("server/ai_crawl.py")
        self.assertFalse((ROOT / "server/offline_search.py").exists())
        self.assertFalse((ROOT / "scripts/build_offline_search_index.py").exists())
        self.assertFalse((ROOT / "scripts/build_offline_search_index.sh").exists())
        self.assertFalse((ROOT / "scripts/offline_search_smoke.sh").exists())
        self.assertFalse((ROOT / "docs/OFFLINE_SEARCH.md").exists())
        self.assertNotIn("offline_search", search_tool)
        self.assertNotIn("search_offline", search_tool)
        self.assertNotIn("offline://", ai_crawl)

    def test_cleanup_manifest_records_semantic_naming_policy(self) -> None:
        manifest = _read("docs/CLEANUP_MANIFEST.md")
        self.assertIn("Future workflow stages use semantic names, not historical R-run IDs.", manifest)
        self.assertIn("No R04" + "0+ naming is used in the current protocol.", manifest)


if __name__ == "__main__":
    unittest.main()
