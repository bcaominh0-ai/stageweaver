from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import string
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import tiktoken
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MEMORY_DIR) not in sys.path:
    sys.path.insert(0, str(MEMORY_DIR))

try:  # script and module execution compatibility
    from client.agent_local_server import (  # type: ignore  # noqa: E402
        DEFAULT_EXECUTOR_GENERATION_HEADROOM,
        DEFAULT_GENERATION_HEADROOM,
        HierarchicalClient,
        _resolve_local_model_path,
    )
except Exception:  # pragma: no cover
    from agent_local_server import (  # type: ignore  # noqa: E402
        DEFAULT_EXECUTOR_GENERATION_HEADROOM,
        DEFAULT_GENERATION_HEADROOM,
        HierarchicalClient,
        _resolve_local_model_path,
    )
from memory.stageweaver_composer import StageWeaverComposer, StageWeaverComposerConfig  # noqa: E402
from memory.stageweaver_projector import StageWeaverProjector  # noqa: E402
from memory.stageweaver_schema import (  # noqa: E402
    StageTuple,
    build_executor_current_state,
    is_role_memory_item,
    load_stage_tuples,
    normalize_optional_text,
    retrieval_query_text,
    retrieval_text,
    require_explicit_executor_memory_jsonl,
    require_positive_executor_cases,
    serialize_role_conditioned_source,
    stage_memory_retrieval_key,
    tuple_role,
)
from memory.semantic_retriever import DEFAULT_SEMANTIC_MODEL_ID, SemanticRetriever  # noqa: E402
from memory.memento_text_memory import load_memento_planner_cases  # noqa: E402
from memory.stageweaver_serializers import render_positive_output_memory  # noqa: E402

ACTIVE_TOOL_SCRIPTS = [
    PROJECT_ROOT / "server" / "code_agent.py",
    PROJECT_ROOT / "server" / "documents_tool.py",
    PROJECT_ROOT / "server" / "image_tool.py",
    PROJECT_ROOT / "server" / "math_tool.py",
    PROJECT_ROOT / "server" / "ai_crawl.py",
    PROJECT_ROOT / "server" / "search_tool.py",
]

ACTIVE_MEMORY_MODES = {"memento_text", "stageweaver", "none"}

JUDGE_PROMPT_TPL = """You will be given a question and its ground truth answer list where each item can be a ground truth answer. Provided a pred_answer, judge if the pred_answer correctly answers the question based on the ground truth answer list.

Here is the criteria for the judgement:
1. The pred_answer doesn't need to be exactly the same as any of the ground truth answers, but should be semantically same for the question.
2. Each item in the ground truth answer list can be viewed as a ground truth answer for the question, and the pred_answer should be semantically same to at least one of them.

question: {question}
ground truth answers: {gt_answer}
pred_answer: {pred_answer}

Output exactly one lowercase word and nothing else:
correct
or
incorrect
"""


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    table = str.maketrans({ch: " " for ch in string.punctuation})
    text = text.translate(table)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_ground_truth(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split("<|answer_split|>")]
        return [p for p in parts if p]
    return [str(value).strip()]


def _exact_match(pred: str, gt_values: list[str]) -> bool:
    pred_n = _normalize_text(pred)
    for gt in gt_values:
        gt_n = _normalize_text(gt)
        if pred_n == gt_n:
            return True
    return False


def _safe_token_trim(text: str, token_budget: int, model: str = "gpt-4.1") -> str:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return " ".join(text.split()[:token_budget]).strip()
    toks = enc.encode(text)
    if len(toks) <= token_budget:
        return text
    return enc.decode(toks[:token_budget]).strip()


def _token_count(text: str, model: str = "gpt-4.1") -> int:
    if not text:
        return 0
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return len(text.split())
    return len(enc.encode(text))


def load_deepresearcher(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def _tuple_to_memory_case(item: StageTuple) -> dict[str, Any]:
    role = tuple_role(item)
    return {
        "state_text": retrieval_text(item),
        "question": str(item.question_text).strip(),
        "target_text": str(item.target_text).strip(),
        "source_id": str(item.source_id),
        "stage": str(item.stage),
        "reward": 1 if is_role_memory_item(item) else int(item.reward),
        "agent_role": role,
        "current_state_text": str(item.current_state_text).strip(),
        "metadata": dict(item.metadata),
    }


def _stageweaver_positive_memory_cases(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items if int(item.get("reward", 0)) == 1]


def _stageweaver_role_memory_cases(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items if is_role_memory_item(item)]


def _memory_key(item: dict[str, Any]) -> str:
    if str(item.get("agent_role", "")).strip() == "planner":
        question = str(item.get("question", "")).strip()
        if question:
            return question
    return str(item.get("question", item.get("state_text", item.get("case", "")))).strip()


def _memory_value(item: dict[str, Any]) -> str:
    return str(item.get("plan", item.get("target_text", ""))).strip()


def _stageweaver_retrieval_key(item: dict[str, Any]) -> str:
    return stage_memory_retrieval_key(item)


def _stageweaver_retrieval_query(*, role: str, question: str, current_state: str) -> str:
    return retrieval_query_text(role=role, question=question, current_state=current_state)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    match = re.search(r"{[\s\S]*}", text)
    return match.group(0) if match else text


def _format_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    exc_type = type(exc).__name__
    return f"{exc_type}: {message}" if message else exc_type


def _is_quota_exception(exc: BaseException) -> bool:
    text = _format_exception(exc)
    return any(
        marker in text
        for marker in (
            "insufficient_user_quota",
            "用户额度不足",
            "预扣费额度失败",
            "Arrearage",
            "overdue-payment",
            "account is in good standing",
            "invalid_parameter_error",
            "parameter of the code model must be in JSON format",
        )
    )


def build_memento_text_prompt(query: str, retrieved_hits: list[dict[str, Any]]) -> str:
    positive_cases: list[dict[str, Any]] = []
    negative_cases: list[dict[str, Any]] = []
    for hit in retrieved_hits:
        case = dict(hit.get("item", {}))
        score = hit.get("score")
        if score is not None:
            case["score"] = score
        reward = int(case.get("reward", 0))
        if reward == 1:
            positive_cases.append(case)
        else:
            negative_cases.append(case)

    prompt_parts: list[str] = []
    if positive_cases:
        prompt_parts.append(f"Positive Examples (reward=1) - Showing {len(positive_cases)} of {len(positive_cases)}:")
        for i, case in enumerate(positive_cases, 1):
            plan_text = _memory_value(case)
            try:
                plan_data = json.loads(plan_text)
                plan_steps = plan_data.get("plan", [])
                plan_text = "\n".join(f"{step['id']}. {step['description']}" for step in plan_steps)
            except Exception:
                pass
            prompt_parts.append(
                f"Example {i}:\nQuestion: {_memory_key(case)}\nPlan:\n{plan_text}\n"
            )
    if negative_cases:
        prompt_parts.append(f"Negative Examples (reward=0) - Showing {len(negative_cases)} of {len(negative_cases)}:")
        for i, case in enumerate(negative_cases, 1):
            plan_text = _memory_value(case)
            try:
                plan_data = json.loads(plan_text)
                plan_steps = plan_data.get("plan", [])
                plan_text = "\n".join(f"{step['id']}. {step['description']}" for step in plan_steps)
            except Exception:
                pass
            prompt_parts.append(
                f"Example {i}:\nQuestion: {_memory_key(case)}\nPlan: {plan_text}\n"
            )
    prompt_parts.append(
        "Based on the above examples, please provide a plan for the current task. "
        "Focus on the positive examples and avoid the patterns shown in negative examples.\n\nYour plan:"
    )
    return "\n".join(prompt_parts)


def build_baseline_memory_prompt(
    *,
    mode: str,
    query: str,
    retrieved_hits: list[dict[str, Any]],
) -> str:
    if mode == "none":
        return ""
    if mode == "memento_text":
        return build_memento_text_prompt(query, retrieved_hits)
    raise ValueError(f"Unsupported baseline memory mode: {mode}")


def _available_tool_names(tools_schema: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for tool in tools_schema:
        function = tool.get("function", {})
        name = str(function.get("name", "")).strip()
        if name:
            names.append(name)
    return names


def _latest_planner_output(trace: dict[str, Any]) -> str:
    for cycle in reversed(trace.get("cycles", [])):
        output = normalize_optional_text(cycle.get("planner_output", ""))
        if output:
            return output
    return ""


def _serialize_role_conditioned_source(
    *,
    role: str,
    current_state: str,
    retrieved_cases_text: str,
    stage: str = "",
) -> str:
    return serialize_role_conditioned_source(
        role=role,
        stage=stage,
        current_state=current_state,
        retrieved_cases_text=retrieved_cases_text,
    )


def _resolve_stageweaver_planner_retriever(
    grouped_retrievers: dict[tuple[str, str], SemanticRetriever],
    stage_name: str,
) -> tuple[tuple[str, str], SemanticRetriever]:
    key = (stage_name, "planner")
    retriever = grouped_retrievers.get(key)
    if retriever is None:
        raise RuntimeError(f"stageweaver planner retriever missing for key={key}")
    return key, retriever


def _merged_stage_mode_label(summaries: list[dict[str, Any]], requested_stage_mode: str) -> str:
    stage_modes = {str(summary.get("stage_mode", "")).strip() for summary in summaries if summary.get("stage_mode")}
    if not stage_modes:
        return requested_stage_mode
    if len(stage_modes) == 1:
        return next(iter(stage_modes))
    return "mixed"


def _generation_cap_tag(planner_max_new_tokens: int, executor_max_new_tokens: int) -> str:
    parts: list[str] = []
    if int(planner_max_new_tokens) != int(DEFAULT_GENERATION_HEADROOM):
        parts.append(f"pmax{planner_max_new_tokens}")
    if int(executor_max_new_tokens) != int(DEFAULT_EXECUTOR_GENERATION_HEADROOM):
        parts.append(f"emax{executor_max_new_tokens}")
    return ("_" + "_".join(parts)) if parts else ""


def _cli_option_present(argv: list[str], option: str) -> bool:
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv[1:])


def _apply_trace_bank_teacher_defaults(args: argparse.Namespace, argv: list[str]) -> None:
    if not args.diagnostic_trace_bank or "none" not in args.modes:
        return
    for option, attr, env_name in (
        ("--openai_base_url", "openai_base_url", "TEACHER_BASE_URL"),
        ("--openai_api_key", "openai_api_key", "TEACHER_API_KEY"),
        ("--meta_model", "meta_model", "TEACHER_MODEL"),
        ("--exec_model", "exec_model", "TEACHER_MODEL"),
    ):
        value = os.getenv(env_name, "").strip()
        if value and not _cli_option_present(argv, option):
            setattr(args, attr, value)


async def llm_judge(
    judge_client: AsyncOpenAI,
    judge_model: str,
    question: str,
    ground_truth: list[str],
    pred_answer: str,
) -> dict[str, Any]:
    prompt = JUDGE_PROMPT_TPL.format(
        question=question,
        gt_answer=json.dumps(ground_truth, ensure_ascii=False),
        pred_answer=pred_answer,
    )
    resp = await judge_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
    )
    content = resp.choices[0].message.content or ""
    content = _strip_fences(content)
    judgement = content.lower().strip().strip(".")
    rationale = ""
    if judgement not in {"correct", "incorrect"}:
        data = json.loads(content)
        judgement = str(data.get("judgement", "incorrect")).lower().strip()
        rationale = str(data.get("rationale", ""))
    if judgement not in {"correct", "incorrect"}:
        raise ValueError(f"invalid judge label: {judgement}")
    return {"judgement": judgement, "rationale": rationale}


class StageWeaverComposerRuntime:
    def __init__(
        self,
        ckpt_path: Path,
        device: str = "cpu",
        max_length: int = 512,
        override_model_name_or_path: str | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.max_length = int(max_length)
        payload: dict[str, Any] = torch.load(str(ckpt_path), map_location=self.device, weights_only=True)
        composer_config = dict(payload.get("composer_config", {}))
        composer_state_dict = payload.get("composer_state_dict")
        projector_config = dict(payload.get("projector_config", {}))
        projector_state_dict = payload.get("projector_state_dict")
        if not composer_config or composer_state_dict is None:
            raise KeyError(f"composer checkpoint is missing composer_config/composer_state_dict: {ckpt_path}")
        if not projector_config or projector_state_dict is None:
            raise KeyError(f"composer checkpoint is missing projector_config/projector_state_dict: {ckpt_path}")
        if override_model_name_or_path:
            composer_config["model_name_or_path"] = override_model_name_or_path

        self.composer = StageWeaverComposer(
            StageWeaverComposerConfig(**composer_config),
            device=self.device,
        )
        self.composer.load_state_dict(composer_state_dict, strict=False)
        self.composer.eval()
        self.projector = StageWeaverProjector(
            composer_hidden_size=int(projector_config["composer_hidden_size"]),
            agent_hidden_size=int(projector_config["agent_hidden_size"]),
            hidden_multiplier=int(projector_config.get("hidden_multiplier", 2)),
        ).to(self.device, dtype=next(self.composer.parameters()).dtype)
        self.projector.load_state_dict(projector_state_dict)
        self.projector.eval()
        self.agent_hidden_size = int(projector_config["agent_hidden_size"])
        self.agent_model_path = str(payload.get("agent_model_path", "")).strip()

    def encode_memory_block(self, source_text: str) -> torch.Tensor:
        tokenized = self.composer.tokenize([source_text], max_length=self.max_length)
        with torch.no_grad():
            composer_latent = self.composer.text_to_latent(
                tokenized["input_ids"].to(self.device),
                tokenized["attention_mask"].to(self.device),
            )
            agent_latent = self.projector(composer_latent)
        return agent_latent.detach()


async def run_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    effective_stage_mode = args.stage_mode
    data_rows = load_deepresearcher(Path(args.data_jsonl), limit=args.limit)
    memory_cases: list[dict[str, Any]] = []
    stageweaver_composer_runtime: StageWeaverComposerRuntime | None = None
    retriever: SemanticRetriever | None = None
    memento_executor_retriever: SemanticRetriever | None = None
    stageweaver_grouped_retrievers: dict[tuple[str, str], SemanticRetriever] = {}
    judge_client = AsyncOpenAI(api_key=args.judge_api_key, base_url=args.judge_base_url)
    if mode == "stageweaver":
        stageweaver_tuples = [item for item in load_stage_tuples(Path(args.stageweaver_bank_jsonl)) if is_role_memory_item(item)]
        override_composer_model = ""
        if args.stageweaver_composer_model or args.stageweaver_composer_model_path:
            try:
                override_composer_model = _resolve_local_model_path(
                    args.stageweaver_composer_model,
                    args.stageweaver_composer_model_path or None,
                )
            except FileNotFoundError:
                override_composer_model = args.stageweaver_composer_model_path or args.stageweaver_composer_model or ""
        stageweaver_composer_runtime = StageWeaverComposerRuntime(
            ckpt_path=Path(args.stageweaver_composer_ckpt),
            device=args.stageweaver_device,
            max_length=args.stageweaver_composer_max_length,
            override_model_name_or_path=(override_composer_model or None),
        )
        memory_cases = [_tuple_to_memory_case(item) for item in stageweaver_tuples]
        if not memory_cases:
            raise RuntimeError("stageweaver mode requires at least one role memory case in the stageweaver bank.")
    if mode in {"memento_text", "stageweaver"}:
        if mode == "stageweaver":
            grouped_items: dict[tuple[str, str], list[dict[str, Any]]] = {}
            for item in _stageweaver_role_memory_cases(memory_cases):
                key = (str(item.get("stage", "")), str(item.get("agent_role", "")))
                grouped_items.setdefault(key, []).append(dict(item))
            for key, items in grouped_items.items():
                grouped_retriever = SemanticRetriever(
                    model_id=args.semantic_model_id,
                    device=args.semantic_device,
                    cache_dir=args.semantic_cache_dir,
                    max_seq_length=args.semantic_max_length,
                )
                grouped_retriever.build(items, _stageweaver_retrieval_key)
                stageweaver_grouped_retrievers[key] = grouped_retriever
            if args.executor_memory_refresh in {"initial", "per_step"} and ("EXEC_STEP", "executor") not in stageweaver_grouped_retrievers:
                require_positive_executor_cases([], args.stageweaver_bank_jsonl)
        else:
            memory_cases = load_memento_planner_cases(Path(args.memory_jsonl))
            retriever = SemanticRetriever(
                model_id=args.semantic_model_id,
                device=args.semantic_device,
                cache_dir=args.semantic_cache_dir,
                max_seq_length=args.semantic_max_length,
            )
            retriever.build(memory_cases, _memory_key)
            if args.executor_memory_refresh == "per_step":
                require_explicit_executor_memory_jsonl(mode, args.executor_memory_refresh, args.executor_memory_jsonl)
                executor_memory_path = Path(args.executor_memory_jsonl)
                executor_cases = [
                    _tuple_to_memory_case(item)
                    for item in load_stage_tuples(executor_memory_path)
                    if is_role_memory_item(item) and tuple_role(item) == "executor"
                ]
                require_positive_executor_cases(executor_cases, executor_memory_path)
                if executor_cases:
                    memento_executor_retriever = SemanticRetriever(
                        model_id=args.semantic_model_id,
                        device=args.semantic_device,
                        cache_dir=args.semantic_cache_dir,
                        max_seq_length=args.semantic_max_length,
                        )
                    memento_executor_retriever.build(executor_cases, _stageweaver_retrieval_key)
    server_scripts = [str(path.resolve()) for path in ACTIVE_TOOL_SCRIPTS]

    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    os.environ["OPENAI_BASE_URL"] = args.openai_base_url
    os.environ["DIRECT_API_KEY"] = args.openai_api_key
    os.environ["DIRECT_BASE_URL"] = args.openai_base_url

    generation_cap_tag = _generation_cap_tag(args.planner_max_new_tokens, args.executor_max_new_tokens)
    if mode == "stageweaver":
        run_tag = f"{mode}_{args.tool_profile}_{effective_stage_mode}_{args.latent_interface}{generation_cap_tag}"
    else:
        run_tag = f"{mode}_{args.tool_profile}_{effective_stage_mode}{generation_cap_tag}"
    if args.trace_jsonl and len(args.modes) == 1:
        if args.trace_jsonl == "__AUTO_TRACE__":
            trace_path = Path(args.output_dir) / f"trace_{run_tag}.jsonl"
        else:
            trace_path = Path(args.trace_jsonl)
    else:
        trace_path = Path(args.output_dir) / f"trace_{run_tag}.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    resume = bool(getattr(args, "resume", False))
    if trace_path.exists() and not resume:
        trace_path.unlink()

    client = HierarchicalClient(
        meta_model=args.meta_model,
        exec_model=args.exec_model,
        is_azure=False,
        trace_jsonl=str(trace_path),
        meta_model_path=(args.meta_model_path or None) if args.agent_backend == "local" else None,
        exec_model_path=(args.exec_model_path or None) if args.agent_backend == "local" else None,
        prefer_local_backend=False,
        backend_mode=args.agent_backend,
    )
    if mode == "stageweaver":
        if not getattr(client.meta_llm, "supports_prefix_embeds", False):
            raise RuntimeError("stageweaver latent runtime requires a direct local planner backend with prefix embedding support.")
        planner_hidden_size = int(getattr(client.meta_llm, "hidden_size"))
        planner_model_path = _resolve_local_model_path(args.meta_model, args.meta_model_path or None)
        executor_model_path = _resolve_local_model_path(args.exec_model, args.exec_model_path or None)
        if stageweaver_composer_runtime is None:
            raise RuntimeError("stageweaver composer runtime is not initialized.")
        if stageweaver_composer_runtime.agent_hidden_size != planner_hidden_size:
            raise RuntimeError(
                "stageweaver composer checkpoint hidden size does not match planner hidden size. "
                f"composer={stageweaver_composer_runtime.agent_hidden_size}, planner={planner_hidden_size}"
            )
        if stageweaver_composer_runtime.agent_model_path:
            expected_agent_path = str(Path(stageweaver_composer_runtime.agent_model_path).resolve())
            if str(Path(planner_model_path).resolve()) != expected_agent_path or str(Path(executor_model_path).resolve()) != expected_agent_path:
                raise RuntimeError(
                    "stageweaver composer checkpoint was trained against a different local agent model path. "
                    f"checkpoint={expected_agent_path}, planner={planner_model_path}, executor={executor_model_path}"
                )
        else:
            raise RuntimeError(
                "stageweaver composer checkpoint is missing agent_model_path. "
                "Refuse to run because hidden-size equality alone is not a safe compatibility check."
            )
        if not getattr(client.exec_llm, "supports_prefix_embeds", False):
            raise RuntimeError("stageweaver executor latent runtime requires a direct local executor backend with prefix embedding support.")
        executor_hidden = int(getattr(client.exec_llm, "hidden_size"))
        if planner_hidden_size != executor_hidden:
            raise RuntimeError(
                "stageweaver requires planner and executor to share the same local hidden size. "
                f"planner={planner_hidden_size}, executor={executor_hidden}"
            )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"results_{run_tag}.jsonl"
    summary_json = out_dir / f"summary_{run_tag}.json"
    existing_by_index: dict[int, dict[str, Any]] = {}
    if resume and out_jsonl.exists():
        with out_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                index = record.get("index")
                if isinstance(index, int):
                    existing_by_index[index] = record
    if out_jsonl.exists() and not resume:
        out_jsonl.unlink()

    records: list[dict[str, Any]] = [existing_by_index[idx] for idx in sorted(existing_by_index)]
    total = len(records)
    correct = sum(1 for record in records if bool(record.get("correct")))
    planner_token_sum = sum(int(record.get("planner_memory_tokens", 0) or 0) for record in records)
    executor_token_sum = sum(int(record.get("executor_memory_tokens", 0) or 0) for record in records)
    planner_prefix_token_sum = sum(int(record.get("planner_prefix_tokens", 0) or 0) for record in records)
    executor_prefix_token_sum = sum(int(record.get("executor_prefix_tokens", 0) or 0) for record in records)
    planner_retrieved_sum = sum(int(record.get("planner_retrieved_cases", 0) or 0) for record in records)
    executor_retrieved_sum = sum(int(record.get("executor_retrieved_cases", 0) or 0) for record in records)
    try:
        await client.connect_to_servers(server_scripts)
        for idx, row in enumerate(data_rows):
            if idx in existing_by_index:
                continue
            query = str(row.get("question", ""))
            query_deadline = (time.monotonic() + float(args.query_timeout_sec)) if args.query_timeout_sec > 0 else None
            memory_prompt = ""
            planner_memory_prompt = ""
            executor_memory_prompt = ""
            planner_memory_role = "system"
            planner_prefix_embeds: torch.Tensor | None = None
            executor_prefix_embeds: torch.Tensor | None = None
            planner_memory_callback = None
            executor_memory_callback = None
            retrieved_hits: list[dict[str, Any]] = []
            gt_values = _parse_ground_truth(row.get("ground_truth"))
            query_memory_stats = {
                "planner_calls": 0,
                "executor_calls": 0,
                "planner_prefix_tokens": 0,
                "executor_prefix_tokens": 0,
                "planner_retrieved_cases": 0,
                "executor_retrieved_cases": 0,
                "planner_text_tokens": 0,
                "executor_text_tokens": 0,
                "planner_text_chars": 0,
                "executor_text_chars": 0,
            }
            try:
                if mode == "stageweaver":
                    if stageweaver_composer_runtime is None:
                        raise RuntimeError("stageweaver composer runtime is not initialized.")
                    executor_top_k = int(args.executor_memory_top_k or args.memory_top_k)

                    async def _planner_memory_callback(context: dict[str, Any]) -> dict[str, Any]:
                        # Planner now relies on shared_history for textual progress state.
                        # Avoid duplicating task/tool summaries in a separate planner memory block.
                        text_memory = ""
                        stage_name = "PLAN_INIT" if int(context["cycle"]) == 0 else "PLAN_REVISE"
                        planner_query = str(context.get("query", "")).strip()
                        current_state = planner_query
                        retrieval_query = _stageweaver_retrieval_query(
                            role="planner",
                            question=planner_query,
                            current_state=current_state,
                        )
                        retriever_key, grouped_retriever = _resolve_stageweaver_planner_retriever(
                            stageweaver_grouped_retrievers,
                            stage_name,
                        )
                        hits = grouped_retriever.retrieve(retrieval_query, top_k=args.memory_top_k)
                        retrieved = _stageweaver_role_memory_cases([dict(hit["item"]) for hit in hits])
                        rendered = render_positive_output_memory(
                            retrieved,
                            budget_tokens=args.memory_budget_tokens,
                            bounded_budget_tokens=args.bounded_memory_budget_tokens,
                            model=args.retrieval_model,
                        )
                        source_text = _serialize_role_conditioned_source(
                            role="planner",
                            stage=stage_name,
                            current_state=normalize_optional_text(current_state) or planner_query or "[NONE]",
                            retrieved_cases_text=rendered["text"],
                        )
                        prefix = stageweaver_composer_runtime.encode_memory_block(source_text)
                        query_memory_stats["planner_calls"] += 1
                        query_memory_stats["planner_prefix_tokens"] += int(prefix.size(1))
                        query_memory_stats["planner_retrieved_cases"] += len(retrieved)
                        query_memory_stats["planner_text_tokens"] += _token_count(text_memory, model=args.retrieval_model)
                        query_memory_stats["planner_text_chars"] += len(text_memory)
                        return {
                            "text_memory": text_memory,
                            "memory_role": "user",
                            "prefix_embeds": prefix,
                            "prefix_injection_mode": args.latent_interface,
                            "trace": {
                                "role": "planner",
                                "stage": stage_name,
                                "prefix_injection_mode": args.latent_interface,
                                "text_memory_chars": len(text_memory),
                                "retrieved_cases": len(retrieved),
                                "retrieved_ids": [str(item.get("source_id", "")) for item in retrieved],
                                "retrieval_query": retrieval_query,
                                "retriever_key": f"{retriever_key[0]}/{retriever_key[1]}",
                                "retriever_size": len(getattr(grouped_retriever, "items", [])),
                                "prefix_tokens": int(prefix.size(1)),
                                "source_text": source_text,
                            },
                        }

                    async def _executor_memory_callback(context: dict[str, Any]) -> dict[str, Any]:
                        task = dict(context["task"])
                        text_memory = ""
                        executor_query = str(context.get("query", "")).strip()
                        task_description = str(task.get("description", "")).strip() or executor_query
                        current_state = task_description
                        retrieval_query = _stageweaver_retrieval_query(
                            role="executor",
                            question=executor_query,
                            current_state=current_state,
                        )
                        grouped_retriever = stageweaver_grouped_retrievers.get(("EXEC_STEP", "executor"))
                        if grouped_retriever is None:
                            raise RuntimeError("stageweaver executor retriever missing for key=('EXEC_STEP', 'executor')")
                        hits = grouped_retriever.retrieve(retrieval_query, top_k=executor_top_k)
                        retrieved = _stageweaver_role_memory_cases([dict(hit["item"]) for hit in hits])
                        rendered = render_positive_output_memory(
                            retrieved,
                            budget_tokens=args.memory_budget_tokens,
                            bounded_budget_tokens=args.bounded_memory_budget_tokens,
                            model=args.retrieval_model,
                        )
                        source_text = _serialize_role_conditioned_source(
                            role="executor",
                            stage="EXEC_STEP",
                            current_state=normalize_optional_text(current_state) or executor_query or "[NONE]",
                            retrieved_cases_text=rendered["text"],
                        )
                        prefix = stageweaver_composer_runtime.encode_memory_block(source_text)
                        source_text_hash = _hash_text(source_text)
                        previous_hash = str(context.get("task_trace", {}).get("executor_memory_last_hash", ""))
                        if isinstance(context.get("task_trace"), dict):
                            context["task_trace"]["executor_memory_last_hash"] = source_text_hash
                        query_memory_stats["executor_calls"] += 1
                        query_memory_stats["executor_prefix_tokens"] += int(prefix.size(1))
                        query_memory_stats["executor_retrieved_cases"] += len(retrieved)
                        query_memory_stats["executor_text_tokens"] += _token_count(text_memory, model=args.retrieval_model)
                        query_memory_stats["executor_text_chars"] += len(text_memory)
                        return {
                            "text_memory": text_memory,
                            "prefix_embeds": prefix,
                            "prefix_injection_mode": args.latent_interface,
                            "trace": {
                                "role": "executor",
                                "stage": "EXEC_STEP",
                                "prefix_injection_mode": args.latent_interface,
                                "text_memory_chars": len(text_memory),
                                "memory_refresh_mode": args.executor_memory_refresh,
                                "step_id": int(context.get("executor_step_id", 0)),
                                "retrieved_cases": len(retrieved),
                                "retrieved_ids": [str(item.get("source_id", "")) for item in retrieved],
                                "retrieved_scores": [float(hit.get("score", 0.0)) for hit in hits],
                                "retrieval_query": retrieval_query,
                                "retrieval_query_hash": _hash_text(retrieval_query),
                                "source_text_hash": source_text_hash,
                                "whether_memory_changed_from_previous_step": previous_hash != source_text_hash,
                                "retriever_key": "EXEC_STEP/executor",
                                "retriever_size": len(getattr(grouped_retriever, "items", [])),
                                "prefix_tokens": int(prefix.size(1)),
                                "source_text": source_text,
                            },
                        }

                    planner_memory_callback = _planner_memory_callback
                    if args.executor_memory_refresh != "none":
                        executor_memory_callback = _executor_memory_callback
                elif mode in {"none", "memento_text"}:
                    if mode == "memento_text":
                        if retriever is None:
                            raise RuntimeError(f"{mode} requires semantic retriever initialization.")
                        retrieved_hits = retriever.retrieve(query, top_k=args.memory_top_k)
                        if args.executor_memory_refresh == "per_step" and memento_executor_retriever is not None:
                            executor_top_k = int(args.executor_memory_top_k or args.memory_top_k)

                            async def _memento_executor_memory_callback(context: dict[str, Any]) -> dict[str, Any]:
                                task = dict(context["task"])
                                task_trace = dict(context.get("task_trace", {}))
                                tool_history = [dict(call) for call in task_trace.get("tool_calls", []) if isinstance(call, dict)]
                                latest_observation = ""
                                if tool_history:
                                    latest = tool_history[-1]
                                    latest_observation = str(latest.get("result_preview") or latest.get("error") or "")
                                failed_calls = [call for call in tool_history if call.get("error")]
                                repeated_calls = [call for call in tool_history if call.get("reused_result")]
                                task_description = str(task.get("description", "")).strip() or str(context.get("query", "")).strip()
                                current_state = build_executor_current_state(
                                    task_description=task_description,
                                    tool_history=tool_history,
                                    latest_observation=latest_observation,
                                    failed_calls=failed_calls,
                                    repeated_calls=repeated_calls,
                                    partial_result=str(task_trace.get("result") or ""),
                                    max_chars=args.executor_state_max_chars,
                                    obs_max_chars=args.executor_obs_max_chars,
                                    tool_history_k=args.executor_tool_history_k,
                                )
                                retrieval_query = _stageweaver_retrieval_query(
                                    role="executor",
                                    question=str(context.get("query", "")).strip(),
                                    current_state=current_state,
                                )
                                hits = memento_executor_retriever.retrieve(retrieval_query, top_k=executor_top_k)
                                retrieved = _stageweaver_positive_memory_cases([dict(hit["item"]) for hit in hits])
                                rendered = render_positive_output_memory(
                                    retrieved,
                                    budget_tokens=args.memory_budget_tokens,
                                    bounded_budget_tokens=args.bounded_memory_budget_tokens,
                                    model=args.retrieval_model,
                                )
                                source_text = _serialize_role_conditioned_source(
                                    role="executor",
                                    current_state=normalize_optional_text(current_state) or task_description or "[NONE]",
                                    retrieved_cases_text=rendered["text"],
                                )
                                source_text_hash = _hash_text(source_text)
                                previous_hash = str(context.get("task_trace", {}).get("executor_memory_last_hash", ""))
                                if isinstance(context.get("task_trace"), dict):
                                    context["task_trace"]["executor_memory_last_hash"] = source_text_hash
                                query_memory_stats["executor_calls"] += 1
                                query_memory_stats["executor_retrieved_cases"] += len(retrieved)
                                query_memory_stats["executor_text_tokens"] += _token_count(source_text, model=args.retrieval_model)
                                query_memory_stats["executor_text_chars"] += len(source_text)
                                return {
                                    "text_memory": source_text,
                                    "trace": {
                                        "role": "executor",
                                        "stage": "EXEC_STEP",
                                        "memory_refresh_mode": args.executor_memory_refresh,
                                        "step_id": int(context.get("executor_step_id", 0)),
                                        "text_memory_chars": len(source_text),
                                        "retrieved_cases": len(retrieved),
                                        "retrieved_ids": [str(item.get("source_id", "")) for item in retrieved],
                                        "retrieved_scores": [float(hit.get("score", 0.0)) for hit in hits],
                                        "retrieval_query": retrieval_query,
                                        "retrieval_query_hash": _hash_text(retrieval_query),
                                        "source_text_hash": source_text_hash,
                                        "whether_memory_changed_from_previous_step": previous_hash != source_text_hash,
                                        "retriever_key": "EXEC_STEP/executor",
                                        "source_text": source_text,
                                    },
                                }

                            executor_memory_callback = _memento_executor_memory_callback
                    memory_prompt = build_baseline_memory_prompt(
                        mode=mode,
                        query=query,
                        retrieved_hits=retrieved_hits,
                    )
                    if mode == "memento_text":
                        planner_memory_role = "user"
                else:
                    raise ValueError(f"Unsupported memory mode: {mode}")
                if memory_prompt:
                    planner_memory_prompt = f"[MEMORY_MODE={mode}]\n{memory_prompt}"
                if args.query_timeout_sec > 0:
                    if query_deadline is not None:
                        remaining = query_deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError(f"query timeout exceeded before process_query for mode={mode}, index={idx}")
                    else:
                        remaining = float(args.query_timeout_sec)
                    answer = await asyncio.wait_for(
                        client.process_query(
                            query,
                            file="",
                            task_id=f"{mode}-{idx}-{uuid.uuid4()}",
                            planner_memory_prompt=planner_memory_prompt,
                            executor_memory_prompt=executor_memory_prompt,
                            planner_memory_role=planner_memory_role,
                            planner_prefix_embeds=planner_prefix_embeds,
                            executor_prefix_embeds=executor_prefix_embeds,
                            planner_prefix_injection_mode=args.latent_interface,
                            executor_prefix_injection_mode=args.latent_interface,
                            planner_max_new_tokens=args.planner_max_new_tokens,
                            executor_max_new_tokens=args.executor_max_new_tokens,
                            planner_memory_callback=planner_memory_callback,
                            executor_memory_callback=executor_memory_callback,
                            executor_memory_refresh=args.executor_memory_refresh,
                        ),
                        timeout=remaining,
                    )
                else:
                    answer = await client.process_query(
                        query,
                        file="",
                        task_id=f"{mode}-{idx}-{uuid.uuid4()}",
                        planner_memory_prompt=planner_memory_prompt,
                        executor_memory_prompt=executor_memory_prompt,
                        planner_memory_role=planner_memory_role,
                        planner_prefix_embeds=planner_prefix_embeds,
                        executor_prefix_embeds=executor_prefix_embeds,
                        planner_prefix_injection_mode=args.latent_interface,
                        executor_prefix_injection_mode=args.latent_interface,
                        planner_max_new_tokens=args.planner_max_new_tokens,
                        executor_max_new_tokens=args.executor_max_new_tokens,
                        planner_memory_callback=planner_memory_callback,
                        executor_memory_callback=executor_memory_callback,
                        executor_memory_refresh=args.executor_memory_refresh,
                    )
                planner_tokens = max(_token_count(planner_memory_prompt, model=args.meta_model), query_memory_stats["planner_text_tokens"])
                executor_tokens = max(_token_count(executor_memory_prompt, model=args.exec_model), query_memory_stats["executor_text_tokens"])
                if args.judge_mode == "llm":
                    judge_result = await llm_judge(judge_client, args.judge_model, query, gt_values, answer)
                    is_correct = judge_result["judgement"] == "correct"
                else:
                    judge_result = {"judgement": "correct" if _exact_match(answer, gt_values) else "incorrect", "rationale": ""}
                    is_correct = judge_result["judgement"] == "correct"
                total += 1
                correct += int(is_correct)
                planner_token_sum += planner_tokens
                executor_token_sum += executor_tokens
                planner_prefix_token_sum += query_memory_stats["planner_prefix_tokens"]
                executor_prefix_token_sum += query_memory_stats["executor_prefix_tokens"]
                planner_retrieved_sum += query_memory_stats["planner_retrieved_cases"]
                executor_retrieved_sum += query_memory_stats["executor_retrieved_cases"]
                record = {
                    "mode": mode,
                    "tool_profile": args.tool_profile,
                    "stage_mode": effective_stage_mode,
                    "index": idx,
                    "question": query,
                    "data_source": str(row.get("data_source", "")).strip(),
                    "protocol_split": str(row.get("protocol_split", "")).strip(),
                    "pred_answer": answer,
                    "ground_truth": gt_values,
                    "correct": is_correct,
                    "judge_mode": args.judge_mode,
                    "judge_rationale": judge_result["rationale"],
                    "memory_prompt_chars": len(memory_prompt),
                    "planner_memory_chars": max(len(planner_memory_prompt), query_memory_stats["planner_text_chars"]),
                    "executor_memory_chars": max(len(executor_memory_prompt), query_memory_stats["executor_text_chars"]),
                    "planner_memory_tokens": planner_tokens,
                    "executor_memory_tokens": executor_tokens,
                    "retrieved_cases": len(retrieved_hits),
                    "planner_prefix_tokens": query_memory_stats["planner_prefix_tokens"],
                    "executor_prefix_tokens": query_memory_stats["executor_prefix_tokens"],
                    "planner_memory_calls": query_memory_stats["planner_calls"],
                    "executor_memory_calls": query_memory_stats["executor_calls"],
                    "planner_retrieved_cases": query_memory_stats["planner_retrieved_cases"],
                    "executor_retrieved_cases": query_memory_stats["executor_retrieved_cases"],
                }
            except Exception as exc:
                if isinstance(exc, asyncio.TimeoutError):
                    print(
                        json.dumps(
                            {
                                "fatal_error": "query_timeout",
                                "index": idx,
                                "error": _format_exception(exc),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                    )
                    raise
                if _is_quota_exception(exc):
                    print(
                        json.dumps(
                            {
                                "fatal_error": "quota_exhausted",
                                "index": idx,
                                "error": _format_exception(exc),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                    )
                    raise
                planner_tokens = max(_token_count(planner_memory_prompt, model=args.meta_model), query_memory_stats["planner_text_tokens"])
                executor_tokens = max(_token_count(executor_memory_prompt, model=args.exec_model), query_memory_stats["executor_text_tokens"])
                total += 1
                planner_token_sum += planner_tokens
                executor_token_sum += executor_tokens
                planner_prefix_token_sum += query_memory_stats["planner_prefix_tokens"]
                executor_prefix_token_sum += query_memory_stats["executor_prefix_tokens"]
                planner_retrieved_sum += query_memory_stats["planner_retrieved_cases"]
                executor_retrieved_sum += query_memory_stats["executor_retrieved_cases"]
                record = {
                    "mode": mode,
                    "tool_profile": args.tool_profile,
                    "stage_mode": effective_stage_mode,
                    "index": idx,
                    "question": query,
                    "data_source": str(row.get("data_source", "")).strip(),
                    "protocol_split": str(row.get("protocol_split", "")).strip(),
                    "pred_answer": "",
                    "ground_truth": gt_values,
                    "correct": False,
                    "error": _format_exception(exc),
                    "judge_mode": args.judge_mode,
                    "memory_prompt_chars": len(memory_prompt),
                    "planner_memory_chars": max(len(planner_memory_prompt), query_memory_stats["planner_text_chars"]),
                    "executor_memory_chars": max(len(executor_memory_prompt), query_memory_stats["executor_text_chars"]),
                    "planner_memory_tokens": planner_tokens,
                    "executor_memory_tokens": executor_tokens,
                    "retrieved_cases": len(retrieved_hits),
                    "planner_prefix_tokens": query_memory_stats["planner_prefix_tokens"],
                    "executor_prefix_tokens": query_memory_stats["executor_prefix_tokens"],
                    "planner_memory_calls": query_memory_stats["planner_calls"],
                    "executor_memory_calls": query_memory_stats["executor_calls"],
                    "planner_retrieved_cases": query_memory_stats["planner_retrieved_cases"],
                    "executor_retrieved_cases": query_memory_stats["executor_retrieved_cases"],
                }
            records.append(record)
            with out_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        await client.cleanup()

    summary = {
        "mode": mode,
        "tool_profile": args.tool_profile,
        "stage_mode": effective_stage_mode,
        "latent_interface": args.latent_interface,
        "executor_memory_refresh": args.executor_memory_refresh,
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "executor_max_new_tokens": args.executor_max_new_tokens,
        "meta_model": args.meta_model,
        "exec_model": args.exec_model,
        "judge_mode": args.judge_mode,
        "judge_model": args.judge_model,
        "retrieval_backend": "sentence-transformers",
        "semantic_model_id": args.semantic_model_id,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total else 0.0,
        "avg_planner_memory_tokens": (planner_token_sum / total) if total else 0.0,
        "avg_executor_memory_tokens": (executor_token_sum / total) if total else 0.0,
        "avg_planner_prefix_tokens": (planner_prefix_token_sum / total) if total else 0.0,
        "avg_executor_prefix_tokens": (executor_prefix_token_sum / total) if total else 0.0,
        "avg_planner_retrieved_cases": (planner_retrieved_sum / total) if total else 0.0,
        "avg_executor_retrieved_cases": (executor_retrieved_sum / total) if total else 0.0,
        "results_jsonl": str(out_jsonl),
        "trace_jsonl": str(trace_path),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def main_async(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[dict[str, Any]] = []
    for mode in args.modes:
        summary = await run_mode(args, mode)
        all_summaries.append(summary)

    merged_stage_mode = _merged_stage_mode_label(all_summaries, args.stage_mode)
    merged = {
        "data_jsonl": args.data_jsonl,
        "limit": args.limit,
        "modes": args.modes,
        "tool_profile": args.tool_profile,
        "stage_mode": merged_stage_mode,
        "latent_interface": args.latent_interface,
        "executor_memory_refresh": args.executor_memory_refresh,
        "planner_max_new_tokens": args.planner_max_new_tokens,
        "executor_max_new_tokens": args.executor_max_new_tokens,
        "semantic_model_id": args.semantic_model_id,
        "summaries": all_summaries,
    }
    mode_sig = "-".join(args.modes)
    generation_cap_tag = _generation_cap_tag(args.planner_max_new_tokens, args.executor_max_new_tokens)
    summary_name = (
        f"summary_all_{mode_sig}_{args.tool_profile}_{merged_stage_mode}_{args.latent_interface}{generation_cap_tag}.json"
        if any(mode == "stageweaver" for mode in args.modes)
        else f"summary_all_{mode_sig}_{args.tool_profile}_{merged_stage_mode}{generation_cap_tag}.json"
    )
    (output_dir / summary_name).write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(merged, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    default_stageweaver_bank = str(
        PROJECT_ROOT / "result" / "stageweaver" / "current" / "stage_bank" / "stage_bank_train.jsonl"
    )
    parser = argparse.ArgumentParser(description="Unified StageWeaver runner for the Memento text-memory baseline and StageWeaver.")
    parser.add_argument(
        "--data_jsonl",
        type=str,
        default=str(PROJECT_ROOT / "data" / "deepresearcher_protocol" / "ood_test.jsonl"),
    )
    parser.add_argument(
        "--memory_jsonl",
        type=str,
        default=default_stageweaver_bank,
        help="Planner-memory source for memento_text. Default uses planner tuples from the current StageWeaver bank built from current-protocol traces.",
    )
    parser.add_argument(
        "--executor_memory_jsonl",
        type=str,
        default="",
        help="Optional executor trajectory-memory source for memento_text per-step executor memory. Defaults to --memory_jsonl.",
    )
    parser.add_argument("--stageweaver_bank_jsonl", type=str, default=default_stageweaver_bank)
    parser.add_argument(
        "--stageweaver_composer_ckpt",
        type=str,
        default="",
        help="Append-aligned composer checkpoint (stageweaver_composer_sft.pt). Required for stageweaver mode.",
    )
    parser.add_argument("--stageweaver_composer_model", type=str, default="")
    parser.add_argument("--stageweaver_composer_model_path", type=str, default="")
    parser.add_argument("--stageweaver_device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--stageweaver_composer_max_length", type=int, default=512)
    parser.add_argument(
        "--latent_interface",
        choices=["append"],
        default="append",
        help="Latent injection placement for local StageWeaver runs. Only append is supported by the current composer-trained mainline.",
    )
    parser.add_argument("--semantic_model_id", type=str, default=DEFAULT_SEMANTIC_MODEL_ID)
    parser.add_argument("--semantic_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--semantic_cache_dir", type=str, default=str(PROJECT_ROOT / ".cache" / "modelscope"))
    parser.add_argument("--semantic_max_length", type=int, default=256)
    parser.add_argument("--retrieval_model", type=str, default="gpt-4.1")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "result" / "stageweaver" / "current" / "eval_ood_test"),
    )
    parser.add_argument("--meta_model", type=str, default=os.getenv("META_MODEL", "qwen3-4b"))
    parser.add_argument("--exec_model", type=str, default=os.getenv("EXEC_MODEL", "qwen3-4b"))
    parser.add_argument(
        "--meta_model_path",
        type=str,
        default=os.getenv("META_MODEL_PATH", ""),
    )
    parser.add_argument(
        "--exec_model_path",
        type=str,
        default=os.getenv("EXEC_MODEL_PATH", ""),
    )
    parser.add_argument("--openai_base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--judge_mode", choices=["llm", "exact_match"], default="llm")
    parser.add_argument("--judge_model", type=str, default=os.getenv("JUDGE_MODEL", os.getenv("EXEC_MODEL", "qwen3-4b")))
    parser.add_argument("--judge_base_url", type=str, default=os.getenv("JUDGE_BASE_URL", os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")))
    parser.add_argument("--judge_api_key", type=str, default=os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")))
    parser.add_argument("--modes", nargs="+", default=["memento_text"])
    parser.add_argument(
        "--memory_mode",
        type=str,
        choices=sorted(ACTIVE_MEMORY_MODES),
        default="",
        help="Single-mode alias for tracker compatibility.",
    )
    parser.add_argument(
        "--stage_mode",
        type=str,
        choices=["both"],
        default="both",
        help="Current StageWeaver protocol requires both planner and executor latent memory.",
    )
    parser.add_argument(
        "--trace_jsonl",
        nargs="?",
        const="__AUTO_TRACE__",
        default="",
        help="Optional trace path override; if provided without a value, auto-uses output_dir/trace_<mode>.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--memory_top_k", type=int, default=8)
    parser.add_argument("--memory_budget_tokens", type=int, default=192)
    parser.add_argument("--bounded_memory_budget_tokens", type=int, default=96)
    parser.add_argument(
        "--executor_memory_refresh",
        choices=["none", "initial", "per_step"],
        default="initial",
        help="Executor memory refresh policy: none disables executor memory, initial preserves the old task-start refresh, per_step refreshes before every executor action.",
    )
    parser.add_argument("--executor_memory_top_k", type=int, default=0, help="Executor memory top-k; 0 reuses --memory_top_k.")
    parser.add_argument("--executor_state_max_chars", type=int, default=4000)
    parser.add_argument("--executor_obs_max_chars", type=int, default=1200)
    parser.add_argument("--executor_tool_history_k", type=int, default=4)
    parser.add_argument(
        "--planner_max_new_tokens",
        type=int,
        default=DEFAULT_GENERATION_HEADROOM,
        help="Planner max_new_tokens cap. Lower this for trace-bank diagnostics or provider TPM limits.",
    )
    parser.add_argument(
        "--executor_max_new_tokens",
        type=int,
        default=DEFAULT_EXECUTOR_GENERATION_HEADROOM,
        help="Executor max_new_tokens cap. Lower this together with planner_max_new_tokens for provider TPM limits.",
    )
    parser.add_argument("--query_timeout_sec", type=int, default=180, help="Per-query timeout in seconds; <=0 disables timeout.")
    parser.add_argument("--resume", action="store_true", help="Append missing indices to an existing results JSONL instead of overwriting it.")
    parser.add_argument("--diagnostic_trace_bank", action="store_true", help="Allow memory_mode=none only for trace-bank construction.")
    parser.add_argument("--tool_profile", choices=["full"], default="full")
    parser.add_argument(
        "--agent_backend",
        choices=["openai", "local"],
        default="openai",
        help="Backend for planner/executor. Use local for direct HF inference with prefix embedding support.",
    )
    args = parser.parse_args()
    if args.memory_mode:
        args.modes = [args.memory_mode]
    _apply_trace_bank_teacher_defaults(args, sys.argv)
    if int(args.planner_max_new_tokens) <= 0:
        raise SystemExit("--planner_max_new_tokens must be > 0.")
    if int(args.executor_max_new_tokens) <= 0:
        raise SystemExit("--executor_max_new_tokens must be > 0.")
    allowed_modes = ACTIVE_MEMORY_MODES
    for mode in args.modes:
        if mode not in allowed_modes:
            raise SystemExit(f"Unsupported mode: {mode}")
    if "none" in args.modes and not args.diagnostic_trace_bank:
        raise SystemExit("memory_mode=none is diagnostic only; pass --diagnostic_trace_bank for trace-bank construction.")
    if any(mode == "stageweaver" for mode in args.modes):
        if args.stage_mode != "both":
            raise SystemExit(
                "Mainline StageWeaver requires --stage_mode both so planner and executor both receive latent prefixes."
            )
        if args.latent_interface != "append":
            raise SystemExit("Mainline StageWeaver requires --latent_interface append.")
        if args.agent_backend != "local":
            raise SystemExit("StageWeaver requires --agent_backend local for latent append support.")
        if not Path(args.stageweaver_composer_ckpt).is_file():
            raise SystemExit(
                "stageweaver mode now requires a valid composer checkpoint: "
                f"{args.stageweaver_composer_ckpt or '[missing --stageweaver_composer_ckpt]'}"
            )
        if str(Path(args.stageweaver_bank_jsonl)) != str(Path(default_stageweaver_bank)):
            raise SystemExit(
                "stageweaver_bank_jsonl must match the training bank to avoid vocab mismatch. "
                f"Use default: {default_stageweaver_bank}"
            )
    return args


def main() -> None:
    load_dotenv()
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
