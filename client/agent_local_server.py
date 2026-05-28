"""
AgentFly - A Hierarchical AI Agent System

This module implements a hierarchical AI system with two main components:
1. META-PLANNER: Breaks down high-level questions into executable tasks
2. EXECUTOR: Executes individual tasks using available tools

The system uses OpenAI models and MCP (Model Context Protocol) for tool integration.
"""

from __future__ import annotations
import asyncio
import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Awaitable, Callable

import torch
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI, AsyncAzureOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
import logging
import colorlog
import tiktoken

# ---------------------------------------------------------------------------
#   Logging setup
# ---------------------------------------------------------------------------
# Configure colored logging for better visibility of log levels
LOG_FORMAT = '%(log_color)s%(levelname)-8s%(reset)s %(message)s'
colorlog.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#   Constants & templates
# ---------------------------------------------------------------------------
# System prompt for the meta-planner agent that breaks down complex problems
META_SYSTEM_PROMPT = (
    "You are the META‑PLANNER in a hierarchical AI system. A user will ask a\n"
    "high‑level question. **First**: break the problem into a *minimal sequence*\n"
    "of executable tasks. Reply ONLY in JSON with the schema:\n"
    "{ \"plan\": [ {\"id\": INT, \"description\": STRING} … ] }\n\n"
    "After each task is executed by the EXECUTOR you will receive its result.\n"
    "Please carefully consider the descriptions of the time of web pages and events in the task, and take these factors into account when planning and giving the final answer.\n"
    "If the final answer is complete, output it with the template:\n"
    "FINAL ANSWER: <answer>\n\n" \
    " YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.\n"
    "Please ensure that the final answer strictly follows the question requirements, without any additional analysis.\n"
    "If the final answer is not complete, emit a *new* JSON plan for the remaining work. Keep cycles as\n"
    "few as possible. Never call tools yourself — that's the EXECUTOR's job."\
    "⚠️  Reply with *pure JSON only*."
)

# System prompt for the executor agent that handles individual tasks
EXEC_SYSTEM_PROMPT = (
    "You are the EXECUTOR sub-agent. You receive one task description at a time\n"
    "from the meta-planner. Your job is to complete the task, using available\n"
    "tools via function calling if needed. Always think step by step but reply\n"
    "with the minimal content needed for the meta-planner. If you must call a\n"
    "tool, produce the appropriate function call instead of natural language.\n"
    "When done, output a concise result. Do NOT output FINAL ANSWER."
)

# Maximum context length for token management
MAX_CTX = 175000
# Default executor model
EXE_MODEL = "qwen3-8b"
DEFAULT_LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "/wufeiyang/mem/M2/model/Qwen3-4B-Instruct-2507")
DEFAULT_LOCAL_8B_MODEL_PATH = os.getenv("QWEN3_8B_MODEL_PATH", "")
DEFAULT_GENERATION_HEADROOM = 15000
_LOCAL_RUNTIME_CACHE: dict[str, "LocalModelRuntime"] = {}

# ---------------------------------------------------------------------------
#   OpenAI backend
# ---------------------------------------------------------------------------
class ChatBackend:
    """Abstract base class for chat backends."""
    async def chat(self, *_, **__) -> Dict[str, Any]:
        raise NotImplementedError

class OpenAIBackend(ChatBackend):
    """OpenAI API backend for chat completions with retry logic."""

    def __init__(self, model: str, is_azure: bool):
        """
        Initialize OpenAI backend with specified model.

        Args:
            model: The OpenAI model to use (e.g., 'gpt-4', 'o3')
        """
        self.model = model
        # Initialize OpenAI client with API key and base URL from environment
        self.client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        ) if not is_azure else AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 15000,
        prefix_embeds: torch.Tensor | None = None,
        prefix_injection_mode: str = "append",
    ) -> Dict[str, Any]:
        """
        Send chat completion request to OpenAI with optional tool calling.

        Args:
            messages: List of message dictionaries with role and content
            tools: Optional list of available tools for function calling
            tool_choice: How to handle tool selection ('auto', 'none', or specific tool)
            max_tokens: Maximum tokens in the response

        Returns:
            Dictionary containing response content and tool calls if any

        Raises:
            Various OpenAI API errors (handled by retry decorator)
        """
        if prefix_embeds is not None:
            raise ValueError("OpenAI-compatible backend does not support custom prefix_embeds.")
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Add tools to payload if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        # Make API call to OpenAI
        resp = await self.client.chat.completions.create(**payload)  # type: ignore[arg-type]
        msg = resp.choices[0].message

        # Extract tool calls if present
        raw_calls = getattr(msg, "tool_calls", None)
        tool_calls = None
        if raw_calls:
            # Convert tool calls to standardized format
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in raw_calls
            ]
        return {"content": msg.content, "tool_calls": tool_calls}

def _resolve_local_model_path(model: str, model_path: str | None = None) -> str:
    candidates: list[str] = []
    if model_path:
        explicit_path = Path(model_path)
        if explicit_path.exists():
            return str(explicit_path.resolve())
        raise FileNotFoundError(f"Explicit model_path does not exist: {model_path}")
    if model:
        candidates.append(model)
    alias = model.lower().strip()
    alias_map = {
        "qwen3-4b": os.getenv("QWEN3_4B_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH),
        "qwen3-4b-instruct": os.getenv("QWEN3_4B_MODEL_PATH", DEFAULT_LOCAL_MODEL_PATH),
        "qwen3-8b": DEFAULT_LOCAL_8B_MODEL_PATH,
        "qwen3-8b-instruct": DEFAULT_LOCAL_8B_MODEL_PATH,
    }
    if alias in alias_map:
        candidates.append(alias_map[alias])
    for cand in candidates:
        if cand and Path(cand).exists():
            return str(Path(cand).resolve())
    raise FileNotFoundError(f"Could not resolve local model path for model={model!r}, model_path={model_path!r}")


def _is_local_model(model: str, model_path: str | None = None) -> bool:
    if model_path:
        _resolve_local_model_path(model, model_path)
        return True
    try:
        _resolve_local_model_path(model, model_path)
        return True
    except FileNotFoundError:
        return False


def _clean_generation_text(text: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    return text


def _extract_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]] | None]:
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if not matches:
        if "<tool_call>" in text:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            tool_name = name_match.group(1) if name_match else "unknown_tool"
            return (
                None,
                [
                    {
                        "id": f"local-tool-partial-{uuid.uuid4()}",
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": text,
                        },
                    }
                ],
            )
        cleaned = _clean_generation_text(text)
        return (cleaned or None), None

    tool_calls: list[dict[str, Any]] = []
    for idx, block in enumerate(matches):
        try:
            payload = json.loads(block)
            tool_name = str(payload["name"])
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                args_json = arguments
            else:
                args_json = json.dumps(arguments, ensure_ascii=False)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', block)
            args_match = re.search(r'"arguments"\s*:\s*(.+)\s*$', block, flags=re.DOTALL)
            tool_name = name_match.group(1) if name_match else "unknown_tool"
            args_json = args_match.group(1).strip() if args_match else block
        tool_calls.append(
            {
                "id": f"local-tool-{idx}-{uuid.uuid4()}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": args_json,
                },
            }
        )
    return None, tool_calls


class LocalModelRuntime:
    def __init__(self, model_ref: str, dtype_name: str = "bfloat16") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_ref = model_ref
        self.tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(dtype_name, torch.bfloat16)
        device_map: str | dict[str, int] = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_ref,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.hidden_size = int(self.model.config.hidden_size)
        self.device = self.model.get_input_embeddings().weight.device


def _get_local_runtime(model: str, model_path: str | None = None) -> LocalModelRuntime:
    resolved = _resolve_local_model_path(model, model_path)
    runtime = _LOCAL_RUNTIME_CACHE.get(resolved)
    if runtime is None:
        runtime = LocalModelRuntime(resolved, dtype_name=os.getenv("LOCAL_MODEL_DTYPE", "bfloat16"))
        _LOCAL_RUNTIME_CACHE[resolved] = runtime
    return runtime


class DirectModelBackend(ChatBackend):
    """Direct local Transformers backend with optional prefix embedding injection."""

    def __init__(
        self,
        model: str = "qwen3-8b",
        model_path: str | None = None,
        default_search_tool: bool = True,
    ) -> None:
        self.model = model
        self.model_path = _resolve_local_model_path(model, model_path)
        self.runtime = _get_local_runtime(model, self.model_path)
        self.hidden_size = self.runtime.hidden_size
        self.supports_prefix_embeds = True
        self.default_search_tool = default_search_tool

        self._fallback_tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Get external knowledge using search engine",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                },
            }
        ]

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]] | None = None,
        tool_choice: str | None = "auto",
        max_tokens: int = 10240,
        prefix_embeds: torch.Tensor | None = None,
        prefix_injection_mode: str = "append",
    ) -> Dict[str, Any]:
        tools_for_prompt = None
        if tool_choice != "none":
            if tools and len(tools) > 0:
                tools_for_prompt = tools
            elif self.default_search_tool:
                tools_for_prompt = self._fallback_tools

        rendered = self.runtime.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools_for_prompt,
        )
        enc = self.runtime.tokenizer(rendered, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.runtime.device)
        attention_mask = enc["attention_mask"].to(self.runtime.device)

        generate_kwargs: Dict[str, Any] = {
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": self.runtime.tokenizer.pad_token_id,
            "eos_token_id": self.runtime.tokenizer.eos_token_id,
        }
        generate_max_time = os.getenv("LOCAL_GENERATE_MAX_TIME_SEC", "").strip()
        if generate_max_time:
            try:
                max_time_sec = float(generate_max_time)
            except ValueError:
                max_time_sec = 0.0
            if max_time_sec > 0:
                # Bound local HF generation internally so outer per-query timeouts are not defeated by a blocking generate().
                generate_kwargs["max_time"] = max_time_sec
        if prefix_embeds is not None:
            if prefix_embeds.dim() != 3:
                raise ValueError(f"prefix_embeds must have shape [batch, prefix_tokens, hidden_size], got {tuple(prefix_embeds.shape)}")
            prefix_embeds = prefix_embeds.to(device=self.runtime.device, dtype=self.runtime.model.get_input_embeddings().weight.dtype)
            if prefix_embeds.size(-1) != self.runtime.hidden_size:
                raise ValueError(
                    f"prefix hidden size mismatch: expected {self.runtime.hidden_size}, got {prefix_embeds.size(-1)}"
                )
            text_embeds = self.runtime.model.get_input_embeddings()(input_ids)
            if text_embeds.size(1) == 0:
                raise ValueError("cannot insert prefix_embeds into an empty prompt")
            prefix_mask = torch.ones(
                (attention_mask.size(0), prefix_embeds.size(1)),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            if prefix_injection_mode == "append":
                # LatentMem-style path: append the latent memory after the prompt hidden states.
                inputs_embeds = torch.cat([text_embeds, prefix_embeds], dim=1)
                generate_kwargs["attention_mask"] = torch.cat([attention_mask, prefix_mask], dim=1)
            else:
                raise ValueError(f"Unsupported prefix_injection_mode: {prefix_injection_mode}")
            generate_kwargs["inputs_embeds"] = inputs_embeds
        else:
            generate_kwargs["input_ids"] = input_ids

        with torch.inference_mode():
            output_ids = self.runtime.model.generate(**generate_kwargs)
        if prefix_embeds is None:
            generated_ids = output_ids[:, input_ids.size(1) :]
        else:
            generated_ids = output_ids
        text = self.runtime.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        content, tool_calls = _extract_tool_calls(text)
        return {"content": content, "tool_calls": tool_calls}

# ---------------------------------------------------------------------------
#   Hierarchical client (trimmed: only essentials kept)
# ---------------------------------------------------------------------------
# Maximum number of conversation turns to keep in memory
MAX_TURNS_MEMORY = 50

def _strip_fences(text: str) -> str:
    """
    Remove markdown code fences and extract JSON content.

    Args:
        text: Text that may contain markdown fences or JSON

    Returns:
        Cleaned text with fences removed
    """
    import re
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n", "", text)
        text = re.sub(r"\n?```$", "", text)
        return text.strip()
    # Extract the first JSON object when the model adds a short natural-language preamble.
    m = re.search(r"\{[\s\S]*\}", text)
    return m.group(0) if m else text


def _fallback_answer_from_trace(trace: Dict[str, Any]) -> str:
    def _strip_task_result_prefix(text: str) -> str:
        cleaned = text.strip()
        while True:
            updated = re.sub(r"^Task\s+\d+\s+result:\s*", "", cleaned, flags=re.IGNORECASE).strip()
            if updated == cleaned:
                return cleaned
            cleaned = updated

    def _extract_structured_result(text: str) -> str:
        text = _strip_task_result_prefix(text)
        if not text.startswith("{"):
            return text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(payload, dict):
            for key in ("result", "answer", "final_answer", "content"):
                candidate = str(payload.get(key, "")).strip()
                if candidate:
                    return candidate
        return text

    for cycle in reversed(trace.get("cycles", [])):
        for task in reversed(cycle.get("tasks", [])):
            raw_result = _strip_task_result_prefix(str(task.get("result", "")).strip())
            if not raw_result:
                continue
            if raw_result.startswith("[TOOL_CALL]"):
                continue
            if raw_result.startswith("[RETURN]"):
                candidate = _extract_structured_result(raw_result[len("[RETURN]") :].strip())
                candidate = _strip_task_result_prefix(candidate)
                if candidate.startswith("[TOOL_CALL]"):
                    continue
                if candidate:
                    return candidate
                continue
            if raw_result.startswith("{"):
                candidate = _extract_structured_result(raw_result)
                candidate = _strip_task_result_prefix(candidate)
                if candidate.startswith("[TOOL_CALL]"):
                    continue
                if candidate:
                    return candidate
            lowered = raw_result.lower()
            if "error" in lowered and "result" not in lowered:
                continue
            return raw_result
    return ""


def _extract_planner_final_answer(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if cleaned.startswith("FINAL ANSWER:"):
        return cleaned[len("FINAL ANSWER:") :].strip()
    cleaned = re.sub(r"^Task\s+\d+\s+result:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if cleaned.startswith("[RETURN]"):
        cleaned = cleaned[len("[RETURN]") :].strip()
        cleaned = re.sub(r"^Task\s+\d+\s+result:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    if not cleaned:
        return ""
    if cleaned.startswith("{"):
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return ""
        if isinstance(payload, dict):
            if "plan" in payload:
                return ""
            for key in ("result", "answer", "final_answer", "content"):
                candidate = str(payload.get(key, "")).strip()
                if candidate:
                    return candidate
        return ""
    if cleaned.startswith("[TOOL_CALL]"):
        return ""
    return cleaned


def _parse_executor_tool_call_text(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned.startswith("[TOOL_CALL]"):
        return None
    payload = cleaned[len("[TOOL_CALL]") :].strip()
    match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)\s*$", payload, flags=re.DOTALL)
    if not match:
        return None
    tool_name = match.group(1)
    raw_arguments = match.group(2).strip() or "{}"
    try:
        parsed_arguments = json.loads(raw_arguments)
        if isinstance(parsed_arguments, dict):
            raw_arguments = json.dumps(parsed_arguments, ensure_ascii=False)
    except json.JSONDecodeError:
        pass
    return {
        "id": f"executor-pseudo-tool-{uuid.uuid4()}",
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": raw_arguments,
        },
    }

def _count_tokens(msg: Dict[str, str], enc) -> int:
    """
    Count tokens in a message for context management.

    Args:
        msg: Message dictionary with role and content
        enc: Tokenizer encoding object

    Returns:
        Number of tokens in the message
    """
    role_tokens = 4  # OpenAI adds 4 tokens for role
    content = msg.get("content") or ""
    return role_tokens + len(enc.encode(content))

def _get_tokenizer(model: str):
    """
    Return a tokenizer for the specified model.

    Args:
        model: Model name to get tokenizer for

    Returns:
        Tokenizer encoding object, falls back to cl100k_base if model unknown
    """
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def trim_messages(messages: List[Dict[str, str]], max_tokens: int, model="gpt-3.5-turbo") -> List[Dict[str, str]]:
    """
    Trim message history to fit within token limit while preserving system message.

    Args:
        messages: List of message dictionaries
        max_tokens: Maximum allowed tokens
        model: Model name for token counting

    Returns:
        Trimmed list of messages that fit within token limit
    """
    enc = _get_tokenizer(model)
    total = sum(_count_tokens(m, enc) for m in messages) + 2

    # If already within limit, return as is
    if total <= max_tokens:
        return messages

    # Always keep system message (first message)
    system_msg = messages[0]
    kept: List[Dict[str, str]] = [system_msg]
    total = _count_tokens(system_msg, enc) + 2

    # Add messages from most recent to oldest until limit is reached
    for msg in reversed(messages[1:]):
        t = _count_tokens(msg, enc)
        if total + t > max_tokens:
            break
        kept.insert(1, msg)  # Insert after system message
        total += t
    return kept


def trim_messages_with_pinned_prefix(
    messages: List[Dict[str, str]],
    max_tokens: int,
    pinned_prefix_count: int,
    model="gpt-3.5-turbo",
) -> List[Dict[str, str]]:
    def _truncate_message(msg: Dict[str, str], budget: int) -> Dict[str, str] | None:
        if budget <= 4:
            return None
        content = msg.get("content") or ""
        toks = enc.encode(content)
        clipped = enc.decode(toks[: max(0, budget - 4)]).strip()
        candidate = {**msg, "content": clipped}
        if _count_tokens(candidate, enc) > budget:
            return None
        return candidate

    enc = _get_tokenizer(model)
    total = sum(_count_tokens(m, enc) for m in messages) + 2
    if total <= max_tokens:
        return messages

    system_msg = messages[0]
    pinned = messages[1 : 1 + max(pinned_prefix_count, 0)]
    rest = messages[1 + max(pinned_prefix_count, 0) :]
    kept_recent: List[Dict[str, str]] = []
    total = _count_tokens(system_msg, enc) + sum(_count_tokens(m, enc) for m in pinned) + 2
    if total > max_tokens:
        available = max_tokens - _count_tokens(system_msg, enc) - 2
        if available <= 0:
            return [system_msg]
        trimmed_pinned: List[Dict[str, str]] = []
        for msg in pinned:
            candidate = _truncate_message(msg, available)
            if candidate is None:
                break
            trimmed_pinned.append(candidate)
            available -= _count_tokens(candidate, enc)
            if available <= 0:
                break
        return [system_msg, *trimmed_pinned]

    for msg in reversed(rest):
        t = _count_tokens(msg, enc)
        if total + t > max_tokens:
            break
        kept_recent.insert(0, msg)
        total += t
    return [system_msg, *pinned, *kept_recent]


def _is_task_result_message(msg: Dict[str, str]) -> bool:
    if msg.get("role") != "assistant":
        return False
    content = str(msg.get("content") or "").strip()
    return bool(re.match(r"^Task\s+\d+\s+result:\s*", content, flags=re.IGNORECASE))


def _history_without_duplicated_task_results(shared_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not shared_history:
        return []
    deduped: List[Dict[str, str]] = [shared_history[0]]
    for msg in shared_history[1:]:
        if _is_task_result_message(msg):
            continue
        deduped.append(msg)
    return deduped

class HierarchicalClient:
    """
    Main client class that orchestrates the hierarchical AI system.

    Manages communication between meta-planner and executor agents,
    handles tool connections, and processes user queries through
    multiple planning and execution cycles.
    """

    # Maximum number of planning cycles before giving up
    MAX_CYCLES = 3
    TRACE_RESULT_CHARS = 4000

    def __init__(
        self,
        meta_model: str,
        exec_model: str,
        is_azure: bool,
        trace_jsonl: str | None = None,
        meta_model_path: str | None = None,
        exec_model_path: str | None = None,
        prefer_local_backend: bool = False,
        backend_mode: str = "auto",
    ):
        """
        Initialize the hierarchical client.

        Args:
            meta_model: Model name for the meta-planner agent
            exec_model: Model name for the executor agent
        """
        if backend_mode == "local":
            _resolve_local_model_path(meta_model, meta_model_path)
            self.meta_llm = DirectModelBackend(model=meta_model, model_path=meta_model_path, default_search_tool=False)
        elif backend_mode == "openai":
            self.meta_llm = OpenAIBackend(meta_model, is_azure)
        else:
            use_local_meta = bool(meta_model_path) or (prefer_local_backend and _is_local_model(meta_model))
            if use_local_meta:
                self.meta_llm = DirectModelBackend(model=meta_model, model_path=meta_model_path, default_search_tool=False)
            else:
                self.meta_llm = OpenAIBackend(meta_model, is_azure)

        if backend_mode == "local":
            _resolve_local_model_path(exec_model, exec_model_path)
            self.exec_llm = DirectModelBackend(model=exec_model, model_path=exec_model_path)
        elif backend_mode == "openai":
            self.exec_llm = OpenAIBackend(exec_model, is_azure)
        else:
            use_local_exec = bool(exec_model_path) or (prefer_local_backend and _is_local_model(exec_model))
            if use_local_exec:
                self.exec_llm = DirectModelBackend(model=exec_model, model_path=exec_model_path)
            else:
                self.exec_llm = OpenAIBackend(exec_model, is_azure)

        self.exec_model = exec_model
        self.trace_jsonl = trace_jsonl
        self.sessions: Dict[str, ClientSession] = {}
        self.shared_history: List[Dict[str, str]] = []

    def _write_trace(self, trace: Dict[str, Any]) -> None:
        if not self.trace_jsonl:
            return

        path = Path(self.trace_jsonl).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def _short_result(self, text: str) -> str:
        if len(text) <= self.TRACE_RESULT_CHARS:
            return text
        omitted = len(text) - self.TRACE_RESULT_CHARS
        return text[: self.TRACE_RESULT_CHARS] + f"\n...[truncated {omitted} chars]"

    def _resolve_tool_name(self, requested: str) -> str:
        if requested in self.sessions:
            return requested

        for name in self.sessions.keys():
            if requested in name or name in requested:
                return name

        raise KeyError(f"No matching tool for '{requested}'. Available: {list(self.sessions.keys())}")

    def _massage_args_for_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        patched = dict(args)
        ln = tool_name.lower()
        return patched

    # ---------- Tool management ----------
    async def connect_to_servers(self, scripts: List[str]):
        """
        Connect to MCP tool servers specified by script paths.

        Args:
            scripts: List of paths to tool server scripts

        Raises:
            RuntimeError: If duplicate tool names are found
        """
        from contextlib import AsyncExitStack
        self.exit_stack = AsyncExitStack()

        for script in scripts:
            path = Path(script)
            # Determine command based on file extension
            cmd = sys.executable if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])

            # Create stdio client and session
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()

            # Register tools from this session
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session

        print("Connected tools:", list(self.sessions.keys()))

    async def _tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get the schema for all available tools in a format suitable for OpenAI.

        Returns:
            List of tool schemas in OpenAI function calling format
        """
        result, cached = [], {}
        for session in self.sessions.values():
            # Cache tool listings to avoid repeated calls
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp

            # Convert MCP tool format to OpenAI function calling format
            for tool in tools_resp.tools:
                result.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return result

    # ---------- Main processing ----------
    async def process_query(
        self,
        query: str,
        file: str,
        task_id: str = "interactive",
        planner_memory_prompt: str = "",
        executor_memory_prompt: str = "",
        planner_memory_role: str = "system",
        planner_prefix_embeds: torch.Tensor | None = None,
        executor_prefix_embeds: torch.Tensor | None = None,
        planner_prefix_injection_mode: str = "append",
        executor_prefix_injection_mode: str = "append",
        executor_memory_refresh: str = "initial",
        planner_max_new_tokens: int = DEFAULT_GENERATION_HEADROOM,
        planner_memory_callback: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]] | None = None,
        executor_memory_callback: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]] | None = None,
    ) -> str:
        """
        Process a user query through the hierarchical AI system.

        This is the main method that:
        1. Gets the meta-planner to break down the query into tasks
        2. Executes each task using the executor agent
        3. Continues planning cycles until a final answer is reached

        Args:
            query: User's question or request
            file: Optional file path context
            task_id: Unique identifier for this query session

        Returns:
            Final answer to the user's query
        """
        tools_schema = await self._tools_schema()
        self.shared_history = []
        trace: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "task_id": task_id,
            "question": query,
            "file_path": file,
            "meta_model": getattr(self.meta_llm, "model", None),
            "exec_model": self.exec_model,
            "connected_tools": list(self.sessions.keys()),
            "cycles": [],
            "final_answer": None,
            "error": None,
        }

        # Initialize conversation with user query
        self.shared_history.append({
            "role": "user",
            "content": f"{query}\ntask_id: {task_id}\nfile_path: {file}\n"
        })
        planner_model_name = getattr(self.meta_llm, "model", "gpt-3.5-turbo")
        planner_prompt_budget = max(1024, MAX_CTX - max(1, int(planner_max_new_tokens)))
        planner_enc = _get_tokenizer(planner_model_name)
        planner_system_tokens = _count_tokens({"role": "system", "content": META_SYSTEM_PROMPT}, planner_enc)
        planner_query_tokens = _count_tokens(self.shared_history[0], planner_enc)
        planner_followup_headroom = 4096 if planner_memory_role == "user" else 0
        planner_memory_budget = max(
            512,
            planner_prompt_budget - planner_system_tokens - planner_query_tokens - planner_followup_headroom - 2,
        )

        def _trim_prompt_tokens(text: str, model_name: str, max_tokens: int = 1024) -> str:
            if not text.strip():
                return ""
            enc = _get_tokenizer(model_name)
            toks = enc.encode(text)
            if len(toks) <= max_tokens:
                return text
            return enc.decode(toks[:max_tokens]).strip()

        static_planner_memory_compact = _trim_prompt_tokens(
            planner_memory_prompt,
            planner_model_name,
            max_tokens=planner_memory_budget if planner_memory_role == "user" else 1024,
        )
        static_executor_memory_compact = _trim_prompt_tokens(
            executor_memory_prompt,
            self.exec_model,
            max_tokens=1024,
        )

        def _build_planner_messages(current_memory_compact: str, current_memory_role: str) -> List[Dict[str, str]]:
            planner_system_prompt = (
                META_SYSTEM_PROMPT
                if not current_memory_compact or current_memory_role == "user"
                else f"{META_SYSTEM_PROMPT}\n\n[PLANNER_MEMORY]\n{current_memory_compact}"
            )
            base: List[Dict[str, str]] = [{"role": "system", "content": planner_system_prompt}]
            if current_memory_compact and current_memory_role == "user" and self.shared_history:
                base.append(self.shared_history[0])
                base.append({"role": "user", "content": current_memory_compact})
                base.extend(self.shared_history[1:])
            else:
                base.extend(self.shared_history)
            return base

        # Main planning and execution loop
        for cycle in range(self.MAX_CYCLES):
            cycle_trace: Dict[str, Any] = {
                "cycle": cycle,
                "planner_output": None,
                "tasks": [],
            }
            trace["cycles"].append(cycle_trace)

            runtime_planner_memory_role = planner_memory_role
            runtime_planner_memory_compact = static_planner_memory_compact
            runtime_planner_prefix_embeds = planner_prefix_embeds
            runtime_planner_prefix_injection_mode = planner_prefix_injection_mode
            if planner_memory_callback is not None:
                planner_bundle = await planner_memory_callback(
                    {
                        "query": query,
                        "file": file,
                        "task_id": task_id,
                        "cycle": cycle,
                        "shared_history": list(self.shared_history),
                        "trace": trace,
                        "tools_schema": tools_schema,
                    }
                )
                runtime_planner_memory_role = str(planner_bundle.get("memory_role", planner_memory_role))
                runtime_planner_memory_compact = _trim_prompt_tokens(
                    str(planner_bundle.get("text_memory", "")),
                    planner_model_name,
                    max_tokens=planner_memory_budget if runtime_planner_memory_role == "user" else 1024,
                )
                runtime_planner_prefix_embeds = planner_bundle.get("prefix_embeds", planner_prefix_embeds)
                runtime_planner_prefix_injection_mode = str(
                    planner_bundle.get("prefix_injection_mode", planner_prefix_injection_mode)
                )
                cycle_trace["planner_memory"] = dict(planner_bundle.get("trace", {}))
            elif runtime_planner_memory_compact:
                cycle_trace["planner_memory"] = {
                    "mode": "static",
                    "memory_role": runtime_planner_memory_role,
                    "text_memory_chars": len(runtime_planner_memory_compact),
                }

            planner_msgs = _build_planner_messages(runtime_planner_memory_compact, runtime_planner_memory_role)

            # Get plan from meta-planner
            pinned_prefix_count = 0
            if self.shared_history:
                pinned_prefix_count = 2 if runtime_planner_memory_compact and runtime_planner_memory_role == "user" else 1
            planner_msgs = trim_messages_with_pinned_prefix(
                planner_msgs,
                planner_prompt_budget,
                pinned_prefix_count=pinned_prefix_count,
                model=planner_model_name,
            )
            meta_reply = await self.meta_llm.chat(
                planner_msgs,
                max_tokens=planner_max_new_tokens,
                prefix_embeds=runtime_planner_prefix_embeds,
                prefix_injection_mode=runtime_planner_prefix_injection_mode,
            )
            meta_content = meta_reply["content"] or ""
            cycle_trace["planner_output"] = meta_content
            self.shared_history.append({"role": "assistant", "content": meta_content})

            # Check if we have a final answer
            planner_answer = _extract_planner_final_answer(meta_content)
            if planner_answer:
                answer = planner_answer
                trace["final_answer"] = answer
                self._write_trace(trace)
                return answer

            # Parse the plan from meta-planner's response
            try:
                tasks = json.loads(_strip_fences(meta_content))["plan"]
            except Exception as e:
                error = f"[planner error] {e}: {meta_content}"
                trace["error"] = error
                self._write_trace(trace)
                return error

            # Execute each task in the plan
            for task in tasks:
                task_desc = f"Task {task['id']}: {task['description']}"
                task_trace: Dict[str, Any] = {
                    "task": task,
                    "tool_calls": [],
                    "result": None,
                }
                cycle_trace["tasks"].append(task_trace)
                runtime_executor_memory_compact = static_executor_memory_compact
                runtime_executor_prefix_embeds = executor_prefix_embeds
                runtime_executor_prefix_injection_mode = executor_prefix_injection_mode
                if executor_memory_callback is not None and executor_memory_refresh != "per_step":
                    executor_bundle = await executor_memory_callback(
                        {
                            "query": query,
                            "file": file,
                            "task_id": task_id,
                            "cycle": cycle,
                            "task": dict(task),
                            "task_trace": task_trace,
                            "executor_step_id": 0,
                            "shared_history": list(self.shared_history),
                            "trace": trace,
                            "tools_schema": tools_schema,
                        }
                    )
                    runtime_executor_memory_compact = _trim_prompt_tokens(
                        str(executor_bundle.get("text_memory", "")),
                        self.exec_model,
                        max_tokens=1024,
                    )
                    runtime_executor_prefix_embeds = executor_bundle.get("prefix_embeds", executor_prefix_embeds)
                    runtime_executor_prefix_injection_mode = str(
                        executor_bundle.get("prefix_injection_mode", executor_prefix_injection_mode)
                    )
                    task_trace["executor_memory"] = dict(executor_bundle.get("trace", {}))
                elif runtime_executor_memory_compact:
                    task_trace["executor_memory"] = {
                        "mode": "static",
                        "text_memory_chars": len(runtime_executor_memory_compact),
                    }
                exec_system_prompt = (
                    EXEC_SYSTEM_PROMPT
                    if not runtime_executor_memory_compact
                    else f"{EXEC_SYSTEM_PROMPT}\n\n[EXECUTOR_MEMORY]\n{runtime_executor_memory_compact}"
                )
                executor_history = (
                    _history_without_duplicated_task_results(self.shared_history)
                    if runtime_executor_memory_compact
                    else self.shared_history
                )
                exec_msgs = (
                    [{"role": "system", "content": exec_system_prompt}]
                    + executor_history
                    + [{"role": "user", "content": task_desc}]
                )

                # Retry malformed tool-call JSON a few times before surfacing an error.
                malformed_tool_retry_count = 0
                max_malformed_tool_retries = 3
                executed_tool_results: Dict[str, str] = {}
                duplicate_tool_call_counts: Dict[str, int] = {}
                executor_step_id = 0

                def _mark_latest_executor_memory_step(status: str) -> None:
                    if task_trace.get("executor_memory_steps"):
                        latest_step = task_trace["executor_memory_steps"][-1]
                    else:
                        latest_step = task_trace.get("executor_memory")
                    if isinstance(latest_step, dict):
                        latest_step["tool_success_or_error"] = status

                # Execute task with potential tool calls
                while True:
                    if executor_memory_callback is not None and executor_memory_refresh == "per_step":
                        executor_bundle = await executor_memory_callback(
                            {
                                "query": query,
                                "file": file,
                                "task_id": task_id,
                                "cycle": cycle,
                                "task": dict(task),
                                "task_trace": task_trace,
                                "executor_step_id": executor_step_id,
                                "shared_history": list(self.shared_history),
                                "trace": trace,
                                "tools_schema": tools_schema,
                            }
                        )
                        runtime_executor_memory_compact = _trim_prompt_tokens(
                            str(executor_bundle.get("text_memory", "")),
                            self.exec_model,
                            max_tokens=1024,
                        )
                        runtime_executor_prefix_embeds = executor_bundle.get("prefix_embeds", executor_prefix_embeds)
                        runtime_executor_prefix_injection_mode = str(
                            executor_bundle.get("prefix_injection_mode", executor_prefix_injection_mode)
                        )
                        memory_trace = dict(executor_bundle.get("trace", {}))
                        task_trace.setdefault("executor_memory_steps", []).append(memory_trace)
                        task_trace["executor_memory"] = memory_trace
                        exec_system_prompt = (
                            EXEC_SYSTEM_PROMPT
                            if not runtime_executor_memory_compact
                            else f"{EXEC_SYSTEM_PROMPT}\n\n[EXECUTOR_MEMORY]\n{runtime_executor_memory_compact}"
                        )
                        if exec_msgs:
                            exec_msgs[0] = {"role": "system", "content": exec_system_prompt}
                    # Trim messages to fit within token limit
                    exec_msgs = trim_messages(exec_msgs, MAX_CTX, model=EXE_MODEL)
                    exec_reply = await self.exec_llm.chat(
                        exec_msgs,
                        tools_schema,
                        max_tokens=10240,
                        prefix_embeds=runtime_executor_prefix_embeds,
                        prefix_injection_mode=runtime_executor_prefix_injection_mode,
                    )
                    current_memory_step = None
                    if task_trace.get("executor_memory_steps"):
                        current_memory_step = task_trace["executor_memory_steps"][-1]
                    elif task_trace.get("executor_memory"):
                        current_memory_step = task_trace["executor_memory"]
                    if isinstance(current_memory_step, dict):
                        tool_calls_for_log = exec_reply.get("tool_calls") or []
                        first_tool = tool_calls_for_log[0] if tool_calls_for_log else None
                        raw_args = ""
                        tool_name = ""
                        if first_tool:
                            tool_name = str(first_tool.get("function", {}).get("name", ""))
                            raw_args = str(first_tool.get("function", {}).get("arguments") or "")
                        current_memory_step["generated_tool_call"] = bool(first_tool)
                        current_memory_step["tool_name"] = tool_name
                        current_memory_step["tool_query_or_args_hash"] = (
                            hashlib.sha256(raw_args.encode("utf-8")).hexdigest()[:16] if raw_args else ""
                        )
                    executor_step_id += 1

                    # If executor has a direct response, use it
                    synthetic_tool_call = None
                    if exec_reply.get("content") and not (exec_reply.get("tool_calls") or []):
                        synthetic_tool_call = _parse_executor_tool_call_text(str(exec_reply["content"]))
                        if synthetic_tool_call is not None:
                            exec_reply["content"] = None
                            exec_reply["tool_calls"] = [synthetic_tool_call]
                        elif str(exec_reply["content"]).strip().startswith("[TOOL_CALL]"):
                            malformed_tool_retry_count += 1
                            error_msg = (
                                "[tool call format error] Executor emitted a textual [TOOL_CALL] reply "
                                "that could not be parsed. Re-emit the same tool call as name({...}) "
                                "with JSON object arguments only."
                            )
                            task_trace["tool_calls"].append({
                                "requested_name": None,
                                "resolved_name": None,
                                "arguments_raw": str(exec_reply["content"]),
                                "error": error_msg,
                            })
                            exec_msgs.extend([
                                {"role": "assistant", "content": str(exec_reply["content"])} ,
                                {"role": "user", "content": error_msg},
                            ])
                            if malformed_tool_retry_count >= max_malformed_tool_retries:
                                final_error = (
                                    "Executor failed to emit a parseable textual tool call "
                                    f"after {max_malformed_tool_retries} attempts."
                                )
                                task_trace["result"] = final_error
                                self.shared_history.append({
                                    "role": "assistant",
                                    "content": f"Task {task['id']} result: {final_error}"
                                })
                                break
                            continue

                    if exec_reply["content"]:
                        result_text = str(exec_reply["content"])
                        _mark_latest_executor_memory_step("direct_text_result")
                        task_trace["result"] = result_text
                        history_result = re.sub(
                            r"^Task\s+\d+\s+result:\s*",
                            "",
                            result_text.strip(),
                            flags=re.IGNORECASE,
                        ).strip()
                        self.shared_history.append({
                            "role": "assistant",
                            "content": f"Task {task['id']} result: {history_result or result_text.strip()}"
                        })
                        break

                    # Handle tool calls from executor
                    retry_executor = False
                    for call in exec_reply.get("tool_calls") or []:
                        t_name = call["function"]["name"]
                        raw_args = call["function"].get("arguments") or "{}"
                        try:
                            t_args = json.loads(raw_args)
                        except json.JSONDecodeError as e:
                            _mark_latest_executor_memory_step("tool_arguments_json_error")
                            malformed_tool_retry_count += 1
                            error_msg = (
                                f"[tool arguments JSON error] Tool '{t_name}' returned invalid JSON "
                                f"arguments ({e}). Re-emit the same tool call with valid JSON only."
                            )
                            task_trace["tool_calls"].append({
                                "requested_name": t_name,
                                "resolved_name": None,
                                "arguments_raw": raw_args,
                                "error": error_msg,
                            })
                            exec_msgs.extend([
                                {"role": "assistant", "content": None, "tool_calls": [call]},
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id", str(uuid.uuid4())),
                                    "name": t_name,
                                    "content": error_msg,
                                },
                            ])

                            if malformed_tool_retry_count >= max_malformed_tool_retries:
                                final_error = (
                                    f"Executor failed to emit valid JSON arguments for tool '{t_name}' "
                                    f"after {max_malformed_tool_retries} attempts."
                                )
                                task_trace["result"] = final_error
                                self.shared_history.append({
                                    "role": "assistant",
                                    "content": f"Task {task['id']} result: {final_error}"
                                })
                            else:
                                retry_executor = True
                            break

                        if not isinstance(t_args, dict):
                            _mark_latest_executor_memory_step("tool_arguments_type_error")
                            malformed_tool_retry_count += 1
                            error_msg = (
                                f"[tool arguments type error] Tool '{t_name}' returned non-object JSON "
                                "arguments. Re-emit the same tool call with a JSON object only."
                            )
                            task_trace["tool_calls"].append({
                                "requested_name": t_name,
                                "resolved_name": None,
                                "arguments_raw": raw_args,
                                "error": error_msg,
                            })
                            exec_msgs.extend([
                                {"role": "assistant", "content": None, "tool_calls": [call]},
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id", str(uuid.uuid4())),
                                    "name": t_name,
                                    "content": error_msg,
                                },
                            ])

                            if malformed_tool_retry_count >= max_malformed_tool_retries:
                                final_error = (
                                    f"Executor failed to emit JSON object arguments for tool '{t_name}' "
                                    f"after {max_malformed_tool_retries} attempts."
                                )
                                task_trace["result"] = final_error
                                self.shared_history.append({
                                    "role": "assistant",
                                    "content": f"Task {task['id']} result: {final_error}"
                                })
                            else:
                                retry_executor = True
                            break

                        try:
                            resolved = self._resolve_tool_name(t_name)
                        except KeyError as e:
                            _mark_latest_executor_memory_step("tool_resolution_error")
                            error_msg = f"[tool resolution error] {e}"
                            task_trace["tool_calls"].append({
                                "requested_name": t_name,
                                "resolved_name": None,
                                "arguments": t_args,
                                "error": error_msg,
                            })
                            exec_msgs.extend([
                                {"role": "assistant", "content": None, "tool_calls": [call]},
                                {"role": "tool", "tool_call_id": call.get("id", str(uuid.uuid4())), "name": t_name, "content": error_msg},
                            ])
                            continue

                        session = self.sessions[resolved]
                        patched_args = self._massage_args_for_tool(resolved, t_args)
                        tool_signature = json.dumps(
                            {"tool": resolved, "arguments": patched_args},
                            ensure_ascii=False,
                            sort_keys=True,
                        )

                        if tool_signature in executed_tool_results:
                            _mark_latest_executor_memory_step("reused_previous_tool_result")
                            duplicate_tool_call_counts[tool_signature] = duplicate_tool_call_counts.get(tool_signature, 0) + 1
                            cached_result = executed_tool_results[tool_signature]
                            guarded_result = (
                                f"[tool repetition guard] Tool '{resolved}' with identical arguments was already "
                                "executed earlier in this task. Do not repeat it. Use the cached result below "
                                "to continue.\n\n"
                                f"{cached_result}"
                            )
                            task_trace["tool_calls"].append({
                                "requested_name": t_name,
                                "resolved_name": resolved,
                                "arguments": patched_args,
                                "result_preview": self._short_result(cached_result),
                                "result_chars": len(cached_result),
                                "reused_result": True,
                                "repeat_count": duplicate_tool_call_counts[tool_signature],
                            })

                            exec_msgs.extend([
                                {"role": "assistant", "content": None, "tool_calls": [call]},
                                {
                                    "role": "tool",
                                    "tool_call_id": call.get("id", str(uuid.uuid4())),
                                    "name": resolved,
                                    "content": guarded_result,
                                },
                            ])
                            continue

                        result_msg = await session.call_tool(resolved, patched_args)
                        _mark_latest_executor_memory_step("tool_success")
                        result_text = str(result_msg.content)
                        executed_tool_results[tool_signature] = result_text
                        task_trace["tool_calls"].append({
                            "requested_name": t_name,
                            "resolved_name": resolved,
                            "arguments": patched_args,
                            "result_preview": self._short_result(result_text),
                            "result_chars": len(result_text),
                        })

                        exec_msgs.extend([
                            {"role": "assistant", "content": None, "tool_calls": [call]},
                            {
                                "role": "tool",
                                "tool_call_id": call.get("id", str(uuid.uuid4())),
                                "name": resolved,
                                "content": result_text
                            },
                        ])

                    if not retry_executor and (exec_reply.get("tool_calls") or []):
                        malformed_tool_retry_count = 0

                    if retry_executor:
                        continue
                    if not (exec_reply.get("tool_calls") or []):
                        _mark_latest_executor_memory_step("empty_executor_reply")
                        empty_error = "Executor returned neither content nor tool calls."
                        task_trace["result"] = empty_error
                        self.shared_history.append({
                            "role": "assistant",
                            "content": f"Task {task['id']} result: {empty_error}"
                        })
                        break
                    if task_trace["result"] is not None:
                        break

        # If the planner never emits FINAL ANSWER, prefer the last grounded task result
        # over recycling the last plan JSON as the final answer.
        fallback_answer = _fallback_answer_from_trace(trace)
        trace["final_answer"] = fallback_answer or meta_content.strip()
        self._write_trace(trace)
        return trace["final_answer"]

    async def cleanup(self):
        """Clean up resources and close tool server connections."""
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()

# ---------------------------------------------------------------------------
#   Command‑line & main routine
# ---------------------------------------------------------------------------

def parse_args():
    """
    Parse command line arguments for the AgentFly client.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="AgentFly – interactive version")
    parser.add_argument("-q", "--question", type=str, help="Your question")
    parser.add_argument("-f", "--file", type=str, default="", help="Optional file path")
    parser.add_argument("-m", "--meta_model", type=str, default="gpt-4.1", help="Meta‑planner model")
    parser.add_argument("-e", "--exec_model", type=str, default="qwen3-8b", help="Executor model")
    parser.add_argument("--meta_model_path", type=str, default="", help="Optional explicit local checkpoint path for the planner model")
    parser.add_argument("--exec_model_path", type=str, default="", help="Optional explicit local checkpoint path for the executor model")
    parser.add_argument("--prefer_local_backend", action="store_true", help="Prefer the direct local Transformers backend for qwen-style models")
    parser.add_argument("--trace_jsonl", type=str, default="", help="Append structured run traces to this JSONL file")
    parser.add_argument("-s", "--servers", type=str, nargs="*", default=[
        "../server/code_agent.py",
        "../server/documents_tool.py",
        "../server/image_tool.py",
        "../server/math_tool.py",
        "../server/ai_crawl.py",
        "../server/search_tool.py",
    ], help="Paths of tool server scripts")
    return parser.parse_args()

async def run_single_query(client: HierarchicalClient, question: str, file_path: str):
    """
    Execute a single query and display the result.

    Args:
        client: Initialized HierarchicalClient instance
        question: User's question
        file_path: Optional file path for context
    """
    answer = await client.process_query(question, file_path, str(uuid.uuid4()))
    print("\nFINAL ANSWER:", answer)

async def main_async(args):
    """
    Main async function that sets up and runs the AgentFly client.

    Args:
        args: Parsed command line arguments
    """
    # Load environment variables (API keys, etc.)
    load_dotenv()

    # Initialize the hierarchical client
    client = HierarchicalClient(
        args.meta_model,
        args.exec_model,
        os.getenv("USE_AZURE_OPENAI") == "True",
        trace_jsonl=args.trace_jsonl or None,
        meta_model_path=args.meta_model_path or None,
        exec_model_path=args.exec_model_path or None,
        prefer_local_backend=args.prefer_local_backend,
        backend_mode="auto",
    )

    # Connect to tool servers
    await client.connect_to_servers(args.servers)

    try:
        if args.question:
            # Run single query mode
            await run_single_query(client, args.question, args.file)
        else:
            # Interactive mode
            print("Enter 'exit' to quit.")
            while True:
                q = input("\nQuestion: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    break
                f = input("File path (optional): ").strip()
                await run_single_query(client, q, f)
    finally:
        # Ensure cleanup happens even if errors occur
        await client.cleanup()

if __name__ == "__main__":
    # Parse arguments and run the main async function
    arg_ns = parse_args()
    asyncio.run(main_async(arg_ns))
