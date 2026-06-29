from __future__ import annotations

import argparse
import json
import os
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _clean_generation_text(text: str) -> str:
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def _extract_tool_calls(text: str) -> tuple[str | None, list[dict[str, Any]] | None]:
    matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.DOTALL)
    if not matches:
        if "<tool_call>" in text:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            tool_name = name_match.group(1) if name_match else "unknown_tool"
            return None, [{
                "id": f"qwen-tool-partial-{uuid.uuid4()}",
                "type": "function",
                "function": {"name": tool_name, "arguments": text},
            }]
        cleaned = _clean_generation_text(text)
        return (cleaned or None), None

    tool_calls: list[dict[str, Any]] = []
    for idx, block in enumerate(matches):
        try:
            payload = json.loads(block)
            tool_name = str(payload["name"])
            arguments = payload.get("arguments", {})
            args_json = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
        except Exception:
            name_match = re.search(r'"name"\s*:\s*"([^"]+)"', block)
            args_match = re.search(r'"arguments"\s*:\s*(.+)\s*$', block, flags=re.DOTALL)
            tool_name = name_match.group(1) if name_match else "unknown_tool"
            args_json = args_match.group(1).strip() if args_match else block
        tool_calls.append({
            "id": f"qwen-tool-{idx}-{uuid.uuid4()}",
            "type": "function",
            "function": {"name": tool_name, "arguments": args_json},
        })
    return None, tool_calls


class ModelRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model_id = args.model_id
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
        print(json.dumps({
            "event": "load_start",
            "model_path": args.model_path,
            "model_id": args.model_id,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_cuda_devices": torch.cuda.device_count(),
            "max_memory_gib": args.max_memory_gib,
            "max_memory_json": os.environ.get("QWEN_SERVER_MAX_MEMORY_JSON", ""),
        }), flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        max_memory = None
        max_memory_json = os.getenv("QWEN_SERVER_MAX_MEMORY_JSON", "").strip()
        if args.device_map == "auto" and torch.cuda.is_available() and max_memory_json:
            raw_max_memory = json.loads(max_memory_json)
            max_memory = {int(key): str(value) for key, value in raw_max_memory.items()}
        elif args.device_map == "auto" and torch.cuda.is_available() and args.max_memory_gib > 0:
            max_memory = {i: f"{args.max_memory_gib}GiB" for i in range(torch.cuda.device_count())}
        load_started = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=dtype,
            device_map=args.device_map,
            max_memory=max_memory,
            trust_remote_code=True,
            local_files_only=args.local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.default_device = self.model.get_input_embeddings().weight.device
        print(json.dumps({
            "event": "load_done",
            "loaded_sec": round(time.time() - load_started, 2),
            "default_device": str(self.default_device),
            "device_map_sample": list(getattr(self.model, "hf_device_map", {}).items())[:20],
        }), flush=True)

    def chat(self, request: dict[str, Any]) -> dict[str, Any]:
        messages = request.get("messages") or []
        tools = request.get("tools") or None
        tool_choice = request.get("tool_choice", "auto")
        tools_for_prompt = tools if tool_choice != "none" else None
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=tools_for_prompt,
        )
        enc = self.tokenizer(rendered, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.default_device)
        attention_mask = enc["attention_mask"].to(self.default_device)
        max_tokens = int(request.get("max_tokens") or request.get("max_completion_tokens") or 1024)
        max_tokens = max(1, min(max_tokens, int(os.getenv("QWEN_SERVER_MAX_NEW_TOKENS", "4096"))))
        temperature = float(request.get("temperature") if request.get("temperature") is not None else 0.0)
        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            if request.get("top_p") is not None:
                generate_kwargs["top_p"] = float(request["top_p"])
        max_time = os.getenv("QWEN_SERVER_GENERATE_MAX_TIME_SEC", "").strip()
        if max_time:
            try:
                generate_kwargs["max_time"] = float(max_time)
            except ValueError:
                pass
        started = time.time()
        with torch.inference_mode():
            output = self.model.generate(**generate_kwargs)
        generated_ids = output[:, input_ids.shape[-1]:]
        raw_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        content, tool_calls = _extract_tool_calls(raw_text)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        finish_reason = "stop"
        if tool_calls:
            message["content"] = None
            message["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        completion_tokens = int(generated_ids.shape[-1])
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": int(input_ids.shape[-1]),
                "completion_tokens": completion_tokens,
                "total_tokens": int(input_ids.shape[-1]) + completion_tokens,
            },
            "stageweaver_runtime": {"elapsed_sec": round(time.time() - started, 3)},
        }


RUNTIME: ModelRuntime | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in {"/health", "/v1/health"}:
            self._send_json({"status": "ok", "model": RUNTIME.model_id if RUNTIME else None})
            return
        if self.path.rstrip("/") == "/v1/models":
            self._send_json({"object": "list", "data": [{"id": RUNTIME.model_id if RUNTIME else "unknown", "object": "model"}]})
            return
        self._send_json({"error": {"message": "not found"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send_json({"error": {"message": "not found"}}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            if payload.get("stream"):
                self._send_json({"error": {"message": "stream=true is not supported"}}, status=400)
                return
            assert RUNTIME is not None
            self._send_json(RUNTIME.chat(payload))
        except Exception as exc:
            self._send_json({"error": {"message": f"{type(exc).__name__}: {exc}"}}, status=500)

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/xiezhen/llm/models/Qwen3-Next-80B-A3B-Instruct")
    parser.add_argument("--model-id", default="qwen3-next-80b-a3b-instruct")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-memory-gib", type=int, default=74)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    global RUNTIME
    args = parse_args()
    RUNTIME = ModelRuntime(args)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"event": "server_ready", "host": args.host, "port": args.port, "model_id": args.model_id}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
