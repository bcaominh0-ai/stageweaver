from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

try:
    from client.agent_local_server import EXEC_SYSTEM_PROMPT, META_SYSTEM_PROMPT, _resolve_local_model_path
except Exception:  # pragma: no cover
    from agent_local_server import EXEC_SYSTEM_PROMPT, META_SYSTEM_PROMPT, _resolve_local_model_path

try:
    from .stageweaver_composer import StageWeaverComposer, StageWeaverComposerConfig
    from .stageweaver_projector import StageWeaverProjector
    from .stageweaver_schema import (
        EXEC_STEP,
        StageTuple,
        is_role_memory_item,
        load_stage_tuples,
        retrieval_text,
        role_memory_bucket_key,
        serialize_role_conditioned_context,
        tuple_question_text,
        tuple_role,
    )
    from .stageweaver_serializers import render_positive_output_memory
except Exception:  # pragma: no cover
    from stageweaver_composer import StageWeaverComposer, StageWeaverComposerConfig
    from stageweaver_projector import StageWeaverProjector
    from stageweaver_schema import (
        EXEC_STEP,
        StageTuple,
        is_role_memory_item,
        load_stage_tuples,
        retrieval_text,
        role_memory_bucket_key,
        serialize_role_conditioned_context,
        tuple_question_text,
        tuple_role,
    )
    from stageweaver_serializers import render_positive_output_memory

try:
    from .semantic_retriever import DEFAULT_SEMANTIC_MODEL_ID, SemanticRetriever
except Exception:  # pragma: no cover
    try:
        from semantic_retriever import DEFAULT_SEMANTIC_MODEL_ID, SemanticRetriever
    except Exception:  # pragma: no cover
        DEFAULT_SEMANTIC_MODEL_ID = "AI-ModelScope/bge-small-en-v1.5"
        SemanticRetriever = None  # type: ignore[assignment]


def _preferred_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


@dataclass
class ComposerSFTExample:
    stage: str
    role: str
    source_id: str
    composer_text: str
    prompt_messages: list[dict[str, str]]
    target_text: str


class ComposerSFTDataset(Dataset):
    def __init__(self, examples: list[ComposerSFTExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> ComposerSFTExample:
        return self.examples[index]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tuple_to_retriever_item(item: StageTuple) -> dict[str, Any]:
    return item.to_dict()


def retrieve_role_memory_neighbors(
    item: StageTuple,
    retriever: SemanticRetriever,
    matched_k: int,
    oversample_k: int = 32,
) -> list[dict[str, Any]]:
    if matched_k <= 0:
        return []
    item_key = retrieval_text(item)
    hits = retriever.retrieve(item_key, top_k=max(matched_k, oversample_k))
    candidates: list[dict[str, Any]] = []
    item_role = tuple_role(item)
    for hit in hits:
        cand = StageTuple.from_dict(dict(hit["item"]))
        cand_role = tuple_role(cand)
        item_trace_id = str((item.metadata or {}).get("trace_id") or "").strip()
        cand_trace_id = str((cand.metadata or {}).get("trace_id") or "").strip()
        same_trace = bool(item_trace_id and cand_trace_id and item_trace_id == cand_trace_id)
        if same_trace or cand.source_id == item.source_id or cand_role != item_role:
            continue
        if cand.stage != item.stage:
            continue
        if not is_role_memory_item(cand):
            continue
        candidates.append(cand.to_dict())
        if len(candidates) >= matched_k:
            break
    return candidates


def _retrieval_bucket_key(item: StageTuple) -> tuple[str, str]:
    return role_memory_bucket_key(item)


def _executor_visible_prompt(item: StageTuple) -> tuple[str, str]:
    system = EXEC_SYSTEM_PROMPT
    user = item.current_state_text or item.state_text or tuple_question_text(item)
    return system, user.strip()


def build_examples(
    data_path: str,
    retrieval_path: str | None,
    matched_k: int,
    memory_budget_tokens: int,
    bounded_budget_tokens: int,
    retrieval_model: str,
    semantic_model_id: str,
    semantic_device: str,
    semantic_cache_dir: str | None,
    semantic_max_length: int,
    max_examples: int,
) -> list[ComposerSFTExample]:
    tuples_ = load_stage_tuples(data_path)
    retrieval_bank = load_stage_tuples(retrieval_path) if retrieval_path else tuples_
    if SemanticRetriever is None:
        raise RuntimeError(
            "SemanticRetriever dependencies are unavailable. Install the sentence-transformers/modelscope stack; "
            "training no longer falls back to token-overlap retrieval."
        )
    grouped_retrievers: dict[tuple[str, str], SemanticRetriever] = {}
    grouped_sizes: dict[tuple[str, str], int] = {}
    grouped_items: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for bank_item in retrieval_bank:
        if not is_role_memory_item(bank_item):
            continue
        key = _retrieval_bucket_key(bank_item)
        grouped_items.setdefault(key, []).append(_tuple_to_retriever_item(bank_item))
    for key, items in grouped_items.items():
        retriever = SemanticRetriever(
            model_id=semantic_model_id,
            device=semantic_device,
            cache_dir=semantic_cache_dir,
            max_seq_length=semantic_max_length,
        )
        retriever.build(
            items,
            key_fn=lambda data: retrieval_text(StageTuple.from_dict(dict(data))),
        )
        grouped_retrievers[key] = retriever
        grouped_sizes[key] = len(items)
    examples: list[ComposerSFTExample] = []
    for item in tuples_:
        retriever_key = _retrieval_bucket_key(item)
        retriever = grouped_retrievers.get(retriever_key)
        if retriever is None:
            neighbors = []
        else:
            neighbors = retrieve_role_memory_neighbors(
                item,
                retriever,
                matched_k=matched_k,
                oversample_k=max(matched_k, grouped_sizes.get(retriever_key, matched_k)),
            )
        rendered = render_positive_output_memory(
            neighbors,
            budget_tokens=memory_budget_tokens,
            bounded_budget_tokens=bounded_budget_tokens,
            model=retrieval_model,
        )
        composer_text = serialize_role_conditioned_context(item, retrieved_cases_text=rendered["text"])
        role = tuple_role(item)
        if item.stage == EXEC_STEP or role == "executor":
            system, user = _executor_visible_prompt(item)
            prompt_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
            system = META_SYSTEM_PROMPT
            prompt_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": tuple_question_text(item) or item.state_text},
            ]
        examples.append(
            ComposerSFTExample(
                stage=item.stage,
                role=role,
                source_id=item.source_id,
                composer_text=composer_text,
                prompt_messages=prompt_messages,
                target_text=item.target_text,
            )
        )
        if max_examples > 0 and len(examples) >= max_examples:
            break
    return examples


def _render_messages(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    lines: list[str] = []
    for message in messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
    lines.append("ASSISTANT:")
    return "\n\n".join(lines)


def build_collate_fn(
    composer: StageWeaverComposer,
    agent_tokenizer: AutoTokenizer,
    composer_max_length: int,
    agent_prompt_max_length: int,
    agent_target_max_length: int,
):
    def _collate(batch: list[ComposerSFTExample]) -> dict[str, Any]:
        composer_batch = composer.tokenize([item.composer_text for item in batch], max_length=composer_max_length)
        prompt_texts = [_render_messages(agent_tokenizer, item.prompt_messages) for item in batch]
        prompt_batch = agent_tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=agent_prompt_max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        targets = [item.target_text for item in batch]
        if agent_tokenizer.eos_token:
            targets = [text + agent_tokenizer.eos_token for text in targets]
        target_batch = agent_tokenizer(
            targets,
            padding=True,
            truncation=True,
            max_length=agent_target_max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        return {
            "composer_input_ids": composer_batch["input_ids"],
            "composer_attention_mask": composer_batch["attention_mask"],
            "prompt_input_ids": prompt_batch["input_ids"],
            "prompt_attention_mask": prompt_batch["attention_mask"],
            "target_input_ids": target_batch["input_ids"],
            "target_attention_mask": target_batch["attention_mask"],
            "stage": [item.stage for item in batch],
            "role": [item.role for item in batch],
            "source_id": [item.source_id for item in batch],
            "target_text": [item.target_text for item in batch],
        }

    return _collate


def _prepare_agent_inputs(
    agent_model: AutoModelForCausalLM,
    prompt_input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    target_input_ids: torch.Tensor,
    target_attention_mask: torch.Tensor,
    latent_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embed = agent_model.get_input_embeddings()
    prompt_embeds = embed(prompt_input_ids)
    target_embeds = embed(target_input_ids)
    batch = prompt_embeds.size(0)
    latent_len = latent_block.size(1)
    prompt_len = prompt_input_ids.size(1)
    target_len = target_input_ids.size(1)
    total_len = prompt_len + latent_len + target_len
    device = prompt_embeds.device
    dtype = prompt_embeds.dtype

    inputs_embeds = torch.zeros((batch, total_len, prompt_embeds.size(-1)), device=device, dtype=dtype)
    attention_mask = torch.zeros((batch, total_len), device=device, dtype=prompt_attention_mask.dtype)
    labels = torch.full((batch, total_len), fill_value=-100, device=device, dtype=target_input_ids.dtype)

    for idx in range(batch):
        p_len = int(prompt_attention_mask[idx].sum().item())
        t_len = int(target_attention_mask[idx].sum().item())
        inputs_embeds[idx, :p_len, :] = prompt_embeds[idx, :p_len, :]
        inputs_embeds[idx, p_len : p_len + latent_len, :] = latent_block[idx]
        inputs_embeds[idx, p_len + latent_len : p_len + latent_len + t_len, :] = target_embeds[idx, :t_len, :]
        attention_mask[idx, : p_len + latent_len + t_len] = 1
        labels[idx, p_len + latent_len : p_len + latent_len + t_len] = target_input_ids[idx, :t_len]
    return inputs_embeds, attention_mask, labels


def _strip_generation(text: str) -> str:
    return text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def _strip_json_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
    return stripped.strip()


def _planner_output_is_valid(text: str) -> bool:
    stripped = _strip_json_fences(_strip_generation(text))
    if not stripped:
        return False
    if stripped.startswith("FINAL ANSWER:"):
        return True
    try:
        parsed = json.loads(stripped)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    if set(parsed.keys()) == {"plan"}:
        return isinstance(parsed.get("plan"), list)
    if set(parsed.keys()) == {"final"}:
        final_payload = parsed.get("final")
        return isinstance(final_payload, dict) and isinstance(final_payload.get("answer"), str)
    return False


def _planner_generation_indices(stages: list[str], remaining: int) -> list[int]:
    if remaining <= 0:
        return []
    indices: list[int] = []
    for idx, stage in enumerate(stages):
        if stage == EXEC_STEP:
            continue
        indices.append(idx)
        if len(indices) >= remaining:
            break
    return indices


@torch.no_grad()
def evaluate(
    composer: StageWeaverComposer,
    projector: StageWeaverProjector,
    agent_model: AutoModelForCausalLM,
    agent_tokenizer: AutoTokenizer,
    loader: DataLoader,
    device: torch.device,
    eval_generation_samples: int,
    eval_max_new_tokens: int,
) -> dict[str, float]:
    composer.eval()
    projector.eval()
    agent_model.eval()
    total_loss = 0.0
    total_items = 0
    planner_valid = 0
    planner_total = 0

    for batch in loader:
        composer_latent = composer.text_to_latent(
            batch["composer_input_ids"].to(device),
            batch["composer_attention_mask"].to(device),
        )
        agent_latent = projector(composer_latent)
        inputs_embeds, attention_mask, labels = _prepare_agent_inputs(
            agent_model,
            batch["prompt_input_ids"].to(device),
            batch["prompt_attention_mask"].to(device),
            batch["target_input_ids"].to(device),
            batch["target_attention_mask"].to(device),
            agent_latent,
        )
        outputs = agent_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            use_cache=False,
        )
        total_loss += float(outputs.loss.item()) * len(batch["stage"])
        total_items += len(batch["stage"])

        remaining_generations = max(eval_generation_samples - planner_total, 0)
        for idx in _planner_generation_indices(batch["stage"], remaining_generations):
            planner_total += 1
            prompt_len = int(batch["prompt_attention_mask"][idx].sum().item())
            prompt_embeds = agent_model.get_input_embeddings()(batch["prompt_input_ids"][idx : idx + 1, :prompt_len].to(device))
            latent = agent_latent[idx : idx + 1]
            full_embeds = torch.cat([prompt_embeds, latent], dim=1)
            full_mask = torch.ones((1, full_embeds.size(1)), device=device, dtype=torch.long)
            output_ids = agent_model.generate(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                max_new_tokens=eval_max_new_tokens,
                do_sample=False,
                pad_token_id=agent_tokenizer.pad_token_id,
                eos_token_id=agent_tokenizer.eos_token_id,
            )
            text = agent_tokenizer.decode(output_ids[0], skip_special_tokens=False)
            planner_valid += int(_planner_output_is_valid(text))

    return {
        "avg_loss": total_loss / max(total_items, 1),
        "planner_json_valid_rate": (planner_valid / planner_total) if planner_total else 0.0,
        "items": total_items,
        "planner_eval_items": planner_total,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    try:
        composer_model_path = _resolve_local_model_path(args.composer_model, args.composer_model_path or None)
    except FileNotFoundError:
        composer_model_path = args.composer_model_path or args.composer_model

    composer_cfg = StageWeaverComposerConfig(
        model_name_or_path=composer_model_path,
        latents_len=args.latents_len,
        lora_r=args.composer_lora_r,
        lora_alpha=args.composer_lora_alpha,
        lora_dropout=args.composer_lora_dropout,
        train_base_model=args.train_composer_base,
    )
    composer = StageWeaverComposer(composer_cfg, device=device)

    agent_model_path = _resolve_local_model_path(args.agent_model, args.agent_model_path or None)
    agent_tokenizer = AutoTokenizer.from_pretrained(agent_model_path, trust_remote_code=True)
    if agent_tokenizer.pad_token_id is None:
        agent_tokenizer.pad_token = agent_tokenizer.eos_token
    agent_model = AutoModelForCausalLM.from_pretrained(
        agent_model_path,
        torch_dtype=_preferred_torch_dtype(device),
        trust_remote_code=True,
    ).to(device)
    for param in agent_model.parameters():
        param.requires_grad_(False)
    agent_hidden_size = int(agent_model.config.hidden_size)
    projector = StageWeaverProjector(
        composer_hidden_size=composer.hidden_size,
        agent_hidden_size=agent_hidden_size,
        projector_type=args.projector_type,
        hidden_multiplier=args.projector_hidden_multiplier,
    ).to(device=device, dtype=next(composer.parameters()).dtype)

    train_examples = build_examples(
        data_path=args.train_jsonl,
        retrieval_path=(args.retrieval_jsonl or args.train_jsonl),
        matched_k=args.matched_k,
        memory_budget_tokens=args.memory_budget_tokens,
        bounded_budget_tokens=args.bounded_budget_tokens,
        retrieval_model=args.retrieval_model,
        semantic_model_id=args.semantic_model_id,
        semantic_device=args.semantic_device,
        semantic_cache_dir=(args.semantic_cache_dir or None),
        semantic_max_length=args.semantic_max_length,
        max_examples=args.max_train_examples,
    )
    val_examples = build_examples(
        data_path=args.val_jsonl,
        retrieval_path=(args.retrieval_jsonl or args.train_jsonl),
        matched_k=args.matched_k,
        memory_budget_tokens=args.memory_budget_tokens,
        bounded_budget_tokens=args.bounded_budget_tokens,
        retrieval_model=args.retrieval_model,
        semantic_model_id=args.semantic_model_id,
        semantic_device=args.semantic_device,
        semantic_cache_dir=(args.semantic_cache_dir or None),
        semantic_max_length=args.semantic_max_length,
        max_examples=args.max_val_examples,
    )
    train_ds = ComposerSFTDataset(train_examples)
    val_ds = ComposerSFTDataset(val_examples)
    collate_fn = build_collate_fn(
        composer=composer,
        agent_tokenizer=agent_tokenizer,
        composer_max_length=args.composer_max_length,
        agent_prompt_max_length=args.agent_prompt_max_length,
        agent_target_max_length=args.agent_target_max_length,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    params = [param for param in list(composer.parameters()) + list(projector.parameters()) if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    history: list[dict[str, Any]] = []
    best_val = None
    best_payload = None
    step = 0

    def _trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
        trainable_names = {name for name, param in module.named_parameters() if param.requires_grad}
        return {
            name: tensor.detach().cpu()
            for name, tensor in module.state_dict().items()
            if name in trainable_names
        }

    for epoch in range(1, args.epochs + 1):
        composer.train()
        projector.train()
        train_loss_sum = 0.0
        train_items = 0
        for batch in train_loader:
            optimizer.zero_grad()
            composer_latent = composer.text_to_latent(
                batch["composer_input_ids"].to(device),
                batch["composer_attention_mask"].to(device),
            )
            agent_latent = projector(composer_latent)
            inputs_embeds, attention_mask, labels = _prepare_agent_inputs(
                agent_model,
                batch["prompt_input_ids"].to(device),
                batch["prompt_attention_mask"].to(device),
                batch["target_input_ids"].to(device),
                batch["target_attention_mask"].to(device),
                agent_latent,
            )
            outputs = agent_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True,
                use_cache=False,
            )
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            train_loss_sum += float(loss.item()) * len(batch["stage"])
            train_items += len(batch["stage"])
            step += 1
            if args.log_every_steps > 0 and step % args.log_every_steps == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "train_avg_loss": train_loss_sum / max(train_items, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if args.max_steps > 0 and step >= args.max_steps:
                break

        train_metrics = {
            "avg_loss": train_loss_sum / max(train_items, 1),
            "items": train_items,
        }
        val_metrics = evaluate(
            composer=composer,
            projector=projector,
            agent_model=agent_model,
            agent_tokenizer=agent_tokenizer,
            loader=val_loader,
            device=device,
            eval_generation_samples=args.eval_generation_samples,
            eval_max_new_tokens=args.eval_max_new_tokens,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        if best_val is None or val_metrics["avg_loss"] < best_val:
            best_val = val_metrics["avg_loss"]
            best_payload = {
                "composer_config": composer_cfg.__dict__,
                "composer_state_dict": _trainable_state_dict(composer),
                "composer_state_dict_type": "trainable_only",
                "projector_state_dict": {k: v.detach().cpu() for k, v in projector.state_dict().items()},
                "projector_config": {
                    "composer_hidden_size": composer.hidden_size,
                    "agent_hidden_size": agent_hidden_size,
                    "projector_type": args.projector_type,
                    "hidden_multiplier": args.projector_hidden_multiplier,
                },
                "agent_model_path": agent_model_path,
                "args": vars(args),
                "best_val_metrics": val_metrics,
            }
        if args.max_steps > 0 and step >= args.max_steps:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if best_payload is not None:
        torch.save(best_payload, output_dir / "stageweaver_composer_sft.pt")
    result = {
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "retrieval_jsonl": args.retrieval_jsonl or args.train_jsonl,
        "semantic_model_id": args.semantic_model_id,
        "seed": args.seed,
        "device": str(device),
        "composer_model": composer_model_path,
        "agent_model_path": agent_model_path,
        "latents_len": args.latents_len,
        "composer_lora": {
            "r": args.composer_lora_r,
            "alpha": args.composer_lora_alpha,
            "dropout": args.composer_lora_dropout,
            "target_modules": list(composer_cfg.lora_target_modules),
        },
        "projector_type": args.projector_type,
        "agent_frozen": all(not param.requires_grad for param in agent_model.parameters()),
        "epochs": len(history),
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_steps": args.max_steps,
        "trainable_params": sum(param.numel() for param in params),
        "history": history,
        "best_val_avg_loss": best_val,
    }
    (output_dir / "train_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, required=True)
    parser.add_argument("--retrieval_jsonl", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--composer_model",
        type=str,
        default="qwen3-4b",
        help="Composer backbone alias or HF path. Default matches LatentMem's released Qwen3-4B-Instruct-2507 family.",
    )
    parser.add_argument("--composer_model_path", type=str, default="")
    parser.add_argument("--agent_model", type=str, default="qwen3-4b")
    parser.add_argument("--agent_model_path", type=str, default="")
    parser.add_argument("--latents_len", type=int, default=8)
    parser.add_argument("--composer_lora_r", type=int, default=16)
    parser.add_argument("--composer_lora_alpha", type=int, default=32)
    parser.add_argument("--composer_lora_dropout", type=float, default=0.1)
    parser.add_argument("--train_composer_base", action="store_true")
    parser.add_argument("--projector_hidden_multiplier", type=int, default=2)
    parser.add_argument("--projector_type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--matched_k", type=int, default=3)
    parser.add_argument("--memory_budget_tokens", type=int, default=192)
    parser.add_argument("--bounded_budget_tokens", type=int, default=96)
    parser.add_argument("--retrieval_model", type=str, default="gpt-4.1")
    parser.add_argument("--semantic_model_id", type=str, default=DEFAULT_SEMANTIC_MODEL_ID)
    parser.add_argument("--semantic_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--semantic_cache_dir", type=str, default=str(PROJECT_ROOT / ".cache" / "modelscope"))
    parser.add_argument("--semantic_max_length", type=int, default=256)
    parser.add_argument("--composer_max_length", type=int, default=384)
    parser.add_argument("--agent_prompt_max_length", type=int, default=768)
    parser.add_argument("--agent_target_max_length", type=int, default=128)
    parser.add_argument("--eval_generation_samples", type=int, default=2)
    parser.add_argument("--eval_max_new_tokens", type=int, default=96)
    parser.add_argument("--max_train_examples", type=int, default=0)
    parser.add_argument("--max_val_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    result = train(args)
    print(json.dumps({"best_val_avg_loss": result["best_val_avg_loss"], "output_dir": args.output_dir}, ensure_ascii=False))


if __name__ == "__main__":
    main()
