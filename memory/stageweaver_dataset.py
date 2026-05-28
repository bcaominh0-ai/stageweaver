from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch.utils.data import Dataset

try:
    from .stageweaver_schema import (
        EXEC_STEP,
        StageTuple,
        load_stage_tuples,
        retrieval_text,
        serialize_role_conditioned_context,
        tuple_role,
    )
    from .stageweaver_serializers import render_positive_output_memory
except Exception:  # pragma: no cover
    from stageweaver_schema import (
        EXEC_STEP,
        StageTuple,
        load_stage_tuples,
        retrieval_text,
        serialize_role_conditioned_context,
        tuple_role,
    )
    from stageweaver_serializers import render_positive_output_memory


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>"]
STAGE_TO_ID = {
    "PLAN_INIT": 0,
    "PLAN_REVISE": 1,
    "EXEC_STEP": 2,
}


def build_char_vocab(items: Iterable[StageTuple]) -> dict[str, int]:
    chars = set()
    for item in items:
        chars.update(item.state_text)
        chars.update(item.target_text)
        chars.update(item.question_text)
        chars.update(item.current_state_text)
        chars.update(item.tool_memory_text)
        chars.update(item.subtask_memory_text)
        for tool_name in item.available_tools:
            chars.update(tool_name)
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for ch in sorted(chars):
        if ch not in vocab:
            vocab[ch] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], bos: bool = False, eos: bool = False) -> list[int]:
    ids: list[int] = []
    if bos:
        ids.append(vocab["<bos>"])
    for ch in text:
        if ch in vocab:
            ids.append(vocab[ch])
    if eos:
        ids.append(vocab["<eos>"])
    return ids


@dataclass
class StageWeaverExample:
    stage_id: int
    source_ids: list[int]
    source_core_ids: list[int]
    source_nomem_ids: list[int]
    memory_ids: list[int]
    target_in_ids: list[int]
    target_out_ids: list[int]


def _token_overlap(a: str, b: str) -> int:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    return len(sa & sb)


def retrieve_positive_neighbors(
    item: StageTuple,
    retrieval_bank: list[StageTuple],
    matched_k: int,
) -> list[dict[str, Any]]:
    item_key = retrieval_text(item)
    item_role = tuple_role(item)
    candidates: list[tuple[int, str, dict[str, Any]]] = []
    for cand in retrieval_bank:
        cand_role = tuple_role(cand)
        if cand.source_id == item.source_id or cand_role != item_role:
            continue
        if item_role != "planner" and cand.stage != item.stage:
            continue
        if int(cand.reward) != 1:
            continue
        overlap = _token_overlap(item_key, retrieval_text(cand))
        candidates.append((overlap, cand.source_id, cand.to_dict()))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [entry for _, _, entry in candidates[:matched_k]]


class StageWeaverDataset(Dataset):
    def __init__(self, examples: list[StageWeaverExample]):
        self.examples = examples

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        vocab: dict[str, int] | None = None,
        matched_k: int = 3,
        memory_budget_tokens: int = 192,
        bounded_budget_tokens: int | None = None,
        retrieval_model: str = "gpt-4.1",
        retrieval_path: str | None = None,
    ) -> tuple["StageWeaverDataset", dict[str, int]]:
        tuples_ = load_stage_tuples(path)
        retrieval_bank = load_stage_tuples(retrieval_path) if retrieval_path else tuples_
        if vocab is None:
            vocab = build_char_vocab(tuples_)

        examples: list[StageWeaverExample] = []
        for item in tuples_:
            if item.stage not in STAGE_TO_ID:
                raise ValueError(f"Unknown stage in dataset tuple: {item.stage}")
            neighbors = retrieve_positive_neighbors(item, retrieval_bank, matched_k=matched_k)
            rendered = render_positive_output_memory(
                neighbors,
                budget_tokens=memory_budget_tokens,
                bounded_budget_tokens=bounded_budget_tokens,
                model=retrieval_model,
            )
            source_nomem_text = serialize_role_conditioned_context(item, retrieved_cases_text="")
            source_text = serialize_role_conditioned_context(item, retrieved_cases_text=rendered["text"])
            source_ids = encode_text(source_text, vocab, bos=False, eos=True)
            source_core_ids = encode_text(source_nomem_text, vocab, bos=False, eos=False)
            source_nomem_ids = encode_text(source_nomem_text, vocab, bos=False, eos=True)
            memory_ids = encode_text(rendered["text"], vocab, bos=False, eos=False)
            target_in_ids = encode_text(item.target_text, vocab, bos=True, eos=False)
            target_out_ids = encode_text(item.target_text, vocab, bos=False, eos=True)
            examples.append(
                StageWeaverExample(
                    stage_id=STAGE_TO_ID[item.stage],
                    source_ids=source_ids,
                    source_core_ids=source_core_ids,
                    source_nomem_ids=source_nomem_ids,
                    memory_ids=memory_ids,
                    target_in_ids=target_in_ids,
                    target_out_ids=target_out_ids,
                )
            )
        return cls(examples), vocab

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> StageWeaverExample:
        return self.examples[idx]


def collate_examples(batch: list[StageWeaverExample], pad_id: int) -> dict[str, torch.Tensor]:
    max_src = max(len(item.source_ids) for item in batch)
    max_src_core = max(len(item.source_core_ids) for item in batch)
    max_src_nomem = max(len(item.source_nomem_ids) for item in batch)
    max_mem = max(max(len(item.memory_ids) for item in batch), 1)
    max_tgt = max(len(item.target_in_ids) for item in batch)

    stage_ids = []
    src_ids = []
    src_mask = []
    src_core_ids = []
    src_core_mask = []
    src_nomem_ids = []
    src_nomem_mask = []
    mem_ids = []
    mem_mask = []
    tgt_in = []
    tgt_out = []

    for item in batch:
        src_pad = max_src - len(item.source_ids)
        src_core_pad = max_src_core - len(item.source_core_ids)
        src_nomem_pad = max_src_nomem - len(item.source_nomem_ids)
        mem_pad = max_mem - len(item.memory_ids)
        tgt_pad = max_tgt - len(item.target_in_ids)
        stage_ids.append(item.stage_id)
        src_ids.append(item.source_ids + [pad_id] * src_pad)
        src_mask.append([1] * len(item.source_ids) + [0] * src_pad)
        src_core_ids.append(item.source_core_ids + [pad_id] * src_core_pad)
        src_core_mask.append([1] * len(item.source_core_ids) + [0] * src_core_pad)
        src_nomem_ids.append(item.source_nomem_ids + [pad_id] * src_nomem_pad)
        src_nomem_mask.append([1] * len(item.source_nomem_ids) + [0] * src_nomem_pad)
        mem_ids.append(item.memory_ids + [pad_id] * mem_pad)
        mem_mask.append([1] * len(item.memory_ids) + [0] * mem_pad)
        tgt_in.append(item.target_in_ids + [pad_id] * tgt_pad)
        tgt_out.append(item.target_out_ids + [pad_id] * tgt_pad)

    return {
        "stage_ids": torch.tensor(stage_ids, dtype=torch.long),
        "source_ids": torch.tensor(src_ids, dtype=torch.long),
        "source_mask": torch.tensor(src_mask, dtype=torch.long),
        "source_core_ids": torch.tensor(src_core_ids, dtype=torch.long),
        "source_core_mask": torch.tensor(src_core_mask, dtype=torch.long),
        "source_nomem_ids": torch.tensor(src_nomem_ids, dtype=torch.long),
        "source_nomem_mask": torch.tensor(src_nomem_mask, dtype=torch.long),
        "memory_ids": torch.tensor(mem_ids, dtype=torch.long),
        "memory_mask": torch.tensor(mem_mask, dtype=torch.long),
        "target_in": torch.tensor(tgt_in, dtype=torch.long),
        "target_out": torch.tensor(tgt_out, dtype=torch.long),
    }
