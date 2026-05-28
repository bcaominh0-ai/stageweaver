from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

DEFAULT_SEMANTIC_MODEL_ID = "AI-ModelScope/bge-small-en-v1.5"


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


class SemanticRetriever:
    def __init__(
        self,
        model_id: str = DEFAULT_SEMANTIC_MODEL_ID,
        device: str = "auto",
        cache_dir: str | None = None,
        batch_size: int = 32,
        max_seq_length: int = 256,
    ) -> None:
        self.model_id = model_id
        self.device = _resolve_device(device)
        self.batch_size = batch_size
        model_dir = snapshot_download(model_id, cache_dir=cache_dir)
        self.model_dir = str(Path(model_dir).resolve())
        self.model = SentenceTransformer(self.model_dir, device=self.device)
        self.model.max_seq_length = max_seq_length
        self.items: list[dict[str, Any]] = []
        self.keys: list[str] = []
        self.embeddings: torch.Tensor | None = None

    def build(self, items: Sequence[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str]) -> None:
        self.items = [dict(item) for item in items]
        self.keys = [str(key_fn(item)).strip() for item in self.items]
        if not self.items:
            self.embeddings = None
            return
        embeddings = self.model.encode(
            self.keys,
            batch_size=self.batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=len(self.keys) >= 128,
        )
        self.embeddings = embeddings.detach().to("cpu")

    def retrieve(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self.embeddings is None or not self.items:
            return []
        query_embedding = self.model.encode(
            [query],
            batch_size=1,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_embedding = query_embedding.detach().to("cpu")
        scores = torch.matmul(query_embedding, self.embeddings.T).squeeze(0)
        k = min(max(top_k, 0), len(self.items))
        if k == 0:
            return []
        top_scores, top_indices = torch.topk(scores, k)
        hits: list[dict[str, Any]] = []
        for rank, (score, index) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), start=1):
            hits.append(
                {
                    "rank": rank,
                    "score": float(score),
                    "key": self.keys[index],
                    "item": dict(self.items[index]),
                }
            )
        return hits
