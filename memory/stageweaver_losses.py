from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_id: int,
) -> torch.Tensor:
    """Cross-entropy over non-padding target positions for composer SFT."""
    if logits.dim() != 3:
        raise ValueError(f"expected logits [batch, seq, vocab], got {tuple(logits.shape)}")
    if targets.dim() != 2:
        raise ValueError(f"expected targets [batch, seq], got {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(f"logits/targets sequence mismatch: {tuple(logits.shape[:2])} vs {tuple(targets.shape)}")
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=int(pad_id),
    )
