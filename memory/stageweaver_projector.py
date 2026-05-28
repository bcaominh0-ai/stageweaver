from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class StageWeaverProjector(nn.Module):
    def __init__(
        self,
        composer_hidden_size: int,
        agent_hidden_size: int,
        hidden_multiplier: int = 2,
    ) -> None:
        super().__init__()
        inner = max(composer_hidden_size, agent_hidden_size) * hidden_multiplier
        self.composer_hidden_size = int(composer_hidden_size)
        self.agent_hidden_size = int(agent_hidden_size)
        self.hidden_multiplier = int(hidden_multiplier)
        self.net = nn.Sequential(
            nn.Linear(self.composer_hidden_size, inner),
            nn.GELU(),
            nn.Linear(inner, self.agent_hidden_size),
        )

    def forward(self, latent_block: torch.Tensor) -> torch.Tensor:
        if latent_block.dim() != 3:
            raise ValueError(f"expected latent_block [batch, latent_len, hidden], got {tuple(latent_block.shape)}")
        weight_dtype = next(self.parameters()).dtype
        if latent_block.dtype != weight_dtype:
            latent_block = latent_block.to(weight_dtype)
        return self.net(latent_block)

    def save_checkpoint(self, output_path: str | Path) -> None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": {
                    "composer_hidden_size": self.composer_hidden_size,
                    "agent_hidden_size": self.agent_hidden_size,
                    "hidden_multiplier": self.hidden_multiplier,
                },
                "state_dict": {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()},
            },
            str(out),
        )

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device: torch.device | str) -> "StageWeaverProjector":
        payload: Any = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        config = dict(payload.get("config", payload.get("projector_config", {})))
        state_dict = payload.get("state_dict", payload.get("projector_state_dict"))
        if not config or state_dict is None:
            raise KeyError("projector checkpoint must contain config/state_dict or projector_config/projector_state_dict")
        model = cls(
            composer_hidden_size=int(config["composer_hidden_size"]),
            agent_hidden_size=int(config["agent_hidden_size"]),
            hidden_multiplier=int(config.get("hidden_multiplier", 2)),
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model
