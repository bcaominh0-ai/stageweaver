from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None


@dataclass
class StageWeaverComposerConfig:
    model_name_or_path: str
    latents_len: int = 8
    lora_r: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    train_base_model: bool = False
    trust_remote_code: bool = True


def _preferred_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class StageWeaverComposer(nn.Module):
    """LatentMem-style composer: text prompt + learned latent queries -> latent block."""

    def __init__(self, config: StageWeaverComposerConfig, device: torch.device | str) -> None:
        super().__init__()
        self.config = config
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=_preferred_torch_dtype(self.device),
            trust_remote_code=config.trust_remote_code,
        ).to(self.device)
        self.hidden_size = int(self.model.config.hidden_size)
        self.latents_len = int(config.latents_len)
        self.query_latents = nn.Parameter(
            torch.randn(
                self.latents_len,
                self.hidden_size,
                device=self.device,
                dtype=self.model.get_input_embeddings().weight.dtype,
            )
            * 0.02
        )

        if config.lora_r > 0:
            if get_peft_model is None or LoraConfig is None or TaskType is None:
                raise RuntimeError("peft is required for LoRA composer training but is not installed.")
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(config.lora_r),
                lora_alpha=int(config.lora_alpha),
                lora_dropout=float(config.lora_dropout),
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_cfg)

        if not config.train_base_model and config.lora_r <= 0:
            for param in self.model.parameters():
                param.requires_grad_(False)

    def _input_embeds(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def text_to_latent(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.dim() != 2 or attention_mask.dim() != 2:
            raise ValueError("composer input_ids and attention_mask must be rank-2 tensors")
        text_embeds = self._input_embeds(input_ids.to(self.device))
        batch = text_embeds.size(0)
        latent_queries = self.query_latents.unsqueeze(0).expand(batch, -1, -1).to(
            device=text_embeds.device,
            dtype=text_embeds.dtype,
        )
        full_embeds = torch.cat([text_embeds, latent_queries], dim=1)
        latent_mask = torch.ones(
            (batch, self.latents_len),
            dtype=attention_mask.dtype,
            device=self.device,
        )
        full_mask = torch.cat([attention_mask.to(self.device), latent_mask], dim=1)
        outputs = self.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        return hidden[:, -self.latents_len :, :]

    def tokenize(
        self,
        texts: list[str],
        max_length: int,
    ) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def trainable_parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def save_checkpoint(self, output_path: str, extra_config: dict[str, Any] | None = None) -> None:
        from pathlib import Path

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "composer_config": {
                "model_name_or_path": self.config.model_name_or_path,
                "latents_len": self.config.latents_len,
                "lora_r": self.config.lora_r,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "train_base_model": self.config.train_base_model,
                "trust_remote_code": self.config.trust_remote_code,
            },
            "query_latents": self.query_latents.detach().cpu(),
            "state_dict": {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()},
            "extra_config": dict(extra_config or {}),
        }
        torch.save(payload, str(path))
