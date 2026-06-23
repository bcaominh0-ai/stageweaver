from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from memory.stageweaver_composer import StageWeaverComposer, StageWeaverComposerConfig
from memory.stageweaver_projector import StageWeaverProjector
from memory.stageweaver_schema import EXEC_STEP, PLAN_INIT, PLAN_REVISE
from memory.train_stageweaver_composer_sft import _planner_generation_indices, build_argparser


ROOT = Path(__file__).resolve().parents[1]


class _DummyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.config = type("Config", (), {"hidden_size": 4})()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class LatentMemArchitectureTests(unittest.TestCase):
    def test_projector_defaults_to_single_trainable_linear_layer(self) -> None:
        projector = StageWeaverProjector(composer_hidden_size=8, agent_hidden_size=12)
        latent = torch.randn(2, 4, 8, requires_grad=True)

        output = projector(latent)
        output.sum().backward()

        self.assertEqual(projector.projector_type, "linear")
        self.assertIsInstance(projector.net, nn.Linear)
        self.assertEqual(tuple(output.shape), (2, 4, 12))
        self.assertIsNotNone(projector.net.weight.grad)
        self.assertIsNotNone(latent.grad)

    def test_latentmem_aligned_training_defaults(self) -> None:
        composer = StageWeaverComposerConfig(model_name_or_path="unused")
        args = build_argparser().parse_args(
            ["--train_jsonl", "train.jsonl", "--val_jsonl", "val.jsonl", "--output_dir", "out"]
        )

        self.assertEqual(composer.latents_len, 8)
        self.assertEqual(composer.lora_r, 16)
        self.assertEqual(composer.lora_alpha, 32)
        self.assertEqual(composer.lora_dropout, 0.1)
        self.assertEqual(composer.lora_target_modules, ("q_proj", "v_proj"))
        self.assertEqual(args.projector_type, "linear")
        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.lr, 1e-5)

    def test_legacy_mlp_projector_remains_available(self) -> None:
        projector = StageWeaverProjector(
            composer_hidden_size=8,
            agent_hidden_size=12,
            projector_type="mlp",
            hidden_multiplier=2,
        )

        self.assertIsInstance(projector.net, nn.Sequential)
        self.assertEqual(tuple(projector(torch.randn(1, 3, 8)).shape), (1, 3, 12))

    def test_generation_sample_limit_is_global_across_batches(self) -> None:
        generated = 0
        for stages in ([EXEC_STEP, PLAN_INIT], [PLAN_REVISE, PLAN_INIT]):
            indices = _planner_generation_indices(stages, remaining=2 - generated)
            generated += len(indices)

        self.assertEqual(generated, 2)

    def test_composer_uses_left_padding_and_base_training_overrides_peft_freeze(self) -> None:
        tokenizer = type("Tokenizer", (), {"padding_side": "right", "pad_token_id": 0})()

        def fake_get_peft_model(model: nn.Module, _config: object) -> nn.Module:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            return model

        with (
            patch("memory.stageweaver_composer.AutoTokenizer.from_pretrained", return_value=tokenizer),
            patch("memory.stageweaver_composer.AutoModelForCausalLM.from_pretrained", return_value=_DummyBackbone()),
            patch("memory.stageweaver_composer.get_peft_model", side_effect=fake_get_peft_model),
        ):
            composer = StageWeaverComposer(
                StageWeaverComposerConfig(model_name_or_path="unused", train_base_model=True),
                device="cpu",
            )

        self.assertEqual(composer.tokenizer.padding_side, "left")
        self.assertTrue(all(parameter.requires_grad for parameter in composer.model.parameters()))

    def test_training_script_checks_explicit_gpu_and_forces_local_caches(self) -> None:
        script = (ROOT / "scripts" / "train_append_sft.sh").read_text(encoding="utf-8")

        self.assertIn('mapfile -t gpu_rows < <(nvidia-smi', script)
        self.assertIn('if ! gpu_is_idle "$requested_gpu"', script)
        self.assertNotIn("Using caller-selected CUDA_VISIBLE_DEVICES", script)
        for setting in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "MODELSCOPE_CACHE"):
            self.assertIn(f'export {setting}="${{REPO_ROOT}}/', script)

    def _run_training_shell(self, gpu_row: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        temp_root = ROOT / ".cache" / "tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(dir=temp_root))
        fake_bin = temp_dir / "bin"
        fake_bin.mkdir()
        capture = temp_dir / "training_env.txt"
        nvidia_smi = fake_bin / "nvidia-smi"
        trainer = fake_bin / "trainer"
        nvidia_smi.write_text(f"#!/bin/sh\nprintf '%s\\n' '{gpu_row}'\n", encoding="ascii")
        trainer.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' \"$CUDA_VISIBLE_DEVICES\" \"$HOME\" \"$TMPDIR\" \"$XDG_CACHE_HOME\" "
            "\"$HF_HOME\" \"$MODELSCOPE_CACHE\" > \"$CAPTURE_FILE\"\n",
            encoding="ascii",
        )
        nvidia_smi.chmod(0o755)
        trainer.chmod(0o755)
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{fake_bin}:{env['PATH']}",
                "PYTHON_BIN": str(trainer),
                "QWEN3_4B_MODEL_PATH": "unused",
                "CUDA_VISIBLE_DEVICES": "3",
                "CAPTURE_FILE": str(capture),
            }
        )
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / "train_append_sft.sh")],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        return result, capture

    def test_training_script_rejects_busy_explicit_gpu(self) -> None:
        result, capture = self._run_training_shell("3, 4096, 0")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("busy", result.stderr)
        self.assertFalse(capture.exists())

    def test_training_script_checks_idle_explicit_gpu_and_passes_local_caches(self) -> None:
        result, capture = self._run_training_shell("3, 4, 0")

        self.assertEqual(result.returncode, 0, result.stderr)
        values = capture.read_text(encoding="utf-8").splitlines()
        self.assertEqual(values[0], "3")
        self.assertTrue(all(Path(value).is_relative_to(ROOT) for value in values[1:]))


if __name__ == "__main__":
    unittest.main()
