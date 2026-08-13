import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch import nn

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface
from starVLA.training import train_starvla


class _DummyAggregate:
    def __init__(self, size=100):
        self.all_steps = [(index, 0) for index in range(size)]
        self.trajectory_ids = np.arange(size)
        self.trajectory_lengths = np.ones(size, dtype=np.int64)
        self.dataset_name = "dummy"
        self.metadata = SimpleNamespace()
        self.tag = "new_embodiment"
        self.epoch = None

    def __len__(self):
        return len(self.all_steps)

    def __getitem__(self, index):
        return int(index)

    def set_epoch(self, epoch):
        self.epoch = epoch


def _mixture(dataset, seed=42):
    with mock.patch.object(LeRobotMixtureDataset, "update_metadata"):
        return LeRobotMixtureDataset(
            [(dataset, 1.0)],
            mode="train",
            seed=seed,
            data_cfg={"aggregate_transition_shuffle": True},
        )


def test_clean50_modality_action_order_and_horizon():
    config = ROBOT_TYPE_CONFIG_MAP["robotwin_clean50"]
    assert DATASET_NAMED_MIXTURES["robotwin_clean50"] == [
        ("robotwin_clean50", 1.0, "robotwin_clean50")
    ]
    assert config.action_indices == list(range(32))
    assert "state" not in config.modality_config()
    train_cfg = OmegaConf.load("examples/simBenchmarks/Robotwin/train_files/qwenpi_clean50.yaml")
    assert train_cfg.datasets.vla_data.per_device_batch_size == 16
    assert train_cfg.trainer.gradient_accumulation_steps == 2
    assert train_starvla.accelerator.gradient_state.plugin_kwargs["sync_each_batch"] is True
    assert train_starvla.accelerator.step_scheduler_with_optimizer is False

    modality = json.loads(
        Path("examples/simBenchmarks/Robotwin/train_files/modality_clean50.json").read_text()
    )
    raw = np.arange(14)
    packed = np.concatenate(
        [raw[modality["action"][key.removeprefix("action.")]["start"]:modality["action"][key.removeprefix("action.")]["end"]]
         for key in config.action_keys]
    )
    np.testing.assert_array_equal(packed, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13])


def test_absolute_action_tail_padding_repeats_episode_end():
    action = np.arange(3 * 14, dtype=np.float32).reshape(3, 14)
    padded = LeRobotSingleDataset.retrieve_data_and_pad(
        object(), action, np.arange(32), max_length=3, padding_strategy="first_last"
    )
    assert padded.shape == (32, 14)
    np.testing.assert_array_equal(padded[:3], action)
    np.testing.assert_array_equal(padded[3:], np.repeat(action[-1:], 29, axis=0))


def test_transition_permutation_is_without_replacement_epoch_varying_and_reproducible():
    first = _mixture(_DummyAggregate())
    epoch0 = first.transition_permutation.copy()
    assert sorted(epoch0.tolist()) == list(range(100))
    first.set_epoch(1)
    assert sorted(first.transition_permutation.tolist()) == list(range(100))
    assert not np.array_equal(epoch0, first.transition_permutation)

    second = _mixture(_DummyAggregate())
    np.testing.assert_array_equal(epoch0, second.transition_permutation)
    assert first.datasets[0].epoch == 1


def test_qwen3_enables_non_reentrant_gradient_checkpointing_and_sdpa():
    fake_model = nn.Linear(1, 1)
    fake_model.config = SimpleNamespace(
        text_config=SimpleNamespace(hidden_size=16), use_cache=True
    )
    fake_model.is_gradient_checkpointing = False
    checkpointing_kwargs = {}

    def enable_checkpointing(**kwargs):
        checkpointing_kwargs.update(kwargs)
        fake_model.is_gradient_checkpointing = True

    fake_model.gradient_checkpointing_enable = enable_checkpointing
    fake_model.enable_input_require_grads = mock.Mock()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side=None))
    cfg = OmegaConf.create(
        {"framework": {"qwenvl": {"base_vlm": "Qwen3-VL-test", "gradient_checkpointing": True}}}
    )
    with (
        mock.patch("starVLA.model.modules.vlm.QWen3.Qwen3VLForConditionalGeneration.from_pretrained", return_value=fake_model),
        mock.patch("starVLA.model.modules.vlm.QWen3.AutoProcessor.from_pretrained", return_value=processor),
    ):
        interface = _QWen3_VL_Interface(cfg)

    assert checkpointing_kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    assert fake_model.enable_input_require_grads.call_count == 1
    assert fake_model.config.use_cache is False
    assert interface.resolved_attn_implementation == "sdpa"


def test_horizon_compatibility_derives_future_window_31():
    cfg = OmegaConf.create({"framework": {"action_model": {"action_horizon": 32}}})
    apply_config_compat(cfg)
    assert cfg.framework.action_model.future_action_window_size == 31


def test_strict_batch_contract_and_missing_full_state_failure(tmp_path):
    backbone_model = SimpleNamespace(is_gradient_checkpointing=True, config=SimpleNamespace(use_cache=False))
    backbone = SimpleNamespace(
        requested_attn_implementation="sdpa",
        resolved_attn_implementation="sdpa",
        gradient_checkpointing_enabled=True,
        model=backbone_model,
    )
    model = SimpleNamespace(
        action_horizon=32,
        action_model=SimpleNamespace(action_horizon=32),
        qwen_vl_interface=backbone,
    )
    accelerator = SimpleNamespace(
        num_processes=8,
        gradient_accumulation_steps=2,
        is_main_process=True,
        unwrap_model=lambda wrapped: wrapped,
    )
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "framework": {"action_model": {"future_action_window_size": 31}},
            "trainer": {"strict_contract": True, "is_resume": True},
        }
    )
    trainer = train_starvla.VLATrainer.__new__(train_starvla.VLATrainer)
    trainer.config = cfg
    trainer.model = model
    trainer.accelerator = accelerator
    trainer.total_batch_size = 256
    trainer._contract_checked = False
    sample = {
        "action": np.zeros((32, 14), dtype=np.float16),
        "image": [Image.new("RGB", (224, 224)) for _ in range(3)],
        "lang": "task",
    }
    trainer._validate_contract([sample.copy() for _ in range(16)])
    assert trainer._contract_checked

    trainer.model = object()
    with pytest.raises(FileNotFoundError, match="complete Accelerate state"):
        trainer._init_checkpointing()


def test_tiny_full_state_restore_and_next_batch_continuity(tmp_path):
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        import torch
        from accelerate import Accelerator
        from torch import nn
        from torch.utils.data import DataLoader

        state_dir = Path({str(tmp_path)!r}) / "state"
        accelerator = Accelerator(cpu=True)
        model = nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
        dataloader = DataLoader(torch.arange(4, dtype=torch.float32).view(-1, 1), batch_size=1)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
        first = next(iter(dataloader))
        accelerator.backward(model(first).sum())
        optimizer.step()
        scheduler.step()
        saved_weight = accelerator.unwrap_model(model).weight.detach().clone()
        accelerator.save_state(str(state_dir))
        with torch.no_grad():
            accelerator.unwrap_model(model).weight.add_(10)
        accelerator.load_state(str(state_dir))
        torch.testing.assert_close(accelerator.unwrap_model(model).weight, saved_weight)
        next_batch = next(iter(accelerator.skip_first_batches(dataloader, num_batches=1)))
        torch.testing.assert_close(next_batch.cpu(), torch.tensor([[1.0]]))
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
