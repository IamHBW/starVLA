import contextlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP
from starVLA.model.framework.VLM4A.QwenPI import Qwen_PI
from starVLA.model.framework.share_tools import apply_config_compat


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILES = ROOT / "examples/simBenchmarks/LIBERO/train_files"
EVAL_FILE = ROOT / "examples/simBenchmarks/LIBERO/eval_files/qwen3_pi_4in1_30k_eval.py"
SPEC = importlib.util.spec_from_file_location("qwen3_pi_4in1_30k_eval", EVAL_FILE)
EVAL = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(EVAL)


def test_h32_registry_is_dedicated_and_equal_weighted():
    assert ROBOT_TYPE_CONFIG_MAP["libero_franka"].action_indices == list(range(8))
    assert ROBOT_TYPE_CONFIG_MAP["libero_franka_h32"].action_indices == list(range(32))
    mixture = DATASET_NAMED_MIXTURES["libero_all_h32"]
    assert len(mixture) == 4
    assert all(weight == 1.0 and robot_type == "libero_franka_h32" for _, weight, robot_type in mixture)


class _CaptureHead(torch.nn.Module):
    def forward(self, vl_embeddings, actions, state, encoder_attention_mask=None):
        self.batch_sizes = (actions.shape[0], *(hidden.shape[0] for hidden in vl_embeddings))
        return actions.square().mean()


def test_qwenpi_forward_honors_repeat_from_config():
    model = object.__new__(Qwen_PI)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        framework=SimpleNamespace(action_model={"repeated_diffusion_steps": 8})
    )
    model.action_horizon = 32
    model.action_model = _CaptureHead()
    model._encode_vl_hidden_states = lambda images, instructions: (
        [torch.zeros(2, 3, 4), torch.zeros(2, 3, 4)],
        torch.ones(2, 3, dtype=torch.bool),
    )
    examples = [
        {"image": [], "lang": "task", "action": np.zeros((32, 7), dtype=np.float32)}
        for _ in range(2)
    ]
    model.forward(examples)
    assert model.action_model.batch_sizes == (16, 16, 16)


def test_train_step_keeps_all_accumulated_microbatches(monkeypatch):
    from starVLA.training.train_starvla import VLATrainer

    class Accelerator:
        sync_gradients = False
        microbatch = 0

        @contextlib.contextmanager
        def accumulate(self, _model):
            self.microbatch += 1
            self.sync_gradients = self.microbatch % 4 == 0
            yield

        @staticmethod
        def backward(loss):
            loss.backward()

    class Optimizer:
        def __init__(self, parameter, accelerator):
            self.base = torch.optim.SGD([parameter], lr=0.1)
            self.accelerator = accelerator
            self.synced_grad = None

        def step(self):
            if self.accelerator.sync_gradients:
                self.synced_grad = model.weight.grad.detach().clone()
                self.base.step()

        def zero_grad(self):
            if self.accelerator.sync_gradients:
                self.base.zero_grad()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, value):
            return {"action_loss": self.weight * value}

    class Scheduler:
        steps = 0

        def step(self):
            self.steps += 1

    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: contextlib.nullcontext())
    model = Model()
    accelerator = Accelerator()
    trainer = object.__new__(VLATrainer)
    trainer.model = model
    trainer.optimizer = Optimizer(model.weight, accelerator)
    trainer.lr_scheduler = Scheduler()
    trainer.accelerator = accelerator
    trainer.config = SimpleNamespace(trainer=SimpleNamespace(strict_contract=False, gradient_clipping=None))
    trainer._contract_checked = False

    for value in (1.0, 2.0, 3.0, 4.0):
        trainer._train_step(value)

    assert trainer.optimizer.synced_grad.item() == pytest.approx(10.0)
    assert model.weight.item() == pytest.approx(0.0, abs=1e-7)
    assert trainer.lr_scheduler.steps == 1


def test_train_config_and_launcher_contract(monkeypatch):
    monkeypatch.setenv("QWEN3_VL_SNAPSHOT", "/tmp/qwen3")
    monkeypatch.setenv("WANDB_ENTITY", "test-entity")
    monkeypatch.setenv("WANDB_PROJECT", "test-project")
    cfg = apply_config_compat(OmegaConf.load(TRAIN_FILES / "qwen3_pi_4in1_30k.yaml"), strict=True)
    assert cfg.framework.name == "QwenPI"
    assert cfg.framework.action_model.action_horizon == 32
    assert cfg.framework.action_model.future_action_window_size == 31
    assert cfg.framework.action_model.repeated_diffusion_steps == 8
    assert cfg.framework.action_model.num_inference_timesteps == 4
    assert cfg.framework.action_model.diffusion_model_cfg.use_canonical_forward is False
    assert cfg.datasets.vla_data.data_mix == "libero_all_h32"
    assert cfg.datasets.vla_data.include_state is False
    assert cfg.datasets.vla_data.per_device_batch_size == 8
    assert cfg.trainer.gradient_accumulation_steps == 4
    assert cfg.datasets.vla_data.per_device_batch_size * 8 * cfg.trainer.gradient_accumulation_steps == 256
    assert cfg.trainer.max_train_steps == 30000
    assert cfg.trainer.freeze_modules == ""
    assert "pretrained_checkpoint" not in cfg.trainer
    launcher = TRAIN_FILES / "qwen3_pi_4in1_30k.sh"
    source = launcher.read_text()
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in source
    assert source.count('bucket_size\"] = 100_000_000') == 2
    assert 'git merge-base --is-ancestor "${starvla_base_revision}" HEAD' in source
    assert '--deepspeed_multinode_launcher standard' in source
    assert '--num_machines "${num_nodes}"' in source
    assert '--num_processes "${total_processes}"' in source
    assert '--machine_rank "${node_rank}"' in source
    assert '--trainer.gradient_accumulation_steps "${gradient_accumulation}"' in source
    assert "unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT" in source
    assert "validate-checkpoint" in source
    subprocess.run(["bash", "-n", launcher], check=True, env=os.environ)
    eval_launcher = ROOT / "examples/simBenchmarks/LIBERO/eval_files/qwen3_pi_4in1_30k_eval.sh"
    eval_source = eval_launcher.read_text()
    assert 'export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"' in eval_source
    assert "--config_override datasets.vla_data.obs_image_size=[224,224]" in eval_source
    subprocess.run(["bash", "-n", eval_launcher], check=True, env=os.environ)
    h8_launcher = TRAIN_FILES / "qwen3_pi_4in1_30k_h8_r8_gb128.sh"
    h8_source = h8_launcher.read_text()
    for expected in (
        "QWEN3_PI_ACTION_HORIZON=8",
        "QWEN3_PI_REPLAN_STEPS=8",
        "QWEN3_PI_DATA_MIX=libero_all",
        "QWEN3_PI_GLOBAL_BATCH_SIZE=128",
        "GRADIENT_ACCUMULATION_STEPS=1",
    ):
        assert expected in h8_source
    subprocess.run(["bash", "-n", h8_launcher], check=True, env=os.environ)
    h8_eval = ROOT / "examples/simBenchmarks/LIBERO/eval_files/qwen3_pi_4in1_30k_h8_r8_gb128_eval.sh"
    assert "QWEN3_PI_EVAL_PROFILE=h8_r8_gb128" in h8_eval.read_text()
    subprocess.run(["bash", "-n", h8_eval], check=True, env=os.environ)


def _make_gate_fixture(
    tmp_path,
    *,
    eval_module=EVAL,
    action_horizon=32,
    replan_steps=24,
    data_mix="libero_all_h32",
    world_size=8,
    gradient_accumulation=4,
    global_batch=256,
):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoints/steps_30000_pytorch_model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"independent-30k-weights")
    stats = run_dir / "dataset_statistics.json"
    stats.write_text("{}\n")
    cfg = {
        "framework": {
            "action_model": {
                "action_dim": 7,
                "action_horizon": action_horizon,
                "repeated_diffusion_steps": 8,
                "num_inference_timesteps": 4,
                "diffusion_model_cfg": {"use_canonical_forward": False},
            }
        },
        "datasets": {
            "vla_data": {
                "data_mix": data_mix,
                "obs_image_size": [224, 224],
                "per_device_batch_size": 8,
                "include_state": False,
            }
        },
        "trainer": {"max_train_steps": 30000, "gradient_accumulation_steps": gradient_accumulation},
    }
    OmegaConf.save(OmegaConf.create(cfg), run_dir / "resolved_input_config.yaml")
    validation = {
        "optimizer_step": 30000,
        "actions_finite": True,
        "normalized_action_shape": [1, action_horizon, 7],
        "unnormalized_action_shape": [action_horizon, 7],
        "weights_sha256": {"checkpoints/steps_30000_pytorch_model.pt": eval_module.sha256(checkpoint)},
        "dataset_statistics_sha256": eval_module.sha256(stats),
    }
    (run_dir / "checkpoint_validation.json").write_text(json.dumps(validation))
    event = {
        "git_sha": eval_module.STARVLA_REVISION,
        "model": {"revision": eval_module.QWEN_REVISION},
        "datasets": {name: {"revision": revision} for name, revision in eval_module.DATASET_REVISIONS.items()},
        "user_overrides": {"action_horizon": action_horizon, "replan_steps": replan_steps},
        "batch": {
            "per_device": 8,
            "world_size": world_size,
            "gradient_accumulation": gradient_accumulation,
            "global": global_batch,
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps({"events": [event]}))
    return run_dir, checkpoint


def test_checkpoint_gate_rejects_100k_and_accepts_30k(tmp_path, monkeypatch):
    run_dir, checkpoint = _make_gate_fixture(tmp_path)
    release = run_dir / "checkpoints/steps_100000_pytorch_model.pt"
    release.write_bytes(b"release")
    with pytest.raises(ValueError, match="only"):
        EVAL.checkpoint_gate(run_dir, release, run_dir / "eval_gate.json")
    gate = EVAL.checkpoint_gate(run_dir, checkpoint, run_dir / "eval_gate.json")
    assert gate["complete"] is True
    assert gate["optimizer_step"] == 30000
    assert gate["checkpoint_sha256"] == EVAL.sha256(checkpoint)
    assert gate["starvla_source_revision"] == EVAL.STARVLA_REVISION

    monkeypatch.setenv("QWEN3_PI_EVAL_PROFILE", "h8_r8_gb128")
    spec = importlib.util.spec_from_file_location("qwen3_pi_4in1_30k_eval_h8", EVAL_FILE)
    h8_eval = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(h8_eval)

    valid_dir, valid_checkpoint = _make_gate_fixture(
        tmp_path / "h8_valid",
        eval_module=h8_eval,
        action_horizon=8,
        replan_steps=8,
        data_mix="libero_all",
        world_size=16,
        gradient_accumulation=1,
        global_batch=128,
    )
    assert h8_eval.checkpoint_gate(valid_dir, valid_checkpoint, valid_dir / "eval_gate.json")["complete"]

    def reject(name, mutate):
        run_dir, candidate = _make_gate_fixture(
            tmp_path / name,
            eval_module=h8_eval,
            action_horizon=8,
            replan_steps=8,
            data_mix="libero_all",
            world_size=16,
            gradient_accumulation=1,
            global_batch=128,
        )
        mutate(run_dir)
        with pytest.raises(ValueError, match="contract"):
            h8_eval.checkpoint_gate(run_dir, candidate, run_dir / "eval_gate.json")

    def change_config(path, key, value):
        config_path = path / "resolved_input_config.yaml"
        config = OmegaConf.load(config_path)
        OmegaConf.update(config, key, value)
        OmegaConf.save(config, config_path)

    def change_manifest(path, key, value):
        manifest_path = path / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["events"][-1][key] = value
        manifest_path.write_text(json.dumps(manifest))

    reject("h8_bad_horizon", lambda path: change_config(path, "framework.action_model.action_horizon", 32))
    reject("h8_bad_mix", lambda path: change_config(path, "datasets.vla_data.data_mix", "libero_all_h32"))
    reject(
        "h8_bad_replan",
        lambda path: change_manifest(path, "user_overrides", {"action_horizon": 8, "replan_steps": 7}),
    )
    reject(
        "h8_bad_batch",
        lambda path: change_manifest(
            path,
            "batch",
            {"per_device": 8, "world_size": 16, "gradient_accumulation": 1, "global": 256},
        ),
    )


def test_checkpoint_gate_accepts_equivalent_two_node_batch(tmp_path):
    run_dir, checkpoint = _make_gate_fixture(tmp_path, world_size=16, gradient_accumulation=2)
    gate = EVAL.checkpoint_gate(run_dir, checkpoint, run_dir / "eval_gate.json")
    assert gate["training_batch"] == {
        "per_device": 8,
        "world_size": 16,
        "gradient_accumulation": 2,
        "global": 256,
    }


def test_eval_executes_24_of_each_32_action_chunk():
    chunk = np.arange(32 * 7, dtype=np.float32).reshape(1, 32, 7)
    executed = EVAL.StarAdapter.execution_prefix(chunk)
    assert executed.shape == (24, 7)
    np.testing.assert_array_equal(executed, chunk[0, :24])


def _result_row(unit_id):
    row = {
        "dataset": "standard",
        "mode": "suite-smoke",
        "unit_id": unit_id,
        "slot": unit_id,
        "suite": "libero_spatial",
        "task_index": unit_id,
        "episode_index": 0,
        "classification_id": "",
        "task_name": f"task-{unit_id}",
        "category": "",
        "difficulty_level": "",
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "normalization_sha256": "stats",
        "gate_sha256": "gate",
    }
    return row


def _result_record(row):
    return {
        **row,
        "status": "completed",
        "success": False,
        "video_required": False,
        "video_path": None,
        "action_horizon": 32,
        "replan_steps": 24,
        "num_inference_steps": 4,
        "finite_action_count": 1,
    }


def test_result_gate_rejects_deleted_and_duplicate_ids(tmp_path):
    rows = [_result_row(0), _result_row(1)]
    gate = {
        "checkpoint_sha256": "checkpoint",
        "config_sha256": "config",
        "normalization_sha256": "stats",
    }
    paths = []
    for row in rows:
        path = EVAL.result_path(tmp_path, row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_result_record(row)))
        paths.append(path)
    records, errors = EVAL.validate_result_set(rows, tmp_path, gate)
    assert len(records) == 2 and not any(errors.values())

    paths[0].unlink()
    _, errors = EVAL.validate_result_set(rows, tmp_path, gate)
    assert errors["missing_unit_ids"] == [0]

    paths[0].write_text(json.dumps(_result_record(rows[0])))
    duplicate = tmp_path / "results/duplicate/result.json"
    duplicate.parent.mkdir(parents=True)
    shutil.copyfile(paths[0], duplicate)
    _, errors = EVAL.validate_result_set(rows, tmp_path, gate)
    assert errors["duplicate_unit_ids"] == [0]
