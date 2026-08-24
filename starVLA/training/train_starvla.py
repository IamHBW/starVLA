# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, setup_optimizer_and_scheduler, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
gradient_accumulation_plugin = GradientAccumulationPlugin(
    num_steps=int(os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1")),
    sync_each_batch=True,
)
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    gradient_accumulation_plugin=gradient_accumulation_plugin,
    step_scheduler_with_optimizer=False,
)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
        fused=True,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # Strip keys unknown to transformers' get_scheduler before passing kwargs.
    sched_kwargs = {k: v for k, v in cfg.trainer.scheduler_specific_kwargs.items()}
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=sched_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self.vla_epoch_count = 0
        self.data_batches_in_epoch = 0
        self._contract_checked = False

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader, self.lr_scheduler = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            self.lr_scheduler,
        )

        if self.resume_from_checkpoint:
            self._restore_full_state()

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if getattr(self.config.trainer, "validate_checkpoint", False):
            self.accelerator.wait_for_everyone()
            return
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        wandb_error = None
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
                wandb_error = str(exc)
        if dist.is_initialized():
            error_payload = [wandb_error]
            dist.broadcast_object_list(error_payload, src=0)
            wandb_error = error_payload[0]
        if wandb_error and getattr(self.config.trainer, "wandb_required", False):
            raise RuntimeError(f"Required W&B online initialization failed: {wandb_error}")
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        is_resume = bool(getattr(self.config.trainer, "is_resume", False))
        full_yaml_path = output_dir / ("config.resume.yaml" if is_resume else "config.full.yaml")
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig) and not is_resume:
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = None

        if is_resume:
            self.resume_from_checkpoint = self._latest_full_state()
            if self.resume_from_checkpoint is None:
                raise FileNotFoundError(
                    f"Resume requires a complete Accelerate state under {self.checkpoint_dir}"
                )
            return

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _latest_full_state(self):
        states = []
        for path in Path(self.checkpoint_dir).glob("full_state_step_*"):
            match = re.fullmatch(r"full_state_step_(\d+)", path.name)
            if path.is_dir() and match and (path / "progress.json").is_file():
                states.append((int(match.group(1)), path))
        return max(states, default=(0, None))[1]

    def _restore_full_state(self):
        progress_path = Path(self.resume_from_checkpoint) / "progress.json"
        progress = json.loads(progress_path.read_text())
        self.completed_steps = int(progress["optimizer_step"])
        self.vla_epoch_count = int(progress["data_epoch"])
        self.data_batches_in_epoch = int(progress["data_batch_offset"])
        self._set_data_epoch(self.vla_epoch_count)
        self.accelerator.load_state(str(self.resume_from_checkpoint))
        logger.info(
            "Restored full state at optimizer_step=%d data_epoch=%d data_batch_offset=%d",
            self.completed_steps,
            self.vla_epoch_count,
            self.data_batches_in_epoch,
        )

    def _set_data_epoch(self, epoch):
        dataloader = self.vla_train_dataloader
        if callable(getattr(getattr(dataloader, "dataset", None), "set_epoch", None)):
            dataloader.dataset.set_epoch(epoch)
        if callable(getattr(getattr(dataloader, "sampler", None), "set_epoch", None)):
            dataloader.sampler.set_epoch(epoch)

    def _save_checkpoint(self):
        """Save current training state."""
        state_dict = self.accelerator.get_state_dict(self.model)
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

        if getattr(self.config.trainer, "full_state_checkpoints", False):
            state_path = Path(self.checkpoint_dir) / f"full_state_step_{self.completed_steps}"
            self.accelerator.save_state(str(state_path))
            if self.accelerator.is_main_process:
                progress = {
                    "optimizer_step": self.completed_steps,
                    "data_epoch": self.vla_epoch_count,
                    "data_batch_offset": self.data_batches_in_epoch,
                    "cumulative_train_samples": self.completed_steps * self.total_batch_size,
                }
                (state_path / "progress.json").write_text(json.dumps(progress, indent=2) + "\n")
                for old_path in Path(self.checkpoint_dir).glob("full_state_step_*"):
                    if old_path != state_path and old_path.is_dir() and old_path.name.startswith("full_state_step_"):
                        shutil.rmtree(old_path)
            self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            metrics["epoch"] = round(
                self.completed_steps * self.accelerator.gradient_accumulation_steps
                / len(self.vla_train_dataloader),
                2,
            )
            metrics["optimizer_step"] = self.completed_steps
            metrics["cumulative_train_samples"] = self.completed_steps * self.total_batch_size
            if torch.cuda.is_available():
                metrics["memory/peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                metrics["memory/peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            with open(Path(self.config.output_dir) / "metrics.jsonl", "a") as metrics_file:
                metrics_file.write(json.dumps(metrics, sort_keys=True) + "\n")
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        if self.data_batches_in_epoch:
            skipped = self.accelerator.skip_first_batches(
                self.vla_train_dataloader, num_batches=self.data_batches_in_epoch
            )
            self.vla_iter = iter(skipped)
        else:
            self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            self.data_batches_in_epoch = 0
            batch_vla = next(self.vla_iter)

        self.data_batches_in_epoch += 1
        return batch_vla

    def _validate_contract(self, batch_vla):
        if self._contract_checked or not getattr(self.config.trainer, "strict_contract", False):
            return

        actions = np.asarray([sample["action"] for sample in batch_vla])
        model = self.accelerator.unwrap_model(self.model)
        backbone = model.qwen_vl_interface
        checks = {
            "num_processes": self.accelerator.num_processes,
            "per_device_batch_size": len(batch_vla),
            "gradient_accumulation_steps": self.accelerator.gradient_accumulation_steps,
            "global_batch_size": self.total_batch_size,
            "action_shape": list(actions.shape),
            "action_horizon": model.action_horizon,
            "action_head_horizon": model.action_model.action_horizon,
            "future_action_window_size": int(self.config.framework.action_model.future_action_window_size),
            "num_cameras": len(batch_vla[0]["image"]),
            "image_sizes": [list(image.size) for image in batch_vla[0]["image"]],
            "image_modes": [image.mode for image in batch_vla[0]["image"]],
            "include_state": any("state" in sample for sample in batch_vla),
            "requested_attention": backbone.requested_attn_implementation,
            "resolved_attention": backbone.resolved_attn_implementation,
            "gradient_checkpointing": bool(
                backbone.gradient_checkpointing_enabled and backbone.model.is_gradient_checkpointing
            ),
            "use_cache": backbone.model.config.use_cache,
        }
        expected = {
            "num_processes": 8,
            "per_device_batch_size": 16,
            "gradient_accumulation_steps": 2,
            "global_batch_size": 256,
            "action_shape": [16, 32, 14],
            "action_horizon": 32,
            "action_head_horizon": 32,
            "future_action_window_size": 31,
            "num_cameras": 3,
            "image_sizes": [[224, 224]] * 3,
            "image_modes": ["RGB"] * 3,
            "include_state": False,
            "resolved_attention": "sdpa",
            "gradient_checkpointing": True,
            "use_cache": False,
        }
        mismatches = {key: (checks.get(key), value) for key, value in expected.items() if checks.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Clean50 training contract mismatch: {mismatches}")
        if self.accelerator.is_main_process:
            (Path(self.config.output_dir) / "resolved_contract.json").write_text(
                json.dumps(checks, indent=2, sort_keys=True) + "\n"
            )
        self._contract_checked = True

    def validate_checkpoint(self):
        if not self.resume_from_checkpoint:
            raise RuntimeError("Checkpoint validation requires trainer.is_resume=true and a full state")
        self._create_data_iterators()
        batch_vla = self._get_next_batch()
        self._validate_contract(batch_vla)
        output = self.accelerator.unwrap_model(self.model).predict_action(examples=batch_vla)
        predicted_shape = list(np.asarray(output["normalized_actions"]).shape)
        if predicted_shape != [16, 32, 14]:
            raise RuntimeError(f"Unexpected checkpoint prediction shape: {predicted_shape}")
        if self.accelerator.is_main_process:
            result = {
                "optimizer_step": self.completed_steps,
                "full_state": str(self.resume_from_checkpoint),
                "predicted_action_shape": predicted_shape,
            }
            (Path(self.config.output_dir) / "checkpoint_validation.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
        self.accelerator.wait_for_everyone()

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
                eval_interval = int(getattr(self.config.trainer, "eval_interval", 0))
                if eval_interval > 0 and self.completed_steps % eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)
                step_metrics["timing/data"] = t_end_data - t_start_data
                step_metrics["timing/model"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)
                if self.completed_steps % self.config.trainer.save_interval == 0:
                    self._save_checkpoint()

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Run simple action-eval on current batch and attach score to metrics."""
        examples = self._get_next_batch()
        actions = [example["action"] for example in examples]
        output_dict = self.accelerator.unwrap_model(self.model).predict_action(
            examples=examples, use_ddim=True, num_ddim_steps=20
        )

        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            actions = np.array(actions)
            num_pots = np.prod(actions.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, actions)
            step_metrics["mse_score"] = score / num_pots

        del examples
        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step."""
        self._validate_contract(batch_vla)
        with self.accelerator.accumulate(self.model):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss

            if not torch.isfinite(total_loss).all():
                raise FloatingPointError(f"Non-finite loss at optimizer step {self.completed_steps}")

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            # Only step the LR scheduler when gradients are actually synced
            # (i.e., not mid-accumulation). Without this guard the scheduler
            # runs gradient_accumulation_steps times faster than intended,
            # causing warmup to end too early and cosine decay to bottom out
            # at min_lr well before max_train_steps is reached.
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        return {
            "action_dit_loss": action_loss.item(),
        }

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process and getattr(self.config.trainer, "save_final_model", True):
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    set_seed(int(cfg.seed))
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    if getattr(cfg.trainer, "validate_checkpoint", False):
        trainer.validate_checkpoint()
    else:
        trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
