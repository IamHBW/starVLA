#!/usr/bin/env python3
"""Strict, resumable LIBERO and LIBERO-Plus evaluation for the 30K QwenPI run."""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import io
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
STANDARD_HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}
PLUS_COUNTS = {
    "libero_spatial": 2402,
    "libero_object": 2518,
    "libero_goal": 2591,
    "libero_10": 2519,
}
CATEGORY_COUNTS = {
    "Background Textures": 1076,
    "Camera Viewpoints": 1599,
    "Language Instructions": 1537,
    "Light Conditions": 1142,
    "Objects Layout": 1525,
    "Robot Initial States": 1550,
    "Sensor Noise": 1601,
}
CATEGORIES = tuple(CATEGORY_COUNTS)
DATASET_REVISIONS = {
    "libero_spatial_no_noops_1.0.0_lerobot": "bf14d6258218d12c2e3c1a3b9922e163cdf6455d",
    "libero_object_no_noops_1.0.0_lerobot": "15657dac2ad1c01b4e94bf54ab0493b46a8d63f9",
    "libero_goal_no_noops_1.0.0_lerobot": "222cf888ed360fad0a5f983748c1cc40743d43e7",
    "libero_10_no_noops_1.0.0_lerobot": "e1a223d30b896c1613f270a2bfc63d382b3de7e1",
}
STARVLA_REVISION = "02861ead680ea648c367ed41cf0d0976581f0467"
QWEN_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PLUS_CLASSIFICATION_SHA256 = "faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6"
PROFILES = {
    "h32_r24_gb256": {"action_horizon": 32, "replan_steps": 24, "global_batch": 256, "data_mix": "libero_all_h32"},
    "h8_r8_gb128": {"action_horizon": 8, "replan_steps": 8, "global_batch": 128, "data_mix": "libero_all"},
}
PROFILE_NAME = os.environ.get("QWEN3_PI_EVAL_PROFILE", "h32_r24_gb256")
try:
    PROFILE = PROFILES[PROFILE_NAME]
except KeyError as exc:
    raise ValueError(f"unknown QwenPI evaluation profile: {PROFILE_NAME}") from exc
ACTION_HORIZON = PROFILE["action_horizon"]
REPLAN_STEPS = PROFILE["replan_steps"]
GLOBAL_BATCH_SIZE = PROFILE["global_batch"]
DATA_MIX = PROFILE["data_mix"]
INFERENCE_STEPS = 4
SEED = 7
SETTLE_STEPS = 10
DUMMY_ACTION = [0.0] * 6 + [-1.0]
FIELDS = (
    "dataset",
    "mode",
    "unit_id",
    "slot",
    "suite",
    "task_index",
    "episode_index",
    "classification_id",
    "task_name",
    "category",
    "difficulty_level",
    "checkpoint_sha256",
    "config_sha256",
    "normalization_sha256",
    "gate_sha256",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(data: object, path: str | Path) -> None:
    atomic_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", path)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_checkpoint(run_dir: str | Path, data_root: str | Path) -> dict:
    """Strictly load the final model and prove it emits one finite action chunk."""
    import av
    import numpy as np
    import torch
    from PIL import Image

    from deployment.model_server.policy_norm_processor import PolicyNormProcessor
    from starVLA.model.framework.base_framework import baseframework

    run_dir = Path(run_dir)
    data_root = Path(data_root)
    checkpoint = run_dir / "checkpoints/steps_30000_pytorch_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = baseframework.from_pretrained(str(checkpoint)).to(torch.bfloat16).cuda().eval()
    dataset = data_root / "libero_spatial_no_noops_1.0.0_lerobot"

    def first_frame(pattern):
        path = next(dataset.glob(pattern))
        with av.open(str(path)) as container:
            return Image.fromarray(next(container.decode(video=0)).to_ndarray(format="rgb24"))

    task = json.loads((dataset / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    example = {
        "image": [
            first_frame("videos/*/observation.images.image/*.mp4"),
            first_frame("videos/*/observation.images.wrist_image/*.mp4"),
        ],
        "lang": task,
    }
    normalized = np.asarray(model.predict_action(examples=[example])["normalized_actions"])
    if normalized.shape != (1, ACTION_HORIZON, 7) or not np.isfinite(normalized).all():
        raise RuntimeError(f"Invalid normalized action: shape={normalized.shape}")
    unnormalized = PolicyNormProcessor(str(checkpoint), unnorm_key="franka").unapply_actions(normalized[0])
    if unnormalized.shape != (ACTION_HORIZON, 7) or not np.isfinite(unnormalized).all():
        raise RuntimeError(f"Invalid unnormalized action: shape={unnormalized.shape}")

    weights = {}
    for step in (10000, 20000, 30000):
        path = run_dir / f"checkpoints/steps_{step}_pytorch_model.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        weights[str(path.relative_to(run_dir))] = sha256(path)
    progress = json.loads((run_dir / "checkpoints/full_state_step_30000/progress.json").read_text())
    if progress["optimizer_step"] != 30000 or progress["cumulative_train_samples"] != 30000 * GLOBAL_BATCH_SIZE:
        raise RuntimeError(f"Invalid final progress: {progress}")
    result = {
        "optimizer_step": 30000,
        "checkpoint": str(checkpoint),
        "weights_sha256": weights,
        "dataset_statistics_sha256": sha256(run_dir / "dataset_statistics.json"),
        "normalized_action_shape": list(normalized.shape),
        "unnormalized_action_shape": list(unnormalized.shape),
        "actions_finite": True,
    }
    atomic_json(result, run_dir / "checkpoint_validation.json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def checkpoint_gate(run_dir: str | Path, checkpoint: str | Path, output: str | Path) -> dict:
    """Validate the exact 30K checkpoint and emit the immutable eval gate."""
    from omegaconf import OmegaConf

    run_dir = Path(run_dir).resolve()
    checkpoint = Path(checkpoint).resolve()
    expected_checkpoint = run_dir / "checkpoints/steps_30000_pytorch_model.pt"
    if checkpoint != expected_checkpoint or checkpoint.name != "steps_30000_pytorch_model.pt":
        raise ValueError(f"evaluation accepts only {expected_checkpoint}, got {checkpoint}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    validation_path = run_dir / "checkpoint_validation.json"
    config_path = run_dir / "resolved_input_config.yaml"
    stats_path = run_dir / "dataset_statistics.json"
    manifest_path = run_dir / "run_manifest.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    events = manifest.get("events", [])
    if not events:
        raise ValueError("run manifest has no provenance event")
    event = events[-1]
    batch = event.get("batch", {})

    action = cfg["framework"]["action_model"]
    diffusion = action["diffusion_model_cfg"]
    vla = cfg["datasets"]["vla_data"]
    trainer = cfg["trainer"]
    contract = (
        action["action_dim"] == 7
        and action["action_horizon"] == ACTION_HORIZON
        and action["repeated_diffusion_steps"] == 8
        and action["num_inference_timesteps"] == INFERENCE_STEPS
        and diffusion["use_canonical_forward"] is False
        and vla["data_mix"] == DATA_MIX
        and vla["obs_image_size"] == [224, 224]
        and vla["per_device_batch_size"] == 8
        and vla["include_state"] is False
        and trainer["max_train_steps"] == 30000
        and batch.get("per_device") == vla["per_device_batch_size"]
        and batch.get("gradient_accumulation") == trainer["gradient_accumulation_steps"]
        and batch.get("world_size", 0) > 0
        and batch["per_device"] * batch["world_size"] * batch["gradient_accumulation"] == GLOBAL_BATCH_SIZE
        and batch.get("global") == GLOBAL_BATCH_SIZE
        and event.get("user_overrides") == {"action_horizon": ACTION_HORIZON, "replan_steps": REPLAN_STEPS}
    )
    if not contract:
        raise ValueError("resolved training config violates the frozen eval contract")
    if validation.get("optimizer_step") != 30000 or validation.get("actions_finite") is not True:
        raise ValueError("checkpoint validation did not prove finite 30K inference")
    if validation.get("normalized_action_shape") != [1, ACTION_HORIZON, 7]:
        raise ValueError("unexpected normalized action shape")
    if validation.get("unnormalized_action_shape") != [ACTION_HORIZON, 7]:
        raise ValueError("unexpected unnormalized action shape")

    checkpoint_hash = sha256(checkpoint)
    stats_hash = sha256(stats_path)
    weights = validation.get("weights_sha256", {})
    if weights.get("checkpoints/steps_30000_pytorch_model.pt") != checkpoint_hash:
        raise ValueError("30K checkpoint hash does not match post-train validation")
    if validation.get("dataset_statistics_sha256") != stats_hash:
        raise ValueError("normalization statistics hash mismatch")

    revisions = {name: item.get("revision") for name, item in event.get("datasets", {}).items()}
    if event.get("starvla_base_sha", event.get("git_sha")) != STARVLA_REVISION:
        raise ValueError("StarVLA base revision mismatch")
    if event.get("model", {}).get("revision") != QWEN_REVISION:
        raise ValueError("Qwen revision mismatch")
    if revisions != DATASET_REVISIONS:
        raise ValueError("training dataset revision set mismatch")

    gate = {
        "complete": True,
        "optimizer_step": 30000,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "normalization": str(stats_path),
        "normalization_key": "franka",
        "normalization_sha256": stats_hash,
        "checkpoint_validation_sha256": sha256(validation_path),
        "starvla_revision": STARVLA_REVISION,
        "starvla_source_revision": event["git_sha"],
        "qwen_revision": QWEN_REVISION,
        "dataset_revisions": DATASET_REVISIONS,
        "action_horizon": ACTION_HORIZON,
        "replan_steps": REPLAN_STEPS,
        "num_inference_steps": INFERENCE_STEPS,
        "use_canonical_forward": False,
        "image_size": [224, 224],
        "image_order": ["primary", "wrist"],
        "include_state": False,
        "training_batch": batch,
        "profile": PROFILE_NAME,
    }
    atomic_json(gate, output)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return gate


def load_gate(path: str | Path) -> dict:
    gate = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "complete": True,
        "optimizer_step": 30000,
        "action_horizon": ACTION_HORIZON,
        "replan_steps": REPLAN_STEPS,
        "num_inference_steps": INFERENCE_STEPS,
        "use_canonical_forward": False,
        "include_state": False,
        "starvla_revision": STARVLA_REVISION,
        "qwen_revision": QWEN_REVISION,
        "profile": PROFILE_NAME,
    }
    for key, value in required.items():
        if gate.get(key) != value:
            raise ValueError(f"invalid checkpoint gate field {key}: {gate.get(key)!r}")
    if gate.get("image_size") != [224, 224] or gate.get("image_order") != ["primary", "wrist"]:
        raise ValueError("invalid image contract in checkpoint gate")
    return gate


def load_catalog(dataset: str) -> tuple[dict[str, list[dict]], str]:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    labels = None
    classification_hash = ""
    if dataset == "plus":
        classification_path = Path(benchmark.__file__).with_name("task_classification.json")
        classification_hash = sha256(classification_path)
        if classification_hash != PLUS_CLASSIFICATION_SHA256:
            raise ValueError("LIBERO-Plus classification SHA256 mismatch")
        labels = json.loads(classification_path.read_text(encoding="utf-8"))

    catalog: dict[str, list[dict]] = {}
    for suite_name in SUITES:
        with contextlib.redirect_stdout(io.StringIO()):
            suite = benchmark_dict[suite_name]()
        expected = PLUS_COUNTS[suite_name] if dataset == "plus" else 10
        if suite.get_num_tasks() != expected:
            raise ValueError(f"unexpected {suite_name} task count: {suite.get_num_tasks()}")
        catalog[suite_name] = []
        for task_index in range(expected):
            task = suite.get_task(task_index)
            row = {
                "suite": suite_name,
                "task_index": task_index,
                "task_name": task.name,
                "classification_id": "",
                "category": "",
                "difficulty_level": "",
            }
            if labels is not None:
                label = labels[suite_name][task_index]
                if label["id"] != task_index + 1 or label["name"] != task.name:
                    raise ValueError(f"classification mismatch: {suite_name} task {task_index}")
                row.update(
                    classification_id=int(label["id"]),
                    category=label["category"],
                    difficulty_level=(
                        "unlabeled" if label.get("difficulty_level") is None else str(label["difficulty_level"])
                    ),
                )
            catalog[suite_name].append(row)
    return catalog, classification_hash


def _standard_full(catalog: dict[str, list[dict]]) -> list[dict]:
    return [
        {**catalog[suite][task_index], "episode_index": episode_index}
        for episode_index in range(50)
        for task_index in range(10)
        for suite in SUITES
    ]


def _plus_full(catalog: dict[str, list[dict]]) -> list[dict]:
    return [
        {**catalog[suite][task_index], "episode_index": 0}
        for task_index in range(max(PLUS_COUNTS.values()))
        for suite in SUITES
        if task_index < PLUS_COUNTS[suite]
    ]


def build_rows(dataset: str, mode: str, slots: int, gate_path: str | Path) -> tuple[list[dict], str]:
    if slots <= 0:
        raise ValueError("slots must be positive")
    gate = load_gate(gate_path)
    gate_hash = sha256(gate_path)
    catalog, classification_hash = load_catalog(dataset)
    force_slot = False
    if dataset == "standard":
        full = _standard_full(catalog)
        if mode == "suite-smoke":
            selected = [{**catalog[suite][0], "episode_index": 0} for suite in SUITES]
        elif mode == "slot-smoke":
            selected = full[:slots]
            force_slot = True
        elif mode == "full":
            selected = full
        else:
            raise ValueError(f"unsupported standard mode: {mode}")
    elif dataset == "plus":
        full = _plus_full(catalog)
        if mode == "category-smoke":
            first_by_category = {}
            for row in full:
                first_by_category.setdefault(row["category"], row)
            if set(first_by_category) != set(CATEGORIES):
                raise ValueError("Plus smoke cannot cover all seven categories")
            selected = [first_by_category[category] for category in CATEGORIES]
        elif mode == "full":
            selected = full
        else:
            raise ValueError(f"unsupported Plus mode: {mode}")
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    expected = {
        ("standard", "suite-smoke"): 4,
        ("standard", "slot-smoke"): slots,
        ("standard", "full"): 2000,
        ("plus", "category-smoke"): 7,
        ("plus", "full"): 10030,
    }[(dataset, mode)]
    if len(selected) != expected or len(
        {(row["suite"], row["task_index"], row["episode_index"]) for row in selected}
    ) != expected:
        raise AssertionError(f"manifest coverage mismatch: {len(selected)} != {expected}")

    rows = []
    for unit_id, item in enumerate(selected):
        rows.append(
            {
                "dataset": dataset,
                "mode": mode,
                "unit_id": unit_id,
                "slot": unit_id if force_slot else unit_id % slots,
                **item,
                "checkpoint_sha256": gate["checkpoint_sha256"],
                "config_sha256": gate["config_sha256"],
                "normalization_sha256": gate["normalization_sha256"],
                "gate_sha256": gate_hash,
            }
        )
    return rows, classification_hash


def write_manifest(dataset: str, mode: str, slots: int, gate: str | Path, output: str | Path) -> None:
    rows, classification_hash = build_rows(dataset, mode, slots, gate)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "classification_sha256": classification_hash or None,
                "dataset": dataset,
                "manifest": str(output),
                "mode": mode,
                "slots": dict(sorted(collections.Counter(row["slot"] for row in rows).items())),
                "units": len(rows),
            },
            indent=2,
            sort_keys=True,
        )
    )


def read_manifest(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("unexpected manifest header")
        rows = []
        for raw in reader:
            rows.append(
                {
                    **raw,
                    "unit_id": int(raw["unit_id"]),
                    "slot": int(raw["slot"]),
                    "task_index": int(raw["task_index"]),
                    "episode_index": int(raw["episode_index"]),
                    "classification_id": int(raw["classification_id"]) if raw["classification_id"] else "",
                }
            )
    if not rows:
        raise ValueError("empty manifest")
    dataset, mode = rows[0]["dataset"], rows[0]["mode"]
    if {(row["dataset"], row["mode"]) for row in rows} != {(dataset, mode)}:
        raise ValueError("manifest mixes datasets or modes")
    expected = {
        ("standard", "suite-smoke"): 4,
        ("standard", "full"): 2000,
        ("plus", "category-smoke"): 7,
        ("plus", "full"): 10030,
    }.get((dataset, mode))
    if mode == "slot-smoke":
        expected = max(row["slot"] for row in rows) + 1
        if {row["slot"] for row in rows} != set(range(expected)):
            raise ValueError("slot smoke does not cover every formal slot")
    identities = {(row["suite"], row["task_index"], row["episode_index"]) for row in rows}
    if expected is None or len(rows) != expected or len(identities) != expected:
        raise ValueError(f"invalid manifest coverage: {len(rows)} != {expected}")
    return rows


def result_path(run_dir: str | Path, row: dict) -> Path:
    return (
        Path(run_dir)
        / "results"
        / row["suite"]
        / f"task_{row['task_index']:04d}"
        / f"episode_{row['episode_index']:03d}"
        / "result.json"
    )


def record_matches(record: dict, row: dict, gate: dict, run_dir: str | Path) -> bool:
    video_required = record.get("video_required") is True
    video_ok = not video_required
    if video_required and isinstance(record.get("video_path"), str):
        video = Path(run_dir) / record["video_path"]
        video_ok = video.is_file() and video.stat().st_size > 0
    return (
        all(record.get(field) == row[field] for field in FIELDS)
        and record.get("gate_sha256") == row["gate_sha256"]
        and record.get("status") == "completed"
        and record.get("unit_id") == row["unit_id"]
        and record.get("suite") == row["suite"]
        and record.get("task_index") == row["task_index"]
        and record.get("episode_index") == row["episode_index"]
        and isinstance(record.get("success"), bool)
        and record.get("checkpoint_sha256") == gate["checkpoint_sha256"] == row["checkpoint_sha256"]
        and record.get("config_sha256") == gate["config_sha256"] == row["config_sha256"]
        and record.get("normalization_sha256") == gate["normalization_sha256"] == row["normalization_sha256"]
        and record.get("action_horizon") == ACTION_HORIZON
        and record.get("replan_steps") == REPLAN_STEPS
        and record.get("num_inference_steps") == INFERENCE_STEPS
        and isinstance(record.get("finite_action_count"), int)
        and record["finite_action_count"] > 0
        and video_ok
    )


def is_complete(path: str | Path, row: dict, gate: dict, run_dir: str | Path) -> bool:
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return record_matches(record, row, gate, run_dir)


class StarAdapter:
    def __init__(self, port: int) -> None:
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy("127.0.0.1", port)
        self.metadata = self.client.get_server_metadata()
        if self.metadata.get("action_chunk_size") != ACTION_HORIZON:
            raise ValueError(f"server action horizon mismatch: {self.metadata}")
        if self.metadata.get("training_data_mix") != DATA_MIX:
            raise ValueError(f"server data mix mismatch: {self.metadata}")
        if self.metadata.get("training_obs_image_size") != [224, 224]:
            raise ValueError(f"server image size mismatch: {self.metadata}")
        if "franka" not in self.metadata.get("available_unnorm_keys", []):
            raise ValueError(f"server lacks franka normalization: {self.metadata}")
        self.actions: deque = deque()
        self.finite_action_count = 0
        self.predict_calls = 0

    def reset(self) -> None:
        self.actions.clear()
        self.finite_action_count = 0
        self.predict_calls = 0

    @staticmethod
    def _resize(image):
        import numpy as np
        from PIL import Image

        array = np.asarray(image)
        if array.shape[:2] != (224, 224):
            array = np.asarray(Image.fromarray(array).resize((224, 224), Image.BILINEAR))
        return array

    @staticmethod
    def execution_prefix(chunk):
        import numpy as np

        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape != (1, ACTION_HORIZON, 7) or not np.isfinite(chunk).all():
            raise ValueError(f"invalid action chunk shape/value: {chunk.shape}")
        return chunk[0, :REPLAN_STEPS]

    def action(self, obs: dict, prompt: str):
        import numpy as np

        if not self.actions:
            primary = self._resize(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
            wrist = self._resize(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]))
            response = self.client.predict_action(
                {
                    "examples": [{"image": [primary, wrist], "lang": str(prompt)}],
                    "unnorm_key": "franka",
                    "do_sample": False,
                    "use_ddim": True,
                    "num_ddim_steps": INFERENCE_STEPS,
                }
            )
            if response.get("status") != "ok":
                raise RuntimeError(f"policy server error: {response}")
            chunk = np.asarray(response.get("data", {}).get("actions"), dtype=np.float32)
            self.actions.extend(self.execution_prefix(chunk))
            self.predict_calls += 1
        raw = np.asarray(self.actions.popleft(), dtype=np.float32)
        action = np.concatenate((raw[:6], np.asarray([1.0 - 2.0 * (raw[6] > 0.5)], dtype=np.float32)))
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("non-finite or malformed LIBERO action")
        self.finite_action_count += 1
        return action


def patch_legacy_renderer() -> None:
    from robosuite.utils import binding_utils

    original = binding_utils.MjRenderContext.render
    if getattr(original, "_qwen3_pi_30k_patched", False):
        return

    def render_with_current_context(self, *args, **kwargs):
        self.gl_ctx.make_current()
        return original(self, *args, **kwargs)

    render_with_current_context._qwen3_pi_30k_patched = True
    binding_utils.MjRenderContext.render = render_with_current_context


def save_video(frames: list, path: str | Path) -> None:
    import imageio
    import numpy as np

    if not frames:
        raise ValueError("no video frames")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.mp4")
    try:
        imageio.mimsave(str(temp), [np.asarray(frame, dtype=np.uint8) for frame in frames], fps=10)
        if not temp.is_file() or temp.stat().st_size == 0:
            raise RuntimeError("video encoder produced an empty file")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def evaluate(args) -> None:
    import numpy as np
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    gate = load_gate(args.gate)
    rows = read_manifest(args.manifest)
    if sha256(args.gate) != rows[0]["gate_sha256"]:
        raise ValueError("manifest checkpoint gate hash mismatch")
    selected = [row for row in rows if row["slot"] == args.slot]
    if not selected:
        raise ValueError(f"slot {args.slot} has no work")
    if rows[0]["dataset"] == "standard":
        patch_legacy_renderer()
    np.random.seed(SEED)
    adapter = StarAdapter(args.port)
    suites = {}
    completed = sum(is_complete(result_path(args.run_dir, row), row, gate, args.run_dir) for row in selected)
    atomic_text(f"{completed}\n", args.progress_file)
    print(f"slot={args.slot} units={len(selected)} resumed={completed}", flush=True)

    for ordinal, row in enumerate(selected, 1):
        output = result_path(args.run_dir, row)
        if is_complete(output, row, gate, args.run_dir):
            continue
        if row["suite"] not in suites:
            with contextlib.redirect_stdout(io.StringIO()):
                suites[row["suite"]] = benchmark.get_benchmark_dict()[row["suite"]]()
        suite = suites[row["suite"]]
        task = suite.get_task(row["task_index"])
        if task.name != row["task_name"]:
            raise ValueError(f"task identity changed: {row['suite']} {row['task_index']}")
        initial_states = suite.get_task_init_states(row["task_index"])
        if row["episode_index"] >= len(initial_states):
            raise ValueError(f"missing init state {row['episode_index']} for {task.name}")

        started = time.monotonic()
        started_utc = utc_now()
        env = None
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=suite.get_task_bddl_file_path(row["task_index"]),
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(SEED)
            adapter.reset()
            env.reset()
            obs = env.set_init_state(initial_states[row["episode_index"]])
            for _ in range(SETTLE_STEPS):
                obs, _, _, _ = env.step(DUMMY_ACTION)
            frames = []
            done = False
            success_step = None
            max_steps = STANDARD_HORIZONS[row["suite"]]
            for step in range(max_steps):
                if args.video_policy != "none":
                    frames.append(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]))
                action = adapter.action(obs, task.language)
                obs, _, done, _ = env.step(action.tolist())
                if done:
                    success_step = step + 1
                    break
            video_required = args.video_policy == "always" or (args.video_policy == "failure" and not done)
            video_relative = (
                Path("results")
                / row["suite"]
                / f"task_{row['task_index']:04d}"
                / f"episode_{row['episode_index']:03d}"
                / "episode.mp4"
            )
            if video_required:
                save_video(frames, Path(args.run_dir) / video_relative)
            record = {
                **row,
                "status": "completed",
                "success": bool(done),
                "success_step": success_step,
                "prompt": task.language,
                "seed": SEED,
                "init_state_index": row["episode_index"],
                "max_steps": max_steps,
                "settle_steps": SETTLE_STEPS,
                "action_horizon": ACTION_HORIZON,
                "replan_steps": REPLAN_STEPS,
                "discarded_actions_per_chunk": ACTION_HORIZON - REPLAN_STEPS,
                "num_inference_steps": INFERENCE_STEPS,
                "use_canonical_forward": False,
                "finite_action_count": adapter.finite_action_count,
                "prediction_calls": adapter.predict_calls,
                "normalization_key": gate["normalization_key"],
                "checkpoint": gate["checkpoint"],
                "starvla_revision": gate["starvla_revision"],
                "libero_revision": args.libero_revision,
                "server_metadata": adapter.metadata,
                "video_required": video_required,
                "video_path": video_relative.as_posix() if video_required else None,
                "started_utc": started_utc,
                "finished_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
            atomic_json(record, output)
            completed += 1
            atomic_text(f"{completed}\n", args.progress_file)
            print(
                f"unit {ordinal}/{len(selected)} id={row['unit_id']} success={bool(done)} "
                f"actions={adapter.finite_action_count} predictions={adapter.predict_calls}",
                flush=True,
            )
        except BaseException as exc:
            error = {
                **row,
                "status": "infra_error",
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "started_utc": started_utc,
                "finished_utc": utc_now(),
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
            error_path = (
                Path(args.run_dir)
                / "infra_errors"
                / row["suite"]
                / f"task_{row['task_index']:04d}"
                / f"episode_{row['episode_index']:03d}"
                / f"attempt_{time.time_ns()}_{os.getpid()}.json"
            )
            atomic_json(error, error_path)
            raise
        finally:
            if env is not None:
                env.close()


def metric(records: list[dict]) -> dict:
    successes = sum(record["success"] for record in records)
    return {"episodes": len(records), "successes": successes, "success_rate": successes / len(records)}


def grouped_metrics(records: list[dict], key: str) -> dict:
    groups = collections.defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return {name: metric(groups[name]) for name in sorted(groups)}


def validate_result_set(rows: list[dict], run_dir: str | Path, gate: dict) -> tuple[list[dict], dict]:
    run_dir = Path(run_dir)
    expected_by_id = {row["unit_id"]: row for row in rows}
    records = []
    invalid = []
    for path in sorted((run_dir / "results").glob("**/result.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            invalid.append(str(path))
            continue
        row = expected_by_id.get(record.get("unit_id"))
        if row is None or not record_matches(record, row, gate, run_dir):
            invalid.append(str(path))
        else:
            records.append(record)
    ids = [record["unit_id"] for record in records]
    duplicates = sorted(unit_id for unit_id, count in collections.Counter(ids).items() if count != 1)
    missing = sorted(set(expected_by_id) - set(ids))
    extras = sorted(set(ids) - set(expected_by_id))
    errors = {"missing_unit_ids": missing, "duplicate_unit_ids": duplicates, "extra_unit_ids": extras, "invalid_paths": invalid}
    return records, errors


def summarize(args) -> None:
    gate = load_gate(args.gate)
    rows = read_manifest(args.manifest)
    records, errors = validate_result_set(rows, args.run_dir, gate)
    run_dir = Path(args.run_dir)
    if any(errors.values()):
        incomplete = {"complete": False, "expected": len(rows), "found_valid": len(records), **errors}
        atomic_json(incomplete, run_dir / "incomplete.json")
        (run_dir / "FINALIZED").unlink(missing_ok=True)
        raise RuntimeError(f"incomplete or duplicate results: {incomplete}")

    dataset, mode = rows[0]["dataset"], rows[0]["mode"]
    summary = {
        "complete": True,
        "dataset": dataset,
        "mode": mode,
        "episodes": len(records),
        "overall": metric(records),
        "per_suite": grouped_metrics(records, "suite"),
        "manifest_sha256": sha256(args.manifest),
        "checkpoint_sha256": gate["checkpoint_sha256"],
        "config_sha256": gate["config_sha256"],
        "normalization_sha256": gate["normalization_sha256"],
        "protocol": {
            "action_horizon": ACTION_HORIZON,
            "replan_steps": REPLAN_STEPS,
            "num_inference_steps": INFERENCE_STEPS,
            "use_canonical_forward": False,
            "seed": SEED,
            "settle_steps": SETTLE_STEPS,
            "max_steps": STANDARD_HORIZONS,
        },
        "infra_retries": (
            len(list((run_dir / "infra_errors").glob("**/attempt_*.json")))
            + sum(
                max(0, len(path.read_text(encoding="utf-8").splitlines()) - 1)
                for path in (run_dir / "worker_restarts").glob("slot_*.tsv")
            )
        ),
        "videos": sum(record["video_path"] is not None for record in records),
    }
    if dataset == "standard":
        summary["per_task"] = grouped_metrics(records, "task_name")
        if mode == "full":
            if any(item["episodes"] != 500 for item in summary["per_suite"].values()):
                raise ValueError("standard suite coverage is not 500 episodes each")
            task_counts = collections.Counter((record["suite"], record["task_index"]) for record in records)
            if len(task_counts) != 40 or set(task_counts.values()) != {50}:
                raise ValueError("standard task coverage is not 50 trials each")
    else:
        summary["classification_sha256"] = PLUS_CLASSIFICATION_SHA256
        summary["per_category"] = grouped_metrics(records, "category")
        summary["per_difficulty"] = grouped_metrics(records, "difficulty_level")
        if mode == "full":
            observed_suites = {key: value["episodes"] for key, value in summary["per_suite"].items()}
            observed_categories = {key: value["episodes"] for key, value in summary["per_category"].items()}
            if observed_suites != PLUS_COUNTS or observed_categories != CATEGORY_COUNTS:
                raise ValueError("LIBERO-Plus suite/category coverage mismatch")

    atomic_json(summary, run_dir / "summary.json")
    report = (
        f"# QwenPI 30K {dataset} {mode}\n\n"
        f"- Complete: true\n- Episodes: {len(records)}\n"
        f"- Successes: {summary['overall']['successes']}\n"
        f"- Success rate: {summary['overall']['success_rate']:.6f}\n"
        f"- Checkpoint SHA256: `{gate['checkpoint_sha256']}`\n"
        f"- Action horizon/replan/inference: {ACTION_HORIZON}/{REPLAN_STEPS}/{INFERENCE_STEPS}\n"
        f"- Infrastructure retries: {summary['infra_retries']}\n"
    )
    atomic_text(report, run_dir / "report.md")
    (run_dir / "incomplete.json").unlink(missing_ok=True)
    atomic_text(utc_now() + "\n", run_dir / "FINALIZED")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    validation = commands.add_parser("validate-checkpoint")
    validation.add_argument("--run-dir", required=True)
    validation.add_argument("--data-root", required=True)

    gate = commands.add_parser("checkpoint-gate")
    gate.add_argument("--run-dir", required=True)
    gate.add_argument("--checkpoint", required=True)
    gate.add_argument("--output", required=True)

    manifest = commands.add_parser("build-manifest")
    manifest.add_argument("--dataset", choices=("standard", "plus"), required=True)
    manifest.add_argument(
        "--mode", choices=("suite-smoke", "slot-smoke", "category-smoke", "full"), required=True
    )
    manifest.add_argument("--slots", type=int, required=True)
    manifest.add_argument("--gate", required=True)
    manifest.add_argument("--output", required=True)

    evaluate_parser = commands.add_parser("eval")
    evaluate_parser.add_argument("--manifest", required=True)
    evaluate_parser.add_argument("--gate", required=True)
    evaluate_parser.add_argument("--slot", type=int, required=True)
    evaluate_parser.add_argument("--port", type=int, required=True)
    evaluate_parser.add_argument("--run-dir", required=True)
    evaluate_parser.add_argument("--progress-file", required=True)
    evaluate_parser.add_argument("--libero-revision", required=True)
    evaluate_parser.add_argument("--video-policy", choices=("always", "failure", "none"), required=True)

    summary_parser = commands.add_parser("summarize")
    summary_parser.add_argument("--manifest", required=True)
    summary_parser.add_argument("--gate", required=True)
    summary_parser.add_argument("--run-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "validate-checkpoint":
        validate_checkpoint(args.run_dir, args.data_root)
    elif args.command == "checkpoint-gate":
        checkpoint_gate(args.run_dir, args.checkpoint, args.output)
    elif args.command == "build-manifest":
        write_manifest(args.dataset, args.mode, args.slots, args.gate, args.output)
    elif args.command == "eval":
        evaluate(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
