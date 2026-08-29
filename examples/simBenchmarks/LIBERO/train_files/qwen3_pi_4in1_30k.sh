#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/../../../.." && pwd)"
python_bin=/mnt/data/users/bowen/workspace/envs/starvla/bin/python
accelerate_bin=/mnt/data/users/bowen/workspace/envs/starvla/bin/accelerate
tokens=/mnt/data/users/bowen/workspace/tokens.sh
data_root=/mnt/data/users/bowen/workspace/data/starvla_libero_lerobot
run_dir="${RUN_DIR:-/mnt/data/users/bowen/workspace/ckpt/qwen3_pi_libero4in1_30k}"
config="${script_dir}/qwen3_pi_4in1_30k.yaml"
action_horizon="${QWEN3_PI_ACTION_HORIZON:-32}"
replan_steps="${QWEN3_PI_REPLAN_STEPS:-24}"
data_mix="${QWEN3_PI_DATA_MIX:-libero_all_h32}"
per_device_batch_size="${QWEN3_PI_PER_DEVICE_BATCH_SIZE:-8}"
global_batch_size="${QWEN3_PI_GLOBAL_BATCH_SIZE:-256}"
task_spec="${QWEN3_PI_TASK_SPEC:-${repo}/docs/spec_qwen3_vl_pi_libero_4in1_30k.md}"
eval_launcher="${QWEN3_PI_EVAL_LAUNCHER:-${repo}/examples/simBenchmarks/LIBERO/eval_files/qwen3_pi_4in1_30k_eval.sh}"
entry_launcher="${QWEN3_PI_ENTRY_LAUNCHER:-${BASH_SOURCE[0]}}"
model_revision=ebb281ec70b05090aa6165b016eac8ec08e71b17
starvla_base_revision=02861ead680ea648c367ed41cf0d0976581f0467
model_snapshot="/mnt/data/users/bowen/workspace/outputs/model_cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/${model_revision}"
num_nodes="${NNODES:-${NUM_NODES:-1}}"
gpus_per_node="${NUM_GPUS_PER_NODE:-${GPUS_PER_NODE:-8}}"
gradient_accumulation="${GRADIENT_ACCUMULATION_STEPS:-4}"
for value in "${num_nodes}" "${gpus_per_node}" "${gradient_accumulation}" "${action_horizon}" \
  "${replan_steps}" "${per_device_batch_size}" "${global_batch_size}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid distributed integer: ${value}" >&2; exit 2; }
done
host="${HOSTNAME:-$(hostname -s)}"
node_rank="${PET_NODE_RANK:-${NODE_RANK:-}}"
if [[ -z "${node_rank}" ]]; then
  if [[ "${host}" =~ -master-([0-9]+)$ ]]; then
    node_rank="${BASH_REMATCH[1]}"
  elif [[ "${host}" =~ -worker-([0-9]+)$ ]]; then
    node_rank="$((BASH_REMATCH[1] + 1))"
  else
    node_rank=0
  fi
fi
master_addr="${PET_MASTER_ADDR:-${MASTER_ADDR:-}}"
if [[ -z "${master_addr}" ]]; then
  if ((num_nodes == 1)); then
    master_addr=127.0.0.1
  elif [[ "${host}" =~ -worker-[0-9]+$ ]]; then
    master_addr="${host%-worker-*}-master-0"
  elif [[ "${host}" =~ -master-[0-9]+$ ]]; then
    master_addr="${host%-master-*}-master-0"
  else
    echo "Cannot infer multi-node master address from host ${host}" >&2
    exit 2
  fi
fi
master_port="${PET_MASTER_PORT:-${MASTER_PORT:-29500}}"
total_processes=$((num_nodes * gpus_per_node))
train_run_id="$(basename "${run_dir}")"
run_root_dir="$(dirname "${run_dir}")"

cd "${repo}"
git merge-base --is-ancestor "${starvla_base_revision}" HEAD || { echo "Unexpected StarVLA base revision" >&2; exit 2; }
[[ -x "${python_bin}" && -x "${accelerate_bin}" ]] || { echo "Missing StarVLA environment" >&2; exit 2; }
[[ -f "${tokens}" && "$(stat -c %U "${tokens}")" == bowen ]] || { echo "Refusing credentials not owned by bowen" >&2; exit 2; }
[[ -d "${data_root}" && -d "${model_snapshot}" && -f "${config}" && -f "${task_spec}" && -x "${entry_launcher}" && -x "${eval_launcher}" ]] || { echo "Missing frozen input" >&2; exit 2; }
for value in "${node_rank}" "${master_port}"; do
  [[ "${value}" =~ ^[0-9]+$ ]] || { echo "Invalid distributed integer: ${value}" >&2; exit 2; }
done
((num_nodes >= 1 && gpus_per_node == 8 && node_rank < num_nodes)) || { echo "Invalid topology ${node_rank}/${num_nodes}x${gpus_per_node}" >&2; exit 2; }
((action_horizon >= 1 && replan_steps >= 1 && replan_steps <= action_horizon)) || { echo "Invalid horizon/replan: ${action_horizon}/${replan_steps}" >&2; exit 2; }
[[ "${run_dir}" == /mnt/data/users/bowen/workspace/ckpt/* && "${train_run_id}" != */* ]] || { echo "Invalid run destination: ${run_dir}" >&2; exit 2; }
global_batch=$((per_device_batch_size * total_processes * gradient_accumulation))
((global_batch == global_batch_size)) || { echo "Invalid global batch: ${per_device_batch_size} x ${total_processes} x ${gradient_accumulation} = ${global_batch}, expected ${global_batch_size}" >&2; exit 2; }
echo "[qwen3-pi] host=${host} node_rank=${node_rank}/${num_nodes} master=${master_addr}:${master_port} world_size=${total_processes} per_device=${per_device_batch_size} accumulation=${gradient_accumulation} global_batch=${global_batch} horizon=${action_horizon} replan=${replan_steps} mix=${data_mix} run_dir=${run_dir}"
if ((node_rank == 0)) && [[ -z "${HTRAIN_JOB_ID:-}" && -n "${HTRAIN_JOB_NAME:-}" ]]; then
  HTRAIN_JOB_ID="$(submit --status "${HTRAIN_JOB_NAME}" 2>/dev/null | awk '/^id:/ {print $2; exit}' || true)"
  export HTRAIN_JOB_ID
fi

resume=false
if [[ "${RESUME:-0}" == 1 ]]; then
  resume=true
  compgen -G "${run_dir}/checkpoints/full_state_step_*" >/dev/null || { echo "No full state to resume" >&2; exit 2; }
elif ((node_rank == 0)) && [[ -e "${run_dir}" && ! -d "${run_dir}" ]]; then
  echo "Run path exists and is not a directory: ${run_dir}" >&2
  exit 2
elif ((node_rank == 0)) && [[ -d "${run_dir}" ]] && find "${run_dir}" -mindepth 1 \
  ! -name 'htrain*.log' \
  ! -path "${run_dir}/failed_attempts" \
  ! -path "${run_dir}/failed_attempts/*" \
  -print -quit | grep -q .; then
  echo "Run already exists; set RESUME=1 only for an infrastructure resume: ${run_dir}" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${tokens}"
export WANDB_MODE=online
export WANDB_PROJECT=StarVLA_LIBERO_Qwen3PI_30K
: "${WANDB_ENTITY:?WANDB_ENTITY must come from current-user tokens.sh}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${train_run_id}}"
export WANDB_RESUME=allow
export QWEN3_VL_SNAPSHOT="${model_snapshot}"
export STARVLA_BASE_REVISION="${starvla_base_revision}"
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

if ((node_rank == 0)); then
  mkdir -p "${run_dir}/wandb" "${run_dir}/cache/huggingface" "${run_dir}/cache/torch" "${run_dir}/cache/triton"
fi
export WANDB_DIR="${run_dir}/wandb"
export WANDB_CACHE_DIR="${run_dir}/cache/wandb"
export WANDB_DATA_DIR="${run_dir}/cache/wandb"
export HF_HOME="${run_dir}/cache/huggingface"
export TORCH_HOME="${run_dir}/cache/torch"
export TRITON_CACHE_DIR="${run_dir}/cache/triton"
export XDG_CACHE_HOME="${run_dir}/cache"

preflight_done="${run_dir}/.preflight_done"
preflight_failed="${run_dir}/.preflight_failed"
if ((node_rank == 0)); then
  if ! "${python_bin}" - "${repo}" "${run_dir}" "${data_root}" "${model_snapshot}" "${config}" "${resume}" \
    "${total_processes}" "${gradient_accumulation}" "${num_nodes}" "${gpus_per_node}" "${train_run_id}" "${run_root_dir}" \
    "${action_horizon}" "${replan_steps}" "${per_device_batch_size}" "${global_batch_size}" "${data_mix}" <<'PY'
import grp
import hashlib
import importlib.metadata
import json
import os
import pwd
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, ROBOT_TYPE_CONFIG_MAP

repo, run_dir, data_root, model, config = map(Path, sys.argv[1:6])
resume = sys.argv[6] == "true"
world_size, gradient_accumulation, num_nodes, gpus_per_node = map(int, sys.argv[7:11])
train_run_id, run_root_dir = sys.argv[11:13]
action_horizon, replan_steps, per_device_batch_size, global_batch_size = map(int, sys.argv[13:17])
data_mix = sys.argv[17]
expected = {
    "libero_spatial_no_noops_1.0.0_lerobot": ("bf14d6258218d12c2e3c1a3b9922e163cdf6455d", 432, 52970),
    "libero_object_no_noops_1.0.0_lerobot": ("15657dac2ad1c01b4e94bf54ab0493b46a8d63f9", 454, 66984),
    "libero_goal_no_noops_1.0.0_lerobot": ("222cf888ed360fad0a5f983748c1cc40743d43e7", 428, 52042),
    "libero_10_no_noops_1.0.0_lerobot": ("e1a223d30b896c1613f270a2bfc63d382b3de7e1", 379, 101469),
}

def run(*args):
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout.strip()

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def owner(path):
    stat = path.stat()
    return f"{pwd.getpwuid(stat.st_uid).pw_name}:{grp.getgrgid(stat.st_gid).gr_name}"

def file_hashes(root, parts):
    files = [p for part in parts for p in (root / part).rglob("*") if p.is_file()]
    return {str(p.relative_to(root)): sha256(p) for p in sorted(files)}

source_modality = repo / "examples/simBenchmarks/LIBERO/train_files/modality.json"
source_modality_sha = sha256(source_modality)
datasets = {}
all_hashes = {}
for name, (revision, episodes, frames) in expected.items():
    root = data_root / name
    info = json.loads((root / "meta/info.json").read_text())
    commits = {
        lines[0]
        for metadata in (root / ".cache/huggingface/download").rglob("*.metadata")
        if (lines := metadata.read_text(errors="replace").splitlines())
    }
    if commits != {revision}:
        raise RuntimeError(f"{name}: commit mismatch {commits}")
    if (info["total_episodes"], info["total_frames"]) != (episodes, frames):
        raise RuntimeError(f"{name}: count mismatch")
    if sha256(root / "meta/modality.json") != source_modality_sha:
        raise RuntimeError(f"{name}: modality mismatch")
    hashes = file_hashes(root, ("data", "videos", "meta"))
    all_hashes[name] = hashes
    content_bytes = sum((root / rel).stat().st_size for rel in hashes if rel.startswith(("data/", "videos/")))
    datasets[name] = {
        "path": str(root), "owner": owner(root), "revision": revision,
        "episodes": episodes, "frames": frames, "data_video_bytes": content_bytes,
        "info_sha256": sha256(root / "meta/info.json"), "modality_sha256": source_modality_sha,
    }

if sum(v["episodes"] for v in datasets.values()) != 1693:
    raise RuntimeError("Dataset episode total mismatch")
if sum(v["frames"] for v in datasets.values()) != 273465:
    raise RuntimeError("Dataset frame total mismatch")
if sum(v["data_video_bytes"] for v in datasets.values()) != 1881101992:
    raise RuntimeError("Dataset byte total mismatch")

(run_dir / "data_file_sha256.json").write_text(json.dumps(all_hashes, indent=2, sort_keys=True) + "\n")

cfg = OmegaConf.load(config)
resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
resolved_cfg["run_id"] = train_run_id
resolved_cfg["run_root_dir"] = run_root_dir
am = resolved_cfg["framework"]["action_model"]
vla = resolved_cfg["datasets"]["vla_data"]
trainer = resolved_cfg["trainer"]
assert (am["action_horizon"], am["repeated_diffusion_steps"], am["num_inference_timesteps"]) == (32, 8, 4)
assert am["diffusion_model_cfg"]["use_canonical_forward"] is False
assert (vla["data_mix"], vla["per_device_batch_size"], vla["include_state"]) == ("libero_all_h32", 8, False)
assert (trainer["max_train_steps"], trainer["gradient_accumulation_steps"], trainer["is_resume"]) == (30000, 4, False)
am["action_horizon"] = action_horizon
am["future_action_window_size"] = action_horizon - 1
vla["data_mix"] = data_mix
vla["per_device_batch_size"] = per_device_batch_size
trainer["gradient_accumulation_steps"] = gradient_accumulation
assert not trainer.get("pretrained_checkpoint") and trainer["freeze_modules"] == ""
mixture = DATASET_NAMED_MIXTURES[data_mix]
assert len(mixture) == 4 and all(weight == 1.0 for _, weight, _ in mixture)
assert all(ROBOT_TYPE_CONFIG_MAP[robot_type].action_indices == list(range(action_horizon)) for _, _, robot_type in mixture)
assert per_device_batch_size * world_size * gradient_accumulation == global_batch_size
OmegaConf.save(OmegaConf.create(resolved_cfg), run_dir / "resolved_input_config.yaml")

ds = json.loads((repo / "starVLA/config/deepseeds/ds_config.yaml").read_text())
ds["train_micro_batch_size_per_gpu"] = per_device_batch_size
ds["gradient_accumulation_steps"] = gradient_accumulation
ds["train_batch_size"] = global_batch_size
ds["zero_optimization"]["allgather_bucket_size"] = 100_000_000
ds["zero_optimization"]["reduce_bucket_size"] = 100_000_000
ds_path = run_dir / "resolved_deepspeed_config.json"
ds_path.write_text(json.dumps(ds, indent=2) + "\n")
accelerate = OmegaConf.load(repo / "starVLA/config/deepseeds/deepspeed_zero2.yaml")
accelerate.num_machines = num_nodes
accelerate.num_processes = world_size
accelerate.deepspeed_config.deepspeed_config_file = str(ds_path)
OmegaConf.save(accelerate, run_dir / "accelerate_config.yaml")

(run_dir / "git_diff.patch").write_text(run("git", "-C", str(repo), "diff", "--binary") + "\n")
(run_dir / "git_status.txt").write_text(run("git", "-C", str(repo), "status", "--short") + "\n")
model_hashes = file_hashes(model, (".",))
packages = {}
for package in ("torch", "transformers", "accelerate", "deepspeed", "wandb", "numpy", "pandas", "pyarrow", "av", "huggingface_hub"):
    try:
        packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        packages[package] = None

event = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "mode": "resume" if resume else "train_from_scratch",
    "git_sha": run("git", "-C", str(repo), "rev-parse", "HEAD"),
    "starvla_base_sha": os.environ["STARVLA_BASE_REVISION"],
    "git_status": run("git", "-C", str(repo), "status", "--short").splitlines(),
    "libero_sha": run("git", "-C", "/mnt/data/users/bowen/workspace/code/LIBERO", "rev-parse", "HEAD"),
    "libero_plus_sha": run("git", "-C", "/mnt/data/users/bowen/workspace/code/LIBERO-plus", "rev-parse", "HEAD"),
    "datasets": datasets,
    "model": {"path": str(model), "owner": owner(model), "revision": model.name, "file_sha256": model_hashes},
    "python": {"configured": sys.executable, "realpath": str(Path(sys.executable).resolve()), "owner": owner(Path(sys.executable).resolve()), "sys_path": sys.path},
    "packages": packages,
    "gpu": run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
    "disk": run("df", "-h", "/mnt/data"),
    "batch": {"per_device": per_device_batch_size, "world_size": world_size, "gradient_accumulation": gradient_accumulation, "global": global_batch_size},
    "expected_action_shape": [per_device_batch_size, action_horizon, 7],
    "topology": {"nodes": num_nodes, "gpus_per_node": gpus_per_node},
    "memory_strategy": {
        "pytorch_cuda_alloc_conf": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        "zero_stage": ds["zero_optimization"]["stage"],
        "allgather_bucket_size": ds["zero_optimization"]["allgather_bucket_size"],
        "reduce_bucket_size": ds["zero_optimization"]["reduce_bucket_size"],
    },
    "previous_failed_job_id": os.environ.get("PREVIOUS_JOB_ID"),
    "user_overrides": {"action_horizon": action_horizon, "replan_steps": replan_steps},
    "shared_raw_source": {
        "path": "/mnt/data/public_data/libero",
        "owner": owner(Path("/mnt/data/public_data/libero")),
        "suite_revisions": {name: revision for name, (revision, _, _) in expected.items()},
    },
    "htrain": {
        "job_id": os.environ.get("HTRAIN_JOB_ID") or os.environ.get("JOB_ID") or os.environ.get("PET_TASK_ID"),
        "job_name": os.environ.get("HTRAIN_JOB_NAME"),
        "project": os.environ.get("HTRAIN_PROJECT"),
    },
    "wandb": {"mode": os.environ["WANDB_MODE"], "entity": os.environ["WANDB_ENTITY"], "project": os.environ["WANDB_PROJECT"], "credentials": "/mnt/data/users/bowen/workspace/tokens.sh"},
}
manifest_path = run_dir / "run_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"events": []}
manifest["events"].append(event)
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY
  then
    touch "${preflight_failed}"
    exit 1
  fi
  cp -f "${config}" "${run_dir}/source_config.yaml"
  cp -f "${BASH_SOURCE[0]}" "${run_dir}/source_launcher.sh"
  cp -f "${entry_launcher}" "${run_dir}/source_entry_launcher.sh"
  cp -f "${eval_launcher}" "${run_dir}/source_eval_launcher.sh"
  cp -f "${task_spec}" "${run_dir}/task_spec.md"
  touch "${preflight_done}"
else
  for _ in {1..3600}; do
    [[ -e "${preflight_failed}" ]] && { echo "Rank-0 preflight failed" >&2; exit 1; }
    [[ -e "${preflight_done}" ]] && break
    sleep 1
  done
  [[ -e "${preflight_done}" ]] || { echo "Timed out waiting for rank-0 preflight" >&2; exit 1; }
fi

train_command=(
  "${accelerate_bin}" launch
  --config_file "${run_dir}/accelerate_config.yaml"
  --deepspeed_multinode_launcher standard
  --num_machines "${num_nodes}"
  --num_processes "${total_processes}"
  --machine_rank "${node_rank}"
  --main_process_ip "${master_addr}"
  --main_process_port "${master_port}"
  --same_network
  --gradient_accumulation_steps "${gradient_accumulation}"
  "${repo}/starVLA/training/train_starvla.py"
  --config_yaml "${config}"
  --run_root_dir "${run_root_dir}"
  --run_id "${train_run_id}"
  --framework.action_model.action_horizon "${action_horizon}"
  --datasets.vla_data.data_mix "${data_mix}"
  --datasets.vla_data.per_device_batch_size "${per_device_batch_size}"
  --trainer.gradient_accumulation_steps "${gradient_accumulation}"
  --trainer.is_resume "${resume}"
)
if ((node_rank == 0)); then
  printf '%q ' "${train_command[@]}" >"${run_dir}/training_command.txt"
  printf '\n' >>"${run_dir}/training_command.txt"
fi
"${train_command[@]}"

((node_rank == 0)) || exit 0

# The post-train gate is single-process; inherited HTrain variables would make
# PartialState wait for the other training ranks after they have already exited.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE
"${python_bin}" "${repo}/examples/simBenchmarks/LIBERO/eval_files/qwen3_pi_4in1_30k_eval.py" \
  validate-checkpoint --run-dir "${run_dir}" --data-root "${data_root}"

if [[ "${RUN_EVAL_AFTER_TRAIN:-1}" == 1 ]]; then
  RUN_DIR="${run_dir}" "${eval_launcher}"
fi
