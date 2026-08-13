#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/../../../.." && pwd)"
cd "${repo}"
tokens=/mnt/data/users/bowen/workspace/tokens.sh
python_bin="${STARVLA_PYTHON:-/mnt/data/users/bowen/workspace/envs/starvla/bin/python}"
accelerate_bin="${python_bin%/python}/accelerate"
source_data=/mnt/data/public_data/robotwin_lerobot_v2
model_revision=ebb281ec70b05090aa6165b016eac8ec08e71b17
model_snapshot="${MODEL_SNAPSHOT:-/mnt/data/users/bowen/workspace/outputs/model_cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/${model_revision}}"
config="${script_dir}/qwenpi_clean50.yaml"
max_steps="${MAX_TRAIN_STEPS:-30000}"
save_interval="${SAVE_INTERVAL:-10000}"

: "${RUN_ID:?RUN_ID is required}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "Invalid RUN_ID: ${RUN_ID}" >&2; exit 2; }
[[ -x "${python_bin}" && -x "${accelerate_bin}" ]] || { echo "Missing StarVLA environment: ${python_bin}" >&2; exit 2; }
[[ -f "${tokens}" && "$(stat -c %U "${tokens}")" == "$(id -un)" ]] || { echo "Refusing non-user W&B credentials" >&2; exit 2; }
[[ -d "${model_snapshot}" ]] || { echo "Missing frozen Qwen3 snapshot: ${model_snapshot}" >&2; exit 2; }

# shellcheck disable=SC1090
source "${tokens}"
export WANDB_MODE=online
export QWEN3_VL_SNAPSHOT="${model_snapshot}"
export PYTHONPATH="${repo}${PYTHONPATH:+:${PYTHONPATH}}"
export ROBOTWIN_OVERLAY_ROOT="/mnt/data/users/bowen/workspace/ckpt/${RUN_ID}/data_overlay"

run_dir="/mnt/data/users/bowen/workspace/ckpt/${RUN_ID}"
cache_dir="/mnt/data/users/bowen/workspace/outputs/${RUN_ID}/cache"
mkdir -p "${run_dir}/wandb" "${cache_dir}/wandb" "${cache_dir}/torch" "${cache_dir}/triton" "${cache_dir}/huggingface"
export WANDB_DIR="${run_dir}/wandb"
export WANDB_CACHE_DIR="${cache_dir}/wandb"
export WANDB_DATA_DIR="${cache_dir}/wandb"
export TORCH_HOME="${cache_dir}/torch"
export TRITON_CACHE_DIR="${cache_dir}/triton"
export HF_HOME="${cache_dir}/huggingface"
export XDG_CACHE_HOME="${cache_dir}"
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

resume=false
if [[ "${RESUME:-0}" == 1 ]]; then
  resume=true
elif [[ -e "${run_dir}/config.full.yaml" ]]; then
  echo "Run already exists; set RESUME=1 only for an infrastructure resume: ${run_dir}" >&2
  exit 2
fi

preflight() {
  "${python_bin}" "${script_dir}/preflight_clean50.py" \
    --repo "${repo}" --run-dir "${run_dir}" --source "${source_data}" \
    --model "${model_snapshot}" --mode "$1"
}

launch() {
  local validation="$1"
  "${accelerate_bin}" launch \
    --config_file "${run_dir}/accelerate_config.yaml" \
    --num_processes 8 \
    --gradient_accumulation_steps 2 \
    "${repo}/starVLA/training/train_starvla.py" \
    --config_yaml "${config}" \
    --run_id "${RUN_ID}" \
    --trainer.max_train_steps "${max_steps}" \
    --trainer.save_interval "${save_interval}" \
    --trainer.is_resume "$([[ "${validation}" == true ]] && echo true || echo "${resume}")" \
    --trainer.validate_checkpoint "${validation}"
}

preflight train
"${python_bin}" - "${run_dir}/accelerate_config.yaml" "${run_dir}/resolved_deepspeed_config.json" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf

accelerate_cfg = OmegaConf.load(sys.argv[1])
deepspeed_cfg = json.load(open(sys.argv[2]))
assert accelerate_cfg.distributed_type == "DEEPSPEED"
assert accelerate_cfg.num_processes == 8
assert Path(accelerate_cfg.deepspeed_config.deepspeed_config_file).resolve() == Path(sys.argv[2]).resolve()
assert deepspeed_cfg["bf16"]["enabled"] is True
assert deepspeed_cfg["zero_optimization"]["stage"] == 2
assert deepspeed_cfg["train_micro_batch_size_per_gpu"] == 16
assert deepspeed_cfg["gradient_accumulation_steps"] == 2
assert deepspeed_cfg["train_batch_size"] == 256
PY
launch false

if [[ "${CHECKPOINT_VALIDATE:-1}" == 1 ]]; then
  preflight validate
  launch true
fi

EXPECTED_STEP="${max_steps}" "${python_bin}" - "${run_dir}" "${save_interval}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
step = int(os.environ["EXPECTED_STEP"])
save_interval = int(sys.argv[2])
checkpoints = run_dir / "checkpoints"
expected_weights = [step] if step < 30000 else list(range(save_interval, step + 1, save_interval))
for expected in expected_weights:
    if not (checkpoints / f"steps_{expected}_pytorch_model.pt").is_file():
        raise FileNotFoundError(f"Missing step-{expected} model weights")
full_state = checkpoints / f"full_state_step_{step}"
progress = json.loads((full_state / "progress.json").read_text())
validation = json.loads((run_dir / "checkpoint_validation.json").read_text())
assert progress["optimizer_step"] == validation["optimizer_step"] == step
assert progress["cumulative_train_samples"] == step * 256
assert validation["predicted_action_shape"] == [16, 32, 14]

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

files = {
    str(path.relative_to(run_dir)): sha256(path)
    for path in sorted(checkpoints.rglob("*"))
    if path.is_file()
}
(run_dir / "checkpoint_manifest.json").write_text(json.dumps({"optimizer_step": step, "files": files}, indent=2) + "\n")
PY
