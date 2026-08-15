#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/data/users/bowen/workspace/code/starVLA
EVAL_DIR="${REPO}/examples/simBenchmarks/Robotwin/eval_files"
CHECKPOINT=/mnt/data/users/bowen/workspace/ckpt/q3pi_c50_train_a2_20260813T090928Z/checkpoints/steps_30000_pytorch_model.pt
CONFIG="${REPO}/examples/simBenchmarks/Robotwin/train_files/qwenpi_clean50.yaml"
CONTRACT=/mnt/data/users/bowen/workspace/ckpt/q3pi_c50_train_a2_20260813T090928Z/resolved_contract.json
ROBOTWIN_PATH=/mnt/data/users/bowen/workspace/code/openpi/third_party/RoboTwin
STARVLA_PYTHON=/mnt/data/users/bowen/workspace/envs/starvla/bin/python
ROBOTWIN_PYTHON=/mnt/data/public_tools/miniconda3/envs/RoboTwin/bin/python
OUTPUT_BASE=/mnt/data/users/bowen/workspace/outputs
CHECKPOINT_SHA256=44c48de47aded109bc668717177339a8f831af1a4dd5b1f762b66c4866fb2aab
CONFIG_SHA256=9a4a71682bb48dbaeb7b0fdcc81f59008432640133d8d29959e1c7e277864ccb
VENDOR_TREE=a620160151ec70e6a4fe9d61c20175d6d0a9754c

# HTrain exports training-world variables even though this entrypoint launches
# eight independent single-GPU servers. Keep them for provenance, then prevent
# Accelerate's logger from waiting for nonexistent peer processes.
export Q3PI_HTRAIN_WORLD_SIZE="${WORLD_SIZE:-}"
export Q3PI_HTRAIN_MASTER_ADDR="${MASTER_ADDR:-}"
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT

phase="${PHASE:?Set PHASE=clean or PHASE=randomized}"
run_id="${RUN_ID:?Set one shared RUN_ID for both phases}"
case "${phase}" in
    clean) task_config=demo_clean ;;
    randomized) task_config=demo_randomized ;;
    *) echo "PHASE must be clean or randomized: ${phase}" >&2; exit 2 ;;
esac
if [[ ! "${run_id}" =~ ^q3pi_c50_eval_[0-9]{8}T[0-9]{6}Z$ ]]; then
    echo "Invalid RUN_ID: ${run_id}" >&2
    exit 2
fi

run_root="${OUTPUT_BASE}/${run_id}"
phase_root="${run_root}/${phase}"
mkdir -p "${phase_root}/logs" "${phase_root}/results"

for path in "${CHECKPOINT}" "${CONFIG}" "${CONTRACT}" "${STARVLA_PYTHON}" "${ROBOTWIN_PYTHON}" \
    "${ROBOTWIN_PATH}/script/eval_policy.py" "${ROBOTWIN_PATH}/README.vendor.md"; do
    [[ -e "${path}" ]] || { echo "Missing required path: ${path}" >&2; exit 1; }
done
[[ -x "${STARVLA_PYTHON}" && -x "${ROBOTWIN_PYTHON}" ]] || { echo "Python is not executable" >&2; exit 1; }

mapfile -t tasks < <(bash "${EVAL_DIR}/start_eval.sh" --list-tasks)
[[ ${#tasks[@]} -eq 50 && $(printf '%s\n' "${tasks[@]}" | sort -u | wc -l) -eq 50 ]] || {
    echo "RoboTwin task set must contain 50 unique tasks" >&2
    exit 1
}
"${ROBOTWIN_PYTHON}" - "${ROBOTWIN_PATH}" "${tasks[@]}" <<'PY'
import json
from pathlib import Path
import sys
import yaml

root = Path(sys.argv[1])
tasks = sys.argv[2:]
limits = yaml.safe_load((root / "task_config/_eval_step_limit.yml").read_text())
assert set(limits) == set(tasks)
assert all((root / "envs" / f"{task}.py").is_file() for task in tasks)
assert all((root / "description/task_instruction" / f"{task}.json").is_file() for task in tasks)
contract = json.loads(Path("/mnt/data/users/bowen/workspace/ckpt/q3pi_c50_train_a2_20260813T090928Z/resolved_contract.json").read_text())
assert contract["global_batch_size"] == 256
assert contract["action_shape"][1:] == [32, 14]
assert contract["image_sizes"] == [[224, 224]] * 3
assert contract["include_state"] is False
PY

if find "${ROBOTWIN_PATH}" -type l -print -quit | grep -q .; then
    echo "RoboTwin vendor must not contain symlinks" >&2
    exit 1
fi
actual_vendor_tree="$(git -C "${ROBOTWIN_PATH}/../.." ls-tree HEAD third_party/RoboTwin | awk '{print $3}')"
[[ "${actual_vendor_tree}" == "${VENDOR_TREE}" ]] || { echo "RoboTwin vendor tree mismatch: ${actual_vendor_tree}" >&2; exit 1; }
[[ -z "$(git -C "${ROBOTWIN_PATH}/../.." status --short -- third_party/RoboTwin)" ]] || {
    echo "RoboTwin vendor has local modifications" >&2
    exit 1
}

actual_config_sha="$(sha256sum "${CONFIG}" | awk '{print $1}')"
[[ "${actual_config_sha}" == "${CONFIG_SHA256}" ]] || { echo "Config hash mismatch: ${actual_config_sha}" >&2; exit 1; }
actual_checkpoint_sha="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
[[ "${actual_checkpoint_sha}" == "${CHECKPOINT_SHA256}" ]] || { echo "Checkpoint hash mismatch: ${actual_checkpoint_sha}" >&2; exit 1; }

export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
"${STARVLA_PYTHON}" -c 'from deployment.model_server.policy_wrapper import PolicyServerWrapper; print("StarVLA imports OK")'
"${ROBOTWIN_PYTHON}" -c 'import cv2, numpy, sapien, yaml; print("RoboTwin imports OK")'
ffmpeg_bin="$("${ROBOTWIN_PYTHON}" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
[[ -x "${ffmpeg_bin}" ]] || { echo "Missing RoboTwin video dependency: ${ffmpeg_bin}" >&2; exit 1; }
mkdir -p "${run_root}/bin"
ln -sfn "${ffmpeg_bin}" "${run_root}/bin/ffmpeg"
export PATH="${run_root}/bin:${PATH}"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    mapfile -t pending < <("${STARVLA_PYTHON}" "${EVAL_DIR}/aggregate_eval.py" missing --run-root "${run_root}" --phase "${phase}")
    echo "[DRY_RUN] phase=${phase} pending=${#pending[@]} run_root=${run_root}"
    exit 0
fi

gpu_count="$(nvidia-smi --list-gpus | wc -l)"
[[ "${gpu_count}" -eq 8 ]] || { echo "Expected 8 GPUs, got ${gpu_count}" >&2; exit 1; }
(cd "${ROBOTWIN_PATH}" && "${ROBOTWIN_PYTHON}" script/test_render.py)

"${STARVLA_PYTHON}" "${EVAL_DIR}/aggregate_eval.py" provenance \
    --run-root "${run_root}" --phase "${phase}" --repo "${REPO}" --vendor "${ROBOTWIN_PATH}" \
    --starvla-python "${STARVLA_PYTHON}" --robotwin-python "${ROBOTWIN_PYTHON}" \
    --checkpoint "${CHECKPOINT}" --checkpoint-sha256 "${actual_checkpoint_sha}" \
    --config "${CONFIG}" --config-sha256 "${actual_config_sha}"

mapfile -t pending < <("${STARVLA_PYTHON}" "${EVAL_DIR}/aggregate_eval.py" missing \
    --run-root "${run_root}" --phase "${phase}" --episodes 20)
if (( ${#pending[@]} == 0 )); then
    echo "[INFO] All ${phase} cells are already complete"
    exit 0
fi

export ROBOTWIN_PATH STARVLA_PYTHON ROBOTWIN_PYTHON
export ROBOTWIN_LOG_ROOT="${phase_root}/logs"
export ROBOTWIN_EVAL_OUTPUT_ROOT="${phase_root}/results"
export ROBOTWIN_EVAL_NUM_EPISODES=20
export ROBOTWIN_SEED=0
export ROBOTWIN_JOBS_PER_GPU=1
export ROBOTWIN_SERVER_TIMEOUT=1800
export ROBOTWIN_SERVER_IDLE_TIMEOUT=-1
export ROBOTWIN_USE_BF16=1

echo "[INFO] Starting formal phase=${phase} pending=${#pending[@]} run_id=${run_id}"
bash "${EVAL_DIR}/start_eval.sh" \
    --mode "${task_config}" --name q3pi_step30000 --ckpt "${CHECKPOINT}" --episodes 20 \
    "${pending[@]}"

remaining="$("${STARVLA_PYTHON}" "${EVAL_DIR}/aggregate_eval.py" missing \
    --run-root "${run_root}" --phase "${phase}" --episodes 20 | wc -l)"
[[ "${remaining}" -eq 0 ]] || { echo "Phase ${phase} ended with ${remaining} incomplete tasks" >&2; exit 1; }
