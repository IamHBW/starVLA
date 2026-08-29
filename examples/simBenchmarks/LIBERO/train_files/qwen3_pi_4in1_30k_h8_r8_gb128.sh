#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/../../../.." && pwd)"

export NNODES="${NNODES:-2}"
export NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
export GRADIENT_ACCUMULATION_STEPS=1
export QWEN3_PI_ACTION_HORIZON=8
export QWEN3_PI_REPLAN_STEPS=8
export QWEN3_PI_DATA_MIX=libero_all
export QWEN3_PI_PER_DEVICE_BATCH_SIZE=8
export QWEN3_PI_GLOBAL_BATCH_SIZE=128
export QWEN3_PI_EVAL_PROFILE=h8_r8_gb128
export QWEN3_PI_TASK_SPEC="${repo}/docs/spec_qwen3_vl_pi_libero_4in1_30k_h8_r8_gb128.md"
export QWEN3_PI_ENTRY_LAUNCHER="${BASH_SOURCE[0]}"
export QWEN3_PI_EVAL_LAUNCHER="${repo}/examples/simBenchmarks/LIBERO/eval_files/qwen3_pi_4in1_30k_h8_r8_gb128_eval.sh"
export RUN_DIR=/mnt/data/users/bowen/workspace/ckpt/qwen3_pi_libero4in1_30k_h8_r8_gb128
export HTRAIN_JOB_NAME="${HTRAIN_JOB_NAME:-q3pi-h8g128-n2}"
export HTRAIN_PROJECT="${HTRAIN_PROJECT:-depth_wam}"

exec "${script_dir}/qwen3_pi_4in1_30k.sh"
