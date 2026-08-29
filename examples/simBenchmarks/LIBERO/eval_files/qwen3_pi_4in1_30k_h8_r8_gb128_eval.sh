#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export QWEN3_PI_EVAL_PROFILE=h8_r8_gb128
export RUN_DIR="${RUN_DIR:-/mnt/data/users/bowen/workspace/ckpt/qwen3_pi_libero4in1_30k_h8_r8_gb128}"

exec "${script_dir}/qwen3_pi_4in1_30k_eval.sh"
