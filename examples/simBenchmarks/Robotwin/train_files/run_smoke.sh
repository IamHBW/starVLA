#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MAX_TRAIN_STEPS=2
export SAVE_INTERVAL=2
export CHECKPOINT_VALIDATE=1
exec bash "${script_dir}/run_train.sh"
