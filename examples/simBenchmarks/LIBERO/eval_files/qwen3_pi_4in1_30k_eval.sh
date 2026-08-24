#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "${script_dir}/../../../.." && pwd)"
runner="${script_dir}/qwen3_pi_4in1_30k_eval.py"
run_dir=/mnt/data/users/bowen/workspace/ckpt/qwen3_pi_libero4in1_30k
checkpoint="${run_dir}/checkpoints/steps_30000_pytorch_model.pt"
model_snapshot=/mnt/data/users/bowen/workspace/outputs/model_cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17
train_python=/mnt/data/users/bowen/workspace/envs/starvla/bin/python
standard_python=/mnt/data/public_tools/miniconda3/envs/libero/bin/python
plus_python=/mnt/data/users/bowen/workspace/envs/openvla-oft-plus/bin/python
standard_repo=/mnt/data/users/bowen/workspace/code/LIBERO
plus_repo=/mnt/data/users/bowen/workspace/code/LIBERO-plus
plus_deps=/mnt/data/users/bowen/workspace/code/lingbot-va/outputs/libero_plus/20260731T075749Z_libero_long/client_deps
standard_revision=8f1084e3132a39270c3a13ebe37270a43ece2a01
plus_revision=4c83d77c807983abf01da2c23bc8d2f72a204912
starvla_base_revision=02861ead680ea648c367ed41cf0d0976581f0467
slots="${EVAL_SLOTS:-32}"
port_base="${EVAL_PORT_BASE:-31500}"
eval_root="${run_dir}/eval/slots_${slots}"
runtime_dir="${run_dir}/eval/runtime"
gate="${run_dir}/eval/checkpoint_gate.json"

[[ "${slots}" =~ ^[0-9]+$ && "${slots}" -ge 1 && "${slots}" -le 32 ]] || {
  echo "EVAL_SLOTS must be an integer in [1,32]" >&2
  exit 2
}
[[ "${port_base}" =~ ^[0-9]+$ && $((port_base + slots - 1)) -le 65535 ]] || {
  echo "invalid EVAL_PORT_BASE" >&2
  exit 2
}
[[ -x "${train_python}" && -x "${standard_python}" && -x "${plus_python}" ]] || {
  echo "missing evaluation Python environment" >&2
  exit 2
}
[[ -f "${runner}" && -f "${checkpoint}" && -d "${model_snapshot}" && -d "${plus_deps}" ]] || {
  echo "missing frozen evaluation input" >&2
  exit 2
}
git -C "${repo}" merge-base --is-ancestor "${starvla_base_revision}" HEAD || {
  echo "StarVLA base revision mismatch" >&2
  exit 2
}
[[ "$(git -C "${standard_repo}" rev-parse HEAD)" == "${standard_revision}" ]] || {
  echo "standard LIBERO revision mismatch" >&2
  exit 2
}
[[ "$(git -C "${plus_repo}" rev-parse HEAD)" == "${plus_revision}" ]] || {
  echo "LIBERO-Plus revision mismatch" >&2
  exit 2
}
[[ "$(sha256sum "${plus_repo}/libero/libero/benchmark/task_classification.json" | cut -d' ' -f1)" == faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6 ]] || {
  echo "LIBERO-Plus classification mismatch" >&2
  exit 2
}

mkdir -p "${eval_root}/logs" "${runtime_dir}/standard_config" "${runtime_dir}/plus_config"
standard_config="${runtime_dir}/standard_config"
plus_config="${runtime_dir}/plus_config"

"${train_python}" - "${standard_config}/config.yaml" "${standard_repo}" <<'PY'
import sys
from pathlib import Path

output, root = map(Path, sys.argv[1:])
values = {
    "assets": root / "libero/libero/assets",
    "bddl_files": root / "libero/libero/bddl_files",
    "benchmark_root": root / "libero/libero",
    "datasets": Path("/mnt/data/public_data/libero"),
    "init_states": root / "libero/libero/init_files",
}
output.write_text("".join(f"{key}: {value}\n" for key, value in values.items()), encoding="utf-8")
PY
"${train_python}" - "${plus_config}/config.yaml" "${plus_repo}" <<'PY'
import sys
from pathlib import Path

output, root = map(Path, sys.argv[1:])
values = {
    "assets": root / "libero/libero/assets",
    "bddl_files": root / "libero/libero/bddl_files",
    "benchmark_root": root / "libero/libero",
    "datasets": root / "libero/datasets",
    "init_states": root / "libero/libero/init_files",
}
output.write_text("".join(f"{key}: {value}\n" for key, value in values.items()), encoding="utf-8")
PY

set +e
"${train_python}" "${runner}" checkpoint-gate \
  --run-dir "${run_dir}" \
  --checkpoint "${run_dir}/checkpoints/steps_100000_pytorch_model.pt" \
  --output "${run_dir}/eval/forbidden_100k_gate.json" \
  >"${run_dir}/eval/acceptance_100k_must_fail.log" 2>&1
forbidden_status=$?
set -e
[[ "${forbidden_status}" -ne 0 ]] || {
  echo "100K release checkpoint unexpectedly passed the 30K gate" >&2
  exit 2
}

"${train_python}" "${runner}" checkpoint-gate \
  --run-dir "${run_dir}" --checkpoint "${checkpoint}" --output "${gate}" \
  >"${run_dir}/eval/checkpoint_gate.log" 2>&1

cp -f "${runner}" "${runtime_dir}/qwen3_pi_4in1_30k_eval.py"
cp -f "${BASH_SOURCE[0]}" "${runtime_dir}/qwen3_pi_4in1_30k_eval.sh"

"${train_python}" - "${eval_root}/runtime_metadata.json" "${runner}" "${BASH_SOURCE[0]}" \
  "${train_python}" "${standard_python}" "${plus_python}" "${standard_repo}" "${plus_repo}" \
  "${plus_deps}" "${slots}" "${port_base}" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pwd
import grp
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output, runner, launcher = map(Path, sys.argv[1:4])
train_python, standard_python, plus_python = map(Path, sys.argv[4:7])
standard_repo, plus_repo, plus_deps = map(Path, sys.argv[7:10])
slots, port_base = map(int, sys.argv[10:12])

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def identity(path):
    real = path.resolve()
    stat = real.stat()
    return {
        "configured": str(path),
        "realpath": str(real),
        "owner": f"{pwd.getpwuid(stat.st_uid).pw_name}:{grp.getgrgid(stat.st_gid).gr_name}",
    }

def packages(python):
    code = (
        "import importlib.metadata,json; "
        "names=('numpy','torch','robosuite','imageio','websockets','msgpack','pillow'); "
        "print(json.dumps({n:(importlib.metadata.version(n) if n in {d.metadata['Name'].lower() for d in importlib.metadata.distributions()} else None) for n in names}))"
    )
    return json.loads(subprocess.run((str(python), "-c", code), check=True, capture_output=True, text=True).stdout)

payload = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "slots": slots,
    "workers_per_gpu": slots / 8,
    "port_base": port_base,
    "runner": {"path": str(runner), "sha256": sha(runner)},
    "launcher": {"path": str(launcher), "sha256": sha(launcher)},
    "environments": {
        "server": {**identity(train_python), "packages": packages(train_python)},
        "standard_client": {**identity(standard_python), "packages": packages(standard_python)},
        "plus_client": {**identity(plus_python), "packages": packages(plus_python)},
    },
    "dependencies": {
        "standard_libero": {"path": str(standard_repo), "revision": subprocess.run(("git", "-C", str(standard_repo), "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()},
        "libero_plus": {"path": str(plus_repo), "revision": subprocess.run(("git", "-C", str(plus_repo), "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()},
        "plus_client_deps": identity(plus_deps),
    },
    "gpu": subprocess.run(("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"), check=True, capture_output=True, text=True).stdout.splitlines(),
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

for slot in $(seq 0 $((slots - 1))); do
  port=$((port_base + slot))
  if ss -ltnH "sport = :${port}" | grep -q .; then
    echo "port already in use: ${port}" >&2
    exit 2
  fi
done

declare -a server_pids=()
declare -a client_pids=()

cleanup() {
  trap - EXIT
  set +e
  local pid
  for pid in "${client_pids[@]:-}"; do
    pkill -TERM -P "${pid}" 2>/dev/null
    kill -TERM "${pid}" 2>/dev/null
  done
  for pid in "${server_pids[@]:-}"; do
    pkill -TERM -P "${pid}" 2>/dev/null
    kill -TERM "${pid}" 2>/dev/null
  done
  sleep 2
  for pid in "${server_pids[@]:-}"; do
    pkill -KILL -P "${pid}" 2>/dev/null
    kill -KILL "${pid}" 2>/dev/null
  done
  set -e
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch_server() {
  local slot=$1
  local gpu=$((slot % 8))
  local port=$((port_base + slot))
  local log="${eval_root}/logs/server_${slot}_gpu${gpu}_port${port}.log"
  (
    cd "${repo}"
    exec env \
      -u WORLD_SIZE -u RANK -u LOCAL_RANK -u LOCAL_WORLD_SIZE \
      -u MASTER_ADDR -u MASTER_PORT -u GROUP_RANK -u ROLE_RANK -u ROLE_WORLD_SIZE \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      HF_HUB_OFFLINE=1 \
      TRANSFORMERS_OFFLINE=1 \
      TORCH_HOME="${run_dir}/cache/torch" \
      PYTHONNOUSERSITE=1 \
      PYTHONUNBUFFERED=1 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      PYTHONPATH="${repo}" \
      "${train_python}" deployment/model_server/server_policy.py \
      --ckpt_path "${checkpoint}" --port "${port}" --use_bf16 --idle_timeout -1 \
      --config_override framework.action_model.diffusion_model_cfg.use_canonical_forward=false \
      --config_override "framework.qwenvl.base_vlm=${model_snapshot}"
  ) >"${log}" 2>&1 &
  server_pids[${slot}]=$!
  echo "${server_pids[${slot}]}" >"${eval_root}/logs/server_${slot}.pid"
}

for slot in $(seq 0 $((slots - 1))); do
  launch_server "${slot}"
done

for attempt in $(seq 1 1350); do
  missing=0
  for slot in $(seq 0 $((slots - 1))); do
    port=$((port_base + slot))
    pid=${server_pids[${slot}]}
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "server slot ${slot} exited before listening" >&2
      set +e
      tail -n 120 "${eval_root}/logs/server_${slot}_gpu$((slot % 8))_port${port}.log" >&2
      set -e
      exit 1
    fi
    if ! ss -ltnH "sport = :${port}" | grep -q .; then
      missing=$((missing + 1))
    fi
  done
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  [[ "${attempt}" -lt 1350 ]] || {
    echo "${missing} policy servers failed to listen within 45 minutes" >&2
    exit 1
  }
  sleep 2
done
nvidia-smi >"${eval_root}/nvidia_smi_after_servers.txt"

run_phase() {
  local dataset=$1
  local mode=$2
  local phase=$3
  local video_policy=$4
  local client_python client_config client_pythonpath libero_revision
  if [[ "${dataset}" == standard ]]; then
    client_python=${standard_python}
    client_config=${standard_config}
    client_pythonpath="${repo}:${standard_repo}"
    libero_revision=${standard_revision}
  else
    client_python=${plus_python}
    client_config=${plus_config}
    client_pythonpath="${plus_deps}:${repo}:${plus_repo}"
    libero_revision=${plus_revision}
  fi

  local phase_dir="${eval_root}/${phase}"
  local manifest="${phase_dir}/manifest.tsv"
  local logs="${phase_dir}/logs"
  local progress="${phase_dir}/progress"
  mkdir -p "${logs}" "${progress}" "${phase_dir}/results"

  env \
    LIBERO_CONFIG_PATH="${client_config}" \
    PYTHONPATH="${client_pythonpath}" \
    PYTHONNOUSERSITE=1 \
    "${client_python}" "${runner}" build-manifest \
    --dataset "${dataset}" --mode "${mode}" --slots "${slots}" --gate "${gate}" --output "${manifest}" \
    >"${logs}/manifest.log" 2>&1

  if [[ -f "${phase_dir}/FINALIZED" ]]; then
    env LIBERO_CONFIG_PATH="${client_config}" PYTHONPATH="${client_pythonpath}" PYTHONNOUSERSITE=1 \
      "${client_python}" "${runner}" summarize \
      --manifest "${manifest}" --gate "${gate}" --run-dir "${phase_dir}" \
      >"${logs}/summary.log" 2>&1
    return 0
  fi

  mkdir -p "${phase_dir}/worker_restarts"

  completed_count() {
    find "${phase_dir}/results" -type f -name result.json -print | wc -l
  }

  run_client_once() {
    local slot=$1
    local gpu=$((slot % 8))
    local port=$((port_base + slot))
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      MUJOCO_GL=egl \
      PYOPENGL_PLATFORM=egl \
      MUJOCO_EGL_DEVICE_ID="${gpu}" \
      MAGICK_HOME=/mnt/data/users/bowen/workspace/envs/openvla-oft-plus \
      LIBERO_CONFIG_PATH="${client_config}" \
      PYTHONPATH="${client_pythonpath}" \
      PYTHONNOUSERSITE=1 \
      PYTHONUNBUFFERED=1 \
      PYTHONFAULTHANDLER=1 \
      OMP_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      "${client_python}" "${runner}" eval \
      --manifest "${manifest}" --gate "${gate}" --slot "${slot}" --port "${port}" \
      --run-dir "${phase_dir}" --progress-file "${progress}/slot_${slot}.txt" \
      --libero-revision "${libero_revision}" --video-policy "${video_policy}"
  }

  supervise_client() {
    local slot=$1
    local no_progress=0
    local before after status
    local restart_log="${phase_dir}/worker_restarts/slot_${slot}.tsv"
    printf 'timestamp_utc\tslot\texit_status\tcompleted_before\tcompleted_after\n' >"${restart_log}"
    before=$(completed_count)
    while true; do
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] starting slot=${slot} completed=${before}"
      if run_client_once "${slot}"; then
        return 0
      else
        status=$?
      fi
      after=$(completed_count)
      printf '%s\t%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${slot}" "${status}" "${before}" "${after}" >>"${restart_log}"
      if [[ "${after}" -gt "${before}" ]]; then
        no_progress=0
      else
        no_progress=$((no_progress + 1))
      fi
      [[ "${no_progress}" -lt 3 ]] || {
        echo "slot ${slot} made no progress across three restarts" >&2
        return "${status}"
      }
      before=${after}
      sleep 2
    done
  }

  mapfile -t phase_slots < <(tail -n +2 "${manifest}" | cut -f4 | sort -n -u)
  client_pids=()
  for slot in "${phase_slots[@]}"; do
    supervise_client "${slot}" >"${logs}/client_${slot}.log" 2>&1 &
    client_pids[${slot}]=$!
    echo "${client_pids[${slot}]}" >"${logs}/client_${slot}.pid"
  done

  local client_status=0
  for slot in "${phase_slots[@]}"; do
    if ! wait "${client_pids[${slot}]}"; then
      echo "client slot ${slot} failed; see ${logs}/client_${slot}.log" >&2
      client_status=1
    fi
  done
  client_pids=()
  [[ "${client_status}" -eq 0 ]] || return 1

  env LIBERO_CONFIG_PATH="${client_config}" PYTHONPATH="${client_pythonpath}" PYTHONNOUSERSITE=1 \
    "${client_python}" "${runner}" summarize \
    --manifest "${manifest}" --gate "${gate}" --run-dir "${phase_dir}" \
    >"${logs}/summary.log" 2>&1
}

run_phase standard suite-smoke standard_suite_smoke always
run_phase standard slot-smoke standard_slot_smoke always
run_phase standard full standard_full failure
run_phase plus category-smoke plus_category_smoke always
run_phase plus full plus_full failure

"${train_python}" - "${eval_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
standard = json.loads((root / "standard_full/summary.json").read_text())
plus = json.loads((root / "plus_full/summary.json").read_text())
if not standard["complete"] or standard["episodes"] != 2000:
    raise RuntimeError("standard evaluation is incomplete")
if not plus["complete"] or plus["episodes"] != 10030:
    raise RuntimeError("LIBERO-Plus evaluation is incomplete")
summary = {"complete": True, "standard": standard, "plus": plus}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
report = (
    "# Qwen3-VL QwenPI LIBERO 4-in-1 30K\n\n"
    f"- Checkpoint: 30,000 optimizer steps\n"
    f"- Horizon/replan: 32/24\n"
    f"- Standard: {standard['overall']['successes']}/2000 ({standard['overall']['success_rate']:.2%})\n"
    f"- LIBERO-Plus: {plus['overall']['successes']}/10030 ({plus['overall']['success_rate']:.2%})\n"
    "- Full per-suite, per-task, per-category metrics and retry evidence are in the adjacent summaries.\n"
)
(root / "report.md").write_text(report)
PY

cat "${eval_root}/summary.json"
