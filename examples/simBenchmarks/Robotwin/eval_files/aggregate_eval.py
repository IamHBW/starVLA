#!/usr/bin/env python3
"""Validate and aggregate the fixed Qwen3-VL-PI RoboTwin evaluation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


MODEL = "qwen3_vl_pi_c50_step30000"
MODEL_LABEL = "Qwen3-VL-PI C50（step 30k）"
PHASES = {"clean": ("demo_clean", "clean"), "randomized": ("demo_randomized", "random")}
TASKS = (
    "adjust_bottle", "beat_block_hammer", "blocks_ranking_rgb", "blocks_ranking_size",
    "click_alarmclock", "click_bell", "dump_bin_bigbin", "grab_roller", "handover_block",
    "handover_mic", "hanging_mug", "lift_pot", "move_can_pot", "move_pillbottle_pad",
    "move_playingcard_away", "move_stapler_pad", "open_laptop", "open_microwave",
    "pick_diverse_bottles", "pick_dual_bottles", "place_a2b_left", "place_a2b_right",
    "place_bread_basket", "place_bread_skillet", "place_burger_fries", "place_can_basket",
    "place_cans_plasticbox", "place_container_plate", "place_dual_shoes", "place_empty_cup",
    "place_fan", "place_mouse_pad", "place_object_basket", "place_object_scale",
    "place_object_stand", "place_phone_stand", "place_shoe", "press_stapler",
    "put_bottles_dustbin", "put_object_cabinet", "rotate_qrcode", "scan_object",
    "shake_bottle_horizontally", "shake_bottle", "stack_blocks_three", "stack_blocks_two",
    "stack_bowls_three", "stack_bowls_two", "stamp_seal", "turn_switch",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PROGRESS = re.compile(r"Success rate:\s*(\d+)/(\d+).*current seed:\s*(\d+)")
WORKER = re.compile(r"_slot(\d+)_gpu([^_]+)_port(\d+)_eval\.log$")
VIDEO = re.compile(r"episode(\d+)_(clean|random)_success-(true|false)\.mp4$")
REFERENCE_HASHES = {
    "summary.json": "07615df675d5e37cdcf8f13e319e2dfcf2ecf3c1e38f29ca921f6145225ca3c9",
    "provenance.json": "5af1bf649ff3d503c05a9491db13e4d20a17b6622347d291e8ef18542926bf9e",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def _run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()


def _markers(text: str, prefix: str) -> list[dict]:
    return [json.loads(line.split(prefix, 1)[1]) for line in text.splitlines() if prefix in line]


def _candidate_logs(root: Path, task: str, task_config: str) -> list[Path]:
    return sorted(
        (root / "logs").glob(f"{task}_{task_config}_slot*_eval.log"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )


def _parse_log(path: Path, episodes: int):
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    prompts = _markers(text, "[ROBOTWIN_EPISODE] ")
    actions = _markers(text, "[ROBOTWIN_ACTION] ")
    progress = [tuple(map(int, match.groups())) for match in PROGRESS.finditer(text)]
    if len(prompts) != episodes or len(actions) != episodes or len(progress) != episodes:
        raise ValueError(
            f"incomplete structured log {path}: prompts={len(prompts)}, "
            f"actions={len(actions)}, progress={len(progress)}"
        )
    if [total for _, total, _ in progress] != list(range(1, episodes + 1)):
        raise ValueError(f"non-contiguous episode counters in {path}")
    for prompt, action in zip(prompts, actions, strict=True):
        if prompt.get("instruction_split") != "unseen" or not str(prompt.get("instruction", "")).strip():
            raise ValueError(f"invalid unseen instruction marker in {path}")
        if action != {"action_chunk_size": 32, "finite": True, "shape": [32, 14]}:
            raise ValueError(f"invalid action contract marker in {path}: {action}")
    return prompts, progress


def parse_cell(run_root: Path, phase: str, task: str, episodes: int) -> list[dict]:
    task_config, suffix = PHASES[phase]
    cell = run_root / phase / "results" / task / task_config
    result = cell / f"_result_{suffix}.txt"
    if not result.is_file() or "Instruction Type: unseen" not in result.read_text(errors="replace"):
        raise ValueError(f"missing unseen result: {result}")

    videos = {}
    for path in cell.glob(f"episode*_{suffix}_success-*.mp4"):
        match = VIDEO.fullmatch(path.name)
        if match:
            index, actual_suffix, success = match.groups()
            if actual_suffix == suffix and path.stat().st_size > 0:
                videos[int(index)] = (path.resolve(), success == "true")
    if sorted(videos) != list(range(episodes)):
        raise ValueError(f"expected {episodes} videos for {task}/{phase}, got {sorted(videos)}")

    parsed = None
    log = None
    errors = []
    for candidate in _candidate_logs(run_root / phase, task, task_config):
        try:
            parsed = _parse_log(candidate, episodes)
            log = candidate
            break
        except ValueError as error:
            errors.append(str(error))
    if parsed is None or log is None:
        raise ValueError("; ".join(errors) or f"missing eval log for {task}/{phase}")
    prompts, progress = parsed
    worker_match = WORKER.search(log.name)
    if not worker_match:
        raise ValueError(f"cannot parse worker from {log}")
    slot, gpu, port = worker_match.groups()

    rows = []
    previous_successes = 0
    previous_seed = 99999
    for index, ((successes, total, seed), prompt) in enumerate(zip(progress, prompts, strict=True)):
        success = successes - previous_successes == 1
        if successes - previous_successes not in (0, 1) or videos[index][1] != success:
            raise ValueError(f"success evidence mismatch for {task}/{phase}/{index}")
        rows.append(
            {
                "model": MODEL,
                "task": task,
                "phase": phase,
                "task_config": task_config,
                "episode_index": index,
                "accepted_seed": seed,
                "rejected_seeds": list(range(previous_seed + 1, seed)),
                "instruction": prompt["instruction"],
                "instruction_split": "unseen",
                "success": success,
                "worker": {"slot": int(slot), "gpu": gpu, "port": int(port)},
                "exception": None,
                "rpc_error": None,
                "simulation_error": None,
                "action_chunk_size": 32,
                "policy_output_shape": [32, 14],
                "policy_actions_finite": True,
                "state_forwarded": False,
                "camera_order": ["head", "left", "right"],
                "image_size": [224, 224],
                "video_path": str(videos[index][0]),
                "eval_log": str(log.resolve()),
            }
        )
        previous_successes = successes
        previous_seed = seed
    rate = float(result.read_text().splitlines()[-1])
    if abs(rate - previous_successes / episodes) > 1e-9:
        raise ValueError(f"result rate mismatch for {task}/{phase}: {rate}")
    return rows


def missing(args) -> None:
    root = Path(args.run_root).resolve()
    for task in TASKS:
        try:
            parse_cell(root, args.phase, task, args.episodes)
        except (OSError, ValueError, json.JSONDecodeError):
            print(task)


def _aggregate(rows: list[dict], episodes: int) -> dict:
    per_task = {
        task: {
            MODEL: {
                phase: sum(row["success"] for row in rows if row["task"] == task and row["phase"] == phase)
                for phase in PHASES
            }
        }
        for task in TASKS
    }
    denominator = len(TASKS) * episodes
    clean = sum(row["success"] for row in rows if row["phase"] == "clean")
    randomized = sum(row["success"] for row in rows if row["phase"] == "randomized")
    return {
        "protocol": {
            "task_count": 50,
            "episodes_per_task_phase": episodes,
            "phases": list(PHASES),
            "training_instruction_type": "seen",
            "evaluation_instruction_type": "unseen",
            "action_chunk_size": 32,
            "total_episodes": len(rows),
        },
        "aggregate": {
            MODEL: {
                "clean_successes": clean,
                "clean_rate": clean / denominator,
                "randomized_successes": randomized,
                "randomized_rate": randomized / denominator,
                "combined_successes": clean + randomized,
                "combined_rate": (clean + randomized) / (2 * denominator),
                "randomized_drop_pp": (randomized - clean) / denominator * 100,
            }
        },
        "per_task": per_task,
    }


def _check_reference(reference: Path) -> dict:
    for name, expected in REFERENCE_HASHES.items():
        actual = hashlib.sha256((reference / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"reference {name} hash mismatch: {actual}")
    return _json(reference / "summary.json")


def _copy_representative_videos(rows: list[dict], report: Path) -> list[dict]:
    output = report / "videos"
    output.mkdir(parents=True, exist_ok=True)
    selected = []
    for phase in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase]
        for success in (True, False):
            row = next((item for item in phase_rows if item["success"] is success), None)
            if row is None:
                continue
            target = output / f"q3pi_{phase}_{row['task']}_success-{str(success).lower()}.mp4"
            shutil.copy2(row["video_path"], target)
            _run(
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(target),
            )
            selected.append({"task": row["task"], "phase": phase, "success": success, "path": str(target)})
    return selected


def _write_report(report: Path, summary: dict, selected: list[dict]) -> None:
    model_keys = list(summary["aggregate"])
    denominator = summary["protocol"]["task_count"] * summary["protocol"]["episodes_per_task_phase"]
    overall = "\n".join(
        f"<tr><td>{escape(MODEL_LABEL if model == MODEL else model)}</td>"
        f"<td>{values['clean_successes']}/{denominator}（{values['clean_rate']:.1%}）</td>"
        f"<td>{values['randomized_successes']}/{denominator}（{values['randomized_rate']:.1%}）</td>"
        f"<td>{values['combined_successes']}/{2 * denominator}（{values['combined_rate']:.1%}）</td>"
        f"<td>{values['randomized_drop_pp']:+.1f} pp</td></tr>"
        for model, values in summary["aggregate"].items()
    )
    tasks = []
    for task in TASKS:
        cells = []
        for model in model_keys:
            values = summary["per_task"][task][model]
            cells.append(
                f"<td>{values['clean']}/20（{values['clean'] / 20:.1%}）</td>"
                f"<td>{values['randomized']}/20（{values['randomized'] / 20:.1%}）</td>"
            )
        tasks.append(f"<tr><td>{escape(task)}</td>{''.join(cells)}</tr>")
    headers = "".join(
        f"<th colspan=\"2\">{escape(MODEL_LABEL if model == MODEL else model)}</th>"
        for model in model_keys
    )
    subheaders = "".join("<th>Clean</th><th>Randomized</th>" for _ in model_keys)
    figures = "\n".join(
        f"<figure><video controls preload=\"metadata\" src=\"{escape(str(Path(item['path']).relative_to(report)))}\"></video>"
        f"<figcaption>{escape(item['phase'])} · {escape(item['task'])} · success={item['success']}</figcaption></figure>"
        for item in selected
    )
    values = summary["aggregate"][MODEL]
    task_values = {task: summary["per_task"][task][MODEL] for task in TASKS}
    clean_ge_50 = sum(item["clean"] >= 10 for item in task_values.values())
    randomized_ge_50 = sum(item["randomized"] >= 10 for item in task_values.values())
    randomized_zero = sum(item["randomized"] == 0 for item in task_values.values())
    randomized_top = sorted(TASKS, key=lambda task: task_values[task]["randomized"], reverse=True)[:5]
    randomized_top_text = "、".join(
        f"<code>{escape(task)}</code> {task_values[task]['randomized']}/20" for task in randomized_top
    )
    body = f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Qwen3-VL-PI RoboTwin Unseen Eval</title>
<style>:root{{color-scheme:light dark;font-family:system-ui,sans-serif}}body{{max-width:1280px;margin:auto;padding:32px 20px 56px;line-height:1.5}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}th,td{{border:1px solid #8886;padding:8px 10px;text-align:right}}th:first-child,td:first-child{{text-align:left}}.scroll{{overflow:auto;margin:20px 0}}.lead,.warning{{padding:14px 16px;border-left:4px solid #4f8cff;background:#4f8cff12}}.warning{{border-color:#e8a43a;background:#e8a43a12}}.videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}}figure{{margin:0;padding:12px;border:1px solid #8885;border-radius:10px}}video{{width:100%;background:#000;border-radius:6px}}figcaption{{margin-top:8px}}</style></head><body>
<h1>Qwen3-VL-PI RoboTwin Unseen Eval</h1><p>step 30k · global batch 256 · train instruction: seen · eval instruction: unseen · 50 tasks × 20 episodes/phase · Clean + Randomized · 2,000 episodes · action chunk 32</p>
<p class=\"warning\"><strong>报告边界：</strong>这是当前 Qwen3-VL-PI checkpoint 的独立描述性报告，不与参考报告中的五个模型合并比较。参考报告仅用于版式，原文件未修改；完整 checkpoint、代码、RoboTwin vendor、环境、HTrain JOBID 与日志溯源见 <code>provenance.json</code>。</p>
<h2>总体结果</h2><div class=\"scroll\"><table><tr><th>模型</th><th>Clean</th><th>Randomized</th><th>两阶段合计</th><th>Clean → Randomized</th></tr>{overall}</table></div>
<p class=\"lead\"><strong>核心结果：</strong>Clean 为 {values['clean_successes']}/{denominator}（{values['clean_rate']:.1%}），Randomized 为 {values['randomized_successes']}/{denominator}（{values['randomized_rate']:.1%}），随机化环境下降 {abs(values['randomized_drop_pp']):.1f} 个百分点；两阶段合计 {values['combined_successes']}/{2 * denominator}（{values['combined_rate']:.1%}）。</p>
<h2>任务分布</h2><p>Clean 有 {clean_ge_50}/50 个任务达到至少 50% 成功率；Randomized 仅 {randomized_ge_50}/50 个达到至少 50%，且 {randomized_zero}/50 个任务为 0/20。Randomized 表现最好的五项是：{randomized_top_text}。</p>
<h2>逐任务</h2><div class=\"scroll\"><table><tr><th rowspan=\"2\">Task</th>{headers}</tr><tr>{subheaders}</tr>{''.join(tasks)}</table></div>
<h2>代表视频</h2><div class=\"videos\">{figures}</div>
<p>逐 episode 证据见 <code>../results_detailed.jsonl</code>；机器可读汇总见 <code>summary.json</code> 和 <code>summary.csv</code>。</p></body></html>"""
    (report / "index.html").write_text(body, encoding="utf-8")
    (report / "index_rgb.html").write_text(body.replace("<h1>", "<p>RGB 输入：head → left → right，224×224。</p><h1>"), encoding="utf-8")


def collect(args) -> None:
    root = Path(args.run_root).resolve()
    rows = [row for phase in PHASES for task in TASKS for row in parse_cell(root, phase, task, args.episodes)]
    keys = {(row["task"], row["phase"], row["episode_index"]) for row in rows}
    expected = len(TASKS) * len(PHASES) * args.episodes
    if len(rows) != expected or len(keys) != expected:
        raise ValueError(f"expected {expected} unique episodes, got rows={len(rows)} keys={len(keys)}")
    detailed = root / "results_detailed.jsonl"
    detailed.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))
    own = _aggregate(rows, args.episodes)
    for phase in PHASES:
        phase_rows = [row for row in rows if row["phase"] == phase]
        _write_json(root / phase / "summary.json", {
            "phase": phase, "episodes": len(phase_rows), "successes": sum(row["success"] for row in phase_rows)
        })

    reference = Path(args.reference).resolve()
    _check_reference(reference)
    report = root / "report"
    report.mkdir(parents=True, exist_ok=True)
    selected = _copy_representative_videos(rows, report)
    _write_json(report / "summary.json", own)
    with (report / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["task"] + [f"{MODEL}_{phase}_successes" for phase in PHASES]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            row = {"task": task}
            for phase in PHASES:
                row[f"{MODEL}_{phase}_successes"] = own["per_task"][task][MODEL][phase]
            writer.writerow(row)
    provenance = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "template_reference": {"path": str(reference), "hashes": REFERENCE_HASHES, "modified": False},
        "qwen3_vl_pi": {
            "phase_provenance": {phase: _json(root / f"provenance_{phase}.json") for phase in PHASES},
            "jobs": _json(root / "jobs.json") if (root / "jobs.json").is_file() else [],
            "results_detailed": str(detailed),
            "selected_videos": selected,
        },
    }
    _write_json(report / "provenance.json", provenance)
    _write_report(report, own, selected)
    print(json.dumps(own["aggregate"][MODEL], indent=2))


def provenance(args) -> None:
    repo = Path(args.repo).resolve()
    vendor = Path(args.vendor).resolve()
    openpi = vendor.parents[1]
    source_files = [
        repo / "examples/simBenchmarks/Robotwin/eval_files/start_eval.sh",
        repo / "examples/simBenchmarks/Robotwin/eval_files/eval.sh",
        repo / "examples/simBenchmarks/Robotwin/eval_files/model2robotwin_interface.py",
        repo / "examples/simBenchmarks/Robotwin/eval_files/run_eval.sh",
        Path(__file__).resolve(),
    ]
    public_code = """import json, importlib.metadata as m
names=('sapien','mplib','toppra','numpy','torch','gymnasium','opencv-python','PyYAML')
print(json.dumps({name:m.version(name) for name in names},sort_keys=True))"""
    payload = {
        "phase": args.phase,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "starvla": {
            "repo": str(repo),
            "commit": _run("git", "rev-parse", "HEAD", cwd=repo),
            "status": _run("git", "status", "--short", cwd=repo).splitlines(),
            "source_hashes": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
            "python": args.starvla_python,
        },
        "checkpoint": {"path": args.checkpoint, "sha256": args.checkpoint_sha256},
        "training_config": {"path": args.config, "sha256": args.config_sha256, "global_batch_size": 256},
        "robotwin_vendor": {
            "path": str(vendor),
            "openpi_commit": _run("git", "rev-parse", "HEAD", cwd=openpi),
            "git_tree": _run("git", "ls-tree", "HEAD", "third_party/RoboTwin", cwd=openpi).split()[2],
            "status": _run("git", "status", "--short", "--", "third_party/RoboTwin", cwd=openpi).splitlines(),
            "upstream_commit": "bf44be51cf5717a5595ce59447f2cf5263d2aa95",
            "copy_source_commit": "c2b62cdd0af0cd97a5840bff49ff3cd379410978",
            "asset_copy_origins": [
                "/mnt/data/users/wanqi/workspace/code/Robot/RoboTwin",
                "/mnt/data/users/wanchi/workspace/code/FloWAM/robotwin_collect_flow_data",
            ],
            "runtime_dependency_on_copy_origins": False,
            "runtime_symlinks": [],
        },
        "environment": {
            "starvla": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "opencv-python", "websockets", "omegaconf")},
            "robotwin_python": args.robotwin_python,
            "robotwin": json.loads(_run(args.robotwin_python, "-c", public_code)),
            "gpus": _run("nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader").splitlines(),
        },
        "job_environment": {
            key: os.environ[key]
            for key in (
                "JOB_ID", "JOBID", "HTRAIN_JOB_ID", "Q3PI_HTRAIN_WORLD_SIZE", "Q3PI_HTRAIN_MASTER_ADDR"
            )
            if key in os.environ
        },
    }
    _write_json(Path(args.run_root).resolve() / f"provenance_{args.phase}.json", payload)


def record_job(args) -> None:
    path = Path(args.run_root).resolve() / "jobs.json"
    jobs = _json(path) if path.is_file() else []
    record = {
        "phase": args.phase, "name": args.name, "job_id": args.job_id, "project": args.project,
        "resources": "1 node × 8 GPUs", "status": args.status, "command": args.command,
        "raw_log": args.raw_log, "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    jobs = [job for job in jobs if job.get("job_id") != args.job_id] + [record]
    _write_json(path, jobs)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    missing_parser = commands.add_parser("missing")
    missing_parser.add_argument("--run-root", required=True)
    missing_parser.add_argument("--phase", choices=PHASES, required=True)
    missing_parser.add_argument("--episodes", type=int, default=20)
    missing_parser.set_defaults(func=missing)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--run-root", required=True)
    collect_parser.add_argument("--reference", required=True)
    collect_parser.add_argument("--episodes", type=int, default=20)
    collect_parser.set_defaults(func=collect)
    provenance_parser = commands.add_parser("provenance")
    for flag in ("run-root", "phase", "repo", "vendor", "starvla-python", "robotwin-python", "checkpoint", "checkpoint-sha256", "config", "config-sha256"):
        provenance_parser.add_argument(f"--{flag}", required=True)
    provenance_parser.set_defaults(func=provenance)
    job_parser = commands.add_parser("record-job")
    for flag in ("run-root", "phase", "name", "job-id", "project", "status", "command", "raw-log"):
        job_parser.add_argument(f"--{flag}", required=True)
    job_parser.set_defaults(func=record_job)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
