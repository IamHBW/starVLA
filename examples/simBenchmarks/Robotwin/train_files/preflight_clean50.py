#!/usr/bin/env python3
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_DATA_HASHES = {
    "conversion_manifest.json": "444a4e1cda53ffd3904cd4217a1e53981eef96e515dac13f307c2dd02b20dc91",
    "meta/info.json": "b3058400f531071f43c18c448eccbef6fc37977b4d4320b01ee1c288e8e8d84d",
}
EXPECTED_MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*command):
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout.strip()


def owner(path):
    stat = path.stat()
    import grp
    import pwd

    return f"{pwd.getpwuid(stat.st_uid).pw_name}:{grp.getgrgid(stat.st_gid).gr_name}"


def ensure_link(link, target):
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"Wrong overlay link: {link} -> {link.resolve()}")
        return
    if link.exists():
        raise FileExistsError(f"Refusing to replace overlay path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "validate"), required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    source = args.source.resolve()
    model = args.model.resolve()
    if model.name != EXPECTED_MODEL_REVISION:
        raise RuntimeError(f"Expected Qwen3 revision {EXPECTED_MODEL_REVISION}, got {model}")

    hashes = {name: sha256(source / name) for name in EXPECTED_DATA_HASHES}
    if hashes != EXPECTED_DATA_HASHES:
        raise RuntimeError(f"Frozen data hash mismatch: {hashes}")

    conversion = json.loads((source / "conversion_manifest.json").read_text())
    info = json.loads((source / "meta/info.json").read_text())
    if (len(conversion["tasks"]), conversion["totals"]["episodes"], info["total_frames"]) != (50, 2500, 560573):
        raise RuntimeError("Clean50 dataset count mismatch")

    overlay_dataset = run_dir / "data_overlay" / "robotwin_clean50"
    overlay_meta = overlay_dataset / "meta"
    overlay_meta.mkdir(parents=True, exist_ok=True)
    ensure_link(overlay_dataset / "data", source / "data")
    ensure_link(overlay_dataset / "videos", source / "videos")
    ensure_link(overlay_dataset / "conversion_manifest.json", source / "conversion_manifest.json")
    for name in ("episodes.jsonl", "episodes_stats.jsonl", "info.json", "tasks.jsonl"):
        ensure_link(overlay_meta / name, source / "meta" / name)
    modality_source = repo / "examples/simBenchmarks/Robotwin/train_files/modality_clean50.json"
    modality_target = overlay_meta / "modality.json"
    if modality_target.exists() and sha256(modality_target) != sha256(modality_source):
        raise RuntimeError(f"Run-local modality changed: {modality_target}")
    if not modality_target.exists():
        shutil.copy2(modality_source, modality_target)

    model_hashes = {
        str(path.relative_to(model)): sha256(path)
        for path in sorted(model.rglob("*"))
        if path.is_file()
    }
    if not model_hashes:
        raise RuntimeError(f"Empty model snapshot: {model}")

    diff = run("git", "-C", str(repo), "diff", "--binary")
    diff_path = run_dir / "git_diff.patch"
    diff_path.write_text(diff + ("\n" if diff else ""))

    accelerate_config = repo / "starVLA/config/deepseeds/deepspeed_zero2.yaml"
    deepspeed_config = repo / "starVLA/config/deepseeds/ds_config.yaml"
    resolved_ds = json.loads(deepspeed_config.read_text())
    resolved_ds["train_micro_batch_size_per_gpu"] = 16
    resolved_ds["gradient_accumulation_steps"] = 2
    resolved_ds["train_batch_size"] = 256
    resolved_ds_path = run_dir / "resolved_deepspeed_config.json"
    resolved_ds_path.write_text(json.dumps(resolved_ds, indent=2) + "\n")
    source_ds_path = "./starVLA/config/deepseeds/ds_config.yaml"
    accelerate_text = accelerate_config.read_text()
    if accelerate_text.count(source_ds_path) != 1:
        raise RuntimeError(f"Unexpected DeepSpeed config reference in {accelerate_config}")
    (run_dir / "accelerate_config.yaml").write_text(
        accelerate_text.replace(source_ds_path, str(resolved_ds_path))
    )

    packages = {}
    for name in ("torch", "transformers", "accelerate", "deepspeed", "wandb", "numpy", "pandas", "pyarrow", "av"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    event = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "command": " ".join(os.sys.argv),
        "git_sha": run("git", "-C", str(repo), "rev-parse", "HEAD"),
        "git_diff_sha256": sha256(diff_path),
        "data": {
            "source": str(source),
            "owner": owner(source),
            "hashes": hashes,
            "tasks": 50,
            "episodes": 2500,
            "transitions": 560573,
            "original_source": "/mnt/data/public_data/robotwin",
            "original_source_owner": owner(Path("/mnt/data/public_data/robotwin")),
        },
        "model": {"snapshot": str(model), "revision": model.name, "file_sha256": model_hashes},
        "overlay": str(overlay_dataset),
        "packages": packages,
        "gpu": run("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"),
        "disk": run("df", "-h", "/", "/mnt/data"),
        "wandb": {
            "mode": os.environ.get("WANDB_MODE"),
            "entity": os.environ.get("WANDB_ENTITY"),
            "project": os.environ.get("WANDB_PROJECT"),
            "credentials_source": "/mnt/data/users/bowen/workspace/tokens.sh",
            "credentials_owner": owner(Path("/mnt/data/users/bowen/workspace/tokens.sh")),
        },
        "job_environment": {
            key: os.environ[key]
            for key in ("JOB_ID", "JOBID", "HTRAIN_JOB_ID", "MASTER_ADDR", "WORLD_SIZE")
            if key in os.environ
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"preflight_events": []}
    manifest["preflight_events"].append(event)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "overlay": str(overlay_dataset), "mode": args.mode}))


if __name__ == "__main__":
    main()
