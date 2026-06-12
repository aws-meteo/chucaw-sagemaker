#!/usr/bin/env python3
"""Run the existing FCN direct tensor smoke script with SageMaker-oriented defaults."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local FourCastNet forward-pass smoke")
    parser.add_argument(
        "--input-tensor",
        default="chucaw-glue-scripts/data/fourcastnet_tensor_real_v1/input_tensor.npy",
        help="Path to FCN input tensor",
    )
    parser.add_argument(
        "--global-means",
        default="chucaw-glue-scripts/data/fourcastnet_assets_v0/global_means.npy",
        help="Path to FCN global means",
    )
    parser.add_argument(
        "--global-stds",
        default="chucaw-glue-scripts/data/fourcastnet_assets_v0/global_stds.npy",
        help="Path to FCN global stds",
    )
    parser.add_argument(
        "--checkpoint",
        default="chucaw-glue-scripts/data/fourcastnet_assets_v0/backbone.ckpt",
        help="Path to FCN checkpoint",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "nvlabs", "earth2studio", "physicsnemo", "dry_random"],
        default="auto",
        help="Backend passed through to direct tensor smoke runner",
    )
    parser.add_argument(
        "--local-repo",
        default="",
        help="Optional local FourCastNet repo path for NVlabs backend",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Requested device",
    )
    parser.add_argument(
        "--allow-random-model",
        action="store_true",
        help="Allow dry_random backend (plumbing only, not FCN proof)",
    )
    parser.add_argument(
        "--output-report",
        default="artifacts/fourcastnet/local_forward_smoke_report.json",
        help="Output report path",
    )
    return parser.parse_args()


def resolve(path: str, repo_root: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    return (repo_root / value).resolve()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    runner = repo_root / "chucaw-glue-scripts" / "scripts" / "dev" / "fourcastnet_direct_tensor_smoke.py"
    if not runner.exists():
        print(f"ERROR: missing runner script: {runner}", file=sys.stderr)
        return 1

    output_report = resolve(args.output_report, Path.cwd())
    output_report.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(runner),
        "--INPUT_TENSOR",
        str(resolve(args.input_tensor, repo_root)),
        "--GLOBAL_MEANS",
        str(resolve(args.global_means, repo_root)),
        "--GLOBAL_STDS",
        str(resolve(args.global_stds, repo_root)),
        "--CHECKPOINT",
        str(resolve(args.checkpoint, repo_root)),
        "--MODEL_BACKEND",
        args.backend,
        "--DEVICE",
        args.device,
        "--OUTPUT_REPORT",
        str(output_report),
    ]
    if args.local_repo.strip():
        command.extend(["--LOCAL_REPO", str(resolve(args.local_repo, repo_root))])
    if args.allow_random_model:
        command.extend(["--ALLOW_RANDOM_MODEL", "true"])

    started = time.time()
    proc = subprocess.run(command, capture_output=True, text=True)
    duration_s = time.time() - started
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)

    wrapper = {
        "ok": proc.returncode == 0,
        "return_code": proc.returncode,
        "duration_seconds": duration_s,
        "command": command,
        "runner_script": str(runner),
        "report_path": str(output_report),
        "report_exists": output_report.exists(),
    }
    if output_report.exists():
        try:
            runner_report = json.loads(output_report.read_text(encoding="utf-8"))
            wrapper["runner_ok"] = runner_report.get("ok")
            wrapper["fourcastnet_proven"] = runner_report.get("fourcastnet_proven")
            wrapper["backend_used"] = runner_report.get("model_backend_used")
        except Exception as exc:  # pragma: no cover - defensive path
            wrapper["report_read_error"] = str(exc)

    wrapper_path = output_report.with_name(output_report.stem + "_wrapper.json")
    wrapper_path.write_text(json.dumps(wrapper, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrapper report: {wrapper_path}")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())

