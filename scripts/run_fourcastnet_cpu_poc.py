#!/usr/bin/env python3
"""One-file FourCastNet CPU Batch Transform canary.

Writes a 1-line JSONL manifest pointing at a single .npy snapshot in S3, then
delegates to scripts/run_fourcastnet_batch_transform.py for the
create_transform_job preview (dry run by default) or real call (--execute).
AWS actions (--create-model, --upload-manifest, --execute) are gated behind an
STS identity check and a no-endpoints pre-check.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fourcastnet_batch_cpu_poc_v1.json"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "fourcastnet" / "cpu_poc"

RUN_BATCH_TRANSFORM = REPO_ROOT / "scripts" / "run_fourcastnet_batch_transform.py"
DESCRIBE_OR_CREATE_MODEL = REPO_ROOT / "scripts" / "describe_or_create_fourcastnet_model.py"
CHECK_NO_ENDPOINTS = REPO_ROOT / "scripts" / "check_no_fourcastnet_endpoints.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare (and optionally run) one SageMaker Batch Transform canary for a single "
            "FourCastNet .npy snapshot. Dry-run by default; no AWS calls unless --create-model, "
            "--upload-manifest, or --execute is passed."
        )
    )
    parser.add_argument("--mode", choices=["metadata_only", "forward"], default="metadata_only")
    parser.add_argument("--input-s3-uri", required=True, help="S3 URI of the single .npy snapshot")
    parser.add_argument("--output-s3-uri", default="", help="Defaults to config's output_s3_uri")
    parser.add_argument("--manifest-s3-uri", default="", help="Defaults to config's input_s3_uri")
    parser.add_argument(
        "--max-runtime-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="max_runtime_guard value embedded in the manifest",
    )
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to CPU POC model config JSON")
    parser.add_argument("--create-model", action="store_true", help="Describe/create the CPU POC model")
    parser.add_argument("--upload-manifest", action="store_true", help="Upload the manifest to --manifest-s3-uri")
    parser.add_argument("--execute", action="store_true", help="Actually create the Batch Transform job")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default="sbnai-725")
    return parser.parse_args()


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    print("Command:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    return result


def main() -> int:
    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    transform = config["transform"]

    output_s3_uri = args.output_s3_uri or config["output_s3_uri"]
    manifest_s3_uri = args.manifest_s3_uri or config["input_s3_uri"]

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "input_s3_uri": args.input_s3_uri,
        "mode": args.mode,
        "output_s3_uri": output_s3_uri,
        "max_runtime_guard": args.max_runtime_guard,
    }
    manifest_path = artifacts_dir / f"{args.mode}_canary_manifest.jsonl"
    manifest_line = json.dumps(manifest)
    manifest_path.write_text(manifest_line + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}\n{manifest_line}")

    needs_aws = args.create_model or args.upload_manifest or args.execute
    if needs_aws:
        print("\n=== AWS identity check ===")
        try:
            sts = _run(["aws", "sts", "get-caller-identity", "--profile", args.profile, "--region", args.region])
        except FileNotFoundError:
            print("ERROR: aws CLI not found.", file=sys.stderr)
            return 1
        if sts.returncode != 0:
            print("ERROR: aws sts get-caller-identity failed; aborting before any AWS mutation.", file=sys.stderr)
            return 1

        print("\n=== Pre-check: no FourCastNet endpoints ===")
        check = _run([sys.executable, str(CHECK_NO_ENDPOINTS), "--region", args.region, "--profile", args.profile])
        if check.returncode != 0:
            return check.returncode

        if args.create_model:
            print("\n=== Describe/create CPU POC model ===")
            result = _run([
                sys.executable, str(DESCRIBE_OR_CREATE_MODEL),
                "--config", args.config, "--region", args.region, "--profile", args.profile,
            ])
            if result.returncode != 0:
                return result.returncode

        if args.upload_manifest:
            print("\n=== Upload manifest ===")
            print("Command:", f"aws s3 cp {manifest_path} {manifest_s3_uri}")
            try:
                import boto3
            except ModuleNotFoundError:
                print("ERROR: boto3 is required for --upload-manifest.", file=sys.stderr)
                return 1
            bucket, _, key = manifest_s3_uri[5:].partition("/")
            session_kwargs: dict[str, str] = {"region_name": args.region}
            if args.profile:
                session_kwargs["profile_name"] = args.profile
            boto3.Session(**session_kwargs).client("s3").upload_file(str(manifest_path), bucket, key)
            print(f"Uploaded {manifest_path} -> {manifest_s3_uri}")

    print("\n=== Transform job ===")
    cmd = [
        sys.executable, str(RUN_BATCH_TRANSFORM),
        "--config", args.config,
        "--model-name", config["model_name"],
        "--input-s3-uri", manifest_s3_uri,
        "--output-s3-uri", output_s3_uri,
        "--instance-type", str(transform["instance_type"]),
        "--instance-count", str(transform["instance_count"]),
        "--max-concurrent-transforms", str(transform["max_concurrent_transforms"]),
        "--max-payload-mb", str(transform["max_payload_in_mb"]),
        "--strategy", transform["batch_strategy"],
        "--content-type", transform["content_type"],
        "--split-type", transform["split_type"],
        "--assemble-with", transform["assemble_with"],
        "--accept", transform["accept"],
        "--region", args.region,
        "--profile", args.profile,
    ]
    if args.execute:
        cmd.append("--execute")
    result = _run(cmd)
    (artifacts_dir / "transform_job.log").write_text((result.stdout or "") + (result.stderr or ""), encoding="utf-8")
    if result.returncode != 0:
        return result.returncode

    if args.execute:
        print("\n=== Post-check: no FourCastNet endpoints ===")
        post = _run([sys.executable, str(CHECK_NO_ENDPOINTS), "--region", args.region, "--profile", args.profile])
        return post.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
