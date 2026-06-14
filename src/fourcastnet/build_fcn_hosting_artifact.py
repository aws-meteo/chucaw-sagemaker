#!/usr/bin/env python3
"""Build FCN model/source hosting artifacts using the separate-source pattern.

This script is local-only by default and never contacts AWS unless ``--execute`` is used.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


MODEL_TAR_NAME = "model.tar.gz"
SOURCE_TAR_NAME = "source.tar.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FCN hosting artifacts")
    parser.add_argument(
        "--assets-dir",
        default="chucaw-glue-scripts/data/fourcastnet_assets_v0",
        help="Directory containing backbone.ckpt and stats files",
    )
    parser.add_argument(
        "--serving-dir",
        default="src/fourcastnet/serving",
        help="Serving source directory relative to chucaw-sagemaker",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/fourcastnet/build",
        help="Output directory for model.tar.gz and source.tar.gz",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute AWS uploads. Without this flag, only local artifacts and commands are produced.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Deprecated alias for enabling uploads. Requires --execute.",
    )
    parser.add_argument("--bucket", default="", help="Common S3 bucket for model+source")
    parser.add_argument("--model-bucket", default="", help="S3 bucket for model artifact")
    parser.add_argument("--source-bucket", default="", help="S3 bucket for source artifact")
    parser.add_argument("--model-prefix", default="", help="S3 key prefix for model artifact")
    parser.add_argument("--source-prefix", default="", help="S3 key prefix for source artifact")
    parser.add_argument("--profile", default="sbnai-725", help="AWS profile for printed/executed commands")
    parser.add_argument("--region", default="us-east-1", help="AWS region for printed/executed commands")
    return parser.parse_args()


def resolve(path: str, root: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return (root / raw).resolve()


def read_tar_entries(tar_path: Path) -> set[str]:
    with tarfile.open(tar_path, "r:gz") as tar:
        return {name.rstrip("/") for name in tar.getnames()}


def require_entries(entries: set[str], expected: Iterable[str], label: str) -> None:
    missing = sorted(set(expected) - entries)
    if missing:
        raise RuntimeError(f"{label} missing required entries: {missing}")


def build_model_tar(assets_dir: Path, model_tar: Path) -> None:
    required = ["backbone.ckpt", "global_means.npy", "global_stds.npy"]
    for name in required:
        file_path = assets_dir / name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required model file: {file_path}")

    model_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(model_tar, "w:gz") as tar:
        for name in required:
            tar.add(assets_dir / name, arcname=name)

    entries = read_tar_entries(model_tar)
    require_entries(entries, required, "model.tar.gz")
    if any(item.startswith("code/") for item in entries):
        raise RuntimeError("model.tar.gz must not include code/ in separate-source pattern")


def build_source_tar(serving_dir: Path, source_tar: Path) -> None:
    required = ["inference.py", "requirements.txt"]
    for name in required:
        file_path = serving_dir / name
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required source file: {file_path}")

    source_tar.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source_tar, "w:gz") as tar:
        for name in required:
            tar.add(serving_dir / name, arcname=name)

    entries = read_tar_entries(source_tar)
    require_entries(entries, required, "source.tar.gz")


def resolve_upload_target(cli_value: str, env_key: str, default: str = "") -> str:
    return (cli_value or os.getenv(env_key, default)).strip()


def build_upload_command(local_path: Path, s3_uri: str, profile: str, region: str) -> str:
    parts = [
        "aws",
        "s3",
        "cp",
        str(local_path),
        s3_uri,
        "--profile",
        profile,
        "--region",
        region,
    ]
    return " ".join(shlex.quote(part) for part in parts)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    sm_root = Path(__file__).resolve().parents[2]

    load_dotenv(sm_root / ".env")
    load_dotenv(Path.cwd() / ".env")

    assets_dir = resolve(args.assets_dir, repo_root)
    serving_dir = resolve(args.serving_dir, sm_root)
    output_dir = resolve(args.output_dir, sm_root)
    model_tar = output_dir / MODEL_TAR_NAME
    source_tar = output_dir / SOURCE_TAR_NAME

    build_model_tar(assets_dir, model_tar)
    build_source_tar(serving_dir, source_tar)

    print(f"Model artifact:  {model_tar}")
    print(f"Source artifact: {source_tar}")

    wants_upload = bool(args.upload or args.execute)
    if args.upload and not args.execute:
        raise ValueError("Deprecated --upload flag now requires --execute for any AWS submission.")
    if not wants_upload:
        return 0

    common_bucket = args.bucket.strip()
    model_bucket = resolve_upload_target(args.model_bucket, "MODEL_S3_BUCKET", default=common_bucket)
    source_bucket = resolve_upload_target(args.source_bucket, "SERVING_SOURCE_S3_BUCKET", default=common_bucket)
    model_prefix = resolve_upload_target(args.model_prefix, "MODEL_S3_PREFIX")
    source_prefix = resolve_upload_target(args.source_prefix, "SERVING_SOURCE_S3_PREFIX")

    if not model_bucket or not source_bucket:
        raise ValueError("Model/source bucket not resolved. Set --bucket or explicit --model-bucket/--source-bucket")
    if not model_prefix or not source_prefix:
        raise ValueError("Model/source prefix not resolved. Set --model-prefix and --source-prefix or env vars")

    profile = args.profile.strip()
    region = args.region.strip()
    if not profile:
        raise ValueError("Missing required value: --profile")
    if not region:
        raise ValueError("Missing required value: --region")

    model_uri = f"s3://{model_bucket}/{model_prefix.strip('/')}/{MODEL_TAR_NAME}"
    source_uri = f"s3://{source_bucket}/{source_prefix.strip('/')}/{SOURCE_TAR_NAME}"
    model_cmd = build_upload_command(model_tar, model_uri, profile, region)
    source_cmd = build_upload_command(source_tar, source_uri, profile, region)

    print("=== AWS Upload Commands ===")
    print(model_cmd)
    print(source_cmd)

    if not args.execute:
        print("Dry-run only. No AWS upload was executed. Re-run with --execute to submit.")
        return 0

    for cmd in (model_cmd, source_cmd):
        result = subprocess.run(cmd, shell=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"AWS upload command failed with exit code {result.returncode}: {cmd}")
    print(f"Uploaded model artifact:  {model_uri}")
    print(f"Uploaded source artifact: {source_uri}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
