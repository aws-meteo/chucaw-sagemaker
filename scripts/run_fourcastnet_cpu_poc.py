#!/usr/bin/env python3
"""One-file FourCastNet CPU Batch Transform canary orchestrator.

Builds the manifest (or resolves the raw .npy S3 URI) and a
``create_transform_job`` payload preview for ONE FourCastNet Batch Transform
canary, then optionally delegates to the existing guardrailed scripts to
describe/create the CPU POC model, upload the manifest, and start the
transform job.

Default invocation (no flags beyond --input-s3-uri) makes ZERO AWS calls: no
boto3, no ``aws`` CLI. AWS actions are gated behind --create-model,
--upload-manifest, and --execute. This script never calls CreateEndpoint,
CreateEndpointConfig, or any real-time endpoint deploy API -- the only
mutating SageMaker call is delegated to
scripts/run_fourcastnet_batch_transform.py's create_transform_job, and only
when --execute is passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "fourcastnet_batch_cpu_poc_v1.json"
DEFAULT_ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "fourcastnet" / "cpu_poc"

RUN_BATCH_TRANSFORM = REPO_ROOT / "scripts" / "run_fourcastnet_batch_transform.py"
DESCRIBE_OR_CREATE_MODEL = REPO_ROOT / "scripts" / "describe_or_create_fourcastnet_model.py"
CHECK_NO_ENDPOINTS = REPO_ROOT / "scripts" / "check_no_fourcastnet_endpoints.py"
CHECK_NO_ALWAYS_ON = REPO_ROOT / "scripts" / "check_no_sagemaker_always_on_compute.py"

DEFAULT_PROFILE = "sbnai-725"
DEFAULT_REGION = "us-east-1"

PLACEHOLDER_MANIFEST_S3_URI = "s3://PLACEHOLDER-SET-WITH---manifest-s3-uri/{mode}_canary_manifest.jsonl"

PAYLOAD_START_MARKER = "=== Planned Transform Job Payload ==="
PAYLOAD_END_MARKER = "====================================="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare (and optionally run) one SageMaker Batch Transform canary for a "
            "single FourCastNet .npy snapshot. Dry-run by default; no AWS calls unless "
            "--create-model, --upload-manifest, or --execute is passed."
        )
    )
    parser.add_argument("--mode", choices=["metadata_only", "forward"], default="metadata_only")
    parser.add_argument(
        "--input-contract", choices=["jsonl_pointer", "raw_npy"], default="jsonl_pointer"
    )
    parser.add_argument("--input-s3-uri", required=True, help="S3 URI of the single .npy snapshot")
    parser.add_argument(
        "--output-s3-uri",
        default="",
        help="Optional output_s3_uri (manifest field + TransformOutput.S3OutputPath fallback)",
    )
    parser.add_argument(
        "--max-runtime-guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="max_runtime_guard value embedded in the generated manifest",
    )
    parser.add_argument(
        "--manifest-s3-uri",
        default="",
        help="S3 URI of the (to-be-)uploaded manifest; required for --execute with jsonl_pointer",
    )
    parser.add_argument("--manifest-path", default="", help="Local path for the generated manifest")
    parser.add_argument(
        "--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR), help="Directory for generated artifacts"
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to CPU POC model config JSON")
    parser.add_argument("--create-model", action="store_true", help="Describe/create the CPU POC model")
    parser.add_argument(
        "--upload-manifest", action="store_true", help="Upload the local manifest to --manifest-s3-uri"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Actually create the SageMaker Batch Transform job"
    )
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--job-name-prefix", default="fcn-cpu-poc")
    return parser.parse_args()


def _extract_payload(stdout: str) -> dict[str, Any] | None:
    try:
        start = stdout.index(PAYLOAD_START_MARKER) + len(PAYLOAD_START_MARKER)
        end = stdout.index(PAYLOAD_END_MARKER, start)
    except ValueError:
        return None
    return json.loads(stdout[start:end].strip())


def _build_batch_transform_cmd(
    args: argparse.Namespace,
    config_path: Path,
    transform_cfg: dict[str, Any],
    model_name: str,
    input_s3_uri: str,
    output_s3_uri: str,
    content_type: str,
    split_type: str,
    *,
    execute: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(RUN_BATCH_TRANSFORM),
        "--config",
        str(config_path),
        "--model-name",
        model_name,
        "--input-s3-uri",
        input_s3_uri,
        "--output-s3-uri",
        output_s3_uri,
        "--instance-type",
        str(transform_cfg["instance_type"]),
        "--instance-count",
        str(transform_cfg["instance_count"]),
        "--max-concurrent-transforms",
        str(transform_cfg["max_concurrent_transforms"]),
        "--max-payload-mb",
        str(transform_cfg["max_payload_in_mb"]),
        "--strategy",
        str(transform_cfg["batch_strategy"]),
        "--content-type",
        content_type,
        "--split-type",
        split_type,
        "--assemble-with",
        str(transform_cfg.get("assemble_with", "Line")),
        "--accept",
        str(transform_cfg.get("accept", "application/json")),
        "--job-name-prefix",
        args.job_name_prefix,
        "--region",
        args.region,
        "--profile",
        args.profile,
    ]
    if args.input_contract == "raw_npy":
        cmd.append("--allow-large-direct-payload")
    if execute:
        cmd.append("--execute")
    return cmd


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

    if args.input_contract == "raw_npy" and args.mode == "metadata_only":
        print(
            "ERROR: --input-contract raw_npy --mode metadata_only is not supported.\n"
            "src/fourcastnet/serving/inference.py input_fn forces mode='forward' for "
            "application/x-npy regardless of the requested mode. Use "
            "--input-contract jsonl_pointer for a metadata_only canary, or pass "
            "--mode forward together with --input-contract raw_npy.",
            file=sys.stderr,
        )
        return 1

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_name = str(config["model_name"])
    transform_cfg = config["transform"]

    output_s3_uri_for_transform = args.output_s3_uri or str(config.get("output_s3_uri", ""))
    if not output_s3_uri_for_transform:
        print(
            "ERROR: --output-s3-uri not provided and config has no fallback 'output_s3_uri'.",
            file=sys.stderr,
        )
        return 1

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_absolute():
        artifacts_dir = (Path.cwd() / artifacts_dir).resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Input contract ===")
    manifest_path: Path | None = None
    if args.input_contract == "jsonl_pointer":
        manifest_record: dict[str, Any] = {"input_s3_uri": args.input_s3_uri, "mode": args.mode}
        if args.output_s3_uri:
            manifest_record["output_s3_uri"] = args.output_s3_uri
        manifest_record["max_runtime_guard"] = args.max_runtime_guard

        manifest_path = (
            Path(args.manifest_path) if args.manifest_path else artifacts_dir / f"{args.mode}_canary_manifest.jsonl"
        )
        if not manifest_path.is_absolute():
            manifest_path = (Path.cwd() / manifest_path).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_line = json.dumps(manifest_record)
        manifest_path.write_text(manifest_line + "\n", encoding="utf-8")
        print(f"Wrote 1-line JSONL manifest: {manifest_path}")
        print(manifest_line)

        content_type = "application/jsonlines"
        split_type = "Line"
        transform_input_s3_uri = args.manifest_s3_uri or PLACEHOLDER_MANIFEST_S3_URI.format(mode=args.mode)
        if not args.manifest_s3_uri:
            print(
                "NOTE: --manifest-s3-uri not provided; using a placeholder S3 URI for the "
                "TransformInput preview. run_fourcastnet_batch_transform.py refuses --execute "
                "with a placeholder URI -- pass --manifest-s3-uri (and --upload-manifest if it "
                "still needs uploading) before executing."
            )
    else:
        print("raw_npy contract: no manifest generated; TransformInput.S3Uri is the .npy file itself.")
        content_type = "application/x-npy"
        split_type = "None"
        transform_input_s3_uri = args.input_s3_uri

    print("\n=== Step 2: Transform job spec preview (dry run, no AWS calls) ===")
    preview_cmd = _build_batch_transform_cmd(
        args,
        config_path,
        transform_cfg,
        model_name,
        transform_input_s3_uri,
        output_s3_uri_for_transform,
        content_type,
        split_type,
        execute=False,
    )
    preview = _run(preview_cmd)
    if preview.returncode != 0:
        print("ERROR: transform job spec preview failed.", file=sys.stderr)
        return preview.returncode

    spec = _extract_payload(preview.stdout)
    if spec is not None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        spec_path = artifacts_dir / f"transform_job_{timestamp}.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        print(f"Saved transform job spec preview: {spec_path}")

    needs_aws = args.create_model or args.upload_manifest or args.execute
    if not needs_aws:
        print("\nPure local mode: no --create-model/--upload-manifest/--execute given; no AWS calls made.")
        return 0

    print("\n=== Step 3: AWS identity validation (STS) ===")
    sts_cmd = ["aws", "sts", "get-caller-identity", "--profile", args.profile, "--region", args.region]
    try:
        sts_result = _run(sts_cmd)
    except FileNotFoundError:
        print("ERROR: aws CLI not found; cannot validate AWS identity.", file=sys.stderr)
        return 1
    if sts_result.returncode != 0:
        print("ERROR: aws sts get-caller-identity failed; aborting before any AWS mutation.", file=sys.stderr)
        return 1

    print("\n=== Step 4: Pre-check - no FourCastNet endpoints / always-on compute ===")
    for check_path in (CHECK_NO_ENDPOINTS, CHECK_NO_ALWAYS_ON):
        check_cmd = [sys.executable, str(check_path), "--region", args.region, "--profile", args.profile]
        result = _run(check_cmd)
        if result.returncode != 0:
            print(f"ERROR: pre-check failed ({check_path.name}); aborting.", file=sys.stderr)
            return result.returncode

    if args.create_model:
        print("\n=== Step 5: Describe/create CPU POC model ===")
        create_cmd = [
            sys.executable,
            str(DESCRIBE_OR_CREATE_MODEL),
            "--config",
            str(config_path),
            "--region",
            args.region,
            "--profile",
            args.profile,
        ]
        result = _run(create_cmd)
        if result.returncode != 0:
            return result.returncode

    if args.upload_manifest:
        print("\n=== Step 6: Upload manifest ===")
        if args.input_contract != "jsonl_pointer":
            print(
                "raw_npy contract: --upload-manifest is a no-op "
                "(the .npy is assumed to already exist at --input-s3-uri)."
            )
        elif not args.manifest_s3_uri:
            print(
                "ERROR: --upload-manifest requires --manifest-s3-uri for the jsonl_pointer contract.",
                file=sys.stderr,
            )
            return 1
        else:
            assert manifest_path is not None
            cp_cmd = ["aws", "s3", "cp", str(manifest_path), args.manifest_s3_uri]
            print("Command:", " ".join(cp_cmd))
            try:
                import boto3
            except ModuleNotFoundError:
                print("ERROR: boto3 is required for --upload-manifest.", file=sys.stderr)
                return 1
            bucket, _, key = args.manifest_s3_uri[5:].partition("/")
            session_kwargs: dict[str, str] = {"region_name": args.region}
            if args.profile:
                session_kwargs["profile_name"] = args.profile
            s3 = boto3.Session(**session_kwargs).client("s3")
            s3.upload_file(str(manifest_path), bucket, key)
            print(f"Uploaded {manifest_path} -> {args.manifest_s3_uri}")

    if args.execute:
        print("\n=== Step 7: Create transform job (--execute) ===")
        exec_cmd = _build_batch_transform_cmd(
            args,
            config_path,
            transform_cfg,
            model_name,
            transform_input_s3_uri,
            output_s3_uri_for_transform,
            content_type,
            split_type,
            execute=True,
        )
        result = _run(exec_cmd)
        if result.returncode != 0:
            return result.returncode

        print("\n=== Step 8: Post-check - no FourCastNet endpoints ===")
        post_cmd = [sys.executable, str(CHECK_NO_ENDPOINTS), "--region", args.region, "--profile", args.profile]
        post = _run(post_cmd)
        if post.returncode != 0:
            return post.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
