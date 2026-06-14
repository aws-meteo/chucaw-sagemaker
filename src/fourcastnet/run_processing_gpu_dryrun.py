#!/usr/bin/env python3
"""Prepare FCN Processing payload locally and optionally execute the AWS CLI command."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


TAGS: list[dict[str, str]] = [
    {"Key": "Project", "Value": "SbnAI"},
    {"Key": "Component", "Value": "FourCastNet"},
    {"Key": "Environment", "Value": "dev"},
    {"Key": "Owner", "Value": "Fabian"},
    {"Key": "CostCenter", "Value": "chucaw"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FCN Processing GPU payload (local-only by default)")
    parser.add_argument("--profile", default="sbnai-725", help="AWS profile for generated command")
    parser.add_argument("--region", default="us-east-1", help="AWS region for generated command")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--processing-image-uri", required=True, help="Container image URI for processing")
    parser.add_argument("--model-assets-s3-uri", required=True, help="S3 URI prefix with FCN assets")
    parser.add_argument("--input-tensor-s3-uri", required=True, help="S3 URI prefix with input tensor")
    parser.add_argument("--code-s3-uri", required=True, help="S3 URI prefix with processing entrypoint")
    parser.add_argument("--output-s3-uri", required=True, help="S3 URI prefix for processing outputs")
    parser.add_argument("--instance-type", default="ml.g4dn.xlarge", help="Processing instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Processing instance count")
    parser.add_argument("--volume-size-gb", type=int, default=100, help="EBS volume size")
    parser.add_argument("--max-runtime-seconds", type=int, default=1800, help="Hard runtime cap")
    parser.add_argument("--job-name-prefix", default="sbnai-fourcastnet-fcn-smoke", help="Job name prefix")
    parser.add_argument(
        "--output-json",
        default="artifacts/fourcastnet/aws_payloads/processing_job.json",
        help="Output payload JSON path",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute create-processing-job via AWS CLI. Without this flag, local dry-run only.",
    )
    return parser.parse_args()


def required(value: str, name: str) -> str:
    checked = value.strip()
    if not checked:
        raise ValueError(f"Missing required value: {name}")
    return checked


def parse_s3_uri(uri: str, field_name: str) -> str:
    value = required(uri, field_name)
    if not value.startswith("s3://"):
        raise ValueError(f"{field_name} must start with s3://, got: {value}")
    without_scheme = value[len("s3://") :]
    bucket, _, _ = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"{field_name} is missing bucket: {value}")
    return value.rstrip("/")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.instance_count < 1:
        raise ValueError("--instance-count must be >= 1")
    if args.volume_size_gb < 30:
        raise ValueError("--volume-size-gb must be >= 30")
    if args.max_runtime_seconds < 60:
        raise ValueError("--max-runtime-seconds must be >= 60")

    # Fail clearly if required local model code for processing is absent.
    sm_root = Path(__file__).resolve().parents[2]
    processing_entrypoint = sm_root / "src" / "fourcastnet" / "processing_entrypoint.py"
    if not processing_entrypoint.exists():
        raise FileNotFoundError(f"Missing FourCastNet processing code: {processing_entrypoint}")

    role_arn = required(args.role_arn, "role_arn")
    image_uri = required(args.processing_image_uri, "processing_image_uri")
    model_assets_s3_uri = parse_s3_uri(args.model_assets_s3_uri, "model_assets_s3_uri")
    input_tensor_s3_uri = parse_s3_uri(args.input_tensor_s3_uri, "input_tensor_s3_uri")
    code_s3_uri = parse_s3_uri(args.code_s3_uri, "code_s3_uri")
    output_s3_uri = parse_s3_uri(args.output_s3_uri, "output_s3_uri")

    job_name = f"{args.job_name_prefix}-{int(time.time())}"
    return {
        "ProcessingJobName": job_name,
        "RoleArn": role_arn,
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": [
                "python",
                "/opt/ml/processing/input/code/processing_entrypoint.py",
                "--input-tensor",
                "/opt/ml/processing/input/tensor/input_tensor.npy",
                "--global-means",
                "/opt/ml/processing/input/assets/global_means.npy",
                "--global-stds",
                "/opt/ml/processing/input/assets/global_stds.npy",
                "--checkpoint",
                "/opt/ml/processing/input/assets/backbone.ckpt",
                "--output-report",
                "/opt/ml/processing/output/processing_smoke_report.json",
            ],
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceCount": args.instance_count,
                "InstanceType": args.instance_type,
                "VolumeSizeInGB": args.volume_size_gb,
            }
        },
        "ProcessingInputs": [
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri": code_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "tensor",
                "S3Input": {
                    "S3Uri": input_tensor_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/tensor",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "assets",
                "S3Input": {
                    "S3Uri": model_assets_s3_uri,
                    "LocalPath": "/opt/ml/processing/input/assets",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
        ],
        "ProcessingOutputConfig": {
            "Outputs": [
                {
                    "OutputName": "report",
                    "S3Output": {
                        "S3Uri": output_s3_uri,
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        "Tags": TAGS,
    }


def build_submit_command(payload_path: Path, profile: str, region: str) -> str:
    command_parts = [
        "aws",
        "sagemaker",
        "create-processing-job",
        "--cli-input-json",
        f"file://{payload_path.as_posix()}",
        "--profile",
        required(profile, "profile"),
        "--region",
        required(region, "region"),
    ]
    return " ".join(shlex.quote(part) for part in command_parts)


def main() -> int:
    args = parse_args()
    output_path = Path(args.output_json).resolve()
    payload = build_payload(args)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    submit_command = build_submit_command(output_path, args.profile.strip(), args.region.strip())
    print("=== Processing Job Payload ===")
    print(json.dumps(payload, indent=2))
    print(f"Payload written: {output_path}")
    print("=== Human-run AWS Command ===")
    print(submit_command)

    if not args.execute:
        print("Dry-run only. No AWS command executed. Re-run with --execute to submit.")
        return 0

    result = subprocess.run(submit_command, shell=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"AWS command failed with exit code {result.returncode}")
    print("Processing job submitted.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
