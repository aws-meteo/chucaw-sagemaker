#!/usr/bin/env python3
"""Prepare a SageMaker create-processing-job payload for FourCastNet smoke runs."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_JSON = "artifacts/fourcastnet/aws_payloads/processing_job.json"
DEFAULT_PROFILE = "sbnai-725"
DEFAULT_REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "ml.g4dn.xlarge"
DEFAULT_VOLUME_GB = 100
DEFAULT_JOB_PREFIX = "sbnai-fourcastnet-fcn-smoke"

TAGS: list[dict[str, str]] = [
    {"Key": "Project", "Value": "SbnAI"},
    {"Key": "Component", "Value": "FourCastNet"},
    {"Key": "Environment", "Value": "dev"},
    {"Key": "Owner", "Value": "Fabian"},
    {"Key": "CostCenter", "Value": "chucaw"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local payload JSON for create-processing-job")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="AWS profile for generated command")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region for generated command")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--processing-image-uri", required=True, help="ECR image URI for processing container")
    parser.add_argument("--code-s3-uri", required=True, help="S3 URI prefix for processing code")
    parser.add_argument("--input-tensor-s3-uri", required=True, help="S3 URI prefix for input tensor")
    parser.add_argument("--model-assets-s3-uri", required=True, help="S3 URI prefix for FCN model assets")
    parser.add_argument("--output-s3-uri", required=True, help="S3 URI prefix for output artifacts")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE, help="Processing instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Processing instance count")
    parser.add_argument("--volume-size-gb", type=int, default=DEFAULT_VOLUME_GB, help="EBS volume size")
    parser.add_argument("--max-runtime-seconds", type=int, default=1800, help="Max runtime")
    parser.add_argument("--job-name-prefix", default=DEFAULT_JOB_PREFIX, help="Processing job name prefix")
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON, help="Payload JSON output path")
    return parser.parse_args()


def required(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Missing required value: {name}")
    return cleaned


def validate_s3_uri(value: str, name: str) -> str:
    uri = required(value, name)
    if not uri.startswith("s3://"):
        raise ValueError(f"{name} must start with s3://, got: {uri}")
    without_scheme = uri[len("s3://") :]
    bucket, _, _ = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"{name} is missing bucket: {uri}")
    return uri.rstrip("/")


def ensure_local_fourcastnet_code_exists() -> None:
    sm_root = Path(__file__).resolve().parents[2]
    required_files = [
        sm_root / "src" / "fourcastnet" / "processing_entrypoint.py",
        sm_root / "src" / "fourcastnet" / "serving" / "inference.py",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing FourCastNet model code files: {missing}")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    ensure_local_fourcastnet_code_exists()
    if args.instance_count < 1:
        raise ValueError("--instance-count must be >= 1")
    if args.volume_size_gb < 30:
        raise ValueError("--volume-size-gb must be >= 30")
    if args.max_runtime_seconds < 60:
        raise ValueError("--max-runtime-seconds must be >= 60")

    job_name = f"{args.job_name_prefix}-{int(time.time())}"
    return {
        "ProcessingJobName": job_name,
        "RoleArn": required(args.role_arn, "role_arn"),
        "AppSpecification": {
            "ImageUri": required(args.processing_image_uri, "processing_image_uri"),
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
                    "S3Uri": validate_s3_uri(args.code_s3_uri, "code_s3_uri"),
                    "LocalPath": "/opt/ml/processing/input/code",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "tensor",
                "S3Input": {
                    "S3Uri": validate_s3_uri(args.input_tensor_s3_uri, "input_tensor_s3_uri"),
                    "LocalPath": "/opt/ml/processing/input/tensor",
                    "S3DataType": "S3Prefix",
                    "S3InputMode": "File",
                },
            },
            {
                "InputName": "assets",
                "S3Input": {
                    "S3Uri": validate_s3_uri(args.model_assets_s3_uri, "model_assets_s3_uri"),
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
                        "S3Uri": validate_s3_uri(args.output_s3_uri, "output_s3_uri"),
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": args.max_runtime_seconds},
        "Tags": TAGS,
    }


def build_command(payload_path: Path, profile: str, region: str) -> str:
    parts = [
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
    return " ".join(shlex.quote(part) for part in parts)


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    command = build_command(output_path, args.profile, args.region)
    print("=== Processing Job Payload ===")
    print(json.dumps(payload, indent=2))
    print(f"Payload written: {output_path}")
    print("=== Human-run AWS Command ===")
    print(command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
