#!/usr/bin/env python3
"""Prepare local SageMaker Model Registry payload JSON files for FourCastNet."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


MODEL_PACKAGE_GROUP_NAME = "sbnai-fourcastnet-fcn-v0"
DEFAULT_PROFILE = "sbnai-725"
DEFAULT_REGION = "us-east-1"
DEFAULT_OUTPUT_DIR = "artifacts/fourcastnet/aws_payloads"

TAGS: list[dict[str, str]] = [
    {"Key": "Project", "Value": "SbnAI"},
    {"Key": "Component", "Value": "FourCastNet"},
    {"Key": "Environment", "Value": "dev"},
    {"Key": "Owner", "Value": "Fabian"},
    {"Key": "CostCenter", "Value": "chucaw"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local payloads for SageMaker Model Registry")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="AWS profile for generated commands")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region for generated commands")
    parser.add_argument(
        "--model-package-group-name",
        default=MODEL_PACKAGE_GROUP_NAME,
        help="Model Package Group name",
    )
    parser.add_argument("--model-artifact-s3-uri", required=True, help="S3 URI to model.tar.gz")
    parser.add_argument("--inference-image-uri", required=True, help="Inference image URI")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for payload files")
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
    return uri


def ensure_local_model_code_exists() -> None:
    sm_root = Path(__file__).resolve().parents[2]
    inference_path = sm_root / "src" / "fourcastnet" / "serving" / "inference.py"
    requirements_path = sm_root / "src" / "fourcastnet" / "serving" / "requirements.txt"
    missing = [str(path) for path in (inference_path, requirements_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing FourCastNet model code files: {missing}")


def build_group_payload(group_name: str) -> dict[str, Any]:
    return {
        "ModelPackageGroupName": group_name,
        "ModelPackageGroupDescription": "FourCastNet FCN v0 package group for wave-1 smoke and promotion",
        "Tags": TAGS,
    }


def build_package_payload(group_name: str, model_artifact_s3_uri: str, inference_image_uri: str) -> dict[str, Any]:
    return {
        "ModelPackageGroupName": group_name,
        "ModelPackageDescription": "FourCastNet FCN wave-1 smoke package",
        "ModelApprovalStatus": "PendingManualApproval",
        "InferenceSpecification": {
            "Containers": [
                {
                    "Image": inference_image_uri,
                    "ModelDataUrl": model_artifact_s3_uri,
                }
            ],
            "SupportedContentTypes": ["application/json", "application/x-npy"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.g4dn.xlarge", "ml.g5.xlarge"],
            "SupportedTransformInstanceTypes": ["ml.g4dn.xlarge", "ml.g5.xlarge"],
        },
        "CustomerMetadataProperties": {
            "Project": "SbnAI",
            "Component": "FourCastNet",
            "Interface": "tensor_nchw_20ch",
            "Wave": "wave1",
            "EndpointRequired": "false",
        },
    }


def build_command(subcommand: str, payload_path: Path, profile: str, region: str) -> str:
    parts = [
        "aws",
        "sagemaker",
        subcommand,
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
    ensure_local_model_code_exists()

    group_name = required(args.model_package_group_name, "model_package_group_name")
    model_artifact_s3_uri = validate_s3_uri(args.model_artifact_s3_uri, "model_artifact_s3_uri")
    inference_image_uri = required(args.inference_image_uri, "inference_image_uri")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    group_path = output_dir / "model_package_group.json"
    package_path = output_dir / "model_package.json"

    group_payload = build_group_payload(group_name)
    package_payload = build_package_payload(group_name, model_artifact_s3_uri, inference_image_uri)
    group_path.write_text(json.dumps(group_payload, indent=2), encoding="utf-8")
    package_path.write_text(json.dumps(package_payload, indent=2), encoding="utf-8")

    group_cmd = build_command("create-model-package-group", group_path, args.profile, args.region)
    package_cmd = build_command("create-model-package", package_path, args.profile, args.region)

    print("=== Model Package Group Payload ===")
    print(json.dumps(group_payload, indent=2))
    print(f"Payload written: {group_path}")
    print("=== Model Package Payload ===")
    print(json.dumps(package_payload, indent=2))
    print(f"Payload written: {package_path}")
    print("=== Human-run AWS Commands ===")
    print(group_cmd)
    print(package_cmd)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
