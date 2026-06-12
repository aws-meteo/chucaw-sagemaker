#!/usr/bin/env python3
"""REALTIME_ENDPOINT_COST_RISK: generate payloads for an experimental async endpoint path."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "sbnai-fourcastnet-fcn-v0-async-model"
DEFAULT_ENDPOINT_CONFIG_NAME = "sbnai-fourcastnet-fcn-v0-async-config"
DEFAULT_MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1"
DEFAULT_ROLE_ARN = "arn:aws:iam::725644097028:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
DEFAULT_VARIANT_NAME = "AllTraffic"
DEFAULT_INITIAL_INSTANCE_COUNT = 1
DEFAULT_OUTPUT_S3_URI = "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-output/"
DEFAULT_MAX_CONCURRENT = 1
DEFAULT_PROFILE = "sbnai-725"
DEFAULT_REGION = "us-east-1"
DEFAULT_OUTPUT_DIR = Path("artifacts/fourcastnet/aws_payloads")

TAGS = [
    {"Key": "Project", "Value": "SbnAI"},
    {"Key": "Component", "Value": "FourCastNet"},
    {"Key": "Environment", "Value": "dev"},
    {"Key": "Owner", "Value": "Fabian"},
    {"Key": "CostCenter", "Value": "chucaw"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "REALTIME_ENDPOINT_COST_RISK. Experimental only. "
            "Default FourCastNet flow must use Batch Transform."
        )
    )
    parser.add_argument("--allow-realtime-endpoint", action="store_true")
    parser.add_argument("--i-understand-this-can-cost-money", action="store_true")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--endpoint-config-name", default=DEFAULT_ENDPOINT_CONFIG_NAME)
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-package-arn", default=DEFAULT_MODEL_PACKAGE_ARN)
    parser.add_argument("--role-arn", default=DEFAULT_ROLE_ARN)
    parser.add_argument("--instance-type", required=True)
    parser.add_argument("--variant-name", default=DEFAULT_VARIANT_NAME)
    parser.add_argument("--initial-instance-count", type=int, default=DEFAULT_INITIAL_INSTANCE_COUNT)
    parser.add_argument("--output-s3-uri", default=DEFAULT_OUTPUT_S3_URI)
    parser.add_argument("--max-concurrent-invocations-per-instance", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def required(value: str, name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"Missing required value: {name}")
    return cleaned


def validate_s3_uri(value: str, name: str) -> str:
    uri = required(value, name)
    if not uri.startswith("s3://"):
        raise ValueError(f"{name} must start with s3://, got {uri}")
    remainder = uri[len("s3://") :]
    bucket, sep, key = remainder.partition("/")
    if not bucket or not sep or not key:
        raise ValueError(f"Invalid {name}: {uri}")
    return uri


def build_create_model_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ModelName": required(args.model_name, "model_name"),
        "ExecutionRoleArn": required(args.role_arn, "role_arn"),
        "PrimaryContainer": {
            "ModelPackageName": required(args.model_package_arn, "model_package_arn"),
            "Environment": {
                "SAGEMAKER_PROGRAM": "inference.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
            },
        },
        "Tags": TAGS,
    }


def build_endpoint_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "EndpointConfigName": required(args.endpoint_config_name, "endpoint_config_name"),
        "ProductionVariants": [
            {
                "VariantName": required(args.variant_name, "variant_name"),
                "ModelName": required(args.model_name, "model_name"),
                "InitialInstanceCount": int(args.initial_instance_count),
                "InstanceType": required(args.instance_type, "instance_type"),
                "InitialVariantWeight": 1.0,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,
            }
        ],
        "AsyncInferenceConfig": {
            "OutputConfig": {
                "S3OutputPath": validate_s3_uri(args.output_s3_uri, "output_s3_uri"),
            },
            "ClientConfig": {
                "MaxConcurrentInvocationsPerInstance": int(args.max_concurrent_invocations_per_instance)
            },
        },
    }


def q(value: str) -> str:
    return shlex.quote(value)


def build_commands(args: argparse.Namespace, model_payload: Path, endpoint_config_payload: Path) -> list[str]:
    profile = required(args.profile, "profile")
    region = required(args.region, "region")
    model_name = required(args.model_name, "model_name")
    endpoint_name = required(args.endpoint_name, "endpoint_name")
    endpoint_config_name = required(args.endpoint_config_name, "endpoint_config_name")

    tag_args = " ".join([f"Key={q(tag['Key'])},Value={q(tag['Value'])}" for tag in TAGS])

    create_model = (
        f"aws sagemaker create-model --cli-input-json file://{model_payload.as_posix()} "
        f"--profile {q(profile)} --region {q(region)}"
    )
    create_config = (
        f"aws sagemaker create-endpoint-config --cli-input-json file://{endpoint_config_payload.as_posix()} "
        f"--profile {q(profile)} --region {q(region)}"
    )
    create_endpoint = (
        f"aws sagemaker create-endpoint --endpoint-name {q(endpoint_name)} "
        f"--endpoint-config-name {q(endpoint_config_name)} --tags {tag_args} "
        f"--profile {q(profile)} --region {q(region)}"
    )
    describe_endpoint = (
        f"aws sagemaker describe-endpoint --endpoint-name {q(endpoint_name)} "
        f"--profile {q(profile)} --region {q(region)}"
    )
    delete_endpoint = (
        f"aws sagemaker delete-endpoint --endpoint-name {q(endpoint_name)} "
        f"--profile {q(profile)} --region {q(region)}"
    )
    delete_endpoint_config = (
        f"aws sagemaker delete-endpoint-config --endpoint-config-name {q(endpoint_config_name)} "
        f"--profile {q(profile)} --region {q(region)}"
    )
    return [create_model, create_config, create_endpoint, describe_endpoint, delete_endpoint, delete_endpoint_config]


def main() -> int:
    args = parse_args()
    if not args.allow_realtime_endpoint or not args.i_understand_this_can_cost_money:
        raise ValueError(
            "Refusing to prepare real-time endpoint payloads without both flags: "
            "--allow-realtime-endpoint and --i-understand-this-can-cost-money"
        )

    if args.initial_instance_count < 1:
        raise ValueError("initial-instance-count must be >= 1")
    if args.max_concurrent_invocations_per_instance < 1:
        raise ValueError("max-concurrent-invocations-per-instance must be >= 1")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_payload = output_dir / "async_create_model.json"
    endpoint_config_payload = output_dir / "async_endpoint_config.json"

    create_model_payload = build_create_model_payload(args)
    create_endpoint_config_payload = build_endpoint_config_payload(args)

    model_payload.write_text(json.dumps(create_model_payload, indent=2), encoding="utf-8")
    endpoint_config_payload.write_text(json.dumps(create_endpoint_config_payload, indent=2), encoding="utf-8")

    print(f"Payload written: {model_payload}")
    print(f"Payload written: {endpoint_config_payload}")
    print(
        "REALTIME_ENDPOINT_COST_RISK: create-endpoint may start 1 instance immediately because InitialInstanceCount=1; "
        "register async autoscaling MinCapacity=0 right after endpoint becomes InService."
    )
    print("=== Human-run AWS commands ===")
    for cmd in build_commands(args, model_payload, endpoint_config_payload):
        print(cmd)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
