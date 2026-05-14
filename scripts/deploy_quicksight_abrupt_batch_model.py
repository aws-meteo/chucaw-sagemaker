#!/usr/bin/env python3
import argparse
import sys
from typing import Dict

import boto3
from botocore.exceptions import ClientError
from sagemaker import image_uris


DEFAULT_MODEL_NAME = "chucaw-quicksight-abrupt-classifier-v0"
DEFAULT_REGION = "us-east-1"
DEFAULT_SKLEARN_VERSION = "1.2-1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or update SageMaker model for QuickSight abrupt CSV batch transform."
    )
    parser.add_argument("--model-artifact-s3-uri", required=True, help="S3 URI to model tar.gz")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="SageMaker model name")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument(
        "--sklearn-version",
        default=DEFAULT_SKLEARN_VERSION,
        help="Sklearn inference container version",
    )
    return parser.parse_args()


def _is_model_not_found(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code", "")
    message = str(exc).lower()
    return code in {"ValidationException", "ResourceNotFound"} and "could not find model" in message


def _base_tags() -> list[Dict[str, str]]:
    return [
        {"Key": "Project", "Value": "chucaw"},
        {"Key": "Component", "Value": "quicksight-sagemaker-augmentation"},
        {"Key": "Environment", "Value": "dev"},
        {"Key": "Owner", "Value": "fabian"},
    ]


def _build_container(region: str, sklearn_version: str, model_artifact_s3_uri: str) -> Dict[str, str]:
    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version=sklearn_version,
        py_version="py3",
        image_scope="inference",
        instance_type="ml.m5.large",
    )
    return {
        "Image": image_uri,
        "ModelDataUrl": model_artifact_s3_uri,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
        },
    }


def main() -> int:
    args = _parse_args()
    sm_client = boto3.client("sagemaker", region_name=args.region)
    container = _build_container(args.region, args.sklearn_version, args.model_artifact_s3_uri)
    tags = _base_tags()

    action = "created"
    try:
        sm_client.describe_model(ModelName=args.model_name)
        sm_client.delete_model(ModelName=args.model_name)
        action = "updated"
    except ClientError as exc:
        if not _is_model_not_found(exc):
            raise

    sm_client.create_model(
        ModelName=args.model_name,
        ExecutionRoleArn=args.role_arn,
        PrimaryContainer=container,
        Tags=tags,
    )

    print(f"Model action: {action}")
    print(f"Model name: {args.model_name}")
    print(f"Region: {args.region}")
    print(f"Model artifact: {args.model_artifact_s3_uri}")
    print("Endpoint action: none (Batch Transform only)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
