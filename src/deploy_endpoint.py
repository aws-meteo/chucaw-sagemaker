#!/usr/bin/env python3
"""Deploy a SageMaker serverless endpoint while also registering the model.

Flow:
1. Ensure Model Package Group exists.
2. Register a model version (create_model_package).
3. Wait until the model version is Completed.
4. Create a SageMaker model from the registered model version ARN.
5. Create/update endpoint config and endpoint.

Environment variables:
  REGION                       required
  AWS_PROFILE                  optional
  ROLE_ARN                     required
  ENDPOINT_NAME                required
  SKLEARN_VERSION              required
  MODEL_S3_BUCKET              required
  MODEL_S3_PREFIX              required
  SERVING_SOURCE_S3_BUCKET     optional unless SAGEMAKER_SUBMIT_DIRECTORY not set
  SERVING_SOURCE_S3_PREFIX     optional unless SAGEMAKER_SUBMIT_DIRECTORY not set
  MODEL_PACKAGE_GROUP_NAME     required
  MODEL_APPROVAL_STATUS        optional, default Approved
  PROJECT_TAG                  required
  OWNER_TAG                    required
  ENV_TAG                      required
  MODEL_DESCRIPTION            optional
  MODEL_PACKAGE_DESCRIPTION    optional
  MEMORY_SIZE_MB               optional, default 2048
  MAX_CONCURRENCY              optional, default 5
  SUPPORTED_CONTENT_TYPES      optional, default application/json
  SUPPORTED_RESPONSE_TYPES     optional, default application/json

CLI overrides:
  --endpoint-name
  --model-s3-bucket
  --model-s3-prefix
  --serving-source-s3-bucket
  --serving-source-s3-prefix
  --sagemaker-program
  --sagemaker-submit-directory
  --delete-failed-endpoint
"""

import argparse
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
# from sagemaker.core import image_uris
from sagemaker import image_uris


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def optional_value(cli_value: str, env_name: str, default: str = "") -> str:
    return (cli_value or os.getenv(env_name, default)).strip()


def csv_env(name: str, default: str):
    raw = os.getenv(name, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def exists_endpoint(sm_client, endpoint_name: str) -> bool:
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except ClientError as exc:
        message = str(exc)
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ValidationException", "ResourceNotFound"} and "Could not find endpoint" in message:
            return False
        raise


def describe_endpoint_or_none(sm_client, endpoint_name: str):
    try:
        return sm_client.describe_endpoint(EndpointName=endpoint_name)
    except ClientError as exc:
        message = str(exc).lower()
        code = exc.response.get("Error", {}).get("Code", "")
        if code in {"ValidationException", "ResourceNotFound"} and (
            "could not find endpoint" in message or "not found" in message
        ):
            return None
        raise


def derive_submit_directory(
    explicit_submit_directory: str,
    serving_source_bucket: str,
    serving_source_prefix: str,
) -> str:
    if explicit_submit_directory:
        return explicit_submit_directory
    if not serving_source_bucket and not serving_source_prefix:
        return ""
    if not serving_source_bucket:
        raise ValueError(
            "SERVING_SOURCE_S3_BUCKET is required when SERVING_SOURCE_S3_PREFIX is set"
        )
    if not serving_source_prefix:
        raise ValueError(
            "SERVING_SOURCE_S3_PREFIX is required when SERVING_SOURCE_S3_BUCKET is set"
        )
    normalized_prefix = serving_source_prefix.strip("/")
    return f"s3://{serving_source_bucket}/{normalized_prefix}/source.tar.gz"


def maybe_delete_failed_endpoint(sm_client, endpoint_name: str, delete_failed_endpoint: bool) -> bool:
    endpoint_description = describe_endpoint_or_none(sm_client, endpoint_name)
    if endpoint_description is None:
        return False

    status = endpoint_description.get("EndpointStatus", "Unknown")
    if status == "Failed":
        if not delete_failed_endpoint:
            raise RuntimeError(
                "Endpoint exists with status Failed. Delete it manually, or rerun with "
                "--delete-failed-endpoint."
            )
        sm_client.delete_endpoint(EndpointName=endpoint_name)
        waiter = sm_client.get_waiter("endpoint_deleted")
        waiter.wait(EndpointName=endpoint_name)
        return False

    return True


def is_missing_model_package_group(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code", "")
    message = str(exc).lower()
    return (
        code in {"ValidationException", "ResourceNotFound", "ResourceNotFoundException"}
        and ("could not find" in message or "does not exist" in message)
    )

def ensure_model_package_group(sm_client, group_name: str, description: str) -> str:
    try:
        response = sm_client.describe_model_package_group(ModelPackageGroupName=group_name)
        return response["ModelPackageGroupArn"]
    except ClientError as exc:
        if not is_missing_model_package_group(exc):
            raise

    response = sm_client.create_model_package_group(
        ModelPackageGroupName=group_name,
        ModelPackageGroupDescription=description
    )
    return response["ModelPackageGroupArn"]


def wait_for_model_package(sm_client, model_package_arn: str, timeout_seconds: int = 1800, poll_seconds: int = 10):
    start = time.time()
    while True:
        response = sm_client.describe_model_package(ModelPackageName=model_package_arn)
        status = response["ModelPackageStatus"]
        if status == "Completed":
            return response
        if status == "Failed":
            reason = response.get("FailureReason", "unknown")
            raise RuntimeError(f"Model package failed: {reason}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for model package: {model_package_arn}")
        time.sleep(poll_seconds)


def wait_for_endpoint(sm_client, endpoint_name: str):
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    description = sm_client.describe_endpoint(EndpointName=endpoint_name)
    return description["EndpointStatus"]


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy/update SageMaker serverless sklearn endpoint")
    parser.add_argument("--endpoint-name", default="", help="Override ENDPOINT_NAME")
    parser.add_argument("--model-s3-bucket", default="", help="Override MODEL_S3_BUCKET")
    parser.add_argument("--model-s3-prefix", default="", help="Override MODEL_S3_PREFIX")
    parser.add_argument(
        "--serving-source-s3-bucket",
        default="",
        help="Override SERVING_SOURCE_S3_BUCKET",
    )
    parser.add_argument(
        "--serving-source-s3-prefix",
        default="",
        help="Override SERVING_SOURCE_S3_PREFIX",
    )
    parser.add_argument(
        "--sagemaker-program",
        default="",
        help="Override SAGEMAKER_PROGRAM (default/env: inference.py)",
    )
    parser.add_argument(
        "--sagemaker-submit-directory",
        default="",
        help="Optional explicit SAGEMAKER_SUBMIT_DIRECTORY, e.g. s3://bucket/prefix/source.tar.gz",
    )
    parser.add_argument(
        "--delete-failed-endpoint",
        action="store_true",
        help="If endpoint exists with status Failed, delete it before create/update",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(Path.cwd() / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()
    role_arn = required_env("ROLE_ARN")
    endpoint_name = optional_value(args.endpoint_name, "ENDPOINT_NAME")
    if not endpoint_name:
        raise ValueError("Missing required endpoint name: pass --endpoint-name or set ENDPOINT_NAME")
    sklearn_version = required_env("SKLEARN_VERSION")
    model_bucket = optional_value(args.model_s3_bucket, "MODEL_S3_BUCKET")
    if not model_bucket:
        raise ValueError("Missing required model bucket: pass --model-s3-bucket or set MODEL_S3_BUCKET")
    model_prefix = optional_value(args.model_s3_prefix, "MODEL_S3_PREFIX").strip("/")
    if not model_prefix:
        raise ValueError("Missing required model prefix: pass --model-s3-prefix or set MODEL_S3_PREFIX")
    serving_source_bucket = optional_value(
        args.serving_source_s3_bucket,
        "SERVING_SOURCE_S3_BUCKET",
    )
    serving_source_prefix = optional_value(
        args.serving_source_s3_prefix,
        "SERVING_SOURCE_S3_PREFIX",
    ).strip("/")
    sagemaker_program = optional_value(args.sagemaker_program, "SAGEMAKER_PROGRAM", "inference.py")
    explicit_submit_directory = optional_value(
        args.sagemaker_submit_directory,
        "SAGEMAKER_SUBMIT_DIRECTORY",
    )
    sagemaker_submit_directory = derive_submit_directory(
        explicit_submit_directory=explicit_submit_directory,
        serving_source_bucket=serving_source_bucket,
        serving_source_prefix=serving_source_prefix,
    )
    model_package_group_name = required_env("MODEL_PACKAGE_GROUP_NAME")
    model_approval_status = os.getenv("MODEL_APPROVAL_STATUS", "Approved").strip()
    model_description = os.getenv("MODEL_DESCRIPTION", "Forecast model deployed from registered model version").strip()
    model_package_description = os.getenv("MODEL_PACKAGE_DESCRIPTION", model_description).strip()
    group_description = os.getenv(
        "MODEL_PACKAGE_GROUP_DESCRIPTION",
        "Registered forecasting models for SageMaker deployments",
    ).strip()

    memory_size_mb = int(os.getenv("MEMORY_SIZE_MB", "2048"))
    max_concurrency = int(os.getenv("MAX_CONCURRENCY", "5"))
    supported_content_types = csv_env("SUPPORTED_CONTENT_TYPES", "application/json")
    supported_response_types = csv_env("SUPPORTED_RESPONSE_TYPES", "application/json")

    project_tag = required_env("PROJECT_TAG")
    owner_tag = required_env("OWNER_TAG")
    env_tag = required_env("ENV_TAG")

    model_data_url = f"s3://{model_bucket}/{model_prefix}/model.tar.gz"
    timestamp = int(time.time())
    model_name = f"{endpoint_name}-model-{timestamp}"
    endpoint_config_name = f"{endpoint_name}-cfg-{timestamp}"

    tags = [
        {"Key": "Project", "Value": project_tag},
        {"Key": "Owner", "Value": owner_tag},
        {"Key": "Environment", "Value": env_tag},
    ]

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    sm_client = session.client("sagemaker")

    image_uri = image_uris.retrieve(
        framework="sklearn",
        region=region,
        version=sklearn_version,
        py_version="py3",
        image_scope="inference",
        instance_type="ml.m5.large",
    )

    container_environment = {"SAGEMAKER_PROGRAM": sagemaker_program}
    if sagemaker_submit_directory:
        container_environment["SAGEMAKER_SUBMIT_DIRECTORY"] = sagemaker_submit_directory

    try:
        group_arn = ensure_model_package_group(
            sm_client,
            group_name=model_package_group_name,
            description=group_description,
        )

        model_package_response = sm_client.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription=model_package_description,
            ModelApprovalStatus=model_approval_status,
            InferenceSpecification={
                "Containers": [
                    {
                        "Image": image_uri,
                        "ModelDataUrl": model_data_url,
                        "Environment": container_environment,
                    }
                ],
                "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.large"],
                "SupportedTransformInstanceTypes": ["ml.m5.large"],
                "SupportedContentTypes": supported_content_types,
                "SupportedResponseMIMETypes": supported_response_types,
            },
        )
        model_package_arn = model_package_response["ModelPackageArn"]
        wait_for_model_package(sm_client, model_package_arn)

        sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            Containers=[{"ModelPackageName": model_package_arn}],
            Tags=tags,
        )

        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": memory_size_mb,
                        "MaxConcurrency": max_concurrency,
                    },
                }
            ],
            Tags=tags,
        )

        endpoint_exists = maybe_delete_failed_endpoint(
            sm_client,
            endpoint_name=endpoint_name,
            delete_failed_endpoint=args.delete_failed_endpoint,
        )
        if endpoint_exists and exists_endpoint(sm_client, endpoint_name):
            sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
            action = "updated"
        else:
            sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=tags,
            )
            action = "created"

        status = wait_for_endpoint(sm_client, endpoint_name)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            "SageMaker deployment with registry failed. "
            f"endpoint={endpoint_name}, model={model_name}, config={endpoint_config_name}, "
            f"group={model_package_group_name}. Error: {exc}"
        ) from exc

    print(f"Model package group: {model_package_group_name}")
    print(f"Model package group ARN: {group_arn}")
    print(f"Registered model version ARN: {model_package_arn}")
    print(f"Model data URL: {model_data_url}")
    print(f"Container environment: {container_environment}")
    print(f"Endpoint name: {endpoint_name}")
    print(f"Endpoint action: {action}")
    print(f"Endpoint status: {status}")
    print("Deployment mode: SageMaker Serverless Inference")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
