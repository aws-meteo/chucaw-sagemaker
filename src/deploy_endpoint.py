#!/usr/bin/env python3
import os
import sys
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from sagemaker import image_uris
from pathlib import Path


def required_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def exists_endpoint(sm_client, endpoint_name):
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True
    except sm_client.exceptions.ClientError as exc:
        if "Could not find endpoint" in str(exc):
            return False
        raise


def wait_for_endpoint(sm_client, endpoint_name):
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    description = sm_client.describe_endpoint(EndpointName=endpoint_name)
    return description["EndpointStatus"]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()
    role_arn = required_env("ROLE_ARN")
    endpoint_name = required_env("ENDPOINT_NAME")
    sklearn_version = required_env("SKLEARN_VERSION")
    model_bucket = required_env("MODEL_S3_BUCKET")
    model_prefix = required_env("MODEL_S3_PREFIX").strip("/")
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

    try:
        sm_client.create_model(
            ModelName=model_name,
            ExecutionRoleArn=role_arn,
            PrimaryContainer={
                "Image": image_uri,
                "ModelDataUrl": model_data_url,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                },
            },
            Tags=tags,
        )

        sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": 2048,
                        "MaxConcurrency": 5,
                    },
                }
            ],
            Tags=tags,
        )

        if exists_endpoint(sm_client, endpoint_name):
            sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
            )
            print(f"Updating existing endpoint: {endpoint_name}")
        else:
            sm_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name,
                Tags=tags,
            )
            print(f"Creating new endpoint: {endpoint_name}")

        status = wait_for_endpoint(sm_client, endpoint_name)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            "SageMaker deployment failed. "
            f"endpoint={endpoint_name}, model={model_name}, config={endpoint_config_name}. "
            f"Error: {exc}"
        ) from exc

    print(f"Endpoint name: {endpoint_name}")
    print(f"Endpoint status: {status}")
    print("Deployment mode: SageMaker Serverless Inference")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
