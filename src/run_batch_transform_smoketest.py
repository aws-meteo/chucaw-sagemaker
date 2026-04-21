#!/usr/bin/env python3
"""
Run a minimal SageMaker Batch Transform job using the same model artifact pattern
already used for endpoint deployment.

Design goals
------------
- Single file
- Reproducible from the repo
- Minimal required configuration
- Safe for a tiny one-file smoke test
- Clear logs and explicit resource names

Recommended first test
----------------------
Use a single JSON input object in S3 that matches the same request body that
already works with your endpoint invocation. Keep the file tiny.

Example input file content:
{"latitude": -33.5, "longitude": -70.5, "date": "2026-04-14T06:00:00Z"}

Environment variables
---------------------
Required:
- REGION
- ROLE_ARN
- SKLEARN_VERSION
- MODEL_S3_BUCKET
- MODEL_S3_PREFIX
- BATCH_INPUT_S3_URI
- BATCH_OUTPUT_S3_URI

Optional:
- AWS_PROFILE
- BATCH_INSTANCE_TYPE=ml.m5.large
- BATCH_INSTANCE_COUNT=1
- BATCH_MAX_CONCURRENT=1
- BATCH_MAX_PAYLOAD_MB=6
- BATCH_CONTENT_TYPE=application/json
- BATCH_ACCEPT=application/json
- BATCH_SPLIT_TYPE=None
- PROJECT_TAG
- OWNER_TAG
- ENV_TAG
"""

import os
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from sagemaker.core import image_uris


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    return int(raw)


def wait_for_transform_job(sm_client, job_name: str, poll_seconds: int = 10, timeout_seconds: int = 3600) -> dict:
    start = time.time()
    while True:
        desc = sm_client.describe_transform_job(TransformJobName=job_name)
        status = desc["TransformJobStatus"]
        if status in {"Completed", "Failed", "Stopped"}:
            return desc
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Timed out waiting for transform job: {job_name}")
        time.sleep(poll_seconds)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")
    load_dotenv(Path.cwd() / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()
    role_arn = required_env("ROLE_ARN")
    sklearn_version = required_env("SKLEARN_VERSION")
    model_bucket = required_env("MODEL_S3_BUCKET")
    model_prefix = required_env("MODEL_S3_PREFIX").strip("/")

    input_s3_uri = required_env("BATCH_INPUT_S3_URI")
    output_s3_uri = required_env("BATCH_OUTPUT_S3_URI")

    batch_instance_type = os.getenv("BATCH_INSTANCE_TYPE", "ml.m5.large").strip()
    batch_instance_count = env_int("BATCH_INSTANCE_COUNT", 1)
    batch_max_concurrent = env_int("BATCH_MAX_CONCURRENT", 1)
    batch_max_payload_mb = env_int("BATCH_MAX_PAYLOAD_MB", 6)
    batch_content_type = os.getenv("BATCH_CONTENT_TYPE", "application/json").strip()
    batch_accept = os.getenv("BATCH_ACCEPT", "application/json").strip()
    batch_split_type = os.getenv("BATCH_SPLIT_TYPE", "None").strip()

    project_tag = os.getenv("PROJECT_TAG", "chucaw").strip()
    owner_tag = os.getenv("OWNER_TAG", "unknown").strip()
    env_tag = os.getenv("ENV_TAG", "dev").strip()

    model_data_url = f"s3://{model_bucket}/{model_prefix}/model.tar.gz"
    timestamp = int(time.time())
    model_name = f"batch-smoketest-model-{timestamp}"
    transform_job_name = f"batch-smoketest-job-{timestamp}"

    tags = [
        {"Key": "Project", "Value": project_tag},
        {"Key": "Owner", "Value": owner_tag},
        {"Key": "Environment", "Value": env_tag},
        {"Key": "Purpose", "Value": "batch-smoke-test"},
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
        instance_type=batch_instance_type,
    )

    print("=== SageMaker Batch Transform smoke test ===")
    print(f"Region:              {region}")
    if profile:
        print(f"AWS profile:         {profile}")
    print(f"Model artifact:      {model_data_url}")
    print(f"Batch input:         {input_s3_uri}")
    print(f"Batch output:        {output_s3_uri}")
    print(f"Instance type:       {batch_instance_type}")
    print(f"Model name:          {model_name}")
    print(f"Transform job name:  {transform_job_name}")

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

        transform_input = {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": input_s3_uri,
                }
            },
            "ContentType": batch_content_type,
        }

        batch_strategy = "SingleRecord"
        if batch_split_type != "None":
            transform_input["SplitType"] = batch_split_type
            batch_strategy = "MultiRecord"

        sm_client.create_transform_job(
            TransformJobName=transform_job_name,
            ModelName=model_name,
            MaxConcurrentTransforms=batch_max_concurrent,
            MaxPayloadInMB=batch_max_payload_mb,
            BatchStrategy=batch_strategy,
            TransformInput=transform_input,
            TransformOutput={
                "S3OutputPath": output_s3_uri,
                "Accept": batch_accept,
                "AssembleWith": "Line",
            },
            TransformResources={
                "InstanceType": batch_instance_type,
                "InstanceCount": batch_instance_count,
            },
            Tags=tags,
        )

        desc = wait_for_transform_job(sm_client, transform_job_name)

    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            "Batch transform smoke test failed. "
            f"model={model_name}, job={transform_job_name}. Error: {exc}"
        ) from exc

    status = desc["TransformJobStatus"]
    print(f"Final status:        {status}")
    print(f"Creation time:       {desc.get('CreationTime')}")
    print(f"Output path:         {output_s3_uri}")
    if status != "Completed":
        print(f"Failure reason:      {desc.get('FailureReason', 'n/a')}")
        raise SystemExit(1)

    print("Batch transform completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
