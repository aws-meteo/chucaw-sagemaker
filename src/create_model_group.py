#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from dotenv import load_dotenv


def required_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default or "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def is_not_found_model_package_group(exc: ClientError) -> bool:
    code = exc.response.get("Error", {}).get("Code", "")
    message = exc.response.get("Error", {}).get("Message", "")
    text = f"{code}: {message}".lower()
    return (
        code in {"ValidationException", "ResourceNotFound", "ResourceNotFoundException"}
        and "does not exist" in text
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()
    group_name = required_env("MODEL_PACKAGE_GROUP_NAME", "ForecastingModels")
    description = os.getenv(
        "MODEL_PACKAGE_GROUP_DESCRIPTION",
        "Registered forecasting models for SageMaker deployments",
    ).strip()

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    sm_client = session.client("sagemaker")

    print("Creating model package group...")
    if profile:
        print(f"Using profile: {profile}")
    print(f"Region: {region}")
    print(f"Group name: {group_name}")

    try:
        try:
            resp = sm_client.describe_model_package_group(
                ModelPackageGroupName=group_name
            )
            print("Model package group already exists.")
            print(resp["ModelPackageGroupArn"])
            return
        except ClientError as exc:
            if not is_not_found_model_package_group(exc):
                raise

        resp = sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=description,
            Tags=[
                {"Key": "Project", "Value": os.getenv("PROJECT_TAG", "chucaw")},
                {"Key": "Owner", "Value": os.getenv("OWNER_TAG", "unknown")},
                {"Key": "Environment", "Value": os.getenv("ENV_TAG", "dev")},
            ],
        )
        print("Model package group created successfully.")
        print(resp["ModelPackageGroupArn"])

    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to ensure model package group '{group_name}': {exc}") from exc


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
