#!/usr/bin/env python3
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


def required_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def main():
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()
    bucket = required_env("MODEL_S3_BUCKET")
    prefix = required_env("MODEL_S3_PREFIX").strip("/")

    tar_path = repo_root / "model.tar.gz"
    if not tar_path.exists():
        raise FileNotFoundError(f"model.tar.gz not found: {tar_path}")

    key = f"{prefix}/model.tar.gz"
    s3_uri = f"s3://{bucket}/{key}"

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    try:
        s3.upload_file(str(tar_path), bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload model.tar.gz to {s3_uri}: {exc}") from exc

    print(f"Uploaded model artifact to: {s3_uri}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
