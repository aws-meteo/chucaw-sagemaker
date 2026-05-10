#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Upload training CSV to S3")
    parser.add_argument("--input", required=True, help="Local training CSV path")
    parser.add_argument("--dataset-id", required=True, help="Dataset identifier for S3 layout")
    parser.add_argument("--bucket", default="", help="Override S3 bucket")
    parser.add_argument("--prefix", default="", help="Override S3 prefix")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()

    bucket = args.bucket or required_env("MODEL_S3_BUCKET")
    base_prefix = args.prefix or required_env("TRAINING_DATA_S3_PREFIX")
    base_prefix = base_prefix.strip("/")

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {input_path}")

    key = f"{base_prefix}/{args.dataset_id}/train.csv"
    s3_uri = f"s3://{bucket}/{key}"

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile

    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    try:
        s3.upload_file(str(input_path), bucket, key)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Failed to upload training data to {s3_uri}: {exc}") from exc

    print(f"Training data uploaded to: {s3_uri}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)