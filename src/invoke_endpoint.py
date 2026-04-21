#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
import argparse


REQUIRED_RESPONSE_KEYS = {"t2m", "lat_grid", "lon_grid", "units", "source"}


def required_env(name):
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value

def main():
    parser = argparse.ArgumentParser(description="Invoke SageMaker endpoint for temperature data")
    parser.add_argument("--lat", type=float, default=-33.5, help="Latitude (default: -33.5)")
    parser.add_argument("--lon", type=float, default=-70.6, help="Longitude (default: -70.6)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    region = required_env("REGION")
    profile = os.getenv("AWS_PROFILE", "").strip()
    endpoint_name = required_env("ENDPOINT_NAME")

    payload = {"lat": args.lat, "lon": args.lon}
    body = json.dumps(payload)

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    runtime = session.client("sagemaker-runtime")

    try:
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=body,
        )
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Endpoint invocation failed for endpoint={endpoint_name}: {exc}") from exc

    raw = response["Body"].read().decode("utf-8")
    print(f"Raw response: {raw}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invocation returned non-JSON response: {exc}") from exc

    missing = sorted(REQUIRED_RESPONSE_KEYS - set(parsed.keys()))
    if missing:
        raise ValueError(f"Invocation response missing required keys: {', '.join(missing)}")

    print(f"Parsed response: {json.dumps(parsed, indent=2)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
