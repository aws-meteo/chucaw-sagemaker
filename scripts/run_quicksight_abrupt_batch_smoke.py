#!/usr/bin/env python3
import argparse
import datetime as dt
import sys
import time

import boto3


DEFAULT_REGION = "us-east-1"
DEFAULT_BUCKET = "chucaw-data-platinum-processed-725644097028-us-east-1-an"
DEFAULT_MODEL_NAME = "chucaw-quicksight-abrupt-classifier-v0"
DEFAULT_INPUT_KEY = "sagemaker/quicksight-abrupt/smoke/input/input.csv"
DEFAULT_OUTPUT_PREFIX = "sagemaker/quicksight-abrupt/smoke/output/"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QuickSight abrupt CSV SageMaker Batch Transform smoke test.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="SageMaker model name")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket")
    parser.add_argument(
        "--input-csv",
        default="examples/quicksight_abrupt_input.csv",
        help="Local CSV used for smoke input upload",
    )
    parser.add_argument("--wait", action="store_true", help="Wait until transform job is terminal")
    return parser.parse_args()


def _wait_for_terminal(sm_client, job_name: str) -> str:
    while True:
        desc = sm_client.describe_transform_job(TransformJobName=job_name)
        status = desc["TransformJobStatus"]
        if status in {"Completed", "Failed", "Stopped"}:
            return status
        time.sleep(10)


def main() -> int:
    args = _parse_args()
    session = boto3.Session(region_name=args.region)
    s3_client = session.client("s3")
    sm_client = session.client("sagemaker")

    input_s3_uri = f"s3://{args.bucket}/{DEFAULT_INPUT_KEY}"
    output_s3_uri = f"s3://{args.bucket}/{DEFAULT_OUTPUT_PREFIX}"

    with open(args.input_csv, "rb") as fh:
        s3_client.upload_fileobj(fh, args.bucket, DEFAULT_INPUT_KEY)

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    transform_job_name = f"chucaw-quicksight-abrupt-smoke-{timestamp}"

    sm_client.create_transform_job(
        TransformJobName=transform_job_name,
        ModelName=args.model_name,
        TransformInput={
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": input_s3_uri}},
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        TransformOutput={
            "S3OutputPath": output_s3_uri,
            "Accept": "text/csv",
            "AssembleWith": "Line",
        },
        TransformResources={"InstanceType": "ml.m5.large", "InstanceCount": 1},
        Tags=[
            {"Key": "Project", "Value": "chucaw"},
            {"Key": "Component", "Value": "quicksight-sagemaker-augmentation"},
            {"Key": "Environment", "Value": "dev"},
            {"Key": "Owner", "Value": "fabian"},
        ],
    )

    print(f"Transform job name: {transform_job_name}")
    print(f"Input uploaded: {input_s3_uri}")
    print(f"Output path: {output_s3_uri}")

    if args.wait:
        final_status = _wait_for_terminal(sm_client, transform_job_name)
        print(f"Final status: {final_status}")
        if final_status != "Completed":
            return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
