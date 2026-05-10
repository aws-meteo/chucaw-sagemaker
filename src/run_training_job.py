#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import boto3
import sagemaker
from dotenv import load_dotenv
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Run SageMaker scikit-learn training job")
    parser.add_argument("--train-s3-uri", required=True, help="S3 URI to training CSV")
    parser.add_argument("--instance-type", default="ml.m5.xlarge", help="Training instance type")
    parser.add_argument("--job-name-prefix", default="chucaw-t2m-train", help="Base training job name")
    parser.add_argument("--framework-version", default="1.2-1", help="SageMaker sklearn framework version")
    parser.add_argument("--output-s3-uri", default="", help="S3 prefix for training outputs")
    parser.add_argument("--wait", action="store_true", help="Wait for completion")
    parser.add_argument("--logs", action="store_true", help="Stream logs while waiting")
    return parser.parse_args()


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env")

    region = required_env("REGION")
    role_arn = required_env("ROLE_ARN")
    bucket = required_env("MODEL_S3_BUCKET")
    model_prefix = required_env("MODEL_S3_PREFIX").strip("/")
    profile = os.getenv("AWS_PROFILE", "").strip()

    output_s3_uri = args.output_s3_uri or f"s3://{bucket}/{model_prefix}/training-jobs"

    boto_session_kwargs = {"region_name": region}
    if profile:
        boto_session_kwargs["profile_name"] = profile

    boto_session = boto3.Session(**boto_session_kwargs)
    sm_session = Session(boto_session=boto_session)

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=str(repo_root / "training"),
        role=role_arn,
        instance_count=1,
        instance_type=args.instance_type,
        framework_version=args.framework_version,
        py_version="py3",
        base_job_name=args.job_name_prefix,
        output_path=output_s3_uri,
        sagemaker_session=sm_session,
        hyperparameters={
            "n_neighbors": 4,
            "weights": "distance",
        },
        tags=[
            {"Key": "Project", "Value": os.getenv("PROJECT_TAG", "unknown")},
            {"Key": "Owner", "Value": os.getenv("OWNER_TAG", "unknown")},
            {"Key": "Environment", "Value": os.getenv("ENV_TAG", "dev")},
        ],
    )

    estimator.fit({"train": args.train_s3_uri}, wait=args.wait, logs=args.logs)

    print(f"Training job name: {estimator.latest_training_job.name}")
    if args.wait:
        print(f"Model artifact S3 URI: {estimator.model_data}")
        print(
            "Next step (build hosting artifact): "
            "python src/build_hosting_artifact.py "
            f"--training-artifact-s3-uri {estimator.model_data} "
            "--upload --bucket <bucket> --model-prefix <model-prefix> --source-prefix <source-prefix>"
        )
    else:
        print(
            "Next step (after job completes): "
            "python src/build_hosting_artifact.py --training-artifact-s3-uri "
            "<training-model-artifact-s3-uri> --upload "
            "--bucket <bucket> --model-prefix <model-prefix> --source-prefix <source-prefix>"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
