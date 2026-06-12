#!/usr/bin/env python3

"""
REALTIME_ENDPOINT_COST_RISK
EXPERIMENTAL: Creates a real-time SageMaker endpoint (costly).
Not for normal usage. FourCastNet default path must use Batch Transform.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _print_verification_commands(region: str, profile: str, label: str) -> None:
    profile_arg = f" --profile {profile}" if profile else ""
    region_arg = f" --region {region}" if region else ""
    print(f"=== {label} verification commands ===")
    print(f"aws sagemaker list-endpoints --name-contains fourcastnet{profile_arg}{region_arg}")
    print(f"aws sagemaker list-transform-jobs --status-equals InProgress{profile_arg}{region_arg}")
    print(f"aws sagemaker list-training-jobs --status-equals InProgress{profile_arg}{region_arg}")


def check_endpoint_exists(sm_client, endpoint_name):
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        return True, response
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False, None
        raise


def delete_endpoint(sm_client, endpoint_name):
    logger.info(f"Deleting endpoint: {endpoint_name}")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    logger.info("Endpoint deletion initiated")


def wait_for_endpoint_deletion(sm_client, endpoint_name, max_wait=1200):
    logger.info(f"Waiting for endpoint {endpoint_name} to be deleted...")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        exists, _ = check_endpoint_exists(sm_client, endpoint_name)
        if not exists:
            logger.info("Endpoint deleted successfully")
            return True
        time.sleep(10)
    logger.error("Endpoint deletion timed out")
    return False


def check_endpoint_config_exists(sm_client, endpoint_config_name):
    try:
        response = sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        return True, response
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False, None
        raise


def check_model_exists(sm_client, model_name):
    try:
        response = sm_client.describe_model(ModelName=model_name)
        return True, response
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            return False, None
        raise


def create_model_if_not_exists(sm_client, model_name, role_arn, image_uri, model_data_url):
    exists, _ = check_model_exists(sm_client, model_name)
    if exists:
        logger.info(f"Model {model_name} already exists")
        return

    logger.info(f"Creating model: {model_name}")
    response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
        },
    )
    logger.info(f"Model created: {response}")


def create_endpoint_config_if_not_exists(
    sm_client, endpoint_config_name, model_name, instance_type, initial_instance_count
):
    exists, _ = check_endpoint_config_exists(sm_client, endpoint_config_name)
    if exists:
        logger.info(f"Endpoint config {endpoint_config_name} already exists")
        return

    logger.info(f"Creating endpoint config: {endpoint_config_name}")
    response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": initial_instance_count,
                "InstanceType": instance_type,
            }
        ],
    )
    logger.info(f"Endpoint config created: {response}")


def create_endpoint(sm_client, endpoint_name, endpoint_config_name):
    logger.info(f"Creating endpoint: {endpoint_name}")
    response = sm_client.create_endpoint(
        EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
    )
    logger.info(f"Endpoint created: {response}")


def wait_for_endpoint_in_service(sm_client, endpoint_name, max_wait=3600):
    logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")
    start_time = time.time()
    while time.time() - start_time < max_wait:
        exists, response = check_endpoint_exists(sm_client, endpoint_name)
        if not exists:
            logger.error("Endpoint no longer exists")
            return False

        status = response["EndpointStatus"]
        logger.info(f"Endpoint status: {status}")

        if status == "InService":
            logger.info("Endpoint is in service")
            return True
        elif status == "Failed":
            logger.error(
                f"Endpoint creation failed: {response.get('FailureReason', 'Unknown reason')}"
            )
            return False

        time.sleep(30)

    logger.error("Endpoint creation timed out")
    return False


def delete_failed_endpoint(sm_client, endpoint_name):
    exists, response = check_endpoint_exists(sm_client, endpoint_name)
    if not exists:
        logger.info(f"Endpoint {endpoint_name} does not exist")
        return True

    status = response["EndpointStatus"]
    if status == "Failed":
        logger.info(f"Deleting failed endpoint: {endpoint_name}")
        delete_endpoint(sm_client, endpoint_name)
        return wait_for_endpoint_deletion(sm_client, endpoint_name)

    logger.info(f"Endpoint {endpoint_name} is not in Failed state (status: {status})")
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="EXPERIMENTAL: Deploy a SageMaker real-time endpoint (costly)."
    )
    parser.add_argument(
        "--allow-realtime-endpoint",
        action="store_true",
        help="Required to proceed (creates a real-time endpoint).",
    )
    parser.add_argument(
        "--i-understand-this-can-cost-money",
        action="store_true",
        help="Second required confirmation flag.",
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        required=True,
        help="Name for the endpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sbnai-pytorch-fcn-model",
        help="Name for the model",
    )
    parser.add_argument(
        "--endpoint-config-name",
        type=str,
        default="sbnai-pytorch-fcn-config",
        help="Name for the endpoint configuration",
    )
    parser.add_argument(
        "--role-arn",
        type=str,
        default=os.environ.get("SAGEMAKER_ROLE_ARN", ""),
        help="IAM role ARN for SageMaker execution",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        required=True,
        help="Instance type for deployment",
    )
    parser.add_argument(
        "--initial-instance-count",
        type=int,
        default=1,
        help="Initial number of instances",
    )
    parser.add_argument(
        "--model-data-url",
        type=str,
        default="s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/model/model.tar.gz",
        help="S3 URL for model artifact",
    )
    parser.add_argument(
        "--image-uri",
        type=str,
        default="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310-ubuntu20.04-sagemaker",
        help="SageMaker inference image URI",
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="",
        help="AWS profile to use",
    )
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing endpoint if it exists",
    )
    parser.add_argument(
        "--delete-failed-endpoint",
        action="store_true",
        help="Delete endpoint if it is in Failed state",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.allow_realtime_endpoint or not args.i_understand_this_can_cost_money:
        print(
            "ERROR: Refusing to create a real-time endpoint. "
            "Required flags: --allow-realtime-endpoint --i-understand-this-can-cost-money",
            file=sys.stderr,
        )
        return 2

    if not args.role_arn:
        logger.error(
            "No role ARN provided. Please set SAGEMAKER_ROLE_ARN environment variable or "
            "pass --role-arn argument."
        )
        return 1

    session_kwargs = {"region_name": args.region}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    session = boto3.Session(**session_kwargs)
    sm_client = session.client("sagemaker")

    _print_verification_commands(args.region, args.profile, "Preflight")

    if args.delete_failed_endpoint:
        success = delete_failed_endpoint(sm_client, args.endpoint_name)
        if success:
            logger.info("Failed endpoint deleted successfully")
        else:
            logger.error("Failed to delete endpoint or endpoint not in Failed state")
        _print_verification_commands(args.region, args.profile, "Postflight")
        return 0 if success else 1

    exists, _ = check_endpoint_exists(sm_client, args.endpoint_name)
    if exists and args.delete_existing:
        delete_endpoint(sm_client, args.endpoint_name)
        wait_for_endpoint_deletion(sm_client, args.endpoint_name)
    elif exists:
        logger.error(
            f"Endpoint {args.endpoint_name} already exists. Use --delete-existing to delete it."
        )
        _print_verification_commands(args.region, args.profile, "Postflight")
        return 1

    create_model_if_not_exists(
        sm_client, args.model_name, args.role_arn, args.image_uri, args.model_data_url
    )

    create_endpoint_config_if_not_exists(
        sm_client,
        args.endpoint_config_name,
        args.model_name,
        args.instance_type,
        args.initial_instance_count,
    )

    create_endpoint(sm_client, args.endpoint_name, args.endpoint_config_name)
    wait_for_endpoint_in_service(sm_client, args.endpoint_name)
    _print_verification_commands(args.region, args.profile, "Postflight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
