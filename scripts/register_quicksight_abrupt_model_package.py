#!/usr/bin/env python3
import argparse
import sys
from typing import Any

try:
    from botocore.exceptions import ClientError
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    class ClientError(Exception):
        pass


DEFAULT_MODEL_PACKAGE_GROUP_NAME = "chucaw-quicksight-abrupt-classifier"
DEFAULT_REGION = "us-east-1"
DEFAULT_SKLEARN_VERSION = "1.2-1"
DEFAULT_APPROVAL_STATUS = "Approved"


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register QuickSight abrupt CSV classifier artifact in SageMaker Model Registry."
    )
    parser.add_argument(
        "--model-package-group-name",
        default=DEFAULT_MODEL_PACKAGE_GROUP_NAME,
        help="SageMaker Model Package Group name",
    )
    parser.add_argument("--model-artifact-s3-uri", required=True, help="S3 URI to model tar.gz")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument(
        "--sklearn-version",
        default=DEFAULT_SKLEARN_VERSION,
        help="Sklearn inference container version",
    )
    parser.add_argument(
        "--approval-status",
        default=DEFAULT_APPROVAL_STATUS,
        choices=["Approved", "Rejected", "PendingManualApproval"],
        help="Model package approval status",
    )
    parser.add_argument("--description", default="", help="Optional model package description")
    parser.add_argument(
        "--create-group-if-missing",
        type=_parse_bool,
        default=True,
        help="Create model package group when missing (true/false)",
    )
    return parser.parse_args()


def resolve_sklearn_image_uri(region: str, sklearn_version: str) -> str:
    from sagemaker import image_uris

    return image_uris.retrieve(
        framework="sklearn",
        region=region,
        version=sklearn_version,
        py_version="py3",
        image_scope="inference",
        instance_type="ml.m5.large",
    )


def build_inference_spec_for_image(image_uri: str, model_artifact_s3_uri: str) -> dict[str, Any]:
    return {
        "Containers": [
            {
                "Image": image_uri,
                "ModelDataUrl": model_artifact_s3_uri,
                "Environment": {
                    "SAGEMAKER_PROGRAM": "inference.py",
                    "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                },
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"],
    }


def build_customer_metadata_properties() -> dict[str, str]:
    return {
        "Project": "chucaw",
        "Component": "quicksight-sagemaker-augmentation",
        "ModelKind": "csv-abrupt-temperature-classifier",
        "InputColumns": "lat/lon/t2m",
        "OutputColumns": "abrupt_temp_change_label/abrupt_temp_change_score",
        "InferenceMode": "BatchTransform",
        "EndpointRequired": "false",
    }


def build_inference_spec(region: str, sklearn_version: str, model_artifact_s3_uri: str) -> dict[str, Any]:
    image_uri = resolve_sklearn_image_uri(region=region, sklearn_version=sklearn_version)
    return build_inference_spec_for_image(image_uri=image_uri, model_artifact_s3_uri=model_artifact_s3_uri)


def _ensure_model_package_group(
    sm_client: Any,
    model_package_group_name: str,
    create_group_if_missing: bool,
) -> None:
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=model_package_group_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code not in {"ValidationException", "ResourceNotFound"}:
            raise
        if not create_group_if_missing:
            raise RuntimeError(
                "Model package group does not exist and --create-group-if-missing is false: "
                f"{model_package_group_name}"
            ) from exc
        sm_client.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription="QuickSight abrupt CSV classifier model package group.",
            Tags=[
                {"Key": "Project", "Value": "chucaw"},
                {"Key": "Component", "Value": "quicksight-sagemaker-augmentation"},
            ],
        )


def main() -> int:
    try:
        import boto3
    except ModuleNotFoundError as exc:
        print("ERROR: boto3 is required to register SageMaker model packages.", file=sys.stderr)
        raise SystemExit(1) from exc

    args = _parse_args()
    sm_client = boto3.client("sagemaker", region_name=args.region)
    _ensure_model_package_group(
        sm_client=sm_client,
        model_package_group_name=args.model_package_group_name,
        create_group_if_missing=args.create_group_if_missing,
    )

    inference_spec = build_inference_spec(args.region, args.sklearn_version, args.model_artifact_s3_uri)
    customer_metadata_properties = build_customer_metadata_properties()
    response = sm_client.create_model_package(
        ModelPackageGroupName=args.model_package_group_name,
        ModelApprovalStatus=args.approval_status,
        ModelPackageDescription=args.description,
        InferenceSpecification=inference_spec,
        CustomerMetadataProperties=customer_metadata_properties,
    )

    package_arn = response["ModelPackageArn"]
    image_uri = inference_spec["Containers"][0]["Image"]
    print(f"Model package group name: {args.model_package_group_name}")
    print(f"Model package ARN: {package_arn}")
    print(f"Approval status: {args.approval_status}")
    print(f"Artifact URI: {args.model_artifact_s3_uri}")
    print(f"Image URI: {image_uri}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
