#!/usr/bin/env python3
import argparse
import sys
from typing import Any

import boto3


DEFAULT_MODEL_PACKAGE_GROUP_NAME = "chucaw-quicksight-abrupt-classifier"
DEFAULT_REGION = "us-east-1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List SageMaker Model Packages for QuickSight abrupt CSV classifier."
    )
    parser.add_argument(
        "--model-package-group-name",
        default=DEFAULT_MODEL_PACKAGE_GROUP_NAME,
        help="SageMaker Model Package Group name",
    )
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    return parser.parse_args()


def _iter_model_packages(sm_client: Any, group_name: str) -> list[dict[str, Any]]:
    paginator = sm_client.get_paginator("list_model_packages")
    pages = paginator.paginate(ModelPackageGroupName=group_name, SortBy="CreationTime", SortOrder="Descending")

    items: list[dict[str, Any]] = []
    for page in pages:
        items.extend(page.get("ModelPackageSummaryList", []))
    return items


def main() -> int:
    args = _parse_args()
    sm_client = boto3.client("sagemaker", region_name=args.region)
    packages = _iter_model_packages(sm_client, args.model_package_group_name)

    if not packages:
        print(f"No model packages found in group: {args.model_package_group_name}")
        return 0

    print(f"Model package group: {args.model_package_group_name}")
    for idx, pkg in enumerate(packages, start=1):
        print(f"#{idx}")
        print(f"  Status: {pkg.get('ModelPackageStatus')}")
        print(f"  Approval status: {pkg.get('ModelApprovalStatus')}")
        print(f"  ARN: {pkg.get('ModelPackageArn')}")
        print(f"  Created at: {pkg.get('CreationTime')}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
