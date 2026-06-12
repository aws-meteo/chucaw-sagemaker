#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Any

import boto3


def _print_verification_commands(region: str, profile: str, label: str) -> None:
    profile_arg = f" --profile {profile}" if profile else ""
    region_arg = f" --region {region}" if region else ""
    print(f"=== {label} verification commands ===")
    print(f"aws sagemaker list-endpoints --name-contains fourcastnet{profile_arg}{region_arg}")
    print(f"aws sagemaker list-transform-jobs --status-equals InProgress{profile_arg}{region_arg}")
    print(f"aws sagemaker list-training-jobs --status-equals InProgress{profile_arg}{region_arg}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail if any FourCastNet endpoints are active.")
    parser.add_argument("--name-contains", default="fourcastnet", help="Substring to match endpoint names")
    parser.add_argument("--region", default="", help="AWS region")
    parser.add_argument("--profile", default="", help="AWS profile")
    return parser.parse_args()


def _list_endpoints(sm_client, name_contains: str) -> list[dict[str, Any]]:
    endpoints: list[dict[str, Any]] = []
    paginator = sm_client.get_paginator("list_endpoints")
    for page in paginator.paginate(NameContains=name_contains):
        endpoints.extend(page.get("Endpoints", []))
    return endpoints


def main() -> int:
    args = _parse_args()
    region = (args.region or "").strip()
    profile = (args.profile or "").strip()
    if not region:
        raise ValueError("Missing required value: region")

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    sm_client = session.client("sagemaker")

    _print_verification_commands(region, profile, "Preflight")
    endpoints = _list_endpoints(sm_client, args.name_contains)
    if not endpoints:
        print(f"No endpoints found matching '{args.name_contains}'.")
        _print_verification_commands(region, profile, "Postflight")
        return 0

    risky_statuses = {"InService", "Creating", "Updating", "SystemUpdating"}
    flagged = [ep for ep in endpoints if ep.get("EndpointStatus") in risky_statuses]

    print("Endpoints found:")
    for ep in endpoints:
        print(f"- {ep.get('EndpointName')} ({ep.get('EndpointStatus')})")

    _print_verification_commands(region, profile, "Postflight")
    if flagged:
        print("ERROR: Active endpoint(s) detected.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
