#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError, ProfileNotFound
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    boto3 = None  # type: ignore[assignment]

    class BotoCoreError(Exception):
        pass

    class ClientError(Exception):
        pass

    class ProfileNotFound(Exception):
        pass


DEFAULT_NAME_FILTERS = ["fourcastnet", "chucaw", "sbnai"]
RISKY_ENDPOINT_STATUSES = {"Creating", "Updating", "InService", "SystemUpdating", "RollingBack"}
RISKY_NOTEBOOK_STATUSES = {"Pending", "InService", "Updating", "Stopping"}
RISKY_APP_STATUSES = {"Pending", "InService", "Deleting"}


@dataclass
class RegionInventory:
    region: str
    endpoints: list[dict[str, Any]]
    notebook_instances: list[dict[str, Any]]
    studio_apps: list[dict[str, Any]]
    training_jobs_in_progress: list[dict[str, Any]]
    processing_jobs_in_progress: list[dict[str, Any]]
    transform_jobs_in_progress: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only SageMaker preflight: detect endpoint/notebook/studio/active job risks."
    )
    parser.add_argument("--profile", default="", help="AWS profile")
    parser.add_argument("--region", default="us-east-1", help="Single AWS region")
    parser.add_argument("--all-regions", action="store_true", help="Scan all opt-in AWS regions")
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="Name substring filter (repeatable). Defaults: fourcastnet, chucaw, sbnai",
    )
    parser.add_argument(
        "--fail-on-any-endpoint",
        action="store_true",
        help="Fail if any matching endpoint exists, regardless of status.",
    )
    parser.add_argument(
        "--fail-on-studio-app",
        action="store_true",
        help="Fail if any matching Studio app exists, regardless of status.",
    )
    parser.add_argument("--json-output", action="store_true", help="Print machine-readable JSON")
    return parser.parse_args()


def _name_filters(args: argparse.Namespace) -> list[str]:
    values = [v.strip().lower() for v in args.name_contains if v and v.strip()]
    return values or DEFAULT_NAME_FILTERS


def _profile_prefix(profile: str) -> str:
    return f" --profile {profile}" if profile else ""


def _print_cli_equivalents(region: str, profile: str, filters: list[str]) -> None:
    profile_arg = _profile_prefix(profile)
    print(f"\n=== Region {region} CLI equivalents ===")
    for needle in filters:
        print(
            f"aws sagemaker list-endpoints --name-contains {needle}{profile_arg} --region {region}"
        )
        print(
            f"aws sagemaker list-notebook-instances --name-contains {needle}{profile_arg} --region {region}"
        )
        print(f"aws sagemaker list-apps{profile_arg} --region {region}  # post-filter by name contains '{needle}'")
    print(f"aws sagemaker list-training-jobs --status-equals InProgress{profile_arg} --region {region}")
    print(f"aws sagemaker list-processing-jobs --status-equals InProgress{profile_arg} --region {region}")
    print(f"aws sagemaker list-transform-jobs --status-equals InProgress{profile_arg} --region {region}")


def _iter_regions(session: boto3.Session, default_region: str, all_regions: bool) -> list[str]:
    if not all_regions:
        return [default_region]
    ec2 = session.client("ec2", region_name=default_region)
    response = ec2.describe_regions(AllRegions=True)
    regions: list[str] = []
    for region in response.get("Regions", []):
        status = region.get("OptInStatus", "")
        if status in {"opt-in-not-required", "opted-in"}:
            regions.append(region["RegionName"])
    return sorted(set(regions))


def _aws_cli_base(service: str, profile: str, region: str) -> list[str]:
    args = ["aws", service]
    if profile:
        args.extend(["--profile", profile])
    args.extend(["--region", region])
    return args


def _run_aws_cli_json(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        args + ["--output", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"AWS CLI command failed: {' '.join(args)}\n{detail}")
    if not result.stdout.strip():
        return {}
    return json.loads(result.stdout)


def _iter_regions_cli(profile: str, default_region: str, all_regions: bool) -> list[str]:
    if not all_regions:
        return [default_region]
    payload = _run_aws_cli_json(
        _aws_cli_base("ec2", profile, default_region) + ["describe-regions", "--all-regions"]
    )
    regions: list[str] = []
    for region in payload.get("Regions", []):
        status = region.get("OptInStatus", "")
        if status in {"opt-in-not-required", "opted-in"}:
            regions.append(region["RegionName"])
    return sorted(set(regions))


def _contains_any(name: str, needles: list[str]) -> bool:
    lowered = (name or "").lower()
    return any(needle in lowered for needle in needles)


def _collect_with_paginator(client: Any, op: str, key: str, **kwargs: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    paginator = client.get_paginator(op)
    for page in paginator.paginate(**kwargs):
        items.extend(page.get(key, []))
    return items


def _filter_named(items: list[dict[str, Any]], key: str, needles: list[str]) -> list[dict[str, Any]]:
    return [item for item in items if _contains_any(str(item.get(key, "")), needles)]


def _inventory_region(session: boto3.Session, region: str, needles: list[str]) -> RegionInventory:
    sm = session.client("sagemaker", region_name=region)
    endpoints = []
    notebook_instances = []
    for needle in needles:
        endpoints.extend(_collect_with_paginator(sm, "list_endpoints", "Endpoints", NameContains=needle))
        notebook_instances.extend(
            _collect_with_paginator(
                sm, "list_notebook_instances", "NotebookInstances", NameContains=needle
            )
        )
    apps = _collect_with_paginator(sm, "list_apps", "Apps")
    studio_apps = _filter_named(apps, "AppName", needles)

    training_jobs_in_progress = _collect_with_paginator(
        sm, "list_training_jobs", "TrainingJobSummaries", StatusEquals="InProgress"
    )
    processing_jobs_in_progress = _collect_with_paginator(
        sm, "list_processing_jobs", "ProcessingJobSummaries", StatusEquals="InProgress"
    )
    transform_jobs_in_progress = _collect_with_paginator(
        sm, "list_transform_jobs", "TransformJobSummaries", StatusEquals="InProgress"
    )
    return RegionInventory(
        region=region,
        endpoints=endpoints,
        notebook_instances=notebook_instances,
        studio_apps=studio_apps,
        training_jobs_in_progress=training_jobs_in_progress,
        processing_jobs_in_progress=processing_jobs_in_progress,
        transform_jobs_in_progress=transform_jobs_in_progress,
    )


def _inventory_region_cli(profile: str, region: str, needles: list[str]) -> RegionInventory:
    endpoints: list[dict[str, Any]] = []
    notebook_instances: list[dict[str, Any]] = []
    for needle in needles:
        endpoints.extend(
            _run_aws_cli_json(
                _aws_cli_base("sagemaker", profile, region)
                + ["list-endpoints", "--name-contains", needle]
            ).get("Endpoints", [])
        )
        notebook_instances.extend(
            _run_aws_cli_json(
                _aws_cli_base("sagemaker", profile, region)
                + ["list-notebook-instances", "--name-contains", needle]
            ).get("NotebookInstances", [])
        )

    apps = _run_aws_cli_json(_aws_cli_base("sagemaker", profile, region) + ["list-apps"]).get("Apps", [])
    training_jobs_in_progress = _run_aws_cli_json(
        _aws_cli_base("sagemaker", profile, region)
        + ["list-training-jobs", "--status-equals", "InProgress"]
    ).get("TrainingJobSummaries", [])
    processing_jobs_in_progress = _run_aws_cli_json(
        _aws_cli_base("sagemaker", profile, region)
        + ["list-processing-jobs", "--status-equals", "InProgress"]
    ).get("ProcessingJobSummaries", [])
    transform_jobs_in_progress = _run_aws_cli_json(
        _aws_cli_base("sagemaker", profile, region)
        + ["list-transform-jobs", "--status-equals", "InProgress"]
    ).get("TransformJobSummaries", [])

    return RegionInventory(
        region=region,
        endpoints=endpoints,
        notebook_instances=notebook_instances,
        studio_apps=_filter_named(apps, "AppName", needles),
        training_jobs_in_progress=training_jobs_in_progress,
        processing_jobs_in_progress=processing_jobs_in_progress,
        transform_jobs_in_progress=transform_jobs_in_progress,
    )


def _region_has_risk(inv: RegionInventory, fail_on_any_endpoint: bool, fail_on_studio_app: bool) -> bool:
    if fail_on_any_endpoint and inv.endpoints:
        return True
    if any(item.get("EndpointStatus") in RISKY_ENDPOINT_STATUSES for item in inv.endpoints):
        return True
    if any(item.get("NotebookInstanceStatus") in RISKY_NOTEBOOK_STATUSES for item in inv.notebook_instances):
        return True
    if fail_on_studio_app and inv.studio_apps:
        return True
    if any(item.get("Status") in RISKY_APP_STATUSES for item in inv.studio_apps):
        return True
    if inv.training_jobs_in_progress or inv.processing_jobs_in_progress or inv.transform_jobs_in_progress:
        return True
    return False


def _print_human_summary(inv: RegionInventory) -> None:
    print(f"\n=== Region {inv.region} summary ===")
    print(f"matching endpoints: {len(inv.endpoints)}")
    for item in inv.endpoints:
        print(f"  - {item.get('EndpointName')} ({item.get('EndpointStatus')})")
    print(f"matching notebook instances: {len(inv.notebook_instances)}")
    for item in inv.notebook_instances:
        print(f"  - {item.get('NotebookInstanceName')} ({item.get('NotebookInstanceStatus')})")
    print(f"matching studio apps: {len(inv.studio_apps)}")
    for item in inv.studio_apps:
        print(
            f"  - {item.get('DomainId')}/{item.get('UserProfileName')}/{item.get('AppType')}/{item.get('AppName')} ({item.get('Status')})"
        )
    print(f"in-progress training jobs: {len(inv.training_jobs_in_progress)}")
    print(f"in-progress processing jobs: {len(inv.processing_jobs_in_progress)}")
    print(f"in-progress transform jobs: {len(inv.transform_jobs_in_progress)}")


def main() -> int:
    args = parse_args()
    needles = _name_filters(args)
    backend = "boto3"
    inventories: list[RegionInventory] = []

    if boto3 is None:
        if shutil.which("aws") is None:
            print(
                "ERROR: boto3 is not installed and AWS CLI was not found; cannot run AWS preflight.",
                file=sys.stderr,
            )
            return 1
        backend = "aws-cli"
        try:
            regions = _iter_regions_cli(args.profile, args.region, args.all_regions)
        except RuntimeError as exc:
            print(f"ERROR: unable to list regions with AWS CLI: {exc}", file=sys.stderr)
            return 1

        for region in regions:
            _print_cli_equivalents(region, args.profile, needles)
            try:
                inv = _inventory_region_cli(args.profile, region, needles)
            except RuntimeError as exc:
                print(f"ERROR: read-only AWS CLI inventory failed in {region}: {exc}", file=sys.stderr)
                return 1
            inventories.append(inv)
            if not args.json_output:
                _print_human_summary(inv)
    else:
        session_kwargs: dict[str, str] = {}
        if args.profile:
            session_kwargs["profile_name"] = args.profile
        try:
            session = boto3.Session(**session_kwargs)
        except ProfileNotFound as exc:
            print(f"ERROR: AWS profile not found: {exc}", file=sys.stderr)
            return 1

        try:
            regions = _iter_regions(session, args.region, args.all_regions)
        except (ClientError, BotoCoreError) as exc:
            print(f"ERROR: unable to list regions: {exc}", file=sys.stderr)
            return 1

        for region in regions:
            _print_cli_equivalents(region, args.profile, needles)
            try:
                inv = _inventory_region(session, region, needles)
            except (ClientError, BotoCoreError) as exc:
                print(f"ERROR: read-only inventory failed in {region}: {exc}", file=sys.stderr)
                return 1
            inventories.append(inv)
            if not args.json_output:
                _print_human_summary(inv)

    has_risk = any(
        _region_has_risk(
            inv,
            fail_on_any_endpoint=args.fail_on_any_endpoint,
            fail_on_studio_app=args.fail_on_studio_app,
        )
        for inv in inventories
    )

    if args.json_output:
        payload = {
            "regions": [asdict(inv) for inv in inventories],
            "filters": needles,
            "fail_on_any_endpoint": args.fail_on_any_endpoint,
            "fail_on_studio_app": args.fail_on_studio_app,
            "inventory_backend": backend,
            "risk_found": has_risk,
        }
        print(json.dumps(payload, indent=2, default=str))

    if has_risk:
        print("RISK: always-on or active SageMaker compute detected.", file=sys.stderr)
        return 2

    print("OK: no risky SageMaker always-on compute detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
