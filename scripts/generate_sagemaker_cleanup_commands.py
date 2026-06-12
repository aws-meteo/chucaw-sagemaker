#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

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


@dataclass(frozen=True)
class EndpointRef:
    region: str
    endpoint_name: str


@dataclass(frozen=True)
class EndpointConfigRef:
    region: str
    endpoint_config_name: str


@dataclass(frozen=True)
class StudioAppRef:
    region: str
    domain_id: str
    user_profile_name: str
    app_type: str
    app_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only inventory, then generate local PowerShell cleanup commands (dry-run by default)."
    )
    parser.add_argument("--profile", default="", help="AWS profile")
    parser.add_argument("--region", default="us-east-1", help="AWS region for single-region scan")
    parser.add_argument("--all-regions", action="store_true", help="Scan all opted-in regions")
    parser.add_argument(
        "--name-contains",
        action="append",
        default=[],
        help="Name substring filter (repeatable). Defaults: fourcastnet, chucaw, sbnai",
    )
    parser.add_argument(
        "--output-script",
        default="generated_cleanup_sagemaker.ps1",
        help="Generated PowerShell script path",
    )
    return parser.parse_args()


def _filters(args: argparse.Namespace) -> list[str]:
    values = [v.strip().lower() for v in args.name_contains if v and v.strip()]
    return values or DEFAULT_NAME_FILTERS


def _contains_any(name: str, filters: list[str]) -> bool:
    lowered = (name or "").lower()
    return any(f in lowered for f in filters)


def _list_regions(session: boto3.Session, default_region: str, all_regions: bool) -> list[str]:
    if not all_regions:
        return [default_region]
    ec2 = session.client("ec2", region_name=default_region)
    response = ec2.describe_regions(AllRegions=True)
    out: list[str] = []
    for item in response.get("Regions", []):
        if item.get("OptInStatus") in {"opt-in-not-required", "opted-in"}:
            out.append(item["RegionName"])
    return sorted(set(out))


def _paginate(sm_client, op: str, key: str, **kwargs):
    paginator = sm_client.get_paginator(op)
    for page in paginator.paginate(**kwargs):
        for item in page.get(key, []):
            yield item


def _inventory(
    session: boto3.Session, regions: list[str], filters: list[str]
) -> tuple[list[EndpointRef], list[EndpointConfigRef], list[StudioAppRef]]:
    endpoints: set[EndpointRef] = set()
    endpoint_configs: set[EndpointConfigRef] = set()
    studio_apps: set[StudioAppRef] = set()

    for region in regions:
        sm = session.client("sagemaker", region_name=region)
        for needle in filters:
            for item in _paginate(sm, "list_endpoints", "Endpoints", NameContains=needle):
                endpoints.add(EndpointRef(region=region, endpoint_name=item["EndpointName"]))
            for item in _paginate(
                sm, "list_endpoint_configs", "EndpointConfigs", NameContains=needle
            ):
                endpoint_configs.add(
                    EndpointConfigRef(
                        region=region, endpoint_config_name=item["EndpointConfigName"]
                    )
                )
        for app in _paginate(sm, "list_apps", "Apps"):
            joined = " ".join(
                [
                    app.get("AppName", ""),
                    app.get("UserProfileName", ""),
                    app.get("AppType", ""),
                    app.get("DomainId", ""),
                ]
            )
            if _contains_any(joined, filters):
                studio_apps.add(
                    StudioAppRef(
                        region=region,
                        domain_id=app.get("DomainId", ""),
                        user_profile_name=app.get("UserProfileName", ""),
                        app_type=app.get("AppType", ""),
                        app_name=app.get("AppName", ""),
                    )
                )

    return sorted(endpoints, key=lambda x: (x.region, x.endpoint_name)), sorted(
        endpoint_configs, key=lambda x: (x.region, x.endpoint_config_name)
    ), sorted(studio_apps, key=lambda x: (x.region, x.domain_id, x.user_profile_name, x.app_type, x.app_name))


def _aws_base(profile: str, region: str) -> str:
    profile_part = f" --profile {profile}" if profile else ""
    return f"aws sagemaker{profile_part} --region {region}"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _render_script(
    profile: str,
    filters: list[str],
    endpoints: list[EndpointRef],
    endpoint_configs: list[EndpointConfigRef],
    studio_apps: list[StudioAppRef],
) -> str:
    lines: list[str] = []
    lines.append("param([switch]$Execute)")
    lines.append("")
    lines.append("$ErrorActionPreference = 'Stop'")
    lines.append("$mode = if ($Execute) { 'EXECUTE' } else { 'DRY-RUN' }")
    lines.append("Write-Host \"SageMaker cleanup mode: $mode\"")
    lines.append(
        "Write-Host 'This script will NOT delete SageMaker Models, S3 artifacts, ECR images, CloudWatch logs, domains, user profiles, or spaces.'"
    )
    lines.append(f"Write-Host \"Name filters used for inventory: {', '.join(filters)}\"")
    lines.append("")
    lines.append("function Invoke-Step([string]$Command) {")
    lines.append("  if ($Execute) {")
    lines.append("    Write-Host \"[EXECUTE] $Command\"")
    lines.append("    Invoke-Expression $Command")
    lines.append("  } else {")
    lines.append("    Write-Host \"[DRY-RUN] $Command\"")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("# Endpoints")
    if endpoints:
        for item in endpoints:
            cmd = f"{_aws_base(profile, item.region)} delete-endpoint --endpoint-name {item.endpoint_name}"
            lines.append(f"Invoke-Step {_ps_quote(cmd)}")
    else:
        lines.append("Write-Host 'No matching endpoints found in read-only inventory.'")
    lines.append("")
    lines.append("# Endpoint configs")
    if endpoint_configs:
        for item in endpoint_configs:
            cmd = (
                f"{_aws_base(profile, item.region)} delete-endpoint-config "
                f"--endpoint-config-name {item.endpoint_config_name}"
            )
            lines.append(f"Invoke-Step {_ps_quote(cmd)}")
    else:
        lines.append("Write-Host 'No matching endpoint configs found in read-only inventory.'")
    lines.append("")
    lines.append("# Studio apps")
    if studio_apps:
        for item in studio_apps:
            cmd = (
                f"{_aws_base(profile, item.region)} delete-app "
                f"--domain-id {item.domain_id} --user-profile-name {item.user_profile_name} "
                f"--app-type {item.app_type} --app-name {item.app_name}"
            )
            lines.append(f"Invoke-Step {_ps_quote(cmd)}")
    else:
        lines.append("Write-Host 'No matching Studio apps found in read-only inventory.'")
    lines.append("")
    lines.append("# Manual-only section (intentionally NOT automated):")
    lines.append("# - SageMaker Models")
    lines.append("# - S3 model artifacts")
    lines.append("# - ECR images")
    lines.append("# - CloudWatch logs")
    lines.append("# - SageMaker domains/user profiles/spaces")
    return "\n".join(lines) + "\n"


def main() -> int:
    if boto3 is None:
        print(
            "ERROR: boto3 is not installed in this interpreter. Install boto3 to generate cleanup inventory.",
            file=sys.stderr,
        )
        return 1

    args = parse_args()
    session_kwargs: dict[str, str] = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile
    try:
        session = boto3.Session(**session_kwargs)
    except ProfileNotFound as exc:
        print(f"ERROR: AWS profile not found: {exc}", file=sys.stderr)
        return 1
    filters = _filters(args)

    try:
        regions = _list_regions(session, args.region, args.all_regions)
        endpoints, endpoint_configs, studio_apps = _inventory(session, regions, filters)
    except (ClientError, BotoCoreError) as exc:
        print(f"ERROR: read-only inventory failed: {exc}", file=sys.stderr)
        return 1

    script_text = _render_script(
        args.profile, filters, endpoints, endpoint_configs, studio_apps
    )
    output_path = Path(args.output_script).resolve()
    output_path.write_text(script_text, encoding="utf-8")

    print(f"Generated: {output_path}")
    print(f"Endpoints: {len(endpoints)}")
    print(f"EndpointConfigs: {len(endpoint_configs)}")
    print(f"StudioApps: {len(studio_apps)}")
    print("Dry-run is default. Add -Execute when running the generated PowerShell script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
