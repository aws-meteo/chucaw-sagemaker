#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from botocore.exceptions import ClientError
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    class ClientError(Exception):
        pass

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fourcastnet_batch_v0.json"


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _coalesce(*values: str) -> str:
    for value in values:
        cleaned = (value or "").strip()
        if cleaned:
            return cleaned
    return ""


def _required(value: str, name: str) -> str:
    cleaned = (value or "").strip()
    if not cleaned:
        raise ValueError(f"Missing required value: {name}")
    return cleaned


def _tag_list(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": key, "Value": value} for key, value in tags.items()]


def _print_verification_commands(region: str, profile: str, label: str) -> None:
    profile_arg = f" --profile {profile}" if profile else ""
    region_arg = f" --region {region}" if region else ""
    print(f"=== {label} verification commands ===")
    print(f"aws sagemaker list-endpoints --name-contains fourcastnet{profile_arg}{region_arg}")
    print(f"aws sagemaker list-transform-jobs --status-equals InProgress{profile_arg}{region_arg}")
    print(f"aws sagemaker list-training-jobs --status-equals InProgress{profile_arg}{region_arg}")


def _describe_model(sm_client, model_name: str) -> dict[str, Any] | None:
    try:
        return sm_client.describe_model(ModelName=model_name)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        message = str(exc).lower()
        if code in {"ValidationException", "ResourceNotFound"} and "could not find model" in message:
            return None
        raise


def _print_model_details(model_desc: dict[str, Any]) -> None:
    primary = model_desc.get("PrimaryContainer") or {}
    containers = model_desc.get("Containers") or []
    if not primary and containers:
        primary = containers[0]

    print(f"ModelName: {model_desc.get('ModelName')}")
    print(f"ExecutionRoleArn: {model_desc.get('ExecutionRoleArn')}")
    print(f"Image: {primary.get('Image')}")
    print(f"ModelDataUrl: {primary.get('ModelDataUrl')}")
    print(f"ModelPackageName: {primary.get('ModelPackageName')}")
    print(f"Environment: {json.dumps(primary.get('Environment') or {}, indent=2)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe or create FourCastNet SageMaker model (batch-safe).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config JSON")
    parser.add_argument("--model-name", default="", help="Override model name")
    parser.add_argument("--model-package-arn", default="", help="Optional model package ARN")
    parser.add_argument("--model-data-url", default="", help="S3 URI to model.tar.gz")
    parser.add_argument("--image-uri", default="", help="Inference image URI")
    parser.add_argument("--role-arn", default="", help="Execution role ARN")
    parser.add_argument("--region", default="", help="AWS region")
    parser.add_argument("--profile", default="", help="AWS profile")
    return parser.parse_args()

def main() -> int:
    args = _parse_args()
    config = _load_config(Path(args.config))

    config_package_arn = str(config.get("model_package_arn", "")).strip()
    if args.model_data_url and (args.model_package_arn or config_package_arn):
        conflict_source = "CLI --model-package-arn" if args.model_package_arn else f"config model_package_arn ('{config_package_arn}')"
        raise ValueError(
            f"Conflict detected: CLI explicitly passed --model-data-url, but {conflict_source} is present. "
            "To use direct-artifact creation, please remove or clear the model_package_arn in the config or CLI arguments."
        )

    region = _coalesce(args.region, str(config.get("region", "")))
    profile = _coalesce(args.profile, str(config.get("aws_profile", "")))
    model_name = _coalesce(args.model_name, str(config.get("model_name", "")))
    model_package_arn = _coalesce(args.model_package_arn, config_package_arn)

    model_data_url = _coalesce(args.model_data_url, str(config.get("model_data_url", "")))
    image_uri = _coalesce(args.image_uri, str(config.get("image_uri", "")))
    role_arn = _coalesce(args.role_arn, str(config.get("execution_role_arn", "")))
    container_env = config.get("container_environment") or {}
    tags = config.get("tags") or {}

    _required(region, "region")
    _required(model_name, "model_name")

    try:
        import boto3
    except ModuleNotFoundError as exc:
        print("ERROR: boto3 is required to describe or create SageMaker models.", file=sys.stderr)
        raise SystemExit(1) from exc

    session_kwargs = {"region_name": region}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.Session(**session_kwargs)
    sm_client = session.client("sagemaker")

    _print_verification_commands(region, profile, "Preflight")
    print(f"DescribeModel command: aws sagemaker describe-model --model-name {model_name}")

    model_desc = _describe_model(sm_client, model_name)
    if model_desc:
        print("Model exists. Details:")
        _print_model_details(model_desc)
        _print_verification_commands(region, profile, "Postflight")
        return 0

    model_package_arn = _coalesce(model_package_arn, "")
    if model_package_arn:
        container = {"ModelPackageName": model_package_arn}
    else:
        _required(model_data_url, "model_data_url")
        _required(image_uri, "image_uri")
        container = {
            "Image": image_uri,
            "ModelDataUrl": model_data_url,
            "Environment": container_env,
        }

    _required(role_arn, "role_arn")
    payload = {
        "ModelName": model_name,
        "ExecutionRoleArn": role_arn,
        "PrimaryContainer": container,
        "Tags": _tag_list(tags) if isinstance(tags, dict) else tags,
    }

    print("Model not found. Creating from config.")
    print("=== AWS CLI create-model payload (save to JSON if needed) ===")
    print(json.dumps(payload, indent=2))
    print(f"CreateModel command: aws sagemaker create-model --cli-input-json file://<payload.json>")

    sm_client.create_model(**payload)
    print("Model created.")
    created_desc = _describe_model(sm_client, model_name)
    if created_desc:
        _print_model_details(created_desc)
    _print_verification_commands(region, profile, "Postflight")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
