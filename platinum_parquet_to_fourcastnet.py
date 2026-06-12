"""Glue job: Platinum Parquet partition -> FourCastNet snapshot artifacts."""

from __future__ import annotations

from pathlib import Path

import boto3
import numpy as np
import pandas as pd

from chucaw_preprocessor.fourcastnet import (
    build_fourcastnet_tensor,
    build_validation_report,
    write_manifest,
)
from chucaw_preprocessor.glue_args import resolve_args

DEFAULT_PLATINUM_BUCKET = "chucaw-data-platinum-processed-725644097028-us-east-1-an"
DEFAULT_PLATINUM_PARQUET_PREFIX = "ecmwf/parquet"
DEFAULT_FOURCASTNET_PREFIX = "ecmwf/fourcastnet"


def _truthy(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _normalize_hour_arg(hour_value: str) -> tuple[str, str]:
    raw = str(hour_value or "").strip().lower()
    if not raw:
        raise ValueError("HOUR cannot be empty")
    if raw.endswith("z"):
        raw = raw[:-1]
    hour_num = int(raw)
    if hour_num < 0 or hour_num > 23:
        raise ValueError(f"Invalid HOUR value: {hour_value}")
    partition_hour = f"{hour_num:02d}"
    run_hour = f"{partition_hour}z"
    return partition_hour, run_hour


def _partition_prefix(base_prefix: str, year: str, month: str, day: str, partition_hour: str) -> str:
    return (
        f"{base_prefix.strip('/')}"
        f"/year={year}/month={month}/day={day}/hour={partition_hour}"
    ).strip("/")


def _read_partition_from_s3(bucket: str, prefix: str, tmp_dir: str) -> list[str]:
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    parquet_keys: list[str] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix.rstrip('/')}/"):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".parquet"):
                parquet_keys.append(key)

    if not parquet_keys:
        raise RuntimeError(f"No parquet files found under s3://{bucket}/{prefix}")

    local_paths: list[str] = []
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    for key in sorted(parquet_keys):
        target = str(Path(tmp_dir) / Path(key).name)
        s3.download_file(bucket, key, target)
        local_paths.append(target)

    return local_paths


def _upload_file(local_path: str, bucket: str, key: str) -> None:
    boto3.client("s3").upload_file(local_path, bucket, key)


def run_job(args: dict[str, str]) -> dict[str, str]:
    platinum_bucket = args.get("PLATINUM_BUCKET") or DEFAULT_PLATINUM_BUCKET
    parquet_prefix = args.get("PLATINUM_PARQUET_PREFIX") or DEFAULT_PLATINUM_PARQUET_PREFIX
    fourcastnet_prefix = args.get("FOURCASTNET_PREFIX") or DEFAULT_FOURCASTNET_PREFIX
    tmp_dir = args.get("TMP_DIR") or "/tmp"
    allow_incomplete = _truthy(args.get("ALLOW_INCOMPLETE") or "false")
    latitude_policy = args.get("LATITUDE_POLICY") or "fail"

    year = f"{int(args['YEAR']):04d}"
    month = f"{int(args['MONTH']):02d}"
    day = f"{int(args['DAY']):02d}"
    partition_hour, run_hour = _normalize_hour_arg(args["HOUR"])

    source_partition = _partition_prefix(parquet_prefix, year, month, day, run_hour)
    target_partition = _partition_prefix(fourcastnet_prefix, year, month, day, run_hour)

    local_paths = _read_partition_from_s3(platinum_bucket, source_partition, tmp_dir)
    
    global_manifest = {
        "status": "ok",
        "platinum_bucket": platinum_bucket,
        "source_partition": source_partition,
        "target_partition": target_partition,
        "year": year,
        "month": month,
        "day": day,
        "hour": partition_hour,
        "run_hour": run_hour,
        "processed_files": []
    }

    local_out = Path(tmp_dir) / "fourcastnet_out"
    local_out.mkdir(parents=True, exist_ok=True)

    for local_path in local_paths:
        file_name = Path(local_path).stem
        step_manifest = {
            "source_file": file_name + ".parquet",
            "allow_incomplete": allow_incomplete,
            "tensor_written": False,
            "latitude_policy": latitude_policy,
            "tensor_write_reason": "validation_pending",
        }
        
        df = pd.read_parquet(local_path)
        validation_report = build_validation_report(df, latitude_policy=latitude_policy)
        
        local_validation = str(local_out / f"{file_name}_validation_report.json")
        write_manifest(local_validation, validation_report)
        _upload_file(local_validation, platinum_bucket, f"{target_partition}/{file_name}_validation_report.json")

        if not validation_report.get("ok", False):
            step_manifest["status"] = "incomplete"
            step_manifest["tensor_write_reason"] = "validation_failed"
            global_manifest["processed_files"].append(step_manifest)
            if not allow_incomplete:
                raise RuntimeError(f"Validation failed for {file_name}: missing required channels/columns; tensor was not written")
            continue

        tensor = build_fourcastnet_tensor(df, latitude_policy=latitude_policy)
        local_tensor = str(local_out / f"{file_name}_tensor.npy")
        np.save(local_tensor, tensor)
        _upload_file(local_tensor, platinum_bucket, f"{target_partition}/{file_name}_tensor.npy")

        step_manifest["tensor_written"] = True
        step_manifest["tensor_write_reason"] = "written"
        step_manifest["tensor_shape"] = list(tensor.shape)
        step_manifest["tensor_dtype"] = str(tensor.dtype)
        step_manifest["tensor_file"] = f"{file_name}_tensor.npy"
        global_manifest["processed_files"].append(step_manifest)

    local_manifest = str(local_out / "manifest.json")
    write_manifest(local_manifest, global_manifest)
    _upload_file(local_manifest, platinum_bucket, f"{target_partition}/manifest.json")

    return global_manifest


def main() -> None:
    args = resolve_args(
        required=["YEAR", "MONTH", "DAY", "HOUR"],
        optional=[
            "PLATINUM_BUCKET",
            "PLATINUM_PARQUET_PREFIX",
            "FOURCASTNET_PREFIX",
            "TMP_DIR",
            "ALLOW_INCOMPLETE",
            "LATITUDE_POLICY",
        ],
    )
    result = run_job(args)
    print(result)


if __name__ == "__main__":
    main()
