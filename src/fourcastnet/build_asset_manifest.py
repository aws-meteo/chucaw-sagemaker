#!/usr/bin/env python3
"""Build and validate a local FourCastNet asset manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/validate FourCastNet local asset manifest")
    parser.add_argument(
        "--assets-dir",
        default="chucaw-glue-scripts/data/fourcastnet_assets_v0",
        help="Relative or absolute path to FCN assets directory",
    )
    parser.add_argument(
        "--tensor-dir",
        default="chucaw-glue-scripts/data/fourcastnet_tensor_real_v1",
        help="Relative or absolute path to FCN tensor directory",
    )
    parser.add_argument(
        "--tensor-report",
        default="tensor_validation_report_with_stats.json",
        help="Report filename under --tensor-dir",
    )
    parser.add_argument(
        "--channel-order",
        default="NVlabs_FCN_v0_official_20ch",
        help="Expected channel-order contract string",
    )
    parser.add_argument(
        "--output",
        default="artifacts/fourcastnet/fcn_asset_manifest.json",
        help="Output JSON path",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw: str, repo_root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def validate_tensor(tensor_path: Path) -> dict[str, Any]:
    tensor = np.load(tensor_path)
    finite = bool(np.isfinite(tensor).all())
    return {
        "path": str(tensor_path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": finite,
        "expected_shape": [1, 20, 720, 1440],
        "expected_dtype": "float32",
        "shape_ok": tuple(tensor.shape) == (1, 20, 720, 1440),
        "dtype_ok": str(tensor.dtype) == "float32",
    }


def validate_stats_shape(stats_path: Path) -> dict[str, Any]:
    arr = np.load(stats_path)
    arr = np.asarray(arr)
    channels = None
    if arr.ndim == 1:
        channels = int(arr.shape[0])
    elif arr.ndim == 4:
        channels = int(arr.shape[1])
    return {
        "path": str(stats_path),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "channels": channels,
        "supports_first_20_policy": bool(channels is not None and channels >= 20),
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]

    assets_dir = resolve_path(args.assets_dir, repo_root)
    tensor_dir = resolve_path(args.tensor_dir, repo_root)
    output_path = resolve_path(args.output, Path.cwd())
    tensor_report_path = tensor_dir / args.tensor_report

    required_assets = {
        "checkpoint": assets_dir / "backbone.ckpt",
        "global_means": assets_dir / "global_means.npy",
        "global_stds": assets_dir / "global_stds.npy",
        "input_tensor": tensor_dir / "input_tensor.npy",
        "tensor_validation_report": tensor_report_path,
    }

    missing = [name for name, path in required_assets.items() if not path.exists()]
    if missing:
        print(f"ERROR: missing required assets: {missing}", file=sys.stderr)
        return 1

    tensor_meta = validate_tensor(required_assets["input_tensor"])
    means_meta = validate_stats_shape(required_assets["global_means"])
    stds_meta = validate_stats_shape(required_assets["global_stds"])

    report_payload = json.loads(required_assets["tensor_validation_report"].read_text(encoding="utf-8"))
    reported_shape = report_payload.get("shape")
    reported_channel_contract = report_payload.get("means_metadata", {}).get("tensor_channel_order")
    means_policy = report_payload.get("means_metadata", {}).get("stats_channel_policy")
    stds_policy = report_payload.get("stds_metadata", {}).get("stats_channel_policy")

    manifest = {
        "created_at_epoch": int(time.time()),
        "repo_root": str(repo_root),
        "assets_dir": str(assets_dir),
        "tensor_dir": str(tensor_dir),
        "target_contract": {
            "tensor_shape": [1, 20, 720, 1440],
            "tensor_dtype": "float32",
            "channel_order_contract": args.channel_order,
            "stats_channel_policy": "first_20_channels",
        },
        "assets": {},
        "checks": {
            "tensor_shape_ok": bool(tensor_meta["shape_ok"]),
            "tensor_dtype_ok": bool(tensor_meta["dtype_ok"]),
            "tensor_finite_ok": bool(tensor_meta["finite"]),
            "means_supports_20ch_policy": bool(means_meta["supports_first_20_policy"]),
            "stds_supports_20ch_policy": bool(stds_meta["supports_first_20_policy"]),
            "report_shape_ok": reported_shape == [1, 20, 720, 1440],
            "report_channel_contract_ok": reported_channel_contract == args.channel_order,
            "report_stats_policy_ok": means_policy == "first_20_channels" and stds_policy == "first_20_channels",
        },
    }

    for name, path in required_assets.items():
        manifest["assets"][name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    manifest["assets"]["input_tensor"]["tensor_meta"] = tensor_meta
    manifest["assets"]["global_means"]["stats_meta"] = means_meta
    manifest["assets"]["global_stds"]["stats_meta"] = stds_meta

    manifest["ok"] = all(bool(v) for v in manifest["checks"].values())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Manifest written: {output_path}")
    print(f"Validation status: {'PASS' if manifest['ok'] else 'FAIL'}")
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

