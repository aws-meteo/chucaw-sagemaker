#!/usr/bin/env python3
"""Package the FourCastNet model into a self-contained or separate-source SageMaker model.tar.gz.

This script ensures that the correct FourCastNet serving handler (from src/fourcastnet/serving/)
is packaged rather than the trivial t2m handler.
"""

from __future__ import annotations

import argparse
import os
import tarfile
import tempfile
from pathlib import Path

DEFAULT_ASSETS = "chucaw-glue-scripts/data/fourcastnet_assets_v0"
DEFAULT_SERVING = "src/fourcastnet/serving"
DEFAULT_OUTPUT = "artifacts/fourcastnet/build"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package FourCastNet model.tar.gz for SageMaker")
    parser.add_argument(
        "--assets-dir",
        default=DEFAULT_ASSETS,
        help="Directory containing backbone.ckpt, global_means.npy, global_stds.npy",
    )
    parser.add_argument(
        "--serving-dir",
        default=DEFAULT_SERVING,
        help="Directory containing inference.py and requirements.txt",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Output directory for the packaged tarball",
    )
    parser.add_argument(
        "--layout",
        choices=["self-contained", "separate-source"],
        default="self-contained",
        help="Layout of model.tar.gz: self-contained (includes code/) or separate-source (no code/)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    parent_root = Path(__file__).resolve().parents[2]

    # Resolve paths
    assets_path = Path(args.assets_dir)
    if not assets_path.is_absolute():
        assets_path = (parent_root / assets_path).resolve()


    serving_path = Path(args.serving_dir)
    if not serving_path.is_absolute():
        serving_path = (repo_root / serving_path).resolve()

    output_path = Path(args.output_dir)
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()

    print(f"Assets directory: {assets_path}")
    print(f"Serving directory: {serving_path}")
    print(f"Output directory: {output_path}")
    print(f"Layout strategy: {args.layout}")

    # Validate inputs
    required_assets = ["backbone.ckpt", "global_means.npy", "global_stds.npy"]
    for asset in required_assets:
        p = assets_path / asset
        if not p.exists():
            print(f"ERROR: Missing required asset: {p}")
            return 1

    required_code = ["inference.py", "requirements.txt"]
    for code_file in required_code:
        p = serving_path / code_file
        if not p.exists():
            print(f"ERROR: Missing required serving file: {p}")
            return 1

    output_path.mkdir(parents=True, exist_ok=True)
    tar_path = output_path / "model.tar.gz"

    with tempfile.TemporaryDirectory() as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)

        # Copy model assets
        for asset in required_assets:
            # We will add files to the root of the tarball
            pass

        # Create tar file
        with tarfile.open(tar_path, "w:gz") as tar:
            # Add assets to root
            for asset in required_assets:
                tar.add(assets_path / asset, arcname=asset)

            if args.layout == "self-contained":
                # Add code/ directory and files
                for code_file in required_code:
                    tar.add(serving_path / code_file, arcname=f"code/{code_file}")
                print("Packaged in self-contained mode (includes code/ folder inside model.tar.gz).")
            else:
                print("Packaged in separate-source mode (model.tar.gz contains assets only).")

    print(f"SUCCESS: Package created at {tar_path}")
    
    # Verify contents
    with tarfile.open(tar_path, "r:gz") as tar:
        names = tar.getnames()
        print("Tarball contents:")
        for name in sorted(names):
            print(f"  - {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
