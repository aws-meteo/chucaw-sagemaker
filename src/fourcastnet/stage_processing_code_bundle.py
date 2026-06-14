#!/usr/bin/env python3
"""Stage local processing code folder for S3 upload."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage FCN processing code bundle")
    parser.add_argument(
        "--staging-dir",
        default="artifacts/fourcastnet/processing_code",
        help="Output directory to upload as S3 prefix",
    )
    parser.add_argument(
        "--fourcastnet-repo",
        default="",
        help="Optional local FourCastNet repo path to include as FourCastNet/",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite staging dir if it already exists",
    )
    return parser.parse_args()


def resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def main() -> int:
    args = parse_args()
    sm_root = Path(__file__).resolve().parents[2]
    entrypoint = sm_root / "src" / "fourcastnet" / "processing_entrypoint.py"
    if not entrypoint.exists():
        raise FileNotFoundError(f"Missing processing entrypoint: {entrypoint}")

    staging_dir = resolve(args.staging_dir, sm_root)
    if staging_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Staging dir exists: {staging_dir}. Use --overwrite to recreate.")
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(entrypoint, staging_dir / "processing_entrypoint.py")
    included_repo = ""
    if args.fourcastnet_repo.strip():
        repo_path = resolve(args.fourcastnet_repo, sm_root)
        if not repo_path.exists():
            raise FileNotFoundError(f"FourCastNet repo path does not exist: {repo_path}")
        target = staging_dir / "FourCastNet"
        shutil.copytree(repo_path, target)
        included_repo = str(target)

    manifest = {
        "created_at_epoch": int(time.time()),
        "staging_dir": str(staging_dir),
        "files": sorted(str(p.relative_to(staging_dir)) for p in staging_dir.rglob("*") if p.is_file()),
        "included_fourcastnet_repo": included_repo,
    }
    (staging_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Staged processing code at: {staging_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

