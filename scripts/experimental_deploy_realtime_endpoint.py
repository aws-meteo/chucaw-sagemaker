#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    experimental_script = (
        repo_root
        / "experimental"
        / "realtime_endpoint_dangerous"
        / "experimental_deploy_realtime_endpoint.py"
    )
    print("ERROR: Real-time endpoints are disabled for FourCastNet by default.", file=sys.stderr)
    print("Use Batch Transform. This script is quarantined.", file=sys.stderr)
    print(f"Dangerous experimental path: {experimental_script}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
