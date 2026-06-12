#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    experimental_script = (
        repo_root
        / "experimental"
        / "realtime_endpoint_dangerous"
        / "prepare_async_endpoint_payloads.py"
    )
    print("ERROR: Real-time endpoint payload generation is disabled by default.", file=sys.stderr)
    print("Real-time endpoints are disabled for FourCastNet by default. Use Batch Transform.", file=sys.stderr)
    print(f"Dangerous experimental path: {experimental_script}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
