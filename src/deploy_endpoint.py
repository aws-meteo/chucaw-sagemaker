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
    print("ERROR: Real-time endpoint deployment is experimental and disabled here.", file=sys.stderr)
    print(
        "Real-time endpoints are disabled for FourCastNet by default. Use Batch Transform.",
        file=sys.stderr,
    )
    print(
        "If you explicitly accept cost risk, use the quarantined script with both flags:",
        file=sys.stderr,
    )
    print(
        f"python {experimental_script} --allow-realtime-endpoint --i-understand-this-can-cost-money --endpoint-name <name> --instance-type <type>",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
