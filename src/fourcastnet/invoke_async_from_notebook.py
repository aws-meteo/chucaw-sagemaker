#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> int:
    print("ERROR: async endpoint invocation helpers are disabled in default repo paths.", file=sys.stderr)
    print(
        "Real-time/async endpoints are disabled for FourCastNet by default. Use Batch Transform.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
