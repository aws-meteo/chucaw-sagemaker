#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> int:
    print("ERROR: async endpoint autoscaling helpers are disabled in default repo paths.", file=sys.stderr)
    print(
        "Endpoint autoscaling still depends on SageMaker endpoint hosting. Use Batch Transform.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
