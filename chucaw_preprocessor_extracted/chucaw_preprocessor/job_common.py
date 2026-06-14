from datetime import datetime
from pathlib import Path


def normalize_run(run: str) -> str:
    run = run.strip().lower()
    if run.endswith("z"):
        return run
    return f"{run}z"


def parse_date(date_value: str) -> datetime:
    return datetime.strptime(date_value, "%Y%m%d")


def partition_prefix(base_prefix: str, date_str: str, run: str) -> str:
    parsed = parse_date(date_str)
    base = Path(base_prefix.strip("/"))
    return str(
        base
        / f"year={parsed.year:04d}"
        / f"month={parsed.month:02d}"
        / f"day={parsed.day:02d}"
        / f"hour={normalize_run(run)}"
    ).replace("\\", "/")
