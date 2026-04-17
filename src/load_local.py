#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"variable", "isobaricInhPa", "latitude", "longitude", "value"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extract local ECMWF parquet snapshot for trivial baseline")
    parser.add_argument("--parquet", required=True, help="Path to local parquet file")
    return parser.parse_args()


def main():
    args = parse_args()
    parquet_path = Path(args.parquet)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Local parquet path not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(
            "Missing required columns in parquet: "
            + ", ".join(missing_columns)
            + ". Required: "
            + ", ".join(sorted(REQUIRED_COLUMNS))
        )

    print(f"Rows before filter: {len(df)}")

    df = df[(df["variable"] == "t") & (df["isobaricInhPa"] == 1000.0)]
    df = df[["latitude", "longitude", "value"]].dropna()

    print(f"Rows after filter: {len(df)}")
    if df.empty:
        raise ValueError(
            "Filtered dataset is empty after applying variable='t' and isobaricInhPa=1000.0"
        )

    out_path = Path(__file__).resolve().parents[1] / "data" / "t2m_snapshot.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Snapshot written to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
