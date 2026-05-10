#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["latitude", "longitude", "value"]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training CSV from materialized snapshot")
    parser.add_argument("--input", required=True, help="Input CSV path, usually data/t2m_snapshot.csv")
    parser.add_argument("--output", required=True, help="Output CSV path, usually data/training/train.csv")
    parser.add_argument("--dataset-name", required=True, help="Logical dataset identifier")
    return parser.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = output_path.with_name("metadata.json")

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV missing required columns: "
            + ", ".join(missing)
            + ". Expected: latitude, longitude, value"
        )

    df = df[EXPECTED_COLUMNS].dropna()
    if df.empty:
        raise ValueError("Prepared training dataset is empty after validation")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    metadata = {
        "dataset_name": args.dataset_name,
        "row_count": int(len(df)),
        "columns": EXPECTED_COLUMNS,
        "source_csv": str(input_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Training CSV written to: {output_path}")
    print(f"Metadata written to: {metadata_path}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)