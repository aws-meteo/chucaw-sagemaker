#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ["latitude", "longitude", "value"]


def parse_args():
    parser = argparse.ArgumentParser(description="Build trivial nearest-neighbor lookup model artifact")
    parser.add_argument("--input", required=True, help="CSV path with latitude,longitude,value")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    missing = [column for column in EXPECTED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Input CSV missing required columns: "
            + ", ".join(missing)
            + ". Expected columns: latitude, longitude, value"
        )

    df = df[EXPECTED_COLUMNS].dropna()
    if df.empty:
        raise ValueError("Input CSV has zero usable rows after selecting expected columns and dropping nulls")

    model = {
        "grid": df[["latitude", "longitude"]].to_numpy(dtype=np.float32),
        "t2m": df["value"].to_numpy(dtype=np.float32),
    }

    if model["grid"].shape[0] != model["t2m"].shape[0]:
        raise ValueError("Model arrays are misaligned: grid row count differs from t2m row count")

    out_path = Path(__file__).resolve().parents[1] / "model.joblib"
    joblib.dump(model, out_path)
    print(f"Model grid shape: {model['grid'].shape}")
    print(f"Model values shape: {model['t2m'].shape}")
    print("Serialization alignment: local artifact targets SageMaker scikit-learn container family (1.2-1)")
    print(f"Model artifact written to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
