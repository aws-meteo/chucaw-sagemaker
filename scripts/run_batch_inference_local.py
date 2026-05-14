#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prediction_schema import build_prediction_frame
from src.predictors import MockPredictor, load_predictor_from_model_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local batch downscaling inference and write Parquet output")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet path")
    parser.add_argument("--output", required=True, help="Output Parquet path")
    parser.add_argument("--model-dir", default=str(REPO_ROOT), help="Directory containing model.joblib")
    parser.add_argument("--model-name", default="knn_baseline")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--source-database", default="silverlayer")
    parser.add_argument("--source-table", default="forecast_parquet")
    parser.add_argument("--source-year", default="2026")
    parser.add_argument("--source-month", default="04")
    parser.add_argument("--source-day", default="09")
    parser.add_argument("--source-hour", default="18")
    return parser.parse_args()


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError("Input must be CSV or Parquet")

    required = {"latitude", "longitude"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input missing required columns: {', '.join(missing)}")
    return df


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    source_df = read_input(input_path).copy()
    model_dir = Path(args.model_dir)
    try:
        predictor = load_predictor_from_model_dir(
            model_dir, model_name=args.model_name, model_version=args.model_version
        )
    except FileNotFoundError:
        predictor = MockPredictor(model_name="mock_predictor", model_version="v0")

    predicted_values = []
    for row in source_df.itertuples(index=False):
        pred = predictor.predict_row(float(row.latitude), float(row.longitude))
        predicted_values.append(float(pred["predicted_value"]))

    prediction_df = build_prediction_frame(
        source_df=source_df,
        predicted_values=predicted_values,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        method=predictor.method,
        target_variable=predictor.target_variable,
        predicted_units=predictor.predicted_units,
        source_database=args.source_database,
        source_table=args.source_table,
        source_year=str(args.source_year),
        source_month=str(args.source_month),
        source_day=str(args.source_day),
        source_hour=str(args.source_hour),
    )
    prediction_df.to_parquet(output_path, index=False)

    print(f"rows={len(prediction_df)}")
    print(f"output={output_path.resolve()}")
    print(f"model_name={predictor.model_name}")
    print(f"model_version={predictor.model_version}")


if __name__ == "__main__":
    main()
