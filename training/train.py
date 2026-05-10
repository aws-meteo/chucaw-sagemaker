import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

EXPECTED_COLUMNS = ["latitude", "longitude", "value"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_neighbors", type=int, default=4)
    parser.add_argument("--weights", type=str, default="distance")
    return parser.parse_args()


def find_training_csv(train_dir: Path) -> Path:
    candidate = train_dir / "train.csv"
    if candidate.exists():
        return candidate

    csv_files = sorted(train_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in training channel: {train_dir}")
    return csv_files[0]


def main():
    args = parse_args()

    train_dir = Path(os.environ["SM_CHANNEL_TRAIN"])
    model_dir = Path(os.environ["SM_MODEL_DIR"])
    output_dir = Path(os.environ["SM_OUTPUT_DIR"])

    train_csv = find_training_csv(train_dir)
    df = pd.read_csv(train_csv)

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Training CSV missing required columns: "
            + ", ".join(missing)
            + ". Expected: latitude, longitude, value"
        )

    df = df[EXPECTED_COLUMNS].dropna()
    if df.empty:
        raise ValueError("Training CSV is empty after validation")

    X = df[["latitude", "longitude"]].to_numpy(dtype=np.float32)
    y = df["value"].to_numpy(dtype=np.float32)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    estimator = KNeighborsRegressor(
        n_neighbors=args.n_neighbors,
        weights=args.weights,
    )
    estimator.fit(X_train, y_train)

    y_pred = estimator.predict(X_valid)
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))

    artifact = {
        "artifact_type": "sklearn_regressor_v1",
        "feature_columns": ["latitude", "longitude"],
        "target_column": "value",
        "estimator": estimator,
        "grid": X,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifact, model_dir / "model.joblib")

    metrics = {
        "row_count": int(len(df)),
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "rmse": rmse,
        "n_neighbors": int(args.n_neighbors),
        "weights": args.weights,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()