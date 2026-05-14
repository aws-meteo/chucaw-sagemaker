from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Dict, Iterable, List

import pandas as pd


REQUIRED_PREDICTION_COLUMNS = [
    "prediction_id",
    "model_name",
    "model_version",
    "prediction_timestamp_utc",
    "source_database",
    "source_table",
    "source_year",
    "source_month",
    "source_day",
    "source_hour",
    "run",
    "latitude",
    "longitude",
    "target_variable",
    "predicted_value",
    "predicted_units",
    "method",
]

PARTITION_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "model_name_partition",
    "model_version_partition",
]


def _prediction_id(parts: Iterable[Any]) -> str:
    material = "|".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_prediction_frame(
    source_df: pd.DataFrame,
    predicted_values: List[float],
    *,
    model_name: str,
    model_version: str,
    method: str,
    target_variable: str,
    predicted_units: str,
    source_database: str,
    source_table: str,
    source_year: str,
    source_month: str,
    source_day: str,
    source_hour: str,
) -> pd.DataFrame:
    if len(source_df) != len(predicted_values):
        raise ValueError("Length mismatch source_df vs predicted_values")

    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []
    for row, pred in zip(source_df.to_dict(orient="records"), predicted_values):
        run_value = row.get("run", "")
        record: Dict[str, Any] = {
            "model_name": model_name,
            "model_version": model_version,
            "prediction_timestamp_utc": timestamp,
            "source_database": source_database,
            "source_table": source_table,
            "source_year": source_year,
            "source_month": source_month,
            "source_day": source_day,
            "source_hour": source_hour,
            "run": run_value,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "target_variable": target_variable,
            "predicted_value": float(pred),
            "predicted_units": predicted_units,
            "method": method,
            # Keep explicit partition columns in the parquet body for simpler Athena validation/debug.
            "year": source_year,
            "month": source_month,
            "day": source_day,
            "hour": source_hour,
            "model_name_partition": model_name,
            "model_version_partition": model_version,
        }
        record["prediction_id"] = _prediction_id(
            [
                model_name,
                model_version,
                source_year,
                source_month,
                source_day,
                source_hour,
                run_value,
                record["latitude"],
                record["longitude"],
                record["target_variable"],
            ]
        )

        if "value" in row and row["value"] is not None:
            observed_value = float(row["value"])
            record["observed_value"] = observed_value
            record["error"] = float(pred) - observed_value
        else:
            record["observed_value"] = None
            record["error"] = None

        snapshot = {"latitude": row.get("latitude"), "longitude": row.get("longitude"), "run": run_value}
        record["feature_snapshot_json"] = json.dumps(snapshot)
        rows.append(record)

    df = pd.DataFrame(rows)
    ordered_columns = REQUIRED_PREDICTION_COLUMNS + ["observed_value", "error", "feature_snapshot_json"] + PARTITION_COLUMNS
    return df[ordered_columns]
