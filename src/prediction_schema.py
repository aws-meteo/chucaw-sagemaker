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
    """
    Genera un identificador único determinista para una predicción.

    Combina las partes proporcionadas en una cadena y calcula su hash SHA-256.

    Parameters
    ----------
    parts : Iterable[Any]
        Elementos que componen la identidad única de la predicción.

    Returns
    -------
    str
        Hash SHA-256 en formato hexadecimal.
    """
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
    """
    Construye un dataframe de predicciones siguiendo el esquema estandarizado de SbnAI.

    Enriquece los resultados con metadatos de auditoría (timestamp, ID único,
    snapshot de características) y columnas de partición para Athena.

    Parameters
    ----------
    source_df : pd.DataFrame
        Dataframe con las características de entrada.
    predicted_values : List[float]
        Lista de valores predichos correspondientes a cada fila de source_df.
    model_name : str
        Nombre del modelo utilizado.
    model_version : str
        Versión del modelo.
    method : str
        Método de predicción.
    target_variable : str
        Variable objetivo (ej. 't').
    predicted_units : str
        Unidades del valor predicho (ej. 'K').
    source_database : str
        Base de datos de origen de las características.
    source_table : str
        Tabla de origen.
    source_year : str
        Año de los datos de origen.
    source_month : str
        Mes.
    source_day : str
        Día.
    source_hour : str
        Hora.

    Returns
    -------
    pd.DataFrame
        Dataframe estructurado y validado listo para su exportación.

    Raises
    ------
    ValueError
        Si la longitud de los valores predichos no coincide con el dataframe de origen.
    """
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
