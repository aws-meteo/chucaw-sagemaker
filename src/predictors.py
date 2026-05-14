from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Protocol

import joblib
import numpy as np


LEGACY_ARTIFACT_TYPE = "legacy_lookup_v1"
SKLEARN_ARTIFACT_TYPE = "sklearn_regressor_v1"


def _coerce_grid(grid: Any) -> np.ndarray:
    grid_arr = np.asarray(grid, dtype=np.float32)
    if grid_arr.ndim != 2 or grid_arr.shape[1] != 2:
        raise ValueError("Invalid model format: 'grid' must be 2D shape (n, 2)")
    if len(grid_arr) == 0:
        raise ValueError("Invalid model format: empty 'grid'")
    return grid_arr


def normalize_artifact(raw_model: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_model, dict):
        raise ValueError("Invalid model format: expected dict artifact")

    artifact_type = raw_model.get("artifact_type")
    if "grid" in raw_model and "t2m" in raw_model and artifact_type in (None, LEGACY_ARTIFACT_TYPE):
        grid = _coerce_grid(raw_model["grid"])
        t2m = np.asarray(raw_model["t2m"], dtype=np.float32).reshape(-1)
        if len(t2m) == 0:
            raise ValueError("Invalid model format: empty 't2m'")
        if len(t2m) != len(grid):
            raise ValueError("Invalid model format: length mismatch grid vs t2m")
        return {"artifact_type": LEGACY_ARTIFACT_TYPE, "grid": grid, "t2m": t2m}

    if artifact_type == SKLEARN_ARTIFACT_TYPE:
        if "estimator" not in raw_model or "grid" not in raw_model:
            raise ValueError("Invalid sklearn model: missing estimator or grid")
        return {
            "artifact_type": SKLEARN_ARTIFACT_TYPE,
            "estimator": raw_model["estimator"],
            "grid": _coerce_grid(raw_model["grid"]),
        }

    raise ValueError(
        "Unsupported model artifact format. "
        f"artifact_type={artifact_type!r}, keys={sorted(raw_model.keys())}"
    )


class BasePredictor(Protocol):
    model_name: str
    model_version: str
    method: str
    predicted_units: str
    target_variable: str

    def predict_row(self, latitude: float, longitude: float) -> Dict[str, float]:
        ...


@dataclass
class KNNPredictor:
    model: Dict[str, Any]
    model_name: str = "knn_baseline"
    model_version: str = "v1"
    method: str = "nearest_grid_lookup"
    predicted_units: str = "K"
    target_variable: str = "t"

    def __post_init__(self) -> None:
        self.model = normalize_artifact(self.model)
        self.grid = self.model["grid"]
        self.artifact_type = self.model.get("artifact_type", LEGACY_ARTIFACT_TYPE)

    def _nearest_grid_idx(self, latitude: float, longitude: float) -> int:
        dists = np.sqrt((self.grid[:, 0] - latitude) ** 2 + (self.grid[:, 1] - longitude) ** 2)
        return int(np.argmin(dists))

    def predict_row(self, latitude: float, longitude: float) -> Dict[str, float]:
        idx = self._nearest_grid_idx(latitude, longitude)

        if self.artifact_type == LEGACY_ARTIFACT_TYPE:
            t2m = self.model.get("t2m")
            if t2m is None:
                raise ValueError("Invalid legacy model: missing 't2m'")
            value = float(t2m[idx])
        elif self.artifact_type == SKLEARN_ARTIFACT_TYPE:
            estimator = self.model.get("estimator")
            if estimator is None:
                raise ValueError("Invalid sklearn model: missing 'estimator'")
            value = float(estimator.predict(np.array([[latitude, longitude]], dtype=np.float32))[0])
        else:
            raise ValueError(f"Unsupported model artifact type: {self.artifact_type!r}")

        return {
            "predicted_value": value,
            "lat_grid": float(self.grid[idx, 0]),
            "lon_grid": float(self.grid[idx, 1]),
        }


@dataclass
class MockPredictor:
    model_name: str = "mock_predictor"
    model_version: str = "v0"
    method: str = "mock_passthrough"
    predicted_units: str = "K"
    target_variable: str = "t"

    def predict_row(self, latitude: float, longitude: float) -> Dict[str, float]:
        value = float(273.15 + ((abs(latitude) + abs(longitude)) % 25))
        return {
            "predicted_value": value,
            "lat_grid": float(latitude),
            "lon_grid": float(longitude),
        }


def load_predictor_from_model_dir(
    model_dir: Path,
    model_name: str = "knn_baseline",
    model_version: str = "v1",
) -> BasePredictor:
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found at: {model_path}")
    raw_model = joblib.load(model_path)
    return KNNPredictor(raw_model, model_name=model_name, model_version=model_version)
