import json
from pathlib import Path

import joblib
import numpy as np


LEGACY_ARTIFACT_TYPE = "legacy_lookup_v1"
SKLEARN_ARTIFACT_TYPE = "sklearn_regressor_v1"


def _coerce_grid(grid):
    grid_arr = np.asarray(grid, dtype=np.float32)
    if grid_arr.ndim != 2 or grid_arr.shape[1] != 2:
        raise ValueError("Invalid model format: 'grid' must be a 2D array with shape (n, 2)")
    if len(grid_arr) == 0:
        raise ValueError("Invalid model format: empty 'grid'")
    return grid_arr


def _normalize_artifact(raw_model):
    if not isinstance(raw_model, dict):
        raise ValueError("Invalid model format: expected dict artifact")

    artifact_type = raw_model.get("artifact_type")

    # Legacy lookup artifact: {"grid": ..., "t2m": ...}
    if "grid" in raw_model and "t2m" in raw_model and artifact_type in (None, LEGACY_ARTIFACT_TYPE):
        grid = _coerce_grid(raw_model["grid"])
        t2m = np.asarray(raw_model["t2m"], dtype=np.float32).reshape(-1)
        if len(t2m) == 0:
            raise ValueError("Invalid model format: empty 't2m'")
        if len(t2m) != len(grid):
            raise ValueError("Invalid model format: length mismatch between 'grid' and 't2m'")
        return {"artifact_type": LEGACY_ARTIFACT_TYPE, "grid": grid, "t2m": t2m}

    # New sklearn artifact from training/train.py
    if artifact_type == SKLEARN_ARTIFACT_TYPE:
        if "estimator" not in raw_model:
            raise ValueError("Invalid model format: missing 'estimator' for sklearn artifact")
        if "grid" not in raw_model:
            raise ValueError("Invalid model format: missing 'grid' for sklearn artifact")
        grid = _coerce_grid(raw_model["grid"])
        return {
            "artifact_type": SKLEARN_ARTIFACT_TYPE,
            "estimator": raw_model["estimator"],
            "grid": grid,
        }

    raise ValueError(
        "Unsupported model artifact format. "
        f"artifact_type={artifact_type!r}, keys={sorted(raw_model.keys())}"
    )


def _nearest_grid_idx(grid, lat, lon):
    dists = np.sqrt((grid[:, 0] - lat) ** 2 + (grid[:, 1] - lon) ** 2)
    return int(np.argmin(dists))


def model_fn(model_dir):
    model_path = Path(model_dir) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found at: {model_path}")

    raw_model = joblib.load(model_path)
    return _normalize_artifact(raw_model)


def input_fn(request_body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Expected application/json")

    try:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        payload = json.loads(request_body)
    except Exception as exc:
        raise ValueError(f"Malformed JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Invalid payload: expected JSON object")

    if "lat" not in payload or "lon" not in payload:
        raise ValueError("Invalid payload: required keys are 'lat' and 'lon'")

    try:
        lat = float(payload["lat"])
        lon = float(payload["lon"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid payload: 'lat' and 'lon' must be numeric") from exc

    return {"lat": lat, "lon": lon}


def predict_fn(input_data, model):
    if not isinstance(input_data, dict):
        raise ValueError("Invalid input_data: expected dict")
    if "lat" not in input_data or "lon" not in input_data:
        raise ValueError("Invalid input_data: required keys are 'lat' and 'lon'")

    if not isinstance(model, dict):
        raise ValueError("Invalid model: expected dict")
    lat = float(input_data["lat"])
    lon = float(input_data["lon"])
    grid = model.get("grid")
    if grid is None:
        raise ValueError("Invalid model: missing 'grid'")

    idx = _nearest_grid_idx(grid, lat, lon)
    artifact_type = model.get("artifact_type", LEGACY_ARTIFACT_TYPE)

    if artifact_type == LEGACY_ARTIFACT_TYPE:
        t2m = model.get("t2m")
        if t2m is None:
            raise ValueError("Invalid legacy model: missing 't2m'")
        value = float(t2m[idx])
    elif artifact_type == SKLEARN_ARTIFACT_TYPE:
        estimator = model.get("estimator")
        if estimator is None:
            raise ValueError("Invalid sklearn model: missing 'estimator'")
        value = float(estimator.predict(np.array([[lat, lon]], dtype=np.float32))[0])
    else:
        raise ValueError(f"Unsupported model artifact type: {artifact_type!r}")

    return {
        "t2m": value,
        "lat_grid": float(grid[idx, 0]),
        "lon_grid": float(grid[idx, 1]),
    }


def output_fn(prediction, accept="application/json"):
    if accept != "application/json":
        raise ValueError(f"Unsupported accept type: {accept}. Expected application/json")

    response = {
        "t2m": float(prediction["t2m"]),
        "lat_grid": float(prediction["lat_grid"]),
        "lon_grid": float(prediction["lon_grid"]),
        "units": "K",
        "source": "ECMWF SCDA",
    }
    return json.dumps(response), accept
