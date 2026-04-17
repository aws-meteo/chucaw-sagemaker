import json
from pathlib import Path

import joblib
import numpy as np


def model_fn(model_dir):
    model_path = Path(model_dir) / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found at: {model_path}")

    model = joblib.load(model_path)
    if not isinstance(model, dict):
        raise ValueError("Invalid model format: expected dict with keys 'grid' and 't2m'")
    if "grid" not in model or "t2m" not in model:
        raise ValueError("Invalid model format: missing 'grid' or 't2m'")
    return model


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

    grid = model.get("grid")
    t2m = model.get("t2m")
    if grid is None or t2m is None:
        raise ValueError("Invalid model: missing 'grid' or 't2m'")
    if len(grid) == 0 or len(t2m) == 0:
        raise ValueError("Invalid model: empty lookup arrays")

    lat = float(input_data["lat"])
    lon = float(input_data["lon"])

    dists = np.sqrt((grid[:, 0] - lat) ** 2 + (grid[:, 1] - lon) ** 2)
    idx = int(np.argmin(dists))

    return {
        "t2m": float(t2m[idx]),
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
