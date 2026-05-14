import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.predictors import load_predictor_from_model_dir


def model_fn(model_dir):
    return load_predictor_from_model_dir(Path(model_dir))


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

    lat = float(input_data["lat"])
    lon = float(input_data["lon"])
    pred = model.predict_row(lat, lon)

    return {
        "t2m": float(pred["predicted_value"]),
        "lat_grid": float(pred["lat_grid"]),
        "lon_grid": float(pred["lon_grid"]),
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
