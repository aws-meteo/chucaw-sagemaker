#!/usr/bin/env python3
import json
import sys
from pathlib import Path


REQUIRED_KEYS = {"t2m", "lat_grid", "lon_grid", "units", "source"}


def main():
    repo_root = Path(__file__).resolve().parents[1]
    model_path = repo_root / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found: {model_path}")

    sys.path.insert(0, str(repo_root))
    from inference.inference import input_fn, model_fn, output_fn, predict_fn  # noqa: E402

    request_json = json.dumps({"lat": -33.5, "lon": -70.6})
    model = model_fn(str(repo_root))
    parsed_input = input_fn(request_json, content_type="application/json")
    prediction = predict_fn(parsed_input, model)
    response_body, content_type = output_fn(prediction, accept="application/json")

    if content_type != "application/json":
        raise ValueError(f"Unexpected content_type: {content_type}")

    response = json.loads(response_body)
    missing = sorted(REQUIRED_KEYS - set(response.keys()))
    if missing:
        raise ValueError(f"Local smoke response missing keys: {', '.join(missing)}")

    if not isinstance(response["t2m"], (float, int)):
        raise TypeError("response['t2m'] must be numeric")
    if not isinstance(response["lat_grid"], (float, int)):
        raise TypeError("response['lat_grid'] must be numeric")
    if not isinstance(response["lon_grid"], (float, int)):
        raise TypeError("response['lon_grid'] must be numeric")
    if response["units"] != "K":
        raise ValueError("response['units'] must be 'K'")
    if response["source"] != "ECMWF SCDA":
        raise ValueError("response['source'] must be 'ECMWF SCDA'")

    print("Local smoke test passed.")
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
