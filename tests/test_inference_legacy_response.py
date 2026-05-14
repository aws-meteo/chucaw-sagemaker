import json

from inference.inference import output_fn, predict_fn
from src.predictors import MockPredictor


def test_predict_fn_preserves_legacy_response_keys():
    prediction = predict_fn({"lat": -33.5, "lon": -70.6}, MockPredictor())
    assert set(prediction.keys()) == {"t2m", "lat_grid", "lon_grid"}


def test_output_fn_preserves_legacy_json_contract():
    body, content_type = output_fn({"t2m": 285.0, "lat_grid": -33.5, "lon_grid": -70.6})
    payload = json.loads(body)
    assert content_type == "application/json"
    assert {"t2m", "lat_grid", "lon_grid", "units", "source"}.issubset(set(payload.keys()))
