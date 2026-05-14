from src.predictors import KNNPredictor


def test_knn_predictor_contract_returns_expected_keys():
    model = {
        "grid": [[-33.5, -70.6], [-34.0, -71.0]],
        "t2m": [285.0, 286.0],
    }
    predictor = KNNPredictor(model=model, model_name="knn_baseline", model_version="v1")
    row = predictor.predict_row(-33.51, -70.59)
    assert set(row.keys()) == {"predicted_value", "lat_grid", "lon_grid"}
