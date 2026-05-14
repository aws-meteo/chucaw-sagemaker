import pandas as pd

from src.prediction_schema import PARTITION_COLUMNS, REQUIRED_PREDICTION_COLUMNS, build_prediction_frame


def test_prediction_schema_contains_required_columns():
    df = pd.DataFrame(
        [
            {"latitude": -33.5, "longitude": -70.6, "run": "18z", "value": 285.0},
            {"latitude": -34.0, "longitude": -71.0, "run": "18z", "value": 286.0},
        ]
    )
    out = build_prediction_frame(
        source_df=df,
        predicted_values=[285.2, 286.4],
        model_name="knn_baseline",
        model_version="v1",
        method="nearest_grid_lookup",
        target_variable="t",
        predicted_units="K",
        source_database="silverlayer",
        source_table="forecast_parquet",
        source_year="2026",
        source_month="04",
        source_day="09",
        source_hour="18",
    )
    for column in REQUIRED_PREDICTION_COLUMNS:
        assert column in out.columns
    for column in PARTITION_COLUMNS:
        assert column in out.columns

    assert out["feature_snapshot_json"].map(type).eq(str).all()
    assert out["prediction_timestamp_utc"].map(type).eq(str).all()
