# Downscaling Inference to QuickSight via Athena

## Why Athena/S3 output, not direct SageMaker to QuickSight

QuickSight reads datasets from query/storage systems (Athena, S3, Redshift), not from live SageMaker endpoint payloads.  
So pipeline should be:

1. Build inputs from `silverlayer.forecast_parquet` (partition-filtered).
2. Run local or SageMaker inference.
3. Write prediction rows to S3 as Parquet.
4. Expose S3 prefix with Glue/Athena external table.
5. Connect QuickSight dataset to Athena table.

This keeps inference decoupled, replayable, and dashboard-friendly.
QuickSight does not directly consume SageMaker model artifacts or endpoint responses.
SageMaker real-time endpoint is optional for demos, and not required for scheduled forecast/downscaling pipelines.
Recommended production path is batch/processing output to S3, then Glue/Athena, then QuickSight.

## KNN baseline in current pipeline

Current baseline predictor is nearest-grid lookup/KNN-style artifact (`model.joblib`).  
New contract uses predictor abstraction:

- `BasePredictor` protocol in `src/predictors.py`
- `KNNPredictor` implementation for current artifact
- `MockPredictor` fallback for local tests when model missing

Future models can replace only predictor implementation while output schema stays fixed.

## Canonical prediction output schema

Implemented in `src/prediction_schema.py` (`build_prediction_frame`):

- `prediction_id`
- `model_name`
- `model_version`
- `prediction_timestamp_utc`
- `source_database`
- `source_table`
- `source_year`
- `source_month`
- `source_day`
- `source_hour`
- `run`
- `latitude`
- `longitude`
- `target_variable`
- `predicted_value`
- `predicted_units`
- `method`
- optional: `observed_value`, `error`, `feature_snapshot_json`

## Local batch inference

Run:

```bash
python scripts/run_batch_inference_local.py \
  --input data/20260409180000-6h-scda-fc.parquet \
  --output artifacts/predictions_local.parquet \
  --source-year 2026 --source-month 04 --source-day 09 --source-hour 18
```

Input requirements:

- Must contain `latitude`, `longitude`
- Can include `run` and `value` (optional)
- CSV and Parquet both supported

Script does not require live SageMaker endpoint.

## Athena table for QuickSight

Use SQL file:

- `sql/create_quicksight_predictions_table.sql`

Creates external table:

- `silverlayer.downscaling_predictions`
- location: configurable, set in SQL LOCATION (`s3://<your-bucket>/<your-prefix>/downscaling/predictions/`)
- partition suggestion: `year, month, day, hour, model_name_partition, model_version_partition`
- partitioned S3 path convention:
  `.../year=YYYY/month=MM/day=DD/hour=HH/model_name_partition=<name>/model_version_partition=<ver>/`

After writing new partition folders, run:

```sql
MSCK REPAIR TABLE silverlayer.downscaling_predictions;
```

Alternative for strict partition registration:

```sql
ALTER TABLE silverlayer.downscaling_predictions ADD IF NOT EXISTS PARTITION (...);
```

## How to swap KNN later

1. Add new predictor class implementing `predict_row(latitude, longitude)`.
2. Return same metadata fields (`model_name`, `model_version`, `method`, `predicted_units`, `target_variable`).
3. Keep `build_prediction_frame` unchanged.
4. Re-run local batch inference and tests.

No QuickSight/Athena schema change needed if contract preserved.
