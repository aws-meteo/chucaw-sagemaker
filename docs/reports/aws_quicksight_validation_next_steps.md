# AWS QuickSight Validation Next Steps

## Prerequisites
- Use explicit interpreter for local commands:
  `C:\ProgramData\miniconda3\envs\chucaw-sagemaker\python.exe`
- Known non-blocker: Windows `python` alias may fail (`WindowsApps\python.exe`), so avoid it.

## 1) Generate Local Prediction Fixture Output
Run from repo root:

```powershell
& 'C:\ProgramData\miniconda3\envs\chucaw-sagemaker\python.exe' scripts/run_batch_inference_local.py `
  --input tests/fixtures/forecast_parquet_like_t1000.csv `
  --output .tmp/predictions_fixture.parquet `
  --source-year 2026 --source-month 04 --source-day 09 --source-hour 18
```

## 2) Upload Parquet to S3 (Partitioned Prefix)
Expected partitioned prefix:

` s3://<your-bucket>/<your-prefix>/downscaling/predictions/year=2026/month=04/day=09/hour=18/model_name_partition=knn_baseline/model_version_partition=v1/ `

Exact upload command:

```bash
aws s3 cp .tmp/predictions_fixture.parquet \
  s3://<your-bucket>/<your-prefix>/downscaling/predictions/year=2026/month=04/day=09/hour=18/model_name_partition=knn_baseline/model_version_partition=v1/predictions_fixture.parquet
```

## 3) Run Athena DDL
DDL file to run in Athena query editor:

- `sql/create_quicksight_predictions_table.sql`

Before running, set the table `LOCATION` in that file to your real bucket/prefix root:

` s3://<your-bucket>/<your-prefix>/downscaling/predictions/ `

## 4) Refresh Partitions
Run:

```sql
MSCK REPAIR TABLE silverlayer.downscaling_predictions;
```

## 5) Validate Data in Athena
Run:

```sql
SELECT
  latitude,
  longitude,
  target_variable,
  predicted_value,
  prediction_timestamp_utc,
  year,
  month,
  day,
  hour,
  model_name_partition,
  model_version_partition
FROM silverlayer.downscaling_predictions
WHERE year='2026'
  AND month='04'
  AND day='09'
  AND hour='18'
  AND model_name_partition='knn_baseline'
  AND model_version_partition='v1'
LIMIT 50;
```

## 6) Create QuickSight Dataset
1. In QuickSight, create a new dataset.
2. Choose Athena as the source.
3. Select database `silverlayer` and table `downscaling_predictions`.
4. Import to SPICE or use direct query.
5. Validate fields exist: `latitude`, `longitude`, `predicted_value`, `target_variable`, `prediction_timestamp_utc`.
6. Build first visual (map or scatter) using latitude/longitude and color/size by `predicted_value`.
