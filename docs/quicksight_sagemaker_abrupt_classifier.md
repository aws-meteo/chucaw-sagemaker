# QuickSight SageMaker Abrupt Classifier (CSV Batch)

## Purpose
This prototype adds a SageMaker Batch Transform compatible classifier for QuickSight dataset augmentation.
It is classification-only and CSV-in/CSV-out. It is not a real-time JSON endpoint.

## QuickSight Contract
- Input columns (no header): `lat,lon,t2m`
- Output columns (no header): `abrupt_temp_change_label,abrupt_temp_change_score`
- Row order is preserved with one output row per input row.
- Supported input content types: `text/csv`, `CSV`, `application/csv`.
- Output is CSV without header.

## Deterministic Prototype Logic
Parameters live in `src/quicksight_abrupt_classifier/model_config.json`:
- `mean_t2m` default `285.0`
- `std_t2m` default `8.0`
- `threshold` default `2.0`

Computation:
- `score = abs(t2m - mean_t2m) / std_t2m`
- `label = 1 if score >= threshold else 0`

`lat` and `lon` are parsed and validated:
- `lat` in `[-90, 90]`
- `lon` in `[-180, 180]`

## Schema
QuickSight schema file:
- `schemas/quicksight_abrupt_classifier_schema.json`
- `schemas/quicksight_abrupt_classifier_schema_textcsv.json`

Recommendation: try `text/csv` schema first. Keep `CSV` schema as fallback because QuickSight UI behavior in this project previously accepted `CSV`.

## CSV Example
Input (`examples/quicksight_abrupt_input.csv`):
```csv
-33.4500,-70.6600,285.4
-34.0000,-71.0000,305.0
45.0000,7.0000,280.0
```

Output example:
```csv
0,0.050000
1,2.500000
0,0.625000
```

## Local Test Command
```bash
python scripts/test_quicksight_csv_local.py --input examples/quicksight_abrupt_input.csv --output /tmp/quicksight_abrupt_output.csv
```

## Packaging Command
```bash
python scripts/package_quicksight_abrupt_model.py --output artifacts/quicksight_abrupt_model.tar.gz
```

## SageMaker Batch Transform Smoke Test Outline
1. Upload `artifacts/quicksight_abrupt_model.tar.gz` to S3.
2. Create a SageMaker model using an sklearn-compatible inference container and this artifact.
3. Configure Batch Transform with `SplitType=Line`, `ContentType=text/csv`, `Accept=text/csv`.
4. Use headerless CSV input (`lat,lon,t2m` per row).
5. Confirm output has one headerless row per input row with `label,score`.

## AWS Batch Transform Smoke Test (Dev)
Batch Transform only. Do not create a real-time endpoint for this classifier.

Region:
- `us-east-1`

Bucket:
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an`

Smoke S3 paths:
- Input: `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/quicksight-abrupt/smoke/input/input.csv`
- Output: `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/quicksight-abrupt/smoke/output/`
- Model artifact example: `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/quicksight-abrupt/model/quicksight_abrupt_model.tar.gz`

Command sequence:
```bash
AWS_PROFILE=sbnai-725 python scripts/package_quicksight_abrupt_model.py --output artifacts/quicksight_abrupt_model.tar.gz

AWS_PROFILE=sbnai-725 aws s3 cp artifacts/quicksight_abrupt_model.tar.gz \
  s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/quicksight-abrupt/model/quicksight_abrupt_model.tar.gz

AWS_PROFILE=sbnai-725 python scripts/deploy_quicksight_abrupt_batch_model.py \
  --region us-east-1 \
  --model-name chucaw-quicksight-abrupt-classifier-v0 \
  --role-arn <SAGEMAKER_EXECUTION_ROLE_ARN> \
  --model-artifact-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/quicksight-abrupt/model/quicksight_abrupt_model.tar.gz

AWS_PROFILE=sbnai-725 python scripts/run_quicksight_abrupt_batch_smoke.py --region us-east-1 --wait
```

Expected output:
- Transform job created and prints `Transform job name`.
- Output object written under:
  `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/quicksight-abrupt/smoke/output/`
- Output is headerless CSV with two columns per line: `abrupt_temp_change_label,abrupt_temp_change_score`.

## Optional: register in SageMaker Model Registry
This exists to add model versioning, visibility, and governance metadata for the QuickSight abrupt CSV classifier while keeping the current Batch Transform flow unchanged.

What this does:
- Creates the Model Package Group if missing.
- Registers a new Model Package version pointing to the uploaded `model.tar.gz`.
- Stores CSV input/output compatibility and project metadata.

What this does not do:
- Does not create or deploy a real-time endpoint.
- Does not replace the current Batch Transform smoke path.
- Does not change QuickSight custom SageMaker augmentation behavior.

Warning:
- This does not create endpoint resources.
- QuickSight custom SageMaker augmentation in this project still uses Batch Transform.

Register command:
```powershell
$env:AWS_PROFILE="sbnai-725"
$env:AWS_REGION="us-east-1"

conda run -n chucaw-sagemaker python scripts/register_quicksight_abrupt_model_package.py `
  --model-package-group-name chucaw-quicksight-abrupt-classifier `
  --region us-east-1 `
  --approval-status Approved `
  --model-artifact-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/quicksight-abrupt/model/quicksight_abrupt_model.tar.gz
```

List registered packages:
```powershell
$env:AWS_PROFILE="sbnai-725"
$env:AWS_REGION="us-east-1"

conda run -n chucaw-sagemaker python scripts/list_quicksight_abrupt_model_packages.py `
  --model-package-group-name chucaw-quicksight-abrupt-classifier `
  --region us-east-1
```

## Athena View Suggestion
```sql
CREATE OR REPLACE VIEW silverlayer.forecast_quicksight_abrupt_input AS
SELECT
  CAST(latitude AS DECIMAL(9,4)) AS lat,
  CAST(longitude AS DECIMAL(9,4)) AS lon,
  CAST(value AS DECIMAL(10,4)) AS t2m
FROM silverlayer.forecast_parquet
WHERE year = '2026'
  AND month = '04'
  AND day = '14'
  AND hour = '06z'
  AND variable = 't2m';
```
