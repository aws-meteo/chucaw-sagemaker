# chucaw-sagemaker

REALTIME_ENDPOINT_COST_RISK

Minimal, testable, failure-oriented SageMaker baseline for ECMWF tabular inference.

Current SageMaker cost-safety policy is documented in [docs/sagemaker_cost_safety.md](docs/sagemaker_cost_safety.md):

- FourCastNet default mode: **CreateModel + Batch Transform only**
- forbidden default mode: real-time endpoint, async endpoint, Studio app, notebook instance, or any always-on hosting
- endpoint deployment and invocation scripts in default paths intentionally fail

If any command in this README conflicts with `docs/team_runbook.md`, use the runbook as source of truth.

## 1. Project purpose

This repository establishes operational foundation, not model quality:

- local-first data extraction and inference workflow
- Athena extraction-only workflow for reproducible snapshot materialization
- deterministic trivial model artifact
- SageMaker Batch Transform deployment path
- strict data and response contracts
- explicit failure points and diagnostics

## 2. Architecture overview

1. Data materialization:
- LOCAL mode: read local parquet snapshot and materialize `data/t2m_snapshot.csv`.
- ATHENA mode: query `silverlayer.forecast_parquet` with strict partition filters, then materialize same CSV schema.

2. Artifact creation:
- `src/train.py` serializes trivial lookup artifact `model.joblib`.
- `src/pack_model.sh` packages `model.tar.gz` with required SageMaker structure.
- local artifact serialization intentionally aligned with SageMaker scikit-learn container family (`scikit-learn 1.2.2` local, `framework_version=1.2-1` in SageMaker) to reduce deserialization risk.

3. Batch serving:
- `src/upload_to_s3.py` uploads `model.tar.gz`.
- `scripts/run_fourcastnet_batch_transform.py` prepares a dry-run Batch Transform payload by default.
- `src/deploy_endpoint.py` intentionally fails because endpoint hosting is not a default path.

4. Local inference parity:
- `src/smoke_test_local.py` executes `inference/inference.py` functions end-to-end without SageMaker.

## 3. Why Batch Transform is the default

Batch Transform is required for FourCastNet because the model input is approximately 1 GB and the previous endpoint incident showed real idle cost risk.

- no always-on endpoint instance
- explicit job lifecycle
- explicit fail-fast behavior around packaging, permissions, endpoint state, and invocation

## 4. Source data contract (`silverlayer.forecast_parquet`)

Table identity:

- Database: `silverlayer`
- Table: `forecast_parquet`
- S3: `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/ecmwf/parquet/`

Non-partition columns:

- `isobaricinhpa` (double)
- `latitude` (double)
- `longitude` (double)
- `variable` (string)
- `value` (float)
- `date` (string)
- `run` (string)

Partition columns:

- `year` (string)
- `month` (string)
- `day` (string)
- `hour` (string)

MVP extraction subset:

- `variable = 't'`
- `isobaricinhpa = 1000.0`

Intermediate schema convergence (both LOCAL and ATHENA):

- `latitude`
- `longitude`
- `value`

## 5. LOCAL mode and ATHENA mode

LOCAL mode:

- input: `data/20260409180000-6h-scda-fc.parquet`
- logic:
```python
df = df[(df["variable"] == "t") & (df["isobaricinhpa"] == 1000.0)]
df = df[["latitude", "longitude", "value"]].dropna()
```
- output: `data/t2m_snapshot.csv`

ATHENA mode:

- uses fixed SQL shape with partition filtering
- only partition values (`year`, `month`, `day`, `hour`) are CLI-parameterized
- defaults: `2026`, `04`, `09`, `18`
- output: same `data/t2m_snapshot.csv`

## 6. Environment setup (exact commands)

```bash
conda env create -f environment.yml
conda activate chucaw-sagemaker
cp .env.example .env
```

If sample parquet is not already inside repository `data/`, copy it before LOCAL run:

```bash
cp ../data/20260409180000-6h-scda-fc.parquet data/20260409180000-6h-scda-fc.parquet
```

## 7. Exact LOCAL parquet execution

```bash
python src/load_local.py --parquet data/20260409180000-6h-scda-fc.parquet
python src/train.py --input data/t2m_snapshot.csv
bash src/pack_model.sh
python src/smoke_test_local.py
```

## 8. Exact ATHENA extraction command

```bash
python src/query_athena.py --year 2026 --month 04 --day 09 --hour 18
python src/train.py --input data/t2m_snapshot.csv
bash src/pack_model.sh
python src/smoke_test_local.py
```

Athena SQL shape used:

```sql
SELECT latitude, longitude, value
FROM silverlayer.forecast_parquet
WHERE variable      = 't'
  AND isobaricinhpa = 1000.0
  AND year  = '2026'
  AND month = '04'
  AND day   = '09'
  AND hour  = '18'
```

## 9. Exact model packaging command

```bash
bash src/pack_model.sh
```

`src/pack_model.sh` validates final tarball structure must be exactly:

```text
model.tar.gz
├── model.joblib
└── code/
    ├── inference.py
    └── requirements.txt
```

## 10. Exact Batch Transform dry-run command

```bash
python scripts/check_no_sagemaker_always_on_compute.py
python scripts/run_fourcastnet_batch_transform.py --input-s3-uri s3://<bucket>/<input-prefix>/ --output-s3-uri s3://<bucket>/<output-prefix>/ --model-name <model-name>
```

`scripts/run_fourcastnet_batch_transform.py` is dry-run by default. It must not create an AWS job unless `--execute` is supplied after preflight review.

## 11. Endpoint commands are disabled

```bash
python src/deploy_endpoint.py
```

There is no FourCastNet invoke script in the default path. Batch Transform writes results to the output S3 prefix instead.

Batch request contract:

```json
{"lat": -33.5, "lon": -70.6}
```

Response contract:

```json
{
  "t2m": 287.3,
  "lat_grid": -33.48,
  "lon_grid": -70.59,
  "units": "K",
  "source": "ECMWF SCDA"
}
```

## 12. Cleanup guidance

Generate local cleanup commands only; dry-run is the default:

```bash
python scripts/generate_sagemaker_cleanup_commands.py
.\generated_cleanup_sagemaker.ps1
```

Actual endpoint/config/app deletion requires:

```powershell
.\generated_cleanup_sagemaker.ps1 -Execute
```

Do not delete SageMaker Models, S3 artifacts, ECR images, CloudWatch logs, domains, user profiles, or spaces from automated cleanup.

## 13. Failure troubleshooting

Explicit failure classes and where they surface:

- local parquet path missing: `src/load_local.py`
- missing required columns: `src/load_local.py`, `src/train.py`, `src/query_athena.py`
- empty filtered dataset: `src/load_local.py`, `src/query_athena.py`
- `.env` missing required variables: `src/query_athena.py`, `src/upload_to_s3.py`
- Athena query failed: `src/query_athena.py`
- Athena query returned zero rows: `src/query_athena.py`
- `model.joblib` missing: `src/pack_model.sh`, `src/smoke_test_local.py`, `inference/model_fn`
- `model.tar.gz` malformed: `src/pack_model.sh`
- S3 upload failure: `src/upload_to_s3.py`
- SageMaker endpoint deployment requested: default scripts refuse and point to Batch Transform
- endpoint invocation is not available in the default FourCastNet path
- malformed request JSON / unsupported content type: `inference/input_fn`

## 14. Known limitations

- model is nearest-grid lookup in degree space; no interpolation, no learning
- lookup uses only `variable='t'` and `isobaricinhpa=1000.0`
- no automated IAM provisioning in this repository
- this baseline assumes valid AWS networking and role trust already in place
- no integration test automation included yet (scripts are explicit runbook smoke checks)

## 15. Architectural decisions and explicit deviations

- **Batch Transform selected over endpoint hosting:** FourCastNet input size and the May 2026 cost incident make real-time hosting the wrong default.
- **Athena extraction-only:** inference does not depend on Athena runtime; Athena used only to materialize deterministic snapshot.
- **Trivial lookup artifact over real ML:** intentional failure-transparent baseline; isolates data/packaging/deploy/invoke path before modeling complexity.
- **Explicit deviation (experimental):** real-time deployment exists only under `experimental/realtime_endpoint_dangerous/` and is not the default workflow.

## Inference implementation notes

`inference/inference.py` provides required SageMaker handlers:

- `model_fn(model_dir)`
- `input_fn(request_body, content_type="application/json")`
- `predict_fn(input_data, model)`
- `output_fn(prediction, accept="application/json")`

`output_fn` appends fixed metadata:

- `"units": "K"`
- `"source": "ECMWF SCDA"`

## IAM minimum permission scope (documented only)

Do not use `AdministratorAccess`. Minimal policy scope should cover:

- SageMaker:
  - `sagemaker:CreateModel`
  - `sagemaker:CreateTransformJob`
  - `sagemaker:DescribeTransformJob`
  - `sagemaker:List*` / `sagemaker:Delete*` for cleanup (optional but practical)
- Athena:
  - `athena:StartQueryExecution`
  - `athena:GetQueryExecution`
  - `athena:GetQueryResults`
- S3:
  - `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on model/artifact and Athena output paths
- IAM pass role:
  - `iam:PassRole` for configured SageMaker execution role only


## 16. Batch inference modes

There are two ways to run batch inference in this repository:

### A) QuickSight-aligned local batch (Parquet)
This is the recommended path for generating dashboard-ready data. It runs locally, uses the canonical prediction schema, and outputs Parquet files suitable for S3/Athena/QuickSight.

```bash
python scripts/run_batch_inference_local.py \
  --input data/20260409180000-6h-scda-fc.parquet \
  --output artifacts/predictions_local.parquet \
  --source-year 2026 --source-month 04 --source-day 09 --source-hour 18
```

For detailed pipeline setup (Athena DDL, QuickSight datasets), see [docs/pipeline_quicksight_integration.md](docs/pipeline_quicksight_integration.md).

### B) SageMaker Batch Transform smoke test (JSON)
This is used to validate the `model.tar.gz` and `inference.py` contract inside a managed SageMaker environment. It uses the legacy JSON response format.

1. Upload test input to S3:
```bash
aws s3 cp data/batch_smoketest_input.json s3://<your-bucket>/batch-smoketest/input/input.json
```

2. Run transform job:
```bash
python src/run_batch_transform_smoketest.py
```

## 17. QuickSight integration

The project now supports a local-to-cloud downscaling pipeline:
- **Local:** `scripts/run_batch_inference_local.py`
- **Athena DDL:** `sql/create_quicksight_predictions_table.sql`
- **Runbook:** [docs/reports/aws_quicksight_validation_next_steps.md](docs/reports/aws_quicksight_validation_next_steps.md)
