# chucaw-sagemaker

Minimal, testable, failure-oriented SageMaker baseline for ECMWF tabular inference.

## 1. Project purpose

This repository establishes operational foundation, not model quality:

- local-first data extraction and inference workflow
- Athena extraction-only workflow for reproducible snapshot materialization
- deterministic trivial model artifact
- SageMaker Serverless Inference deployment path
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

3. Deployment and serving:
- `src/upload_to_s3.py` uploads `model.tar.gz`.
- `src/deploy_endpoint.py` deploys/updates SageMaker **Serverless Inference** endpoint.
- `src/invoke_endpoint.py` validates online inference contract.

4. Local inference parity:
- `src/smoke_test_local.py` executes `inference/inference.py` functions end-to-end without SageMaker.

## 3. Why serverless inference was selected

Serverless inference is preferred for this MVP baseline because traffic is low/variable and the objective is deployment clarity with minimal always-on cost.

- no always-on instance management
- simpler first production path for deterministic lookup baseline
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

## 10. Exact deploy commands

```bash
python src/upload_to_s3.py
python src/deploy_endpoint.py
```

## 11. Exact invoke command

```bash
python src/invoke_endpoint.py
```

Request contract:

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

Delete endpoint resources when finished:

```bash
aws sagemaker delete-endpoint --endpoint-name chucaw-t2m-trivial --region us-east-1
```

Delete old endpoint configs/models created by this repo (names prefixed with endpoint name):

```bash
aws sagemaker list-endpoint-configs --name-contains chucaw-t2m-trivial --region us-east-1
aws sagemaker delete-endpoint-config --endpoint-config-name <name> --region us-east-1
aws sagemaker list-models --name-contains chucaw-t2m-trivial --region us-east-1
aws sagemaker delete-model --model-name <name> --region us-east-1
```

## 13. Failure troubleshooting

Explicit failure classes and where they surface:

- local parquet path missing: `src/load_local.py`
- missing required columns: `src/load_local.py`, `src/train.py`, `src/query_athena.py`
- empty filtered dataset: `src/load_local.py`, `src/query_athena.py`
- `.env` missing required variables: `src/query_athena.py`, `src/upload_to_s3.py`, `src/deploy_endpoint.py`, `src/invoke_endpoint.py`
- Athena query failed: `src/query_athena.py`
- Athena query returned zero rows: `src/query_athena.py`
- `model.joblib` missing: `src/pack_model.sh`, `src/smoke_test_local.py`, `inference/model_fn`
- `model.tar.gz` malformed: `src/pack_model.sh`
- S3 upload failure: `src/upload_to_s3.py`
- SageMaker deployment failure: `src/deploy_endpoint.py`
- endpoint invocation failure: `src/invoke_endpoint.py`
- malformed request JSON / unsupported content type: `inference/input_fn`

## 14. Known limitations

- model is nearest-grid lookup in degree space; no interpolation, no learning
- lookup uses only `variable='t'` and `isobaricinhpa=1000.0`
- no automated IAM provisioning in this repository
- this baseline assumes valid AWS networking and role trust already in place
- no integration test automation included yet (scripts are explicit runbook smoke checks)

## 15. Architectural decisions and explicit deviations

- **Serverless selected over always-on endpoint:** baseline traffic profile and MVP objective favor serverless simplicity/cost posture.
- **Athena extraction-only:** inference does not depend on Athena runtime; Athena used only to materialize deterministic snapshot.
- **Trivial lookup artifact over real ML:** intentional failure-transparent baseline; isolates data/packaging/deploy/invoke path before modeling complexity.
- **Explicit deviation:** deployment uses low-level SageMaker API calls (`create_model`, `create_endpoint_config`, `create_endpoint`) instead of high-level estimator deploy helper. Reason: clearer serverless config, explicit resource tags, explicit container environment wiring, and clearer failure diagnostics.

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
  - `sagemaker:CreateEndpointConfig`
  - `sagemaker:CreateEndpoint`
  - `sagemaker:UpdateEndpoint`
  - `sagemaker:DescribeEndpoint`
  - `sagemaker:InvokeEndpoint`
  - `sagemaker:List*` / `sagemaker:Delete*` for cleanup (optional but practical)
- Athena:
  - `athena:StartQueryExecution`
  - `athena:GetQueryExecution`
  - `athena:GetQueryResults`
- S3:
  - `s3:GetObject`, `s3:PutObject`, `s3:ListBucket` on model/artifact and Athena output paths
- IAM pass role:
  - `iam:PassRole` for configured SageMaker execution role only
