# Team Runbook

## Executive summary

Operational baseline is now **separate-source serving bundle**.

Attempt 02 was the validated proof point and successfully created/invoked endpoint:

- `chucaw-t2m-trained-knn-attempt-02`

Clear statement: **Attempt 02 succeeded because source code is uploaded separately and referenced through `SAGEMAKER_SUBMIT_DIRECTORY` as an S3 tarball.**

## Final architecture

Athena/local data -> training dataset -> SageMaker Training Job -> training model artifact -> serving source bundle -> deploy endpoint -> invoke endpoint.

Two S3 artifacts are required for deploy:

1. model data artifact (`model.tar.gz`) from `MODEL_S3_BUCKET/MODEL_S3_PREFIX`
2. source code artifact (`source.tar.gz`) from `SERVING_SOURCE_S3_BUCKET/SERVING_SOURCE_S3_PREFIX`

Training model artifact is model data only. Serving source bundle is code artifact only.

## Required `.env` variables

```env
REGION=us-east-1
AWS_PROFILE=<your-profile>
ROLE_ARN=arn:aws:iam::<account-id>:role/<sagemaker-exec-role>

ATHENA_DATABASE=silverlayer
ATHENA_TABLE=forecast_parquet
ATHENA_OUTPUT_BUCKET=<athena-output-bucket>
ATHENA_OUTPUT_PREFIX=<athena-output-prefix>

TRAINING_DATA_S3_PREFIX=sagemaker/chucaw-t2m-trivial/training-data

MODEL_S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an
MODEL_S3_PREFIX=sagemaker/chucaw-t2m-trained-knn/current/model

SERVING_SOURCE_S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an
SERVING_SOURCE_S3_PREFIX=sagemaker/chucaw-t2m-trained-knn/current/source

ENDPOINT_NAME=chucaw-t2m-trained-knn
SAGEMAKER_PROGRAM=inference.py
SKLEARN_VERSION=1.2-1

MODEL_PACKAGE_GROUP_NAME=chucaw-t2m-models
MODEL_APPROVAL_STATUS=Approved

PROJECT_TAG=chucaw
OWNER_TAG=<owner>
ENV_TAG=prod
```

## Exact command sequence

### A) Prepare data

```bash
python src/query_athena.py --year 2026 --month 04 --day 09 --hour 18z
python src/prepare_training_data.py --input data/t2m_snapshot.csv --output data/training/train.csv --dataset-name 2026-04-09-18
python src/upload_training_data.py --input data/training/train.csv --dataset-id 2026-04-09-18
```

Expected outputs:

- `data/t2m_snapshot.csv`
- `data/training/train.csv`
- printed S3 URI for uploaded training CSV

### B) Train

```bash
python src/run_training_job.py --train-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/chucaw-t2m-trivial/training-data/2026-04-09-18/train.csv --wait --logs
```

Expected output:

- training job name
- model artifact S3 URI from training output

### C) Build and upload serving artifacts

Use the exact training model artifact URI printed by step B.

```bash
python src/build_hosting_artifact.py --training-artifact-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/chucaw-t2m-trivial/training-jobs/<job-name>/output/model.tar.gz --upload --bucket chucaw-data-platinum-processed-725644097028-us-east-1-an --model-prefix sagemaker/chucaw-t2m-trained-knn/current/model --source-prefix sagemaker/chucaw-t2m-trained-knn/current/source
```

Expected outputs:

- local: `artifacts/build/model.tar.gz`
- local: `artifacts/build/source.tar.gz`
- printed S3 URI for `model.tar.gz`
- printed S3 URI for `source.tar.gz`

How to verify locally:

```bash
tar -tzf artifacts/build/model.tar.gz
tar -tzf artifacts/build/source.tar.gz
python src/smoke_test_local.py --artifact-local artifacts/build/model.tar.gz --source-artifact-local artifacts/build/source.tar.gz
```

### D) Deploy

```bash
python src/deploy_endpoint.py
```

Expected outputs:

- model package ARN
- model data URL
- container environment including `SAGEMAKER_PROGRAM=inference.py`
- container environment with `SAGEMAKER_SUBMIT_DIRECTORY=s3://.../source.tar.gz`
- endpoint status `InService`

### E) Invoke

```bash
python src/invoke_endpoint.py --lat -33.5 --lon -70.6
```

Expected output:

- JSON response with unchanged contract keys:
  - `t2m`
  - `lat_grid`
  - `lon_grid`
  - `units`
  - `source`

## Already-validated attempt-02 environment

```env
MODEL_S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an
MODEL_S3_PREFIX=sagemaker/chucaw-t2m-trained-knn/attempt-02
SERVING_SOURCE_S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an
SERVING_SOURCE_S3_PREFIX=sagemaker/chucaw-t2m-trained-knn/attempt-02/source
ENDPOINT_NAME=chucaw-t2m-trained-knn-attempt-02
SAGEMAKER_PROGRAM=inference.py
```

## Recommended stable environment

```env
MODEL_S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an
MODEL_S3_PREFIX=sagemaker/chucaw-t2m-trained-knn/current/model
SERVING_SOURCE_S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an
SERVING_SOURCE_S3_PREFIX=sagemaker/chucaw-t2m-trained-knn/current/source
ENDPOINT_NAME=chucaw-t2m-trained-knn
SAGEMAKER_PROGRAM=inference.py
```

## Common failure modes

1. `FileNotFoundError: /opt/ml/model/code`
   - cause: old single-artifact pattern or wrong submit directory
   - fix: use separate-source bundle and set `SERVING_SOURCE_*` (or explicit `SAGEMAKER_SUBMIT_DIRECTORY` to `source.tar.gz`)

2. Endpoint exists in `Failed`
   - fix: delete manually, or use:
   - `python src/deploy_endpoint.py --delete-failed-endpoint`
   - deletion is opt-in and not default

3. Training artifact invalid
   - cause: missing `model.joblib` in training output
   - fix: verify training job completed and pass correct output `model.tar.gz` URI

4. Source bundle invalid
   - cause: missing `inference.py` in `source.tar.gz`
   - fix: rebuild with `src/build_hosting_artifact.py` and verify tar listing

## Cleanup notes

- Do not delete remote resources as part of normal runbook execution.
- Keep `artifacts/hosting_attempts/` for investigation evidence if needed.
- Generated local files (`artifacts/build`, `tmp_model`, tarballs) are ignored in git.

## Diagnostic-only files

- `src/generate_hosting_attempts.py` is diagnostic-only.
- `docs/hosting_artifact_investigation.md` records attempt evidence.
- Operational baseline commands are in this runbook only.
