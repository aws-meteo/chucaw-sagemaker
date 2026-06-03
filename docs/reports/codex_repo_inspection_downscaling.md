# Repo Inspection: Downscaling Inference Contract

## Scope

Repository: `chucaw-sagemaker`  
Goal inspected: generic, pipeline-friendly inference output for Athena/QuickSight consumption.

## Current relevant files

- `inference/inference.py` - SageMaker inference entrypoint (`model_fn`, `input_fn`, `predict_fn`, `output_fn`)
- `src/train.py` - local trivial artifact builder (`grid` + `t2m`)
- `training/train.py` - SageMaker training job script (`KNeighborsRegressor`)
- `src/load_local.py` - local parquet extraction (`variable='t'`, `isobaricInhPa=1000.0`)
- `src/query_athena.py` - Athena extraction with partition predicates
- `src/deploy_endpoint.py`, `src/upload_to_s3.py` - batch-only deployment flow
- `src/pack_model.sh` / `src/pack_model.ps1` - model packaging
- `src/smoke_test_local.py` - local handler smoke test

## Current KNN entrypoints

- Training path 1: `src/train.py` (legacy nearest-grid lookup artifact)
- Training path 2: `training/train.py` (sklearn KNN regressor artifact)
- Inference path: `inference/inference.py` chooses behavior by artifact type

## Current training/inference/deployment scripts

- Data extract local: `python src/load_local.py --parquet ...`
- Data extract Athena: `python src/query_athena.py --year ... --month ... --day ... --hour ...`
- Build model artifact: `python src/train.py --input data/t2m_snapshot.csv` (or SageMaker `training/train.py`)
- Package: `src/pack_model.sh` or `src/pack_model.ps1`
- Local smoke: `python src/smoke_test_local.py`
- Deploy: `python src/upload_to_s3.py` then `python src/deploy_endpoint.py`
- Batch outputs are consumed from S3; there is no default `invoke_endpoint` path for FourCastNet

## Current assumptions

- Baseline target variable fixed to `t` at `isobaricInhPa=1000.0`
- Feature space uses only `latitude`, `longitude`
- Online response shape optimized for endpoint (`t2m`, `lat_grid`, `lon_grid`)
- Athena used as extraction source, not online dependency
- Partition filtering expected in Athena queries (`year/month/day/hour`)

## Gaps blocking QuickSight consumption

1. No generic predictor contract abstraction; inference logic hard-coded in handler.
2. No canonical prediction output schema for downstream Athena table.
3. No local batch prediction path writing Parquet for BI consumption.
4. No SQL artifact to create QuickSight-facing Athena external table.
5. No docs connecting SageMaker inference outputs to Athena/QuickSight ingestion flow.
6. No tests validating schema contract for batch output.

## Change strategy selected

Smallest clean change set:

- Add predictor interface + KNN implementation wrapper.
- Keep existing inference handlers and output keys stable.
- Add schema builder utility and local batch inference script.
- Add Athena DDL and integration docs.
- Add minimal fast tests for schema/contract/batch writer.
