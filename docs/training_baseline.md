# Training Baseline

REALTIME_ENDPOINT_COST_RISK

This baseline now defines one safe operational path:

Athena/local data -> training dataset -> model artifact -> SageMaker Model -> Batch Transform.

The old deploy/invoke endpoint sequence is disabled by default. FourCastNet must not use real-time or async endpoint hosting unless the user explicitly accepts the experimental cost risk in the same session.

## Data contract

Source of truth in Athena:

- database: `silverlayer`
- table: `forecast_parquet`
- filter: `variable='t'` and `isobaricinhpa=1000.0`

Both LOCAL and ATHENA flows must converge to:

- `latitude`
- `longitude`
- `value`

## Commands

```bash
python src/query_athena.py --year 2026 --month 04 --day 09 --hour 18z
python src/prepare_training_data.py --input data/t2m_snapshot.csv --output data/training/train.csv --dataset-name 2026-04-09-18
python src/upload_training_data.py --input data/training/train.csv --dataset-id 2026-04-09-18
python src/run_training_job.py --train-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/chucaw-t2m-trivial/training-data/2026-04-09-18/train.csv --wait --logs
python src/build_hosting_artifact.py --training-artifact-s3-uri s3://<bucket>/<training-job-output>/model.tar.gz --upload --bucket chucaw-data-platinum-processed-725644097028-us-east-1-an --model-prefix sagemaker/chucaw-t2m-trained-knn/current/model --source-prefix sagemaker/chucaw-t2m-trained-knn/current/source
python scripts/check_no_sagemaker_always_on_compute.py
python scripts/run_fourcastnet_batch_transform.py --input-s3-uri s3://<bucket>/<input-prefix>/ --output-s3-uri s3://<bucket>/<output-prefix>/ --model-name <model-name>
```

## Artifact contract

Training output artifact (SageMaker Training Job):

- `output/model.tar.gz` from the job output location

Batch-serving artifacts:

- model artifact `model.tar.gz`: contains only `model.joblib`
- source artifact `source.tar.gz`: contains `inference.py` and `requirements.txt`

This is retained for packaging compatibility, but the default SageMaker execution mode is Batch Transform.
