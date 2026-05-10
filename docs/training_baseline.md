# Training Baseline

This baseline defines one operational path:

Athena/local data -> training dataset -> SageMaker training job -> training `model.tar.gz` -> separate-source serving bundle (`model.tar.gz` + `source.tar.gz`) -> deploy endpoint -> invoke endpoint.

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
python src/deploy_endpoint.py
python src/invoke_endpoint.py --lat -33.5 --lon -70.6
```

## Artifact contract

Training output artifact (SageMaker Training Job):

- `output/model.tar.gz` from the job output location

Serving artifacts (built by `src/build_hosting_artifact.py`):

- model artifact `model.tar.gz`: contains only `model.joblib`
- source artifact `source.tar.gz`: contains `inference.py` and `requirements.txt`

This is the official **separate-source serving bundle** design.
