# Hosting Artifact Investigation

## Outcome

Validated successful endpoint:

- `chucaw-t2m-trained-knn-attempt-02`

Validated pattern:

- **separate-source serving bundle**
- model artifact: `model.tar.gz` with only `model.joblib`
- source artifact: `source.tar.gz` with `inference.py` and minimal requirements

Key reason it worked:

`SAGEMAKER_SUBMIT_DIRECTORY` must point to a downloadable source tarball in S3, not to `/opt/ml/model/code`.

## Failure that was resolved

Observed startup failure in CloudWatch from previous pattern:

`FileNotFoundError: [Errno 2] No such file or directory: '/opt/ml/model/code'`

Previous failing strategy relied on model-embedded `code/` and/or `SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code`.

## Attempt status

- Attempt 01: diagnostic-only, failed for operational baseline.
- Attempt 02: succeeded and is now baseline.
- Attempt 03: diagnostic-only, not baseline.

`src/generate_hosting_attempts.py` remains for diagnostics and evidence generation only. It is not part of normal deploy workflow.
