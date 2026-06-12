# FourCastNet Batch Inference (SageMaker)

REALTIME_ENDPOINT_COST_RISK

FourCastNet inputs are ~1 GB, which makes real-time endpoints expensive and brittle for routine use. This workflow is **batch-first**: create or reuse a SageMaker Model, run Batch Transform on S3 input, and finish with zero idle cost.

## Why not real-time endpoints

- The input is large (~1 GB), so real-time inference is slow and costly.
- Endpoints bill while idle; Batch Transform only bills while running.

## Resource glossary

- **Model**: metadata + container image + model artifact pointer. No cost while idle.
- **Endpoint Config**: deployment config for endpoints. No cost while idle.
- **Endpoint**: live real-time service. **Costs while idle**.
- **Batch Transform Job**: ephemeral offline inference job. Costs only while running.

## Safe lifecycle (batch-only)

1. **Create/keep model** (idempotent).
2. **Run batch transform** (S3 input ➜ S3 output).
3. **Inspect outputs** and stop. **No endpoint remains.**

> [!WARNING]
> Do not use `.deploy()` for FourCastNet unless you intentionally want a real-time endpoint.

Defaults are defined in `configs/fourcastnet_batch_v0.json` (including the conservative default instance type `ml.g4dn.xlarge`).

## Input contract (JSONL indirection)

FourCastNet inputs are too large to pass directly in the Batch Transform payload. Use JSONL where **each line is a small JSON object** that points to the large tensor in S3.

Example `requests.jsonl` (one JSON object per line):

```json
{"input_s3_uri":"s3://<bucket>/<prefix>/input_tensor.npy","output_s3_uri":"s3://<bucket>/<prefix>/outputs/run-id/","mode":"metadata_only"}
```

Recommended transform settings:
- **ContentType**: `application/json`
- **SplitType**: `Line` (one JSON object per record)
- **BatchStrategy**: `SingleRecord`
- **MaxPayloadInMB**: `10` (conservative)

## PowerShell commands

### Describe or create the model (idempotent)

```powershell
python scripts\describe_or_create_fourcastnet_model.py --config configs\fourcastnet_batch_v0.json
```

### Run batch transform (no endpoint)

```powershell
python scripts\run_fourcastnet_batch_transform.py `
  --model-name sbnai-fourcastnet-fcn-v0-2026-05-27-19-10-45-401 `
  --input-s3-uri s3://<bucket>/<prefix>/input/requests.jsonl `
  --output-s3-uri s3://<bucket>/<prefix>/output/ `
  --instance-type ml.g4dn.xlarge `
  --instance-count 1 `
  --content-type application/json `
  --split-type Line `
  --batch-strategy SingleRecord `
  --max-payload-mb 10 `
  --region us-east-1 `
  --profile sbnai-725 `
  --wait `
  --execute
```

### List transform jobs

```powershell
aws sagemaker list-transform-jobs --status-equals InProgress --region us-east-1 --profile sbnai-725
```

### List endpoints

```powershell
aws sagemaker list-endpoints --name-contains fourcastnet --region us-east-1 --profile sbnai-725
```

### Confirm no endpoint exists (CI/preflight)

```powershell
python scripts\check_no_fourcastnet_endpoints.py --region us-east-1 --profile sbnai-725
```

### Describe model

```powershell
aws sagemaker describe-model --model-name sbnai-fourcastnet-fcn-v0-2026-05-27-19-10-45-401 --region us-east-1 --profile sbnai-725
```

### Cleanup (failed transform jobs)

```powershell
aws sagemaker stop-transform-job --transform-job-name <job-name> --region us-east-1 --profile sbnai-725
```

## Notes on cost

- **Endpoints** and **endpoint configs** are not required for batch inference.
- Batch Transform jobs run only when invoked and then stop.
