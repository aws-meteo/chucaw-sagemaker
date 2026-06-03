# SageMaker Batch Transform for FourCastNet

REALTIME_ENDPOINT_COST_RISK

This document outlines the required offline inference pattern for FourCastNet using SageMaker Batch Transform.

## Difference between SageMaker Model and Batch Transform Job

A **SageMaker Model** is merely a database entry in AWS linking a Docker image to an S3 model artifact. It incurs **$0 cost** and provides no compute by itself. 

A **Batch Transform Job** spins up ephemeral compute nodes (e.g. `ml.g4dn.xlarge`), copies the model, streams S3 input to it, captures the output back to S3, and immediately shuts down the nodes. You only pay for the exact duration of the job.

**No Endpoint is Needed**: Creating a real-time endpoint spins up permanent, idle compute that bills continuously (up to $0.7+ per hour per GPU) until explicitly deleted.

## Why FourCastNet Input Size Makes Real-Time Unsuitable

FourCastNet handles massive weather tensors (~1 GB per request). Real-time endpoints have a hard 5MB payload limit and strict timeout bounds. Passing 1GB payloads synchronously over HTTP is brittle, anti-pattern, and typically fails. Batch Transform efficiently handles large data by streaming inputs (like JSONL pointers to S3 paths) asynchronously directly from S3.

## Example Commands

### Dry Run (Default)
By default, the script will strictly print the configuration without executing anything on AWS.
```bash
python scripts/run_fourcastnet_batch_transform.py \
  --model-name sbnai-fourcastnet-fcn-v0-2026-05-27-19-10-45-401 \
  --input-s3-uri s3://your-bucket/input/data.jsonl \
  --output-s3-uri s3://your-bucket/output/ \
  --instance-type ml.g4dn.xlarge
```

### Execute (Manual Only)
Add `--execute` to actually create the Transform Job.
```bash
python scripts/run_fourcastnet_batch_transform.py \
  --model-name sbnai-fourcastnet-fcn-v0-2026-05-27-19-10-45-401 \
  --input-s3-uri s3://your-bucket/input/data.jsonl \
  --output-s3-uri s3://your-bucket/output/ \
  --instance-type ml.g4dn.xlarge \
  --execute
```

## How to Check Job Status

You can monitor it on the AWS SageMaker Console or run:
```bash
aws sagemaker describe-transform-job --transform-job-name <JobName> --query "TransformJobStatus"
```

## How to Inspect S3 Output
```bash
aws s3 ls s3://your-bucket/output/
```

## Confirming No Endpoints Exist
Ensure there is no leakage of expensive real-time endpoints:
```bash
python scripts/check_no_sagemaker_always_on_compute.py
```

## Cost Guardrails
- **No Deploy Policy**: The codebase strictly forbids `.deploy()`, `create_endpoint`, etc.
- **Cost Tags**: the batch config carries `CostMode=batch-only` and `Lifecycle=finished`; the runner includes them in the planned transform job tags.
- **Idempotency**: Execution scripts require explicit opt-in (`--execute`).

## Inference Contract Audit (BATCH_CONTRACT_CONFIRMED)

The model's inference handler (`src/fourcastnet/serving/inference.py`) has been audited for Batch Transform compatibility:

### Accepted Input Format
- **Content Types**: `application/json`, `application/jsonlines`, `application/x-jsonlines`, `application/x-npy`.
- **Data Handling**: For JSON, the handler expects pointers (`input_s3_uri`, `output_s3_uri`) instead of raw data. The tensor is loaded into memory directly from S3 (`np.load(io.BytesIO(blob))`).
- **1 GB Support**: Yes, if `max_runtime_guard=False`. By default, `max_runtime_guard=True` blocks arrays > 50,000,000 elements to prevent accidental OOMs. It successfully loads into memory if the instance (e.g. `ml.g4dn.xlarge` with 16GB RAM) has enough capacity.

### Batch Transform Compatibility
- **SplitType=None**: Supported (treats the entire file as one payload, e.g. for a single `application/x-npy` or a single JSON object).
- **SplitType=Line**: Supported. Required when processing `application/jsonlines` so that Batch Transform feeds one JSON line to `input_fn` per request.
- **BatchStrategy=SingleRecord**: Supported and **Recommended**. Ensures `input_fn` gets exactly one valid JSON object per request.
- **BatchStrategy=MultiRecord**: **Not Supported**. Passing multiple concatenated JSON objects in a single payload will cause `json.loads` to throw a JSONDecodeError inside `input_fn`.
- **MaxPayloadInMB <= 100**: Supported. The actual tensors stay in S3. The payload is merely a JSON pointer string (a few kilobytes).
- **AssembleWith=None or Line**: Supported. The `output_fn` always returns JSON. `AssembleWith=Line` is the proper way to recombine the results.

## The Canonical JSONLines Manifest

Due to the size of FourCastNet inputs (~1 GB), the canonical input type for Batch Transform is `application/jsonlines`.
Instead of passing huge blobs, each JSON line acts as a manifest pointing to the data in S3. 

**Required parameters for JSONL:**
- `ContentType: application/jsonlines`
- `SplitType: Line`
- `BatchStrategy: SingleRecord` (MultiRecord is explicitly rejected to avoid JSON parse errors)
- `AssembleWith: Line`

## Canary-First Workflow

Before running a full 1 GB batch inference, always run a "metadata_only" canary test to ensure IAM permissions, S3 access, and basic model setup are healthy.

1. **Generate the Canary Payload** (Local only first):
```bash
python scripts/create_fourcastnet_canary_payload.py \
  --input-s3-uri s3://your-bucket/fourcastnet/input/ \
  --output-s3-uri s3://your-bucket/fourcastnet/output/
```

2. **Upload Canary to S3** (MANUAL):
```bash
python scripts/create_fourcastnet_canary_payload.py \
  --input-s3-uri s3://your-bucket/fourcastnet/input/ \
  --output-s3-uri s3://your-bucket/fourcastnet/output/ \
  --execute-upload --profile sbnai-725
```

3. **Dry-Run the Batch Transform Job**:
```bash
python scripts/run_fourcastnet_batch_transform.py \
  --config configs/fourcastnet_batch_v0.json \
  --input-s3-uri s3://your-bucket/fourcastnet/input/canary_manifest.jsonl \
  --output-s3-uri s3://your-bucket/fourcastnet/output/canary_out/
```

4. **Execute Canary Job** (MANUAL ONLY):
```bash
python scripts/run_fourcastnet_batch_transform.py \
  --config configs/fourcastnet_batch_v0.json \
  --input-s3-uri s3://your-bucket/fourcastnet/input/canary_manifest.jsonl \
  --output-s3-uri s3://your-bucket/fourcastnet/output/canary_out/ \
  --execute
```

5. **Validate Canary Output**:
```bash
python scripts/validate_fourcastnet_batch_output.py \
  --uri s3://your-bucket/fourcastnet/output/canary_out/canary_manifest.jsonl.out \
  --profile sbnai-725
```

## Full 1 GB Workflow

Once the canary passes cleanly (e.g., `CONTROLLED_ERROR` or `SUCCESS`), you can proceed to the 1 GB processing. The structure is identical to the canary, except the JSONL manifest will point to real S3 numpy arrays and specify `"max_runtime_guard": false`.

*Note*: Direct `application/x-npy` via `run_fourcastnet_batch_transform.py` is blocked by default because S3 pointer streaming (via `application/jsonlines`) is much more reliable than squeezing large blobs through the HTTP request payload limit.
