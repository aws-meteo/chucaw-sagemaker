# FourCastNet Model Deployment Guide

REALTIME_ENDPOINT_COST_RISK

This guide records what was done in the recent deployment work, what belongs in git,
and the batch-only line of execution for future model updates.

## What changed

The recent useful source changes are:

- `scripts/run_fourcastnet_cpu_poc.py`
  - derives manifest and output S3 paths from the input tensor URI
  - writes one JSONL manifest line for `metadata_only` or `forward`
  - previews the transform job by default; only `--execute` creates it
  - uploads the manifest before `--execute`, because the transform job reads that manifest
- `src/fourcastnet/serving/inference.py`
  - defines the canonical 20-channel FourCastNet order
  - reports per-channel input/output stats
  - ignores extra normalization channels when 21-channel stats feed the 20-channel model
  - optionally writes the denormalized forecast tensor to S3 when
    `write_output_tensor=true`
- `tests/test_run_fourcastnet_cpu_poc.py` and
  `tests/test_channel_stats_and_tensor_write.py`
  - cover the derived S3 paths, dry-run behavior, channel stats, normalization warning,
    and tensor-write success/failure semantics

The untracked dry-run files and output folders are operational evidence, not source.
Keep them only while investigating a run. Do not commit them.

## Commit boundary

Commit:

- source code that changes runtime behavior
- config JSON files that define reusable model or transform contracts
- tests that lock those contracts
- docs that explain the execution line
- GitHub workflow files that enforce the same line

Do not commit:

- `dryrun_*.txt`
- `*_output/`
- `*_success_report.json`
- local tool state such as `.serena/`
- generated model archives, local artifacts, tensors, or copied AWS CLI payloads

For the current dirty tree, the important commit set is:

- `.gitignore`
- `.github/workflows/fourcastnet-model-deploy.yml`
- `docs/fourcastnet_model_deployment_guide.md`
- `scripts/run_fourcastnet_cpu_poc.py`
- `src/fourcastnet/serving/inference.py`
- `tests/test_run_fourcastnet_cpu_poc.py`
- `tests/test_channel_stats_and_tensor_write.py`
- `tests/test_github_workflow_guardrails.py`
- `configs/fourcastnet_batch_cpu_forward_v2.json`

`configs/fourcastnet_batch_cpu_forward_v1.json` is only worth committing if you want a
checked-in rollback/reference for the first forward artifact. The automated workflow uses
`v2`.

## Execution line

Use this order for a model-code update:

1. Validate locally.

```bash
python -m pytest tests/test_run_fourcastnet_cpu_poc.py tests/test_channel_stats_and_tensor_write.py tests/test_run_fourcastnet_batch_transform.py
```

2. Build a self-contained model artifact from the current handler.

```bash
python scripts/package_fourcastnet_model.py \
  --assets-dir <local_assets_dir> \
  --requirements-file requirements-forward.txt \
  --layout self-contained
```

3. Run the SageMaker safety checks before any AWS write.

```bash
python scripts/check_no_sagemaker_always_on_compute.py --region us-east-1
python scripts/check_no_fourcastnet_endpoints.py --region us-east-1
```

4. Upload the versioned artifact.

```bash
aws s3 cp artifacts/fourcastnet/build/model.tar.gz \
  s3://<bucket>/sagemaker/fourcastnet/fcn-v1/model/model-<git-sha>.tar.gz
```

5. Create a new SageMaker Model record that points at that artifact.

```bash
python scripts/describe_or_create_fourcastnet_model.py \
  --config configs/fourcastnet_batch_cpu_forward_v2.json \
  --model-name sbnai-fourcastnet-fcn-v1-cpu-forward-<git-sha12> \
  --model-data-url s3://<bucket>/sagemaker/fourcastnet/fcn-v1/model/model-<git-sha>.tar.gz \
  --region us-east-1
```

6. Validate the deploy command with a dry-run transform plan.

```bash
python scripts/run_fourcastnet_cpu_poc.py \
  --config configs/fourcastnet_batch_cpu_forward_v2.json \
  --mode forward \
  --input-s3-uri s3://<bucket>/ecmwf/fourcastnet/year=2026/month=06/day=13/hour=06z/20260613060000-24h-oper-fc_tensor.npy \
  --no-max-runtime-guard
```

No real-time endpoint is part of this path. The cost mode is `batch-only`: model
artifact upload plus SageMaker Model metadata creation, then Batch Transform only when a
human explicitly runs a transform job.

## GitHub workflow

`.github/workflows/fourcastnet-model-deploy.yml` automates the same line.

- it triggers manually from GitHub Actions with `workflow_dispatch`
- run it from branch `main`; the deploy job is guarded with `github.ref == 'refs/heads/main'`
- the workflow runs source guardrails and the focused tests before AWS writes
- the workflow downloads model assets from S3, uploads a versioned `model.tar.gz`, and
  creates a new SageMaker Model record
- the workflow does not execute a Batch Transform job

Required GitHub repo secrets:

- secret `AWS_ROLE_TO_ASSUME`: IAM role trusted by GitHub OIDC
- secret `AWS_REGION`: defaults to `us-east-1` when absent
- secret `FCN_ASSETS_S3_URI`: S3 prefix containing `backbone.ckpt`, `global_means.npy`,
  and `global_stds.npy`
- secret `FCN_MODEL_S3_PREFIX`: defaults to
  `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v1/model`
- secret `FCN_TEST_INPUT_S3_URI`: defaults to the June 13, 2026 24h tensor when absent

Configure them with `gh`:

```bash
gh secret set AWS_ROLE_TO_ASSUME --repo aws-meteo/chucaw-sagemaker --body "arn:aws:iam::725644097028:role/<github-oidc-role>"
gh secret set AWS_REGION --repo aws-meteo/chucaw-sagemaker --body "us-east-1"
gh secret set FCN_ASSETS_S3_URI --repo aws-meteo/chucaw-sagemaker --body "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/"
gh secret set FCN_MODEL_S3_PREFIX --repo aws-meteo/chucaw-sagemaker --body "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v1/model"
gh secret set FCN_TEST_INPUT_S3_URI --repo aws-meteo/chucaw-sagemaker --body "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/ecmwf/fourcastnet/year=2026/month=06/day=13/hour=06z/20260613060000-24h-oper-fc_tensor.npy"
gh secret list --repo aws-meteo/chucaw-sagemaker
```

Run the manual deploy from GitHub:

```bash
gh workflow run fourcastnet-model-deploy.yml --repo aws-meteo/chucaw-sagemaker --ref main
gh run watch --repo aws-meteo/chucaw-sagemaker
```

The workflow must keep `scripts/check_no_sagemaker_always_on_compute.py` before every AWS
write. If that check fails, deployment stops.
