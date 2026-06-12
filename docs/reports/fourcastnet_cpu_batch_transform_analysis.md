# FourCastNet CPU Batch Transform — Code-Level Analysis & Fix

_Date: 2026-06-12 · Branch: `claude/fourcastnet-cpu-batch-transform-hfq19w`_

This report answers the brief's open blocker — **"can `sbnai-fourcastnet-fcn-v1`
run on CPU, and is `inference.py` CPU-compatible?"** — at the code level, since
no AWS CLI access was available in this session. It also delivers the tooling to
verify and (if needed) fix it the moment AWS access is restored.

## TL;DR

1. **The FourCastNet PyTorch/CUDA inference code is NOT in this repository.** It
   lives only inside the S3 artifact `model.tar.gz` under `code/inference.py`
   (the registered model sets `SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code`).
   Therefore the CPU/GPU question **cannot be settled by reading repo files** —
   it requires inspecting the downloaded artifact.
2. The repo's `inference/inference.py` is a **different model**: a NumPy/sklearn
   nearest-grid `lat`/`lon` lookup baseline. It has **no torch and no CUDA**, and
   only accepts `application/json` — it is unrelated to the FourCastNet tensor
   path described in the brief.
3. The docs reference `src/fourcastnet/serving/inference.py` as "audited", but
   **that file does not exist in the repo** — the audit claim is unverifiable
   from source.
4. **Fix delivered:** a reviewable, unit-tested tool
   (`scripts/inspect_and_patch_fourcastnet_inference.py`) that inspects a
   downloaded `model.tar.gz` for hardcoded CUDA and, if found, emits a CPU-patched
   copy **without touching the original**; plus a gated PowerShell runner
   (`scripts/run_fourcastnet_cpu_poc.ps1`) that wires the whole safe flow together.

## What is actually in the repo

| File | Role | torch/CUDA? | Input contract |
|------|------|-------------|----------------|
| `inference/inference.py` | KNN/sklearn `lat`/`lon` baseline handler | none | `application/json` `{lat,lon}` |
| `src/predictors.py` | loads `model.joblib`, nearest-grid lookup | none | n/a |
| `scripts/run_fourcastnet_batch_transform.py` | boto3 TransformJob spec builder | none | builds JSON spec only |
| `configs/fourcastnet_batch_v0.json` | GPU (`ml.g4dn.xlarge`) batch config | n/a | `application/jsonlines` |

None of these contain the FourCastNet model. The runner script only *constructs
and submits* a `CreateTransformJob` request — it never imports torch and never
selects a device, so there is nothing CPU/GPU to "fix" in repo Python. The device
decision is made by:

- **the container image** (GPU vs CPU PyTorch ECR image), and
- **`code/inference.py` inside the artifact** (whether it forces `.cuda()`).

## The two-model confusion (important)

The brief conflates two distinct artifacts:

- **`sbnai-fourcastnet-fcn-v1`** — GPU image
  `pytorch-inference:2.1.0-gpu-py310`, artifact contains `backbone.ckpt`,
  `global_means.npy`, `global_stds.npy`, `code/inference.py`. **This is the real
  FourCastNet** and the subject of the POC.
- **The repo baseline** — separate-source bundle (`build_hosting_artifact.py`
  explicitly forbids a `code/` dir in its `model.tar.gz`), `model.joblib` only,
  served by `inference/inference.py`. **Not FourCastNet.**

Do not assume the repo `inference/inference.py` reflects what runs inside the
FourCastNet artifact. They are unrelated.

## CPU readiness decision tree

```
Download s3://.../fcn-v1/model/model.tar.gz
        │
        ▼
Inspect code/inference.py  (scripts/inspect_and_patch_fourcastnet_inference.py)
        │
   ┌────┴─────────────────────────┐
   │ no hard CUDA blockers         │ hardcoded CUDA found
   │ (e.g. is_available() / cpu)   │ (.cuda(), device("cuda"), .to("cuda"))
   ▼                               ▼
Create CPU model on the SAME       Emit CPU-patched model.tar.gz to a NEW key,
model.tar.gz + CPU image           review change log, then create CPU model on it
        │                               │
        └───────────────┬───────────────┘
                        ▼
        ml.m5.large transform quota >= 1  (CPU quota exists; GPU = 0)
                        ▼
        Validate workflow → start run → verify S3 output → confirm no endpoint
```

### What "CPU-safe" means here
- **Blockers** (will crash on CPU): `torch.device("cuda")`, `.cuda()`,
  `.to("cuda")`, `torch.cuda.set_device(...)`.
- **Safe signals**: `torch.cuda.is_available()` fallback,
  `map_location="cpu"`.
- **Warning** (verify, non-fatal): `torch.load(...)` **without** `map_location`
  — a CUDA-serialized checkpoint can fail to load on CPU. The patcher injects
  `map_location="cpu"` for these.

## ⚠️ Second, independent risk: content-type mismatch

Separate from CPU/GPU, the MWAA workflow
(`workflows/fourcastnet_cpu_batch_transform_poc_v2.yaml`) feeds the 82 MB tensor
as `ContentType: application/x-npy` with `SplitType: None`. But
`docs/fourcastnet_batch_transform.md` states the **canonical/required** path is a
**JSONL S3-pointer manifest** (`application/jsonlines`), and that direct
`application/x-npy` is *"blocked by default"* in the runner because streaming via
S3 pointers is more reliable. The runner script enforces this:
`application/x-npy` requires `--allow-large-direct-payload`.

**Implication:** even after CPU is sorted, the run can still fail inside
`input_fn` if the real handler does not accept raw `x-npy`. The inspector prints
the handler source signals (`input_fn`, `np.load`, accepted content types) so you
can confirm before spending a transform. If `x-npy` is not supported, switch the
workflow to a JSONL manifest per the docs.

## Deliverables in this change

| Path | Purpose |
|------|---------|
| `scripts/inspect_and_patch_fourcastnet_inference.py` | Inspect/patch `model.tar.gz` for CUDA (no AWS calls). Exit 0=safe, 2=needs patch, 3=no handler. |
| `tests/test_inspect_and_patch_fourcastnet_inference.py` | Unit tests for scan/patch/tar-roundtrip (6 tests, all pass). |
| `scripts/run_fourcastnet_cpu_poc.ps1` | Gated end-to-end runner (STS → endpoints → input → CPU model → quota → workflow → run → output). Dry-run by default. |
| `configs/fourcastnet_batch_cpu_v1.json` | CPU batch config (`ml.m5.large`, CPU model name). |
| `workflows/fourcastnet_cpu_batch_transform_poc_v2.yaml` | Version-controlled copy of the MWAA workflow already created in AWS. |

## Quick test plan (run when AWS access is restored)

All PowerShell, from repo root, profile `sbnai-725`, region `us-east-1`.

```powershell
# 0. Validate the patcher logic locally (no AWS needed)
python -m pytest tests/test_inspect_and_patch_fourcastnet_inference.py -q

# 1. Full read-only dry run: identity, endpoint gate, input, inference.py
#    inspection, quota, workflow validation. Changes NOTHING in AWS.
./scripts/run_fourcastnet_cpu_poc.ps1

# 2a. If inference.py is CPU-safe: create the CPU model, then start the run.
./scripts/run_fourcastnet_cpu_poc.ps1 -CreateModel -Execute

# 2b. If inference.py hardcodes CUDA: a patched model.tar.gz is written to
#     artifacts/fcn-v1-cpu-model.tar.gz. REVIEW the change log, then:
./scripts/run_fourcastnet_cpu_poc.ps1 -CreateModel -AllowPatchUpload -Execute
#     (uploads patched artifact to a NEW key:
#      s3://<bucket>/sagemaker/fourcastnet/fcn-v1-cpu-poc/model/model.tar.gz)

# 3. Monitor
aws mwaa-serverless list-workflow-runs --workflow-arn <ARN> --profile sbnai-725 --region us-east-1 --output table
aws sagemaker list-transform-jobs --name-contains fourcastnet-cpu-poc --profile sbnai-725 --region us-east-1 `
  --query "TransformJobSummaries[].{Name:TransformJobName,Status:TransformJobStatus}" --output table

# 4. Output + endpoint post-flight
aws s3 ls s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/batch-transform/fourcastnet-poc/2026-06-06-18z/ --recursive --profile sbnai-725 --region us-east-1
aws sagemaker list-endpoints --profile sbnai-725 --region us-east-1 --output table
```

## Safety properties preserved
- **No endpoints, ever.** Pre- and post-flight gates stop on any
  FourCastNet/FCN/Chucaw endpoint. No `deploy`, no `CreateEndpoint`.
- **Original artifact is never overwritten.** Patched copies go to a distinct S3
  key and a distinct local file.
- **Dry-run by default.** Resource creation requires explicit `-CreateModel` /
  `-AllowPatchUpload` / `-Execute`.
- **CPU only.** Quota gate confirms `ml.m5.large` (GPU quotas are zero).

## One concrete next step
Restore AWS access and run step 1 (read-only dry run). Its `inference.py`
inspection output is the single missing fact that decides whether path 2a or 2b
applies.
