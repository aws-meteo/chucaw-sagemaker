# FourCastNet Notebook Execution Report

> Generated: 2026-05-27T04:51:00Z (UTC-4 / local time 2026-05-27T00:51)
> Author: Antigravity agent (automated)

---

## 1. Endpoint Route: Corrected and Deferred

The async endpoint route has been **explicitly deferred** as not recommended for the current goal.

All 5 async endpoint documents now carry the required banner at the top:

```
⚠️ DEFERRED / NOT RECOMMENDED FOR CURRENT GOAL
Use the Studio/Notebook GPU route instead.
```

Documents updated:
- `docs/reports/fourcastnet_async_endpoint_manual_commands.md`
- `docs/reports/fourcastnet_async_endpoint_ready_report.md`
- `docs/reports/fourcastnet_async_scale_to_zero_ready_report.md`
- `docs/reports/fourcastnet_execution_wave1_runbook.md`
- `docs/reports/fourcastnet_no_processing_alternative_plan.md`

**Current active route:** SageMaker Studio/Notebook GPU + Model Registry + S3 artifacts. No endpoint is created or used.

---

## 2. Read-Only AWS Checks

All 5 AWS read-only checks executed successfully.

### 2.1 Identity check
```
Command: aws sts get-caller-identity --profile sbnai-725 --region us-east-1

Result:
{
    "UserId": "AROA2R46CGYCEIVEYBSUE:fabian_trigo",
    "Account": "725644097028",
    "Arn": "arn:aws:sts::725644097028:assumed-role/AWSReservedSSO_AdministratorAccess_bd0d038c02eeb95c/fabian_trigo"
}
```

**Status:** ✅ Authenticated as fabian_trigo, account 725644097028.

---

### 2.2 Describe model package
```
Command: aws sagemaker describe-model-package --model-package-name arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --profile sbnai-725 --region us-east-1
```

Key fields returned:

| Field | Value |
|-------|-------|
| ModelPackageGroupName | sbnai-fourcastnet-fcn-v0 |
| ModelPackageVersion | 1 |
| ModelPackageArn | arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 |
| ModelPackageStatus | **Completed** |
| ModelApprovalStatus | **PendingManualApproval** |
| ModelDataUrl | s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/model/model.tar.gz |
| Image | 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310 |
| ModelDataETag | 7149b5066f8a1aa8ea0af2c8f93f8dbe-99 (multipart, ~1 GB) |
| CustomerMetadata.EndpointRequired | false |

**Status:** ✅ Model package is readable. ModelDataUrl resolved. PendingManualApproval is acceptable for notebook usage.

---

### 2.3 List FourCastNet endpoints
```
Command: aws sagemaker list-endpoints --name-contains fourcastnet --profile sbnai-725 --region us-east-1

Result: {"Endpoints": []}
```

**Status:** ✅ No FourCastNet endpoints exist. Zero cost from endpoints.

---

### 2.4 List notebook instances InService
```
Command: aws sagemaker list-notebook-instances --status-equals InService --profile sbnai-725 --region us-east-1

Result: {"NotebookInstances": []}
```

**Status:** ✅ No notebook instances currently InService. Zero cost from notebook instances.

---

### 2.5 List Studio apps InService
```
Command: aws sagemaker list-apps --profile sbnai-725 --region us-east-1 --query "Apps[?Status=='InService'].[...]" --output table

Result: (empty table)
```

**Status:** ✅ No Studio apps currently InService. Zero cost from Studio.

---

## 3. Local metadata_only Result

> Script: `src/fourcastnet/notebook_forward_smoke.py`
> Mode: `metadata_only`
> Profile: `sbnai-725`

The script was invoked with the following command (single line, no continuations):
```
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode metadata_only --region us-east-1 --profile sbnai-725 --output-report-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/local_metadata_only_report.json
```

### Result: ✅ PASS — `ok: true`, `result: metadata_collected`

Full JSON report captured:

```json
{
  "timestamp_utc": "2026-05-27T04:49:14.115548+00:00",
  "mode": "metadata_only",
  "ok": true,
  "fourcastnet_proven": false,
  "model_package_status": "Completed",
  "model_approval_status": "PendingManualApproval",
  "model_data_url": "s3://.../sagemaker/fourcastnet/fcn-v0/model/model.tar.gz",
  "model_tar_size_bytes": 829298764,
  "result": "metadata_collected",
  "report_uploaded_to_s3": "s3://.../notebook-smoke/local_metadata_only_report.json"
}
```

Key findings from the run:

| Field | Value |
|-------|-------|
| `ok` | `true` |
| `result` | `metadata_collected` |
| `fourcastnet_proven` | `false` (correct — metadata_only does not run forward) |
| model.tar.gz size | **829,298,764 bytes (~791 MB)** |
| `backbone.ckpt` exists | ✅ **true** |
| `global_means.npy` exists | ✅ **true** |
| `global_stds.npy` exists | ✅ **true** |
| input_tensor shape | `[1, 20, 720, 1440]` float32, finite, nan_count=0 |
| global_means shape | `[1, 21, 1, 1]` float32, finite |
| global_stds shape | `[1, 21, 1, 1]` float32, finite |
| torch_available (local) | `false` (no torch in aws_backend env — expected) |
| cuda_available (local) | `false` (local PC, no GPU — expected) |
| backend_probe `ok` | `false` — `modulus`, `physicsnemo`, `fourcastnet` NOT installed locally (expected) |
| Report uploaded to S3 | ✅ **Yes** |

**The backend_not_found on local PC is expected and is NOT an AWS failure.**
`modulus` / `physicsnemo` must be installed in the Studio/Notebook GPU environment before running `--mode forward`.

---

## 4. FourCastNet Endpoint Status

| Check | Result |
|-------|--------|
| Endpoints with `fourcastnet` in name | **0** (none exist) |
| Endpoint cost incurring | **No** |

---

## 5. Notebook and Studio App Status

| Resource | Count InService | Cost |
|----------|----------------|------|
| Notebook instances | 0 | $0 |
| Studio JupyterLab apps | 0 | $0 |

**Nothing related to FourCastNet is currently incurring cost.**

---

## 6. Model Package Readability

- **Model package is readable:** ✅ `describe-model-package` returned full metadata.
- **ModelDataUrl resolved:** ✅ `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/model/model.tar.gz`
- **PendingManualApproval:** Acceptable for notebook usage. The script calls `describe-model-package` (read-only); approval is not required to read artifacts.

---

## 7. Artifact Download and Verification

| Artifact | Status |
|----------|--------|
| model.tar.gz download | ✅ **829 MB downloaded successfully** |
| model.tar.gz extraction | ✅ Extracted to temp dir |
| `backbone.ckpt` exists | ✅ **true** |
| `global_means.npy` exists | ✅ **true** |
| `global_stds.npy` exists | ✅ **true** |
| `input_tensor.npy` download | ✅ Shape `[1, 20, 720, 1440]`, finite, no NaN |

---

## 8. Torch and CUDA Status

| Environment | torch | CUDA |
|-------------|-------|------|
| Local PC (`aws_backend` env) | `false` (not installed) | `false` (no GPU) |
| Studio/Notebook GPU (expected) | Must be present | Must be `true` on ml.g4dn.xlarge |

**CUDA is only required for `--mode forward` on Studio/Notebook GPU.**
The `metadata_only` run confirmed all artifact plumbing works without torch.

---

## 9. S3 Report Upload

✅ **Report uploaded successfully** to:
```
s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/local_metadata_only_report.json
```

Local copy also written to:
```
artifacts/fourcastnet/notebook_smoke/notebook_forward_smoke_report.json
```

---

## 10. Next Command for Studio/Notebook GPU

Once a Studio JupyterLab app or Notebook instance is launched on `ml.g4dn.xlarge`:

### Step A — metadata_only (run first)
```bash
python src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode metadata_only --region us-east-1 --output-report-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/studio_metadata_only_report.json
```

### Step B — forward (only after metadata_only PASS)
```bash
python src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode forward --region us-east-1 --output-report-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/studio_forward_report.json
```

---

## 11. Local Validation (Task 6)

| Check | Result |
|-------|--------|
| `py_compile src/fourcastnet/notebook_forward_smoke.py` | ✅ **PASSED** (exit 0) |
| `notebooks/fourcastnet_registry_gpu_smoke.ipynb` JSON parse | ✅ **PASSED** (5 cells, valid JSON) |
| `git status --short` | 8 modified tracked files; 17+ untracked FourCastNet/docs files (all new work) |
| `git diff --stat HEAD` | 8 files changed, 589 insertions(+), 58 deletions(-) in tracked baseline files |

---

## 12. Explicit Final Status

| Status Item | Value |
|-------------|-------|
| **FourCastNet registered** | ✅ **Yes** — model package `sbnai-fourcastnet-fcn-v0/1` status `Completed` |
| **FourCastNet endpoint** | ✅ **No** — zero endpoints exist; zero cost |
| **FourCastNet proven** | ❌ **No** — forward pass not yet executed; `fourcastnet_proven=false` until a Studio/Notebook GPU forward run succeeds |
| **Current FourCastNet cost** | ✅ **Zero** — no endpoints, no notebook instances, no Studio apps InService |
| **ModelDataUrl resolved** | ✅ **Yes** |
| **Report uploaded to S3** | ✅ **Yes** — `s3://.../notebook-smoke/local_metadata_only_report.json` |
| **Next step** | Launch Studio JupyterLab on ml.g4dn.xlarge; run metadata_only, then forward |
