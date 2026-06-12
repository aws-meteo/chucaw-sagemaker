# FourCastNet Notebook Registry Ready Report

## Status
FourCastNet is registered but not yet proven; proof requires successful forward pass in Studio/Notebook GPU.

## Scope implemented
- `src/fourcastnet/notebook_forward_smoke.py` hardened for Model Registry flow.
- `notebooks/fourcastnet_registry_gpu_smoke.ipynb` created for metadata and forward smoke sequence.
- `docs/reports/fourcastnet_notebook_registry_runbook.md` created as execution runbook.
- Async endpoint docs marked as deferred/not recommended for current phase.

## Key behavior
- Uses `describe-model-package` (read-only) to fetch:
  - `ModelPackageStatus`
  - `ModelApprovalStatus`
  - `ModelDataUrl`
- Accepts `ModelApprovalStatus=PendingManualApproval`.
- Loads model artifacts from `ModelDataUrl` tarball and input tensor from S3.
- Executes:
  - `metadata_only` (plumbing and metadata)
  - `forward` (proof path)

## Non-goals enforced
- No model package approval.
- No endpoint create/invoke/autoscale/delete.
- No Processing, Batch Transform, or Training usage.
