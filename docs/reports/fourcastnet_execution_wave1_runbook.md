# FourCastNet SageMaker Wave-1 Runbook (Quota-aware Alternative)

> ⚠️ **DEFERRED / NOT RECOMMENDED FOR CURRENT GOAL** (async endpoint route only)
> Use the Studio/Notebook GPU route instead.
> El endpoint async listado como "Selected Execution Route #2" queda diferido indefinidamente.
> La ruta activa es exclusivamente Studio/Notebook GPU (Route #1 de este documento).

## Mandatory Guardrails
- Codex must not run AWS commands.
- Human-only AWS execution.
- Single-line PowerShell commands only (no line continuation).
- Do not approve model package yet.
- Endpoint must be temporary and deleted after test.
- Do not use SageMaker Processing, Batch Transform, or Training as runner in this wave.

## Current Quota Constraint
- Processing GPU quota (`ml.g4dn.xlarge`, `ml.g5.xlarge`): blocked (0 usage allowed).
- Batch Transform GPU quota (`ml.g4dn.xlarge`, `ml.g5.xlarge`): blocked (0 usage allowed).
- Training GPU quota (`ml.g4dn.xlarge`, `ml.g5.xlarge`): blocked (0 usage allowed).
- Endpoint GPU quota and Notebook/Studio GPU quota: available.

## Current Registry State
- Model Package Group: `sbnai-fourcastnet-fcn-v0`
- Model package version: `arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1`
- ModelPackageStatus: `Completed`
- ModelApprovalStatus: `PendingManualApproval`
- Status note: registered but not yet proven in runtime inference.

## Selected Execution Route
1. Notebook/Studio GPU smoke route (`ml.g4dn.xlarge` first, `ml.g5.xlarge` fallback).
2. Temporary Async Inference endpoint route with S3 input/output.
3. Cleanup endpoint resources immediately after smoke.

## Local Tooling Prepared
- Async endpoint payload generator: `src/fourcastnet/prepare_async_endpoint_payloads.py`
- Async invoke payload generator: `src/fourcastnet/prepare_async_invoke_payload.py`
- Notebook/Studio smoke helper: `src/fourcastnet/notebook_forward_smoke.py`
- Async-compatible serving entrypoint: `src/fourcastnet/serving/inference.py`

## Human Command Sheets
- Async endpoint manual commands: `docs/reports/fourcastnet_async_endpoint_manual_commands.md`
- Notebook/Studio GPU smoke guide: `docs/reports/fourcastnet_notebook_gpu_smoke.md`
- No-processing alternative plan: `docs/reports/fourcastnet_no_processing_alternative_plan.md`
