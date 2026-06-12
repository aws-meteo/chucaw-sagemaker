# FourCastNet No-Processing Alternative Plan

> ⚠️ **DEFERRED / NOT RECOMMENDED FOR CURRENT GOAL** (endpoint route)
> Use the Studio/Notebook GPU route instead.
> El endpoint route mencionado como disponible queda diferido indefinidamente.
> La ruta activa es exclusivamente Studio/Notebook GPU sin endpoint.

## Why Processing Is Blocked
GPU quotas for SageMaker Processing job types are currently zero for both `ml.g4dn.xlarge` and `ml.g5.xlarge`, so Processing cannot be used as execution runner.

## Why Batch Transform Is Blocked
GPU quotas for Batch Transform job types are currently zero for both `ml.g4dn.xlarge` and `ml.g5.xlarge`, so Transform cannot be used.

## Why Endpoint/Notebook Routes Are Available
- Endpoint usage quota is available for `ml.g4dn.xlarge` and `ml.g5.xlarge`.
- Notebook instance and Studio GPU quotas are available.
- Endpoint path is currently deferred; validation route uses Notebook/Studio GPU only.

## Recommended Order
1. Notebook `metadata_only` on GPU.
2. Notebook `forward` only if metadata-only passes.
3. Confirm `fourcastnet_proven=true` from forward report.

## What Counts As Model Proven
- Checkpoint loads successfully in runtime.
- Backend is present and model forward pass executes successfully.
- Forward result report marks `fourcastnet_proven=true` based on actual execution evidence.

## What Does NOT Count As Model Proven
- Model package registered in SageMaker Model Registry.
- Endpoint creation succeeds but no forward run.
- Metadata-only run succeeds.
- Any path that skips real model forward execution.
