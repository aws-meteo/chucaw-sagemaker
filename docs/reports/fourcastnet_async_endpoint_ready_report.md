# FourCastNet Async Endpoint Ready Report

> ⚠️ **DEFERRED / NOT RECOMMENDED FOR CURRENT GOAL**
> Use the Studio/Notebook GPU route instead.
> Este reporte queda diferido indefinidamente para la etapa actual.
> La validacion recomendada de FourCastNet es Studio/Notebook GPU via Model Registry + S3, sin endpoint.

## Scope Completed
Prepared a no-Processing execution path using SageMaker Notebook/Studio GPU and temporary SageMaker Async Inference endpoint GPU.

## Files Added or Updated
- `src/fourcastnet/serving/inference.py` (async-safe serving logic with metadata_only/forward modes)
- `src/fourcastnet/serving/requirements.txt` (added `boto3`)
- `src/fourcastnet/prepare_async_endpoint_payloads.py` (local payload generator)
- `src/fourcastnet/prepare_async_invoke_payload.py` (local invoke payload generator)
- `src/fourcastnet/notebook_forward_smoke.py` (Notebook/Studio smoke script)
- `docs/reports/fourcastnet_notebook_gpu_smoke.md`
- `docs/reports/fourcastnet_async_endpoint_manual_commands.md`
- `docs/reports/fourcastnet_execution_wave1_runbook.md`
- `docs/reports/fourcastnet_no_processing_alternative_plan.md`

## Generated Payloads
- `artifacts/fourcastnet/aws_payloads/async_create_model.json`
- `artifacts/fourcastnet/aws_payloads/async_endpoint_config.json`
- `artifacts/fourcastnet/async_inputs/metadata_only_request.json`
- `artifacts/fourcastnet/async_inputs/forward_request.json`

## Local Validation Results
- Python compile check:
  - `python -m py_compile` passed for new/modified `src/fourcastnet/*.py` and `src/fourcastnet/serving/inference.py`.
- JSON parse check:
  - Generated payloads parsed successfully (`json-ok:7`).
- Command sheet presence:
  - `docs/reports/fourcastnet_async_endpoint_manual_commands.md` exists.
- AWS safety:
  - No AWS command was executed by Codex in this run.
  - Only local file generation and local validation commands were executed.

## Baseline Repo State Not Reverted
- Existing unrelated modifications were already present before this work, including:
  - `scripts/register_quicksight_abrupt_model_package.py`
  - `src/build_hosting_artifact.py`
- No pre-existing user changes were reverted.

## Recommended Human Path
1. Run Notebook/Studio `metadata_only` smoke on GPU.
2. Run Notebook/Studio `forward` smoke only if metadata-only passes.
3. Create temporary async endpoint and run metadata-only async invocation.
4. Run async forward invocation only after metadata-only async is healthy.
5. Delete endpoint and endpoint config immediately after test.

## First 5 Human Commands
```powershell
aws sts get-caller-identity --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker describe-model-package --model-package-name arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --profile sbnai-725 --region us-east-1
```
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/prepare_async_endpoint_payloads.py --model-name sbnai-fourcastnet-fcn-v0-async-model --endpoint-config-name sbnai-fourcastnet-fcn-v0-async-config --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --role-arn arn:aws:iam::725644097028:role/service-role/AmazonSageMakerAdminIAMExecutionRole --instance-type ml.g4dn.xlarge --initial-instance-count 1 --output-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-output/ --max-concurrent-invocations-per-instance 1 --profile sbnai-725 --region us-east-1
```
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/prepare_async_invoke_payload.py --input-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --output-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-output/ --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --content-type application/json --accept application/json --include-forward --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker create-model --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_create_model.json --profile sbnai-725 --region us-east-1
```

The model is registered but not yet proven. A successful metadata-only async invocation proves endpoint plumbing only. A successful forward invocation proves runtime inference.
