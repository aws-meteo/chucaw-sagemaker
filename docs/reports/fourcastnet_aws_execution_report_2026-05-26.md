# FourCastNet AWS Execution Report (2026-05-26)

## Executive Summary
- Se ejecutó la secuencia en orden desde terminal de Codex con acceso fuera del sandbox.
- Cargas S3: completadas.
- Model Package Group: creado.
- Model Package: creado (tras corregir payload e image tag).
- Processing Job: no creado por cuota de SageMaker Processing en 0 para `ml.g4dn.xlarge` y `ml.g5.xlarge`.

## Variables usadas en ejecución
- `ACCOUNT_ID=725644097028`
- `S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an`
- `SAGEMAKER_ROLE_ARN=arn:aws:iam::725644097028:role/service-role/AmazonSageMakerAdminIAMExecutionRole`
- `PYTORCH_INFERENCE_IMAGE_URI` inicial: `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker`
- `PYTORCH_PROCESSING_IMAGE_URI`: `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker`

## Command-by-Command Outcome

### 1) Uploads S3 (OK)
- `aws s3 cp ...model.tar.gz...` -> OK
- `aws s3 cp ...source.tar.gz...` -> OK
- `aws s3 cp ...processing_code/ --recursive...` -> OK
- `aws s3 cp ...input_tensor.npy...` -> OK
- `aws s3 cp ...backbone.ckpt...` -> OK
- `aws s3 cp ...global_means.npy...` -> OK
- `aws s3 cp ...global_stds.npy...` -> OK

Evidencia de objetos en S3:
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/model/model.tar.gz`
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/source/source.tar.gz`
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/code/processing_entrypoint.py`
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/code/bundle_manifest.json`
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy`
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/backbone.ckpt`
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/global_means.npy`
- `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/global_stds.npy`

### 2) Model Registry
- `create-model-package-group` -> OK
  - ARN: `arn:aws:sagemaker:us-east-1:725644097028:model-package-group/sbnai-fourcastnet-fcn-v0`

- `create-model-package` primer intento -> FAIL
  - Error: `Tags are not supported in Model Package versions`
  - Acción correctiva: quitar `Tags` de `artifacts/fourcastnet/aws_payloads/model_package.json`.

- `create-model-package` segundo intento -> FAIL
  - Error: image no encontrada para tag `...pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker`
  - Acción correctiva: consulta ECR y actualización del image tag a `763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310`.

- `create-model-package` tercer intento -> OK
  - ARN: `arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1`

- `list-model-packages` -> OK
  - Version 1, status `Completed`, approval `PendingManualApproval`.

- `describe-model-package` -> OK
  - Confirmado `ModelDataUrl` en S3 y `ImageDigest` resuelto.

### 3) Processing Job
- `create-processing-job` -> FAIL
  - Error: `ResourceLimitExceeded`
  - Detalle: cuota de cuenta para `ml.g4dn.xlarge for processing job usage` = `0`, delta solicitado `1`.

- Verificación de fallback (`ml.g5.xlarge`) en Service Quotas:
  - `ml.g5.xlarge for processing job usage` = `0`
  - `ml.g4dn.xlarge for processing job usage` = `0`
  - Conclusión: no hay capacidad de arranque para processing GPU en esta cuenta/región actualmente.

- `describe-processing-job` -> FAIL esperado (job inexistente)
- `list-processing-jobs --name-contains sbnai-fourcastnet-fcn-smoke` -> `[]`
- `aws logs tail /aws/sagemaker/ProcessingJobs ...` -> FAIL esperado (log group no existe al no existir job)

## Cambios técnicos aplicados durante la ejecución
- Se corrigió payload runtime de model package:
  - Remoción de `Tags` (incompatibles con `create-model-package` en esta cuenta/API).
  - Image tag de inference ajustado a uno existente.
- Se dejó hardening persistente en código:
  - `src/fourcastnet/prepare_model_registry_payloads.py` actualizado para no emitir `Tags` en payload de model package.

## Estado final
- Model Registry listo y operativo:
  - Group: creado.
  - Package v1: creado y en `PendingManualApproval`.
- Processing smoke job: bloqueado por cuota AWS (no por payload ni por permisos).

## Siguiente acción requerida (humano)
- Solicitar aumento de cuota en `us-east-1` para SageMaker Processing GPU:
  - `ml.g4dn.xlarge for processing job usage`
  - `ml.g5.xlarge for processing job usage`
- Tras aprobación de cuota, reintentar `aws sagemaker create-processing-job --cli-input-json file://artifacts/fourcastnet/aws_payloads/processing_job.json --profile sbnai-725 --region us-east-1`.
