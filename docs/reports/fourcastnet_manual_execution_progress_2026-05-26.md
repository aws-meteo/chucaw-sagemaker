# FourCastNet Manual Execution Progress Report (2026-05-26)

## 1) Estado actual
- La preparación local de FourCastNet para SageMaker avanzó correctamente.
- Los artefactos locales y payloads JSON necesarios ya existen.
- No se ejecutaron comandos AWS de creación/subida desde Codex.

## 2) Variables operativas definidas
- `ACCOUNT_ID=725644097028`
- `S3_BUCKET=chucaw-data-platinum-processed-725644097028-us-east-1-an`
- `SAGEMAKER_ROLE_ARN=arn:aws:iam::725644097028:role/service-role/AmazonSageMakerAdminIAMExecutionRole`
- `PYTORCH_INFERENCE_IMAGE_URI=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker`
- `PYTORCH_PROCESSING_IMAGE_URI=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker`

## 3) Comandos ya ejecutados por humano (confirmados)
- `build_asset_manifest.py`: `PASS`.
- `stage_processing_code_bundle.py`: staging generado en `artifacts/fourcastnet/processing_code`.
- `build_fcn_hosting_artifact.py`: generados `model.tar.gz` y `source.tar.gz`.
- `prepare_model_registry_payloads.py`: generados:
  - `artifacts/fourcastnet/aws_payloads/model_package_group.json`
  - `artifacts/fourcastnet/aws_payloads/model_package.json`
- `prepare_processing_job_payload.py`: generado:
  - `artifacts/fourcastnet/aws_payloads/processing_job.json`

## 4) Verificaciones locales ejecutadas por Codex
- Re-definición local de variables en terminal de trabajo: OK.
- Inventario de artefactos en `artifacts/fourcastnet/`: OK.
- Validación de contenido de tarballs:
  - `model.tar.gz` contiene exactamente `backbone.ckpt`, `global_means.npy`, `global_stds.npy`.
  - `source.tar.gz` contiene exactamente `inference.py`, `requirements.txt`.
- Validación semántica de payloads:
  - `ModelPackageGroupName == sbnai-fourcastnet-fcn-v0`: OK.
  - `ModelApprovalStatus == PendingManualApproval`: OK.
  - `processing instance == ml.g4dn.xlarge`: OK.
  - `processing volume == 100 GB`: OK.
  - `ProcessingInputs == [code, tensor, assets]`: OK.
  - Tags requeridos presentes (5/5) en model package y processing payload: OK.

## 5) Nota de calidad detectada
- En la ejecución humana de `prepare_processing_job_payload.py` se pasó `--model-assets-s3-uri` dos veces (una con `/input/` y otra con `/assets/`).
- Resultado observado: el payload final quedó correcto apuntando a `/assets/` (último argumento prevalece).
- Recomendación: usar el comando canónico sin argumentos duplicados para evitar errores silenciosos.

## 6) Próximos pasos (human-only AWS), orden estricto
1. Subir artefactos y entradas a S3 (`model.tar.gz`, `source.tar.gz`, `processing_code/`, `input_tensor.npy`, `backbone.ckpt`, `global_means.npy`, `global_stds.npy`).
2. Ejecutar `create-model-package-group` con `model_package_group.json`.
3. Ejecutar `create-model-package` con `model_package.json`.
4. Ejecutar `create-processing-job` con `processing_job.json`.
5. Monitorear con `describe-processing-job`, `list-processing-jobs` y `aws logs tail`.
6. Si el job queda colgado, detener con `stop-processing-job`.

## 7) Restricción de seguridad aplicada
- Codex no autenticó SSO.
- Codex no ejecutó `aws s3 cp`.
- Codex no ejecutó `create-processing-job`, `create-model-package-group`, `create-model-package`, ni operaciones de borrado/endpoint.
- Toda la actividad de Codex fue local/read-only o generación local de archivos.
