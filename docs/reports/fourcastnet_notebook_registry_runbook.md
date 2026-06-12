# FourCastNet Notebook/Studio Registry GPU Runbook

## Scope
Ruta oficial para validación de FourCastNet:
- SageMaker Studio/Notebook GPU
- Model Registry (`describe-model-package`)
- S3 artifacts (`ModelDataUrl` + `input_tensor.npy`)

Fuera de alcance:
- Endpoints (real-time/async)
- Processing
- Batch Transform
- Training

## Inputs
- Model Package ARN: `arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1`
- Input tensor S3: `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy`

## Preconditions
- Instancia Studio/Notebook con GPU disponible.
- Permisos de lectura a:
  - `sagemaker:DescribeModelPackage`
  - `s3:GetObject` para `ModelDataUrl` e input tensor.
- Dependencias del entorno incluyen `numpy`, `boto3` y backend FourCastNet (`modulus` o `physicsnemo` según imagen).

## Execution path
1. Abrir `notebooks/fourcastnet_registry_gpu_smoke.ipynb`.
2. Ejecutar celda de configuración.
3. Ejecutar `metadata_only`.
4. Revisar `artifacts/fourcastnet/notebook_smoke/notebook_metadata_only_report.json`.
5. Si metadata está OK, ejecutar `forward`.
6. Revisar `artifacts/fourcastnet/notebook_smoke/notebook_forward_report.json`.

## CLI alternative (same logic)
```powershell
python src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode metadata_only --region us-east-1 --output-report artifacts/fourcastnet/notebook_smoke/notebook_metadata_only_report.json
```
```powershell
python src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode forward --max-runtime-guard --region us-east-1 --output-report artifacts/fourcastnet/notebook_smoke/notebook_forward_report.json
```

## Acceptance criteria
- Metadata step:
  - `ok=true`
  - `result=metadata_collected`
  - `model_package_status=Completed`
  - `model_approval_status=PendingManualApproval` is acceptable.
- Forward step:
  - `ok=true`
  - `result=forward_succeeded`
  - `fourcastnet_proven=true`

## Interpretation
- `Completed + PendingManualApproval` means model package is registered and readable, but not automatically deploy-approved.
- Proof of model validity requires successful forward pass on GPU, not only registration metadata.
