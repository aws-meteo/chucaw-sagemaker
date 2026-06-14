# FourCastNet Notebook/Studio GPU Smoke (Sin Processing)

## Objetivo
Probar carga de assets y forward de FourCastNet dentro de SageMaker Studio/Notebook con GPU disponible (`ml.g4dn.xlarge` recomendado, `ml.g5.xlarge` fallback), sin usar Processing, Batch Transform ni Training.

## Requisitos
- Cuenta/Region: `725644097028` en `us-east-1`
- Rol SageMaker: `arn:aws:iam::725644097028:role/service-role/AmazonSageMakerAdminIAMExecutionRole`
- Input tensor S3: `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy`
- Assets S3 prefix: `s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/`

## Flujo recomendado
1. Levantar Studio/Notebook con `ml.g4dn.xlarge`.
2. Descargar `model.tar.gz` o assets individuales (`backbone.ckpt`, `global_means.npy`, `global_stds.npy`) y `input_tensor.npy`.
3. Verificar version de Python/PyTorch/CUDA.
4. Ejecutar `metadata_only`.
5. Ejecutar `forward` solo si metadata-only pasa.
6. Guardar reporte JSON local o en S3.
7. Cerrar app/notebook al finalizar.

## Comandos ejemplo dentro de Studio/Notebook
```bash
python src/fourcastnet/notebook_forward_smoke.py --input-tensor s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --checkpoint s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/backbone.ckpt --global-means s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/global_means.npy --global-stds s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/global_stds.npy --mode metadata_only --output-report s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-output/notebook_metadata_report.json
```

```bash
python src/fourcastnet/notebook_forward_smoke.py --input-tensor s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --checkpoint s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/backbone.ckpt --global-means s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/global_means.npy --global-stds s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/assets/global_stds.npy --mode forward --max-runtime-guard --output-report s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-output/notebook_forward_report.json
```

## Criterio de resultado
- `metadata_only` exitoso: prueba paths/lectura/input stats/disponibilidad GPU, pero no prueba inferencia real.
- `forward` exitoso: prueba inferencia runtime real; recien ahi `fourcastnet_proven=true`.

## Cierre
- Detener Notebook Instance o cerrar Studio JupyterLab App al terminar para evitar costo.
