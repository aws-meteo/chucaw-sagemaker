REALTIME_ENDPOINT_COST_RISK

# FourCastNet Async Scale-to-Zero Ready Report

> ⚠️ **DEFERRED / NOT RECOMMENDED FOR CURRENT GOAL**
> Use the Studio/Notebook GPU route instead.
> Esta ruta de endpoint async con autoscaling queda diferida indefinidamente para la fase actual.
> La ruta recomendada es prueba de registro en Studio/Notebook GPU sin endpoint.

## Status
- FourCastNet endpoint has not been created by Codex.
- The intended endpoint mode is async inference with autoscaling MinCapacity=0.
- Se prepararon payloads locales y comandos humanos para crear endpoint async y aplicar autoscaling a cero.

## Files changed
- `src/fourcastnet/prepare_async_endpoint_payloads.py`
- `src/fourcastnet/prepare_async_autoscaling_payloads.py`
- `src/fourcastnet/invoke_async_from_notebook.py`
- `docs/reports/fourcastnet_async_endpoint_manual_commands.md`
- `docs/reports/fourcastnet_cost_safety_checks.md`
- `docs/reports/fourcastnet_async_scale_to_zero_ready_report.md`

## Payloads generated
- `artifacts/fourcastnet/aws_payloads/async_create_model.json`
- `artifacts/fourcastnet/aws_payloads/async_endpoint_config.json`
- `artifacts/fourcastnet/aws_payloads/async_register_scalable_target.json`
- `artifacts/fourcastnet/aws_payloads/async_scale_from_zero_policy.json`
- `artifacts/fourcastnet/aws_payloads/async_backlog_target_tracking_policy.json`
- `artifacts/fourcastnet/async_inputs/notebook_metadata_only_request.json`
- `artifacts/fourcastnet/async_inputs/notebook_forward_request.json`

## Validation results (local only)
- `py_compile`: OK for `src/fourcastnet/*.py` and `src/fourcastnet/serving/*.py`.
- JSON parse: `json-ok:12` para payloads en `artifacts/fourcastnet/aws_payloads/*.json` y `artifacts/fourcastnet/async_inputs/*.json`.
- `git diff --stat`: ejecutado.
- `git status --short`: ejecutado.
- No AWS mutating command was executed by Codex.

## Remaining risks
- `InitialInstanceCount=1` en endpoint config puede iniciar 1 GPU al crear endpoint.
- Si no se aplica autoscaling + alarma de `HasBacklogWithoutCapacity`, el endpoint puede quedar facturando idle.
- El scale-to-zero introduce cold start de varios minutos.
- La política StepScaling de scale-from-zero requiere CloudWatch alarm asociada al Policy ARN.

## First 6 human commands
```powershell
aws sts get-caller-identity --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker create-model --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_create_model.json --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker create-endpoint-config --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_endpoint_config.json --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker create-endpoint --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --endpoint-config-name sbnai-fourcastnet-fcn-v0-async-config --tags Key=Project,Value=SbnAI Key=Component,Value=FourCastNet Key=Environment,Value=dev Key=Owner,Value=Fabian Key=CostCenter,Value=chucaw --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker wait endpoint-in-service --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --profile sbnai-725 --region us-east-1
```
```powershell
aws application-autoscaling register-scalable-target --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_register_scalable_target.json --profile sbnai-725 --region us-east-1
```

