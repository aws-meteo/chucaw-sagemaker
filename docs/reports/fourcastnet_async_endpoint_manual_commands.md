REALTIME_ENDPOINT_COST_RISK

# FourCastNet Async Endpoint Manual Commands (Human-run)

> ⚠️ **DEFERRED / NOT RECOMMENDED FOR CURRENT GOAL**
> Use the Studio/Notebook GPU route instead.
> Ruta de endpoint async diferida indefinidamente.
> La ruta recomendada es Studio/Notebook GPU con Model Registry + S3 artifacts.
> No ejecutar ninguno de los comandos de creación de endpoint de este documento.

## Persistent but scale-to-zero async endpoint
Objetivo: mantener un endpoint async disponible para notebooks/CLI, pero minimizar costo idle con autoscaling `MinCapacity=0` y `MaxCapacity=1`.

1) Identity check
```powershell
aws sts get-caller-identity --profile sbnai-725 --region us-east-1
```

2) Create model
```powershell
aws sagemaker create-model --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_create_model.json --profile sbnai-725 --region us-east-1
```

3) Create endpoint config
```powershell
aws sagemaker create-endpoint-config --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_endpoint_config.json --profile sbnai-725 --region us-east-1
```

4) Create endpoint
```powershell
aws sagemaker create-endpoint --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --endpoint-config-name sbnai-fourcastnet-fcn-v0-async-config --tags Key=Project,Value=SbnAI Key=Component,Value=FourCastNet Key=Environment,Value=dev Key=Owner,Value=Fabian Key=CostCenter,Value=chucaw --profile sbnai-725 --region us-east-1
```

5) Wait/describe endpoint
```powershell
aws sagemaker wait endpoint-in-service --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker describe-endpoint --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --profile sbnai-725 --region us-east-1 --query "{EndpointStatus:EndpointStatus,EndpointArn:EndpointArn,LastModifiedTime:LastModifiedTime}"
```

6) Register scalable target min=0 max=1
```powershell
aws application-autoscaling register-scalable-target --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_register_scalable_target.json --profile sbnai-725 --region us-east-1
```

7) Attach scale-from-zero/backlog scaling policy
```powershell
aws application-autoscaling put-scaling-policy --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_scale_from_zero_policy.json --profile sbnai-725 --region us-east-1
```
```powershell
aws application-autoscaling put-scaling-policy --cli-input-json file://artifacts/fourcastnet/aws_payloads/async_backlog_target_tracking_policy.json --profile sbnai-725 --region us-east-1
```
```powershell
aws cloudwatch put-metric-alarm --alarm-name sbnai-fourcastnet-fcn-v0-async-has-backlog-without-capacity --metric-name HasBacklogWithoutCapacity --namespace AWS/SageMaker --statistic Average --period 60 --evaluation-periods 2 --datapoints-to-alarm 2 --threshold 1 --comparison-operator GreaterThanOrEqualToThreshold --treat-missing-data missing --dimensions Name=EndpointName,Value=sbnai-fourcastnet-fcn-v0-async-endpoint --alarm-actions $(aws application-autoscaling describe-scaling-policies --service-namespace sagemaker --resource-id endpoint/sbnai-fourcastnet-fcn-v0-async-endpoint/variant/AllTraffic --scalable-dimension sagemaker:variant:DesiredInstanceCount --profile sbnai-725 --region us-east-1 --query "ScalingPolicies[?PolicyName=='HasBacklogWithoutCapacity-ScalingPolicy'].PolicyARN | [0]" --output text) --profile sbnai-725 --region us-east-1
```

8) Describe scalable target
```powershell
aws application-autoscaling describe-scalable-targets --service-namespace sagemaker --resource-ids endpoint/sbnai-fourcastnet-fcn-v0-async-endpoint/variant/AllTraffic --scalable-dimension sagemaker:variant:DesiredInstanceCount --profile sbnai-725 --region us-east-1
```
```powershell
aws application-autoscaling describe-scaling-policies --service-namespace sagemaker --resource-id endpoint/sbnai-fourcastnet-fcn-v0-async-endpoint/variant/AllTraffic --scalable-dimension sagemaker:variant:DesiredInstanceCount --profile sbnai-725 --region us-east-1
```

9) Invoke metadata_only
```powershell
aws s3 cp artifacts/fourcastnet/async_inputs/notebook_metadata_only_request.json s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-input/notebook_metadata_only_request.json --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker-runtime invoke-endpoint-async --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --input-location s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-input/notebook_metadata_only_request.json --content-type application/json --accept application/json --profile sbnai-725 --region us-east-1
```

10) Check async output S3
```powershell
aws s3 ls s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-output/ --recursive --profile sbnai-725 --region us-east-1
```

11) Invoke forward only after metadata passes
```powershell
aws s3 cp artifacts/fourcastnet/async_inputs/notebook_forward_request.json s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-input/notebook_forward_request.json --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker-runtime invoke-endpoint-async --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --input-location s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/async-input/notebook_forward_request.json --content-type application/json --accept application/json --profile sbnai-725 --region us-east-1
```

12) Cleanup endpoint if cost concern
```powershell
aws sagemaker delete-endpoint --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --profile sbnai-725 --region us-east-1
```

13) Delete endpoint config if cleaning up
```powershell
aws sagemaker delete-endpoint-config --endpoint-config-name sbnai-fourcastnet-fcn-v0-async-config --profile sbnai-725 --region us-east-1
```

14) Optional delete model
```powershell
aws sagemaker delete-model --model-name sbnai-fourcastnet-fcn-v0-async-model --profile sbnai-725 --region us-east-1
```

## Cost warnings
- Endpoint creation may start one GPU instance because endpoint config uses `InitialInstanceCount=1`.
- After autoscaling `MinCapacity=0` is active and queue is empty, the endpoint should scale down to zero.
- Cold start after scale-to-zero can take several minutes before first request is served.
- If autoscaling or CloudWatch alarm is misconfigured, endpoint can continue billing.
- Always verify desired/current instance count after setup using `describe-endpoint` and `describe-scalable-targets`.

