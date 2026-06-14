REALTIME_ENDPOINT_COST_RISK

# FourCastNet Cost Safety Checks (Human-run)

## DEFERRED / NOT RECOMMENDED
Chequeos orientados a endpoint async. Esta ruta queda diferida para la fase actual.
Usar validacion Studio/Notebook GPU con Model Registry + S3.

## One-line checks
```powershell
aws sagemaker list-endpoints --name-contains fourcastnet --profile sbnai-725 --region us-east-1 --query "Endpoints[].{Name:EndpointName,Status:EndpointStatus,Created:CreationTime}"
```
```powershell
aws sagemaker describe-endpoint --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --profile sbnai-725 --region us-east-1 --query "{EndpointStatus:EndpointStatus,Config:EndpointConfigName,LastModifiedTime:LastModifiedTime}"
```
```powershell
aws sagemaker describe-endpoint-config --endpoint-config-name sbnai-fourcastnet-fcn-v0-async-config --profile sbnai-725 --region us-east-1
```
```powershell
aws application-autoscaling describe-scalable-targets --service-namespace sagemaker --resource-ids endpoint/sbnai-fourcastnet-fcn-v0-async-endpoint/variant/AllTraffic --scalable-dimension sagemaker:variant:DesiredInstanceCount --profile sbnai-725 --region us-east-1
```
```powershell
aws application-autoscaling describe-scaling-policies --service-namespace sagemaker --resource-id endpoint/sbnai-fourcastnet-fcn-v0-async-endpoint/variant/AllTraffic --scalable-dimension sagemaker:variant:DesiredInstanceCount --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker list-notebook-instances --profile sbnai-725 --region us-east-1 --query "NotebookInstances[?NotebookInstanceStatus=='InService'].{Name:NotebookInstanceName,Type:InstanceType,Status:NotebookInstanceStatus}"
```
```powershell
aws sagemaker list-apps --profile sbnai-725 --region us-east-1 --query "Apps[?Status=='InService'].{DomainId:DomainId,UserProfileName:UserProfileName,AppType:AppType,AppName:AppName,Status:Status}"
```
```powershell
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/Endpoints/sbnai-fourcastnet-fcn-v0-async-endpoint --profile sbnai-725 --region us-east-1
```

## Emergency stop command
```powershell
aws sagemaker delete-endpoint --endpoint-name sbnai-fourcastnet-fcn-v0-async-endpoint --profile sbnai-725 --region us-east-1
```

