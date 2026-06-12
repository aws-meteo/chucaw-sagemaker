# FourCastNet Manual AWS Commands (Human-Run Only)

## A) Login and Identity Check
```powershell
aws sso login --profile sbnai-725
```
```powershell
aws sts get-caller-identity --profile sbnai-725 --region us-east-1
```

## B) Local Artifact and Payload Generation
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/build_asset_manifest.py --assets-dir ../chucaw-glue-scripts/data/fourcastnet_assets_v0 --tensor-dir ../chucaw-glue-scripts/data/fourcastnet_tensor_real_v1 --output artifacts/fourcastnet/fcn_asset_manifest.json
```
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/stage_processing_code_bundle.py --staging-dir artifacts/fourcastnet/processing_code --overwrite
```
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/build_fcn_hosting_artifact.py --assets-dir ../chucaw-glue-scripts/data/fourcastnet_assets_v0 --serving-dir src/fourcastnet/serving --output-dir artifacts/fourcastnet/build
```
Placeholder: `<S3_BUCKET>` is the target S3 bucket for FCN artifacts and payload references; `<PYTORCH_INFERENCE_IMAGE_URI>` is the full ECR URI for the PyTorch inference image (for example with `<ACCOUNT_ID>` embedded in the account segment).
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/prepare_model_registry_payloads.py --model-package-group-name sbnai-fourcastnet-fcn-v0 --model-artifact-s3-uri s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/model/model.tar.gz --inference-image-uri <PYTORCH_INFERENCE_IMAGE_URI> --profile sbnai-725 --region us-east-1
```
Placeholder: `<SAGEMAKER_ROLE_ARN>` is the SageMaker execution role ARN for Processing; `<PYTORCH_PROCESSING_IMAGE_URI>` is the full ECR URI for the processing image; `<S3_BUCKET>` is the same bucket used for code/input/assets/output.
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/prepare_processing_job_payload.py --role-arn <SAGEMAKER_ROLE_ARN> --processing-image-uri <PYTORCH_PROCESSING_IMAGE_URI> --code-s3-uri s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/code/ --input-tensor-s3-uri s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/input/ --model-assets-s3-uri s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/assets/ --output-s3-uri s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/output/ --instance-type ml.g4dn.xlarge --volume-size-gb 100 --profile sbnai-725 --region us-east-1
```

## C) S3 Upload Commands
Placeholder: `<S3_BUCKET>` is the target bucket for model artifacts.
```powershell
aws s3 cp artifacts/fourcastnet/build/model.tar.gz s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/model/model.tar.gz --profile sbnai-725 --region us-east-1
```
Placeholder: `<S3_BUCKET>` is the target bucket for serving source artifact.
```powershell
aws s3 cp artifacts/fourcastnet/build/source.tar.gz s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/source/source.tar.gz --profile sbnai-725 --region us-east-1
```
Placeholder: `<S3_BUCKET>` is the target bucket for processing code bundle.
```powershell
aws s3 cp artifacts/fourcastnet/processing_code/ s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/code/ --recursive --profile sbnai-725 --region us-east-1
```
Placeholder: `<S3_BUCKET>` is the target bucket for the FCN smoke input tensor.
```powershell
aws s3 cp ../chucaw-glue-scripts/data/fourcastnet_tensor_real_v1/input_tensor.npy s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --profile sbnai-725 --region us-east-1
```
Placeholder: `<S3_BUCKET>` is the target bucket for FCN assets consumed by the processing smoke.
```powershell
aws s3 cp ../chucaw-glue-scripts/data/fourcastnet_assets_v0/backbone.ckpt s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/assets/backbone.ckpt --profile sbnai-725 --region us-east-1
```
Placeholder: `<S3_BUCKET>` is the target bucket for FCN assets consumed by the processing smoke.
```powershell
aws s3 cp ../chucaw-glue-scripts/data/fourcastnet_assets_v0/global_means.npy s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/assets/global_means.npy --profile sbnai-725 --region us-east-1
```
Placeholder: `<S3_BUCKET>` is the target bucket for FCN assets consumed by the processing smoke.
```powershell
aws s3 cp ../chucaw-glue-scripts/data/fourcastnet_assets_v0/global_stds.npy s3://<S3_BUCKET>/sagemaker/fourcastnet/fcn-v0/assets/global_stds.npy --profile sbnai-725 --region us-east-1
```

## D) Model Registry Commands
```powershell
aws sagemaker create-model-package-group --cli-input-json file://artifacts/fourcastnet/aws_payloads/model_package_group.json --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker create-model-package --cli-input-json file://artifacts/fourcastnet/aws_payloads/model_package.json --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker list-model-packages --model-package-group-name sbnai-fourcastnet-fcn-v0 --sort-by CreationTime --sort-order Descending --profile sbnai-725 --region us-east-1
```
```powershell
$MP_ARN=(aws sagemaker list-model-packages --model-package-group-name sbnai-fourcastnet-fcn-v0 --sort-by CreationTime --sort-order Descending --max-results 1 --query "ModelPackageSummaryList[0].ModelPackageArn" --output text --profile sbnai-725 --region us-east-1); aws sagemaker describe-model-package --model-package-name $MP_ARN --profile sbnai-725 --region us-east-1
```

## E) Processing Job Commands
```powershell
aws sagemaker create-processing-job --cli-input-json file://artifacts/fourcastnet/aws_payloads/processing_job.json --profile sbnai-725 --region us-east-1
```
```powershell
$PROCESSING_JOB_NAME=(Get-Content artifacts/fourcastnet/aws_payloads/processing_job.json | ConvertFrom-Json).ProcessingJobName; aws sagemaker describe-processing-job --processing-job-name $PROCESSING_JOB_NAME --profile sbnai-725 --region us-east-1
```
```powershell
aws sagemaker list-processing-jobs --name-contains sbnai-fourcastnet-fcn-smoke --sort-by CreationTime --sort-order Descending --profile sbnai-725 --region us-east-1
```
```powershell
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/ProcessingJobs --profile sbnai-725 --region us-east-1
```
```powershell
aws logs tail /aws/sagemaker/ProcessingJobs --since 30m --follow --profile sbnai-725 --region us-east-1
```
```powershell
$PROCESSING_JOB_NAME=(Get-Content artifacts/fourcastnet/aws_payloads/processing_job.json | ConvertFrom-Json).ProcessingJobName; aws sagemaker stop-processing-job --processing-job-name $PROCESSING_JOB_NAME --profile sbnai-725 --region us-east-1
```

## F) Cost Guardrails
```powershell
aws sagemaker list-processing-jobs --max-results 20 --sort-by CreationTime --sort-order Descending --profile sbnai-725 --region us-east-1
```
```powershell
$PROCESSING_JOB_NAME=(Get-Content artifacts/fourcastnet/aws_payloads/processing_job.json | ConvertFrom-Json).ProcessingJobName; aws sagemaker describe-processing-job --processing-job-name $PROCESSING_JOB_NAME --profile sbnai-725 --region us-east-1
```
Manual reminder: if a job is stuck or no longer needed, run the emergency stop command immediately.
