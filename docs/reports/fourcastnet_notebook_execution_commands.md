# FourCastNet Notebook Execution Commands

> **Current Route: SageMaker Studio/Notebook GPU + Model Registry + S3 artifacts**
> No endpoint is created or used.

---

## 1. Preflight – from local terminal

### 1.1 Identity check
```powershell
aws sts get-caller-identity --profile sbnai-725 --region us-east-1
```

### 1.2 Describe model package (read-only)
```powershell
aws sagemaker describe-model-package --model-package-name arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --profile sbnai-725 --region us-east-1
```

### 1.3 List endpoints with fourcastnet in name (expect empty)
```powershell
aws sagemaker list-endpoints --name-contains fourcastnet --profile sbnai-725 --region us-east-1
```

### 1.4 List notebook instances InService
```powershell
aws sagemaker list-notebook-instances --status-equals InService --profile sbnai-725 --region us-east-1
```

### 1.5 List Studio apps InService
```powershell
aws sagemaker list-apps --profile sbnai-725 --region us-east-1 --query "Apps[?Status=='InService'].[AppName,AppType,DomainId,UserProfileName]" --output table
```

---

## 2. Local metadata_only dry run

> Run from the repo root on local PC. Does NOT run forward pass.
> Downloads model.tar.gz and input_tensor.npy; writes and uploads a JSON report.

### 2.1 Run metadata_only with profile and S3 report upload
```powershell
C:\ProgramData\miniconda3\envs\aws_backend\python.exe src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode metadata_only --region us-east-1 --profile sbnai-725 --output-report-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/local_metadata_only_report.json
```

### 2.2 Verify report was uploaded to S3
```powershell
aws s3 ls s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/ --profile sbnai-725 --region us-east-1
```

### 2.3 Download and inspect report locally
```powershell
aws s3 cp s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/local_metadata_only_report.json artifacts/fourcastnet/notebook_smoke/local_metadata_only_report.json --profile sbnai-725 --region us-east-1
```

---

## 3. Studio/Notebook GPU execution

> Open a SageMaker Studio JupyterLab app or Notebook instance on **ml.g4dn.xlarge** (primary) or **ml.g5.xlarge** (fallback).
> Clone or sync the repo. Run commands below inside the terminal of the Studio/Notebook.

### 3.1 Run metadata_only inside Studio/Notebook (no --profile)
```bash
python src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode metadata_only --region us-east-1 --output-report-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/studio_metadata_only_report.json
```

### 3.2 Run forward inside Studio/Notebook — ONLY after metadata_only PASS
```bash
python src/fourcastnet/notebook_forward_smoke.py --model-package-arn arn:aws:sagemaker:us-east-1:725644097028:model-package/sbnai-fourcastnet-fcn-v0/1 --input-tensor-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/input/input_tensor.npy --mode forward --region us-east-1 --output-report-s3-uri s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/fourcastnet/fcn-v0/notebook-smoke/studio_forward_report.json
```

### 3.3 Alternative: Run notebook interactively
Open `notebooks/fourcastnet_registry_gpu_smoke.ipynb` in Studio JupyterLab.
Execute cells in order. The notebook calls the same script above.

---

## 4. Cost safety

### 4.1 List Studio apps InService
```powershell
aws sagemaker list-apps --profile sbnai-725 --region us-east-1 --query "Apps[?Status=='InService'].[AppName,AppType,DomainId,UserProfileName,ResourceSpec.InstanceType]" --output table
```

### 4.2 List Notebook instances InService
```powershell
aws sagemaker list-notebook-instances --status-equals InService --profile sbnai-725 --region us-east-1
```

### 4.3 Stop a notebook instance (template — fill in INSTANCE_NAME)
> ⚠️ Mutating command. Run only when you want to stop billing for that instance.
```powershell
aws sagemaker stop-notebook-instance --notebook-instance-name INSTANCE_NAME --profile sbnai-725 --region us-east-1
```

### 4.4 Delete a Studio JupyterLab app (template — fill in DOMAIN_ID, USER_PROFILE, APP_NAME, APP_TYPE)
> ⚠️ Mutating command. Run only when you want to shut down the GPU app.
```powershell
aws sagemaker delete-app --domain-id DOMAIN_ID --user-profile-name USER_PROFILE --app-type APP_TYPE --app-name APP_NAME --profile sbnai-725 --region us-east-1
```

---

## 5. Interpretation

| Result | Meaning |
|--------|---------|
| `metadata_only` PASS (`ok=true`, `result=metadata_collected`) | Artifact access and environment plumbing are confirmed. Model Registry is readable. S3 download works. `fourcastnet_proven` remains `false`. |
| `forward` PASS (`ok=true`, `result=forward_succeeded`, `fourcastnet_proven=true`) | Runtime inference is confirmed. This is the only condition under which `fourcastnet_proven=true`. |
| `forward` FAIL with `reason=backend_not_found` | **Not an AWS failure.** The FourCastNet Python package (`modulus`, `physicsnemo`, or equivalent) is missing from the Studio/Notebook environment. Fix: install the package, then re-run `--mode forward`. |
| `forward` FAIL with `reason=model_instantiation_failed` | Checkpoint or model class instantiation error. Check `load_attempts` in the report. |
| `forward` FAIL with `reason=forward_failed` | Model loaded but inference raised an exception. Check `error_message` and `traceback` in the report. |

> **FourCastNet is never considered proven unless `fourcastnet_proven=true` in a `forward` mode report.**
