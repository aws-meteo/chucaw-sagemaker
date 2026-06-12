<#
.SYNOPSIS
  Safe, gated CPU Batch Transform POC for FourCastNet (no real-time endpoints).

.DESCRIPTION
  Implements the brief's sections 14 + 17 as a single guarded flow:

    1. STS identity gate            (stop on wrong account / failed call)
    2. Endpoint pre-flight gate     (stop if any FourCastNet/FCN/Chucaw endpoint exists)
    3. Input .npy existence gate
    4. CPU model: describe; if missing, download model.tar.gz, inspect
       code/inference.py for hardcoded CUDA, then create CPU model
    5. CPU transform quota gate      (ml.m5.large >= 1)
    6. Validate MWAA workflow by ARN
    7. (with -Execute) start workflow run and monitor
    8. Validate S3 output + endpoint POST-flight gate

  DRY RUN BY DEFAULT. Nothing that creates AWS resources or starts a run is
  performed unless you pass -Execute. Creating the CPU model is additionally
  gated behind -CreateModel, and re-uploading a patched artifact behind
  -AllowPatchUpload.

.NOTES
  - Never creates endpoints, never calls deploy. CPU only.
  - Requires: aws cli, tar, and python (for the inference inspector).
  - The FourCastNet torch handler is NOT in this repo; it lives inside the S3
    model.tar.gz. This script downloads and inspects it at runtime.

.EXAMPLE
  # Read-only: validate everything, inspect inference.py, do not change AWS.
  ./scripts/run_fourcastnet_cpu_poc.ps1

.EXAMPLE
  # Create the CPU model (only if inference.py is CPU-safe) then start the run.
  ./scripts/run_fourcastnet_cpu_poc.ps1 -CreateModel -Execute
#>

[CmdletBinding()]
param(
  [string]$Profile      = "sbnai-725",
  [string]$Region       = "us-east-1",
  [string]$AccountId    = "725644097028",

  [string]$Bucket       = "chucaw-data-platinum-processed-725644097028-us-east-1-an",
  [string]$InputS3Uri   = "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/ecmwf/fourcastnet/year=2026/month=06/day=06/hour=18z/20260606180000-0h-oper-fc_tensor.npy",
  [string]$OutputS3Uri  = "s3://chucaw-data-platinum-processed-725644097028-us-east-1-an/sagemaker/batch-transform/fourcastnet-poc/2026-06-06-18z/",

  [string]$OriginalModelName = "sbnai-fourcastnet-fcn-v1",
  [string]$CpuModelName      = "sbnai-fourcastnet-fcn-v1-cpu-poc",
  [string]$CpuImage          = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-cpu-py310",
  [string]$CpuInstanceType   = "ml.m5.large",

  [string]$WorkflowArn  = "arn:aws:airflow-serverless:us-east-1:725644097028:workflow/fourcastnet-cpu-batch-transform-poc-5k2ShVoqUy",

  # Behaviour switches (all off => pure read-only dry run)
  [switch]$CreateModel,        # allow create-model when CPU model is missing
  [switch]$AllowPatchUpload,   # allow uploading a CPU-patched model.tar.gz if CUDA is hardcoded
  [switch]$Execute             # allow start-workflow-run
)

$ErrorActionPreference = "Stop"

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$ArtifactDir = Join-Path $RepoRoot "artifacts"
New-Item -ItemType Directory -Force $ArtifactDir | Out-Null
$Inspector  = Join-Path $RepoRoot "scripts/inspect_and_patch_fourcastnet_inference.py"

function Write-Step { param([string]$Msg) Write-Host "`n=== $Msg ===" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host "OK: $Msg"   -ForegroundColor Green }

function Invoke-Aws {
  # Thin wrapper that fails hard on non-zero exit (avoids using stale variables).
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AwsArgs)
  $out = & aws @AwsArgs --profile $Profile --region $Region 2>&1
  if ($LASTEXITCODE -ne 0) {
    throw "STOP: aws $($AwsArgs -join ' ') failed:`n$out"
  }
  return $out
}

# --- 1. STS identity gate ----------------------------------------------------
Write-Step "1. STS identity gate"
$Identity = (Invoke-Aws sts get-caller-identity --output json) | ConvertFrom-Json
if ($Identity.Account -ne $AccountId) {
  throw "STOP: Wrong account. Expected=$AccountId Actual=$($Identity.Account)"
}
Write-Ok "Account=$($Identity.Account), Arn=$($Identity.Arn)"

# --- 2. Endpoint pre-flight gate --------------------------------------------
Write-Step "2. Endpoint pre-flight gate (must be NONE for FourCastNet/FCN/Chucaw)"
$EndpointsJson = Invoke-Aws sagemaker list-endpoints `
  --query "Endpoints[?EndpointStatus=='InService' || EndpointStatus=='Creating' || EndpointStatus=='Updating'].{Name:EndpointName,Status:EndpointStatus}" `
  --output json
$BadEndpoints = ($EndpointsJson | ConvertFrom-Json) | Where-Object { $_.Name -match "(?i)fourcast|fcn|chucaw" }
if ($BadEndpoints) {
  $BadEndpoints | Format-Table
  throw "STOP: Active/creating FourCastNet/Chucaw endpoint exists. This POC forbids endpoints."
}
Write-Ok "No active FourCastNet/FCN/Chucaw endpoints."

# --- 3. Input existence gate -------------------------------------------------
Write-Step "3. Input .npy existence gate"
$InputLs = Invoke-Aws s3 ls $InputS3Uri
Write-Host $InputLs
Write-Ok "Input present: $InputS3Uri"

# --- 4. CPU model: describe or create ---------------------------------------
Write-Step "4. CPU model discovery: $CpuModelName"
$CpuModelExists = $false
$cpuDescRaw = & aws sagemaker describe-model --model-name $CpuModelName --profile $Profile --region $Region --output json 2>&1
if ($LASTEXITCODE -eq 0) {
  $CpuModelExists = $true
  $cpuDesc = $cpuDescRaw | ConvertFrom-Json
  $img = $cpuDesc.PrimaryContainer.Image
  Write-Ok "CPU model exists. Image=$img"
  if ($img -match "(?i)-gpu-") {
    throw "STOP: '$CpuModelName' uses a GPU image ($img). Refusing to run on GPU-imaged model."
  }
} else {
  Write-Host "CPU model not found. Will inspect original artifact to decide if it can be created."

  Write-Step "4a. Describe original model: $OriginalModelName"
  $origDesc = (Invoke-Aws sagemaker describe-model --model-name $OriginalModelName --output json) | ConvertFrom-Json
  $ModelDataUrl = $origDesc.PrimaryContainer.ModelDataUrl
  $RoleArn      = $origDesc.ExecutionRoleArn
  if (-not $ModelDataUrl) { throw "STOP: Could not read ModelDataUrl from $OriginalModelName" }
  if (-not $RoleArn)      { throw "STOP: Could not read ExecutionRoleArn from $OriginalModelName" }
  Write-Ok "ModelDataUrl=$ModelDataUrl"
  Write-Ok "ExecutionRoleArn=$RoleArn"

  Write-Step "4b. Download + inspect code/inference.py for hardcoded CUDA"
  $LocalTar = Join-Path $ArtifactDir "fcn-v1-model.tar.gz"
  Remove-Item $LocalTar -Force -ErrorAction SilentlyContinue
  Invoke-Aws s3 cp $ModelDataUrl $LocalTar | Out-Null

  & python $Inspector --model-tar $LocalTar
  $InspectExit = $LASTEXITCODE   # 0 = CPU-safe, 2 = hardcoded CUDA, 3 = no inference.py

  $UploadModelDataUrl = $ModelDataUrl
  if ($InspectExit -eq 0) {
    Write-Ok "inference.py is CPU-safe. CPU model can reuse the original model.tar.gz."
  }
  elseif ($InspectExit -eq 2) {
    Write-Host "inference.py hardcodes CUDA." -ForegroundColor Yellow
    $PatchedTar = Join-Path $ArtifactDir "fcn-v1-cpu-model.tar.gz"
    & python $Inspector --model-tar $LocalTar --emit-patched $PatchedTar
    if ($LASTEXITCODE -ne 0) { throw "STOP: failed to produce a CPU-safe patched artifact." }
    Write-Host "A CPU-patched artifact was written: $PatchedTar" -ForegroundColor Yellow
    Write-Host "REVIEW the change log above before uploading." -ForegroundColor Yellow

    if (-not $AllowPatchUpload) {
      throw "STOP: inference.py needed CPU patching. Re-run with -AllowPatchUpload after reviewing $PatchedTar."
    }
    # Upload patched artifact to a DISTINCT key so the original is never overwritten.
    $PatchedKey = "sagemaker/fourcastnet/fcn-v1-cpu-poc/model/model.tar.gz"
    $UploadModelDataUrl = "s3://$Bucket/$PatchedKey"
    if ($Execute -or $CreateModel) {
      Invoke-Aws s3 cp $PatchedTar $UploadModelDataUrl | Out-Null
      Write-Ok "Uploaded patched artifact to $UploadModelDataUrl"
    } else {
      Write-Host "[DRY RUN] Would upload patched artifact to $UploadModelDataUrl" -ForegroundColor Yellow
    }
  }
  else {
    throw "STOP: inspector exit=$InspectExit (no inference.py / read error). Cannot proceed."
  }

  Write-Step "4c. Create CPU model (gated by -CreateModel)"
  $CpuModelSpec = @{
    ModelName        = $CpuModelName
    ExecutionRoleArn = $RoleArn
    PrimaryContainer = @{
      Image        = $CpuImage
      Mode         = "SingleModel"
      ModelDataUrl = $UploadModelDataUrl
      Environment  = @{
        SAGEMAKER_PROGRAM           = "inference.py"
        SAGEMAKER_SUBMIT_DIRECTORY  = "/opt/ml/model/code"
      }
    }
    Tags = @(
      @{ Key = "Project";     Value = "SbnAI-Chucaw" },
      @{ Key = "Purpose";     Value = "fourcastnet-cpu-batch-transform-poc" },
      @{ Key = "Environment"; Value = "dev" },
      @{ Key = "NoEndpoint";  Value = "true" }
    )
  }
  $SpecPath = Join-Path $ArtifactDir "create-model-$CpuModelName.json"
  $CpuModelSpec | ConvertTo-Json -Depth 20 | Set-Content -Encoding utf8 $SpecPath
  Write-Host "Wrote create-model spec: $SpecPath"

  if ($CreateModel) {
    Invoke-Aws sagemaker create-model --cli-input-json "file://$SpecPath" | Out-Null
    $CpuModelExists = $true
    Write-Ok "Created CPU model: $CpuModelName"
  } else {
    Write-Host "[DRY RUN] CPU model not created. Re-run with -CreateModel to create it." -ForegroundColor Yellow
  }
}

# --- 5. CPU transform quota gate --------------------------------------------
Write-Step "5. CPU transform quota gate ($CpuInstanceType for transform job usage >= 1)"
$QuotasJson = Invoke-Aws service-quotas list-service-quotas --service-code sagemaker --output json
$Quota = ($QuotasJson | ConvertFrom-Json).Quotas | Where-Object {
  $_.QuotaName -match [regex]::Escape("$CpuInstanceType for transform job usage")
} | Select-Object -First 1
if (-not $Quota) {
  Write-Host "WARN: could not find quota entry for '$CpuInstanceType for transform job usage'." -ForegroundColor Yellow
} elseif ([double]$Quota.Value -lt 1) {
  throw "STOP: quota '$($Quota.QuotaName)' = $($Quota.Value) (<1). Request an increase or pick another CPU type."
} else {
  Write-Ok "Quota '$($Quota.QuotaName)' = $($Quota.Value)"
}

# --- 6. Validate workflow by ARN --------------------------------------------
Write-Step "6. Validate MWAA Serverless workflow by ARN"
$wf = (Invoke-Aws mwaa-serverless get-workflow --workflow-arn $WorkflowArn --output json) | ConvertFrom-Json
Write-Ok "Workflow OK: $($wf.Name) (status=$($wf.Status))"

# --- 7. Start workflow run (gated by -Execute) ------------------------------
Write-Step "7. Start workflow run (gated by -Execute)"
if (-not $CpuModelExists) {
  Write-Host "[BLOCKED] CPU model does not exist yet; will not start run. Re-run with -CreateModel." -ForegroundColor Yellow
}
elseif (-not $Execute) {
  Write-Host "[DRY RUN] Pre-conditions satisfied. Re-run with -Execute to start the workflow run." -ForegroundColor Yellow
}
else {
  $run = (Invoke-Aws mwaa-serverless start-workflow-run --workflow-arn $WorkflowArn --output json) | ConvertFrom-Json
  Write-Ok "Started workflow run: $($run | ConvertTo-Json -Compress)"
  Write-Host "Monitor with:" -ForegroundColor Cyan
  Write-Host "  aws mwaa-serverless list-workflow-runs --workflow-arn $WorkflowArn --profile $Profile --region $Region --output table"
  Write-Host "  aws sagemaker list-transform-jobs --name-contains fourcastnet-cpu-poc --profile $Profile --region $Region --output table"
}

# --- 8. Output + endpoint post-flight gate ----------------------------------
Write-Step "8. Output listing + endpoint POST-flight gate"
$OutLs = & aws s3 ls $OutputS3Uri --recursive --profile $Profile --region $Region 2>&1
Write-Host "Output prefix contents ($OutputS3Uri):"
Write-Host $OutLs

$PostJson = Invoke-Aws sagemaker list-endpoints `
  --query "Endpoints[?EndpointStatus=='InService' || EndpointStatus=='Creating' || EndpointStatus=='Updating'].{Name:EndpointName,Status:EndpointStatus}" `
  --output json
$PostBad = ($PostJson | ConvertFrom-Json) | Where-Object { $_.Name -match "(?i)fourcast|fcn|chucaw" }
if ($PostBad) {
  $PostBad | Format-Table
  throw "STOP: An endpoint appeared during this run. Investigate immediately."
}
Write-Ok "POST-flight: still no FourCastNet/FCN/Chucaw endpoints."

Write-Host "`n=== DONE ===" -ForegroundColor Cyan
Write-Host "CpuModelExists : $CpuModelExists"
Write-Host "Mode           : $([string]::Join(' ', @(if($CreateModel){'-CreateModel'}; if($AllowPatchUpload){'-AllowPatchUpload'}; if($Execute){'-Execute'}) ))"
