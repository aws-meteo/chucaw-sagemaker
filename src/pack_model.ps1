$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$modelJoblib = Join-Path $repoRoot "model.joblib"
$inferencePy = Join-Path $repoRoot "inference\inference.py"
$inferenceReq = Join-Path $repoRoot "inference\requirements.txt"
$buildDir = Join-Path $repoRoot ".build_model"
$codeDir = Join-Path $buildDir "code"
$tarPath = Join-Path $repoRoot "model.tar.gz"

function Require-Path {
    param([string]$PathValue, [string]$Label)
    if (-not (Test-Path $PathValue)) {
        throw "Missing required $Label: $PathValue"
    }
}

Require-Path $modelJoblib "model artifact"
Require-Path $inferencePy "inference entrypoint"
Require-Path $inferenceReq "inference requirements"

if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}
New-Item -ItemType Directory -Path $codeDir | Out-Null

Copy-Item $modelJoblib (Join-Path $buildDir "model.joblib")
Copy-Item $inferencePy (Join-Path $codeDir "inference.py")
Copy-Item $inferenceReq (Join-Path $codeDir "requirements.txt")

if (Test-Path $tarPath) {
    Remove-Item -Force $tarPath
}

tar -czf $tarPath -C $buildDir .

Write-Host "Packaged tarball: $tarPath"

$entries = tar -tzf $tarPath
$required = @(
    "./model.joblib",
    "./code/inference.py",
    "./code/requirements.txt"
)

foreach ($item in $required) {
    if ($entries -notcontains $item) {
        Write-Host "Tarball contents:"
        $entries | ForEach-Object { Write-Host $_ }
        throw "Malformed model.tar.gz: missing $item"
    }
}

Write-Host "Validated structure:"
$entries | ForEach-Object { Write-Host $_ }

if (Test-Path $buildDir) {
    Remove-Item -Recurse -Force $buildDir
}