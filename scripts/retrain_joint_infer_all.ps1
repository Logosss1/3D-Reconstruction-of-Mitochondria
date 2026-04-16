# Joint train Hela2 + Hela3, then infer-all for both (np_s0 crops from manifest).
# Run: powershell -ExecutionPolicy Bypass -File scripts\retrain_joint_infer_all.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $repo "hela2_mito_pipeline.py"))) {
    Write-Error "hela2_mito_pipeline.py not found under $repo"
}
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Python not found: $py — edit scripts/retrain_joint_infer_all.ps1"
}
$log = Join-Path $repo "result\retrain_full.log"
New-Item -ItemType Directory -Force -Path (Join-Path $repo "result") | Out-Null
$env:PYTHONUNBUFFERED = "1"
Set-Location $repo

function Log([string]$m) {
    $line = "$(Get-Date -Format o) $m"
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

function RunStep([string]$title, [string[]]$extraArgs) {
    Log "======== $title ========"
    $scriptPath = Join-Path $repo "hela2_mito_pipeline.py"
    $all = @($scriptPath, "--python", $py) + $extraArgs
    & $py -u @all 2>&1 | ForEach-Object {
        $_ | Out-File -FilePath $log -Append -Encoding utf8
        Write-Host $_
    }
    if ($LASTEXITCODE -ne 0) {
        Log "FAILED: $title (exit $LASTEXITCODE)"
        exit $LASTEXITCODE
    }
}

Log "=== QUEUE START ==="
RunStep "joint hela2" @("joint", "--dataset", "jrc_hela-2", "--epochs", "300", "--fg_query_fraction", "0.38")
RunStep "joint hela3" @("joint", "--dataset", "jrc_hela-3", "--epochs", "300", "--fg_query_fraction", "0.38")
RunStep "infer-all hela2" @("infer-all", "--dataset", "jrc_hela-2")
RunStep "infer-all hela3" @("infer-all", "--dataset", "jrc_hela-3")
Log "=== QUEUE DONE ==="
