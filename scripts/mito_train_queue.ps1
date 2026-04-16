# Sequential GPU jobs: Hela2 per-crop rerun -> Hela3 joint -> Hela3 infer-all -> Hela3 per-crop.
# Log: result/mito_train_queue.log
$ErrorActionPreference = "Continue"
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
$repo = "D:\1AAA\mit_rec\cursor\Mito3D_Reconstruction_Thesis"
$log = Join-Path $repo "result\mito_train_queue.log"
Set-Location $repo
New-Item -ItemType Directory -Path (Join-Path $repo "result") -Force | Out-Null
function Log([string]$m) {
    $line = "$(Get-Date -Format o) $m"
    Add-Content -Path $log -Value $line
    Write-Host $line
}
function RunStep([string]$name, [string[]]$argList) {
    Log "STEP: $name"
    & $py @argList 2>&1 | Tee-Object -FilePath $log -Append
    if ($LASTEXITCODE -ne 0) {
        Log "FAILED (exit $LASTEXITCODE): $name"
        exit $LASTEXITCODE
    }
}
Log "=== START queue ==="
$pipe = (Join-Path $repo "hela2_mito_pipeline.py")
RunStep "hela2 per-crop" @($pipe, "--python", $py, "per-crop", "--data_root", "data", "--dataset", "jrc_hela-2", "--epochs", "300")
RunStep "hela3 joint" @($pipe, "--python", $py, "joint", "--data_root", "data", "--dataset", "jrc_hela-3", "--epochs", "300")
RunStep "hela3 infer-all" @($pipe, "--python", $py, "infer-all", "--data_root", "data", "--dataset", "jrc_hela-3")
RunStep "hela3 per-crop" @($pipe, "--python", $py, "per-crop", "--data_root", "data", "--dataset", "jrc_hela-3", "--epochs", "300")
Log "=== QUEUE DONE ==="
