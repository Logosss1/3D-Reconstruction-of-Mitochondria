# UTF-8 log. Completes: (1) Hela2 per-crop 23,3,6,9  (2) Hela3 joint  (3) Hela3 infer-all  (4) Hela3 per-crop
$ErrorActionPreference = "Continue"
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
$repo = "D:\1AAA\mit_rec\cursor\Mito3D_Reconstruction_Thesis"
$log = Join-Path $repo "result\mito_remaining_queue.log"
Set-Location $repo
New-Item -ItemType Directory -Path (Join-Path $repo "result") -Force | Out-Null
$utf8 = New-Object System.Text.UTF8Encoding $false
function Log([string]$m) {
    $line = "$(Get-Date -Format o) $m"
    [System.IO.File]::AppendAllText($log, "$line`n", $utf8)
    Write-Host $line
}
function RunStep([string]$name, [string[]]$argList) {
    Log "STEP: $name"
    & $py @argList 2>&1 | ForEach-Object { $_; [void][System.IO.File]::AppendAllText($log, "$_`n", $utf8) }
    if ($LASTEXITCODE -ne 0) {
        Log "FAILED (exit $LASTEXITCODE): $name"
        exit $LASTEXITCODE
    }
}
Log "=== START remaining queue ==="
$pipe = (Join-Path $repo "hela2_mito_pipeline.py")
RunStep "hela2 per-crop crops 23,3,6,9" @(
    $pipe, "--python", $py, "per-crop",
    "--data_root", "data", "--dataset", "jrc_hela-2",
    "--crop_ids", "23,3,6,9", "--epochs", "300"
)
RunStep "hela3 joint" @(
    $pipe, "--python", $py, "joint",
    "--data_root", "data", "--dataset", "jrc_hela-3", "--epochs", "300"
)
RunStep "hela3 infer-all" @(
    $pipe, "--python", $py, "infer-all",
    "--data_root", "data", "--dataset", "jrc_hela-3"
)
RunStep "hela3 per-crop" @(
    $pipe, "--python", $py, "per-crop",
    "--data_root", "data", "--dataset", "jrc_hela-3", "--epochs", "300"
)
Log "=== QUEUE DONE ==="
