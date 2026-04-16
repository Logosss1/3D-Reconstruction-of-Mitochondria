# Train model_hela2_hela3_mixed.pth then infer-all on both datasets (meshes + montage).
# Use the interpreter that has PyTorch. If you have two conda envs named mito3d_env
# (e.g. D:\anacondaEnvs\... vs E:\Anaconda\envs\...), only one may include torch â€?see scripts/diag_mito3d_env.ps1.
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Edit scripts/mixed_train_infer.ps1: set `$py to your torch Python."
}
# Preflight: must use GPU (CPU training is impractically slow)
$cudaOk = & $py -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "CUDA not available in $py â€?training will be very slow. Fix PyTorch/CUDA driver, then retry."
}
Write-Host "OK: CUDA available (training will use GPU)."
$log = Join-Path $repo "result\mixed_train_infer.log"
New-Item -ItemType Directory -Force -Path (Join-Path $repo "result") | Out-Null
$env:PYTHONUNBUFFERED = "1"
Set-Location $repo
$pipe = Join-Path $repo "hela2_mito_pipeline.py"

function Log([string]$m) {
    $line = "$(Get-Date -Format o) $m"
    Add-Content -Path $log -Value $line -Encoding utf8
    Write-Host $line
}

function Run([string]$title, [string[]]$argv) {
    Log "=== $title ==="
    $all = @($pipe, "--python", $py) + $argv
    & $py -u @all 2>&1 | ForEach-Object {
        $_ | Out-File -FilePath $log -Append -Encoding utf8
        Write-Host $_
    }
    if ($LASTEXITCODE -ne 0) {
        Log "FAILED $title exit $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Log "=== START mixed train + infer ==="
# Higher num_points + steps_per_epoch => heavier GPU work per logged epoch (see train.py).
# Total optimizer steps = epochs * steps_per_epoch (e.g. 300 * 6 = 1800).
Run "joint mixed" @(
    "joint",
    "--datasets", "jrc_hela-2,jrc_hela-3",
    "--epochs", "300",
    "--num_points", "28000",
    "--steps_per_epoch", "6",
    "--fg_query_fraction", "0.38",
    "--native_required",
    "--weight_decay", "1e-5",
    "--train_log_csv", "result/mixed_joint_train_log.csv",
    "--train_log_every", "20",
    "--val_every", "10",
    "--val_log_csv", "result/mixed_joint_val_metrics.csv",
    "--val_num_points", "4096",
    "--aug_contrast",
    "--aug_contrast_scale", "0.12",
    "--aug_noise_std", "4",
    "--aug_prob", "0.85",
    "--aug_geometric",
    "--aug_gamma", "0.12",
    "--aug_percentile_stretch"
)
Run "infer-all hela2 (mixed ckpt)" @(
    "infer-all",
    "--dataset", "jrc_hela-2",
    "--checkpoint", "checkpoints/model_hela2_hela3_mixed.pth",
    "--output_dir", "result/hela2_mito_mixed_infer",
    "--native_required",
    "--native_tile", "160",
    "--native_overlap", "24",
    "--query_chunk", "200000",
    "--mc_level_mode", "percentile",
    "--mc_percentile", "90",
    "--mc_prob_smooth_sigma", "0.75",
    "--validate",
    "--validate_train_mode", "mixed_joint_infer"
)
Run "infer-all hela3 (mixed ckpt)" @(
    "infer-all",
    "--dataset", "jrc_hela-3",
    "--checkpoint", "checkpoints/model_hela2_hela3_mixed.pth",
    "--output_dir", "result/hela3_mito_mixed_infer",
    "--native_required",
    "--native_tile", "160",
    "--native_overlap", "24",
    "--query_chunk", "200000",
    "--mc_level_mode", "percentile",
    "--mc_percentile", "90",
    "--mc_prob_smooth_sigma", "0.75",
    "--validate",
    "--validate_train_mode", "mixed_joint_infer"
)
Log "=== DONE ==="
