# Sequential full retrains: Hela2 joint -> Hela3 joint -> Hela2+Hela3 mixed (then infer for each).
# Strict native-size reconstruction: infer-all runs generate.py with --native_required (tiled, no downsample).
# Note: native reconstruction is much slower; adjust $nativeTile / $queryChunk if you hit OOM.
# Matches advisor-style aug: contrast, noise, rot90+flip, gamma, percentile stretch; Adam weight_decay.
# GPU: high num_points + steps_per_epoch (override below if VRAM is tight).
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
if (-not (Test-Path $py)) {
    Write-Error "Set `$py to E:\Anaconda\envs\mito3d_env\python.exe (torch build)."
}
$cudaOk = & $py -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "CUDA not available in $py"
}
$log = Join-Path $repo "result\rerun_all_training.log"
New-Item -ItemType Directory -Force -Path (Join-Path $repo "result") | Out-Null
$env:PYTHONUNBUFFERED = "1"
Set-Location $repo
$pipe = Join-Path $repo "hela2_mito_pipeline.py"

# Shared training knobs (raise GPU load vs single-step 10k points)
$epochs = "300"
$numPoints = "28000"
$stepsPerEpoch = "6"
$fg = "0.38"
$wd = "1e-5"
$nativeTile = "160"
$nativeOverlap = "24"
$queryChunk = "200000"

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

$aug = @(
    "--aug_contrast",
    "--aug_contrast_scale", "0.12",
    "--aug_noise_std", "4",
    "--aug_prob", "0.85",
    "--aug_geometric",
    "--aug_gamma", "0.12",
    "--aug_percentile_stretch",
    "--weight_decay", $wd
)
$val = @(
    "--train_log_every", "20",
    "--val_every", "10",
    "--val_num_points", "4096"
)

Log "=== START rerun_all_training (native_required reconstruction) ==="

Run "joint hela2 only" (@(
    "joint",
    "--dataset", "jrc_hela-2",
    "--epochs", $epochs,
    "--encoder_spatial", "128",
    "--num_points", $numPoints,
    "--steps_per_epoch", $stepsPerEpoch,
    "--fg_query_fraction", $fg,
    "--native_required",
    "--out", "checkpoints/model_hela2_all_mito.pth",
    "--train_log_csv", "result/hela2_joint_train_log.csv",
    "--val_log_csv", "result/hela2_joint_val_metrics.csv"
) + $val + $aug)

Run "infer-all hela2 (hela2_all ckpt)" @(
    "infer-all",
    "--dataset", "jrc_hela-2",
    "--checkpoint", "checkpoints/model_hela2_all_mito.pth",
    "--output_dir", "result/hela2_mito_joint_infer",
    "--native_required",
    "--native_tile", $nativeTile,
    "--native_overlap", $nativeOverlap,
    "--query_chunk", $queryChunk,
    "--mc_level_mode", "percentile",
    "--mc_percentile", "90",
    "--mc_prob_smooth_sigma", "0.75",
    "--validate",
    "--validate_train_mode", "joint_infer_hela2"
)

Run "joint hela3 only" (@(
    "joint",
    "--dataset", "jrc_hela-3",
    "--epochs", $epochs,
    "--encoder_spatial", "128",
    "--num_points", $numPoints,
    "--steps_per_epoch", $stepsPerEpoch,
    "--fg_query_fraction", $fg,
    "--native_required",
    "--out", "checkpoints/model_hela3_all_mito.pth",
    "--train_log_csv", "result/hela3_joint_train_log.csv",
    "--val_log_csv", "result/hela3_joint_val_metrics.csv"
) + $val + $aug)

Run "infer-all hela3 (hela3_all ckpt)" @(
    "infer-all",
    "--dataset", "jrc_hela-3",
    "--checkpoint", "checkpoints/model_hela3_all_mito.pth",
    "--output_dir", "result/hela3_mito_joint_infer",
    "--native_required",
    "--native_tile", $nativeTile,
    "--native_overlap", $nativeOverlap,
    "--query_chunk", $queryChunk,
    "--mc_level_mode", "percentile",
    "--mc_percentile", "90",
    "--mc_prob_smooth_sigma", "0.75",
    "--validate",
    "--validate_train_mode", "joint_infer_hela3"
)

Run "joint mixed hela2+hela3" (@(
    "joint",
    "--datasets", "jrc_hela-2,jrc_hela-3",
    "--epochs", $epochs,
    "--encoder_spatial", "128",
    "--num_points", $numPoints,
    "--steps_per_epoch", $stepsPerEpoch,
    "--fg_query_fraction", $fg,
    "--native_required",
    "--out", "checkpoints/model_hela2_hela3_mixed.pth",
    "--train_log_csv", "result/mixed_joint_train_log.csv",
    "--val_log_csv", "result/mixed_joint_val_metrics.csv"
) + $val + $aug)

Run "infer-all hela2 (mixed ckpt)" @(
    "infer-all",
    "--dataset", "jrc_hela-2",
    "--checkpoint", "checkpoints/model_hela2_hela3_mixed.pth",
    "--output_dir", "result/hela2_mito_mixed_infer",
    "--native_required",
    "--native_tile", $nativeTile,
    "--native_overlap", $nativeOverlap,
    "--query_chunk", $queryChunk,
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
    "--native_tile", $nativeTile,
    "--native_overlap", $nativeOverlap,
    "--query_chunk", $queryChunk,
    "--mc_level_mode", "percentile",
    "--mc_percentile", "90",
    "--mc_prob_smooth_sigma", "0.75",
    "--validate",
    "--validate_train_mode", "mixed_joint_infer"
)

Log "=== DONE rerun_all_training ==="
