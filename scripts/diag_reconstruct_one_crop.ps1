# Diagnostic pipeline for steps 1–4 (one crop, short epochs):
#   (1) Log occupancy + marching-cubes from generate.py
#   (2) Compare fixed vs percentile (+ optional smooth) MC
#   (3) Same fg_query_fraction / val as your mixed script
#   (4) Compare joint(single-crop) vs per-crop checkpoints on validate_crop
#
# Requires: E:\Anaconda\envs\mito3d_env\python.exe (or edit $py below)
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
if (-not (Test-Path $py)) { $py = "D:\anacondaEnvs\envs\mito3d_env\python.exe" }
if (-not (Test-Path $py)) { Write-Error "Set `$py to your torch Python." }
$crop = 9
$epochs = 5
$export = Join-Path $repo "data\crop_exports_hela2_mito_bg"
$out = Join-Path $repo "result\diag_reconstruct_one_crop"
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location $repo

function Run([string]$title, [string[]]$argv) {
    Write-Host "=== $title ==="
    & $py -u @argv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

# 1) Joint train (same crop only — same as multi-crop schedule but one volume)
Run "joint train" @(
    "hela2_mito_pipeline.py", "--python", $py, "joint",
    "--dataset", "jrc_hela-2", "--crop_ids", "$crop",
    "--epochs", "$epochs",
    "--fg_query_fraction", "0.38",
    "--out", "checkpoints/diag_joint_crop${crop}.pth",
    "--train_log_csv", "result/diag_joint_train_log.csv",
    "--train_log_every", "1",
    "--val_every", "1",
    "--val_log_csv", "result/diag_joint_val_metrics.csv"
)

# 2) Per-crop train + infer (default MC)
Run "per-crop" @(
    "hela2_mito_pipeline.py", "--python", $py, "per-crop",
    "--only-crop", "$crop",
    "--epochs", "$epochs",
    "--fg_query_fraction", "0.38",
    "--checkpoint_dir", "checkpoints/diag_per_crop",
    "--result_dir", "result/diag_per_crop_infer"
)

# 3) Joint infer: fixed vs percentile
$jointCk = "checkpoints/diag_joint_crop${crop}.pth"
$prefix = "joint_"
Run "generate fixed MC" @(
    "generate.py",
    "--data_root", "data", "--dataset", "jrc_hela-2", "--crop_id", "$crop",
    "--checkpoint", $jointCk,
    "--crop_export_dir", $export,
    "--output_dir", (Join-Path $out "joint_fixed"),
    "--name_prefix", $prefix,
    "--mc_level_mode", "fixed", "--mc_fixed_level", "0.5",
    "--mesh_strip_mc_shell_cells", "3.0"
)
Run "generate percentile + smooth" @(
    "generate.py",
    "--data_root", "data", "--dataset", "jrc_hela-2", "--crop_id", "$crop",
    "--checkpoint", $jointCk,
    "--crop_export_dir", $export,
    "--output_dir", (Join-Path $out "joint_pct90_smooth"),
    "--name_prefix", $prefix,
    "--mc_level_mode", "percentile", "--mc_percentile", "90",
    "--mc_prob_smooth_sigma", "0.75",
    "--mesh_strip_mc_shell_cells", "3.0",
    "--save_binary_preview"
)

# 4) Validate metrics
$jFixed = Join-Path $out "joint_fixed\${prefix}final_mitochondria.obj"
$jPct = Join-Path $out "joint_pct90_smooth\${prefix}final_mitochondria.obj"
$pc = Join-Path $repo "result\diag_per_crop_infer\crop${crop}_final_mitochondria.obj"
$rep = Join-Path $out "metrics_compare.txt"
"" | Out-File -FilePath $rep -Encoding utf8
foreach ($pair in @(
    @("joint_fixed", $jFixed),
    @("joint_pct90_smooth075", $jPct),
    @("per_crop", $pc)
)) {
    $tag = $pair[0]
    $mesh = $pair[1]
    if (-not (Test-Path $mesh)) { continue }
    & $py "validate_crop.py" "--data_root", "data", "--dataset", "jrc_hela-2", "--crop_id", "$crop",
        "--mesh", $mesh, "--out_json", (Join-Path $out "metrics_${tag}.json"),
        "--crop_export_dir", $export | Tee-Object -FilePath $rep -Append
}
Write-Host "Wrote $rep"
Write-Host "Done."
