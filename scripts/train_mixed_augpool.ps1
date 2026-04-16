# Train mixed model using original + 6 offline-augmented export folders.
# This script assumes you've already generated the augmented folders via:
#   python scripts/export_augmented_crop_exports.py --dataset jrc_hela-2 --in_dir data/crop_exports_hela2_mito_bg --mode all
#   python scripts/export_augmented_crop_exports.py --dataset jrc_hela-3 --in_dir data/crop_exports_hela3_mito_bg --mode all
#
# Result: one big pool of volumes (original + aug_*), sampled uniformly across steps.

$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
if (-not (Test-Path $py)) { Write-Error "Set `$py to your torch python." }
Set-Location $repo

$hela2 = "data/crop_exports_hela2_mito_bg"
$hela3 = "data/crop_exports_hela3_mito_bg"
$modes = @("contrast","stretch","geo","gamma","noise","combo")

$script:dirs = @()
$script:datasets = @()

function AddDirs([string]$ds, [string]$base) {
  $script:datasets += $ds
  $script:dirs += $base
  foreach ($m in $modes) {
    $script:datasets += $ds
    $script:dirs += "${base}_aug_${m}"
  }
}

AddDirs "jrc_hela-2" $hela2
AddDirs "jrc_hela-3" $hela3

$datasetsCsv = ($script:datasets -join ",")
$dirsCsv = ($script:dirs -join ",")
Write-Host "datasets entries: $($script:datasets.Count) dirs entries: $($script:dirs.Count)"
Write-Host "datasetsCsv length: $($datasetsCsv.Length) dirsCsv length: $($dirsCsv.Length)"
Write-Host "datasetsCsv head: $($datasetsCsv.Substring(0,[Math]::Min(120,$datasetsCsv.Length)))"
Write-Host "dirsCsv head: $($dirsCsv.Substring(0,[Math]::Min(120,$dirsCsv.Length)))"

$argv = @(
  "train.py",
  "--data_root","data",
  "--datasets",$datasetsCsv,
  "--crop_export_dirs",$dirsCsv,
  "--epochs","300",
  "--encoder_spatial","128",
  "--num_points","28000",
  "--steps_per_epoch","6",
  "--fg_query_fraction","0.38",
  "--lr","5e-4",
  "--weight_decay","1e-5",
  "--out","checkpoints/model_hela2_hela3_mixed_augpool_modes.pth",
  "--train_log_csv","result/mixed_augpool_train_log.csv",
  "--train_log_every","20",
  "--val_every","10",
  "--val_log_csv","result/mixed_augpool_val_metrics.csv",
  "--val_num_points","4096",
  "--native_required"
)

& $py @argv

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done: checkpoints/model_hela2_hela3_mixed_augpool_modes.pth"

