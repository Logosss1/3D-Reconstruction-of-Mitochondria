# One-off: infer on a NEW spatial window inside an existing crop (raw_subcrop), then save float + binary TIFF.
# MC occupancy is always mc_resolution^3 (default 128^3) on the encoder input — physical size is the subcrop box.
# Example: crop 155 is 800^3; take a 384^3 block at offset (100,100,100) — not used as a single fixed crop during training.
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$py = "E:\Anaconda\envs\mito3d_env\python.exe"
Set-Location $repo

$ckpt = Join-Path $repo "checkpoints\model_hela2_hela3_mixed.pth"
$out = Join-Path $repo "result\explore_subcrop_binary"
New-Item -ItemType Directory -Force -Path $out | Out-Null

& $py "generate.py" `
  --crop_export_dir (Join-Path $repo "data\crop_exports_hela2_mito_bg") `
  --dataset "jrc_hela-2" `
  --crop_id 155 `
  --checkpoint $ckpt `
  --output_dir $out `
  --name_prefix "sub384_" `
  --raw_subcrop "100,100,100,384,384,384" `
  --infer_max_spatial 128 `
  --mc_resolution 128 `
  --mc_level_mode "percentile" `
  --mc_percentile "90" `
  --mc_prob_smooth_sigma "0.75" `
  --mesh_strip_mc_shell_cells "3.0" `
  --save_mc_prob_tiff `
  --save_mc_prob_tiff_binary `
  --mc_binary_threshold "0.5"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Done. See $out for sub384_mc_prob_float32.tif and sub384_mc_prob_binary_uint8.tif"
