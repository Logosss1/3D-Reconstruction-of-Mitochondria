# Quick verification for this repo using the PyTorch conda env (mito3d_env).
# On this machine, `conda run -n mito3d_env` may bind to D:\anacondaEnvs\... (no torch).
# Use the explicit interpreter that has torch — typically:
#   E:\Anaconda\envs\mito3d_env\python.exe
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent
$candidates = @(
    "E:\Anaconda\envs\mito3d_env\python.exe",
    "D:\anacondaEnvs\envs\mito3d_env\python.exe"
)
$py = $null
foreach ($c in $candidates) {
    if (Test-Path $c) {
        & $c -c "import torch" 2>$null
        if ($LASTEXITCODE -eq 0) { $py = $c; break }
    }
}
if (-not $py) {
    Write-Error "No mito3d_env Python with torch found. Tried: $($candidates -join ', ')"
}
Write-Host "Using: $py"
Set-Location $repo
$smoke = "import numpy as np; import torch; from src.train_val_batch import build_fixed_val_batch; r=np.random.rand(64,64,64).astype('float32')*255; l=(np.random.rand(64,64,64)>0.85).astype('float32'); t=build_fixed_val_batch(r,l,32,1024,0.38,0,torch.device('cpu')); assert t is not None; print('OK: train_val_batch')"
& $py -c $smoke
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
Write-Host "Done."
