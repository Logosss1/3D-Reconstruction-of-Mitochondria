# Reproducibility Notes

## Environment

The baseline environment is defined in [`environment.yml`](environment.yml):

- Python 3.9
- PyTorch 2.0.1
- CUDA toolkit 11.8 (GPU runs)
- NumPy, SciPy, scikit-image, Zarr, trimesh, TensorBoard, and related tools

Create the environment with:

```bash
conda env create -f environment.yml
conda activate mito3d_env
```

## Main commands

```bash
python hela2_mito_pipeline.py joint --epochs 300
python hela2_mito_pipeline.py per-crop --only-crop 9 --epochs 300
python hela2_mito_pipeline.py infer-all --checkpoint checkpoints/model_hela2_all_mito.pth --validate
```

## Record with each run

For a reproducible comparison, record the dataset accession, crop ID, model
checkpoint, training mode, epoch count, random seed (when set), inference
resolution, occupancy threshold, post-processing settings, GPU/CPU details,
and the commit hash of the code used.

Exact metrics can vary with crop selection, hardware, dependency resolution,
and post-processing. The repository should therefore be treated as an
experimental research pipeline rather than a bit-for-bit benchmark package.
