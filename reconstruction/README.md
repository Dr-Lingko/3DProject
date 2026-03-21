# Reconstruction Registration

Minimal helpers for point-cloud registration with Open3D.

## Quick Start

Run the multi-scale registration pipeline for a dataset folder:

```powershell
python E:\3DProject\reconstruction\main.py
```

Register a single pair (debug harness):

```powershell
python E:\3DProject\reconstruction\run_pair.py --source E:\3DProject\D2\main_object_fixed\1.ply --target E:\3DProject\D2\main_object_fixed\2.ply --output E:\3DProject\D2\main_object_fixed\pair_merged.ply
```

## Notes

- The pipeline uses RANSAC for coarse alignment and multi-scale ICP for refinement.
- Parameters are estimated per pair using nearest-neighbor distances.

