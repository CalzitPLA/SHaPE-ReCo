#!/usr/bin/env python3
"""PyTorch Dataset wrapping manifest Records + their .bin voxel volumes."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from pathlib import Path

from training.manifest import Record

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    # load_voxel() below has no torch dependency - only VoxelDataset does.
    # Keep this module importable (for reco_engine.predict's subprocess-
    # mode fallback path) even where torch isn't installed.
    Dataset = object


def load_voxel(bin_path, resolution: int) -> np.ndarray:
    """Load a resolution^3 uint8 {0,1} volume, normalized to {-1, +1}."""
    arr = np.fromfile(bin_path, dtype=np.uint8)
    expected = resolution ** 3
    if arr.size != expected:
        raise ValueError(
            f"{bin_path}: expected {expected} voxels for resolution "
            f"{resolution}, got {arr.size}"
        )
    vol = arr.reshape(resolution, resolution, resolution).astype(np.float32)
    # {0, 1} -> {-1, +1}
    return vol * 2.0 - 1.0


class VoxelDataset(Dataset):
    """Base dataset: yields (voxel_tensor[C,R,R,R], Record). `input_kind` "occupancy" ({-1,+1}, the default, C=1), "sdf"
    (training/sdf_input.py: the signed distance of the grid, truncated and scaled to [-1, 1], cached in `sdf_cache`, C=1), "field" (the vector to
    the nearest surface point, C=3), "fused" (occupancy + sdf + vector, C=5), "voxel_field" (occupancy + vector, C=4, no sdf channel;
    training/field_input.py, computed on the fly) or "exact_field" (the mesh's own closest-surface-point field, C=3,
    training/exact_field_input.py, cached - at its own `field_resolution`, independent of `resolution`, default Nyquist = 2x) or
    "vf_exact" (occupancy + sdf + V of the exact mesh field, C=5, "fused" layout, read from the sample's .field.npz -
    dataset_generator.py --with-field; training/field_input.load_vf_exact) or "voxfield" (grid + exact surface field, C=6,
    training/inputs.py, sparse-cached) or "dirlog" (material, unit normal, log distance of the grid, C=5, on the fly) or
    "dirlog_inside" (dirlog + the base-point bit, C=6: <stem>.inside.npy, else <stem>.clean.bin, else the grid itself -
    the same order as gpu_encode.load_raw and inputs._sample_files) or "dirlog_exact" / "dirlog_curv" (C=5 / 8, from
    the .field.npz, cached; field_input.load_exact_kind). The CPU path; --gpu-encode uses training/gpu_encode.py for
    occupancy, dirlog and dirlog_inside."""

    def __init__(self, records: List[Record], resolution: int = 64, input_kind: str = "occupancy",
                 sdf_truncation: float = 8.0, sdf_cache=None, field_resolution: int = None,
                 field_grid_mm: float = 40.0, field_cache=None):
        self.records = records
        self.resolution = resolution
        self.input_kind, self.sdf_truncation, self.sdf_cache = input_kind, sdf_truncation, sdf_cache
        from training.exact_field_input import field_resolution_for
        self.field_resolution = field_resolution or field_resolution_for(resolution)
        self.field_grid_mm, self.field_cache = field_grid_mm, field_cache

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, idx: int) -> Tuple[torch.Tensor, Record]:
        record = self.records[idx]
        if self.input_kind == "sdf":
            from training.sdf_input import load_sdf_input
            vol = load_sdf_input(record.voxel_path, self.resolution, self.sdf_truncation, self.sdf_cache)
            tensor = torch.from_numpy(vol).unsqueeze(0)  # (1, R, R, R)
        elif self.input_kind in ("field", "fused", "voxel_field", "dirlog"):
            from training.field_input import encode
            tensor = torch.from_numpy(encode(load_voxel(record.voxel_path, self.resolution), self.input_kind, self.sdf_truncation))
        elif self.input_kind == "dirlog_inside":
            from training.field_input import encode
            vol = load_voxel(record.voxel_path, self.resolution)
            from dataset_gen.src.inside_bits import read_inside           # the stored standard: parity of the placed mesh
            inside = read_inside(record.voxel_path, self.resolution)
            if inside is None:                                             # old data: the clean copy, else the grid itself
                clean = Path(record.voxel_path).with_name(Path(record.voxel_path).stem + ".clean.bin")
                inside = load_voxel(clean, self.resolution) > 0 if clean.exists() else vol > 0
            tensor = torch.from_numpy(encode(vol, "dirlog_inside", self.sdf_truncation, inside=inside))
        elif self.input_kind in ("vf_exact", "dirlog_exact", "dirlog_curv", "voxfield"):
            from training.field_input import load_exact_kind
            tensor = torch.from_numpy(load_exact_kind(record.voxel_path, self.input_kind, self.sdf_truncation))
        elif self.input_kind == "exact_field":
            from training.exact_field_input import load_exact_field_input
            grid_mm = (record.placement or {}).get("grid_mm", self.field_grid_mm)
            tensor = torch.from_numpy(load_exact_field_input(record.stl_path, self.field_resolution, grid_mm, self.field_cache))
        else:
            tensor = torch.from_numpy(load_voxel(record.voxel_path, self.resolution)).unsqueeze(0)  # (1, R, R, R)
        return tensor, record
