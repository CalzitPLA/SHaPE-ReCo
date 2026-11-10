#!/usr/bin/env python3
"""
Signed distance field as network input, computed from the binary occupancy grid the pipeline already has.

The SDF of a binary grid is a distance transform of the body and of its complement (distances between voxel centres); the
surface lies half a voxel from the nearest centre of the other class, so half a voxel is subtracted on both sides
(measured: 0.19 voxel mean error near the surface on clean grids, docs/sdf_background.tex). Negative inside, positive
outside, in voxels, then truncated at `truncation` voxels and scaled to [-1, 1] - the range of the {-1, +1} occupancy input.

`encode` is the one function training, evaluation and the probes use to turn a loaded grid into a network input, so a network
trained on SDF input is always fed the same encoding it was trained on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_TRUNCATION = 8.0                 # voxels (0.625 mm each in the in-scale mode)
KINDS = ("occupancy", "sdf")


def sdf_from_occupancy(occupied: np.ndarray, truncation: float = DEFAULT_TRUNCATION) -> np.ndarray:
    """(R,R,R) float32 in [-1, 1]: signed distance / truncation, negative inside."""
    from scipy import ndimage
    occ = np.asarray(occupied) > 0
    if not occ.any():
        return np.ones(occ.shape, np.float32)                            # nothing inside: everything is far outside
    if occ.all():
        return -np.ones(occ.shape, np.float32)
    d_out = ndimage.distance_transform_edt(~occ)
    d_in = ndimage.distance_transform_edt(occ)
    sdf = np.where(occ, -(d_in - 0.5), d_out - 0.5)
    return np.clip(sdf / truncation, -1.0, 1.0).astype(np.float32)


def encode(volume_pm1: np.ndarray, kind: str = "occupancy", truncation: float = DEFAULT_TRUNCATION) -> np.ndarray:
    """Network input from a grid in the pipeline's {-1, +1} convention (see training.voxel_dataset.load_voxel)."""
    if kind == "occupancy":
        return volume_pm1
    if kind == "sdf":
        return sdf_from_occupancy(volume_pm1 > 0, truncation)
    raise ValueError(f"input kind must be one of {KINDS}, got {kind!r}")


def cache_path(cache_dir: Path, voxel_path, truncation: float) -> Path:
    """The SDF input's file in `cache_dir`, keyed (dataset_gen/src/cache_keys.py) by the .bin's identity (path, size,
    mtime: scripts/revoxelize_parity.py and add_voxel_noise.py rewrite it in place), the truncation and this module's
    code. The resolution is the .bin's size."""
    from dataset_gen.src.cache_keys import cache_key
    key = cache_key("sdf", files=[voxel_path], code=[__name__], params=dict(truncation=float(truncation)), length=24)
    return Path(cache_dir) / f"{key}.npy"


def load_sdf_input(voxel_path, resolution: int, truncation: float = DEFAULT_TRUNCATION,
                   cache_dir: Optional[Path] = None) -> np.ndarray:
    """The SDF input of a stored .bin grid, from the cache when there is one (float16 files, computed once)."""
    from training.voxel_dataset import load_voxel
    if cache_dir is not None:
        path = cache_path(cache_dir, voxel_path, truncation)
        if path.is_file():
            return np.load(path).astype(np.float32)
    sdf = sdf_from_occupancy(load_voxel(voxel_path, resolution) > 0, truncation)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(path, sdf.astype(np.float16))
    return sdf


def _fill_one(args):
    voxel_path, resolution, truncation, cache_dir = args
    load_sdf_input(voxel_path, resolution, truncation, cache_dir)


def precompute(voxel_paths: Sequence, resolution: int, truncation: float, cache_dir: Path, workers: int = 12) -> None:
    """Compute and cache the SDF input of many grids in parallel (each takes about 65 ms)."""
    from multiprocessing import Pool
    todo = [(p, resolution, truncation, str(cache_dir)) for p in voxel_paths if not cache_path(cache_dir, p, truncation).is_file()]
    if not todo:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with Pool(min(workers, len(todo))) as pool:
        for _ in pool.imap_unordered(_fill_one, todo, chunksize=16):
            pass
