#!/usr/bin/env python3
"""
Exact vector-field network input: the closest-surface-point field V(x) = c(x) - x of the STL mesh itself
(dataset_gen/src/closest_point_field.py), not the smoothed-gradient approximation training/field_input.py derives
from the occupancy grid (2-5 deg off exact there). Needs the mesh, so it is keyed by `Record.stl_path`, not by
the voxel grid - and its resolution is therefore independent of the occupancy grid's `--resolution`.

Resolution: the occupancy grid can only resolve detail down to one voxel; sampling the (continuous) exact field on
that same grid would alias it to the same limit for nothing gained. Nyquist-Shannon says at least 2x the voxel
grid's sample rate is needed to resolve everything the voxel grid itself resolves, so `field_resolution_for`
defaults the field grid to 2x the occupancy resolution (`--field-resolution` overrides it).

    load_exact_field_input(stl_path, resolution, grid_mm, cache_dir)   ->  (3, resolution, resolution, resolution) float32
    precompute(records, resolution, grid_mm, cache_dir)                ->  fills the cache in parallel (one mesh load per sample)

Cost: exact=False grid_vectors at resolution R takes about 0.9 s (R=64) / 2.9 s (R=96) / 5.5 s (R=128) per mesh on one
core (measured on a carve-shard STL); always cached (float16 .npy, like training/sdf_input.py) since it is far more
expensive than the on-the-fly channels of field_input.py.

LEGACY (state of 2026-10-01; review R-04 F5, R-05 B8). No production code calls this module (only its tests and the
CHANNELS constant in reco_engine/model_registry.py); the stored .field.npz (field_channels) and "vf_exact"/"voxfield"
superseded it. Known limits: `SurfaceField.grid_vectors` centres the grid on the MESH's bounding-box centre, while an
in-scale sample's grid is centred on the world origin with the part at a random position, and for main-generator
samples `Record.stl_path` is the UNPLACED macro STL (the placed solid is <stem>.solid.stl, dataset_gen/src/
inside_bits.py), so this field is not aligned with the sample's voxels. The cache key (`cache_path`) holds the STL path,
resolution and grid_mm only (no mtime, size or backend): an STL rewritten at the same path is served stale. The engine
refuses the kind (model_registry._type_cnn_input_kind).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

DEFAULT_GRID_MM = 40.0
DEFAULT_OVERSAMPLE = 2          # Nyquist: >=2x the voxel grid's sample rate to resolve what it resolves
CHANNELS = 3

_DATASET_GEN = Path(__file__).resolve().parent.parent / "dataset_gen"


def field_resolution_for(voxel_resolution: int, oversample: int = DEFAULT_OVERSAMPLE) -> int:
    """The default field-grid resolution for a voxel grid of `voxel_resolution` (Nyquist: oversample x)."""
    return int(voxel_resolution) * int(oversample)


def cache_path(cache_dir: Path, stl_path, resolution: int, grid_mm: float) -> Path:
    """The exact-field input's file in `cache_dir`, keyed (dataset_gen/src/cache_keys.py) by the STL's identity (path,
    size, mtime), the grid, the requested field backend ($SHAPE_RECO_FIELD_BACKEND: "igl" is exact, the others are the
    one-pass query; read from the env, not resolved by field_backend(), which would import torch before precompute's
    fork) and the code of this module and of closest_point_field."""
    import os
    if str(_DATASET_GEN) not in sys.path:
        sys.path.insert(0, str(_DATASET_GEN))
    from src.cache_keys import cache_key
    backend = os.environ.get("SHAPE_RECO_FIELD_BACKEND", "auto").strip().lower() or "auto"
    key = cache_key("exact_field", files=[stl_path], code=[__name__, "src.closest_point_field"],
                    params=dict(resolution=int(resolution), grid_mm=float(grid_mm), backend=backend), length=24)
    return Path(cache_dir) / f"{key}.npy"


def compute_exact_field(stl_path, resolution: int, grid_mm: float = DEFAULT_GRID_MM) -> np.ndarray:
    """(3, resolution, resolution, resolution) float32, array axes, in voxels of the FIELD grid (not the occupancy
    grid's) - the exact closest-surface-point vector of the mesh itself, centred on the mesh's own bounding box.
    That is NOT the in-scale occupancy grid's centre (the world origin) for a placed sample, and for a main-generator
    record `stl_path` is the unplaced macro STL (module docstring)."""
    if str(_DATASET_GEN) not in sys.path:
        sys.path.insert(0, str(_DATASET_GEN))
    from src.closest_point_field import load_field
    field = load_field(stl_path)
    return field.grid_vectors(resolution, grid_mm).astype(np.float32)


def load_exact_field_input(stl_path, resolution: int, grid_mm: float = DEFAULT_GRID_MM,
                           cache_dir: Optional[Path] = None) -> np.ndarray:
    """The exact-field input of one STL, from the cache when there is one (float16 files, computed once)."""
    if cache_dir is not None:
        path = cache_path(cache_dir, stl_path, resolution, grid_mm)
        if path.is_file():
            return np.load(path).astype(np.float32)
    field = compute_exact_field(stl_path, resolution, grid_mm)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        np.save(path, field.astype(np.float16))
    return field


def _fill_one(args):
    stl_path, resolution, grid_mm, cache_dir = args
    load_exact_field_input(stl_path, resolution, grid_mm, cache_dir)


def precompute(stl_paths: Sequence, resolution: int, cache_dir: Path, grid_mm: float = DEFAULT_GRID_MM, workers: int = 12) -> None:
    """Compute and cache the exact-field input of many meshes in parallel.

    The workers are SPAWNED, not forked (T-06, 2026-10-01): once torch has run a parallel op here (GNU OpenMP, the
    libgomp torch ships), a forked worker's own first parallel op waits at a barrier for pool threads that exist only
    in the parent - a hang at 0 % CPU (gdb: gomp_team_barrier_wait_end in at::index_kernel; the laptop and server_b).
    A worker's start (about a second) is small against one mesh (0.9-5.5 s)."""
    from multiprocessing import get_context
    todo = [(p, resolution, grid_mm, str(cache_dir)) for p in stl_paths if not cache_path(cache_dir, p, resolution, grid_mm).is_file()]
    if not todo:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(min(workers, len(todo))) as pool:
        for _ in pool.imap_unordered(_fill_one, todo, chunksize=4):
            pass
