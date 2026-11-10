#!/usr/bin/env python3
"""
The base-point bit, stored as standard (user decision 2026-09-23): for every voxel centre of a sample's in-scale grid,
is that point (the base point of the closest-point vector) inside the MESH? One bool per voxel, written next to the .bin
as `<stem>.inside.npy` (np.packbits, 32 KB at 64^3), in the .bin array order [z, -y, x].

Computed from the mesh, never from voxels: scanline ray parity (field_channels.parity_occupancy) of the PLACED solid on
the exact in-scale grid (field_channels.inscale_spec). So it is free of voxel noise and of the stltovoxel half-voxel anchor
offset by construction, and it equals the reconstruction loop's inference grid (the loop uses the same parity on its exact
grid) and the parity .bin of a clean sample (voxelization._voxelize_parity, the default voxelizer since 2026-10-01).

The placed solid: `<stem>.solid.stl` when it exists (the main generator writes the placed, modified solid there; the
record's stl_path is then the UNPLACED macro STL), else the STL itself (sample_writer's special and I-beam samples are
placed already).

    write_inside(stl_path, bin_path, grid_mm=40.0, resolution=64)  -> "<stem>.inside.npy" (name relative to the .bin)
    read_inside(bin_path, resolution=64)                            -> bool (R, R, R) in .bin order, or None if missing
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

SUFFIX = ".inside.npy"


def inside_path(bin_path) -> Path:
    b = Path(bin_path)
    return b.with_name(b.stem + SUFFIX)


def placed_solid(stl_path) -> Path:
    """The solid the .bin describes: <stem>.solid.stl if present (main generator), else the STL itself."""
    s = Path(stl_path)
    solid = s.with_name(s.stem + ".solid.stl")
    return solid if solid.exists() else s


def compute_inside(solid_stl, grid_mm: float = 40.0, resolution: int = 64) -> np.ndarray:
    """bool (R, R, R) in the .bin order: parity of the placed solid at the voxel centres of the in-scale grid."""
    from src.field_channels import inscale_spec, parity_of_file, world_to_bin
    world = parity_of_file(solid_stl, inscale_spec(float(grid_mm), int(resolution)))     # shared with the voxelizer / field
    return world_to_bin(world).astype(bool)


def write_inside(stl_path, bin_path, grid_mm: float = 40.0, resolution: int = 64) -> str:
    """Compute and store the bits for one sample; returns the file name relative to the .bin's directory."""
    inside = compute_inside(placed_solid(stl_path), grid_mm, resolution)
    out = inside_path(bin_path)
    np.save(out, np.packbits(inside.ravel()))
    return out.name


def read_inside(bin_path, resolution: int = 64) -> Optional[np.ndarray]:
    """The stored bits as bool (R, R, R) in .bin order, None without a file. The size comes from `resolution`, not from
    the file (B-051): a 64^3 file read with resolution 32 returns its first 32 768 bits reshaped, a scrambled grid, with
    no error; a 32^3 file read with 64 raises (measured by review R-04, 2026-10-01). The training readers pass the
    resolution of the sample's .bin (training.inputs derives it from the .bin size; voxel_dataset.load_voxel and the
    reshape in gpu_encode.load_raw check the .bin against it), so only a stale .inside.npy of another resolution next
    to a valid .bin goes undetected."""
    p = inside_path(bin_path)
    if not p.exists():
        return None
    bits = np.unpackbits(np.load(p))[: resolution ** 3]
    return bits.reshape((resolution,) * 3).astype(bool)
