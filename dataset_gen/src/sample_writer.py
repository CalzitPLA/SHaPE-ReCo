#!/usr/bin/env python3
"""
The one writer that turns a FreeCAD-built STL into a registered training sample - shared by dataset_generator.py
(--with-field, --emit-negative-residual) and the other generators (I-beam, special shapes), so every sample of every
generator is voxelised, fielded and labelled the same way and lands in a manifest training/manifest.py reads.

    rec = finish_sample(stl_path, {"topology": "box", "subclass": "ibeam", "parameters": {...},
                                   "rotation": {...}, "placement": {"grid_mm": 40.0}}, out_root)

What it does, in order:
  1. voxelises the STL on the in-scale grid (voxelization.voxelize_stl_to_bin, scale_mode fixed, anchor origin: a cube
     of placement["grid_mm"] (default 40) centred on the WORLD origin) - the STL must already be placed in that frame;
     by ray parity at the voxel centres (D-vox; stltovoxel only if asked for: SHAPE_RECO_VOXELIZER=stltovoxel, the
     legacy voxelizer with its half-voxel anchor offset), placement.voxelizer = the voxelizer that ran (T-19, 2026-10-01);
     writes <stem>.bin next to it (array order [z, -y, x], as every .bin of the project). placement.degenerate = fewer than `min_voxels` (200) clean voxels;
     placement.manifold of the clean grid; with `noise`, the clean grid is kept as <stem>.clean.bin;
  2. <stem>.inside.npy, the base-point bits (src/inside_bits.py: parity of the placed solid on the same grid, whatever the
     voxelizer of step 1 was);
  3. with_field: <stem>.field.npz next to it (src/field_channels.py: occupancy, sdf positive inside with the parity sign,
     V = c(x) - x in mm, world axes, + `field_points` surface points with normals, + the grid spec);
  4. completes the record: stl_path / voxel_path relative to out_root, placement.voxels and voxel_volume_mm3 (the .bin's
     count x pitch^3), placement.field, subclass checked against SUBCLASSES, labels dims_mm / center_mm /
     rotation_canonical from an optional "_shape" entry ({"prim": sdf_primitives primitive in its local frame, "R": 3x3
     world rotation, "t": world translation of the local origin}, popped, never written); a FreeCAD volume the caller put
     in the record (any key) is kept as it is;
  5. appends it as one JSON line to <out_root>/<topology>/dataset_metadata.jsonl under an exclusive fcntl lock, so
     several processes can write into one shard directory (and each shard its own directory) without torn lines.

Nothing here builds geometry: the STL comes from FreeCAD (the project rule for all 3-D training data).
"""

from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import field_channels, inscale
from .voxelization import voxelize_stl_to_bin

SUBCLASSES = {
    "box": ("full", "hollow", "ibeam", "rhs", "frame"),
    "cylinder": ("full", "sector", "hollow", "tube", "frame"),
    "cone": ("full", "sector", "hollow", "frame"),
    "sphere": ("full", "sector", "band", "hollow", "frame"),
    "torus": ("full", "sector", "band", "hollow", "frame"),
}
DEFAULT_GRID_MM = 40.0
IDENTITY_ROTATION = dict(rx=0.0, ry=0.0, rz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)


def check_subclass(topology: str, subclass: str) -> None:
    if topology not in SUBCLASSES:
        raise ValueError(f"topology must be one of {tuple(SUBCLASSES)}, got {topology!r}")
    if subclass not in SUBCLASSES[topology]:
        raise ValueError(f"subclass of {topology} must be one of {SUBCLASSES[topology]}, got {subclass!r}")


def voxelize(stl_path: Path, bin_path: Path, resolution: int = 64, grid_mm: float = DEFAULT_GRID_MM,
             executable: str = "stltovoxel", voxelizer: Optional[str] = None, report: Optional[Dict] = None) -> np.ndarray:
    """The in-scale grid of the placed STL (the generator's own call; parity unless `voxelizer`/the env asks for
    stltovoxel); returns the (R, R, R) uint8 volume, .bin order. `report["voxelizer"]`: the voxelizer that ran."""
    ok = voxelize_stl_to_bin(stl_path=str(stl_path), output_bin=str(bin_path), resolution=resolution, executable=executable,
                             scale_mode="fixed", scale_mm=grid_mm, anchor="origin", voxelizer=voxelizer, report=report)
    if not ok:
        raise RuntimeError(f"voxelization failed: {stl_path}")
    return np.fromfile(bin_path, dtype=np.uint8).reshape((resolution,) * 3)


def write_field(stl_path: Path, npz_path: Path, resolution: int = 64, grid_mm: float = DEFAULT_GRID_MM,
                field_points: int = 2048) -> Dict:
    """<npz_path> with the field of the (placed) STL on the in-scale grid; returns the placement["field"] record.

    Field backend for NEW data (user decision 2026-09-24): libigl's exact closest point (SHAPE_RECO_FIELD_BACKEND=igl)
    whenever libigl is installed and no backend is set explicitly - exact everywhere, where the one-pass query missed by up
    to 0.24 mm, and 43x faster in generation (docs/data/generator_igl_e2e.json). The backend is recorded per sample.
    Setting the variable changes `field_backend()` for the rest of the CALLING process too (the GUI or a loop that calls
    `finish_sample`), not only for this sample. Without libigl and without the variable the field takes the slower
    refined-mesh path of field_channels.mesh_channels."""
    import os
    from .closest_point_field import field_backend
    if not os.environ.get("SHAPE_RECO_FIELD_BACKEND"):
        try:
            import igl  # noqa: F401
            os.environ["SHAPE_RECO_FIELD_BACKEND"] = "igl"             # this (generator) process only
        except ImportError:
            pass
    backend = field_backend()
    spec = field_channels.inscale_spec(grid_mm, resolution)
    t0 = time.time()
    field = field_channels.field_of_file(stl_path)                    # the STL read and the parity are shared with the
    inside = field_channels.parity_of_file(stl_path, spec)            # voxelizer and inside_bits of the same sample
    channels = field_channels.mesh_channels(field, spec, inside=inside)
    points = field_channels.surface_points(field, field_points, seed=0)
    seconds = round(time.time() - t0, 3)
    field_channels.save_field_npz(npz_path, channels, spec, points, source="mesh", seconds=seconds)
    return dict(path=Path(npz_path).name, source="mesh", closed=bool(field.is_closed()), seconds=seconds, backend=backend)


def append_record(metadata_file: Path, record: Dict) -> None:
    """One JSON line, appended under an exclusive lock (parallel writers into the same file never interleave)."""
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"     # = dataset_utils.json_dump_compact
    with open(metadata_file, "a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _labels_from_shape(shape: Dict) -> Dict:
    from . import solid_builder
    R = np.asarray(shape.get("R", np.eye(3)), float)
    placement = None
    t = shape.get("t")
    if t is not None:
        placement = dict(scale_factor=1.0, center_mm=list(np.asarray(t, float)))
    return solid_builder.labels(shape["prim"], R, placement, np.zeros(3) if placement else None)


DEGENERATE_VOXELS = 200     # the project's canon (CLAUDE.md): fewer clean voxels -> placement.degenerate, skipped in training


def finish_sample(stl_path: Path, record: dict, out_root: Path, *, resolution: int = 64, with_field: bool = True,
                  field_points: int = 2048, noise: Optional[Dict] = None, min_voxels: int = DEGENERATE_VOXELS,
                  voxelizer: Optional[str] = None) -> dict:
    """Voxelise + field + label + register one FreeCAD-built, already placed STL (see the module docstring). `record`
    needs at least "topology" (a key of SUBCLASSES); "subclass" defaults to "full", "mode" to "type", "parameters" to
    {}, "rotation" to the identity. Returns the record as written. `out_root` is the shard directory: the STL must be
    inside it (paths in the manifest are relative to it). `voxelizer`: None = $SHAPE_RECO_VOXELIZER, else parity
    (voxelization.voxelizer_for); "stltovoxel" only for legacy data.

    `noise` ({"config": voxel_noise.VoxelNoiseConfig, "prob": p}, voxel_noise.noise_spec_from_args): with probability p
    the .bin is noised in place after voxelization (the voxelization layer, as dataset_generator.py does), with an rng
    seeded from the sample's path, so a rerun gives the same grid. placement.noise records whether it was; voxels describes the
    stored (noised) grid, voxels_clean and manifold the clean one (as dataset_generator.py judges a sample), whose grid is kept as
    <stem>.clean.bin (training/field_input dirlog_inside reads it). The .field.npz stays the clean solid's."""
    stl_path, out_root = Path(stl_path).resolve(), Path(out_root).resolve()
    if out_root not in stl_path.parents:
        raise ValueError(f"{stl_path} is not inside the output root {out_root}")
    rec = dict(record)
    shape = rec.pop("_shape", None)
    topology = rec.get("topology")
    rec.setdefault("subclass", "full")
    check_subclass(topology, rec["subclass"])
    rec.setdefault("mode", "type")
    rec.setdefault("parameters", {})
    rec.setdefault("rotation", dict(IDENTITY_ROTATION))
    placement = dict(rec.get("placement") or {})
    grid_mm = float(placement.setdefault("grid_mm", DEFAULT_GRID_MM))
    placement.setdefault("mode", "fixed")
    placement.setdefault("cuts", [])

    bin_path = stl_path.with_suffix(".bin")
    ran: Dict = {}
    volume = voxelize(stl_path, bin_path, resolution, grid_mm, voxelizer=voxelizer, report=ran)
    placement["voxelizer"] = ran["voxelizer"]                          # the voxelizer that ran, not the env's wish
    placement["degenerate"] = bool(int((volume > 0).sum()) < min_voxels)  # canon: marked and skipped, never dropped
    placement["manifold"] = inscale.manifold_report(volume)             # of the CLEAN grid, as dataset_generator judges it
    if noise:
        import zlib
        from .voxel_noise import apply_voxel_noise
        rng = np.random.default_rng(zlib.crc32(str(stl_path.relative_to(out_root)).encode()))
        placement["voxels_clean"] = int((volume > 0).sum())
        placement["noise"] = bool(rng.random() < float(noise.get("prob", 1.0)))
        if placement["noise"]:
            volume.astype(np.uint8).tofile(bin_path.with_suffix(".clean.bin"))   # the clean grid (dirlog_inside's bit)
            volume = apply_voxel_noise(volume, noise["config"], rng)
            volume.astype(np.uint8).tofile(bin_path)
    placement["voxels"] = int((volume > 0).sum())
    from .inside_bits import write_inside                                  # the standard base-point bit, from the mesh
    placement["inside"] = write_inside(stl_path, bin_path, grid_mm, resolution)
    placement.setdefault("noise", False)
    rec["voxel_volume_mm3"] = float(placement["voxels"] * (grid_mm / resolution) ** 3)
    if with_field:
        placement["field"] = write_field(stl_path, stl_path.with_name(stl_path.stem + ".field.npz"), resolution, grid_mm,
                                         field_points)
    rec["placement"] = placement
    if shape is not None:
        for key, value in _labels_from_shape(shape).items():
            rec.setdefault(key, value)
    rec["stl_path"] = str(stl_path.relative_to(out_root))
    rec["voxel_path"] = str(bin_path.relative_to(out_root))
    append_record(out_root / topology / "dataset_metadata.jsonl", rec)
    return rec
