#!/usr/bin/env python3
"""
Fused network input from a binary occupancy grid: occupancy + signed distance + the vector to the nearest surface point.

The three views of one solid that this project has used separately:

  occupancy   {-1, +1}                         the voxel CNN's input (training/voxel_dataset.py)
  sdf         signed distance / truncation     negative inside, truncated at T voxels, in [-1, 1]   (training/sdf_input.py)
  field V     c(x) - x, scaled by 1/T          the vector from every voxel to the nearest surface point (the vector-field route,
                                               docs/VECTORFIELD.md): direction = the surface normal, length = |sdf|

V is derived from the grid itself, V = -f grad f with f the SDF (exact wherever f is differentiable), so the network needs no mesh and works
on any voxel grid, a scan included. The direction is taken from the gradient of the Gaussian-smoothed SDF (sigma 1 voxel): measured against the exact
mesh field on synthetic solids at 64^3 the median direction error is 2-5 degrees (box 5.3, cylinder 3.0, sphere 2.2), against 17-20 degrees for the
vector to the nearest voxel of the other class; the length error is 0.2 voxel. Vectors are in the grid's array axes (units of voxels, then / T), so
a rotation of the grid by a signed permutation of its axes (the augmentation of train_rotation_sym) must rotate the OCCUPANCY first and encode after:
`encode` does exactly that.

    encode(volume_pm1, "fused")       ->  (5, R, R, R) float32: occupancy, sdf, Vx, Vy, Vz
    encode(volume_pm1, "voxel_field") ->  (4, R, R, R) float32: occupancy, Vx, Vy, Vz  (voxelisation first, the field stacked on top;
                                          no separate sdf channel - the vector's length already IS the distance, so it adds nothing
                                          the network could not read off "field" alone, only redundant capacity to fit)
    load_vf_exact(voxel_path)         ->  (5, R, R, R) float32 "vf_exact": the same five channels as "fused" (same order, sign,
                                          scaling and array frame) but read from the sample's `<stem>.field.npz`, written by
                                          dataset_generator.py --with-field from the exact MESH field (dataset_gen/src/
                                          field_channels.py: parity sign, exact closest-point |sdf| and V) - not derivable
                                          from the grid, so `encode` refuses it
    encode(volume_pm1, "dirlog")      ->  (5, R, R, R) float32: material {0, 1}, unit outward normal nx, ny, nz, log distance
                                          (`dirlog_channels`). The "fused" layout multiplies the direction by the distance
                                          (V = -f grad f, clipped at T): the direction fades to 0 exactly at the surface,
                                          where it IS the normal, and beyond T voxels every channel saturates, a larger share
                                          of the grid the finer it is. Here the direction has full weight everywhere and
                                          the distance is compressed by a logarithm instead of truncated.
    load_exact_kind(voxel_path, "voxfield")  ->  (6, R, R, R) float32 "voxfield": the voxel grid and the EXACT surface field in
                                          one tensor - material, unit outward normal, saturated log distance (from the
                                          .field.npz), parity inside bit; stored sparse, computed once per sample. Defined in
                                          training/inputs.py (the train/infer input contract), which also holds the
                                          documented voxel-only fallback; `encode` refuses it like the other FROM_NPZ kinds.
    KINDS / CHANNELS                  the input kinds and their channel counts

Inference (state of 2026-10-01): reco_engine.predict._type_tensor encodes the loop's grid with `encode`, so a Type net
can be fed at inference only if its kind is grid-derived (occupancy, sdf, field, fused, voxel_field, dirlog,
dirlog_inside). The FROM_NPZ kinds (vf_exact, dirlog_exact, dirlog_curv, voxfield) load (model_registry knows their
channel counts) and then raise at the first prediction; dirlog_residual is built by the loop itself. The registry's type
stage fuses the descriptors with type_hier_dirlog_inside_medium (input dirlog_inside; models/registry.json, "type_stage");
its "default_type_entry" (the $SHAPE_RECO_REGISTRY override) is still the legacy occupancy net type_v3_B.
"""

from __future__ import annotations

import numpy as np

DEFAULT_TRUNCATION = 8.0
DEFAULT_SIGMA = 1.0
DEFAULT_LOG_SCALE = 1.0          # voxels: the distance below which d_log is ~linear (dirlog_channels)
KINDS = ("occupancy", "sdf", "field", "fused", "voxel_field", "vf_exact", "dirlog", "dirlog_exact", "dirlog_curv", "dirlog_residual",
         "dirlog_inside", "voxfield")
CHANNELS = {"occupancy": 1, "sdf": 1, "field": 3, "fused": 5, "voxel_field": 4, "vf_exact": 5, "dirlog": 5, "dirlog_exact": 5,
            "dirlog_curv": 8, "dirlog_residual": 7, "dirlog_inside": 6, "voxfield": 6}
FROM_NPZ = ("vf_exact", "dirlog_exact", "dirlog_curv", "voxfield")   # read from the sample's .field.npz (`load_exact_kind`)
VECTOR_CHANNELS = {"vf_exact": (2, 5), "dirlog_exact": (1, 4), "dirlog_curv": (1, 4), "dirlog_residual": (2, 5),
                   "voxfield": (1, 4)}
# "dirlog_residual" (dataset_gen/src/field_channels.dirlog_residual) needs a model as well as the input: built by the
# add-cut loop (reco_engine/residual_loop._State.dirlog_residual), not from one sample's files.


def _signed_distance(occ: np.ndarray) -> np.ndarray:
    """Signed distance in voxels between voxel centres, the surface half a voxel from the nearest centre of the other class."""
    from scipy import ndimage
    if not occ.any():
        return np.full(occ.shape, 1e3, np.float32)
    if occ.all():
        return np.full(occ.shape, -1e3, np.float32)
    d_out = ndimage.distance_transform_edt(~occ)
    d_in = ndimage.distance_transform_edt(occ)
    return np.where(occ, -(d_in - 0.5), d_out - 0.5).astype(np.float32)


def sdf_and_field(occupied: np.ndarray, truncation: float = DEFAULT_TRUNCATION, sigma: float = DEFAULT_SIGMA):
    """(sdf (R,R,R), V (3,R,R,R)) both scaled by 1/truncation with |V| <= 1; V = -sdf * unit gradient of the smoothed SDF."""
    from scipy import ndimage
    occ = np.asarray(occupied) > 0
    sdf = _signed_distance(occ)
    if sigma > 0:
        smooth = ndimage.gaussian_filter(sdf, sigma)
    else:
        smooth = sdf
    g = np.stack(np.gradient(smooth), axis=0)                                  # (3, R, R, R), array axes
    norm = np.maximum(np.linalg.norm(g, axis=0, keepdims=True), 1e-6)
    scaled = np.clip(sdf / truncation, -1.0, 1.0)
    field = (-scaled[None] * g / norm).astype(np.float32)
    return scaled.astype(np.float32), field


def log_distance(sdf_voxels: np.ndarray, resolution: int, log_scale: float = DEFAULT_LOG_SCALE) -> np.ndarray:
    """sign(d) log(1 + |d|/s) / log(1 + R/s): odd, monotone, ~linear within s voxels of the surface, no truncation;
    a distance of the whole grid size R maps to 1, so the range is resolution-independent."""
    d = np.asarray(sdf_voxels, np.float64)
    return (np.sign(d) * np.log1p(np.abs(d) / log_scale) / np.log1p(resolution / log_scale)).astype(np.float32)


def dirlog_channels(occupied: np.ndarray, sigma: float = DEFAULT_SIGMA, log_scale: float = DEFAULT_LOG_SCALE) -> np.ndarray:
    """(5, R, R, R) float32 [material {0,1}, nx, ny, nz, d_log] from a boolean grid (array axes). n is the UNIT outward
    normal field grad f / |grad f| of the smoothed SDF (f negative inside): continuous across the surface (the
    direction of V = -f grad f flips there; the flip is the material channel's job), length 1 everywhere, so no
    voxel's direction is weighted by its distance. d_log = `log_distance` of the (unsmoothed) SDF, negative inside."""
    from scipy import ndimage
    occ = np.asarray(occupied) > 0
    sdf = _signed_distance(occ)
    smooth = ndimage.gaussian_filter(sdf, sigma) if sigma > 0 else sdf
    g = np.stack(np.gradient(smooth), axis=0)
    norm = np.linalg.norm(g, axis=0, keepdims=True)
    n = np.where(norm > 1e-6, g / np.maximum(norm, 1e-6), 0.0)
    out = np.empty((5,) + occ.shape, np.float32)
    out[0] = occ
    out[1:4] = n
    out[4] = log_distance(np.clip(sdf, -1e3, 1e3), occ.shape[0], log_scale)
    return out


def dirlog_inside_channels(occupied: np.ndarray, inside: np.ndarray, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """(6, R, R, R) float32 "dirlog_inside": the five dirlog channels of the (possibly noisy) grid plus the BASE-POINT bit -
    is this voxel centre inside the exact mesh? - as channel 5, in {0, 1}.

    Why a separate bit: the dirlog channels are derived from the grid, and in training half the grids carry voxel noise
    (the project rule: every class noisy and clean), so the material channel and everything computed from it are
    disturbed. For an exact (closed) mesh, "is the voxel centre inside" is exactly what the voxeliser computes, so the bit is
    the CLEAN voxelisation. In training it is read in this order (training/voxel_dataset.py, gpu_encode.load_raw and
    training/inputs._sample_files each implement it): the sample's stored base-point bits `<stem>.inside.npy`
    (dataset_gen/src/inside_bits.py: ray parity of the PLACED solid on the exact in-scale grid, the standard since
    2026-09-23), else its `.clean.bin` when it was noised (scripts/add_voxel_noise.py keeps it), else the grid itself - which
    for a noisy sample without either file is the NOISY grid, silently. At inference: the loop's own grid, ray parity of the
    input STL/STEP (reco_engine/residual_loop, dataset_gen/src/field_channels.parity_occupancy), clean by construction,
    resampled onto the in-scale grid by reco_engine.cascade.inscale_target. Do not recompute it from the record's
    `stl_path`: for main-generator samples that is the UNPLACED macro STL (the placed solid is `<stem>.solid.stl`).
    Known offset (measured 2026-09-23, docs/training_plan_v5.tex F7): a legacy stltovoxel .bin sits up to half a voxel off the
    exact grid (anchor rounding), so channels 0-4 of a sample voxelised that way carry the offset while `.inside.npy` (and
    the inference grid) do not; on tiny bodies (a 26-voxel torus) such a .bin misses the solid entirely (IoU 0.0 against
    parity of the .solid.stl, which the .field.npz matches at 1.0). A parity .bin (the generators' default since
    2026-10-01, T-19; scripts/revoxelize_parity.py for older sets) has no offset; placement.voxelizer is the one that ran."""
    out = np.empty((6,) + np.asarray(occupied).shape, np.float32)
    out[:5] = dirlog_channels(occupied, sigma)
    out[5] = np.asarray(inside) > 0
    return out


def encode(volume_pm1: np.ndarray, kind: str = "fused", truncation: float = DEFAULT_TRUNCATION, sigma: float = DEFAULT_SIGMA,
           inside: np.ndarray = None) -> np.ndarray:
    """(C, R, R, R) network input for `kind` from a grid in the pipeline's {-1, +1} convention (training.voxel_dataset.load_voxel).
    `inside` (bool grid, same axes): the base-point bit of "dirlog_inside"; without it the grid's own occupancy is used,
    which is exact wherever the grid itself is clean (at inference: the loop's grid from the input mesh)."""
    if kind not in KINDS:
        raise ValueError(f"input kind must be one of {KINDS}, got {kind!r}")
    if kind == "dirlog_inside":
        occ = np.asarray(volume_pm1) > 0
        return dirlog_inside_channels(occ, occ if inside is None else inside, sigma)
    if kind == "dirlog_residual":
        raise ValueError("dirlog_residual needs the current model: field_channels.dirlog_residual / residual_loop._State.dirlog_residual")
    if kind in FROM_NPZ:
        raise ValueError(f"{kind} is read from the sample's .field.npz (load_exact_kind), it cannot be derived from a grid")
    if kind == "occupancy":
        return np.asarray(volume_pm1, np.float32)[None]
    if kind == "dirlog":
        return dirlog_channels(np.asarray(volume_pm1) > 0, sigma)
    sdf, field = sdf_and_field(np.asarray(volume_pm1) > 0, truncation, sigma)
    if kind == "sdf":
        return sdf[None]
    if kind == "field":
        return field
    occ = np.asarray(volume_pm1, np.float32)[None]
    if kind == "voxel_field":
        return np.concatenate([occ, field], axis=0)
    return np.concatenate([occ, sdf[None], field], axis=0)


def field_npz_path(voxel_path) -> "Path":
    """`<stem>.field.npz` next to a sample's `<stem>.bin` (or `.stl`) - where dataset_generator.py --with-field writes it."""
    from pathlib import Path
    p = Path(voxel_path)
    return p.with_name(p.stem + ".field.npz")


def load_vf_exact(voxel_path, truncation: float = DEFAULT_TRUNCATION) -> np.ndarray:
    """The "vf_exact" network input of one sample: its .field.npz through dataset_gen/src/field_channels.network_input -
    the SAME function the residual loop calls at inference, so training and inference cannot encode differently."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "dataset_gen"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.field_channels import load_field_npz, network_input
    path = field_npz_path(voxel_path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no field for this sample - generate the dataset with --with-field")
    channels, spec, _ = load_field_npz(path)
    return network_input(channels, spec, truncation)


EXACT_KIND_CACHE_VERSION = "1"    # part of the cache key with the rest of the code (dataset_gen/src/cache_keys.py)


def exact_kind_cache_path(voxel_path, kind: str):
    """Where `load_exact_kind` keeps a dirlog kind of one sample: $SHAPE_RECO_CACHE or ~/.cache/shape_reco, subdir
    field_kinds/ (float16 .npy), keyed (dataset_gen/src/cache_keys.py) by the .field.npz's identity (path, size, mtime),
    the kind and the code of this module and of field_channels (dirlog_input), constants included."""
    from dataset_gen.src.cache_keys import cache_key, cache_root
    key = cache_key("field_kinds", files=[field_npz_path(voxel_path)], code=[__name__, "src.field_channels"],
                    params=dict(kind=kind), length=24)
    return cache_root("field_kinds") / f"{key}.npy"


def load_exact_kind(voxel_path, kind: str, truncation: float = DEFAULT_TRUNCATION, cache: bool = True) -> np.ndarray:
    """Any of FROM_NPZ for one sample: "vf_exact" (`load_vf_exact`), "dirlog_exact" / "dirlog_curv"
    (dataset_gen/src/field_channels.dirlog_input: material, unit normal, log distance [, curvature, collapse]); the
    dirlog kinds are cached (`exact_kind_cache_path`): the curvature needs an eigen-decomposition per voxel, 1-10 s.
    The first (uncached) call returns the float32 tensor and stores it as float16; later calls return the float16-rounded
    copy, so one sample's tensor differs by the float16 rounding between the first epoch and the later ones.
    "voxfield": training/inputs.load_voxfield (its own sparse cache; without a .field.npz its documented voxel-only
    fallback, `inputs.voxfield_source` tells which)."""
    if kind == "vf_exact":
        return load_vf_exact(voxel_path, truncation)
    if kind == "voxfield":
        from training.inputs import load_voxfield
        return load_voxfield(voxel_path, cache=cache)
    import sys
    from pathlib import Path
    path = field_npz_path(voxel_path)
    if not path.is_file():
        raise FileNotFoundError(f"{path}: no field for this sample - generate the dataset with --with-field")
    cpath = exact_kind_cache_path(voxel_path, kind) if cache else None
    if cpath is not None and cpath.is_file():
        try:
            return np.load(cpath).astype(np.float32)
        except (OSError, ValueError):
            pass                                                         # a partly written file: recompute
    root = Path(__file__).resolve().parent.parent / "dataset_gen"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from src.field_channels import dirlog_input, load_field_npz
    channels, spec, _ = load_field_npz(path)
    x = dirlog_input(channels, spec, curvature=(kind == "dirlog_curv"))
    if cpath is not None:
        import os
        cpath.parent.mkdir(parents=True, exist_ok=True)
        tmp = cpath.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, x.astype(np.float16))
        os.replace(tmp, cpath)
    return x


def _precompute_one(args):
    load_exact_kind(*args)


def precompute_exact_kind(voxel_paths, kind: str, workers: int = 4) -> None:
    """Fill the cache of a dirlog kind for many samples in parallel."""
    from multiprocessing import Pool
    if kind == "voxfield":
        from training.inputs import voxfield_cache_path
        todo = [(p, kind) for p in voxel_paths if not voxfield_cache_path(p).is_file()]
    else:
        todo = [(p, kind) for p in voxel_paths if not exact_kind_cache_path(p, kind).is_file()]
    if todo:
        with Pool(min(workers, len(todo))) as pool:
            for _ in pool.imap_unordered(_precompute_one, todo, chunksize=2):
                pass
