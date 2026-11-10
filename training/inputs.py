#!/usr/bin/env python3
"""
The network input contract (plan item S4): ONE encoding for training and inference, "voxfield" - the voxel grid and the
exact surface vector field combined in one tensor (user requirement 2026-09-27: "voxels plus the surface vector field,
combined in the training data; computed once; sparse to save memory").

State of 2026-10-01: the ENCODER is shared (`voxfield_channels`; tests/test_inputs_voxfield.py checks train = infer),
but no Type net reads voxfield at inference: reco_engine.predict._type_tensor goes through training.field_input.encode,
which refuses voxfield (a FROM_NPZ kind), reco_engine.stage_registry.INPUT_KINDS does not list it, and
model_registry.registry_type_dir admits only occupancy/sdf/fused. A voxfield Type checkpoint therefore loads
(model_registry._type_cnn_input_kind knows its 6 channels) and fails at the first prediction. The one inference
consumer is reco_engine.feature_groups.piece_tensor, which takes the voxel-only fallback below. The registry's type
network reads "dirlog_inside" (training/field_input.py), whose channel order this kind shares.

    voxfield_channels(material, inside, sdf, vector, pitch)  -> (6, X, Y, Z) float32   the pure encoding (no files, no torch)
    voxfield_from_world(material_w, inside_w, sdf_w, vector_w, pitch)                   the same from WORLD-axes arrays
    load_voxfield(voxel_path)                                -> (6, R, R, R) float32   one sample (training), cached, sparse
    voxfield_to_sparse / voxfield_from_sparse / save_voxfield_sparse / load_voxfield_sparse    the on-disk band format

Channels (the order of "dirlog_inside", so the two are drop-in replacements for each other and VECTOR_CHANNELS = (1, 4)):

  0     material  {0, 1}   the (possibly noisy) voxel grid - where noise lives (project rule: every class noisy and clean,
                           noise in the voxelisation layer only)
  1..3  n                  UNIT outward normal of the level set through the voxel centre, from the EXACT field:
                           n = V/|V| inside, -V/|V| outside, V = c(x) - x the vector to the closest surface point. This is the
                           direction to the closest surface point signed by the parity bit (it points to the surface from
                           inside, away from it outside). Not the raw direction to the surface: that one flips at the surface,
                           this one is continuous across it (the flip is carried by channel 5), as in dirlog. Where |V| is
                           below 1e-6 voxel (a centre on the surface) n is the unit gradient of the signed distance.
                           0 where |d| >= SATURATION_VOXELS (see below).
  4     d_log              sign(d) log(1 + min(|d|, S)/s) / log(1 + LOG_NORM_VOXELS/s), d the distance to the surface in
                           voxels, NEGATIVE inside, the sign taken from the parity bit (channel 5), never from the field's
                           own sign; s = log_scale (1 voxel). The normaliser is a FIXED 64 voxels, not the grid's shape, so a
                           value means the same distance on the 64^3 training grid, on the loop's box grid (88 x 562 x 419)
                           and in "dirlog_inside" (whose d_log on a 64^3 grid uses the same normaliser) - for |d| < S the
                           two kinds' d_log scales are identical.
  5     inside    {0, 1}   the base-point bit: ray parity of the exact mesh at the voxel centre (dataset_gen/src/
                           inside_bits.py, the D-vox ground truth) - the CLEAN voxelisation.

Grid-derived direction/distance (dirlog's channels 1..4): REPLACED, not kept next to the exact ones. Reasons:
  - at inference on the input mesh the loop's grid is clean, so dirlog's channels are just a degraded copy of the exact field
    (median direction error 2-5 degrees, field_input.py's docstring) - redundant capacity to fit;
  - in training they are computed from the NOISY grid, so they would carry the voxel noise into four more channels while at
    inference they are clean: a train/infer gap that the exact channels do not have (noise stays in channel 0);
  - four more channels in memory and on disk for every sample.
The grid-derived field survives in exactly one place: the FALLBACK below.

Fallback when only voxels exist (`sdf` and `vector` both None) - the loop's leftover pieces, which have no mesh
(reco_engine/residual_loop.py `_voxel_channels`), and training samples without a `.field.npz`: d and n are computed from the
INSIDE bits exactly as dirlog computes them from a grid (field_input._signed_distance: Euclidean distance transform, the
surface half a voxel from the nearest centre of the other class; n = unit gradient of the sigma-1 Gaussian-smoothed SDF),
then encoded with the SAME saturation, log and quantisation as the exact path. Not the blocky-surface field of
field_channels.voxel_channels: its closest points lie on axis-aligned voxel faces, so its directions are staircase normals.
The fallback is a different input distribution (the Type net is trained on the exact path); it is selected explicitly by
passing no field (`encode(..., "voxfield")` refuses, so a checkpoint cannot be fed it by accident), `load_voxfield` reports
which path it took, and train_type_cnn refuses a training set that mixes the two unless told to (a class-correlated field
coverage - the v5 medium set: 19 % of the boxes have a .field.npz, 68 % of the spheres - would be a shortcut).

Saturation (why the tensor is sparse): beyond SATURATION_VOXELS from the surface n = 0 and d_log = +-d_log(S). So outside
the band |d| < S the tensor is fully determined by channel 5 and is not stored: the sparse file holds the two bit grids
(np.packbits) plus int32 linear indices and float16 values of the four field channels for every voxel that differs from
that rule. Every value is quantised to float16 by the encoding itself (the output is float32 holding float16 values), so
the sparse round trip is bit-exact and a cached sample equals a freshly encoded one bit for bit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from training.field_input import DEFAULT_SIGMA, _signed_distance

VOXFIELD = "voxfield"
CHANNELS = 6
VECTOR_CHANNELS = (1, 4)
SATURATION_VOXELS = 16.0         # |d| beyond this: n = 0, d_log saturated (measured choice, see SATURATION_NOTE)
LOG_NORM_VOXELS = 64.0           # d_log = 1 at this distance (not reached below S); the in-scale grid's resolution
DEFAULT_LOG_SCALE = 1.0          # voxels: d_log ~linear within this distance of the surface (field_input.DEFAULT_LOG_SCALE)
TINY_VOXELS = 1e-6               # |V| below this (in voxels): the centre is on the surface, n from the distance gradient
CACHE_VERSION = "1"              # part of the cache key, as is the code of VOXFIELD_CODE (dataset_gen/src/cache_keys.py)
# the modules whose code makes a cached voxfield tensor (`voxfield_cache_path`), named as the encoder imports them
VOXFIELD_CODE = ("training.inputs", "training.field_input", "src.field_channels", "dataset_gen.src.inside_bits")


def _quantise(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).astype(np.float16).astype(np.float32)


def saturated_log(saturation: float = SATURATION_VOXELS, log_scale: float = DEFAULT_LOG_SCALE) -> float:
    """|d_log| of every voxel at or beyond the saturation distance (before quantisation)."""
    return float(np.log1p(saturation / log_scale) / np.log1p(LOG_NORM_VOXELS / log_scale))


def _grid_distance_and_normal(inside: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """The fallback's (d in voxels, negative inside; unit normal (3, ...)) from a bool grid - the arithmetic of
    field_input.dirlog_channels, for any grid shape."""
    from scipy import ndimage
    d = _signed_distance(inside).astype(np.float64)
    smooth = ndimage.gaussian_filter(d, sigma) if sigma > 0 else d
    g = np.stack(np.gradient(smooth), axis=0)
    norm = np.linalg.norm(g, axis=0, keepdims=True)
    return d, np.where(norm > 1e-6, g / np.maximum(norm, 1e-6), 0.0)


def voxfield_channels(material: np.ndarray, inside: np.ndarray, sdf: Optional[np.ndarray] = None,
                      vector: Optional[np.ndarray] = None, pitch: float = 1.0, log_scale: float = DEFAULT_LOG_SCALE,
                      saturation: float = SATURATION_VOXELS, sigma: float = DEFAULT_SIGMA) -> np.ndarray:
    """(6, X, Y, Z) float32 "voxfield" (layout in the module docstring), values exactly representable in float16.

    material  (X, Y, Z) grid, > 0 = material (the possibly noisy voxelisation)
    inside    (X, Y, Z) bool: the parity base-point bits (clean); the sign of d
    sdf       (X, Y, Z) distance to the surface in the units of `pitch` (mm); only |sdf| is used - any sign convention
    vector    (3, X, Y, Z) V = c(x) - x in the units of `pitch`, components in the SAME axes as the array axes (use
              `voxfield_from_world` for world-axes data); only its direction is used
    pitch     voxel size in the units of sdf/vector (distances are converted to voxels)
    sdf and vector are rounded to float16 first - the precision of the stored .field.npz - so a loop passing its float32
    mesh_channels gets bit for bit what training read from the file of the same solid.
    sdf = vector = None: the voxel-only fallback (module docstring). Any grid shape; no file access."""
    occ = np.asarray(material) > 0
    ins = np.asarray(inside) > 0
    if occ.shape != ins.shape:
        raise ValueError(f"voxfield: material {occ.shape} and inside {ins.shape} differ")
    if (sdf is None) != (vector is None):
        raise ValueError("voxfield: pass both sdf and vector (exact field) or neither (voxel-only fallback)")
    if sdf is None:
        d, n = _grid_distance_and_normal(ins, sigma)
    else:
        a = np.abs(_quantise(sdf).astype(np.float64)) / float(pitch)        # the field as the dataset stores it (float16,
        V = _quantise(vector).astype(np.float64)                             # field_channels.save_field_npz): train = infer
        if a.shape != ins.shape or V.shape != (3,) + ins.shape:
            raise ValueError(f"voxfield: sdf {a.shape} / vector {V.shape} do not match the grid {ins.shape}")
        d = np.where(ins, -a, a)
        length = np.linalg.norm(V, axis=0, keepdims=True)
        n = np.where(ins[None], 1.0, -1.0) * V / np.maximum(length, 1e-300)
        tiny = length[0] < TINY_VOXELS * float(pitch)
        if tiny.any():                                                          # on the surface: the distance gradient
            g = np.stack(np.gradient(d), axis=0)
            g /= np.maximum(np.linalg.norm(g, axis=0, keepdims=True), 1e-12)
            n[:, tiny] = g[:, tiny]
    far = np.abs(d) >= saturation
    n[:, far] = 0.0
    out = np.empty((CHANNELS,) + ins.shape, np.float32)
    out[0] = occ
    out[1:4] = n
    out[4] = np.sign(d) * np.log1p(np.minimum(np.abs(d), saturation) / log_scale) / np.log1p(LOG_NORM_VOXELS / log_scale)
    out[5] = ins
    return _quantise(out)


def world_to_array(scalar_w: np.ndarray) -> np.ndarray:
    """A world-axes [x, y, z] grid -> the network's array frame [z, -y, x] (field_channels.world_to_bin)."""
    return np.ascontiguousarray(np.asarray(scalar_w).transpose(2, 1, 0)[:, ::-1, :])


def vector_world_to_array(vector_w: np.ndarray) -> np.ndarray:
    """(3, X, Y, Z) world-axes vectors -> array frame, grid AND components: (a0, a1, a2) = (vz, -vy, vx)."""
    v = np.asarray(vector_w)
    return np.stack([world_to_array(v[2]), -world_to_array(v[1]), world_to_array(v[0])])


def voxfield_from_world(material_w: np.ndarray, inside_w: np.ndarray, sdf_w: Optional[np.ndarray] = None,
                        vector_w: Optional[np.ndarray] = None, pitch: float = 1.0, **kwargs) -> np.ndarray:
    """`voxfield_channels` of world-axes arrays ([x, y, z], field_channels.mesh_channels / parity_occupancy), returned in the
    array frame [z, -y, x] of the type nets. The inputs are moved into the array frame FIRST and encoded there - the
    training path (`load_voxfield`) does the same, so a loop that calls this gets the training encoding bit for bit."""
    vec = None if vector_w is None else vector_world_to_array(vector_w)
    sdf = None if sdf_w is None else world_to_array(sdf_w)
    return voxfield_channels(world_to_array(material_w), world_to_array(inside_w), sdf, vec, pitch, **kwargs)


# ---------------------------------------------------------------------------------------------------- sparse band format

def voxfield_to_sparse(x: np.ndarray, saturation: float = SATURATION_VOXELS, log_scale: float = DEFAULT_LOG_SCALE) -> dict:
    """The arrays of the sparse file: shape, packed material and inside bits, and (int32 linear index, float16 [nx, ny, nz,
    d_log]) of every voxel whose field channels differ from the outside rule (n = 0, d_log = -+saturated by the inside bit)."""
    x = np.asarray(x, np.float32)
    shape = x.shape[1:]
    ins = x[5] > 0.5
    rule = _outside_rule(ins, saturation, log_scale)
    diff = np.any(x[1:5] != rule, axis=0).ravel()
    idx = np.flatnonzero(diff).astype(np.int32)
    val = x[1:5].reshape(4, -1)[:, idx].T.astype(np.float16)
    return dict(shape=np.asarray(shape, np.int32), material=np.packbits(x[0].ravel() > 0.5), inside=np.packbits(ins.ravel()),
                index=idx, values=val, saturation=np.float64(saturation), log_scale=np.float64(log_scale),
                version=np.asarray(CACHE_VERSION))


def _outside_rule(ins: np.ndarray, saturation: float, log_scale: float) -> np.ndarray:
    """(4, ...) float32 the field channels outside the band: n = 0, d_log = +-saturated_log (negative inside), quantised."""
    rule = np.zeros((4,) + ins.shape, np.float32)
    rule[3] = np.where(ins, -1.0, 1.0) * saturated_log(saturation, log_scale)
    return _quantise(rule)


def voxfield_from_sparse(data) -> np.ndarray:
    """Inverse of `voxfield_to_sparse`: the dense (6, X, Y, Z) float32 tensor, bit-identical to the one that was stored."""
    shape = tuple(int(v) for v in data["shape"])
    size = int(np.prod(shape))
    ins = np.unpackbits(data["inside"])[:size].reshape(shape).astype(bool)
    out = np.empty((CHANNELS,) + shape, np.float32)
    out[0] = np.unpackbits(data["material"])[:size].reshape(shape)
    out[1:5] = _outside_rule(ins, float(data["saturation"]), float(data["log_scale"]))
    flat = out[1:5].reshape(4, -1)
    flat[:, np.asarray(data["index"], np.int64)] = np.asarray(data["values"], np.float32).T
    out[5] = ins
    return out


def save_voxfield_sparse(path, x: np.ndarray, saturation: float = SATURATION_VOXELS,
                         log_scale: float = DEFAULT_LOG_SCALE) -> None:
    """Write one tensor as a compressed .npz (atomic: a temporary file, then a rename)."""
    import os
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **voxfield_to_sparse(x, saturation, log_scale))
    os.replace(tmp, path)


def load_voxfield_sparse(path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return voxfield_from_sparse({k: z[k] for k in z.files})


# ---------------------------------------------------------------------------------------------------- one sample

def _sample_files(voxel_path, resolution: int):
    """(material (R,R,R) bool, inside bool, field npz path or None, the files the result depends on), .bin order. Inside
    bits: <stem>.inside.npy, else <stem>.clean.bin, else the grid (training/voxel_dataset.py's order)."""
    import sys
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dataset_gen.src.inside_bits import inside_path, read_inside
    from training.field_input import field_npz_path
    vp = Path(voxel_path)
    material = np.fromfile(vp, dtype=np.uint8).reshape((resolution,) * 3) > 0
    deps = [vp]
    inside = read_inside(vp, resolution)
    if inside is not None:
        deps.append(inside_path(vp))
    else:
        clean = vp.with_name(vp.stem + ".clean.bin")
        if clean.exists():
            inside = np.fromfile(clean, dtype=np.uint8).reshape((resolution,) * 3) > 0
            deps.append(clean)
        else:
            inside = material
    npz = field_npz_path(vp)
    if npz.is_file():
        deps.append(npz)
    return material, inside, (npz if npz.is_file() else None), deps


def _resolution_of(voxel_path) -> int:
    r = round(Path(voxel_path).stat().st_size ** (1.0 / 3.0))
    if r ** 3 != Path(voxel_path).stat().st_size:
        raise ValueError(f"{voxel_path}: {Path(voxel_path).stat().st_size} bytes is not a cube grid")
    return int(r)


def voxfield_cache_path(voxel_path, resolution: Optional[int] = None, saturation: float = SATURATION_VOXELS,
                        log_scale: float = DEFAULT_LOG_SCALE) -> Path:
    """$SHAPE_RECO_CACHE or ~/.cache/shape_reco, subdir voxfield/, keyed (dataset_gen/src/cache_keys.py) by every file the
    tensor is made of (path, size, mtime of the .bin, the inside bits and the .field.npz), the parameters and the code of
    the encoder - this module, field_input, field_channels and inside_bits, their constants included (CACHE_VERSION is
    one of them)."""
    from dataset_gen.src.cache_keys import cache_key, cache_root
    r = resolution or _resolution_of(voxel_path)
    _, _, _, deps = _sample_files(voxel_path, r)
    key = cache_key("voxfield", files=deps, code=VOXFIELD_CODE,
                    params=dict(resolution=int(r), saturation=float(saturation), log_scale=float(log_scale)), length=24)
    return cache_root("voxfield") / f"{key}.npz"


def encode_sample(voxel_path, resolution: Optional[int] = None, saturation: float = SATURATION_VOXELS,
                  log_scale: float = DEFAULT_LOG_SCALE) -> Tuple[np.ndarray, str]:
    """(tensor, source) of one sample, uncached: source "field" (its .field.npz, the exact mesh field on the in-scale grid,
    moved into the .bin array frame) or "voxels" (the documented fallback from the inside bits)."""
    import sys
    r = resolution or _resolution_of(voxel_path)
    material, inside, npz, _ = _sample_files(voxel_path, r)
    if npz is None:
        return voxfield_channels(material, inside, saturation=saturation, log_scale=log_scale), "voxels"
    root = str(Path(__file__).resolve().parent.parent / "dataset_gen")
    if root not in sys.path:
        sys.path.insert(0, root)
    from src.field_channels import load_field_npz
    channels, spec, _ = load_field_npz(npz)
    if spec.resolution != r:
        raise ValueError(f"{npz}: field resolution {spec.resolution} != grid resolution {r}")
    x = voxfield_channels(material, inside, world_to_array(channels[1]), vector_world_to_array(channels[2:5]), spec.pitch,
                          log_scale=log_scale, saturation=saturation)
    return x, "field"


def voxfield_source(voxel_path) -> str:
    """"field" when the sample has a .field.npz (the exact path), else "voxels" (the fallback) - without encoding it."""
    from training.field_input import field_npz_path
    return "field" if field_npz_path(voxel_path).is_file() else "voxels"


def load_voxfield(voxel_path, resolution: Optional[int] = None, cache: bool = True, saturation: float = SATURATION_VOXELS,
                  log_scale: float = DEFAULT_LOG_SCALE) -> np.ndarray:
    """The "voxfield" tensor of one sample (`encode_sample`), computed once and kept as a sparse file
    (`voxfield_cache_path`); every later epoch/run reads it back bit-identically."""
    cpath = voxfield_cache_path(voxel_path, resolution, saturation, log_scale) if cache else None
    if cpath is not None and cpath.is_file():
        try:
            return load_voxfield_sparse(cpath)
        except (OSError, ValueError, KeyError):
            pass                                                         # a partly written file: recompute
    x, _ = encode_sample(voxel_path, resolution, saturation, log_scale)
    if cpath is not None:
        save_voxfield_sparse(cpath, x, saturation, log_scale)
    return x
