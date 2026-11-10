#!/usr/bin/env python3
"""
The input tensors of one run - parity occupancy and the closest-point field channels - computed ONCE and kept sparse.

Why: the add-cut loop (reco_engine/residual_loop.py) holds T_in = (5, X, Y, Z) float32 [occupancy, sdf, vx, vy, vz]
(dataset_gen/src/field_channels.py) densely: 20 bytes per cell, 414 MB on the frame's wall-thickness grid
(88 x 562 x 419), before any copy of it. Far from the surface those numbers are not read by the fits, and the occupancy
is constant there. So:

  SparseGrid   a (C, X, Y, Z) field of which only a BAND of cells around the surface is stored: sorted int32 linear
               indices (C order, x major) + an (n, C - 1) value table, looked up with np.searchsorted; the occupancy
               channel is NOT in the table but a bit per cell of the whole grid (np.packbits along z), so it is exact
               everywhere. Outside the band a float channel reads as a constant per side (`fill`, one row for empty cells
               and one for material). `dense(window=...)` builds only that index window - nothing of the size of the grid
               is allocated. No scipy.sparse: its matrices are 2-D, a 4-D tensor would have to be flattened to
               (cells, channels) CSR, which stores the same indices plus a row pointer per CELL (83 MB on the frame grid)
               and gives no window access; the sorted-index table is smaller and does.
  RunTensors   the once-per-run holder: parity first (cheap), the band from the occupancy boundary by binary dilation,
               then the closest-point query ONLY at the band's voxel centres. Built lazily on first access, under a lock
               (one worker thread + the GUI thread reading); every later request is a lookup (`builds` stays 1).
  piece_channels   `field_channels.voxel_channels` of a leftover piece on a window around it instead of the whole grid.

    rt = RunTensors(field, spec)                      # nothing computed yet; spec: GridSpec (cube) or local_grids.BoxGrid
    occ = rt.occupancy()                              # (X, Y, Z) bool, channel 0 of mesh_channels, exact everywhere
    win = window_of(piece_mask, margin=4)             # ((x0, x1), (y0, y1), (z0, z1)) around a piece
    t = rt.channels(window=win)                       # (5, wx, wy, wz) float32; exact in the band, `fill` outside it
    t = rt.channels(window=win, complete=True)        # the cells outside the band queried now (not stored): exact everywhere
    rt.grid.save(path); SparseGrid.load(path)         # one .npz

Grids: `field_channels.GridSpec` supports cubes only (lo, hi, resolution), so every function here also takes a
`local_grids.BoxGrid` (lo, shape, pitch; its `crop` is ignored) or any object with `.centres()` -> (X, Y, Z, 3) and
`.pitch`. `dense_channels` is `mesh_channels` for such a box (bit-identical to it on a GridSpec, tested).

Band: `band_voxels` dilation steps (26-neighbourhood) of the occupancy and of its complement; a cell outside the band has
only cells of its own side within that Chebyshev distance. DEFAULT_BAND_VOXELS = 9 = the networks' SDF truncation
(field_channels.DEFAULT_TRUNCATION = 8 voxels) + 1 voxel for the offset between the occupancy boundary and the surface:
measured below, every cell with |sdf| <= 8 voxels was inside the band on the three parts, so `network_input`'s clipped SDF
channel is unchanged by the default fill (sdf = +-band_voxels * pitch, V = 0). What DOES change outside the band is V:
`network_input` keeps V's direction at any distance and `dirlog_input` its unit normal and an untruncated log distance -
a caller that feeds a network the whole grid must use `complete=True` or a window inside the band.

Disk cache (optional, off unless asked for, like reco_engine/cache.py): `RunTensors(..., cache=<dir>)`, else
$SHAPE_RECO_TENSOR_CACHE = <dir> | off, else $SHAPE_RECO_CACHE/run_tensors. Keyed by the mesh bytes, the grid's axes, the
band, the query mode and backend, the value dtype and RUN_TENSORS_VERSION.

Measured 2026-09-27 (scripts/measure_run_tensors.py -> docs/data/run_tensors.json; every build in its own subprocess,
peak RSS = resource.getrusage ru_maxrss of that process, "base" = imports + the loaded mesh; other sessions' jobs shared
the laptop, the cuda rows were not repeated). MB = 2^20 bytes, in memory, float32; seconds = parity + band + query:

  part           grid        backend band  dense MB sparse MB cells   dense s  band s  max|diff|  RSS dense  band  base
  box_with_bore  96x96x96    igl     9      16.9      9.9     58 %     0.29    0.24    0            236    184   105
  box_with_bore  96x96x96    igl     3      16.9      3.8     21 %     0.29    0.12    0            236    138   105
  box_with_bore  96x96x96    cuda    9      16.9      9.9     58 %    54.87   31.80    0           1104   1053   491
  box_with_bore  96x96x96    cuda    3      16.9      3.8     21 %    54.87   11.92    0           1104   1023   491
  cy_t           96x96x96    igl     9      16.9      9.7     56 %     0.32    0.21    0            238    184   107
  cy_t           96x96x96    igl     3      16.9      3.4     19 %     0.32    0.11    0            238    135   107
  cy_t           96x96x96    cuda    9      16.9      9.7     56 %   102.44   58.20    0           1153   1109   490
  cy_t           96x96x96    cuda    3      16.9      3.4     19 %   102.44   19.86    0           1153   1094   490
  liner 9.6 mm   44x24x170   igl     9       3.4      3.4     99 %     0.41    0.45    0            147    154   138
  liner 9.6 mm   44x24x170   igl     3       3.4      2.5     71 %     0.41    0.31    0            147    144   138
  liner 9.6 mm   44x24x170   cuda    9       3.4      3.4     99 %    80.76   80.71    0           1433   1436   504
  liner 9.6 mm   44x24x170   cuda    3       3.4      2.5     71 %    80.76   58.66    0           1433   1430   504
  liner 2.4 mm   164x84x668  igl     9     175.5    117.9     66 %    14.12    9.07    0           1513    481   138
  liner 2.4 mm   164x84x668  igl     3     175.5     50.9     28 %    14.12    4.32    0           1513    317   138

  - max|diff| is over the band's cells, all five channels, against `dense_channels`; the occupancy channel was equal in
    EVERY cell of the grid in all 14 rows.
  - Band 9 left no cell with |sdf| <= 8 voxels outside the band (0 in all rows; the nearest left-out cell was 9.02 to
    11.6 voxels from the surface); band 3 leaves them out by construction (nearest 3.02 to 3.12 voxels).
  - The saving is the band's share of the cells, so it depends on the grid: the liner is a thin shell, at 9.6 mm the
    grid is 24 cells across and a 9-voxel band IS the grid (99 %, no saving, the index costs 4 bytes per cell on top);
    at 2.4 mm 66 %. The larger effect is the build's peak: 1513 -> 481 MB (the dense query holds ~100 bytes per point
    for all points at once, the band build QUERY_BLOCK points at a time).
  - One 32^3 window out of the built grid: 0.3 to 2.2 ms. On disk (.npz, compressed) band 9: 1.1 to 2.3 MB float32,
    0.4 to 1.2 MB float16 for the three small grids; float16 rounds by up to 0.002 mm (96^3 parts) and 0.031 mm (liner
    9.6 mm).
  - NOT measured: the frame (no frame runs in this session), the peel route, the loop itself - nothing calls this
    module yet.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

CHANNELS = ("occupancy", "sdf", "vx", "vy", "vz")       # field_channels.CHANNELS
DEFAULT_BAND_VOXELS = 9          # dilation steps; field_channels.DEFAULT_TRUNCATION (8) + 1, see the module docstring
QUERY_BLOCK = 262144             # band centres per closest-point query (bounds the Query's ~100 bytes per point)
PARITY_SLAB = 64                 # grid rows (y) per parity pass, as local_grids.iter_slabs
RUN_TENSORS_VERSION = "1"        # bump when what a build stores changes for the same inputs

Window = Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]


# ------------------------------------------------------------------------------------------------ grids

def grid_axes(spec) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(xs, ys, zs, pitch): the voxel-centre coordinates of `spec` per world axis, WITHOUT building the (X, Y, Z, 3)
    centre array (24 bytes per cell). GridSpec: the arithmetic of `GridSpec.centres` (linspace edges + half a step), so
    the values agree with it bit for bit (tested); BoxGrid: its `axis(d)`; anything else: read off `spec.centres()`."""
    if hasattr(spec, "axis") and hasattr(spec, "shape"):                          # local_grids.BoxGrid
        return spec.axis(0), spec.axis(1), spec.axis(2), float(spec.pitch)
    if all(hasattr(spec, a) for a in ("lo", "hi", "resolution")):                 # field_channels.GridSpec
        lo, hi, r = np.asarray(spec.lo, float), np.asarray(spec.hi, float), int(spec.resolution)
        axes = []
        for d in range(3):
            edges = np.linspace(lo[d], hi[d], r + 1)
            axes.append(edges[:-1] + (edges[1] - edges[0]) / 2.0)
        return axes[0], axes[1], axes[2], float(spec.pitch)
    c = np.asarray(spec.centres(), float)
    pitch = float(getattr(spec, "pitch", c[1, 0, 0, 0] - c[0, 0, 0, 0]))
    return c[:, 0, 0, 0].copy(), c[0, :, 0, 1].copy(), c[0, 0, :, 2].copy(), pitch


def grid_shape(spec) -> Tuple[int, int, int]:
    xs, ys, zs, _ = grid_axes(spec)
    return len(xs), len(ys), len(zs)


def _grid_lo(spec, axes) -> np.ndarray:
    if hasattr(spec, "lo"):
        return np.asarray(spec.lo, float)
    return np.array([a[0] for a in axes[:3]]) - axes[3] / 2.0


def full_window(shape) -> Window:
    return tuple((0, int(n)) for n in shape)


def _window(window, shape) -> Window:
    """`window` normalised to ((x0, x1), (y0, y1), (z0, z1)), half-open index ranges clipped to the grid; None = all."""
    if window is None:
        return full_window(shape)
    out = []
    for d in range(3):
        w = window[d]
        a, b = (0, shape[d]) if w is None else (w.start, w.stop) if isinstance(w, slice) else (w[0], w[1])
        a = 0 if a is None else int(a)
        b = shape[d] if b is None else int(b)
        a, b = max(0, min(a, shape[d])), max(0, min(b, shape[d]))
        out.append((a, max(a, b)))
    return tuple(out)


def window_of(mask: np.ndarray, margin: int = 0) -> Optional[Window]:
    """The index window around the True cells of `mask` (X, Y, Z), widened by `margin` cells and clipped to the grid;
    None if `mask` is empty."""
    mask = np.asarray(mask, bool)
    out = []
    for d in range(3):
        hit = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != d)))
        if not len(hit):
            return None
        out.append((max(0, int(hit[0]) - margin), min(mask.shape[d], int(hit[-1]) + 1 + margin)))
    return tuple(out)


def window_slices(window: Window) -> Tuple[slice, slice, slice]:
    return tuple(slice(a, b) for a, b in window)


def window_centres(spec, window=None) -> np.ndarray:
    """(wx, wy, wz, 3) world centres of the cells of `window` - `spec.centres()[window]` without the full array."""
    xs, ys, zs, _ = grid_axes(spec)
    (x0, x1), (y0, y1), (z0, z1) = _window(window, (len(xs), len(ys), len(zs)))
    return np.stack(np.meshgrid(xs[x0:x1], ys[y0:y1], zs[z0:z1], indexing="ij"), axis=-1)


# ------------------------------------------------------------------------------------------------ the sparse container

class SparseGrid:
    """A (C, X, Y, Z) float32 field stored in a band of cells only (module docstring).

    index       (n,) int32, sorted: linear index (x * Y + y) * Z + z of every stored cell
    values      (n, F) float32 or float16: the FLOAT channels of those cells, in channel order without the occupancy one
    occupancy_bits   (X, Y, ceil(Z / 8)) uint8, np.packbits along z: the occupancy channel of EVERY cell; None if the
                field has no occupancy channel (`occupancy_channel` None; then all C channels are in `values`)
    fill        (2, C) float32: what a float channel reads outside the band, row 0 in empty cells, row 1 in material
                (without occupancy bits only row 0 is used). The entry of the occupancy channel is ignored."""

    def __init__(self, shape, index, values, occupancy_bits=None, channels: Sequence[str] = CHANNELS,
                 occupancy_channel: Optional[int] = 0, fill=None):
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3:
            raise ValueError(f"SparseGrid: shape must be (X, Y, Z), got {shape}")
        if int(np.prod(self.shape, dtype=np.int64)) >= 2 ** 31:
            raise ValueError(f"SparseGrid: {self.shape} has more cells than an int32 index can address")
        self.channels = tuple(str(c) for c in channels)
        self.occupancy_channel = None if occupancy_channel is None else int(occupancy_channel)
        index = np.asarray(index)
        values = np.asarray(values)
        if values.dtype not in (np.float32, np.float16):
            values = values.astype(np.float32)
        if values.ndim != 2 or values.shape != (len(index), len(self.float_channels)):
            raise ValueError(f"SparseGrid: values {values.shape} do not match {len(index)} cells x "
                             f"{len(self.float_channels)} float channels")
        if len(index) and (index.min() < 0 or index.max() >= np.prod(self.shape, dtype=np.int64)):
            raise ValueError("SparseGrid: a band index lies outside the grid")
        if len(index) > 1 and not np.all(np.diff(index.astype(np.int64)) > 0):                # sort once, drop nothing
            order = np.argsort(index, kind="stable")
            index, values = index[order], values[order]
            if not np.all(np.diff(index.astype(np.int64)) > 0):
                raise ValueError("SparseGrid: a cell is listed twice in the band")
        self.index = np.ascontiguousarray(index, dtype=np.int32)
        self.values = np.ascontiguousarray(values)
        if (occupancy_bits is None) != (self.occupancy_channel is None):
            raise ValueError("SparseGrid: occupancy bits and occupancy_channel go together")
        if occupancy_bits is not None:
            occupancy_bits = np.ascontiguousarray(occupancy_bits, dtype=np.uint8)
            want = (self.shape[0], self.shape[1], (self.shape[2] + 7) // 8)
            if occupancy_bits.shape != want:
                raise ValueError(f"SparseGrid: occupancy bits {occupancy_bits.shape}, expected {want}")
        self.occupancy_bits = occupancy_bits
        f = np.zeros((2, len(self.channels)), np.float32)
        if fill is not None:
            f[:] = np.asarray(fill, np.float32)                     # (C,) for both sides or (2, C)
        self.fill = f

    # ------------------------------------------------------------------------------------------- construction
    @property
    def float_channels(self) -> Tuple[int, ...]:
        return tuple(c for c in range(len(self.channels)) if c != self.occupancy_channel)

    @staticmethod
    def pack_occupancy(occ: np.ndarray) -> np.ndarray:
        return np.packbits(np.asarray(occ, bool), axis=2)

    @classmethod
    def from_dense(cls, dense: np.ndarray, band: np.ndarray, channels: Sequence[str] = CHANNELS,
                   occupancy_channel: Optional[int] = 0, fill=None, dtype=np.float32) -> "SparseGrid":
        """From a dense (C, X, Y, Z) array and the bool (X, Y, Z) `band` of cells to keep. The occupancy channel
        (`dense[occupancy_channel] > 0.5`) is kept for every cell, as bits."""
        dense = np.asarray(dense)
        band = np.asarray(band, bool)
        if dense.ndim != 4 or dense.shape[1:] != band.shape:
            raise ValueError(f"SparseGrid.from_dense: dense {dense.shape} and band {band.shape} do not match")
        index = np.flatnonzero(band)
        floats = [c for c in range(dense.shape[0]) if c != occupancy_channel]
        values = np.empty((len(index), len(floats)), dtype)
        for col, c in enumerate(floats):
            values[:, col] = dense[c].reshape(-1)[index]
        bits = None if occupancy_channel is None else cls.pack_occupancy(dense[occupancy_channel] > 0.5)
        return cls(band.shape, index, values, bits, channels, occupancy_channel, fill)

    # ------------------------------------------------------------------------------------------- reading
    @property
    def n_band(self) -> int:
        return int(len(self.index))

    @property
    def nbytes(self) -> int:
        return int(self.index.nbytes + self.values.nbytes + self.fill.nbytes
                   + (0 if self.occupancy_bits is None else self.occupancy_bits.nbytes))

    @property
    def dense_nbytes(self) -> int:
        """What the same field costs as a dense float32 array."""
        return int(4 * len(self.channels) * np.prod(self.shape, dtype=np.int64))

    def _channel_ids(self, channels) -> Tuple[int, ...]:
        if channels is None:
            return tuple(range(len(self.channels)))
        if isinstance(channels, (str, int, np.integer)):
            channels = [channels]
        return tuple(self.channels.index(c) if isinstance(c, str) else int(c) for c in channels)

    def _band_in(self, window: Window) -> Tuple[np.ndarray, np.ndarray]:
        """(rows of `index`/`values`, flat position in the window's own C order) of the band cells inside `window`.
        One np.searchsorted pair per (x, y) row of the window: the cells of a row are consecutive in the sorted index."""
        X, Y, Z = self.shape
        (x0, x1), (y0, y1), (z0, z1) = window
        wx, wy, wz = x1 - x0, y1 - y0, z1 - z0
        if wx * wy * wz == 0 or not len(self.index):
            return np.empty(0, np.int64), np.empty(0, np.int64)
        if (wx, wy, wz) == (X, Y, Z):
            return np.arange(len(self.index), dtype=np.int64), self.index.astype(np.int64)
        base = ((np.arange(x0, x1, dtype=np.int64)[:, None] * Y + np.arange(y0, y1, dtype=np.int64)[None, :]) * Z).reshape(-1)
        start = np.searchsorted(self.index, base + z0, side="left")
        count = np.searchsorted(self.index, base + z1, side="left") - start
        total = int(count.sum())
        if not total:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        first = np.cumsum(count) - count                                     # where each row's cells begin in the output
        rows = np.repeat(start - first, count) + np.arange(total, dtype=np.int64)
        z = self.index[rows].astype(np.int64) - np.repeat(base, count)
        local = np.repeat(np.arange(wx * wy, dtype=np.int64), count) * wz + (z - z0)
        return rows, local

    def occupancy(self, window=None) -> np.ndarray:
        """(wx, wy, wz) bool: the occupancy channel in `window` (None: the whole grid), exact in every cell."""
        if self.occupancy_bits is None:
            raise ValueError("SparseGrid: this field has no occupancy channel")
        (x0, x1), (y0, y1), (z0, z1) = _window(window, self.shape)
        if z1 == z0:
            return np.zeros((x1 - x0, y1 - y0, 0), bool)
        b0 = z0 // 8
        bits = np.unpackbits(self.occupancy_bits[x0:x1, y0:y1, b0:(z1 + 7) // 8], axis=2)
        return bits[:, :, z0 - 8 * b0:z1 - 8 * b0].astype(bool)

    def band_mask(self, window=None) -> np.ndarray:
        """(wx, wy, wz) bool: which cells of `window` are stored."""
        w = _window(window, self.shape)
        out = np.zeros(tuple(b - a for a, b in w), bool)
        out.reshape(-1)[self._band_in(w)[1]] = True
        return out

    def dense(self, channels=None, window=None) -> np.ndarray:
        """(len(channels), wx, wy, wz) float32 of `window` = ((x0, x1), (y0, y1), (z0, z1)) (None: the whole grid):
        the stored values in the band, `fill` outside it, the occupancy channel from its bits. `channels`: indices or
        names, None = all. Allocates the window (and index arrays of its band cells), never the grid."""
        ids = self._channel_ids(channels)
        w = _window(window, self.shape)
        size = tuple(b - a for a, b in w)
        out = np.empty((len(ids),) + size, np.float32)
        occ = self.occupancy(w) if self.occupancy_bits is not None else None
        rows = local = None
        for k, c in enumerate(ids):
            if c == self.occupancy_channel:
                out[k] = occ
                continue
            if occ is None or self.fill[0, c] == self.fill[1, c]:
                out[k] = self.fill[0, c]
            else:
                out[k] = np.where(occ, self.fill[1, c], self.fill[0, c])
            if rows is None:
                rows, local = self._band_in(w)
            out[k].reshape(-1)[local] = self.values[rows, self.float_channels.index(c)]
        return out

    # ------------------------------------------------------------------------------------------- storage
    def save(self, path, dtype="float32", **extra) -> None:
        """One compressed .npz at exactly `path`. `dtype`: "float32" (exact round trip) or "float16" (half the value
        table; what `field_channels.save_field_npz` stores). `extra`: further arrays, returned by `load(..., extra=True)`."""
        dtype = np.dtype(dtype)
        if dtype not in (np.float32, np.float16):
            raise ValueError(f"SparseGrid.save: dtype must be float32 or float16, got {dtype}")
        arrays = dict(version=np.array(RUN_TENSORS_VERSION), shape=np.asarray(self.shape, np.int64), index=self.index,
                      values=self.values.astype(dtype, copy=False), channels=np.array(list(self.channels)),
                      occupancy_channel=np.array(-1 if self.occupancy_channel is None else self.occupancy_channel),
                      fill=self.fill)
        if self.occupancy_bits is not None:
            arrays["occupancy_bits"] = self.occupancy_bits
        for k, v in extra.items():
            arrays["extra_" + k] = np.asarray(v)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        from reco_engine.cache import _atomic

        def write(tmp):
            with open(tmp, "wb") as f:              # a path would make numpy append ".npz" and write elsewhere
                np.savez_compressed(f, **arrays)
        _atomic(path, write)

    @classmethod
    def load(cls, path, extra: bool = False):
        """The SparseGrid of `save` (values in the dtype they were stored in); with `extra` also the dict of extra arrays."""
        with np.load(path, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        oc = int(data["occupancy_channel"])
        grid = cls(tuple(int(v) for v in data["shape"]), data["index"], data["values"], data.get("occupancy_bits"),
                   [str(c) for c in data["channels"]], None if oc < 0 else oc, data["fill"])
        if extra:
            return grid, {k[len("extra_"):]: v for k, v in data.items() if k.startswith("extra_")}
        return grid


# ------------------------------------------------------------------------------------------------ building blocks

@contextmanager
def _field_backend(name: Optional[str]):
    """$SHAPE_RECO_FIELD_BACKEND = `name` for the duration (None: unchanged), as residual_loop._field_backend."""
    if not name:
        yield
        return
    old = os.environ.get("SHAPE_RECO_FIELD_BACKEND")
    os.environ["SHAPE_RECO_FIELD_BACKEND"] = name
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("SHAPE_RECO_FIELD_BACKEND", None)
        else:
            os.environ["SHAPE_RECO_FIELD_BACKEND"] = old


def parity_grid(field, spec) -> np.ndarray:
    """(X, Y, Z) bool: `field_channels.parity_occupancy` on any box grid, in slabs of PARITY_SLAB rows along y so the hit
    counters never have the size of the grid (rows are independent: the same bits as one pass, tested)."""
    from src.field_channels import parity_axes
    xs, ys, zs, pitch = grid_axes(spec)
    ys = ys + 1.2345e-4 * pitch                                  # parity_occupancy's shifts off the round coordinates
    zs = zs + 2.3456e-4 * pitch
    V, F = np.asarray(field.vertices, float), np.asarray(field.faces)
    ty = V[F][:, :, 1]
    tmin, tmax = ty.min(axis=1), ty.max(axis=1)
    out = np.zeros((len(xs), len(ys), len(zs)), bool)
    for j0 in range(0, len(ys), PARITY_SLAB):
        yy = ys[j0:j0 + PARITY_SLAB]
        sel = (tmax >= yy[0]) & (tmin <= yy[-1])
        if sel.any():
            out[:, j0:j0 + len(yy)] = parity_axes(V, F[sel], xs, yy, zs)
    return out


def surface_band(occ: np.ndarray, band_voxels: int) -> np.ndarray:
    """(X, Y, Z) bool: the cells within `band_voxels` dilation steps (26-neighbourhood, i.e. Chebyshev distance) of the
    occupancy boundary, on both sides: dilate(occ) and not erode(occ). The grid's edge counts as empty, so material
    touching it is in the band. Bool arrays only (a distance transform would take 8 bytes per cell)."""
    from scipy import ndimage
    occ = np.asarray(occ, bool)
    n = int(band_voxels)
    if n <= 0 or not occ.any():
        return np.zeros(occ.shape, bool)
    box = np.ones((3, 3, 3), bool)
    grown = ndimage.binary_dilation(occ, structure=box, iterations=n)
    grown &= ~ndimage.binary_erosion(occ, structure=box, iterations=n)
    return grown


def _source(field, pitch: float):
    """The field the distances are read from: `field` itself under the igl backend, else its refined copy - the choice
    `field_channels.mesh_channels` makes."""
    from src.closest_point_field import field_backend
    from src.field_channels import REFINE_EDGE_VOXELS, refined_field
    return field if field_backend() == "igl" else refined_field(field, REFINE_EDGE_VOXELS * pitch)


def _query_values(src, points: np.ndarray, inside: np.ndarray, exact: bool) -> np.ndarray:
    """(n, 4) float32 [sdf, vx, vy, vz] of `points`, the sign from `inside`; in blocks of QUERY_BLOCK."""
    out = np.empty((len(points), 4), np.float32)
    for lo in range(0, len(points), QUERY_BLOCK):
        q = src.query(points[lo:lo + QUERY_BLOCK], exact=exact)
        out[lo:lo + QUERY_BLOCK, 0] = np.where(inside[lo:lo + QUERY_BLOCK], q.distance, -q.distance)
        out[lo:lo + QUERY_BLOCK, 1:] = q.vector
    return out


def dense_channels(field, spec, exact: bool = False, backend: Optional[str] = None,
                   inside: Optional[np.ndarray] = None) -> np.ndarray:
    """(5, X, Y, Z) float32: `field_channels.mesh_channels` (sign "parity") on any box grid - the dense reference the
    sparse build is measured against. One query over all centres, as mesh_channels does."""
    xs, ys, zs, pitch = grid_axes(spec)
    shape = (len(xs), len(ys), len(zs))
    inside = parity_grid(field, spec) if inside is None else np.asarray(inside, bool)
    pts = window_centres(spec).reshape(-1, 3)
    with _field_backend(backend):
        q = _source(field, pitch).query(pts, exact=exact)
    sdf = np.where(inside.reshape(-1), q.distance, -q.distance)
    out = np.empty((5,) + shape, np.float32)
    out[0] = (sdf > 0).reshape(shape)
    out[1] = sdf.reshape(shape)
    out[2:] = q.vector.reshape(shape + (3,)).transpose(3, 0, 1, 2)
    return out


def piece_channels(occ: np.ndarray, spec, window=None, margin: int = 4) -> Tuple[Window, np.ndarray]:
    """(window, (5, wx, wy, wz) float32): `field_channels.voxel_channels(occ, spec)` of a leftover piece, computed on an
    index window around it (`window`, default `window_of(occ, margin)`) instead of the whole grid. `occ`: (X, Y, Z) bool
    on the run's grid; every True cell must lie inside the window (the blocky surface is that of the piece alone, as in
    voxel_channels, so the numbers do not depend on the window). An empty `occ` gives an empty window and array."""
    from src.closest_point_field import SurfaceField, voxel_surface_mesh
    occ = np.asarray(occ, bool)
    axes = grid_axes(spec)
    shape = tuple(len(a) for a in axes[:3])
    if occ.shape != shape:
        raise ValueError(f"piece_channels: occupancy {occ.shape} does not match the grid {shape}")
    tight = window_of(occ, 0)
    if tight is None:
        return ((0, 0),) * 3, np.zeros((5, 0, 0, 0), np.float32)
    w = _window(window, shape) if window is not None else window_of(occ, margin)
    if any(t[0] < a or t[1] > b for t, (a, b) in zip(tight, w)):
        raise ValueError(f"piece_channels: the piece {tight} reaches outside the window {w}")
    part = occ[window_slices(w)]
    v, f = voxel_surface_mesh(part)
    v = v + np.array([a for a, _ in w], float)                             # window index -> grid index, exact in float
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    behind = np.floor(tri.mean(axis=1) - 0.5 * n).astype(int)              # field_channels.voxel_field: faces outward
    inb = np.all((behind >= 0) & (behind < np.array(shape)), axis=1)
    solid = np.zeros(len(f), bool)
    solid[inb] = occ[behind[inb, 0], behind[inb, 1], behind[inb, 2]]
    f = np.where(solid[:, None], f, f[:, ::-1])
    field = SurfaceField(v * axes[3] + _grid_lo(spec, axes), f)
    q = field.query(window_centres(spec, w).reshape(-1, 3), exact=False)
    size = part.shape
    out = np.empty((5,) + size, np.float32)
    out[0] = part
    out[1] = np.where(part.reshape(-1), q.distance, -q.distance).reshape(size)
    out[2:] = q.vector.reshape(size + (3,)).transpose(3, 0, 1, 2)
    return w, out


# ------------------------------------------------------------------------------------------------ disk cache

def tensor_cache_dir(cache=None) -> Optional[Path]:
    """Where built tensors are kept: `cache` (a directory; "off"/False = none), else $SHAPE_RECO_TENSOR_CACHE
    (<dir> | off), else $SHAPE_RECO_CACHE/run_tensors, else None - off unless asked for, as reco_engine/cache.py."""
    if cache is False:
        return None
    if cache is None:
        cache = os.environ.get("SHAPE_RECO_TENSOR_CACHE", "").strip()
        if not cache:
            root = os.environ.get("SHAPE_RECO_CACHE", "").strip()
            return Path(root) / "run_tensors" if root else None
    if str(cache).lower() in ("off", "0", "none", "false"):
        return None
    return Path(cache)


# ------------------------------------------------------------------------------------------------ once per run

class RunTensors:
    """Parity occupancy and field channels of ONE input on ONE grid, built once (lazily, under a lock) and kept as a
    SparseGrid (module docstring).

    field        a `closest_point_field.SurfaceField` (vertices, faces, query)
    spec         GridSpec, BoxGrid, or an object with `.centres()` and `.pitch`
    band_voxels  width of the stored band in dilation steps on either side of the occupancy boundary
    backend      SHAPE_RECO_FIELD_BACKEND for the queries (None: whatever the environment says)
    exact        `SurfaceField.query(exact=...)`, as ResidualConfig.exact_channels
    cache        directory of the disk cache, "off", or None = the environment (`tensor_cache_dir`)
    dtype        "float32" (default, exact) or "float16" for the value table in memory and on disk

    `builds` counts the builds (0 before the first access, 1 ever after), `seconds` their parts, `from_cache` whether
    the build was a disk hit."""

    def __init__(self, field, spec, band_voxels: int = DEFAULT_BAND_VOXELS, backend: Optional[str] = None,
                 exact: bool = False, cache=None, dtype="float32"):
        self.field = field
        self.spec = spec
        self.band_voxels = int(band_voxels)
        self.backend = backend
        self.exact = bool(exact)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.float32, np.float16):
            raise ValueError(f"RunTensors: dtype must be float32 or float16, got {dtype}")
        self.cache_dir = tensor_cache_dir(cache)
        self.builds = 0
        self.from_cache = False
        self.seconds: Dict[str, float] = {}
        self._axes = grid_axes(spec)
        self.shape = tuple(len(a) for a in self._axes[:3])
        self.pitch = float(self._axes[3])
        self._grid: Optional[SparseGrid] = None
        self._parity_bits: Optional[np.ndarray] = None
        self._src = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------------------- the build
    def _key(self) -> str:
        from src.closest_point_field import field_backend
        from src.field_channels import REFINE_EDGE_VOXELS, REFINE_K
        with _field_backend(self.backend):
            backend = field_backend()
        h = hashlib.sha1(("run_tensors:" + RUN_TENSORS_VERSION).encode())
        parts = [np.ascontiguousarray(self.field.vertices, dtype=np.float64).tobytes(),
                 np.ascontiguousarray(self.field.faces, dtype=np.int64).tobytes()]
        parts += [np.ascontiguousarray(a, dtype=np.float64).tobytes() for a in self._axes[:3]]
        parts.append(f"pitch={self.pitch!r};band={self.band_voxels};exact={self.exact};backend={backend};"
                     f"dtype={self.dtype.name};refine={REFINE_EDGE_VOXELS!r},{REFINE_K}".encode())
        for part in parts:
            h.update(len(part).to_bytes(8, "little"))
            h.update(part)
        return h.hexdigest()

    def _build(self) -> None:
        path = None
        if self.cache_dir is not None:
            path = self.cache_dir / f"{self._key()}.npz"
            try:
                grid, extra = SparseGrid.load(path, extra=True)
                if grid.shape == self.shape and "parity_bits" in extra:
                    self._parity_bits, self._grid, self.from_cache = extra["parity_bits"], grid, True
                    return
            except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
                pass
        t0 = time.perf_counter()
        inside = parity_grid(self.field, self.spec)
        t1 = time.perf_counter()
        index = np.flatnonzero(surface_band(inside, self.band_voxels))
        t2 = time.perf_counter()
        xs, ys, zs, _ = self._axes
        X, Y, Z = self.shape
        pts = np.stack([xs[index // (Y * Z)], ys[(index // Z) % Y], zs[index % Z]], axis=1)
        with _field_backend(self.backend):
            values = _query_values(self._source(), pts, inside.reshape(-1)[index], self.exact)
        del pts
        # channel 0 of mesh_channels is (sdf > 0): parity, except in a centre ON the surface (distance 0), which is in the band
        occ = inside.copy()
        occ.reshape(-1)[index[~(values[:, 0] > 0)]] = False
        far = self.band_voxels * self.pitch
        fill = np.array([[0.0, -far, 0.0, 0.0, 0.0], [1.0, far, 0.0, 0.0, 0.0]], np.float32)
        grid = SparseGrid(self.shape, index, values.astype(self.dtype, copy=False), SparseGrid.pack_occupancy(occ),
                          CHANNELS, 0, fill)
        self._parity_bits = SparseGrid.pack_occupancy(inside)
        self.seconds = dict(parity=t1 - t0, band=t2 - t1, field=time.perf_counter() - t2)
        if path is not None:
            grid.save(path, dtype=self.dtype, parity_bits=self._parity_bits)
        self._grid = grid

    def _source(self):
        if self._src is None:
            self._src = _source(self.field, self.pitch)
        return self._src

    @property
    def grid(self) -> SparseGrid:
        """The SparseGrid of the five channels; the first access builds it, every later one returns the same object."""
        if self._grid is None:
            with self._lock:
                if self._grid is None:
                    self._build()
                    self.builds += 1
        return self._grid

    # ------------------------------------------------------------------------------------------- reading
    @property
    def nbytes(self) -> int:
        return self.grid.nbytes + self._parity_bits.nbytes

    def occupancy(self, window=None) -> np.ndarray:
        """(wx, wy, wz) bool: channel 0 of `mesh_channels` (sdf > 0), exact in every cell of the grid."""
        return self.grid.occupancy(window)

    def parity(self, window=None) -> np.ndarray:
        """(wx, wy, wz) bool: `parity_occupancy(field, spec)` - equal to `occupancy` except in a voxel centre that lies
        exactly on the surface."""
        grid = self.grid
        return SparseGrid(grid.shape, np.empty(0, np.int32), np.empty((0, 0), np.float32), self._parity_bits,
                          ("occupancy",), 0).occupancy(window)

    def band_mask(self, window=None) -> np.ndarray:
        return self.grid.band_mask(window)

    def channels(self, window=None, channels=None, complete: bool = False) -> np.ndarray:
        """(len(channels), wx, wy, wz) float32 of `window` (None: the whole grid - dense, avoid on a large grid).
        In the band: what `mesh_channels` gives there. Outside it: occupancy exact, sdf = +-band_voxels * pitch by side,
        V = 0 - unless `complete`, which queries the field at the window's cells outside the band now (nothing is
        stored; the cost of a query per such cell) so the window equals the dense tensor's everywhere."""
        grid = self.grid
        out = grid.dense(channels, window)
        if not complete:
            return out
        ids = grid._channel_ids(channels)
        if not any(c != 0 for c in ids):
            return out
        w = _window(window, self.shape)
        miss = np.flatnonzero(~grid.band_mask(w).reshape(-1))
        if not len(miss):
            return out
        (x0, x1), (y0, y1), (z0, z1) = w
        wy, wz = y1 - y0, z1 - z0
        xs, ys, zs, _ = self._axes
        pts = np.stack([xs[x0 + miss // (wy * wz)], ys[y0 + (miss // wz) % wy], zs[z0 + miss % wz]], axis=1)
        inside = self.parity(w).reshape(-1)[miss]
        with self._lock, _field_backend(self.backend):
            values = _query_values(self._source(), pts, inside, self.exact)
        for k, c in enumerate(ids):
            if c != 0:
                out[k].reshape(-1)[miss] = values[:, c - 1]
        return out
