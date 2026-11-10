#!/usr/bin/env python3
"""
Thickness-adaptive parity grids: make "voxels are the ground truth" true where the grid is used.

Ray parity at voxel centres is point sampling of the solid's indicator. It is an unbiased volume estimate, but it keeps
the TOPOLOGY only when the pitch is well below the thinnest wall (an r-regular solid needs h below a constant times r,
r <= t/2 for a wall of thickness t). The frame's 12 mm webs are 0.11 voxel at the 64^3 cube over the 6.9 m part: that
grid is empty (docs/FRAME_FAILURE.md, "Thickness-adaptive local grids"). This module picks the pitch from the part's
own wall thickness and evaluates parity over any axis-aligned box at that pitch, in slabs, so a 4 mm grid over a 7 m
part needs no dense array. It returns the Betti numbers of the voxel solid, which is what "the grid sees the part"
means in a testable form.

    pts, n, tau = wall_thickness(vertices, faces)        # thickness along the inward normal at surface samples
    h = adaptive_pitch(tau)                              # tau_q / 3 (q-th percentile, default the 5th)
    topo = grid_topology(vertices, faces, lo, hi, h)     # b0, b1, b2, chi, voxel count and volume of the parity grid

The voxel solid is the union of CLOSED cubes at the occupied centres: objects are 26-connected, the complement
6-connected, b1 = b0 + b2 - chi. `crop` restricts the solid to the centres inside a box (mesh intersect box, sampled),
so a member crop is compared with the exact common of the B-rep and the same box (scripts/local_grid_frame.py).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator, Optional, Tuple

import numpy as np
from scipy import ndimage

from .field_channels import parity_axes

ROW_SHIFT = (1.2345e-4, 2.3456e-4)   # fractions of a pitch, as parity_occupancy: rows never run through a mesh vertex


def wall_thickness(vertices: np.ndarray, faces: np.ndarray, n: int = 4000, seed: int = 0):
    """(points (n, 3), outward normals (n, 3), thickness (n,)): area-weighted surface samples and the distance along the
    inward normal to the first surface hit, the shape-diameter of a wall (a plate of thickness t gives t exactly; a ray
    that escapes gives inf). For a closed, outward-oriented mesh."""
    import trimesh
    mesh = trimesh.Trimesh(np.asarray(vertices, float), np.asarray(faces), process=False)
    pts, idx = trimesh.sample.sample_surface(mesh, n, seed=seed)
    nrm = mesh.face_normals[idx]
    eps = 1e-6 * float(np.linalg.norm(mesh.extents))
    origins = pts - eps * nrm                     # start just inside, so the ray does not hit its own face
    try:
        import igl
    except ImportError:
        igl = None
    tau = np.full(len(pts), np.inf)
    if igl is not None:
        # libigl's AABB tree, first hit per ray: 13-300x faster than trimesh's numpy rays (bs_hollow_sphere, 53k faces:
        # 38.7 s -> 0.19 s; the frame 0.29 -> 0.017 s), measured 2026-09-24. T is the distance from the origin, which lies
        # eps below the surface point: the wall is T + eps.
        V = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
        F = np.ascontiguousarray(mesh.faces, dtype=np.int64)
        tree = igl.AABB()
        tree.init(V, F)
        hit, t, _ = tree.intersect_ray_first(V, F, np.ascontiguousarray(origins, dtype=np.float64),
                                             np.ascontiguousarray(-nrm, dtype=np.float64))
        ok = np.asarray(hit) >= 0
        tau[ok] = np.asarray(t)[ok] + eps
        return pts, nrm, tau
    loc, ray, _ = mesh.ray.intersects_location(origins, -nrm, multiple_hits=False)
    tau[ray] = np.linalg.norm(loc - origins[ray], axis=1) + eps      # from the surface point: t + eps (not t + 2 eps)
    return pts, nrm, tau


def adaptive_pitch(tau: np.ndarray, fraction: float = 1.0 / 3.0, q: float = 5.0) -> float:
    """The pitch that resolves the thin walls: `fraction` of the q-th percentile of the finite thicknesses.

    The default fraction 1/3 is D-005's rule (pitch <= tau/3). The loop's grid "tau" (reco_engine.residual_loop._tau_spec)
    calls this with `ResidualConfig.tau_fraction`, default 1.0, i.e. the pitch tau_5 itself (state of 2026-10-01; an open
    decision, review R-03 5.1). Measured on the frame (docs/data/local_grid_frame.json, tau_5 = 12.0 mm): the parity
    grid has the exact Betti numbers (1, 17, 0) and 0.98-1.03 of the volume at every pitch tau/1 ... tau/4, and wrong
    ones on the 64^3-256^3 cubes over the part (105-26 mm)."""
    t = np.asarray(tau, float)
    t = t[np.isfinite(t) & (t > 0)]
    if not len(t):
        raise ValueError("adaptive_pitch: no finite wall thickness")
    return float(fraction * np.percentile(t, q))


# ------------------------------------------------------------------------------------------------ the grid, in slabs

@dataclass(frozen=True)
class BoxGrid:
    """Voxel centres lo + (index + 0.5) * pitch over a box (any aspect ratio, one pitch); `crop` masks the solid to the
    centres inside [crop_lo, crop_hi]."""
    lo: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    pitch: float
    crop_lo: Optional[Tuple[float, float, float]] = None
    crop_hi: Optional[Tuple[float, float, float]] = None

    def axis(self, d: int) -> np.ndarray:
        return self.lo[d] + (np.arange(self.shape[d]) + 0.5) * self.pitch


def box_grid(lo, hi, pitch: float, crop: bool = False, margin: int = 2) -> BoxGrid:
    """A grid over [lo, hi] plus `margin` empty voxels on every side (so the outside is one connected region). With
    `crop` the solid is restricted to [lo, hi]."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / pitch)) + 2 * margin for d in range(3))
    start = tuple(float(lo[d] - margin * pitch) for d in range(3))
    return BoxGrid(start, shape, float(pitch), tuple(lo) if crop else None, tuple(hi) if crop else None)


def iter_slabs(vertices: np.ndarray, faces: np.ndarray, grid: BoxGrid, slab: int = 64) -> Iterator[np.ndarray]:
    """The parity occupancy of `grid`, slab by slab along y: (nx, <= slab, nz) bool arrays in order."""
    V, F = np.asarray(vertices, float), np.asarray(faces)
    xs, ys, zs = grid.axis(0), grid.axis(1), grid.axis(2)
    ys = ys + ROW_SHIFT[0] * grid.pitch
    zs = zs + ROW_SHIFT[1] * grid.pitch
    ty = V[F][:, :, 1]
    tmin, tmax = ty.min(axis=1), ty.max(axis=1)
    cx = cz = None
    if grid.crop_lo is not None:
        cx = (xs >= grid.crop_lo[0]) & (xs <= grid.crop_hi[0])
        cz = (zs >= grid.crop_lo[2]) & (zs <= grid.crop_hi[2])
    for j0 in range(0, len(ys), slab):
        yy = ys[j0:j0 + slab]
        # parity along x only needs the triangles whose y range meets the slab's rows
        sel = (tmax >= yy[0]) & (tmin <= yy[-1])
        occ = parity_axes(V, F[sel], xs, yy, zs) if sel.any() else np.zeros((len(xs), len(yy), len(zs)), bool)
        if cx is not None:
            cy = (yy >= grid.crop_lo[1]) & (yy <= grid.crop_hi[1])
            occ &= cx[:, None, None] & cy[None, :, None] & cz[None, None, :]
        yield occ


# ------------------------------------------------------------------------------------------------ topology

def _euler_lut() -> np.ndarray:
    """8 x the Euler contribution of one lattice vertex, indexed by the occupancy code of its 2x2x2 voxel window (bit
    4*dx + 2*dy + dz). A vertex is in the closed-cube complex if any voxel is; the 6 edges, 12 faces and 8 cubes around
    it are shared by 2, 4 and 8 vertices, so chi = sum over vertices of [v] - E_v/2 + F_v/4 - C_v/8."""
    lut = np.zeros(256, np.int64)
    for code in range(256):
        o = np.array([(code >> (4 * dx + 2 * dy + dz)) & 1 for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
                     bool).reshape(2, 2, 2)
        v = int(o.any())
        e = sum(int(np.take(o, s, axis=d).any()) for d in range(3) for s in (0, 1))
        f = 0
        for d in range(3):                            # faces normal to... lying in the plane spanned by the two other axes
            a, b = [x for x in range(3) if x != d]
            for sa in (0, 1):
                for sb in (0, 1):
                    idx = [slice(None)] * 3
                    idx[a], idx[b] = sa, sb
                    f += int(o[tuple(idx)].any())
        c = int(o.sum())
        lut[code] = 8 * v - 4 * e + 2 * f - c
    return lut


_LUT = _euler_lut()


def _window_codes(a: np.ndarray) -> np.ndarray:
    """Codes of every 2x2x2 window of `a` (already padded on the sides that need it)."""
    a = a.astype(np.uint8)
    code = np.zeros(tuple(n - 1 for n in a.shape), np.uint8)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                code |= a[dx:dx + code.shape[0], dy:dy + code.shape[1], dz:dz + code.shape[2]] << (4 * dx + 2 * dy + dz)
    return code


class _UnionFind:
    def __init__(self):
        self.parent = np.zeros(0, np.int64)

    def add(self, n: int) -> int:
        start = len(self.parent)
        self.parent = np.concatenate([self.parent, np.arange(start, start + n)])
        return start

    def find(self, i: int) -> int:
        p = self.parent
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return root

    def union_pairs(self, a: np.ndarray, b: np.ndarray) -> None:
        for x, y in set(zip(a.tolist(), b.tolist())):
            rx, ry = self.find(x), self.find(y)
            if rx != ry:
                self.parent[max(rx, ry)] = min(rx, ry)

    def roots(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.parent))], np.int64)


_S26 = np.ones((3, 3, 3), bool)
_S6 = ndimage.generate_binary_structure(3, 1)


def _pairs26(p: np.ndarray, q: np.ndarray):
    """Label pairs (p[x, z], q[x', z']) of two adjacent planes that touch in 26-connectivity (|x - x'|, |z - z'| <= 1)."""
    out_a, out_b = [], []
    nx, nz = p.shape
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            ps = p[max(0, -dx):nx - max(0, dx), max(0, -dz):nz - max(0, dz)]
            qs = q[max(0, dx):nx - max(0, -dx), max(0, dz):nz - max(0, -dz)]
            m = (ps > 0) & (qs > 0)
            out_a.append(ps[m]); out_b.append(qs[m])
    return np.concatenate(out_a), np.concatenate(out_b)


@dataclass
class Topology:
    b0: int
    b1: int
    b2: int
    chi: int
    voxels: int
    volume: float
    pitch: float
    shape: Tuple[int, int, int]

    def to_record(self) -> dict:
        return asdict(self)


def slab_topology(slabs: Iterator[np.ndarray], pitch: float) -> Topology:
    """Betti numbers of the closed-cube solid of an occupancy grid delivered as consecutive y-slabs (the grid must have
    an empty margin, as box_grid gives). chi from the vertex-window table, b0 from 26-connected labels and b2 from the
    6-connected complement, both stitched across slab boundaries with a union-find."""
    chi8 = 0
    voxels = 0
    uf_o, uf_c = _UnionFind(), _UnionFind()
    border_c = set()
    prev_occ = prev_lo = prev_lc = None
    shape = [0, 0, 0]
    for occ in slabs:
        nx, ny, nz = occ.shape
        shape[0], shape[1], shape[2] = nx, shape[1] + ny, nz
        voxels += int(occ.sum())
        # Euler: windows whose vertex lies at y-planes 0..ny-1 of this slab use the previous slab's last voxel plane
        below = prev_occ[:, -1:, :] if prev_occ is not None else np.zeros((nx, 1, nz), bool)
        a = np.pad(np.concatenate([below, occ], axis=1), ((1, 1), (0, 0), (1, 1)))
        chi8 += int(_LUT[_window_codes(a)].sum())
        # object components
        lo, n_o = ndimage.label(occ, structure=_S26)
        base_o = uf_o.add(n_o)
        lo = np.where(lo > 0, lo + base_o, 0)
        if prev_lo is not None and n_o:
            pa, pb = _pairs26(prev_lo[:, -1, :], lo[:, 0, :])
            uf_o.union_pairs(pa - 1, pb - 1)
        # complement components (6-connected); x/z faces of every slab are the grid border
        lc, n_c = ndimage.label(~occ, structure=_S6)
        base_c = uf_c.add(n_c)
        lc = np.where(lc > 0, lc + base_c, 0)
        edge = np.concatenate([lc[0].ravel(), lc[-1].ravel(), lc[:, :, 0].ravel(), lc[:, :, -1].ravel()])
        if prev_lc is None:
            edge = np.concatenate([edge, lc[:, 0, :].ravel()])
        border_c.update((edge[edge > 0] - 1).tolist())
        if prev_lc is not None:
            m = (prev_lc[:, -1, :] > 0) & (lc[:, 0, :] > 0)
            uf_c.union_pairs(prev_lc[:, -1, :][m] - 1, lc[:, 0, :][m] - 1)
        prev_occ, prev_lo, prev_lc = occ, lo, lc
    # the vertex planes above the last voxel plane
    if prev_occ is not None:
        nx, _, nz = prev_occ.shape
        a = np.pad(np.concatenate([prev_occ[:, -1:, :], np.zeros((nx, 1, nz), bool)], axis=1), ((1, 1), (0, 0), (1, 1)))
        chi8 += int(_LUT[_window_codes(a)].sum())
        lc = prev_lc
        border_c.update((lc[:, -1, :][lc[:, -1, :] > 0] - 1).tolist())
    b0 = len(set(uf_o.roots().tolist())) if len(uf_o.parent) else 0
    roots_c = uf_c.roots() if len(uf_c.parent) else np.zeros(0, np.int64)
    outer = {int(roots_c[i]) for i in border_c}
    b2 = len(set(roots_c.tolist()) - outer)
    assert chi8 % 8 == 0, chi8
    chi = chi8 // 8
    return Topology(b0=b0, b1=b0 + b2 - chi, b2=b2, chi=chi, voxels=voxels, volume=voxels * pitch ** 3,
                    pitch=float(pitch), shape=tuple(shape))


def occupancy_topology(occ: np.ndarray, pitch: float = 1.0, slab: int = 64) -> Topology:
    """slab_topology of an in-memory grid (padded with an empty margin here)."""
    occ = np.pad(np.asarray(occ, bool), 1)
    return slab_topology((occ[:, j:j + slab, :] for j in range(0, occ.shape[1], slab)), pitch)


def grid_topology(vertices, faces, lo, hi, pitch: float, crop: bool = False, slab: int = 64) -> Topology:
    """Topology of the parity grid of a closed mesh over [lo, hi] at `pitch` (with `crop`, of mesh intersect box)."""
    grid = box_grid(lo, hi, pitch, crop=crop)
    return slab_topology(iter_slabs(vertices, faces, grid, slab), pitch)
