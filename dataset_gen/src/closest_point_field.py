#!/usr/bin/env python3
"""
The closest-surface-point vector field of a solid, computed from its GEOMETRY (an STL mesh, or a STEP model tessellated by
FreeCAD) - not from a voxel grid.

For a query point x the field is V(x) = c(x) - x, where c(x) is the point of the surface nearest to x. It points to the surface
from both sides, |V| is the unsigned distance, and where the SDF f is differentiable V = -f grad f (so it carries the SDF
and its gradient in one vector). Unlike the SDF it also tells WHERE on the surface the nearest point is, and the direction of
V at a surface point is the surface normal. It is discontinuous on the medial axis (the set of points with two nearest
surface points), as every distance-based field is.

The SIGN of the distance comes from the STL's own face normal at c(x), not from a separate inside/outside test: x is on the
side the normal points away from x sits on the OUTSIDE (negative) and the opposite side is INSIDE (positive) - the sign of
dot(V(x), normal), since V(x) = c(x) - x points back towards the surface from x. This needs the mesh to be closed and its
normals consistently oriented (`is_closed()`); on an open mesh, or off the medial axis on a very coarse tessellation, the
"normal at c(x)" can be ambiguous and the sign then only reflects that one nearest face, not the whole solid.

`SurfaceField.query(points)` is exact (the closest point on the triangle set) and vectorised: a KD-tree over the centroids of
the small triangles proposes candidates, the few large triangles are always checked, the exact point-to-triangle projection of
Ericson (Real-Time Collision Detection, 5.1.5) picks the winner, and a bounding argument proves nothing was missed (checked against
trimesh's exact closest_point, tests/test_closest_point_field.py).

    field = load_field("part.stl")                      # or "part.step" (FreeCAD tessellates it)
    q = field.query(points)                             # q.closest, q.vector, q.distance, q.signed_distance, q.normal, q.face
    pts = field.sample_points(20000, band=2.0)          # points within `band` of the surface, for fitting
    grid = field.grid_vectors(64, 40.0)                 # (3, 64, 64, 64) on the project's voxel grid

Pure numpy + scipy (+ trimesh only to read an STL). STEP needs `freecadcmd`. The one-pass query (exact=False) runs in torch
when it is installed (`field_backend`; same answer, 10-20x faster on a GPU). SHAPE_RECO_FIELD_BACKEND=igl (needs libigl)
answers BOTH modes with libigl's exact AABB-tree closest point over all triangles (`_query_igl`): exact by construction,
50-100x faster than the one-pass query on the CPU, no refinement needed (docs/data/field_igl_bench.json). Who uses which
(2026-10-01): the library default is "auto" (cuda, else torch, else numpy); the generator's field writer
(sample_writer.write_field) switches its process to igl when libigl is installed and no backend is set (user decision
2026-09-24; the backend is recorded per sample); the GUI and scripts/time_cascade.py pass igl; ResidualConfig's
field backend None keeps the environment's (docs/CODEMAP.md section 2).

Second order: with SHAPE_RECO_MESH_ORDER=2 `load_field` returns a quadratic_mesh.QuadraticField for a STEP file (six-node
triangles from gmsh); an STL stays first order.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def closest_points_on_triangles(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Closest point on triangle (A, B, C) to P, all (m, 3) - Ericson 5.1.5, vectorised over m."""
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    denom = va + vb + vc
    safe = np.where(np.abs(denom) > 1e-30, denom, 1.0)
    inside = A + ab * (vb / safe)[:, None] + ac * (vc / safe)[:, None]                 # face interior

    e = np.where(np.abs(d1 - d3) > 1e-30, d1 - d3, 1.0)
    on_ab = A + ab * (d1 / e)[:, None]
    e = np.where(np.abs(d2 - d6) > 1e-30, d2 - d6, 1.0)
    on_ac = A + ac * (d2 / e)[:, None]
    w = (d4 - d3) + (d5 - d6)
    w = np.where(np.abs(w) > 1e-30, w, 1.0)
    on_bc = B + (C - B) * ((d4 - d3) / w)[:, None]

    out = inside
    for cond, alt in (((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), on_bc), ((vb <= 0) & (d2 >= 0) & (d6 <= 0), on_ac),
                      ((vc <= 0) & (d1 >= 0) & (d3 <= 0), on_ab), ((d6 >= 0) & (d5 <= d6), C),
                      ((d3 >= 0) & (d4 <= d3), B), ((d1 <= 0) & (d2 <= 0), A)):
        out = np.where(cond[:, None], alt, out)
    return out


def voxel_surface_mesh(occ: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """A closed triangle mesh of a binary voxel grid's boundary, in voxel-index units: one quad (as two triangles) per
    face between an occupied voxel and an empty one (a grid edge counts as empty on its far side). The "blocky"/Minecraft
    surface of the grid - no marching-cubes smoothing step that could change its topology. (None, None) if `occ` holds no
    material. Pure numpy (only `trimesh`, already a dependency, to drop unreferenced corners of the padded corner grid and
    weld nothing else, since every corner already has a unique index by construction)."""
    occ = np.asarray(occ, bool)
    if not occ.any():
        return None, None
    nz, ny, nx = occ.shape
    corners = np.stack(np.meshgrid(np.arange(nz + 1), np.arange(ny + 1), np.arange(nx + 1), indexing="ij"), axis=-1)
    corner_id = np.arange((nz + 1) * (ny + 1) * (nx + 1)).reshape(nz + 1, ny + 1, nx + 1)
    quads = []
    for axis in range(3):
        for sign in (0, 1):                                                    # the face on the LOW (0) or HIGH (1) side
            pad = [(0, 0)] * 3
            pad[axis] = (1, 0) if sign == 0 else (0, 1)
            padded = np.pad(occ, pad, constant_values=False)
            sl = [slice(None)] * 3
            sl[axis] = slice(0, -1) if sign == 0 else slice(1, None)
            neighbor = padded[tuple(sl)]
            need = np.argwhere(occ & ~neighbor)                                # (n, 3) voxel indices needing this face
            if not len(need):
                continue
            base = need.copy()
            base[:, axis] += sign                                              # the shared coordinate of all 4 corners
            other = [a for a in range(3) if a != axis]
            offsets = [(0, 0), (1, 0), (1, 1), (0, 1)]                         # a consistent loop around the quad
            quad = np.empty((len(need), 4, 3), dtype=int)
            for k, (d0, d1) in enumerate(offsets):
                c = base.copy()
                c[:, other[0]] += d0
                c[:, other[1]] += d1
                quad[:, k] = c
            quads.append(quad)
    quads = np.concatenate(quads, axis=0)                                      # (Q, 4, 3) corner grid-indices
    ids = corner_id[quads[..., 0], quads[..., 1], quads[..., 2]]               # (Q, 4)
    faces = np.concatenate([ids[:, [0, 1, 2]], ids[:, [0, 2, 3]]], axis=0)      # two triangles per quad
    vertices = corners.reshape(-1, 3).astype(float)
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)       # drops the (many) unreferenced corners
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def _closest_points_torch(P, A, B, C):
    """closest_points_on_triangles in torch, broadcasting over leading dims (same branches, same order)."""
    import torch
    ab, ac, ap = B - A, C - A, P - A
    d1, d2 = (ab * ap).sum(-1), (ac * ap).sum(-1)
    bp = P - B
    d3, d4 = (ab * bp).sum(-1), (ac * bp).sum(-1)
    cp = P - C
    d5, d6 = (ab * cp).sum(-1), (ac * cp).sum(-1)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    def safe(x):
        return torch.where(x.abs() > 1e-30, x, torch.ones_like(x))
    denom = safe(va + vb + vc)
    out = A + ab * (vb / denom)[..., None] + ac * (vc / denom)[..., None]
    on_ab = A + ab * (d1 / safe(d1 - d3))[..., None]
    on_ac = A + ac * (d2 / safe(d2 - d6))[..., None]
    on_bc = B + (C - B) * ((d4 - d3) / safe((d4 - d3) + (d5 - d6)))[..., None]
    for cond, alt in (((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), on_bc), ((vb <= 0) & (d2 >= 0) & (d6 <= 0), on_ac),
                      ((vc <= 0) & (d1 >= 0) & (d3 <= 0), on_ab), ((d6 >= 0) & (d5 <= d6), C),
                      ((d3 >= 0) & (d4 <= d3), B), ((d1 <= 0) & (d2 <= 0), A)):
        out = torch.where(cond[..., None], alt, out)
    return out


def field_backend() -> str:
    """Backend of the one-pass query (exact=False): env SHAPE_RECO_FIELD_BACKEND = numpy | torch | cuda | auto (default) | igl.
    auto: cuda if available, else torch on the CPU, else numpy. igl: libigl's exact closest point for both query modes
    (`_query_igl`; an explicit opt-in, so a missing package raises instead of silently falling back)."""
    want = os.environ.get("SHAPE_RECO_FIELD_BACKEND", "auto").lower()
    if want == "numpy":
        return "numpy"
    if want == "igl":
        import igl  # noqa: F401
        return "igl"
    try:
        import torch
    except ImportError:
        return "numpy"
    if want in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    return "torch"


def kdtree_workers() -> int:
    """Threads of the cKDTree k-nearest-centroid query (scipy's `workers`): SHAPE_RECO_KDTREE_WORKERS, default -1 = every
    core. Identical indices (scipy orders each point's neighbours the same way whatever the thread count); measured on a
    13k-face v5 cylinder, 16 384 centres, k = 64: 0.30 s one thread -> 0.05 s (docs/data/field_igl_bench.json)."""
    env = os.environ.get("SHAPE_RECO_KDTREE_WORKERS", "")
    if env:
        return int(env)
    omp = os.environ.get("OMP_NUM_THREADS", "")      # a process that pins its threads (MPI ranks, pool workers) is obeyed:
    return int(omp) if omp.isdigit() and int(omp) > 0 else -1   # N ranks x every core was the "load 70" pattern


def _query_igl(field: "SurfaceField", P: np.ndarray):
    """libigl's exact closest point of every P on the field's triangles (an AABB tree over ALL of them, float64):
    (closest (n, 3), face (n,), squared distance (n,)). No candidate set, no k, no refinement - exact by construction, so
    it stands in for both query modes under SHAPE_RECO_FIELD_BACKEND=igl. Measured against the default path on v5 solids
    at 64^3: identical to 1e-14 mm where the one-pass query is right, and where that one is wrong (its documented
    candidate misses: 20 centres off by up to 0.044 mm on a cylinder, 6798 by up to 0.24 mm on examples/box_with_bore)
    the distance here is the smaller, true one. On the medial axis (exact ties) the point picked may differ."""
    import igl
    if not len(P):
        return np.empty((0, 3)), np.empty(0, int), np.empty(0)
    d2, face, closest = igl.point_mesh_squared_distance(np.ascontiguousarray(P, dtype=np.float64),
                                                       np.ascontiguousarray(field.vertices, dtype=np.float64),
                                                       np.ascontiguousarray(field.faces, dtype=np.int64))
    return (np.asarray(closest, float).reshape(len(P), 3), np.asarray(face, int).reshape(-1),
            np.maximum(np.asarray(d2, float).reshape(-1), 0.0))


def _knn_gpu(centroids: np.ndarray, P: np.ndarray, k: int, extra: int = 16, rows: Optional[int] = None) -> np.ndarray:
    """Indices (n, k) of the k nearest centroids of every point, nearest first, on the GPU: a float32 brute-force
    pre-selection of k + `extra` candidates, re-ranked by exact float64 distances (stable), so the set is cKDTree's
    (measured on 5 v5 solids, 262k voxel centres each: identical sets everywhere, 8-15x faster than one CPU thread)."""
    import torch
    c32 = torch.as_tensor(centroids, device="cuda", dtype=torch.float32)
    c64 = torch.as_tensor(centroids, device="cuda", dtype=torch.float64)
    cn = (c32 * c32).sum(1)
    kk = min(k + extra, len(centroids))
    if rows is None:        # the (rows, n_centroids) distance matrix (+ topk scratch) takes 30 % of the FREE GPU memory, capped by
        budget = 0.3 * torch.cuda.mem_get_info()[0]              # SHAPE_RECO_FIELD_GPU_MB when set (parallel ranks sharing a GPU)
        if os.environ.get("SHAPE_RECO_FIELD_GPU_MB"):
            budget = min(budget, float(os.environ["SHAPE_RECO_FIELD_GPU_MB"]) * 2 ** 20)
        rows = int(max(64, min(4096, budget // (3 * 4 * max(len(centroids), 1)))))   # measured: 324 -> 2048 rows, 21 -> 15 s
    out = np.empty((len(P), k), np.int64)
    for lo in range(0, len(P), rows):
        p32 = torch.as_tensor(np.ascontiguousarray(P[lo:lo + rows]), device="cuda", dtype=torch.float32)
        cand = torch.topk((p32 * p32).sum(1)[:, None] - 2 * p32 @ c32.T + cn[None], kk, dim=1, largest=False).indices
        p64 = torch.as_tensor(np.ascontiguousarray(P[lo:lo + rows]), device="cuda", dtype=torch.float64)
        order = torch.argsort(((c64[cand] - p64[:, None, :]) ** 2).sum(-1), dim=1, stable=True)[:, :k]
        out[lo:lo + rows] = torch.gather(cand, 1, order).cpu().numpy()
    return out


DENSE_BLOCK = 32768        # query points per block of _dense_one_pass (bounded memory, identical answers)


def _dense_one_pass(field: "SurfaceField", P: np.ndarray):
    """The one-pass query (exact=False) with every candidate projected at once in torch (float64): the k nearest small
    triangles by centroid plus all large ones, first minimum in that order - the same answer as `_query_chunk(exact=False)`
    (its pruning only skips candidates that cannot win, its ties keep the earlier candidate). Measured on 6 v5 training
    solids at 64^3: max distance difference 7e-15 mm, 10-20x faster on an RTX 4050, 2-4x on the CPU. None -> numpy path.
    On the GPU the k-nearest-centroid search runs there too (`_knn_gpu`); it was the dominant cost of a generated sample
    (a single-threaded cKDTree query took 5-30 s per 64^3 grid)."""
    backend = field_backend()
    if backend == "numpy" or field._tree is None or not len(P):
        return None
    if len(P) > DENSE_BLOCK:
        # blocks of query points: the k candidates of ALL points at once cost ~1.6 KB per point (+3.2 GB at 128^3,
        # +10.9 GB at 192^3, measured by another session's profile); every point's answer is independent, so the result
        # is bit-identical
        parts = [_dense_one_pass(field, P[lo:lo + DENSE_BLOCK]) for lo in range(0, len(P), DENSE_BLOCK)]
        if any(part is None for part in parts):
            return None
        return tuple(np.concatenate([part[i] for part in parts]) for i in range(3))
    import torch
    k = min(field._k, len(field._small))
    idx = None
    if backend == "cuda":
        try:
            centroids = field.vertices[field.faces[field._small]].mean(axis=1)
            idx = _knn_gpu(centroids, P, k)
        except RuntimeError as e:                           # CUDA out of memory -> the CPU tree
            if "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
    if idx is None:
        _, idx = field._tree.query(P, k=k, workers=kdtree_workers())
    cand = field._small[np.asarray(idx).reshape(len(P), -1)]
    if len(field._large):
        cand = np.concatenate([np.broadcast_to(field._large, (len(P), len(field._large))), cand], axis=1)
    m = cand.shape[1]
    step = max(256, int(1_000_000 // m))                    # about 24 MB per (step, m, 3) float64 tensor
    for device in (["cuda", "cpu"] if backend == "cuda" else ["cpu"]):
        try:
            a, b, c = (torch.as_tensor(x, device=device) for x in (field._a, field._b, field._c))
            closest, face, best = np.empty_like(P), np.empty(len(P), int), np.empty(len(P))
            for lo in range(0, len(P), step):
                p = torch.as_tensor(np.ascontiguousarray(P[lo:lo + step]), device=device)[:, None, :]
                ci = torch.as_tensor(cand[lo:lo + step], device=device)
                cp = _closest_points_torch(p, a[ci], b[ci], c[ci])
                d2 = ((cp - p) ** 2).sum(-1)
                j = d2.argmin(dim=1)
                rows = torch.arange(len(j), device=device)
                closest[lo:lo + len(j)] = cp[rows, j].cpu().numpy()
                face[lo:lo + len(j)] = ci[rows, j].cpu().numpy()
                best[lo:lo + len(j)] = d2[rows, j].cpu().numpy()
            return closest, face, best
        except RuntimeError as e:                           # CUDA out of memory (parallel generator workers) -> CPU
            if device == "cpu" or "out of memory" not in str(e).lower():
                raise
            torch.cuda.empty_cache()
    return None


@dataclass
class Query:
    closest: np.ndarray        # (n, 3) nearest surface point c(x)
    vector: np.ndarray         # (n, 3) V(x) = c(x) - x
    distance: np.ndarray       # (n,)   |V(x)|, the unsigned distance
    normal: np.ndarray         # (n, 3) unit normal of the face holding c(x) (orientation as in the mesh)
    face: np.ndarray           # (n,)   index of that face
    signed_distance: np.ndarray  # (n,) distance, signed by the STL normal at c(x): negative outside, positive inside


class SurfaceField:
    """Exact closest-point queries on a triangle mesh (vertices (v, 3), faces (f, 3)).

    Small triangles are indexed by their centroids in a KD-tree; the few LARGE ones (flat faces of CAD models: a box face, a
    cylinder cap) are checked for every query by brute force. A query examines the `k` nearest small triangles plus all large
    ones and is accepted only when a bounding argument proves that no other triangle can be closer: every triangle lies within
    its own circumradius-like radius of its centroid, so a small triangle whose centroid is farther than
    (best distance so far + largest small radius) cannot beat it. Otherwise `k` is doubled for that query. The answer is exact
    for any mix of triangle sizes."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, k: int = 24, large_factor: float = 3.0, max_large: int = 4096):
        from scipy.spatial import cKDTree
        self.vertices = np.asarray(vertices, float)
        self.faces = np.asarray(faces, int)
        tri = self.vertices[self.faces]
        self._a, self._b, self._c = tri[:, 0], tri[:, 1], tri[:, 2]
        cross = np.cross(self._b - self._a, self._c - self._a)
        area2 = np.linalg.norm(cross, axis=1)
        self.face_normals = cross / np.maximum(area2, 1e-30)[:, None]
        self.area = float(0.5 * area2.sum())
        centroid = tri.mean(axis=1)
        radius = np.linalg.norm(tri - centroid[:, None, :], axis=2).max(axis=1)          # every point of the triangle is within it
        cap = large_factor * max(float(np.median(radius)), 1e-12)
        large = radius > cap
        if large.sum() > max_large:                                                          # keep the brute-force list short
            large = radius >= np.sort(radius)[-max_large]
        self._radius = radius
        self._large = np.nonzero(large)[0]
        self._small = np.nonzero(~large)[0]
        self._small_radius = float(radius[self._small].max()) if len(self._small) else 0.0
        self._k = k
        self._tree = cKDTree(centroid[self._small]) if len(self._small) else None
        self.bounds = (self.vertices.min(axis=0), self.vertices.max(axis=0))

    # ------------------------------------------------------------------------------------------------- queries
    def _project(self, P: np.ndarray, faces: np.ndarray):
        """Closest points of P (n, 3) on the given faces (n, m): (points (n, m, 3), squared distances (n, m))."""
        n, m = faces.shape
        pp = np.repeat(P, m, axis=0)
        f = faces.reshape(-1)
        cp = closest_points_on_triangles(pp, self._a[f], self._b[f], self._c[f])
        return cp.reshape(n, m, 3), ((cp - pp) ** 2).sum(-1).reshape(n, m)

    def query(self, points: np.ndarray, chunk: int = 4000, exact: bool = True) -> Query:
        """Closest points of `points` (n, 3). exact=False takes one pass over the k nearest small triangles: faster for points far from the surface (where
        the exactness proof needs many candidates), but it is wrong whenever a triangle outside those k is nearer - which happens on
        CAD meshes whose walls are long sliver triangles (their centroids are far from nearby points). Use it only where an
        approximate vector is enough."""
        P = np.asarray(points, float).reshape(-1, 3)
        dense = _query_igl(self, P) if field_backend() == "igl" else (None if exact else _dense_one_pass(self, P))
        if dense is not None:
            closest, face, best = dense
        else:
            closest, face, best = np.empty_like(P), np.empty(len(P), int), np.empty(len(P))
            for lo in range(0, len(P), chunk):
                p = P[lo:lo + chunk]
                c, f, d2 = self._query_chunk(p, exact)
                closest[lo:lo + len(p)], face[lo:lo + len(p)], best[lo:lo + len(p)] = c, f, d2
        vector, distance, normal = closest - P, np.sqrt(best), self.face_normals[face]
        # V(x) = c(x) - x points back towards the surface; that is along the OUTWARD normal iff x is inside, so
        # dot(V, normal) > 0 <=> inside (positive), < 0 <=> outside (negative) - sign(0) (on the surface) counts as inside.
        sign = np.where((vector * normal).sum(-1) >= 0.0, 1.0, -1.0)
        return Query(closest, vector, distance, normal, face, sign * distance)

    def _query_chunk(self, p: np.ndarray, exact: bool = True):
        n = len(p)
        cl, fc, d2 = np.empty((n, 3)), np.empty(n, int), np.full(n, np.inf)
        if len(self._large):
            faces = np.broadcast_to(self._large, (n, len(self._large)))
            cp, dd = self._project(p, faces)
            j = dd.argmin(axis=1)
            rows = np.arange(n)
            cl, fc, d2 = cp[rows, j], faces[rows, j], dd[rows, j]
        if self._tree is None:
            return cl, fc, d2
        todo = np.arange(n)
        k = min(self._k, len(self._small))
        first = min(4, k)
        while len(todo):
            dist, idx = self._tree.query(p[todo], k=k, workers=kdtree_workers())
            dist, idx = dist.reshape(len(todo), -1), idx.reshape(len(todo), -1)
            cand = self._small[idx]
            rows = np.arange(len(todo))
            # a good upper bound from the nearest few, then only the candidates that can still beat it
            cp0, dd0 = self._project(p[todo], cand[:, :first])
            j0 = dd0.argmin(axis=1)
            bp, bf, bd = cp0[rows, j0], cand[rows, j0], dd0[rows, j0]
            if k > first:
                rest = cand[:, first:]
                keep = (dist[:, first:] - self._radius[rest]) < np.sqrt(bd)[:, None]                  # lower bound < best so far
                r_i, c_i = np.nonzero(keep)
                if len(r_i):
                    pp = p[todo][r_i]
                    fs = rest[r_i, c_i]
                    cpk = closest_points_on_triangles(pp, self._a[fs], self._b[fs], self._c[fs])
                    ddk = ((cpk - pp) ** 2).sum(-1)
                    order = np.lexsort((ddk, r_i))                                                     # per query: smallest first
                    r_s = r_i[order]
                    firstpos = np.r_[True, r_s[1:] != r_s[:-1]]
                    rr, cpb, ffb, ddb = r_s[firstpos], cpk[order][firstpos], fs[order][firstpos], ddk[order][firstpos]
                    win = ddb < bd[rr]
                    bp[rr[win]], bf[rr[win]], bd[rr[win]] = cpb[win], ffb[win], ddb[win]
            better = bd < d2[todo]
            sel = todo[better]
            cl[sel], fc[sel], d2[sel] = bp[better], bf[better], bd[better]
            done = (dist[:, -1] > np.sqrt(d2[todo]) + self._small_radius) | (k >= len(self._small)) | (not exact)
            todo = todo[~done]
            k = min(k * 2, len(self._small))
        return cl, fc, d2

    def sample_points(self, n: int, band: float = 2.0, seed: int = 0) -> np.ndarray:
        """`n` random points within `band` of the surface, on both sides: a surface point (area-weighted) moved along its face
        normal by a uniform offset in [-band, band] - the region where the field carries information about the shape."""
        rng = np.random.default_rng(seed)
        area = 0.5 * np.linalg.norm(np.cross(self._b - self._a, self._c - self._a), axis=1)
        f = rng.choice(len(self.faces), size=n, p=area / area.sum())
        r1, r2 = rng.random((n, 1)), rng.random((n, 1))
        flip = (r1 + r2) > 1
        r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
        on = self._a[f] + (self._b[f] - self._a[f]) * r1 + (self._c[f] - self._a[f]) * r2
        return on + self.face_normals[f] * rng.uniform(-band, band, (n, 1))

    def grid_vectors(self, resolution: int = 64, grid_mm: float = 40.0, anchor_centre: Optional[np.ndarray] = None,
                     exact: bool = False) -> np.ndarray:
        """V on the project's voxel grid: (3, R, R, R), components in the ARRAY axes [z, y (reversed), x] and in VOXELS, on the
        voxel centres of a grid of `grid_mm` centred on `anchor_centre` (default: the mesh's bounding-box centre). Approximate by default
        (`exact=False`, see `query`): most voxels are far from the surface, where the exact proof is expensive and the vectors matter less.
        Not the in-scale grid of a placed sample (that one is centred on the WORLD origin, field_channels.inscale_spec): only the legacy
        training/exact_field_input.py (and a timing in scripts/make_vectorfield_tex.py) call this; stored fields come from
        field_channels.mesh_channels."""
        v = grid_mm / resolution
        c = (np.arange(resolution) + 0.5 - resolution / 2.0) * v
        z, yr, x = np.meshgrid(c, c, c, indexing="ij")
        centre = (self.bounds[0] + self.bounds[1]) / 2.0 if anchor_centre is None else np.asarray(anchor_centre, float)
        pts = np.stack([x, -yr, z], axis=-1).reshape(-1, 3) + centre
        vec = self.query(pts, exact=exact).vector.reshape(resolution, resolution, resolution, 3)      # world (x, y, z), mm
        return np.stack([vec[..., 2], -vec[..., 1], vec[..., 0]]) / v                   # array axes, voxels

    def _unique_edges(self, faces: Optional[np.ndarray] = None):
        """(unique undirected edges (e, 2), how many faces share each) — shared by `is_closed` and `genus`. Degenerate
        triangles (two of their three vertex indices equal - zero area, no real geometry) are dropped first: a common
        real-world tessellation artefact at a pole/apex (a fan of triangles collapsing to a point can produce a few of
        these) that would otherwise show up as a self-loop edge and a couple of edges shared by the "wrong" number of
        faces, making a genuinely closed surface look open. Measured on `dataset_gen/examples/bs_sphere.stl`: 2 of
        26 718 faces are degenerate; dropping them is what makes this otherwise-closed sphere read as closed at all."""
        faces = self.faces if faces is None else faces
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
        e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
        return np.unique(e, axis=0, return_counts=True)

    def is_closed(self) -> bool:
        """True if every edge is shared by exactly two faces (a watertight mesh, so that inside/outside is defined)."""
        _, counts = self._unique_edges()
        return bool((counts == 2).all())

    def genus(self) -> float:
        """Total genus of the surface, summed over its connected boundary shells, from each shell's own Euler
        characteristic V - E + F = 2 - 2g (NaN if any edge is shared by other than two faces - `is_closed`). Summing per
        shell (rather than treating the whole boundary as one surface) matters whenever there is an internal cavity: a
        hollow box's boundary is TWO disjoint closed shells (the outer skin and the cavity's own wall), each individually a
        topological sphere contributing genus 0 - correctly 0 total (a hollow box has no through-features), where naively
        plugging combined V/E/F into 2-2g would give a meaningless negative number for more than one shell.

        A sphere or a box is genus 0; a plain ring (a picture frame with ONE opening) is genus 1, like a torus; each
        further independent through-feature (another opening, a rod or a rib bridging two otherwise-unconnected parts of
        the solid) adds 1 more - see docs/FRAME_FAILURE.md for a real example (genus 17, a two-way lattice of ~150 bridging
        rods, not a plain frame). Exact and cheap (milliseconds): a topological invariant, unaffected by how fine the
        tessellation is. Every SMALL real hole (a bolt hole, a rivet, a drilled feature) counts the same as a large one, so
        on a real mechanical part this number can be dominated by tiny features - `coarse_genus` below is the rough,
        scale-aware alternative."""
        if not self.is_closed():
            return float("nan")
        import trimesh
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        shells = mesh.split(only_watertight=False) if len(mesh.faces) else [mesh]
        total = 0.0
        for shell in shells:
            faces = shell.faces[(shell.faces[:, 0] != shell.faces[:, 1]) & (shell.faces[:, 1] != shell.faces[:, 2])
                                & (shell.faces[:, 0] != shell.faces[:, 2])]
            if not len(faces):
                continue                                                          # an "island" of nothing but degenerate
                                                                                    # triangles (mesh.split can isolate one
                                                                                    # as its own tiny shell) - not a real
                                                                                    # boundary component, so it adds no genus
            edges, _ = self._unique_edges(faces)
            v, e, f = len(np.unique(faces)), len(edges), len(faces)               # V from the vertices FACES actually use
            total += (2 - (v - e + f)) / 2.0
        return total

    def coarse_genus(self, resolution: int = 32) -> float:
        """Genus of a COARSE voxelization of this surface (`resolution` voxels along its largest extent), not of the exact
        mesh. Motivation: `genus()` counts every real hole alike, but a mechanical part often has many small ones (bolt
        holes, rivets) that are not what "how many big loops does this part have" usually means - voxelizing first, at a
        resolution too coarse to resolve a small hole, merges it back into the surrounding material and leaves only the
        topology of features bigger than roughly one voxel. NaN if the mesh is not closed, or if it voxelizes to nothing.

        Known limitation, confirmed not just suspected (`tests/test_closest_point_field.py`): `.fill()` below turns the
        raw surface voxelization into a solid by flood-filling from the grid's border and taking everything the flood
        never reaches as material - which cannot tell a genuinely solid interior from a fully SEALED cavity, since
        both are equally "not reached from outside". A cavity with no topology of its own washes out harmlessly (both
        `genus()` and this end up calling it genus 0), but if the cavity's OWN boundary has a handle - a strut spanning
        it, entirely inside the seal - that handle's genus is erased by the fill at every resolution, so this function
        can never converge to the exact value for such a part (`genus()`'s per-shell computation still gets it right,
        since it never fills). A correct fix would replace the fast rasterised voxelization with a point-in-solid test
        that respects every shell's winding (`trimesh.contains`), which is far slower - not worth it for a `coarse_genus`
        whose whole point is to be cheap, and not something any topology in this project's own data currently needs
        (`dataset_gen/src/inscale.hollow()`'s cavities never have this kind of internal structure)."""
        if not self.is_closed():
            return float("nan")
        import trimesh
        extent = float((self.bounds[1] - self.bounds[0]).max())
        pitch = extent / resolution
        occ = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False).voxelized(pitch).fill().matrix
        v, f = voxel_surface_mesh(occ)
        if v is None:
            return float("nan")
        return SurfaceField(v, f).genus()

    def genus_by_resolution(self, resolutions: Tuple[int, ...] = (4, 8, 16, 32, 64, 128)) -> dict:
        """{resolution: coarse_genus(resolution)} over a sweep of "growing" voxel resolutions - "at what scale does each
        feature start to matter". Read the HIGH-resolution end, where the number has settled and stays there as resolution
        grows further: that is the genus of the part's BIG features. A single mid-sweep value can be unreliable on its
        own - measured on a synthetic two-tunnel block (exact genus 2, tunnel radius a tenth of the block): 0, 0, 0, 2, 0,
        2, 2, 2 at resolutions 4, 5, 6, 8, 10, 16, 32, 64 - a dip back to 0 at resolution 10 even though 8 already read 2,
        a voxel-grid-alignment aliasing effect right at the resolution that just barely resolves a feature, not a
        monotonic function of resolution. Settling at the high end is the reliable signal; a single value near where a
        feature's own size roughly equals the voxel pitch is not."""
        return {r: self.coarse_genus(r) for r in resolutions}


# ------------------------------------------------------------------------------------------------------- loading

FREECAD_STEP_SCRIPT = '''
import os, Part, MeshPart
shape = Part.read(os.environ["STEP_IN"])
mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=float(os.environ["DEFLECTION"]), AngularDeflection=0.1, Relative=False)
mesh.write(os.environ["STL_OUT"])
'''


try:                                                     # $SHAPE_RECO_FREECADCMD, PATH, /usr/bin; re-exported
    from .freecad_path import find_freecadcmd  # noqa: E402,F401  (as src.* or dataset_gen.src.*)
except ImportError:                                      # run as a script, not as a package module
    from src.freecad_path import find_freecadcmd  # noqa: E402,F401


def tessellate_step(step_path, deflection: float = 0.02, freecad: Optional[str] = None) -> Path:
    """STEP/STP -> an STL file (in a temp directory) via FreeCAD's tessellator; `deflection` in the model's units."""
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        raise RuntimeError("STEP input needs FreeCAD: freecadcmd was not found")
    tmp = Path(tempfile.mkdtemp(prefix="step_mesh_"))
    script, out = tmp / "tessellate.py", tmp / (Path(step_path).stem + ".stl")
    script.write_text(FREECAD_STEP_SCRIPT)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(STEP_IN=str(Path(step_path).resolve()), STL_OUT=str(out), DEFLECTION=str(deflection))
    proc = subprocess.run([freecad, str(script)], capture_output=True, text=True, timeout=300, env=env)
    if not out.is_file():
        raise RuntimeError(f"FreeCAD could not tessellate {step_path}: {(proc.stderr or proc.stdout)[-400:]}")
    return out


def load_mesh(path, deflection: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices, faces) of an .stl, or of a .step/.stp after tessellation. Identical vertices are merged."""
    import trimesh
    p = Path(path)
    if p.suffix.lower() in (".step", ".stp"):
        p = tessellate_step(p, deflection)
    mesh = trimesh.load(str(p), force="mesh", process=True)
    return np.asarray(mesh.vertices), np.asarray(mesh.faces)


def mesh_order() -> int:
    """$SHAPE_RECO_MESH_ORDER: 1 (default) = the flat tessellation, 2 = six-node triangles with the closest point on
    their quadratic patches (src.quadratic_mesh; STEP input only, needs gmsh)."""
    return 2 if os.environ.get("SHAPE_RECO_MESH_ORDER", "1").strip() == "2" else 1


def load_field(path, deflection: float = 0.02, order: Optional[int] = None, **kwargs) -> SurfaceField:
    """The SurfaceField of an STL or STEP file. `order` 2 (default: `mesh_order()`) gives a STEP file's second-order
    field (src.quadratic_mesh.QuadraticField: measured 2026-09-27 on a sphere r = 10, 320 six-node triangles against
    the 8000 flat ones of deflection 0.02: distance error 5.2e-4 against 6.1e-3 mm, normal 0.066 against 1.14 deg).
    An STL has no exact surface to put the midpoints on: it stays first order, whatever `order` says."""
    if (mesh_order() if order is None else order) == 2 and Path(path).suffix.lower() in (".step", ".stp"):
        from .quadratic_mesh import load_quadratic_field
        return load_quadratic_field(path, **kwargs)
    v, f = load_mesh(path, deflection)
    return SurfaceField(v, f, **kwargs)
