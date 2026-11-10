#!/usr/bin/env python3
"""
Second-order surface meshes of a STEP model: six-node triangles whose nodes lie on the exact surface.

A linear tessellation replaces a cylinder by flat facets: the closest point of the field is then a point of a facet,
up to the chord height (sagitta) inside the true surface, and the normal is constant per facet. A six-node triangle
carries a quadratic patch S(L) = sum_i N_i(L) x_i through three corners and three edge midpoints that all lie on the
surface (gmsh meshes the STEP's faces in their own parameter space), so the patch follows the curvature and its normal
dS/du x dS/dv varies inside the element.

    nodes, tri6 = mesh_step_second_order("part.step", max_size=4.0)     # gmsh, order 2
    surface = QuadraticSurface(nodes, tri6)
    q = surface.query(points)            # q.closest, q.distance, q.normal, q.face, q.bary, q.iterations

The query: libigl's exact closest point on the once-subdivided LINEAR mesh (every six-node triangle as four flat ones)
names the element and a starting point; a Gauss-Newton iteration in the element's barycentric coordinates then moves
the point onto the quadratic patch. A step that would leave the element is cut at its edge; points that end on an edge
are tried on the neighbour across it as well, and the nearer result is kept.

Measured 2026-09-27 (scripts/measure_second_order.py, docs/data/second_order_mesh.json; 20000 points within 2 mm of
the surface, truth analytic, r = 10 mm):

    sphere    FreeCAD tessellation at deflection 0.02 (the loader's default): 8000 triangles, distance error 6.1e-3 mm
              mean / 2.3e-2 max, normal error 1.14 deg mean / 10.7 max
              320 six-node triangles (size 4): 5.2e-4 mm mean / 3.3e-3 max, normal 0.066 deg mean / 0.57 max
              3168 six-node triangles (size 1): 5.1e-6 mm mean / 3.3e-4 max, normal 0.002 deg mean / 0.19 max
    cylinder  FreeCAD tessellation: 500 triangles, 2.1e-3 mm mean / 3.1e-3 max, normal 0.72 deg mean / 1.65 max
              438 six-node triangles (size 4): 1.3e-4 mm mean / 6.5e-4 max, normal 0.017 deg mean / 0.14 max
    the same six-node meshes read as four flat triangles each are WORSE than FreeCAD's tessellation (sphere, size 4:
    3.0e-2 mm mean): the gain is in the patches, not in the node positions.
    query: 0.09-0.10 s for the 20000 points (flat query on the same points: 0.003-0.02 s); gmsh meshes either solid
    in under 0.2 s.
The liner (docs/data/second_order_liner.json, 2026-09-27; 1500 points within 2 mm, truth by OCCT distToShape on the
B-rep, commit da3c5521): 321 582 six-node triangles against 251 062 flat ones (deflection 0.02); distance error 3.2e-4 mm
mean (median 1.2e-6) against 4.2e-3, 95.3 % of the points within 1e-3 mm against 37.6 %, the same maximum (0.115 against
0.118 mm); gmsh 73 s against 3.9 s for the tessellation, query 1.13 s against 0.19 s.
Not measured: the frame, the effect on the reconstruction loop. Used only when SHAPE_RECO_MESH_ORDER=2 (and only for STEP
input); field_channels.mesh_channels queries the patches only under the igl backend (see its docstring).

Node order (gmsh, element type 9): corners 0 1 2, then the midpoints of the edges 0-1, 1-2, 2-0.
Shape functions: N_i = L_i (2 L_i - 1) for the corners, N_3 = 4 L_0 L_1, N_4 = 4 L_1 L_2, N_5 = 4 L_2 L_0.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

TRI3, TRI6 = 2, 9                                  # gmsh element types
SUBDIVISION = np.array([[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]])   # a six-node triangle as four flat ones
# the barycentric coordinates (L0, L1, L2) of the six nodes, and of the corners of the four flat triangles
NODE_BARY = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [0, .5, .5], [.5, 0, .5]], float)


def find_gmsh() -> Optional[str]:
    return os.environ.get("SHAPE_RECO_GMSH") or shutil.which("gmsh")


def read_msh2(path) -> Tuple[np.ndarray, np.ndarray, int]:
    """(nodes (n, 3), triangles (m, 3 or 6) as row indices into nodes, order) of a gmsh 2.2 ASCII file; only the
    surface triangles are read (lines and points of the STEP's edges and vertices are skipped)."""
    lines = Path(path).read_text().split("\n")
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    rows = [l.split() for l in lines[i + 2:i + 2 + n]]
    nodes = np.array([r[1:4] for r in rows], float)
    row_of = {int(r[0]): k for k, r in enumerate(rows)}
    j = lines.index("$Elements")
    m = int(lines[j + 1])
    tris, kind = [], None
    for l in lines[j + 2:j + 2 + m]:
        t = l.split()
        if int(t[1]) in (TRI3, TRI6):
            kind = int(t[1])
            tris.append([row_of[int(x)] for x in t[3 + int(t[2]):]])
    if not tris:
        raise ValueError(f"{path}: no surface triangles")
    return nodes, np.array(tris, np.int64), 2 if kind == TRI6 else 1


def mesh_step(step_path, max_size: float, min_size: Optional[float] = None, order: int = 2,
              gmsh: Optional[str] = None, timeout: float = 600.0, per_circle: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """The surface mesh of a STEP model by gmsh: (nodes, triangles (m, 3) for order 1, (m, 6) for order 2). `max_size`
    / `min_size`: the element size bounds in the model's units (min default: max / 20). `per_circle`: elements per
    full turn of a curved face (gmsh's size from curvature), so that a bore of radius 2 gets as many elements around
    as one of radius 100; the size bounds still hold."""
    exe = gmsh or find_gmsh()
    if exe is None:
        raise RuntimeError("gmsh not found (apt install gmsh, or set SHAPE_RECO_GMSH)")
    tmp = Path(tempfile.mkdtemp(prefix="gmsh_"))
    out = tmp / "surface.msh"
    cmd = [exe, str(Path(step_path).resolve()), "-2", "-order", str(order), "-clmax", str(max_size),
           "-clmin", str(min_size if min_size is not None else max_size / 20.0), "-format", "msh2", "-o", str(out), "-v", "0"]
    if per_circle:
        cmd += ["-clcurv", str(int(per_circle))]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(f"gmsh could not mesh {step_path}: {(proc.stderr or proc.stdout)[-400:]}")
        nodes, tris, got = read_msh2(out)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if got != order:
        raise RuntimeError(f"gmsh wrote order {got}, asked for {order}")
    return nodes, tris


def mesh_step_second_order(step_path, max_size: float, min_size: Optional[float] = None, **kwargs):
    return mesh_step(step_path, max_size, min_size, order=2, **kwargs)


def shape_functions(bary: np.ndarray) -> np.ndarray:
    """(n, 6) N_i at the barycentric coordinates (n, 3)."""
    l0, l1, l2 = bary[:, 0], bary[:, 1], bary[:, 2]
    return np.stack([l0 * (2 * l0 - 1), l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), 4 * l0 * l1, 4 * l1 * l2, 4 * l2 * l0], axis=1)


def shape_derivatives(bary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(dN/du, dN/dv), each (n, 6), with u = L1, v = L2 and L0 = 1 - u - v."""
    l0, l1, l2 = bary[:, 0], bary[:, 1], bary[:, 2]
    z = np.zeros_like(l0)
    du = np.stack([-(4 * l0 - 1), 4 * l1 - 1, z, 4 * (l0 - l1), 4 * l2, -4 * l2], axis=1)
    dv = np.stack([-(4 * l0 - 1), z, 4 * l2 - 1, -4 * l1, 4 * l1, 4 * (l0 - l2)], axis=1)
    return du, dv


def subdivide(nodes: np.ndarray, tri6: np.ndarray, levels: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The quadratic patches as flat triangles: `levels` = 1 uses the six nodes themselves (4 triangles per element),
    `levels` = k evaluates every patch on a regular barycentric grid of 2^k segments per edge (4^k triangles).
    Returns (vertices, faces, parent element of every face, barycentric coordinates (f, 3, 3) of every face's corners
    in its parent). Vertices are not merged between elements for levels > 1."""
    if levels == 1:
        faces = tri6[:, SUBDIVISION].reshape(-1, 3)
        parent = np.repeat(np.arange(len(tri6)), 4)
        bary = np.tile(NODE_BARY[SUBDIVISION], (len(tri6), 1, 1))
        return nodes, faces, parent, bary
    s = 2 ** levels
    grid = [(i, j) for i in range(s + 1) for j in range(s + 1 - i)]
    index = {g: k for k, g in enumerate(grid)}
    local = np.array([[1 - (i + j) / s, i / s, j / s] for i, j in grid])
    tris = []
    for i in range(s):
        for j in range(s - i):
            tris.append([index[(i, j)], index[(i + 1, j)], index[(i, j + 1)]])
            if i + j < s - 1:
                tris.append([index[(i + 1, j)], index[(i + 1, j + 1)], index[(i, j + 1)]])
    tris = np.array(tris)
    n_el, n_loc = len(tri6), len(grid)
    verts = np.einsum("gk,ekd->egd", shape_functions(local), nodes[tri6]).reshape(-1, 3)
    faces = (tris[None] + (np.arange(n_el) * n_loc)[:, None, None]).reshape(-1, 3)
    parent = np.repeat(np.arange(n_el), len(tris))
    bary = np.tile(local[tris], (n_el, 1, 1))
    return verts, faces, parent, bary


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


@dataclass
class QuadraticQuery:
    closest: np.ndarray        # (n, 3) on the quadratic patches
    distance: np.ndarray       # (n,) >= 0
    normal: np.ndarray         # (n, 3) unit normal of the patch at the closest point (the mesh's orientation)
    face: np.ndarray           # (n,) six-node element
    bary: np.ndarray           # (n, 3) barycentric coordinates in it
    iterations: np.ndarray     # (n,) Gauss-Newton steps taken
    linear_distance: np.ndarray  # (n,) the distance to the flat subdivision the iteration started from


class QuadraticSurface:
    """The closest-point query on a mesh of six-node triangles (see the module docstring)."""

    def __init__(self, nodes: np.ndarray, tri6: np.ndarray):
        self.nodes = np.asarray(nodes, float)
        self.tri6 = np.asarray(tri6, np.int64)
        if self.tri6.ndim != 2 or self.tri6.shape[1] != 6:
            raise ValueError(f"QuadraticSurface needs six-node triangles, got {self.tri6.shape}")
        self.vertices, self.faces, self.parent, self.face_bary = subdivide(self.nodes, self.tri6, 1)
        self._neighbour = self._edge_neighbours()

    def _edge_neighbours(self) -> np.ndarray:
        """(m, 3) the element across the edge opposite to corner k (edge 1-2, 2-0, 0-1), -1 on an open edge; the
        edge's midpoint node identifies it (one node per edge, shared by both elements)."""
        mid_of_opposite = self.tri6[:, [4, 5, 3]]                     # opposite corner 0: edge 1-2 = midpoint node 4 ...
        flat = mid_of_opposite.reshape(-1)
        order = np.argsort(flat, kind="stable")
        out = np.full(flat.shape, -1, np.int64)
        same = flat[order][1:] == flat[order][:-1]
        a, b = order[:-1][same], order[1:][same]
        out[a], out[b] = b // 3, a // 3
        return out.reshape(-1, 3)

    def evaluate(self, face: np.ndarray, bary: np.ndarray) -> np.ndarray:
        return np.einsum("nk,nkd->nd", shape_functions(bary), self.nodes[self.tri6[face]])

    def normals(self, face: np.ndarray, bary: np.ndarray) -> np.ndarray:
        x = self.nodes[self.tri6[face]]
        du, dv = shape_derivatives(bary)
        n = np.cross(np.einsum("nk,nkd->nd", du, x), np.einsum("nk,nkd->nd", dv, x))
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-300)

    def _newton(self, p: np.ndarray, face: np.ndarray, bary: np.ndarray, max_iter: int, tol: float):
        """Gauss-Newton for min |S(u, v) - p|^2 inside each element; returns (bary, iterations)."""
        x = self.nodes[self.tri6[face]]
        uv = bary[:, 1:3].copy()
        its = np.zeros(len(p), int)
        active = np.ones(len(p), bool)
        for _ in range(max_iter):
            if not active.any():
                break
            k = np.nonzero(active)[0]
            b = np.column_stack([1 - uv[k].sum(1), uv[k]])
            r = np.einsum("nk,nkd->nd", shape_functions(b), x[k]) - p[k]
            du, dv = shape_derivatives(b)
            su, sv = np.einsum("nk,nkd->nd", du, x[k]), np.einsum("nk,nkd->nd", dv, x[k])
            a11, a12, a22 = (su * su).sum(1), (su * sv).sum(1), (sv * sv).sum(1)
            g1, g2 = (su * r).sum(1), (sv * r).sum(1)
            det = np.maximum(a11 * a22 - a12 * a12, 1e-300)
            step = np.column_stack([-(a22 * g1 - a12 * g2) / det, -(a11 * g2 - a12 * g1) / det])
            new = uv[k] + step
            # back into the element: clip u, v >= 0, then u + v <= 1 along the direction to the edge
            new = np.maximum(new, 0.0)
            over = new.sum(1) > 1.0
            new[over] = new[over] / new[over].sum(1, keepdims=True)
            moved = np.abs(new - uv[k]).max(1)
            uv[k] = new
            its[k] += 1
            active[k[moved < tol]] = False
        return np.column_stack([1 - uv.sum(1), uv]), its

    def query(self, points: np.ndarray, max_iter: int = 12, tol: float = 1e-10, edge: float = 1e-9) -> QuadraticQuery:
        import igl
        p = np.ascontiguousarray(np.asarray(points, float).reshape(-1, 3))
        d2, f, c = igl.point_mesh_squared_distance(p, np.ascontiguousarray(self.vertices), np.ascontiguousarray(self.faces))
        f = np.asarray(f, int).reshape(-1)
        c = np.asarray(c, float).reshape(-1, 3)
        # the barycentric coordinates of the flat closest point in its flat triangle, mapped to the parent element
        a, b_, cc = (self.vertices[self.faces[f, k]] for k in range(3))
        v0, v1, v2 = b_ - a, cc - a, c - a
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
        w1, w2 = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        w = np.clip(np.column_stack([1 - w1 - w2, w1, w2]), 0.0, 1.0)
        w /= w.sum(1, keepdims=True)
        start = np.einsum("nk,nkd->nd", w, self.face_bary[f])
        face = self.parent[f]
        bary, its = self._newton(p, face, start, max_iter, tol)
        closest = self.evaluate(face, bary)
        dist = np.linalg.norm(closest - p, axis=1)
        # a point that ended on an edge may be nearer to the neighbour's patch
        for k in range(3):
            on_edge = (bary[:, k] <= edge) & (self._neighbour[face, k] >= 0)
            if not on_edge.any():
                continue
            idx = np.nonzero(on_edge)[0]
            nb = self._neighbour[face[idx], k]
            guess = self._bary_of(nb, closest[idx])
            b2, it2 = self._newton(p[idx], nb, guess, max_iter, tol)
            c2 = self.evaluate(nb, b2)
            d2n = np.linalg.norm(c2 - p[idx], axis=1)
            better = d2n < dist[idx] - 1e-14
            j = idx[better]
            face[j], bary[j], closest[j], dist[j] = nb[better], b2[better], c2[better], d2n[better]
            its[j] += it2[better]
        return QuadraticQuery(closest, dist, self.normals(face, bary), face, bary, its, np.sqrt(np.maximum(np.asarray(d2, float).reshape(-1), 0.0)))

    def _bary_of(self, face: np.ndarray, x: np.ndarray) -> np.ndarray:
        """A starting point in `face` for the 3-D point x: its barycentric coordinates in the flat corner triangle."""
        t = self.nodes[self.tri6[face][:, :3]]
        v0, v1, v2 = t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], x - t[:, 0]
        d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
        d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
        den = np.maximum(d00 * d11 - d01 * d01, 1e-300)
        w1, w2 = (d11 * d20 - d01 * d21) / den, (d00 * d21 - d01 * d20) / den
        w = np.clip(np.column_stack([1 - w1 - w2, w1, w2]), 0.0, 1.0)
        return w / w.sum(1, keepdims=True)


def outward(nodes: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """The triangles with outward normals (positive enclosed volume); a six-node triangle is turned over by swapping
    the corners 1 and 2 and the midpoints that go with them."""
    corners = tri[:, :3]
    if signed_volume(nodes - nodes.mean(axis=0), corners) >= 0:
        return tri
    return tri[:, [0, 2, 1]] if tri.shape[1] == 3 else tri[:, [0, 2, 1, 5, 4, 3]]


def default_sizes(step_nodes_extent: float) -> Tuple[float, float]:
    """(max, min) element size for a part of the given bounding-box diagonal: 1/40 and 1/4000 of it."""
    return step_nodes_extent / 40.0, step_nodes_extent / 4000.0


def load_quadratic_field(step_path, max_size: Optional[float] = None, min_size: Optional[float] = None,
                         per_circle: int = 12, **kwargs) -> "QuadraticField":
    """The second-order field of a STEP file. Sizes default to 1/40 and 1/4000 of the bounding-box diagonal (a first,
    coarse gmsh pass finds it), 12 elements per full turn of a curved face."""
    if max_size is None:
        coarse, _ = mesh_step(step_path, 1e22, 0.0, order=1)
        diag = float(np.linalg.norm(coarse.max(axis=0) - coarse.min(axis=0)))
        max_size, auto_min = default_sizes(diag)
        min_size = auto_min if min_size is None else min_size
    nodes, tri6 = mesh_step(step_path, max_size, min_size, order=2, per_circle=per_circle)
    return QuadraticField(nodes, outward(nodes, tri6), **kwargs)


def _surface_field_class():
    try:
        from src.closest_point_field import Query, SurfaceField
    except ImportError:                                   # imported as dataset_gen.src.quadratic_mesh
        from .closest_point_field import Query, SurfaceField
    return Query, SurfaceField


_Query, _SurfaceField = _surface_field_class()


class QuadraticField(_SurfaceField):
    """A SurfaceField whose closest points lie on the quadratic patches of a six-node mesh. It IS a SurfaceField of the
    flat subdivision (vertices, faces, area, bounds, sampling, genus and ray parity work on that), and `query` refines
    every answer onto the patch: closest, vector, distance and normal are the patch's. `face` is the FIRST of the four
    flat triangles of the element that holds the answer (not necessarily the one under the point), so a caller that
    looks up `face_normals[face]` gets a flat normal - use the query's `normal`. `exact` is ignored (the patch query is
    always libigl's start plus the Gauss-Newton refinement, so libigl is needed whatever SHAPE_RECO_FIELD_BACKEND says).
    `order` = 2 tells a caller which field it holds."""

    order = 2

    def __init__(self, nodes: np.ndarray, tri6: np.ndarray, **kwargs):
        self.surface = QuadraticSurface(nodes, tri6)
        super().__init__(self.surface.vertices, self.surface.faces, **kwargs)

    def query(self, points: np.ndarray, chunk: int = 4000, exact: bool = True):
        p = np.asarray(points, float).reshape(-1, 3)
        if not len(p):
            return super().query(p, chunk, exact)
        q = self.surface.query(p)
        vector = q.closest - p
        sign = np.where((vector * q.normal).sum(-1) >= 0.0, 1.0, -1.0)
        flat = np.searchsorted(self.surface.parent, q.face)          # the first flat triangle of the element
        return _Query(q.closest, vector, q.distance, q.normal, flat, sign * q.distance)
