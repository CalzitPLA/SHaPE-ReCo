#!/usr/bin/env python3
"""
The model of the accepted operations as an exact mesh, built by mesh booleans (manifold3d) one operation at a time.

Why not the SDF: marching cubes on a grid over the whole part (reco_field.peel.mesh_from_sdf, 72^3) never sees a wall
thinner than a grid cell. On the frame (walls 0.12-0.23 STL units, cells ~1.07) the model mesh came out EMPTY from the
4th operation on, while re-evaluating the whole program at every accepted step (16.8 s of a 46.7 s run). Here every
primitive is its own exact solid (box, cylinder, cone, sphere; the rest - torus, partial sweeps - from the primitive's
own SDF on a grid that fits it), and the model of ops[:n] is the model of ops[:n-1] plus one boolean, cached by prefix.

Without manifold3d (`pip install --user manifold3d`) `available()` is False and the caller keeps its SDF path.
"""

from __future__ import annotations

import atexit
import weakref
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

SEGMENTS = 64                       # circle segments of cylinders, cones, spheres
_LIVE: "weakref.WeakSet" = weakref.WeakSet()


@atexit.register
def _release():
    """Drop the cached Manifolds before the interpreter tears the binding down (else nanobind reports them as leaks)."""
    for mesher in list(_LIVE):
        mesher.clear()


def available() -> bool:
    try:
        import manifold3d  # noqa: F401
        return True
    except ImportError:
        return False


def _dim(d: Dict[str, float], *names) -> float:
    for n in names:
        if d.get(n) is not None:
            return float(d[n])
    raise KeyError(f"none of {names} in {sorted(d)}")


def _placement(p) -> np.ndarray:
    """3x4 [R | centre] of a result_schema.Primitive (local axes: the columns of R)."""
    from reco_gui.cascade_session import quaternion_matrix
    return np.hstack([quaternion_matrix(p.quaternion), np.asarray(p.center_mm, float).reshape(3, 1)])


def _from_triangles(vertices: np.ndarray, triangles: np.ndarray):
    """A Manifold from a triangle soup whose coincident vertices are merged; None when it is not a closed manifold."""
    import manifold3d as m
    v = np.asarray(vertices, np.float64)
    if not len(v):
        return None
    key = np.round(v / (1e-7 * max(float(np.ptp(v, axis=0).max()), 1e-12))).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    tris = inverse.reshape(-1)[np.asarray(triangles, np.int64)]
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])]
    solid = m.Manifold(m.Mesh(np.ascontiguousarray(v[first], np.float32), np.ascontiguousarray(tris, np.uint32)))
    return None if solid.status() != m.Error.NoError or solid.is_empty() else solid


def primitive_solid(p, sdf_mesh: Optional[Callable] = None, crop=None):
    """The exact Manifold of a result_schema.Primitive in world coordinates (the conventions of
    cascade_session.primitive_sdf: box half extents, cylinder/cone along local z centred on the centre, radius1 at -z).
    Full boxes, cylinders, cones and spheres are built analytically; anything else (torus, a u/v sweep) from
    `sdf_mesh(p) -> MeshData` if given. A "composite" is its parts joined by its combine (^ common, - cut, + fuse); a
    "halfspace" is a block on the inner side of its plane (local -z), as large as `crop` = (lo, hi) (the part's padded
    box) or, without one, 1e4 mm. None if it cannot be built."""
    import manifold3d as m
    if p.kind == "composite":
        if len(p.parts) != 2:
            return None
        a, b = (primitive_solid(q, sdf_mesh, crop) for q in p.parts)
        if a is None or b is None:
            return None
        return {"common": a ^ b, "cut": a - b, "fuse": a + b}.get(p.combine)
    if p.kind == "halfspace":
        size = 1e4 if crop is None else 4.0 * float(np.linalg.norm(np.asarray(crop[1]) - np.asarray(crop[0])))
        block = m.Manifold.cube((size, size, size), center=True).translate((0.0, 0.0, -size / 2.0))   # z <= 0
        solid = block.transform(_placement(p))
        if crop is not None:
            lo, hi = (np.asarray(v, float) for v in crop)
            solid = solid ^ m.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
        return solid
    d = dict(p.dims_mm or {})
    full = not p.u and not p.v
    try:
        if p.kind == "box":
            h = np.array([_dim(d, "x_half", "hx"), _dim(d, "y_half", "hy"), _dim(d, "z_half", "hz")])
            if not (h > 0).all():
                return None
            local = m.Manifold.cube(tuple(2 * h), center=True)
        elif p.kind in ("cylinder", "cone") and full:
            h = _dim(d, "half_length", "half_height")
            r1 = _dim(d, "radius") if p.kind == "cylinder" else _dim(d, "radius1")
            r2 = r1 if p.kind == "cylinder" else _dim(d, "radius2")
            if h <= 0 or max(r1, r2) <= 0:
                return None
            local = m.Manifold.cylinder(2 * h, r1, r2, circular_segments=SEGMENTS, center=True)
        elif p.kind == "sphere" and full:
            r = _dim(d, "radius")
            if r <= 0:
                return None
            local = m.Manifold.sphere(r, circular_segments=SEGMENTS)
        else:
            local = None
    except KeyError:
        return None
    if local is not None:
        return local.transform(_placement(p))
    if sdf_mesh is None:
        return None
    mesh = sdf_mesh(p)
    return None if mesh is None else _from_triangles(mesh.vertices, mesh.triangles)


def to_mesh_data(solid):
    from reco_gui.field_session import MeshData
    if solid is None or solid.is_empty():
        return None
    mesh = solid.to_mesh()
    v = np.asarray(mesh.vert_properties, float)[:, :3]
    return MeshData(v, np.asarray(mesh.tri_verts, np.int64))


class ModelMesher:
    """`mesh(ops, key_of)` -> MeshData of the program `ops` (Operations: add = union, cut = difference, a leading cut
    has nothing to cut from and is skipped - as the SDF model of cascade_session did). The Manifold of every prefix seen
    is kept, so the next step costs one primitive and one boolean."""

    def __init__(self, sdf_mesh: Optional[Callable] = None, union_all: bool = False, crop: Optional[Callable] = None):
        self.sdf_mesh = sdf_mesh            # fallback mesher for primitives without an analytic solid
        self.union_all = union_all          # every op is a union (the voxel-cube fallbacks)
        self.crop = crop                    # () -> (lo, hi) of the part's padded box, for half-spaces
        self._models: Dict[Tuple, object] = {(): None}
        self._solids: Dict[Tuple, object] = {}
        _LIVE.add(self)

    def clear(self):
        self._models = {(): None}
        self._solids = {}

    def _solid(self, key, p):
        if key not in self._solids:
            self._solids[key] = primitive_solid(p, self.sdf_mesh, crop=self.crop() if self.crop else None)
        return self._solids[key]

    def model(self, ops: Sequence, key_of: Callable):
        keys = tuple((o.op,) + key_of(o.primitive) for o in ops)
        n = len(keys)
        while n and keys[:n] not in self._models:                     # the longest prefix already built
            n -= 1
        model = self._models[keys[:n]]
        for i in range(n, len(keys)):
            o = ops[i]
            s = self._solid(keys[i][1:], o.primitive)
            if s is not None:
                if model is None:
                    model = s if (o.op == "add" or self.union_all) else None
                elif o.op == "add" or self.union_all:
                    model = model + s
                elif o.op == "intersect":
                    model = model ^ s
                else:
                    model = model - s
            self._models[keys[:i + 1]] = model
        return model

    def mesh(self, ops: Sequence, key_of: Callable):
        return to_mesh_data(self.model(ops, key_of))
