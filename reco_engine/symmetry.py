#!/usr/bin/env python3
"""
The symmetry of a part, and exact copies of a tool onto a symmetric feature (the user's rule 2026-10-29: "always
check if symmetrical features exist").

    sym = detect(V, F)                                   # mirror planes and rotation axes of the closed surface
    describe(sym)                                        # "mirror planes x, y; C2 about z" for the log
    for name, T in partner_transforms(sym, src_c, dst_c, tol): q = apply_transform(p, T)

A plane (or an axis of order n) counts when the surface maps onto itself: area-uniform samples of the surface, mirrored
(rotated), all lie within `tol` (1e-3 of the diagonal) of the surface - the 99th percentile of their distance to it
(src.closest_point_field.SurfaceField, the exact closest point), so a few samples on a mesh's facets do not decide.
The candidates go through the area centroid of the surface: the world axes (CAD parts are mostly axis-aligned), the
principal axes, the most common directions of the facets' normals (area-weighted means) as normals of planes, refined
by a few ICP steps; the same directions and the found planes' normals as rotation axes. Each candidate is first tried
on 128 samples, so the many that fail cost little.

Why exact copies: the feature phase moves a tool fitted on one piece onto a congruent piece by the pieces' centroids and
principal axes (feature_stage.transfer_candidates). On a meshed piece those carry float noise. Measured 2026-10-29 on
ABC part 00009283_..._step_001: four mirror-image corner roundings. The copy onto the fourth had box half extents
25.400009 / 25.400002 / 6.3499999 instead of 25.4 / 25.4 / 6.35, and the cut left a 0.00094 mm^3 sliver that the
one-solid check refused. The plate's mirror planes map the corner exactly.

A mirror is improper (det -1). A placed primitive is mirrored as x' = M x + t with the frame M A D, D flipping the local
x axis: boxes, and the full cylinder, cone, sphere and torus, are symmetric under that flip. A sweep u about the local
z becomes [pi - u1, pi - u0]. A composite's parts are mirrored one by one.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

TOL_REL = 1e-3                      # a symmetry holds within this share of the diagonal
SAMPLES = 4000
QUANTILE = 99.0
ORDERS = (12, 8, 6, 5, 4, 3, 2)     # rotation orders tried about an axis, the highest first
NORMAL_DIRECTIONS = 6               # the most common facet normal directions tried as plane normals
SAME_DIRECTION = math.cos(math.radians(1.0))


class _Surface:
    """Exact closest points on the mesh: libigl's AABB tree when installed (as src.local_grids.wall_thickness; ~100x
    faster than SurfaceField's queries), else src.closest_point_field.SurfaceField."""
    def __init__(self, V, F, field=None):
        self.V, self.F = np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64)
        self.tree, self.field = None, field
        try:
            import igl
            self.tree = igl.AABB()
            self.tree.init(self.V, self.F)
        except Exception:
            self.tree = None
            if self.field is None:
                from src.closest_point_field import SurfaceField
                self.field = SurfaceField(self.V, self.F)

    def closest(self, P) -> tuple:
        """(distance (n,), closest points (n, 3))."""
        P = np.ascontiguousarray(P, np.float64)
        if self.tree is not None:
            sq, _, C = self.tree.squared_distance(self.V, self.F, P)
            return np.sqrt(np.maximum(np.asarray(sq, float), 0.0)), np.asarray(C, float)
        q = self.field.query(P)
        return np.abs(np.asarray(q.distance, float)), np.asarray(q.closest, float)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _canon(v) -> np.ndarray:
    v = _unit(v)
    return v if v[int(np.argmax(np.abs(v)))] > 0 else -v


def reflection(normal, point) -> Tuple[np.ndarray, np.ndarray]:
    """(M, t) of the mirror at the plane through `point` with `normal`: x -> M x + t."""
    n = _unit(normal)
    M = np.eye(3) - 2.0 * np.outer(n, n)
    return M, 2.0 * float(n @ np.asarray(point, float)) * n


def rotation(axis, point, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """(R, t) of the turn by `angle` about the line through `point` along `axis`: x -> R x + t."""
    a = _unit(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    p = np.asarray(point, float)
    return R, p - R @ p


def _samples(V: np.ndarray, F: np.ndarray, n: int, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Area-uniform surface samples, the area centroid, the total area."""
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    total = float(area.sum())
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=n, p=area / max(total, 1e-30))
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    pts = tri[idx, 0] * (1 - s)[:, None] + tri[idx, 1] * (s * (1 - r2))[:, None] + tri[idx, 2] * (s * r2)[:, None]
    centroid = (tri.mean(axis=1) * area[:, None]).sum(axis=0) / max(total, 1e-30)
    return pts, centroid, total


def _normal_directions(V: np.ndarray, F: np.ndarray, k: int) -> List[np.ndarray]:
    """The k facet-normal directions (up to sign) that carry the most area, each the area-weighted mean of its group
    (the large flat faces dominate it; the first facet of a group may be a rounded corner's, 0.9 deg off: measured).
    Vectorised: the normals binned at 0.01, the 60 heaviest bins merged within 1 degree."""
    tri = V[F]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cr, axis=1)
    keep = area > 0
    nrm, w = cr[keep] / area[keep, None], area[keep]
    sign = np.sign(nrm[np.arange(len(nrm)), np.argmax(np.abs(nrm), axis=1)])
    nrm = nrm * sign[:, None]
    keys, inv = np.unique(np.round(nrm * 100).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    bin_area = np.bincount(inv, weights=w)
    bin_sum = np.stack([np.bincount(inv, weights=w * nrm[:, j]) for j in range(3)], axis=1)
    groups: List[List] = []
    for b in np.argsort(-bin_area)[:60]:
        n = _canon(bin_sum[b])
        for g in groups:
            if abs(float(n @ g[2])) >= SAME_DIRECTION:
                g[0] += bin_area[b]
                g[1] = g[1] + bin_sum[b] * (1.0 if float(n @ g[2]) > 0 else -1.0)
                break
        else:
            groups.append([bin_area[b], bin_sum[b].copy(), n])
    groups.sort(key=lambda g: -g[0])
    return [_canon(g[1]) for g in groups[:k]]


def _refine_plane(field, pts: np.ndarray, normal: np.ndarray, point: np.ndarray, steps: int = 4):
    """A few ICP steps of a mirror plane: every sample's mirror image is matched to its closest surface point q, and
    the plane that bisects the samples and their matches is fitted: normal along the main direction of p - q (points
    near the plane carry no weight), through the mean of (p + q) / 2."""
    n, c = _unit(normal), np.asarray(point, float)
    for _ in range(steps):
        M, t = reflection(n, c)
        q = field.closest(pts @ M.T + t)[1]                     # the matches of the mirrored samples
        d = pts - q
        _, vecs = np.linalg.eigh(d.T @ d)
        m = vecs[:, -1]
        n = _canon(m if float(m @ n) > 0 else -m)
        c = 0.5 * (pts + q).mean(axis=0)
    return n, c


def detect(V, F, tol_rel: float = TOL_REL, samples: int = SAMPLES, field=None, quick: int = 128) -> Dict:
    """The mirror planes and rotation axes of the closed surface (V, F): dict(centre, diag, tol, planes=[dict(normal,
    point, p99)], axes=[dict(axis, point, order, p99)]). `field`: its SurfaceField when the caller has one. Every
    candidate is first tried on `quick` samples (3 tol), mirror planes are refined by ICP before the full test."""
    V, F = np.asarray(V, float), np.asarray(F, np.int64)
    field = _Surface(V, F, field)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = tol_rel * diag
    pts, c, _ = _samples(V, F, samples)
    few = pts[:quick]
    X = pts - c
    _, vecs = np.linalg.eigh(X.T @ X)
    principal = [_canon(vecs[:, i]) for i in range(3)]
    world = [np.eye(3)[i] for i in range(3)]

    def p99(mapped: np.ndarray) -> float:
        return float(np.percentile(field.closest(mapped)[0], QUANTILE))

    def distinct(vs):
        out: List[np.ndarray] = []
        for v in vs:
            if not any(abs(float(v @ m)) >= SAME_DIRECTION for m in out):
                out.append(v)
        return out

    planes = []
    for n in distinct(world + principal + _normal_directions(V, F, NORMAL_DIRECTIONS)):
        M, t = reflection(n, c)
        e0 = p99(few @ M.T + t)
        if e0 > 30 * tol:                                       # far off: not even worth refining
            continue
        n2, c2 = (n, c) if e0 <= 0.5 * tol else _refine_plane(field, pts[: 4 * quick], n, c)
        if any(abs(float(n2 @ np.asarray(q["normal"]))) >= SAME_DIRECTION for q in planes):
            continue
        M, t = reflection(n2, c2)
        if p99(few @ M.T + t) > 3 * tol:
            continue
        e = p99(pts @ M.T + t)
        if e <= tol:
            planes.append(dict(normal=n2.tolist(), point=c2.tolist(), p99=e))
    axes = []
    candidates = distinct(world + principal + [np.asarray(q["normal"]) for q in planes])
    for a in candidates:
        for order in ORDERS:
            R, t = rotation(a, c, 2 * math.pi / order)
            if p99(few @ R.T + t) > 3 * tol:
                continue
            e = p99(pts @ R.T + t)
            if e <= tol:
                axes.append(dict(axis=a.tolist(), point=c.tolist(), order=order, p99=e))
                break
    return dict(centre=c.tolist(), diag=diag, tol=tol, planes=planes, axes=axes)


def _axis_name(v) -> str:
    v = _canon(v)
    for i, name in enumerate("xyz"):
        if abs(v[i]) >= SAME_DIRECTION:
            return name
    return "(" + ", ".join(f"{x:.2f}" for x in v) + ")"


def describe(sym: Optional[Dict]) -> str:
    """The symmetry as text for the log: "mirror planes x, y; C2 about z" or "none"."""
    if not sym or not (sym.get("planes") or sym.get("axes")):
        return "none"
    parts = []
    if sym.get("planes"):
        parts.append("mirror plane" + ("s " if len(sym["planes"]) > 1 else " ") +
                     ", ".join(_axis_name(p["normal"]) for p in sym["planes"]))
    if sym.get("axes"):
        parts.append(", ".join(f"C{a['order']} about {_axis_name(a['axis'])}" for a in sym["axes"]))
    return "; ".join(parts)


def transforms(sym: Dict) -> List[Tuple[str, np.ndarray, np.ndarray, bool]]:
    """Every symmetry as (name, R, t, improper): the mirrors, the turns k 2 pi / n, and the products of two mirrors."""
    out = []
    mirrors = []
    for p in sym.get("planes") or []:
        M, t = reflection(p["normal"], p["point"])
        name = f"mirror {_axis_name(p['normal'])}"
        mirrors.append((name, M, t))
        out.append((name, M, t, True))
    for a in sym.get("axes") or []:
        for k in range(1, int(a["order"])):
            R, t = rotation(a["axis"], a["point"], 2 * math.pi * k / a["order"])
            out.append((f"C{a['order']}^{k} about {_axis_name(a['axis'])}", R, t, False))
    for i in range(len(mirrors)):
        for j in range(i + 1, len(mirrors)):
            (n1, M1, t1), (n2, M2, t2) = mirrors[i], mirrors[j]
            out.append((f"{n1} + {n2}", M2 @ M1, M2 @ t1 + t2, False))
    return out


def partner_transforms(sym: Optional[Dict], src_centre, dst_centre, tol: float) -> List[Tuple[str, tuple]]:
    """The symmetries that map the source centre onto the target centre within `tol`: [(name, (R, t, improper))]."""
    if not sym:
        return []
    s, d = np.asarray(src_centre, float), np.asarray(dst_centre, float)
    return [(name, (R, t, imp)) for name, R, t, imp in transforms(sym) if float(np.linalg.norm(R @ s + t - d)) <= tol]


def mirror_primitive(p, normal, point):
    """The placed primitive (reco_engine.result_schema.Primitive) mirrored at the plane: centre M c + t, frame M A D
    (D flips the local x axis, under which every kind is symmetric), a sweep u -> [pi - u1, pi - u0]; composites part by
    part."""
    from reco_engine.result_to_program import quaternion_of, rotation_matrix
    M, t = reflection(normal, point)
    q = copy.copy(p)
    q.center_mm = tuple(float(x) for x in M @ np.asarray(p.center_mm, float) + t)
    q.quaternion = quaternion_of(M @ rotation_matrix(p) @ np.diag([-1.0, 1.0, 1.0]))
    if p.u is not None:
        q.u = (math.pi - float(p.u[1]), math.pi - float(p.u[0]))
    q.parts = [mirror_primitive(part, normal, point) for part in (p.parts or [])]
    q.alternatives = []
    return q


def apply_transform(p, T: tuple, normal=None, point=None):
    """The primitive under a symmetry (R, t, improper): a proper one by result_to_program.transform_primitive, a mirror
    by mirror_primitive (its plane recovered from R when not given)."""
    from reco_engine.result_to_program import transform_primitive
    R, t, improper = T
    if not improper:
        return transform_primitive(copy.deepcopy(p), R, t)
    if normal is None:                                          # M = I - 2 n n^T: n is the eigenvector of -1
        w, v = np.linalg.eigh((np.asarray(R) + np.asarray(R).T) / 2)
        normal = v[:, int(np.argmin(w))]
        point = 0.5 * np.asarray(t, float)                      # t = 2 (n . p) n: a point of the plane
    return mirror_primitive(copy.deepcopy(p), normal, point)
