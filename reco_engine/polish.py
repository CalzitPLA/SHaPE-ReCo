#!/usr/bin/env python3
"""Polish: refit the program's curved primitives to the input mesh's VERTICES after the loop (2026-10-02).

Why the vertices. A CAD tessellation (an STL export, or the loop's own tessellation of a STEP) puts its vertices on the
surface and its triangles inside a convex face (outside a concave one), by up to the chord deviation. The loop fits the
triangulated surface, so every curved primitive inherits the triangles' mean sag. On three v5 spheres of the
calibration corpus calib_20261002 the loop's radius was 6.4 / 9.5 / 10.8 um short - the same as a sphere fitted to area
samples of the triangles (6.3 / 9.3 / 10.6 um) - and the IoU against the exact tree 0.998 (tier B, "fit deviation");
the vertices lie on the true sphere within 1e-6 mm, and a sphere fitted to them is exact to 0.01 um (IoU 1.0000).
Planar faces have no sag: boxes come out exact without this.

Per accepted operation with a curved surface (CURVED), in program order:
  1. its faces: the input triangles whose centroid lies within `band` of the operation's curved surface (and of its
     bounded solid) and whose normal faces the way that surface does (an add: along its normal; a cut: against it),
     nearer to it than to any other operation's curved surface. Faces, not vertices: a CAD tessellation of a cylinder
     or cone has vertices only on its rims, which lie on the cap planes too - the solid's SDF is zero there for every
     radius above the true one, so only "this vertex is on the curved surface" pins the radius;
  2. a robust least-squares fit of the distance to the curved surface alone (sphere |p| - r, cylinder rho - r, cone the
     distance to its profile line, torus the tube) at those faces' vertices, area-weighted (a vertex carries a third of
     its assigned faces' area): soft-L1, then Tukey reweighting, as user_guess.refine. Only the curved surface moves
     (FREE): sphere centre + radius; cylinder / cone the axis (shift and tilt across it) + radii, the cap planes stay
     where the loop put them (planar faces have no sag); torus centre, axis tilt and both radii. Window: `band` for
     positions and sizes, WINDOW_DEG for the tilt;
  3. kept only if the fit is CONSISTENT - its vertices' RMS distance falls to at most CONSISTENT x what it was: a
     tessellation's bias vanishes once the right surface is fitted (B -> A runs of calib/eval 2026-10-02: ratio
     ~1e-4), while a primitive that sits on faces of another true surface (a failed program) only improves a little
     and moves far (ratios 0.71-0.81, moved 0.4-1.8 mm, IoU -0.039 on aegir_n2_S_000713_L before this rule) - and if
     the area-weighted RMS of the whole program's SDF over all vertices does not rise (a guard against an
     interaction with other operations; the vertices are samples of the true surface).

polish_result(result, vertices, faces) changes the accepted operations in place and returns a PolishReport.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

CURVED = ("sphere", "cylinder", "cone", "torus")
# the free parameters per kind: local centre shift axes, local tilt axes, dimensions
FREE = {"sphere": ((0, 1, 2), (), ("radius",)),
        "cylinder": ((0, 1), (0, 1), ("radius",)),
        "cone": ((0, 1), (0, 1), ("radius1", "radius2")),
        "torus": ((0, 1, 2), (0, 1), ("major_radius", "minor_radius"))}
WINDOW_DEG = 2.0                 # the axis may tilt by at most this many degrees
BAND_DIAGONAL = 0.01             # the face band: half the median edge length, at most this share of the diagonal
FACING = 0.8                     # a face belongs to a surface only if cos(normal angle) >= this
MAX_FACES = 600_000              # a larger mesh is subsampled (deterministically, by faces)
MIN_POINTS = 12                  # fewer vertices than this (or than 2 x the free parameters): the operation stays
CONSISTENT = 0.25                # a refit is kept only if its vertices' RMS falls to at most this share of what it was
EXACT_DIAGONAL = 1e-7            # vertices' RMS below this share of the diagonal: already on its vertices, left as it is


@dataclass
class PolishedOp:
    index: int                   # the operation's index in result.operations
    op: str
    kind: str
    faces: int                   # input faces assigned to its curved surface
    points: int                  # their vertices
    rms_before: float            # area-weighted RMS distance of those vertices to its curved surface (mm)
    rms_after: float
    moved_mm: float
    turned_deg: float
    scaled: Dict[str, float]     # new / old per free dimension
    kept: bool
    reason: str = ""


@dataclass
class PolishReport:
    vertices: int
    faces: int
    band_mm: float
    rms_before: float            # area-weighted RMS of the program's SDF over all vertices (mm)
    rms_after: float
    ops: List[PolishedOp] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return sum(o.kept for o in self.ops)

    def summary(self) -> str:
        return f"{self.kept} of {len(self.ops)} curved operations refitted to the input's vertices"

    def as_dict(self) -> dict:
        d = asdict(self)
        d["kept"] = self.kept
        return d


@dataclass
class MeshArrays:
    V: np.ndarray                # vertices, exact duplicates merged (an STL triangle soup has them)
    F: np.ndarray                # faces into V
    area: np.ndarray             # per face
    normal: np.ndarray           # per face, unit
    centroid: np.ndarray         # per face
    vertex_weight: np.ndarray    # a third of the adjacent faces' area, mean 1 over the used vertices
    used: np.ndarray             # indices of the vertices a face uses
    edge: float                  # median edge length


def mesh_arrays(vertices, faces, max_faces: int = MAX_FACES, seed: int = 0) -> MeshArrays:
    V = np.asarray(vertices, float)
    F = np.asarray(faces, np.int64)
    V, inverse = np.unique(V, axis=0, return_inverse=True)
    F = inverse.reshape(-1)[F]
    if len(F) > max_faces:
        F = F[np.sort(np.random.default_rng(seed).choice(len(F), max_faces, replace=False))]
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    keep = norm > 0                                                    # degenerate triangles carry nothing
    F, tri, cross, norm = F[keep], tri[keep], cross[keep], norm[keep]
    weight = np.zeros(len(V))
    for k in range(3):
        np.add.at(weight, F[:, k], norm / 6.0)
    used = np.nonzero(weight > 0)[0]
    weight = weight / weight[used].mean() if len(used) else weight
    edges = np.linalg.norm(tri - np.roll(tri, 1, axis=1), axis=2).ravel()
    return MeshArrays(V, F, norm / 2.0, cross / norm[:, None], tri.mean(axis=1), weight, used,
                      float(np.median(edges)) if len(edges) else 0.0)


def vertex_samples(vertices, faces, max_faces: int = MAX_FACES, seed: int = 0):
    """(points, weights, median edge length) of the vertices the faces use (duplicates merged, weights mean 1)."""
    m = mesh_arrays(vertices, faces, max_faces, seed)
    return m.V[m.used], m.vertex_weight[m.used], m.edge


def _frame(prim):
    from scipy.spatial.transform import Rotation
    return np.asarray(prim.center_mm, float), Rotation.from_quat(np.asarray(prim.quaternion, float)).as_matrix()


def surface_distance(prim, points) -> np.ndarray:
    """Signed distance of `points` to the primitive's CURVED surface, unbounded (positive outside the solid): sphere
    |p| - r, cylinder rho - r, cone the distance to its profile line through (radius1, -h) and (radius2, +h), torus
    the tube. The caps and sector planes are not part of it."""
    c, R = _frame(prim)
    q = (np.asarray(points, float) - c) @ R                           # local coordinates (R's columns: local axes)
    d = prim.dims_mm
    rho = np.hypot(q[:, 0], q[:, 1])
    if prim.kind == "sphere":
        return np.linalg.norm(q, axis=1) - d["radius"]
    if prim.kind == "cylinder":
        return rho - d["radius"]
    if prim.kind == "cone":
        r1, r2, h = d["radius1"], d["radius2"], d["half_length"]
        return (rho - (r1 + (r2 - r1) * (q[:, 2] + h) / (2.0 * h))) * (2.0 * h) / math.hypot(2.0 * h, r2 - r1)
    if prim.kind == "torus":
        return np.hypot(rho - d["major_radius"], q[:, 2]) - d["minor_radius"]
    raise ValueError(prim.kind)


def _surface_normal(prim, points, h: float) -> np.ndarray:
    g = np.stack([(surface_distance(prim, points + h * e) - surface_distance(prim, points - h * e)) / (2 * h)
                  for e in np.eye(3)], axis=1)
    return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)


def _program_rms(result, points, weights) -> float:
    from reco_engine.result_to_program import program_model
    model = program_model(result)
    if model is None:
        return float("nan")
    d = model.sdf(points)
    return float(np.sqrt(np.sum(weights * d * d) / np.sum(weights)))


def _refit(prim, P, W, band: float, diagonal: float):
    """(refitted primitive, rms before, rms after, moved mm, tilted deg, {dim: new/old}): the robust least squares of
    user_guess.refine on surface_distance at the points P (weights W), over the kind's FREE parameters."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    shift_axes, tilt_axes, names = FREE[prim.kind]
    c0, R0 = _frame(prim)
    d0 = np.array([prim.dims_mm[k] for k in names], float)
    ns, nt = len(shift_axes), len(tilt_axes)

    def build(th):
        shift = np.zeros(3)
        shift[list(shift_axes)] = th[:ns]
        tilt = np.zeros(3)
        tilt[list(tilt_axes)] = th[ns:ns + nt]
        R = R0 @ Rotation.from_rotvec(tilt).as_matrix()               # tilt about the primitive's own axes
        return replace(prim, center_mm=tuple(float(x) for x in c0 + R0 @ shift),
                       quaternion=tuple(float(x) for x in Rotation.from_matrix(R).as_quat()),
                       dims_mm=dict(prim.dims_mm, **{k: float(v) for k, v in zip(names, d0 * np.exp(th[ns + nt:]))}))

    sw = np.sqrt(W)

    def rms(p) -> float:
        r = surface_distance(p, P)
        return float(np.sqrt(np.sum(W * r * r) / np.sum(W)))

    x0 = np.zeros(ns + nt + len(names))
    hi = np.r_[[band] * ns, [math.radians(WINDOW_DEG)] * nt, [math.log1p(band / max(d, 1e-12)) for d in d0]]
    scale = max(float(np.median(np.abs(surface_distance(prim, P)))), 1e-9 * diagonal)
    th = least_squares(lambda t: sw * surface_distance(build(t), P), x0, bounds=(-hi, hi), loss="soft_l1",
                       f_scale=scale, max_nfev=200).x
    for _ in range(3):                                               # Tukey: drop vertices of other surfaces
        r = surface_distance(build(th), P)
        s = max(1.4826 * float(np.median(np.abs(r))), 1e-9 * diagonal)
        w = np.clip(1.0 - (r / (4.685 * s)) ** 2, 0.0, 1.0) ** 2
        if (w > 0).sum() < max(MIN_POINTS, 2 * len(x0)):
            break
        sww = sw * np.sqrt(w)
        th = least_squares(lambda t: sww * surface_distance(build(t), P), np.clip(th, -hi * 0.999999, hi * 0.999999),
                           bounds=(-hi, hi), max_nfev=100).x
    new = build(th)
    c1, R1 = _frame(new)
    turned = math.degrees(math.acos(float(np.clip(R0[:, 2] @ R1[:, 2], -1.0, 1.0))))
    return (new, rms(prim), rms(new), float(np.linalg.norm(c1 - c0)), turned,
            {k: float(new.dims_mm[k] / prim.dims_mm[k]) for k in names if prim.dims_mm[k]})


def polish_result(result, vertices, faces, max_faces: int = MAX_FACES, seed: int = 0) -> PolishReport:
    """Refit the accepted curved operations of `result` to the mesh's vertices, in place (module docstring)."""
    from reco_engine.user_guess import shape_sdf
    m = mesh_arrays(vertices, faces, max_faces, seed)
    P, W = m.V[m.used], m.vertex_weight[m.used]
    diagonal = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0))) if len(P) else 0.0
    band = min(0.5 * m.edge, BAND_DIAGONAL * diagonal)
    ops = [(i, o) for i, o in enumerate(result.operations) if o.accepted]
    report = PolishReport(vertices=int(len(P)), faces=int(len(m.F)), band_mm=band,
                          rms_before=_program_rms(result, P, W) if ops and len(P) else float("nan"), rms_after=float("nan"))
    curved = [(i, o) for i, o in ops if o.primitive.kind in CURVED]
    if not curved or not len(P):
        report.rms_after = report.rms_before
        return report
    # the faces of each curved surface: near it (and near the bounded solid), facing its way; the nearest one wins
    near = np.full((len(curved), len(m.F)), np.inf)
    for j, (i, o) in enumerate(curved):
        d = np.abs(surface_distance(o.primitive, m.centroid))
        cand = (d <= band) & (np.abs(shape_sdf(o.primitive).sdf(m.centroid)) <= band)
        if cand.any():
            side = 1.0 if o.op == "add" else -1.0
            n = _surface_normal(o.primitive, m.centroid[cand], 1e-4 * max(band, 1e-9))
            facing = np.einsum("ij,ij->i", n, m.normal[cand]) * side >= FACING
            idx = np.nonzero(cand)[0][facing]
            near[j, idx] = d[idx]
    owner = np.where(np.isfinite(near).any(axis=0), np.argmin(near, axis=0), -1)
    current = report.rms_before
    for j, (i, o) in enumerate(curved):
        prim = o.primitive
        mine = m.F[owner == j]
        verts = np.unique(mine)
        weight = np.zeros(len(m.V))
        for k in range(3):
            np.add.at(weight, mine[:, k], m.area[owner == j] / 3.0)
        n_free = sum(len(x) for x in FREE[prim.kind])
        entry = PolishedOp(index=i, op=o.op, kind=prim.kind, faces=int(len(mine)), points=int(len(verts)),
                           rms_before=float("nan"), rms_after=float("nan"), moved_mm=0.0, turned_deg=0.0, scaled={},
                           kept=False)
        report.ops.append(entry)
        if len(verts) < max(MIN_POINTS, 2 * n_free):
            entry.reason = "too few vertices"
            continue
        w = weight[verts] / weight[verts].mean()
        r0 = surface_distance(prim, m.V[verts])
        if float(np.sqrt(np.sum(w * r0 * r0) / np.sum(w))) <= EXACT_DIAGONAL * diagonal:
            entry.reason = "already on its vertices"
            continue
        try:
            new, entry.rms_before, entry.rms_after, entry.moved_mm, entry.turned_deg, entry.scaled = \
                _refit(prim, m.V[verts], w, band, diagonal)
        except Exception as exc:                                       # a degenerate fit: the operation stays
            entry.reason = f"fit failed: {str(exc)[-120:]}"
            continue
        if not entry.rms_after <= CONSISTENT * entry.rms_before:
            entry.reason = (f"its vertices do not lie on one {prim.kind} (RMS {entry.rms_before:.3g} -> "
                            f"{entry.rms_after:.3g} mm)")
            continue
        o.primitive = new
        trial = _program_rms(result, P, W)
        if trial <= current * (1.0 + 1e-9) + 1e-15:
            entry.kept, current = True, trial
        else:
            o.primitive = prim
            entry.reason = f"the program's surface RMS would rise ({current:.3g} -> {trial:.3g} mm)"
    report.rms_after = current
    return report
