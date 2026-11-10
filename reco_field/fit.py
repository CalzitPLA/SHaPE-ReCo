#!/usr/bin/env python3
"""
Fit analytic primitives to a SurfaceField (the closest-surface-point vector field of an STL/STEP model), or to a cluster of its
surface points.

What the field gives a fit, per primitive with SDF f_theta:
  * SURFACE points c_i with their normals: the primitive's surface must pass through them:  f_theta(c_i) = 0
  * OFF-SURFACE points x_j with the unsigned distance d_j = |V(x_j)| to the true surface: | f_theta(x_j) | = d_j
    (pins the size and the extent of the body; only used when the fit is to the WHOLE part - for a cluster the distance to the
    whole surface is not the distance to the cluster)
A robust loss makes the fit ignore surface the primitive does not describe; a trimmed refit then drops the outliers and refits on
the rest, which removes the bias a large unmodelled region causes. `explained` is the share of the surface points within `tol` of
the fitted surface.

Starting points come from the exact normals: a box frame from a Manhattan-frame fit to the normals (training/box_alignment), a
cylinder from RANSAC on pairs of oriented points, a cone from RANSAC on triples (the normals of a cone lie on a small circle of
the unit sphere), a torus from a scan of the minor radius (offset the surface by r along its normals and the points fall on the
core circle), a sphere from the algebraic least-squares fit, a plane from the principal axes.

SEGMENTS. Everything with a circle (sphere, cylinder, cone, torus) also gets its sweep angle u about the axis (and the sphere's
latitude range, the torus' minor-circle range v) as fitted parameters: the angles of the points on the lateral surface leave an
empty arc where the body is cut open, the frame is turned so that the sector starts at u = 0, and u = (0, sweep) is refined with
the other parameters against the side faces. A segment is kept only if it explains more surface than the full body (its side
faces are supported by data); otherwise the gap is a cut, which the peeling loop handles with other operations.

    field = load_field("part.stl")
    ranked = fit_best(field)                              # every kind, best first
    cluster = subset(make_samples(field), mask)
    fit_best(None, samples=cluster, kinds=("sphere", "box", "cylinder", "plane"))

Sphere, box, cylinder, cone, torus, plane (a half space: the flat face of a cut). Deterministic (fixed seeds).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402
from training.box_alignment import fit_frame, frame_score  # noqa: E402

TWO_PI = 2.0 * np.pi
POLE = 0.2                  # a latitude limit this close to a pole (radians) is the pole itself
SEGMENT_MIN = 0.25          # a body that explains less than this is not worth a segment search (its side faces are a minor part)


@dataclass
class FitResult:
    kind: str
    params: Dict[str, float]
    primitive: sp.Primitive
    cost: float                  # share of the samples NOT explained (lower is better)
    explained: float             # share of the samples' surface points within `tol` of the fitted surface
    rms_explained: float         # rms distance of those points to the fitted surface
    tol: float
    n_params: int
    frame: Optional[np.ndarray] = None       # rotation (columns = local axes; z = the axis of a body of revolution)
    centre: Optional[np.ndarray] = None      # box / sphere / torus: its centre; cylinder / cone: the middle of the axis segment
    half: Optional[np.ndarray] = None        # box: half extents; cylinder: (radius, radius, half length); sphere: (radius,)
                                             # cone: (radius1, radius2, half height); torus: (major, minor)
    u: Optional[tuple] = None                # sweep about the axis (0, w) in radians when the body is a sector, else None
    v: Optional[tuple] = None                # sphere: latitude range; torus: minor-circle range; else None
    latitude: str = "cone"                   # sphere with v: "cone" (the parameter region) or "plane" (FreeCAD's makeSphere)


@dataclass
class Samples:
    surface: np.ndarray          # (n, 3) surface points
    normals: np.ndarray          # (n, 3) outward face normals
    near: np.ndarray             # (m, 3) points on both sides of the surface (empty for a cluster)
    near_distance: np.ndarray    # (m,)  |V| at them
    extent: float                # diagonal of the whole part's bounding box (the length scale of every tolerance)


def make_samples(field: SurfaceField, n_surface: int = 3000, n_near: int = 2000, band_fraction: float = 0.15, seed: int = 0) -> Samples:
    rng = np.random.default_rng(seed)
    area = 0.5 * np.linalg.norm(np.cross(field._b - field._a, field._c - field._a), axis=1)
    f = rng.choice(len(field.faces), size=n_surface, p=area / area.sum())
    r1, r2 = rng.random((n_surface, 1)), rng.random((n_surface, 1))
    flip = (r1 + r2) > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    surface = field._a[f] + (field._b[f] - field._a[f]) * r1 + (field._c[f] - field._a[f]) * r2
    extent = float(np.linalg.norm(field.bounds[1] - field.bounds[0]))
    near = field.sample_points(n_near, band=band_fraction * extent, seed=seed + 1)
    return Samples(surface, field.face_normals[f], near, field.query(near).distance, extent)


def subset(s: Samples, mask: np.ndarray) -> Samples:
    """The surface points selected by `mask`, without the off-surface terms."""
    return Samples(s.surface[mask], s.normals[mask], np.empty((0, 3)), np.empty(0), s.extent)


def tolerance(s: Samples) -> float:
    return 0.004 * s.extent                                                    # well above the tessellation error, below small features


# ------------------------------------------------------------------------------------------------------ helpers

def _rotation(rv):
    return Rotation.from_rotvec(rv).as_matrix()


def _residuals(prim: sp.Primitive, s: Samples, ms: Optional[np.ndarray] = None, mn: Optional[np.ndarray] = None) -> np.ndarray:
    surf = s.surface if ms is None else s.surface[ms]
    r = prim.sdf(surf)
    if len(s.near):
        near, d = (s.near, s.near_distance) if mn is None else (s.near[mn], s.near_distance[mn])
        r = np.concatenate([r, np.abs(prim.sdf(near)) - d])
    return r


def _unexplained(prim: sp.Primitive, s: Samples, tol: float) -> float:
    return float(1.0 - np.mean(np.abs(prim.sdf(s.surface)) < tol))


def _irls(build: Callable, th: np.ndarray, s: Samples, tol: float, rounds: int = 5) -> np.ndarray:
    """Tukey-biweight iteratively reweighted least squares: residuals beyond 3 tolerances get weight 0, so surface the primitive
    does not describe stops pulling the fit (a soft-L1 loss only discounts it and settles on a compromise)."""
    for _ in range(rounds):
        r = _residuals(build(th), s)
        w = np.clip(1.0 - (r / (3.0 * tol)) ** 2, 0.0, 1.0) ** 2
        sw = np.sqrt(w)
        if (w > 0).sum() < max(12, 0.05 * len(r)):
            break
        th = least_squares(lambda t: sw * _residuals(build(t), s), th, loss="linear", max_nfev=20).x
    return th


def _optimise(build: Callable, x0: np.ndarray, s: Samples):
    """A soft-L1 fit from x0, then Tukey IRLS from both x0 and that fit; the candidate explaining the most surface wins."""
    tol = tolerance(s)
    th1 = least_squares(lambda th: _residuals(build(th), s), x0, loss="soft_l1", f_scale=2 * tol, max_nfev=120).x
    best = None
    for start in (th1, x0):
        th = _irls(build, start, s, tol)
        cost = _unexplained(build(th), s, tol)
        if best is None or cost < best[1] - 1e-9:
            best = (th, cost)
    return best


def _finish(kind, names, theta, build, s, cost, frame=None, centre=None, half=None, u=None, v=None, latitude="cone") -> FitResult:
    prim = build(theta)
    d = np.abs(prim.sdf(s.surface))
    tol = tolerance(s)
    ok = d <= tol
    res = FitResult(kind, dict(zip(names, np.round(theta, 5).tolist())), prim, float(cost), float(ok.mean()),
                    float(np.sqrt((d[ok] ** 2).mean())) if ok.any() else float("nan"), tol, len(theta), frame, centre, half,
                    u, v, latitude)
    return replace(res, explained=0.0) if _disguised(kind, half, s.extent) else res


def _disguised(kind: str, half, extent: float) -> bool:
    """A round body whose radius is far beyond the part is a plane (or, as a segment, the wedge between two planes) in disguise."""
    if kind not in ("sphere", "cylinder", "cone", "torus") or half is None:
        return False
    radius = {"sphere": half[0], "cylinder": half[0], "cone": max(half[0], half[1]) if len(half) > 1 else 0.0,
              "torus": half[0] + half[1] if len(half) > 1 else 0.0}[kind]
    return bool(radius > 4 * extent)


def _failed(kind: str, s: Samples) -> FitResult:
    """No hypothesis for this kind: explains nothing, so it never wins."""
    return FitResult(kind, {}, sp.Sphere(1e-6), 1.0, 0.0, float("nan"), tolerance(s), 99)


def _axis_frame(axis) -> np.ndarray:
    """A proper rotation whose third column is `axis`."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(a, u), a], axis=1)


def _turn(R: np.ndarray, angle: float) -> np.ndarray:
    """The frame R turned about its own z axis by `angle` (a point's angle in the new frame is its old angle minus `angle`)."""
    c, s_ = np.cos(angle), np.sin(angle)
    return R @ np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _arc_range(angles: np.ndarray, min_gap: float = 0.25, ratio: float = 10.0):
    """(start, width) of the smallest arc containing every angle, or None when the angles cover the circle: the largest empty gap
    must exceed `min_gap` and `ratio` times the mean spacing inside the arc (a random sample leaves gaps of about ln(n) spacings)."""
    if len(angles) < 12:
        return None
    a = np.sort(np.mod(angles, TWO_PI))
    gaps = np.diff(np.r_[a, a[0] + TWO_PI])
    k = int(np.argmax(gaps))
    width = TWO_PI - gaps[k]
    if gaps[k] < max(min_gap, ratio * width / len(a)):
        return None
    return float(a[(k + 1) % len(a)]), float(width)


def _pca(points: np.ndarray):
    c = points.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov((points - c).T) + 1e-12 * np.eye(3))
    vecs = vecs[:, ::-1]
    if np.linalg.det(vecs) < 0:
        vecs[:, 2] *= -1
    return c, vecs


def _proper(R):
    R = np.array(R, float)
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


# --------------------------------------------------------------------------------------------------------- fitters

def _sphere_builder(latitude: str = "cone"):
    """th = [centre(3), radius]; a segment adds [rotvec(3) (axis = z), sweep u, (v0, v1)] as the layout says: 8 = u only,
    9 = v only, 10 = u and v."""
    def build(th):
        if len(th) == 4:
            return sp.Placed(sp.Sphere(abs(th[3]) + 1e-9), translation=th[:3])
        u, v, rest = sp.FULL, (-np.pi / 2, np.pi / 2), th[7:]
        if len(th) in (8, 10):
            u, rest = (0.0, float(np.clip(rest[0], 0.05, TWO_PI))), rest[1:]
        if len(th) in (9, 10):
            v0 = float(np.clip(rest[0], -np.pi / 2, np.pi / 2 - 0.05))
            v = (v0, float(np.clip(rest[1], v0 + 0.05, np.pi / 2)))
        return sp.Placed(sp.Sphere(abs(th[3]) + 1e-9, u, v, latitude=latitude), rotation=_rotation(th[4:7]), translation=th[:3])
    return build


def _sphere_ransac(s: Samples, tol: float, n_hyp: int = 400, seed: int = 0):
    """A sphere from a PAIR of oriented surface points: the two normal lines meet at the centre, and the distance from there to
    either point is the radius. Scored by the surface points within `tol` of the sphere. Returns (centre, radius, inliers) of the
    best hypothesis, or None. Robust where the algebraic fit is not: it ignores the faces that are not on the sphere."""
    P, N = s.surface, s.normals
    n = len(P)
    if n < 30:
        return None
    rng = np.random.default_rng(seed)
    i, j = rng.integers(0, n, (2, n_hyp))
    d = np.einsum("ij,ij->i", N[i], N[j])
    w = P[i] - P[j]
    denom = 1.0 - d ** 2
    ok = denom > 0.04                                                       # normals not (anti)parallel
    b, e = np.einsum("ij,ij->i", N[i], w), np.einsum("ij,ij->i", N[j], w)
    si = np.where(ok, (d * e - b) / np.where(ok, denom, 1.0), 0.0)            # closest points of the two normal lines
    sj = np.where(ok, (e - d * b) / np.where(ok, denom, 1.0), 0.0)
    centre = 0.5 * ((P[i] + si[:, None] * N[i]) + (P[j] + sj[:, None] * N[j]))
    radius = 0.5 * (np.linalg.norm(P[i] - centre, axis=1) + np.linalg.norm(P[j] - centre, axis=1))
    ok &= (radius > 3 * tol) & (radius < 4 * s.extent)
    best = None
    for k in np.nonzero(ok)[0]:
        count = int((np.abs(np.linalg.norm(P - centre[k], axis=1) - radius[k]) < tol).sum())
        if best is None or count > best[2]:
            best = (centre[k], float(radius[k]), count)
    return best if best is not None and best[2] >= 12 else None


def fit_sphere(s: Samples) -> FitResult:
    P = s.surface
    A = np.c_[2 * P, np.ones(len(P))]                                      # |p - c|^2 = r^2  =>  2 p.c + k = |p|^2, k = r^2 - |c|^2
    sol = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)[0]
    c = sol[:3]
    r = np.sqrt(max(sol[3] + c @ c, 1e-12))
    build = _sphere_builder()
    th, cost = _optimise(build, np.r_[c, r], s)
    if cost > 0.05:                                                         # part of the surface is not on the sphere: RANSAC start
        hyp = _sphere_ransac(s, tolerance(s))
        if hyp is not None:
            th2, cost2 = _optimise(build, np.r_[hyp[0], hyp[1]], s)
            if cost2 < cost:
                th, cost = th2, cost2
    res = _finish("sphere", ["cx", "cy", "cz", "radius"], th, build, s, cost, half=np.array([abs(th[3])]), centre=th[:3])
    return _sphere_segment(s, th, res)


def _sphere_axes(P: np.ndarray, N: np.ndarray, s: Samples) -> List[np.ndarray]:
    """Candidate axes of a sphere segment from the surface that is NOT on the sphere (its cut faces): the directions of the
    dominant plane normals (a latitude cap, or the normal of a longitude face), the directions perpendicular to two of them (the
    axis of a sector lies in both side planes), and the axis of a cone through the leftover (latitude faces are cones from the
    centre)."""
    rng = np.random.default_rng(0)
    remaining = np.ones(len(N), bool)
    dirs = []
    for _ in range(3):
        idx = np.nonzero(remaining)[0]
        if len(idx) < 8:
            break
        cand = idx[rng.choice(len(idx), min(60, len(idx)), replace=False)]
        best = cand[np.argmax((np.abs(N[idx] @ N[cand].T) > 0.99).sum(0))]
        m = np.abs(N @ N[best]) > 0.99
        if m.sum() < 8:
            break
        d = (np.sign(N[m] @ N[best])[:, None] * N[m]).mean(axis=0)
        dirs.append(d / np.linalg.norm(d))
        remaining &= ~m
    axes = list(dirs)
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            x = np.cross(dirs[i], dirs[j])
            if np.linalg.norm(x) > 0.3:
                axes.append(x / np.linalg.norm(x))
    if len(P) >= 20:
        sub = Samples(P, N, np.empty((0, 3)), np.empty(0), s.extent)
        axes += [R[:, 2] for R, *_ in _cone_starts(sub, tolerance(s), top=2)]
    return axes


def _sphere_segment(s: Samples, th: np.ndarray, full: FitResult) -> FitResult:
    """Longitude sector u and latitude range v of a sphere whose surface is not all explained: the axis comes from the leftover's
    cut faces (`_sphere_axes`), the ranges from the angles of the points on the sphere, and the variant (u, v, both; latitude limit
    a cone or FreeCAD's planes) that explains most surface is refined. Kept only if it beats the full sphere."""
    if full.explained > 0.995 or full.explained < SEGMENT_MIN:
        return full
    tol = tolerance(s)
    c, r = th[:3], abs(th[3])
    dist = np.linalg.norm(s.surface - c, axis=1)
    on = np.abs(dist - r) < tol
    left = ~on & (dist < r + tol)
    if on.sum() < 12 or left.sum() < 12:
        return full
    best = None                                                                # (explained, theta, latitude)
    for axis in _sphere_axes(s.surface[left], s.normals[left], s):
        R = _axis_frame(axis)
        local = (s.surface[on] - c) @ R
        rho = np.hypot(local[:, 0], local[:, 1])
        arc = _arc_range(np.arctan2(local[rho > 3 * tol, 1], local[rho > 3 * tol, 0]))
        lat = np.arcsin(np.clip(local[:, 2] / r, -1.0, 1.0))
        margin = 0.15
        vseg = lat.min() > -np.pi / 2 + margin or lat.max() < np.pi / 2 - margin
        if arc is None and not vseg:
            continue
        R2 = _turn(R, arc[0]) if arc is not None else R
        base = np.r_[c, r, Rotation.from_matrix(R2).as_rotvec()]
        tail_u = [arc[1]] if arc is not None else []
        tail_v = [lat.min(), lat.max()] if vseg else []
        for latitude in (("cone", "plane") if vseg else ("cone",)):
            trial = np.r_[base, tail_u, tail_v]
            e = float(np.mean(np.abs(_sphere_builder(latitude)(trial).sdf(s.surface)) < tol))
            if best is None or e > best[0] + 1e-9:
                best = (e, trial, latitude)
    if best is None or best[0] < full.explained + 0.005:
        return full
    _, x0, latitude = best
    build = _sphere_builder(latitude)
    th2, cost = _optimise(build, x0, s)
    names = ["cx", "cy", "cz", "radius", "rx", "ry", "rz"] + (["u_sweep"] if len(th2) in (8, 10) else []) + (["v0", "v1"] if len(th2) in (9, 10) else [])
    u = (0.0, float(th2[7])) if len(th2) in (8, 10) else None
    v = None
    if len(th2) in (9, 10):
        th2 = th2.copy()
        th2[-2] = -np.pi / 2 if th2[-2] < -np.pi / 2 + POLE else th2[-2]        # a pole end is unobservable: snap it to the pole
        th2[-1] = np.pi / 2 if th2[-1] > np.pi / 2 - POLE else th2[-1]
        v0 = float(np.clip(th2[-2], -np.pi / 2, np.pi / 2 - 0.05))
        v = (v0, float(np.clip(th2[-1], v0 + 0.05, np.pi / 2)))
    seg = _finish("sphere", names, th2, build, s, cost, frame=_rotation(th2[4:7]), centre=th2[:3], half=np.array([abs(th2[3])]),
                  u=u, v=v, latitude=latitude)
    return seg if seg.explained >= full.explained + 0.005 else full


def fit_plane(s: Samples) -> FitResult:
    """The half space on the -normal side of the plane through the points, `normal` the mean outward normal."""
    c, vecs = _pca(s.surface)
    n = vecs[:, 2]
    if n @ s.normals.mean(axis=0) < 0:
        n = -n
    build = lambda th: sp.HalfSpace(th[:3] / np.linalg.norm(th[:3]), th[3])                  # noqa: E731
    th, cost = _optimise(build, np.r_[n, n @ c], s)
    r = _finish("plane", ["nx", "ny", "nz", "offset"], th, build, s, cost)
    # a plane fits any planar patch; it only counts if the normals agree with it
    agree = float(np.mean(s.normals @ (th[:3] / np.linalg.norm(th[:3])) > 0.98))
    return replace(r, explained=min(r.explained, agree))


def _box_from(s: Samples, R):
    c = s.surface.mean(axis=0)
    local = (s.surface - c) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    return c + R @ ((lo + hi) / 2), (hi - lo) / 2


def _plane_interval(local: np.ndarray, normals: np.ndarray, R, axis: int, tol: float):
    """(lo, hi) along `axis`: the pair of opposite faces (normals along -axis / +axis) with the most supporting surface points -
    a start that survives a union (a second box on top would otherwise stretch the extent). None where a side has no face."""
    def peaks(values):
        if len(values) == 0:
            return []
        bins = np.round(values / tol).astype(int)
        u, c = np.unique(bins, return_counts=True)
        return [(u[i] * tol, c[i]) for i in np.argsort(-c)[:6]]
    pos = peaks(local[normals @ R[:, axis] > 0.98, axis])
    neg = peaks(local[normals @ R[:, axis] < -0.98, axis])
    best = None
    for hp, cp in pos:
        for ln, cn in neg:
            if hp > ln and (best is None or cp + cn > best[2]):
                best = (ln, hp, cp + cn)
    return None if best is None else (best[0], best[1])


def _box_start_from_planes(s: Samples, R):
    tol = tolerance(s)
    c = s.surface.mean(axis=0)
    local = (s.surface - c) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    for a in range(3):
        iv = _plane_interval(local, s.normals, R, a, tol)
        if iv is not None:
            lo[a], hi[a] = iv
    return c + R @ ((lo + hi) / 2), (hi - lo) / 2


def fit_box(s: Samples, hint: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None) -> FitResult:
    """`hint`, if given: (centre (3,), rotation matrix (3,3), half-extents (3,)) - an ADDITIONAL starting
    hypothesis tried alongside the usual PCA/Manhattan-frame starts below (never instead of them), e.g. from
    a trained network (training/field_point_nets.py, reco_field/field_hints.py) - `_optimise`'s own
    lowest-cost-wins logic decides whether it beats the geometric starts, exactly like any other start here."""
    build = lambda th: sp.Placed(sp.Box(np.abs(th[6:9]) + 1e-6), rotation=_rotation(th[3:6]), translation=th[:3])   # noqa: E731
    _, pca_frames = _pca(s.surface)
    starts = []
    for R0 in (pca_frames, np.eye(3)):
        R = fit_frame(s.normals, _proper(R0))                                   # a Manhattan frame of the exact normals
        starts.append((frame_score(s.normals, R), _proper(R)))
    best = None
    for _, R in sorted(starts, key=lambda t: -t[0])[:2]:
        for c, h in (_box_start_from_planes(s, R), _box_from(s, R)):
            th, cost = _optimise(build, np.r_[c, Rotation.from_matrix(R).as_rotvec(), h], s)
            if best is None or cost < best[1]:
                best = (th, cost)
    if hint is not None:
        hint_c, hint_R, hint_h = hint
        th, cost = _optimise(build, np.r_[hint_c, Rotation.from_matrix(_proper(hint_R)).as_rotvec(), hint_h], s)
        if cost < best[1]:
            best = (th, cost)
    th = best[0]
    return _finish("box", ["cx", "cy", "cz", "rx", "ry", "rz", "hx", "hy", "hz"], th, build, s, best[1],
                   frame=_rotation(th[3:6]), centre=th[:3], half=np.abs(th[6:9]))


def fit_box_bounding(s: Samples, orientation_hint: Optional[np.ndarray] = None) -> FitResult:
    """A box that COVERS every point of `s`, not the tightest fit to most of them: `_box_from` (the true
    axis-aligned-in-frame bounding box, min/max over ALL points) with NO `_optimise`/Tukey-IRLS refinement
    afterward - the deliberate difference from `fit_box`. IRLS is a ROBUST estimator: it down-weights points
    far from the current fit as outliers, which is exactly wrong for a cluster whose own "outlier" points are
    real geometry the fit should include (the concave 45-degree chamfer where an I-beam's flange meets its
    web - `docs/FRAME_FAILURE.md`'s "saddle" - gets shrunk away by IRLS, never by a plain bounding box, since
    a bounding box mathematically cannot exclude any point of `s` along its own axes). For an ADDITION
    candidate, covering more of the cluster (even with some slack past its true boundary) is what
    `reco_field.peel`'s gain gate rewards; a converged, tight fit that quietly excludes real material is not
    an improvement just because its own cost is lower.

    `orientation_hint`: a rotation matrix to use directly (e.g. from `reco_field.field_hints.FieldNetHints`)
    instead of the PCA/Manhattan-frame search below - "CNN-oriented" coverage, still no IRLS."""
    if orientation_hint is not None:
        R = _proper(orientation_hint)
    else:
        _, pca_frames = _pca(s.surface)
        starts = [(frame_score(s.normals, _proper(fit_frame(s.normals, _proper(R0)))), _proper(fit_frame(s.normals, _proper(R0))))
                 for R0 in (pca_frames, np.eye(3))]
        R = max(starts, key=lambda t: t[0])[1]
    c, h = _box_from(s, R)
    build = lambda th: sp.Placed(sp.Box(np.abs(th[6:9]) + 1e-6), rotation=_rotation(th[3:6]), translation=th[:3])   # noqa: E731
    th = np.r_[c, Rotation.from_matrix(R).as_rotvec(), h]
    cost = _unexplained(build(th), s, tolerance(s))
    return _finish("box", ["cx", "cy", "cz", "rx", "ry", "rz", "hx", "hy", "hz"], th, build, s, cost,
                   frame=R, centre=c, half=h)


def _cylinder_builder(seg_u: bool = False):
    """th = [centre (the middle of the axis segment)(3), rotvec(3), radius, half length] (+ sweep u)."""
    def build(th):
        R = _rotation(th[3:6])
        half = abs(th[7])
        u = (0.0, float(np.clip(th[8], 0.05, TWO_PI))) if seg_u else sp.FULL
        return sp.Placed(sp.Cylinder(abs(th[6]) + 1e-6, 2 * half + 1e-6, u), rotation=R, translation=th[:3] - R @ np.array([0.0, 0.0, half]))
    return build


CYLINDER_NAMES = ["cx", "cy", "cz", "rx", "ry", "rz", "radius", "half_length"]


def fit_cylinder(s: Samples) -> FitResult:
    build = _cylinder_builder()
    c0, pca = _pca(s.surface)
    _, nvecs = np.linalg.eigh(s.normals.T @ s.normals)                          # smallest eigenvector: normals perpendicular to the axis
    best = None
    for axis in (nvecs[:, 0], pca[:, 0], pca[:, 2], nvecs[:, 2]):
        R = _axis_frame(axis)
        local = (s.surface - c0) @ R
        radial = np.hypot(local[:, 0], local[:, 1])
        lo, hi = local[:, 2].min(), local[:, 2].max()
        iv = _plane_interval(local, s.normals, R, 2, tolerance(s))
        for l, h in ((lo, hi),) + ((iv,) if iv is not None else ()):
            c = c0 + R @ np.array([0.0, 0.0, (l + h) / 2])
            th, cost = _optimise(build, np.r_[c, Rotation.from_matrix(R).as_rotvec(), np.median(radial), (h - l) / 2], s)
            if best is None or cost < best[1]:
                best = (th, cost)
    for R, mid, r, cnt, half in _cylinder_ransac(s, tolerance(s)):
        th, cost = _optimise(build, np.r_[mid, Rotation.from_matrix(_proper(R)).as_rotvec(), r, half], s)
        if best is None or cost < best[1]:
            best = (th, cost)
    th = best[0]
    full = _finish("cylinder", CYLINDER_NAMES, th, build, s, best[1],
                   frame=_rotation(th[3:6]), centre=th[:3], half=np.array([abs(th[6]), abs(th[6]), abs(th[7])]))
    return _segment("cylinder", th, s, full, lambda su, sv: _cylinder_builder(su), CYLINDER_NAMES)


def _cylinder_ransac(s: Samples, tol: float, n_hyp: int = 300, top: int = 3, seed: int = 0):
    """Cylinder hypotheses from PAIRS of oriented surface points: the normals of a cylinder's lateral surface are perpendicular to
    its axis, so two of them give the axis (their cross product), the centre (where the two normal lines meet in the plane
    perpendicular to it) and the radius. Scored by the number of surface points within `tol` of the lateral surface, so a leftover
    made of several cylinders (intersecting bores) still yields a start on each. Returns [(rotation R with axis = column 2,
    centre, radius, inliers)] for the best distinct hypotheses."""
    rng = np.random.default_rng(seed)
    n = len(s.surface)
    if n < 10:
        return []
    i, j = rng.integers(0, n, n_hyp), rng.integers(0, n, n_hyp)
    P, N = s.surface, s.normals
    axis = np.cross(N[i], N[j])
    norm = np.linalg.norm(axis, axis=1)
    ok = norm > 0.35
    hyps = []
    for k in np.nonzero(ok)[0]:
        a = axis[k] / norm[k]
        u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(a, u)
        B = np.stack([u, v])                                                   # basis of the plane perpendicular to the axis
        p1, p2 = B @ P[i[k]], B @ P[j[k]]
        n1, n2 = B @ N[i[k]], B @ N[j[k]]
        M = np.stack([n1, -n2], axis=1)
        if abs(np.linalg.det(M)) < 1e-6:
            continue
        t = np.linalg.solve(M, p2 - p1)                                        # p1 + t0 n1 = p2 + t1 n2
        c2 = p1 + t[0] * n1
        r = np.linalg.norm(p1 - c2)
        if r < 3 * tol or r > s.extent:
            continue
        centre3 = P[i[k]] - (P[i[k]] @ a) * a + (B.T @ c2) - (B.T @ p1) + (P[i[k]] - (P[i[k]] @ a) * a) * 0
        centre3 = B.T @ c2 + (P[i[k]] @ a) * a                                  # a point on the axis
        rel = P - centre3
        rho = np.linalg.norm(rel - np.outer(rel @ a, a), axis=1)
        inl = np.abs(rho - r) < tol
        hyps.append((int(inl.sum()), a, centre3, r, inl))
    hyps.sort(key=lambda h: -h[0])
    out = []
    for cnt, a, c3, r, inl in hyps:
        if cnt < 12:
            break
        if any(abs(a @ o[0][:, 2]) > 0.98 and np.linalg.norm(np.cross(c3 - o[1], a)) < 2 * tol for o in out):        # same cylinder
            continue
        t = (P[inl] - c3) @ a
        mid = c3 + a * (t.min() + t.max()) / 2
        u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        out.append((np.stack([u, np.cross(a, u), a], axis=1), mid, r, cnt, (t.max() - t.min()) / 2))
        if len(out) >= top:
            break
    return out


# ------------------------------------------------------------------------------------------------------- cone

def _cone_builder(seg_u: bool = False):
    """th = [centre (the middle of the axis segment)(3), rotvec(3), radius1 (at the bottom, z = -half), radius2 (top), half height]
    (+ sweep u)."""
    def build(th):
        R = _rotation(th[3:6])
        half = abs(th[8])
        u = (0.0, float(np.clip(th[9], 0.05, TWO_PI))) if seg_u else sp.FULL
        return sp.Placed(sp.Cone(abs(th[6]) + 1e-6, abs(th[7]) + 1e-6, 2 * half + 1e-6, u), rotation=R,
                         translation=th[:3] - R @ np.array([0.0, 0.0, half]))
    return build


CONE_NAMES = ["cx", "cy", "cz", "rx", "ry", "rz", "radius1", "radius2", "half_height"]


def _cone_starts(s: Samples, tol: float, n_hyp: int = 600, top: int = 3, seed: int = 0):
    """Cone hypotheses from TRIPLES of oriented surface points. The normals of a cone make one angle with its axis, so their end
    points on the unit sphere lie on a small circle: three normals give the circle's plane, i.e. the axis a (n . a = k for all of
    them, k = sin of the half angle). Every tangent plane passes through the apex, n_i . (apex - p_i) = 0, a linear least squares.
    Scored by the points that agree in normal AND lie on the cone surface. Returns [(R with the axis = column 2 pointing from the
    apex into the body, centre of the axis segment, radius1, radius2, half height, count)]."""
    rng = np.random.default_rng(seed)
    P, N = s.surface, s.normals
    n = len(P)
    if n < 20:
        return []
    i, j, k = rng.integers(0, n, (3, n_hyp))
    a = np.cross(N[j] - N[i], N[k] - N[i])
    norm = np.linalg.norm(a, axis=1)
    keep = norm > 0.02
    a = a[keep] / norm[keep, None]
    kk = np.einsum("ij,ij->i", N[i][keep], a)
    a[kk < 0] *= -1
    kk = np.abs(kk)
    good = (kk > 0.03) & (kk < 0.9986)                                          # half angle between 1.7 and 87 degrees
    a, kk = a[good], kk[good]
    if len(a) == 0:
        return []
    agree = np.abs(N @ a.T - kk) < 0.03
    order = np.argsort(-agree.sum(0))[:40]
    out = []
    for h in order:
        inl = agree[:, h]
        if inl.sum() < max(12, 0.08 * n):
            break
        apex = np.linalg.lstsq(N[inl], np.einsum("ij,ij->i", N[inl], P[inl]), rcond=None)[0]
        rel = P - apex
        t = rel @ a[h]
        rho = np.linalg.norm(rel - np.outer(t, a[h]), axis=1)
        alpha = np.arcsin(kk[h])
        side = np.sign(np.median(t[inl]))
        on = inl & (np.sign(t) == side) & (np.abs(rho * np.cos(alpha) - np.abs(t) * np.sin(alpha)) < 1.5 * tol) & (np.abs(t) > 3 * tol)
        if on.sum() < max(12, 0.05 * n):
            continue
        b = side * a[h]                                                          # from the apex into the body
        tb = (rel[on] @ b)
        lo, hi = float(tb.min()), float(tb.max())
        if hi - lo < 3 * tol:
            continue
        if any(abs(a[h] @ o[0][:, 2]) > 0.98 and np.linalg.norm(apex - o[6]) < 3 * tol for o in out):
            continue
        tan = np.tan(alpha)
        out.append((_axis_frame(b), apex + b * (lo + hi) / 2, lo * tan, hi * tan, (hi - lo) / 2, int(on.sum()), apex))
        if len(out) >= top:
            break
    return [o[:6] for o in out]


def fit_cone(s: Samples) -> FitResult:
    build = _cone_builder()
    best = None
    for R, mid, r1, r2, half, _ in _cone_starts(s, tolerance(s)):
        th, cost = _optimise(build, np.r_[mid, Rotation.from_matrix(R).as_rotvec(), r1, r2, half], s)
        if best is None or cost < best[1]:
            best = (th, cost)
    if best is None:
        return _failed("cone", s)
    th = best[0]
    full = _finish("cone", CONE_NAMES, th, build, s, best[1], frame=_rotation(th[3:6]), centre=th[:3],
                   half=np.array([abs(th[6]), abs(th[7]), abs(th[8])]))
    return _segment("cone", th, s, full, lambda su, sv: _cone_builder(su), CONE_NAMES)


# ------------------------------------------------------------------------------------------------------ torus

def _torus_builder(seg_u: bool = False, seg_v: bool = False):
    """th = [centre(3), rotvec(3), major, minor] (+ sweep u) (+ minor-circle range v0, v1)."""
    def build(th):
        u, v, rest = sp.FULL, sp.FULL, th[8:]
        if seg_u:
            u, rest = (0.0, float(np.clip(rest[0], 0.05, TWO_PI))), rest[1:]
        if seg_v:
            v = (float(rest[0]), float(np.clip(rest[1], rest[0] + 0.05, rest[0] + TWO_PI)))
        return sp.Placed(sp.Torus(abs(th[6]) + 1e-6, abs(th[7]) + 1e-6, u, v), rotation=_rotation(th[3:6]), translation=th[:3])
    return build


TORUS_NAMES = ["cx", "cy", "cz", "rx", "ry", "rz", "major", "minor"]


def _circle_3d(Q: np.ndarray):
    """Least-squares circle through 3D points: (centre, axis, radius, rms distance of the points to the circle)."""
    c0 = Q.mean(axis=0)
    _, _, vt = np.linalg.svd(Q - c0, full_matrices=False)
    xy = (Q - c0) @ vt[:2].T
    A = np.c_[2 * xy, np.ones(len(xy))]
    sol = np.linalg.lstsq(A, (xy ** 2).sum(1), rcond=None)[0]
    R = np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-12))
    resid = np.hypot(np.hypot(xy[:, 0] - sol[0], xy[:, 1] - sol[1]) - R, (Q - c0) @ vt[2])
    return c0 + vt[:2].T @ sol[:2], vt[2], float(R), float(np.sqrt((resid ** 2).mean())), resid


def _torus_starts(s: Samples, tol: float, top: int = 2):
    """Torus hypotheses by a scan of the signed minor radius r: shifted by r along the surface normals, the points of a torus
    fall on its core circle, so the r that makes p - r n the best circle (planar and round) is the minor radius (its sign says
    whether the normals point out of the tube or into it, as on a groove). Two trimmed rounds against outliers. Returns
    [(centre, axis, major, minor, rms)]."""
    P, N = s.surface, s.normals
    if len(P) < 30:
        return []
    radii = np.geomspace(2 * tol, 0.5 * s.extent, 40)
    grid = np.r_[radii, -radii]
    found = []
    keep = np.ones(len(P), bool)
    for _ in range(3):
        scores = []
        for r in grid:
            Q = P[keep] - r * N[keep]
            scores.append(_circle_3d(Q)[3])
        scores = np.array(scores)
        order = np.argsort(scores)
        r = grid[order[0]]
        c, axis, R, rms, _ = _circle_3d(P[keep] - r * N[keep])
        resid = _circle_3d_resid(P - r * N, c, axis, R)
        keep = resid < max(2.5 * np.median(resid[keep]), 2 * tol)
        if keep.sum() < 20:
            break
        found.append((rms, c, axis, R, abs(r)))
        # the best of the other sign of r (a cutter versus a solid)
        other = np.argsort(scores + np.where(np.sign(grid) == np.sign(r), 1e9, 0.0))[0]
        ro = grid[other]
        co, ao, Ro, rmso, _ = _circle_3d(P[keep] - ro * N[keep])
        found.append((rmso, co, ao, Ro, abs(ro)))
    found.sort(key=lambda f: f[0])
    out = []
    for rms, c, axis, R, r in found:
        if R < 2 * tol or any(abs(axis @ o[1]) > 0.98 and abs(R - o[2]) < 3 * tol and abs(r - o[3]) < 3 * tol for o in out):
            continue
        out.append((c, axis, R, r, rms))
        if len(out) >= top:
            break
    return out


def _circle_3d_resid(Q, c, axis, R):
    rel = Q - c
    z = rel @ axis
    return np.hypot(np.linalg.norm(rel - np.outer(z, axis), axis=1) - R, z)


def fit_torus(s: Samples) -> FitResult:
    build = _torus_builder()
    best = None
    for c, axis, R, r, _ in _torus_starts(s, tolerance(s)):
        th, cost = _optimise(build, np.r_[c, Rotation.from_matrix(_axis_frame(axis)).as_rotvec(), R, r], s)
        if best is None or cost < best[1]:
            best = (th, cost)
    if best is None:
        return _failed("torus", s)
    th = best[0]
    full = _finish("torus", TORUS_NAMES, th, build, s, best[1], frame=_rotation(th[3:6]), centre=th[:3],
                   half=np.array([abs(th[6]), abs(th[7])]))
    return _segment("torus", th, s, full, lambda su, sv: _torus_builder(su, sv), TORUS_NAMES)


# ---------------------------------------------------------------------------------------------- segments (u, v)

def _lateral(kind: str, local: np.ndarray, th: np.ndarray, tol: float) -> np.ndarray:
    """The points (in the body's local frame) that lie on the lateral surface of the full body of revolution."""
    rho = np.hypot(local[:, 0], local[:, 1])
    if kind == "cylinder":
        return np.abs(rho - abs(th[6])) < tol
    if kind == "torus":
        return np.abs(np.hypot(rho - abs(th[6]), local[:, 2]) - abs(th[7])) < tol
    half = abs(th[8])
    slope = (abs(th[7]) - abs(th[6])) / (2 * half)
    radius = 0.5 * (abs(th[6]) + abs(th[7])) + slope * local[:, 2]
    return (np.abs(rho - radius) / np.sqrt(1 + slope ** 2) < tol) & (np.abs(local[:, 2]) < half + tol)


def _segment(kind: str, th: np.ndarray, s: Samples, full: FitResult, factory, names) -> FitResult:
    """Sweep u (and for a torus the minor-circle range v) of a fitted body of revolution. The angles of the points on its lateral
    surface about the axis leave an empty arc where the body is cut open; the frame is turned so that the sector starts at
    u = 0, and u = (0, sweep) is refined together with the other parameters against the side faces. Returns the segment if it
    explains at least half a percent more surface than the full body, else `full`."""
    if full.explained < SEGMENT_MIN:
        return full
    tol = tolerance(s)
    R = _rotation(th[3:6])
    local = (s.surface - th[:3]) @ R
    rho = np.hypot(local[:, 0], local[:, 1])
    lat = _lateral(kind, local, th, tol) & (rho > 3 * tol)
    if lat.sum() < 12:
        return full
    arc = _arc_range(np.arctan2(local[lat, 1], local[lat, 0]))
    varc = _arc_range(np.arctan2(local[lat, 2], rho[lat] - abs(th[6]))) if kind == "torus" else None
    if arc is None and varc is None:
        return full
    x0 = th.copy()
    tail = []
    if arc is not None:
        x0[3:6] = Rotation.from_matrix(_turn(R, arc[0])).as_rotvec()
        tail.append(arc[1])
    if varc is not None:
        tail += [varc[0], varc[0] + varc[1]]
    build = factory(arc is not None, varc is not None)
    th2, cost = _optimise(build, np.r_[x0, tail], s)
    extra = (["u_sweep"] if arc is not None else []) + (["v0", "v1"] if varc is not None else [])
    R2 = _rotation(th2[3:6])
    n = len(th)
    u = (0.0, float(th2[n])) if arc is not None else None
    v = (float(th2[-2]), float(th2[-1])) if varc is not None else None
    seg = _finish(kind, list(names) + extra, th2, build, s, cost, frame=R2, centre=th2[:3], half=full.half, u=u, v=v)
    return seg if seg.explained >= full.explained + 0.005 else full


FITTERS = {"sphere": fit_sphere, "box": fit_box, "cylinder": fit_cylinder, "cone": fit_cone, "torus": fit_torus, "plane": fit_plane}
WHOLE_KINDS = ("sphere", "box", "cylinder", "cone", "torus")


def rank(results: Sequence[FitResult]) -> List[FitResult]:
    """Most surface explained first; when two explain (about) the same, the one with fewer parameters, then the lower cost."""
    priority = {"plane": 0, "sphere": 1, "cylinder": 2, "cone": 3, "torus": 4, "box": 5}
    return sorted(results, key=lambda r: (-round(r.explained, 2), r.n_params, priority[r.kind], r.cost))


FIT_ORDER = ("plane", "sphere", "cylinder", "torus", "cone", "box")            # fewest parameters first (the tie-break of `rank`)


def fit_best(field: Optional[SurfaceField], kinds=None, samples: Optional[Samples] = None, stop_at: float = 0.995) -> List[FitResult]:
    """Fit the kinds in `kinds` (default: sphere, box, cylinder, cone, torus) to the whole part (`field`) or to `samples`, best
    first. The kinds are tried with the cheapest description first, and the search stops at a fit that explains `stop_at` of the
    surface: later kinds have at least as many parameters, so they could only tie (`stop_at` = 1.0 fits them all)."""
    s = samples if samples is not None else make_samples(field)
    wanted = set(kinds or WHOLE_KINDS)
    results = []
    for kind in FIT_ORDER:
        if kind in wanted:
            results.append(FITTERS[kind](s))
            if results[-1].explained >= stop_at:
                break
    return rank(results)
