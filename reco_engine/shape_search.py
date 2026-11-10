"""The dumb shape search (the user's rule 2026-10-16: "a mode which tries each base shape and searches for a good fit;
pretty stupid; graded by the percent of exact surface coverage"): `ResidualConfig.mode = "search"`.

For one residual component the loop hands over, the input's exact surface samples near the component are collected
(the samples whose voxel lies in the component dilated by one voxel). Then, for EVERY base shape (box, sphere, cylinder,
cone, torus), initial guesses are built from those points alone - their centroid, principal axes and the world axes,
extents along the axis, radii from the radial distances - and each guess is hill-climbed by random perturbation of its
centre, axis and dimensions. The one grade is the COVERAGE: the share of the nearby surface samples that lie on the
shape's surface (|sdf| <= tolerance, tolerance = `search_tolerance_share` x the part's extent, the fit's own 0.004).
No network, no RANSAC, no priors: the search knows nothing but the points and the five shapes.

The loop takes the result like any candidate list (`best_candidate`: the dict `_best_candidate` returns - best, tried,
why): every kind's best is judged by the loop's own material checks (`residual_loop._evaluate`: net voxel gain, void
purity, the sign check, the severing checks) and the passing one with the highest coverage wins, by gain among equal
coverages. The sources of such a primitive say `search:coverage`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from reco_engine.result_schema import KINDS, Primitive, StageSource

SOURCE = "search:coverage"
DEFAULT_TOLERANCE_SHARE = 0.004          # x the part's extent: reco_field.fit.tolerance
MIN_POINTS = 40                          # fewer nearby samples: the search says so and gives nothing


# ------------------------------------------------------------------------------------------------ geometry helpers

def frame_with_z(axis: np.ndarray) -> np.ndarray:
    """A rotation whose third column is `axis` (start_shape._frame)."""
    z = np.asarray(axis, float)
    n = np.linalg.norm(z)
    z = z / n if n > 0 else np.array([0.0, 0.0, 1.0])
    u = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(z, u), z], axis=1)


def _quat(R: np.ndarray) -> Tuple[float, float, float, float]:
    return tuple(float(x) for x in Rotation.from_matrix(R).as_quat())


def _axes_of(points: np.ndarray) -> List[np.ndarray]:
    """The principal axes of the points (largest spread first) and the world axes: six candidate directions."""
    c = points.mean(axis=0)
    q = points - c
    cov = q.T @ q / max(1, len(q))
    w, v = np.linalg.eigh(cov)
    pca = [v[:, i] for i in np.argsort(w)[::-1]]
    return pca + [np.eye(3)[i] for i in range(3)]


def _extent_along(points: np.ndarray, c: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    t = (points - c) @ z
    return float(t.min()), float(t.max())


def _radial(points: np.ndarray, c: np.ndarray, z: np.ndarray) -> np.ndarray:
    q = points - c
    t = q @ z
    return np.linalg.norm(q - np.outer(t, z), axis=1)


# ------------------------------------------------------------------------------------------------ the parameter vector

@dataclass
class _Shape:
    """One candidate as the search moves it: kind, centre, axis (the local z), dims in the kind's order."""
    kind: str
    center: np.ndarray
    R: np.ndarray                         # 3 x 3, local frame (z = axis)
    dims: np.ndarray                      # box: x_half y_half z_half; sphere: r; cylinder: r h; cone: r1 r2 h; torus: R r

    def primitive(self, floor: float) -> Primitive:
        d = np.maximum(self.dims, floor)
        k = self.kind
        if k == "box":
            dims = dict(x_half=d[0], y_half=d[1], z_half=d[2])
        elif k == "sphere":
            dims = dict(radius=d[0])
        elif k == "cylinder":
            dims = dict(radius=d[0], half_length=d[1])
        elif k == "cone":
            dims = dict(radius1=d[0], radius2=d[1], half_length=d[2])
        elif k == "torus":
            major, minor = max(d[0], d[1] + floor), d[1]          # the tube must not reach the axis
            dims = dict(major_radius=major, minor_radius=minor)
        else:
            raise ValueError(k)
        p = Primitive(kind=k, quaternion=_quat(self.R), center_mm=tuple(float(x) for x in self.center),
                      dims_mm={a: float(b) for a, b in dims.items()}, confidence=0.0)
        return p


def _initial(kind: str, points: np.ndarray, axes: Sequence[np.ndarray]) -> List[_Shape]:
    """The guesses of one kind from the points: one per candidate axis, the sphere one."""
    c = points.mean(axis=0)
    out = []
    if kind == "sphere":
        r = np.linalg.norm(points - c, axis=1)
        for radius in (float(np.median(r)), float(r.max())):
            out.append(_Shape("sphere", c.copy(), np.eye(3), np.array([radius])))
        return out
    for z in axes:
        R = frame_with_z(z)
        lo, hi = _extent_along(points, c, R[:, 2])
        mid = c + R[:, 2] * 0.5 * (lo + hi)
        h = 0.5 * (hi - lo)
        rad = _radial(points, mid, R[:, 2])
        if kind == "box":
            q = (points - mid) @ R
            half = np.maximum(np.abs(q).max(axis=0), 1e-9)
            out.append(_Shape("box", mid, R, half))
        elif kind == "cylinder":
            for radius in (float(np.median(rad)), float(rad.max())):
                out.append(_Shape("cylinder", mid, R, np.array([radius, h])))
        elif kind == "cone":
            t = (points - mid) @ R[:, 2]
            lo_t, hi_t = np.quantile(t, 0.2), np.quantile(t, 0.8)
            r1 = float(np.median(rad[t <= lo_t])) if np.any(t <= lo_t) else float(np.median(rad))
            r2 = float(np.median(rad[t >= hi_t])) if np.any(t >= hi_t) else float(np.median(rad))
            out.append(_Shape("cone", mid, R, np.array([r1, r2, h])))
        elif kind == "torus":
            major = float(np.median(rad))
            t = (points - mid) @ R[:, 2]
            minor = float(np.median(np.hypot(rad - major, t)))
            out.append(_Shape("torus", mid, R, np.array([major, max(minor, 1e-3 * major)])))
    return out


# ------------------------------------------------------------------------------------------------ the grade

def coverage(prim: Primitive, points: np.ndarray, tol: float) -> float:
    """The share of `points` on the primitive's surface: |sdf| <= tol. 0.0 for a primitive that cannot be built."""
    from reco_engine.result_to_program import primitive_to_sp
    if prim.defect() is not None or not len(points):
        return 0.0
    d = primitive_to_sp(prim).sdf(np.asarray(points, float))
    return float(np.mean(np.abs(d) <= tol))


def swallowed(prim: Primitive, points: np.ndarray, tol: float) -> float:
    """The share of `points` strictly inside the primitive (sdf < -tol). On the input's surface samples: material a
    cutter would remove, or air an add would fill - a shape found by coverage alone reaches into the part behind the
    surface it follows (the smoke run of 2026-10-16: every cut of the step and the pockets failed the void check,
    purity 0.81-0.96), so the search's score is coverage - swallowed."""
    from reco_engine.result_to_program import primitive_to_sp
    if prim.defect() is not None or not len(points):
        return 0.0
    d = primitive_to_sp(prim).sdf(np.asarray(points, float))
    return float(np.mean(d < -tol))


def score(prim: Primitive, points: np.ndarray, tol: float, forbid: Optional[np.ndarray]) -> float:
    """coverage of `points` minus the share of `forbid` swallowed (0 without `forbid`)."""
    c = coverage(prim, points, tol)
    return c - (swallowed(prim, forbid, tol) if forbid is not None and len(forbid) else 0.0)


def _perturb(s: _Shape, rng: np.random.Generator, scale: float, step: float) -> _Shape:
    """A random neighbour: the centre moved by up to `scale x step`, the axis tilted by up to `step` rad (not the
    sphere), each dim scaled by 1 +- step."""
    center = s.center + rng.normal(0.0, scale * step, 3)
    R = s.R
    if s.kind != "sphere":
        R = Rotation.from_rotvec(rng.normal(0.0, step, 3)).as_matrix() @ s.R
    dims = s.dims * np.exp(rng.normal(0.0, step, len(s.dims)))
    return _Shape(s.kind, center, R, dims)


def refine(s: _Shape, points: np.ndarray, tol: float, evals: int, rng: np.random.Generator, scale: float,
           floor: float, forbid: Optional[np.ndarray] = None) -> Tuple[_Shape, float, int]:
    """Hill climbing on the score (coverage - swallowed): `evals` random neighbours, the step shrinks after 8 misses in
    a row. Returns (shape, score, evaluations made)."""
    best, best_cov = s, score(s.primitive(floor), points, tol, forbid)
    step, misses, made = 0.15, 0, 0
    for _ in range(max(0, evals)):
        cand = _perturb(best, rng, scale, step)
        made += 1
        cov = score(cand.primitive(floor), points, tol, forbid)
        if cov > best_cov:
            best, best_cov, misses = cand, cov, 0
        else:
            misses += 1
            if misses >= 8:
                step, misses = max(step * 0.6, 0.005), 0
    return best, best_cov, made


@dataclass
class Found:
    primitive: Primitive
    coverage: float                      # the grade: the share of the nearby surface samples on the shape
    swallowed: float                     # the share of the part's surface samples strictly inside it (0 wanted)
    evaluations: int

    @property
    def score(self) -> float:
        return self.coverage - self.swallowed


def search(points: np.ndarray, extent: float, kinds: Sequence[str] = KINDS, evals_per_kind: int = 120,
           tolerance_share: float = DEFAULT_TOLERANCE_SHARE, max_points: int = 3000, seed: int = 0,
           keep_per_kind: int = 2, forbid: Optional[np.ndarray] = None, max_forbid: int = 4000,
           seed_points: Optional[np.ndarray] = None) -> List[Found]:
    """Every kind's best fit to `points`, best score first (coverage - the share of `forbid` swallowed; `forbid`: the
    part's surface samples, None = coverage alone). `extent`: the part's extent (the tolerance's scale). The points
    are subsampled to `max_points`, `forbid` to `max_forbid`; the `keep_per_kind` best initial guesses of a kind are
    refined. `seed_points`: a second source of initial guesses (the component's own voxel centres: a box over the
    excess voxels swallows nothing by construction - the step part's cut box from the surface points alone stayed in a
    local optimum, 46 % covered, 14.5 % swallowed, 2026-10-16)."""
    pts = np.asarray(points, float)
    rng = np.random.default_rng(seed)
    if len(pts) > max_points:
        pts = pts[rng.choice(len(pts), max_points, replace=False)]
    seeds = None
    if seed_points is not None and len(seed_points) >= 4:
        seeds = np.asarray(seed_points, float)
        if len(seeds) > max_points:
            seeds = seeds[rng.choice(len(seeds), max_points, replace=False)]
    if forbid is not None:
        forbid = np.asarray(forbid, float)
        if len(forbid) > max_forbid:
            forbid = forbid[rng.choice(len(forbid), max_forbid, replace=False)]
    tol = tolerance_share * float(extent)
    floor = 1e-3 * float(extent)
    axes = _axes_of(pts)
    span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or float(extent)
    out = []
    for kind in kinds:
        guesses = _initial(kind, pts, axes)
        if seeds is not None:
            guesses += _initial(kind, seeds, _axes_of(seeds))
        graded = sorted(((score(g.primitive(floor), pts, tol, forbid), i, g) for i, g in enumerate(guesses)),
                        key=lambda t: (-t[0], t[1]))
        best: Optional[Tuple[_Shape, float]] = None
        made = len(guesses)
        for cov0, _, g in graded[:keep_per_kind]:
            s, sc, n = refine(g, pts, tol, evals_per_kind, rng, 0.1 * span, floor, forbid)
            made += n
            if best is None or sc > best[1]:
                best = (s, sc)
        if best is None:
            continue
        p = best[0].primitive(floor)
        cov = coverage(p, pts, tol)
        sw = swallowed(p, forbid, tol) if forbid is not None and len(forbid) else 0.0
        p.label = (f"search {kind}: {cov:.0%} of {len(pts)} nearby surface samples on it"
                   + (f", {sw:.1%} of the part's surface swallowed" if sw > 0 else ""))
        p.confidence = float(cov)
        p.sources = [StageSource(st, SOURCE, float(cov), detail=dict(coverage=cov, swallowed=sw, points=int(len(pts)),
                                                                    evaluations=made))
                     for st in ("type", "rotation", "dimensions", "position")]
        out.append(Found(p, float(cov), float(sw), made))
    out.sort(key=lambda f: -f.score)
    return out


# ------------------------------------------------------------------------------------------------ the loop's hook

_SAMPLES: Dict[int, Tuple[object, np.ndarray, float]] = {}


def input_surface_samples(field, n: int = 30000) -> Tuple[np.ndarray, float]:
    """(points, extent): the input's exact surface samples (reco_field.fit.make_samples), cached per field."""
    key = id(field)
    if key not in _SAMPLES:
        from reco_field.fit import make_samples
        _SAMPLES.clear()
        s = make_samples(field, n_surface=n, n_near=0)
        _SAMPLES[key] = (field, np.asarray(s.surface, float), float(s.extent))
    return _SAMPLES[key][1], _SAMPLES[key][2]


def nearby_points(state, comp: np.ndarray, points: np.ndarray, dilate: int = 1) -> np.ndarray:
    """The surface samples whose voxel lies in the component dilated by `dilate` voxels (the samples on the
    component's own surface: what a cutter or an add must coincide with)."""
    from scipy import ndimage
    c0 = np.asarray(state.centres[0, 0, 0], float)
    idx = np.rint((points - c0) / state.pitch).astype(int)
    shape = np.array(comp.shape)
    inside = np.all((idx >= 0) & (idx < shape), axis=1)
    mask = ndimage.binary_dilation(comp, iterations=dilate) if dilate > 0 else comp
    hit = np.zeros(len(points), bool)
    sel = idx[inside]
    hit[inside] = mask[sel[:, 0], sel[:, 1], sel[:, 2]]
    return points[hit]


def best_candidate(state, comp: np.ndarray, op: str, field, cfg, purity_min: float, evaluate) -> Dict:
    """residual_loop's candidate dict (best, tried, why) for one component, from the search alone: the nearby surface
    samples, every kind's best by coverage, each judged by `evaluate(state, prim, op, purity_min)` (the loop's material
    checks); the passing one with the highest coverage wins (gain decides among equal coverages). Without a field, or
    with fewer than MIN_POINTS nearby samples, nothing is proposed and `why` says so."""
    if field is None:
        return dict(best=None, tried=[], why="search: no field (no exact surface samples)")
    points, extent = input_surface_samples(field, n=int(getattr(cfg, "search_samples", 30000)))
    near = nearby_points(state, comp, points)
    if len(near) < MIN_POINTS:
        return dict(best=None, tried=[], why=f"search: only {len(near)} surface samples near the component (< {MIN_POINTS})")
    kinds = tuple(getattr(cfg, "search_kinds", None) or KINDS)
    found = search(near, extent, kinds=kinds, evals_per_kind=int(getattr(cfg, "search_evals", 120)),
                   tolerance_share=float(getattr(cfg, "search_tolerance_share", DEFAULT_TOLERANCE_SHARE)),
                   forbid=points,                                     # the whole part's surface: nothing swallowed
                   seed_points=np.asarray(state.centres[comp], float))  # the component's voxels: guesses of their own
    tried, best = [], None
    min_gain = float(getattr(cfg, "min_gain", 0) or 0)
    min_cov = float(getattr(cfg, "search_min_coverage", 0.3) or 0.0)
    for f in found:
        ev = evaluate(state, f.primitive, op, purity_min)
        ev["op"] = op
        ev["surface"] = f.coverage
        tried.append((f.primitive, ev))
        ev["swallowed"] = f.swallowed
        if ev["ok"] and f.coverage < min_cov:                             # a passing junk shape: gains voxels, follows
            ev["ok"] = False                                              # no surface (the step part's torus cuts)
            ev["reason"] = f"only {f.coverage:.0%} of the nearby surface on it (the search needs {min_cov:.0%})"
        if ev["ok"] and ev["gain"] >= min_gain and (best is None or
                                                   (round(f.coverage, 3), ev["gain"]) > (round(best[1]["surface"], 3), best[1]["gain"])):
            best = (f.primitive, ev)
    if best is not None:
        best[0].alternatives = [p for p, _ in tried if p is not best[0]][:3]
        for a in best[0].alternatives:
            a.alternatives = []
    why = ("search: no kind fits the nearby surface" if not found
           else f"search: no kind's best passes the loop's checks with a gain and >= {min_cov:.0%} of the nearby surface")
    return dict(best=best, tried=tried, why=why)
