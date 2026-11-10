#!/usr/bin/env python3
"""
Straight prismatic MEMBERS of a mesh (I-beams, rods, bars, tubes, rectangular hollow sections) and closed CAVITIES,
detected from the exact mesh - not from voxels: a frame's 12 mm webs are far below the pitch of any whole-part grid
(60 mm at 128^3 on the 6.7 m Frame_4_x_6), but every section of the mesh shows them exactly.

Members (`detect_members`):
  1. candidate directions: the directions of the long mesh edges (a tessellated rod or a strut has its generators along
     its axis; two-level clusters - 0.02 deg, then 0.5 deg represented by the heaviest fine one, so a facet diagonal a
     fraction of a degree off the axis never stands for it), the dominant face-normal frame (three axes), and the cross
     products of the main face-normal clusters (the axis of a bar is normal to two side faces); deduplicated at 1 deg.
     Members at any orientation are found; nothing assumes world axes.
  2. per direction, `n_stations` plane sections (`trimesh.Trimesh.section(...).discrete`) along the part, each loop
     projected on two in-plane axes; a loop inside another loop of the same station is a HOLE of it.
  3. a track = the same outer loop (same bbox and area in the plane) over consecutive stations; >= 3 stations.
  4. extent: from the last matching station outwards while the section still COVERS the loop (sample points just inside
     the loop's material are all inside that station's section) - so a member runs on through a joint where its loop
     merges with another member's - then a bisection to the exact end, snapped to the nearest vertex plane.
  5. kept when its length is at least `min_length_ratio` x its largest section side; one member per physical run (a
     slightly oblique section of a rod is still nearly a circle: of near-parallel coincident members the one matched
     over most stations is kept).

Sections (`classify_section`): rod (circle), bar (rectangle, any in-plane angle: minimum-area rectangle), tube
(circle + concentric circular hole), rhs (rectangle + rectangular hole), ibeam (`fit_ibeam_section`), else other.

I-sections (`fit_ibeam_section`): in the loop's minimum-area-rectangle frame, for both choices of the web direction:
h, b = the rectangle; the width profile on ~400 scanlines (every line must cross ONE interval): tw = median width over
the middle 30 % of h, tf = the width step at each flange found by bisection (both flanges must agree), r = (width just
past the flange - tw) / 2; area = 2 b tf + (h - 2 tf) tw + (4 - pi) r^2 within 3 %; web centred. Confidence = scanline
IoU of the loop against the fitted I with its fillets (>= 0.97). The dims are snapped to
dataset_gen/src/din_profiles.nearest when every one of h, b, tw, tf is within 5 % (at one of the unit scales
UNIT_SCALES: the frame's STL is at 1/100 mm); otherwise the measured dims are kept and the nearest profile is recorded
as the envelope (the frame's posts: HEA 360 envelope with a 12 mm web against the standard 10 mm).

Primitives (`member_operations`): ibeam -> 3 box adds (top flange, bottom flange, web; the web reaches half-way into the
flanges, so no two faces are coplanar); rod -> cylinder add; bar -> box add; tube -> cylinder add + cylinder cut;
rhs -> box add + box cut. A cutter runs 5 % past a free end (a through hole, no coplanar cap).

Cavities (`find_cavities`): the connected surface components of the mesh; a component inside the outer one is the
shell of a closed cavity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from reco_engine.timing import timed, timed_fn  # noqa: E402

from reco_engine.result_schema import Primitive, StageSource
from reco_engine.result_to_program import quaternion_of

MODE_IBEAM = "analytic:ibeam_section"
MODE_MEMBER = "analytic:member_section"
STAGE_NAMES = ("type", "rotation", "dimensions", "position")
UNIT_SCALES = (1.0, 10.0, 100.0, 1000.0, 0.1, 25.4)      # model units -> mm tried by the profile snap
SNAP_MAX_REL = 0.05
MEMBER_KINDS = ("ibeam", "rod", "bar", "tube", "rhs", "other")


# ------------------------------------------------------------------------------------------------ 2D geometry

def _open(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, float)
    if len(L) > 1 and np.allclose(L[0], L[-1]):
        L = L[:-1]
    return L


def signed_area(L: np.ndarray) -> float:
    x, y = L[:, 0], L[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def centroid(L: np.ndarray) -> np.ndarray:
    x, y = L[:, 0], L[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    c = x * y1 - x1 * y
    a = c.sum() / 2.0
    if abs(a) < 1e-300:
        return L.mean(axis=0)
    return np.array([((x + x1) * c).sum(), ((y + y1) * c).sum()]) / (6.0 * a)


def points_inside(P: np.ndarray, loops: Sequence[np.ndarray], chunk: int = 2048) -> np.ndarray:
    """Even-odd inside test of 2D points against all loops together (a hole loop makes its inside 'outside')."""
    P = np.asarray(P, float).reshape(-1, 2)
    inside = np.zeros(len(P), bool)
    for L in loops:
        a = L
        b = np.roll(L, -1, axis=0)
        ya, yb = a[:, 1], b[:, 1]
        dy = np.where(yb == ya, 1e-300, yb - ya)
        for i in range(0, len(P), chunk):
            p = P[i:i + chunk]
            py = p[:, 1:2]
            cond = (ya[None, :] > py) != (yb[None, :] > py)
            xint = a[None, :, 0] + (py - ya[None, :]) * (b[None, :, 0] - a[None, :, 0]) / dy[None, :]
            inside[i:i + chunk] ^= ((cond & (p[:, 0:1] < xint)).sum(axis=1) % 2) == 1
    return inside


def scan_intervals(loops: Sequence[np.ndarray], y: float) -> Optional[np.ndarray]:
    """The (k, 2) x-intervals of the material on the horizontal line at height y (even-odd over all loops)."""
    xs = []
    for L in loops:
        a, b = L, np.roll(L, -1, axis=0)
        m = (a[:, 1] > y) != (b[:, 1] > y)
        if m.any():
            xs.append(a[m, 0] + (y - a[m, 1]) * (b[m, 0] - a[m, 0]) / (b[m, 1] - a[m, 1]))
    if not xs:
        return np.empty((0, 2))
    xs = np.sort(np.concatenate(xs))
    if len(xs) % 2:
        return None
    return xs.reshape(-1, 2)


def min_area_rectangle(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centre, R2, half): R2's columns are the rectangle's axes (in the loop's coordinates), half its half sides."""
    from scipy.spatial import ConvexHull
    try:
        hull = L[ConvexHull(L).vertices]
    except Exception:
        hull = L
    best = None
    for i in range(len(hull)):
        e = hull[(i + 1) % len(hull)] - hull[i]
        n = np.linalg.norm(e)
        if n < 1e-12:
            continue
        d1 = e / n
        R2 = np.array([[d1[0], -d1[1]], [d1[1], d1[0]]])
        q = hull @ R2
        lo, hi = q.min(axis=0), q.max(axis=0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0] - 1e-12 * max(area, 1.0):
            best = (area, R2, lo, hi)
    _, R2, lo, hi = best
    # canonical: the first axis the one closest to +u, both axes right-handed
    if abs(R2[0, 1]) > abs(R2[0, 0]):
        R2 = np.stack([R2[:, 1], -R2[:, 0]], axis=1)
        lo, hi = np.array([lo[1], -hi[0]]), np.array([hi[1], -lo[0]])
    if R2[0, 0] < 0:
        R2, lo, hi = -R2, -hi, -lo
    return R2 @ ((lo + hi) / 2.0), R2, (hi - lo) / 2.0


# ------------------------------------------------------------------------------------------------ section fits

@dataclass
class IBeamSection:
    h: float
    b: float
    tw: float
    tf: float
    r: float
    centre: np.ndarray               # 2D, in the member's plane coordinates
    h_dir: np.ndarray                # 2D unit vector along the depth h (the web)
    b_dir: np.ndarray                # 2D unit vector along the flange width b
    area: float
    confidence: float                # scanline IoU of the loop against the fitted I with fillets
    measured: Dict[str, float] = dc_field(default_factory=dict)
    profile: Optional[str] = None    # standard name (snapped) or the envelope's name (not snapped)
    standard: Optional[str] = None
    series: Optional[str] = None
    rel_distance: Optional[float] = None
    snapped: bool = False
    unit_scale: float = 1.0          # model units x unit_scale = mm of the profile table

    @property
    def label(self) -> str:
        if self.profile and self.snapped:
            return self.profile
        if self.profile:
            return f"I {self.h:.4g}x{self.b:.4g}x{self.tw:.3g}x{self.tf:.3g} (~{self.profile} envelope)"
        return f"I {self.h:.4g}x{self.b:.4g}x{self.tw:.3g}x{self.tf:.3g}"


@dataclass
class SectionFit:
    kind: str                        # one of MEMBER_KINDS
    centre: np.ndarray               # 2D
    axes: np.ndarray                 # 2x2, columns: the section's local x, y in the plane coordinates
    params: Dict[str, float] = dc_field(default_factory=dict)
    confidence: float = 0.0
    ibeam: Optional[IBeamSection] = None

    @property
    def label(self) -> str:
        if self.ibeam is not None:
            return self.ibeam.label
        p = self.params
        if self.kind == "rod":
            return f"rod r={p['r']:.4g}"
        if self.kind == "tube":
            return f"tube r={p['r']:.4g}/{p['r_in']:.4g}"
        if self.kind == "bar":
            return f"bar {2 * p['hx']:.4g}x{2 * p['hy']:.4g}"
        if self.kind == "rhs":
            return f"rhs {2 * p['hx']:.4g}x{2 * p['hy']:.4g} (hole {2 * p['ix']:.4g}x{2 * p['iy']:.4g})"
        return self.kind


def snap_profile(h, b, tw, tf, max_rel: float = SNAP_MAX_REL):
    """(IProfile, rel_distance, unit_scale) of the nearest DIN profile over UNIT_SCALES, or None without din_profiles."""
    try:
        from src import din_profiles as dp
    except ImportError:
        try:
            from dataset_gen.src import din_profiles as dp
        except ImportError:
            return None
    best = None
    for s in UNIT_SCALES:
        try:
            p, d = dp.nearest(h * s, b * s, tw * s, tf * s)
        except Exception:
            continue
        if best is None or d < best[1]:
            best = (p, float(d), s)
    return best


def _i_width(y, h, b, tw, tf, r):
    """Width of an I (centred at 0 in y from -h/2..h/2) at height y, with root fillets r."""
    d = h / 2.0 - np.abs(y)                     # distance from the outer flange face
    w = np.full_like(np.asarray(y, float), tw)
    w = np.where(d <= tf, b, w)
    if r > 0:
        z = d - tf                              # height above the flange's inner face
        f = (z > 0) & (z < r)
        w = np.where(f, tw + 2.0 * (r - np.sqrt(np.clip(r * r - (r - z) ** 2, 0.0, None))), w)
    return w


def fit_ibeam_section(loop: np.ndarray, holes: Sequence[np.ndarray] = (), n_lines: int = 400,
                      min_confidence: float = 0.97, snap: bool = True) -> Optional[IBeamSection]:
    """The I-section of one closed 2D loop, or None when the loop is no I (see the module docstring)."""
    loop = _open(loop)
    if len(holes) or len(loop) < 12:
        return None
    area = abs(signed_area(loop))
    c, R2, half = min_area_rectangle(loop)
    q = (loop - c) @ R2
    best = None
    for k in (0, 1):                                   # k: the rectangle axis the web (h) runs along
        X, Y = q[:, 1 - k], q[:, k]
        pts = np.stack([X, Y], axis=1)
        h, b = 2.0 * half[k], 2.0 * half[1 - k]
        if h <= 0 or b <= 0:
            continue
        ys = -h / 2 + (np.arange(n_lines) + 0.5 + 0.01234) / (n_lines + 0.1) * h

        def iv(y):
            s = scan_intervals([pts], y)
            return s if s is not None and len(s) == 1 else None
        ivs = [iv(y) for y in ys]
        if any(v is None for v in ivs):
            continue
        ivs = np.array([v[0] for v in ivs])
        w = ivs[:, 1] - ivs[:, 0]
        mid = np.abs(ys) <= 0.15 * h
        if mid.sum() < 3:
            continue
        tw = float(np.median(w[mid]))
        if not (tw < 0.6 * b) or np.ptp(w[mid]) > 0.05 * tw + 1e-9 * b:
            continue
        thr = 0.75 * b + 0.25 * tw

        def width(y):
            s = scan_intervals([pts], y)
            return float((s[:, 1] - s[:, 0]).sum()) if s is not None and len(s) else 0.0

        def step(sign):                                # distance from the outer face where the width drops below thr
            lo, hi = sign * h / 2.0 * (1 - 1e-9), 0.0      # lo: in the flange, hi: the centre (web)
            if width(sign * (h / 2.0 - 1e-6 * h)) < thr:
                return None
            for _ in range(48):
                m = (lo + hi) / 2.0
                if width(m) >= thr:
                    lo = m
                else:
                    hi = m
            return h / 2.0 - abs((lo + hi) / 2.0)
        t_bot, t_top = step(-1.0), step(1.0)
        if t_bot is None or t_top is None or abs(t_bot - t_top) > 0.03 * max(t_bot, t_top) + 1e-9 * h:
            continue
        tf = (t_bot + t_top) / 2.0
        if not (0 < tf < 0.35 * h):
            continue
        eps = 1e-4 * h
        w_root = 0.5 * (width(-h / 2 + tf + eps) + width(h / 2 - tf - eps))
        r = max(0.0, (w_root - tw) / 2.0)
        if r < 0.05 * tw:
            r = 0.0
        r = min(r, 0.5 * (b - tw), h / 2 - tf)
        a_fit = 2 * b * tf + (h - 2 * tf) * tw + (4 - math.pi) * r * r
        if abs(area - a_fit) > 0.03 * area:
            continue
        centre_off = ivs[mid].mean(axis=1)
        if np.abs(centre_off).max() > 0.02 * b:
            continue
        wf = _i_width(ys, h, b, tw, tf, r)
        lo_i, hi_i = np.maximum(ivs[:, 0], -wf / 2), np.minimum(ivs[:, 1], wf / 2)
        inter = np.clip(hi_i - lo_i, 0.0, None)
        union = w + wf - inter
        conf = float(inter.sum() / union.sum())
        if conf < min_confidence:
            continue
        h_dir, b_dir = R2[:, k], R2[:, 1 - k]
        sec = IBeamSection(h=h, b=b, tw=tw, tf=tf, r=r, centre=c.copy(), h_dir=h_dir.copy(), b_dir=b_dir.copy(),
                           area=area, confidence=conf, measured=dict(h=h, b=b, tw=tw, tf=tf, r=r))
        if best is None or conf > best.confidence:
            best = sec
    if best is None:
        return None
    if snap:
        s = snap_profile(best.h, best.b, best.tw, best.tf)
        if s is not None:
            p, d, scale = s
            best.profile, best.standard, best.series, best.rel_distance = p.name, p.standard, p.series, d
            best.unit_scale = scale
            if d < SNAP_MAX_REL:
                best.snapped = True
                best.h, best.b, best.tw, best.tf, best.r = (p.h / scale, p.b / scale, p.tw / scale, p.tf / scale,
                                                           p.r / scale)
    return best


def _circle(L: np.ndarray, tol: float = 0.06):
    """(centre, radius) when the loop is a (tessellated) circle: every vertex within tol of the circumradius."""
    if len(L) < 8:
        return None
    c = centroid(L)
    rr = np.linalg.norm(L - c, axis=1)
    r = float(rr.max())
    if r <= 0 or (r - rr.min()) > tol * r:
        return None
    a = abs(signed_area(L))
    if not (0.88 * math.pi * r * r <= a <= 1.001 * math.pi * r * r):
        return None
    return c, r


def classify_section(loop: np.ndarray, holes: Sequence[np.ndarray] = (), snap: bool = True) -> SectionFit:
    """The member kind of an outer loop with its holes (see the module docstring)."""
    loop = _open(loop)
    holes = [_open(hh) for hh in holes]
    area = abs(signed_area(loop))
    eye = np.eye(2)
    if not holes:
        circ = _circle(loop)
        if circ is not None:
            return SectionFit("rod", circ[0], eye, dict(r=circ[1]), confidence=area / (math.pi * circ[1] ** 2))
        c, R2, half = min_area_rectangle(loop)
        rect = 4.0 * half[0] * half[1]
        if rect > 0 and area / rect > 0.97:
            return SectionFit("bar", c, R2, dict(hx=half[0], hy=half[1]), confidence=area / rect)
        ib = fit_ibeam_section(loop, snap=snap)
        if ib is not None:
            ax = np.stack([ib.b_dir, ib.h_dir], axis=1)
            return SectionFit("ibeam", ib.centre, ax, dict(h=ib.h, b=ib.b, tw=ib.tw, tf=ib.tf, r=ib.r),
                              confidence=ib.confidence, ibeam=ib)
        return SectionFit("other", centroid(loop), eye, {}, confidence=0.0)
    if len(holes) == 1:
        hole = holes[0]
        co, ci = _circle(loop), _circle(hole)
        if co is not None and ci is not None and np.linalg.norm(co[0] - ci[0]) < 0.02 * co[1] and ci[1] < co[1]:
            return SectionFit("tube", co[0], eye, dict(r=co[1], r_in=ci[1]), confidence=1.0)
        c, R2, half = min_area_rectangle(loop)
        ch, R2h, halfh = min_area_rectangle(hole)
        rect, recth = 4 * half[0] * half[1], 4 * halfh[0] * halfh[1]
        ah = abs(signed_area(hole))
        parallel = abs(abs(float(R2[:, 0] @ R2h[:, 0])) - 1.0) < 1e-3 or abs(float(R2[:, 0] @ R2h[:, 0])) < 1e-3
        if rect > 0 and recth > 0 and area / rect > 0.96 and ah / recth > 0.96 and parallel \
                and np.linalg.norm(c - ch) < 0.02 * half.max():
            # express the hole's half sides along the outer rectangle's axes
            if abs(float(R2[:, 0] @ R2h[:, 0])) > 0.5:
                ix, iy = halfh
            else:
                iy, ix = halfh
            return SectionFit("rhs", c, R2, dict(hx=half[0], hy=half[1], ix=ix, iy=iy),
                              confidence=min(area / rect, ah / recth))
    return SectionFit("other", centroid(loop), eye, {}, confidence=0.0)


# ------------------------------------------------------------------------------------------------ members

@dataclass
class Member:
    axis: np.ndarray                 # world unit vector
    e_u: np.ndarray                  # world in-plane axes: plane coordinates (u, v) = (p . e_u, p . e_v)
    e_v: np.ndarray
    loop: np.ndarray                 # (n, 2) outer section loop in (u, v)
    holes: List[np.ndarray]
    bbox: Tuple[float, float, float, float]    # umin, vmin, umax, vmax
    area: float                      # outer loop area (holes not subtracted)
    start: float                     # extent along the axis (p . axis)
    end: float
    stations: List[float]            # axis coordinates where the loop matched exactly
    joined: Tuple[bool, bool] = (False, False)   # material beyond the start / end touches the section
    section: Optional[SectionFit] = None

    @property
    def length(self) -> float:
        return self.end - self.start

    @property
    def kind(self) -> str:
        return self.section.kind if self.section is not None else "other"

    def to_world(self, uv, s) -> np.ndarray:
        uv = np.asarray(uv, float)
        return uv[0] * self.e_u + uv[1] * self.e_v + float(s) * self.axis

    def summary(self) -> Dict:
        return dict(kind=self.kind, axis=tuple(float(x) for x in self.axis), start=float(self.start),
                    end=float(self.end), length=float(self.length), stations=len(self.stations),
                    joined=tuple(self.joined), section=self.section.label if self.section is not None else None,
                    bbox=tuple(float(x) for x in self.bbox), area=float(self.area))


def _perp_basis(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = a / np.linalg.norm(a)
    t = np.eye(3)[int(np.argmin(np.abs(a)))]
    u = np.cross(a, t)
    u /= np.linalg.norm(u)
    return u, np.cross(a, u)


def _fold(d: np.ndarray) -> np.ndarray:
    """Antipodal fold: the component of largest magnitude made positive."""
    d = np.asarray(d, float)
    idx = np.argmax(np.abs(d), axis=-1)
    sgn = np.sign(np.take_along_axis(d, idx[..., None], axis=-1))
    return d * np.where(sgn == 0, 1.0, sgn)


def _clusters(dirs: np.ndarray, weights: np.ndarray, cos_tol: float, max_n: int = 64):
    """Greedy direction clusters, heaviest representative first: [(representative, total weight)]."""
    order = np.argsort(-weights)
    cap = 4 * max_n
    reps = np.zeros((min(cap, len(order)) + 1, 3))
    w = np.zeros(len(reps))
    n = 0
    for i in order:
        d = dirs[i]
        if n:
            dots = np.abs(reps[:n] @ d)
            j = int(np.argmax(dots))
            if dots[j] >= cos_tol:
                w[j] += weights[i]
                continue
        if n >= cap:
            continue
        reps[n], w[n] = d, float(weights[i])
        n += 1
    o = np.argsort(-w[:n])
    return [(reps[i].copy(), float(w[i])) for i in o]


EXACT_DEG = 1e-3       # directions this close are one direction (mesh coordinates are float32: about 1e-5 deg on a long edge)


def _two_level(dirs: np.ndarray, weights: np.ndarray, coarse_deg: float, fine_deg: float = 0.02):
    """Clusters at coarse_deg, each represented by its heaviest fine_deg sub-cluster (exact directions win over a spread
    of near-misses such as facet diagonals or the facets of a tessellated cylinder).

    Before that, directions within EXACT_DEG are summed to one: a cluster's representative is its heaviest member, and
    the heaviest single edge of a filleted member is a facet DIAGONAL of a fillet strip, which is longer than the
    strip's own edges and 0.014 deg off the axis on a 5.3 m girder with strips 1.3 mm wide. With it as the axis the
    section drifts by 0.8 mm over 3.2 m and the member ended there (measured 2026-09-28 on the frame review set: the
    girder of r01 3236 of 5290 mm, the chords of r02 3894 of 4300 mm). The generators of a prism all have the SAME
    direction and add up; the diagonals of the strips around a fillet all differ."""
    exact = _clusters(dirs, weights, math.cos(math.radians(EXACT_DEG)), max_n=1500)
    dirs, weights = np.array([d for d, _ in exact]), np.array([w for _, w in exact])
    fine = _clusters(dirs, weights, math.cos(math.radians(fine_deg)), max_n=1500)
    return _clusters(np.array([d for d, _ in fine]), np.array([w for _, w in fine]), math.cos(math.radians(coarse_deg)))


def dominant_frame(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """3x3, columns: the dominant face-normal direction, the dominant one perpendicular to it, their cross product."""
    tri = vertices[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    ok = a > 1e-300
    n, a = _fold(n[ok] / a[ok, None]), a[ok] / 2.0
    cl = _two_level(n, a, 2.0)
    e1 = cl[0][0]
    e2 = None
    for d, _ in cl[1:]:
        if abs(d @ e1) < 0.05:
            e2 = d - (d @ e1) * e1
            e2 /= np.linalg.norm(e2)
            break
    if e2 is None:
        e2, _ = _perp_basis(e1)
    return np.stack([e1, e2, np.cross(e1, e2)], axis=1)


def candidate_directions(vertices: np.ndarray, faces: np.ndarray, max_dirs: int = 48,
                         min_edge_fraction: float = 0.02) -> List[np.ndarray]:
    """Directions a member may run along (see the module docstring), deduplicated (1 deg): long-edge directions
    first (exact for a prism's generators), then the face-normal frame, then the normal clusters' cross products."""
    V = np.asarray(vertices, float)
    F = np.asarray(faces)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    frame = dominant_frame(V, F)
    out = []
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    e = np.unique(e, axis=0)
    vec = V[e[:, 1]] - V[e[:, 0]]
    L = np.linalg.norm(vec, axis=1)
    keep = L >= max(float(np.median(L)), min_edge_fraction * diag)       # the longer half, and not tiny
    if keep.any():
        idx = np.nonzero(keep)[0]
        idx = idx[np.argsort(-L[idx])][:6000]
        # a long facet DIAGONAL, a fraction of a degree off the axis, must not stand for the axis: two-level clusters
        cl = _two_level(_fold(vec[idx] / L[idx, None]), L[idx], 0.5)
        total = sum(w for _, w in cl)
        out += [d for d, w in cl if w >= 0.005 * total]
    out += [frame[:, i] for i in range(3)]
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    a = np.linalg.norm(n, axis=1)
    ok = a > 1e-300
    ncl = _two_level(_fold(n[ok] / a[ok, None]), a[ok], 2.0)[:8]
    for i in range(len(ncl)):
        for j in range(i + 1, len(ncl)):
            c = np.cross(ncl[i][0], ncl[j][0])
            if np.linalg.norm(c) > 0.5:
                out.append(c / np.linalg.norm(c))
    res = []
    cos1 = math.cos(math.radians(1.0))
    for d in out:
        d = d / np.linalg.norm(d)
        if all(abs(float(d @ r)) < cos1 for r in res):
            res.append(d)
        if len(res) >= max_dirs:
            break
    return res


class _Sectioner:
    """Plane sections of one mesh along one direction, in the plane coordinates (e_u, e_v), cached per station."""

    def __init__(self, mesh, axis, e_u, e_v):
        self.mesh, self.axis, self.e_u, self.e_v = mesh, axis, e_u, e_v
        self._cache: Dict[float, List[np.ndarray]] = {}

    def loops(self, s: float) -> List[np.ndarray]:
        s = float(s)
        if s not in self._cache:
            out = []
            try:
                sec = self.mesh.section(plane_origin=self.axis * s, plane_normal=self.axis)
            except Exception:
                sec = None
            if sec is not None:
                for L in sec.discrete:
                    L = _open(np.asarray(L, float))
                    if len(L) >= 3:
                        out.append(np.stack([L @ self.e_u, L @ self.e_v], axis=1))
            self._cache[s] = out
        return self._cache[s]


def _outer_with_holes(loops: List[np.ndarray]):
    """[(outer, [holes], bbox, area)] - a loop whose first vertex lies inside a bigger loop is a hole of the smallest such."""
    areas = [abs(signed_area(L)) for L in loops]
    parent = [-1] * len(loops)
    for i, L in enumerate(loops):
        best = None
        for j, M in enumerate(loops):
            if j == i or areas[j] <= areas[i]:
                continue
            if points_inside(L[:1], [M])[0] and (best is None or areas[j] < areas[best]):
                best = j
        parent[i] = best if best is not None else -1
    depth = []
    for i in range(len(loops)):
        d, p = 0, parent[i]
        while p != -1 and d < 16:
            d, p = d + 1, parent[p]
        depth.append(d)
    out = []
    for i, L in enumerate(loops):
        if depth[i] % 2 == 0:
            holes = [loops[j] for j in range(len(loops)) if parent[j] == i and depth[j] == depth[i] + 1]
            lo, hi = L.min(axis=0), L.max(axis=0)
            out.append((L, holes, (lo[0], lo[1], hi[0], hi[1]), areas[i]))
    return out


def _material_samples(loop: np.ndarray, holes: Sequence[np.ndarray], n: int = 96) -> np.ndarray:
    """Points just inside the material of a section: edge points of every loop offset a hair into the material."""
    pts = []
    size = float(np.ptp(loop, axis=0).max())
    eps = 2e-3 * size
    for L, into in [(loop, 1.0)] + [(hh, -1.0) for hh in holes]:
        sgn = 1.0 if signed_area(L) > 0 else -1.0             # counter-clockwise: inward normal is the left normal
        a, b = L, np.roll(L, -1, axis=0)
        d = b - a
        ln = np.linalg.norm(d, axis=1)
        ok = ln > 1e-12 * size
        a, d, ln = a[ok], d[ok], ln[ok]
        cum = np.concatenate([[0.0], np.cumsum(ln)])
        m = max(8, int(n * (1.0 if into > 0 else 0.5)))
        t = (np.arange(m) + 0.5) / m * cum[-1]
        k = np.clip(np.searchsorted(cum, t, side="right") - 1, 0, len(ln) - 1)
        f = (t - cum[k]) / ln[k]
        p = a[k] + d[k] * f[:, None]
        nrm = np.stack([-d[k, 1], d[k, 0]], axis=1) / ln[k, None] * sgn * into
        pts.append(p + eps * nrm)
    return np.vstack(pts)


def _same(desc_a, desc_b, tol_abs: float) -> bool:
    (_, ha, ba, aa), (_, hb, bb, ab) = desc_a, desc_b
    size = max(ba[2] - ba[0], ba[3] - ba[1])
    tol = max(tol_abs, 2e-3 * size)
    return (max(abs(x - y) for x, y in zip(ba, bb)) <= tol and abs(aa - ab) <= 0.01 * max(aa, ab)
            and len(ha) == len(hb))


@timed_fn("detect members")
def detect_members(field, min_length_ratio: float = 2.0, n_stations: int = 64, directions=None,
                   max_dirs: int = 48, snap: bool = True) -> List[Member]:
    """Straight prismatic members of the mesh of `field` (a SurfaceField, a trimesh, or (vertices, faces)), longest first,
    each with its section classified (Member.section)."""
    import trimesh
    if isinstance(field, trimesh.Trimesh):
        mesh = field
    elif isinstance(field, tuple):
        mesh = trimesh.Trimesh(np.asarray(field[0], float), np.asarray(field[1]), process=False)
    else:
        mesh = trimesh.Trimesh(np.asarray(field.vertices, float), np.asarray(field.faces), process=False)
    V = np.asarray(mesh.vertices, float)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol_abs = 1e-5 * diag
    dirs = directions if directions is not None else candidate_directions(V, np.asarray(mesh.faces), max_dirs=max_dirs)
    members: List[Member] = []
    for a in dirs:
        a = np.asarray(a, float) / np.linalg.norm(a)
        e_u, e_v = _perp_basis(a)
        proj = V @ a
        lo, hi = float(proj.min()), float(proj.max())
        span = hi - lo
        if span <= 0:
            continue
        sec = _Sectioner(mesh, a, e_u, e_v)
        ss = lo + (np.arange(n_stations) + 0.5 + 0.0123457) / (n_stations + 0.05) * span
        per = [_outer_with_holes(sec.loops(s)) for s in ss]
        tracks = []                                                   # [first k, last k, desc]
        open_ = []
        for k, descs in enumerate(per):
            nxt = []
            used = set()
            for d in descs:
                hit = None
                for ti, t in enumerate(open_):
                    if ti not in used and t[1] == k - 1 and _same(t[2], d, tol_abs):
                        hit = ti
                        break
                if hit is not None:
                    used.add(hit)
                    open_[hit][1] = k
                    nxt.append(open_[hit])
                else:
                    nxt.append([k, k, d])
            tracks += [t for ti, t in enumerate(open_) if ti not in used]
            open_ = nxt
        tracks += open_
        found: List[Member] = []
        for k0, k1, d in tracks:
            if k1 - k0 + 1 < 3:
                continue
            loop, holes, bbox, area = d
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if span < min_length_ratio * size:
                continue
            probe = _material_samples(loop, holes)

            def covered(s):
                loops = sec.loops(s)
                return bool(loops) and bool(points_inside(probe, loops).all())

            def touches(s):
                loops = sec.loops(s)
                return bool(loops) and bool(points_inside(probe, loops).any())

            def extend(k, step):
                kk = k
                while 0 <= kk + step < n_stations and covered(ss[kk + step]):
                    kk += step
                a_in = ss[kk]
                b_out = ss[kk + step] if 0 <= kk + step < n_stations else (lo if step < 0 else hi) - step * 1e-9 * span
                for _ in range(30):
                    m = 0.5 * (a_in + b_out)
                    if covered(m):
                        a_in = m
                    else:
                        b_out = m
                end = 0.5 * (a_in + b_out)
                near = proj[np.abs(proj - end) <= max(4.0 * abs(b_out - a_in), 1e-9 * span)]
                if len(near):
                    end = float(near[np.argmin(np.abs(near - end))])
                gap = max(1e-4 * span, 1e-3 * size)
                joined = touches(end + step * gap) if (lo < end + step * gap < hi) else False
                return end, joined
            s0, j0 = extend(k0, -1)
            s1, j1 = extend(k1, +1)
            if s1 - s0 < min_length_ratio * size:
                continue
            m = Member(axis=a.copy(), e_u=e_u, e_v=e_v, loop=loop, holes=list(holes), bbox=tuple(float(x) for x in bbox),
                       area=float(area), start=float(s0), end=float(s1), stations=[float(x) for x in ss[k0:k1 + 1]],
                       joined=(bool(j0), bool(j1)))
            dup = False
            for o in found:                                           # the same loop broken by a joint: merge
                if _same((o.loop, o.holes, o.bbox, o.area), d, tol_abs) and o.start <= m.end + 1e-9 * span \
                        and m.start <= o.end + 1e-9 * span:
                    if m.start < o.start:
                        o.start, o.joined = m.start, (m.joined[0], o.joined[1])
                    if m.end > o.end:
                        o.end, o.joined = m.end, (o.joined[0], m.joined[1])
                    o.stations = sorted(set(o.stations) | set(m.stations))
                    dup = True
                    break
            if not dup:
                found.append(m)
        members += found
    members = _dedupe(members)
    for m in members:
        m.section = classify_section(m.loop, m.holes, snap=snap)
    members.sort(key=lambda m: -m.length)
    return members


def _dedupe(members: List[Member], max_deg: float = 10.0) -> List[Member]:
    """One member per physical run: a slightly oblique section of a rod is still (nearly) a circle, so the same rod shows
    up along several near-parallel directions. Of members with axes within max_deg whose centre lines are closer than
    half a section size and whose extents overlap, the one matched over most stations (the exact direction) is kept."""
    def centre_line(m):
        b = m.bbox
        return m.to_world(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 0.0), 0.5 * max(b[2] - b[0], b[3] - b[1])
    order = sorted(members, key=lambda m: (-len(m.stations), -m.length))
    kept: List[Member] = []
    cos_t = math.cos(math.radians(max_deg))
    for m in order:
        p, half = centre_line(m)
        mid = p + m.axis * 0.5 * (m.start + m.end)
        dup = False
        for k in kept:
            if abs(float(m.axis @ k.axis)) < cos_t:
                continue
            q, khalf = centre_line(k)
            d = mid - q
            off = np.linalg.norm(d - (d @ k.axis) * k.axis)
            if off > 0.5 * max(half, khalf):
                continue
            a0 = sorted([float((p + m.axis * m.start) @ k.axis), float((p + m.axis * m.end) @ k.axis)])
            if a0[0] <= k.end and a0[1] >= k.start:
                dup = True
                break
        if not dup:
            kept.append(m)
    return kept


# ------------------------------------------------------------------------------------------------ primitives

def _sources(mode: str, conf: float) -> List[StageSource]:
    return [StageSource(s, mode, float(conf)) for s in STAGE_NAMES]


def _frame(member: Member, sec: SectionFit) -> np.ndarray:
    """World rotation, columns: section local x, section local y, the member axis (right-handed)."""
    x = sec.axes[0, 0] * member.e_u + sec.axes[1, 0] * member.e_v
    y = sec.axes[0, 1] * member.e_u + sec.axes[1, 1] * member.e_v
    R = np.stack([x, y, member.axis], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    return R


def _box(R, centre, half, label, mode, conf) -> Primitive:
    return Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in centre),
                     dims_mm=dict(x_half=float(half[0]), y_half=float(half[1]), z_half=float(half[2])),
                     confidence=float(conf), sources=_sources(mode, conf), label=label)


def _cyl(R, centre, r, half_len, label, mode, conf) -> Primitive:
    return Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in centre),
                     dims_mm=dict(radius=float(r), half_length=float(half_len)), confidence=float(conf),
                     sources=_sources(mode, conf), label=label)


def ibeam_boxes(member: Member, section=None) -> List[Primitive]:
    """The three boxes of an I member (top flange, bottom flange, web) in the world frame. `section`: an IBeamSection or a
    SectionFit (default: member.section)."""
    sec = section if section is not None else member.section
    if isinstance(sec, IBeamSection):
        ib = sec
        sec = SectionFit("ibeam", ib.centre, np.stack([ib.b_dir, ib.h_dir], axis=1), confidence=ib.confidence, ibeam=ib)
    ib = sec.ibeam
    R = _frame(member, sec)
    y = R[:, 1]
    s0, s1 = member.start, member.end
    c0 = member.to_world(sec.centre, 0.5 * (s0 + s1))
    L2 = 0.5 * (s1 - s0)
    yf = 0.5 * (ib.h - ib.tf)
    name = ib.label
    out = []
    for part, off, half in (("top flange", yf, (ib.b / 2, ib.tf / 2, L2)),
                            ("bottom flange", -yf, (ib.b / 2, ib.tf / 2, L2)),
                            ("web", 0.0, (ib.tw / 2, ib.h / 2 - ib.tf / 2, L2))):
        out.append(_box(R, c0 + off * y, half, f"{name} {part}", MODE_IBEAM, ib.confidence))
    return out


def member_operations(member: Member) -> List[Tuple[str, Primitive]]:
    """[(op, primitive)] of one member by its section kind; [] for "other"."""
    sec = member.section
    if sec is None or sec.kind == "other":
        return []
    if sec.kind == "ibeam":
        return [("add", p) for p in ibeam_boxes(member, sec)]
    R = _frame(member, sec)
    s0, s1 = member.start, member.end
    L2 = 0.5 * (s1 - s0)
    c = member.to_world(sec.centre, 0.5 * (s0 + s1))
    # a cutter runs 5 % past a free end (through hole), and stops at a joined end
    e0 = 0.0 if member.joined[0] else 0.05 * (s1 - s0)
    e1 = 0.0 if member.joined[1] else 0.05 * (s1 - s0)
    cc = member.to_world(sec.centre, 0.5 * (s0 - e0 + s1 + e1))
    cL2 = L2 + 0.5 * (e0 + e1)
    p, conf = sec.params, sec.confidence
    if sec.kind == "rod":
        return [("add", _cyl(R, c, p["r"], L2, sec.label, MODE_MEMBER, conf))]
    if sec.kind == "bar":
        return [("add", _box(R, c, (p["hx"], p["hy"], L2), sec.label, MODE_MEMBER, conf))]
    if sec.kind == "tube":
        return [("add", _cyl(R, c, p["r"], L2, sec.label, MODE_MEMBER, conf)),
                ("cut", _cyl(R, cc, p["r_in"], cL2, sec.label + " bore", MODE_MEMBER, conf))]
    if sec.kind == "rhs":
        return [("add", _box(R, c, (p["hx"], p["hy"], L2), sec.label, MODE_MEMBER, conf)),
                ("cut", _box(R, cc, (p["ix"], p["iy"], cL2), sec.label + " hole", MODE_MEMBER, conf))]
    return []


def member_prism_mask(member: Member, centres: np.ndarray) -> np.ndarray:
    """Grid voxels (centres (..., 3)) inside the member's section bounding box x extent: its region on the grid."""
    c = np.asarray(centres, float)
    s = c @ member.axis
    u, v = c @ member.e_u, c @ member.e_v
    b = member.bbox
    return (s >= member.start) & (s <= member.end) & (u >= b[0]) & (u <= b[2]) & (v >= b[1]) & (v <= b[3])


def member_bounding_box(member: Member) -> Primitive:
    """The exact bounding box of a member (section bbox x extent): the old path's bounding part for an unknown section."""
    b = member.bbox
    R = np.stack([member.e_u, member.e_v, member.axis], axis=1)
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    c = member.to_world(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), 0.5 * (member.start + member.end))
    return _box(R, c, ((b[2] - b[0]) / 2, (b[3] - b[1]) / 2, member.length / 2), "member bounding box",
                "prior:member_bbox", 1.0)


# ------------------------------------------------------------------------------------------------ cavities

@dataclass
class Cavities:
    outer: Tuple[np.ndarray, np.ndarray]                 # (vertices, faces) of the outer shell
    inner: List[Tuple[np.ndarray, np.ndarray]]           # the cavity shells, faces flipped to enclose a positive volume


@timed_fn("find cavities")
def find_cavities(field) -> Optional[Cavities]:
    """The closed cavities of the mesh: surface components inside the largest one. None when there is none."""
    import trimesh
    mesh = trimesh.Trimesh(np.asarray(field.vertices, float), np.asarray(field.faces), process=False)
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:
        return None
    if len(comps) < 2:
        return None
    comps = sorted(comps, key=lambda c: -abs(float(c.volume)))
    outer = comps[0]
    if outer.volume < 0:
        outer.invert()
    inner = []
    for c in comps[1:]:
        if len(c.faces) < 4 or not c.is_watertight or abs(float(c.volume)) < 1e-6 * abs(float(outer.volume)):
            continue                                                  # a stray triangle, not a shell
        try:
            if not outer.contains(c.vertices[:1])[0]:
                continue
        except Exception:
            continue
        if c.volume < 0:
            c.invert()
        inner.append((np.asarray(c.vertices, float), np.asarray(c.faces)))
    if not inner:
        return None
    return Cavities((np.asarray(outer.vertices, float), np.asarray(outer.faces)), inner)


# ------------------------------------------------------------------------------------------------ exact evaluation

def primitive_volume(p: Primitive) -> float:
    d = p.dims_mm
    if p.kind == "box":
        return 8.0 * d["x_half"] * d["y_half"] * d["z_half"]
    if p.kind == "cylinder":
        return math.pi * d["radius"] ** 2 * 2.0 * d["half_length"]
    if p.kind == "sphere":
        return 4.0 / 3.0 * math.pi * d["radius"] ** 3
    return _sdf_sampled(p, 4000, 0)[1]


def _sdf_sampled(p: Primitive, n: int, seed: int) -> Tuple[np.ndarray, float]:
    """(up to n uniform points inside, volume) of any primitive - cone, torus, composite - by rejection on its own SDF in
    its bounds (the hollow sub-loop's any-kind cavity cutter, 2026-10-29: a cone crashed the box/cylinder/sphere samplers)."""
    from reco_engine.result_to_program import primitive_to_sp
    s = primitive_to_sp(p)
    lo, hi = (np.asarray(b, float) for b in s.bounds())
    rng = np.random.default_rng(seed)
    got, kept, drawn = [], 0, 0
    for _ in range(64):
        P = rng.uniform(lo, hi, size=(4 * n, 3))
        inside = P[np.asarray(s.sdf(P)) <= 0]
        drawn, kept = drawn + len(P), kept + len(inside)
        got.append(inside)
        if kept >= n:
            break
    pts = np.concatenate(got)[:n] if got else np.zeros((0, 3))
    return pts, float(np.prod(hi - lo) * kept / max(drawn, 1))


def primitive_samples(p: Primitive, n: int = 1500, seed: int = 0) -> np.ndarray:
    """~n uniform points inside a primitive (world coordinates): closed forms for a box / cylinder / sphere, rejection on
    the SDF for any other kind."""
    from reco_engine.result_to_program import rotation_matrix
    rng = np.random.default_rng(seed)
    R, c, d = rotation_matrix(p), np.asarray(p.center_mm, float), p.dims_mm
    if p.kind == "box":
        h = np.array([d["x_half"], d["y_half"], d["z_half"]])
        loc = rng.uniform(-1, 1, (n, 3)) * h
    elif p.kind == "cylinder":
        rr = d["radius"] * np.sqrt(rng.uniform(0, 1, n))
        t = rng.uniform(0, 2 * math.pi, n)
        loc = np.stack([rr * np.cos(t), rr * np.sin(t), rng.uniform(-1, 1, n) * d["half_length"]], axis=1)
    elif p.kind == "sphere":
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        loc = v * d["radius"] * rng.uniform(0, 1, n)[:, None] ** (1 / 3)
    else:
        return _sdf_sampled(p, n, seed)[0]
    return c + loc @ R.T


class ExactJudge:
    """Exact (mesh) volume accounting of an operation, for walls far thinner than the grid pitch: points sampled in the
    primitive, inside the input by the generalized winding number (libigl, exact for a closed mesh; trimesh `contains`,
    ray parity, without libigl), inside the model by the program's own SDF. Measured on the frame, 1500 points: igl
    0.22 s, trimesh 2.2 s, the same answer for all of 3000 points (uniform and within 0.02 of the surface)."""

    def __init__(self, field, n: int = 1500):
        import trimesh
        self.mesh = trimesh.Trimesh(np.asarray(field.vertices, float), np.asarray(field.faces), process=False)
        self._V = np.ascontiguousarray(field.vertices, dtype=np.float64)
        self._F = np.ascontiguousarray(field.faces, dtype=np.int64)
        self.n = n

    @timed_fn("inside input")
    def contains(self, pts: np.ndarray) -> np.ndarray:
        try:
            import igl
        except ImportError:
            return np.asarray(self.mesh.contains(pts), bool)
        return np.abs(igl.winding_number(self._V, self._F, np.ascontiguousarray(pts, dtype=np.float64))) > 0.5

    def judge(self, prim: Primitive, op: str, model=None, seed: int = 0, exclude=None) -> Dict[str, float]:
        """dict(volume, in_input, in_model, new, excess, removed_good, removed_bad, gain_volume, precision). `exclude`: an
        SDF solid whose inside is not counted - for an add, the cutter that follows it (a tube's bore)."""
        pts = primitive_samples(prim, self.n, seed)
        vol = primitive_volume(prim)
        inside = self.contains(pts)
        with timed("model sdf"):
            in_model = (np.asarray(model.sdf(pts)) <= 0) if model is not None else np.zeros(len(pts), bool)
        counted = np.ones(len(pts), bool) if exclude is None else (np.asarray(exclude.sdf(pts)) > 0)
        out = dict(volume=vol, in_input=float(inside.mean()), in_model=float(in_model.mean()),
                   counted=float(counted.mean()))
        if op == "add":
            new = float((inside & ~in_model & counted).mean())
            excess = float((~inside & ~in_model & counted).mean())
            prec = float((inside & counted).sum() / max(1, counted.sum()))
            out.update(new=new, excess=excess, gain_volume=vol * (new - excess), precision=prec)
        else:
            good = float((in_model & ~inside).mean())
            bad = float((in_model & inside).mean())
            m = float(in_model.mean())
            out.update(removed_good=good, removed_bad=bad, gain_volume=vol * (good - bad),
                       precision=(good / m) if m > 0 else 0.0)
        return out
