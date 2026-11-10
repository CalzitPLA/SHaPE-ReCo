#!/usr/bin/env python3
"""
The exact boundary of a CSG tree (dataset_gen/src/sdf_primitives.py), as area-uniform surface samples with outward
normals and the leaf that owns each one - the common ground of the validation cycle (reco_engine/validation.py), on which
a reconstruction and its true tree are compared surface against surface.

    tree = compile_tree(sp_tree, owners)          # leaves with world frames, a fast evaluator with the deciding leaf
    s = boundary_samples(tree, n_target=200_000)   # points, outward normals, leaf, owner, density, spacing
    d = surface_distances(s.points, s.normals, other_tree, other_samples)   # bracketed distances to the other boundary

How a boundary point is found. Every leaf's boundary lies on a few simple surfaces (walls, caps, sector planes, latitude
cones or planes, tube surfaces, a clipped plane). Each of them is sampled area-uniformly over a region that holds the
leaf's part of it; a sample is kept when
  1. the leaf's sign changes across it (it is on the leaf's boundary, not on the unused part of the surface),
  2. the TREE's sign changes across it (it is on the solid's boundary: faces inside the solid or outside it are dropped,
     and so is a face shared by two leaves inside a union), and
  3. the leaf that produced it is the leaf that decides the tree's value there (active_leaf): where two leaves' faces
     coincide on the boundary, only one copy survives, so the density stays uniform.
The sign of an sp tree is exact although its values are only bounds, so the tests are exact up to the step delta (a
millionth of the tree's size). The outward normal is the side on which the tree is positive.

Distances: |sdf| of a Boolean tree is only a LOWER bound of the distance to its boundary (`d_lo`). The distance to the
nearest boundary sample, and to the boundary point a Newton projection reaches, are UPPER bounds (`d_hi`); the validation
grades on d_hi and reports how often the two differ by more than a tolerance.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402

TWO_PI = 2.0 * np.pi
CHUNK = 200_000                                  # points per evaluation chunk (memory: a few x CHUNK x leaves doubles)


# ------------------------------------------------------------------------------------------------- the compiled tree

@dataclass
class Leaf:
    """A leaf primitive with its world frame (x_world = R x_local + t), the operation that owns it, and its role:
    'add' (its inside is the solid's inside), 'cut' (its inside is removed) or 'clip' (an operand of an intersection)."""
    prim: sp.Primitive
    R: np.ndarray
    t: np.ndarray
    owner: int
    role: str
    index: int

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return self.prim.sdf((np.asarray(p, float) - self.t) @ self.R)

    def to_local(self, p: np.ndarray) -> np.ndarray:
        return (np.asarray(p, float) - self.t) @ self.R

    def to_world(self, p: np.ndarray, n: Optional[np.ndarray] = None):
        w = np.asarray(p, float) @ self.R.T + self.t
        return w if n is None else (w, np.asarray(n, float) @ self.R.T)


class CompiledTree:
    """An sp tree as a list of leaves and a nested node tuple ('leaf', i) | (op, a, b) with op in union/inter/sub, so the
    value and the DECIDING leaf of every point come out of one pass."""

    def __init__(self, node, leaves: List[Leaf]):
        self.node, self.leaves = node, leaves

    def sdf(self, p: np.ndarray) -> np.ndarray:
        return self.evaluate(p)[0]

    def evaluate(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(value, index of the deciding leaf) at every point; chunked."""
        p = np.asarray(p, float).reshape(-1, 3)
        if len(p) <= CHUNK:
            return self._eval(self.node, p)
        out_v, out_i = np.empty(len(p)), np.empty(len(p), np.int64)
        for k in range(0, len(p), CHUNK):
            out_v[k:k + CHUNK], out_i[k:k + CHUNK] = self._eval(self.node, p[k:k + CHUNK])
        return out_v, out_i

    def _eval(self, node, p):
        kind = node[0]
        if kind == "leaf":
            leaf = self.leaves[node[1]]
            return leaf.sdf(p), np.full(len(p), node[1], np.int64)
        va, ia = self._eval(node[1], p)
        vb, ib = self._eval(node[2], p)
        if kind == "union":
            take_a = va <= vb
            return np.where(take_a, va, vb), np.where(take_a, ia, ib)
        if kind == "inter":
            take_a = va >= vb
            return np.where(take_a, va, vb), np.where(take_a, ia, ib)
        nb = -vb                                                            # sub: max(a, -b)
        take_a = va >= nb
        return np.where(take_a, va, nb), np.where(take_a, ia, ib)

    def owners(self) -> np.ndarray:
        return np.array([leaf.owner for leaf in self.leaves], np.int64)


def compile_tree(tree: sp.Primitive, owner: int = -1, leaves: Optional[List[Leaf]] = None) -> CompiledTree:
    """Collapse the Placed chains into world frames and number the leaves in traversal order. `owner` labels every leaf
    (an operation index); compile_program gives each operation's leaves their own owner."""
    leaves = [] if leaves is None else leaves
    node = _compile(tree, np.eye(3), np.zeros(3), +1, False, owner, leaves)
    return CompiledTree(node, leaves)


def _compile(node, R, t, polarity, clip, owner, leaves):
    if isinstance(node, sp.Placed):
        Rn, tn = np.asarray(node.rotation, float), np.asarray(node.translation, float)
        return _compile(node.primitive, R @ Rn, R @ tn + t, polarity, clip, owner, leaves)
    if isinstance(node, sp.Union):
        return ("union", _compile(node.a, R, t, polarity, clip, owner, leaves),
                _compile(node.b, R, t, polarity, clip, owner, leaves))
    if isinstance(node, sp.Intersection):
        return ("inter", _compile(node.a, R, t, polarity, clip, owner, leaves),
                _compile(node.b, R, t, polarity, True, owner, leaves))
    if isinstance(node, sp.Subtraction):
        return ("sub", _compile(node.a, R, t, polarity, clip, owner, leaves),
                _compile(node.b, R, t, -polarity, clip, owner, leaves))
    role = "cut" if polarity < 0 else ("clip" if clip else "add")
    leaves.append(Leaf(node, R.copy(), t.copy(), owner, role, len(leaves)))
    return ("leaf", len(leaves) - 1)


def compile_program(ops: Sequence[Tuple[str, sp.Primitive]]) -> CompiledTree:
    """The left fold of program_model (first op the base; then union for 'add', subtraction for 'cut', intersection for
    'intersect'), with every leaf owned by the index of its operation (a composite's parts share it)."""
    leaves: List[Leaf] = []
    node = None
    for k, (op, prim) in enumerate(ops):
        sub = _compile(prim, np.eye(3), np.zeros(3), -1 if op == "cut" else +1, op == "intersect", k, leaves)
        if node is None:
            if op != "add":
                raise ValueError("the first operation of a program must be an add (the base)")
            node = sub
        else:
            node = ({"add": "union", "cut": "sub", "intersect": "inter"}[op], node, sub)
    if node is None:
        raise ValueError("empty program")
    return CompiledTree(node, leaves)


def finite_bounds(leaves: Sequence[Leaf], roles=("add",)) -> Tuple[np.ndarray, np.ndarray]:
    """The world box of the leaves of the given roles (material comes only from adds); half spaces are skipped."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for leaf in leaves:
        if leaf.role not in roles or isinstance(leaf.prim, sp.HalfSpace):
            continue
        a, b = leaf.prim.bounds()
        corners = np.array([[x, y, z] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])
        w = corners @ leaf.R.T + leaf.t
        lo, hi = np.minimum(lo, w.min(0)), np.maximum(hi, w.max(0))
    if not np.all(np.isfinite(lo)):
        raise ValueError("no finite leaf of the requested roles")
    return lo, hi


# ---------------------------------------------------------------------------------------------- per-leaf candidates

def _range(u, full=(0.0, TWO_PI)):
    u0, u1 = float(u[0]), float(u[1])
    return (full if u1 - u0 >= TWO_PI - 1e-9 else (u0, u1)), (u1 - u0 < TWO_PI - 1e-9)


def _count(area: float, density: float, rng) -> int:
    return int(rng.poisson(max(area, 0.0) * density)) if area > 0 else 0


def _ring(rng, n, r_lo, r_hi):
    """Radii of n area-uniform points of an annulus r_lo..r_hi."""
    return np.sqrt(r_lo ** 2 + rng.random(n) * (r_hi ** 2 - r_lo ** 2))


def _sector_planes(rng, density, u, rho_lo, rho_hi, z_lo, z_hi):
    """The two half planes through the z axis at the sector's ends, over rho in [rho_lo, rho_hi], z in [z_lo, z_hi]."""
    out = []
    for phi in u:
        n = _count((rho_hi - rho_lo) * (z_hi - z_lo), density, rng)
        rho = rho_lo + rng.random(n) * (rho_hi - rho_lo)
        z = z_lo + rng.random(n) * (z_hi - z_lo)
        c, s = np.cos(phi), np.sin(phi)
        out.append((np.c_[rho * c, rho * s, z], np.tile([-s, c, 0.0], (n, 1))))
    return out


def _wall(rng, density, radius, u, z_lo, z_hi, sign=1.0):
    n = _count(radius * (u[1] - u[0]) * (z_hi - z_lo), density, rng)
    phi = u[0] + rng.random(n) * (u[1] - u[0])
    z = z_lo + rng.random(n) * (z_hi - z_lo)
    c, s = np.cos(phi), np.sin(phi)
    return np.c_[radius * c, radius * s, z], sign * np.c_[c, s, np.zeros(n)]


def _disc(rng, density, z, r_lo, r_hi, u, sign):
    n = _count(0.5 * (u[1] - u[0]) * (r_hi ** 2 - r_lo ** 2), density, rng)
    rho = _ring(rng, n, r_lo, r_hi)
    phi = u[0] + rng.random(n) * (u[1] - u[0])
    return np.c_[rho * np.cos(phi), rho * np.sin(phi), np.full(n, z)], np.tile([0.0, 0.0, sign], (n, 1))


def _box(prim: sp.Box, rng, density):
    h = np.asarray(prim.half, float)
    out = []
    for ax in range(3):
        o = [a for a in range(3) if a != ax]
        for sgn in (-1.0, 1.0):
            n = _count(4.0 * h[o[0]] * h[o[1]], density, rng)
            p = np.zeros((n, 3))
            p[:, ax] = sgn * h[ax]
            p[:, o[0]] = (rng.random(n) * 2 - 1) * h[o[0]]
            p[:, o[1]] = (rng.random(n) * 2 - 1) * h[o[1]]
            nn = np.zeros((n, 3))
            nn[:, ax] = sgn
            out.append((p, nn))
    return out


def _cylinder(prim: sp.Cylinder, rng, density):
    u, partial = _range(prim.u)
    ro, ri, hz = float(prim.radius), float(prim.inner_radius), float(prim.height)
    out = [_wall(rng, density, ro, u, 0.0, hz)]
    if ri > 0:
        out.append(_wall(rng, density, ri, u, 0.0, hz, -1.0))
    out += [_disc(rng, density, 0.0, ri, ro, u, -1.0), _disc(rng, density, hz, ri, ro, u, 1.0)]
    if partial:
        out += _sector_planes(rng, density, u, ri, ro, 0.0, hz)
    return out


def _cone(prim: sp.Cone, rng, density):
    u, partial = _range(prim.u)
    r1, r2, hz = float(prim.radius1), float(prim.radius2), float(prim.height)
    slant = np.hypot(hz, r2 - r1)
    n = _count(0.5 * (u[1] - u[0]) * (r1 + r2) * slant, density, rng)
    if abs(r2 - r1) < 1e-12:
        r = np.full(n, r1)
        z = rng.random(n) * hz
    else:
        r = np.sqrt(r1 ** 2 + rng.random(n) * (r2 ** 2 - r1 ** 2))   # area grows with r: r^2 uniform
        z = (r - r1) * hz / (r2 - r1)
    phi = u[0] + rng.random(n) * (u[1] - u[0])
    c, s = np.cos(phi), np.sin(phi)
    k = -(r2 - r1) / hz
    nrm = np.c_[c, s, np.full(n, k)] / np.sqrt(1.0 + k * k)
    out = [(np.c_[r * c, r * s, z], nrm)]
    if r1 > 0:
        out.append(_disc(rng, density, 0.0, 0.0, r1, u, -1.0))
    if r2 > 0:
        out.append(_disc(rng, density, hz, 0.0, r2, u, 1.0))
    if partial:
        out += _sector_planes(rng, density, u, 0.0, max(r1, r2), 0.0, hz)
    return out


def _sphere(prim: sp.Sphere, rng, density):
    u, upartial = _range(prim.u)
    v0, v1 = float(prim.v[0]), float(prim.v[1])
    vpartial = not (v0 <= -np.pi / 2 + 1e-9 and v1 >= np.pi / 2 - 1e-9)
    R, ri = float(prim.radius), float(prim.inner_radius)
    out = []
    for rad, sign in ((R, 1.0), (ri, -1.0)):
        if rad <= 0:
            continue
        if prim.latitude == "plane" and vpartial:
            z0, z1 = max(-rad, R * np.sin(v0)), min(rad, R * np.sin(v1))
        else:
            z0, z1 = rad * np.sin(v0), rad * np.sin(v1)
        if z1 <= z0:
            continue
        n = _count(rad * (u[1] - u[0]) * (z1 - z0), density, rng)      # Archimedes: area-uniform = z uniform
        z = z0 + rng.random(n) * (z1 - z0)
        phi = u[0] + rng.random(n) * (u[1] - u[0])
        rho = np.sqrt(np.maximum(rad ** 2 - z ** 2, 0.0))
        p = np.c_[rho * np.cos(phi), rho * np.sin(phi), z]
        out.append((p, sign * p / rad))
    if vpartial:
        for v in (v0, v1):
            if abs(abs(v) - np.pi / 2) < 1e-9:
                continue
            if prim.latitude == "plane":
                z = R * np.sin(v)
                rmax = np.sqrt(max(R ** 2 - z ** 2, 0.0))
                out.append(_disc(rng, density, z, 0.0, rmax, u, 1.0))
            else:                                                        # the cone from the centre at latitude v
                cv, sv = np.cos(v), np.sin(v)
                n = _count(0.5 * (u[1] - u[0]) * abs(cv) * (R ** 2 - ri ** 2), density, rng)
                srad = _ring(rng, n, ri, R)
                phi = u[0] + rng.random(n) * (u[1] - u[0])
                c, s = np.cos(phi), np.sin(phi)
                p = np.c_[srad * cv * c, srad * cv * s, srad * sv]
                out.append((p, np.c_[-sv * c, -sv * s, np.full(n, cv)]))
    if upartial:
        out += _sector_planes(rng, density, u, 0.0, R, -R, R)
    return out


def _torus(prim: sp.Torus, rng, density):
    u, upartial = _range(prim.u)
    v, vpartial = _range(prim.v)
    Rm, r, ri = float(prim.major), float(prim.minor), float(prim.inner_minor)
    out = []
    for rad, sign in ((r, 1.0), (ri, -1.0)):
        if rad <= 0:
            continue
        n_prop = _count(rad * (u[1] - u[0]) * (v[1] - v[0]) * (Rm + rad), density, rng)
        th = v[0] + rng.random(n_prop) * (v[1] - v[0])
        w = np.abs(Rm + rad * np.cos(th)) / (Rm + rad)
        th = th[rng.random(n_prop) < w]                                  # area weight (R + r cos theta)
        n = len(th)
        phi = u[0] + rng.random(n) * (u[1] - u[0])
        c, s, ct, st = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th)
        rho = Rm + rad * ct
        out.append((np.c_[rho * c, rho * s, rad * st], sign * np.c_[ct * c, ct * s, st]))
    if vpartial:
        for th in v:
            ct, st = np.cos(th), np.sin(th)
            n_prop = _count((u[1] - u[0]) * (r - ri) * (Rm + r), density, rng)
            srad = ri + rng.random(n_prop) * (r - ri)
            w = np.abs(Rm + srad * ct) / (Rm + r)
            srad = srad[rng.random(n_prop) < w]
            n = len(srad)
            phi = u[0] + rng.random(n) * (u[1] - u[0])
            c, s = np.cos(phi), np.sin(phi)
            rho = Rm + srad * ct
            out.append((np.c_[rho * c, rho * s, srad * st], np.c_[-st * c, -st * s, np.full(n, ct)]))
    if upartial:
        out += _sector_planes(rng, density, u, max(Rm - r, 0.0), Rm + r, -r, r)
    return out


def _halfspace(leaf: Leaf, rng, density, clip):
    nrm = np.asarray(leaf.prim.normal, float)
    lo, hi = clip
    m = leaf.to_local(0.5 * (lo + hi))
    half = 0.5 * float(np.linalg.norm(hi - lo))
    c = m - (m @ nrm - float(leaf.prim.offset)) * nrm
    e1 = np.cross(nrm, [1.0, 0.0, 0.0] if abs(nrm[0]) < 0.9 else [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(nrm, e1)
    n = _count((2 * half) ** 2, density, rng)
    a, b = (rng.random(n) * 2 - 1) * half, (rng.random(n) * 2 - 1) * half
    return [(c + a[:, None] * e1 + b[:, None] * e2, np.tile(nrm, (n, 1)))]


def _polyhedron(prim, rng, density):
    from scipy.spatial import ConvexHull
    hull = ConvexHull(prim.vertices)
    out = []
    for simplex, eq in zip(hull.simplices, hull.equations):
        a, b, c = prim.vertices[simplex]
        n = _count(0.5 * np.linalg.norm(np.cross(b - a, c - a)), density, rng)
        r1, r2 = rng.random(n), rng.random(n)
        flip = r1 + r2 > 1
        r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
        out.append((a + r1[:, None] * (b - a) + r2[:, None] * (c - a), np.tile(eq[:3], (n, 1))))
    return out


def leaf_candidates(leaf: Leaf, density: float, rng, clip) -> Tuple[np.ndarray, np.ndarray]:
    """World points (and candidate normals, either orientation) on the surfaces that hold the leaf's boundary."""
    prim = leaf.prim
    if isinstance(prim, sp.HalfSpace):
        parts = _halfspace(leaf, rng, density, clip)
    elif isinstance(prim, sp.Box):
        parts = _box(prim, rng, density)
    elif isinstance(prim, sp.Cylinder):
        parts = _cylinder(prim, rng, density)
    elif isinstance(prim, sp.Cone):
        parts = _cone(prim, rng, density)
    elif isinstance(prim, sp.Sphere):
        parts = _sphere(prim, rng, density)
    elif isinstance(prim, sp.Torus):
        parts = _torus(prim, rng, density)
    elif isinstance(prim, sp.ConvexPolyhedron):
        parts = _polyhedron(prim, rng, density)
    else:
        raise NotImplementedError(f"no surface sampler for {type(prim).__name__}")
    parts = [(p, n) for p, n in parts if len(p)]
    if not parts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    p = np.concatenate([q for q, _ in parts])
    n = np.concatenate([m for _, m in parts])
    return leaf.to_world(p, n)


def leaf_candidate_area(leaf: Leaf, clip) -> float:
    """The area of the candidate surfaces (density 1 sampled once with a fixed generator is too noisy: use the expectation)."""
    rng = np.random.default_rng(0)
    pts, _ = leaf_candidates(leaf, 64.0 / max(_scale(leaf, clip) ** 2, 1e-12), rng, clip)
    return len(pts) / (64.0 / max(_scale(leaf, clip) ** 2, 1e-12))


def _scale(leaf: Leaf, clip) -> float:
    if isinstance(leaf.prim, sp.HalfSpace):
        return float(np.linalg.norm(clip[1] - clip[0]))
    a, b = leaf.prim.bounds()
    return float(np.linalg.norm(np.asarray(b) - np.asarray(a)))


# ---------------------------------------------------------------------------------------------- boundary samples

@dataclass
class BoundarySamples:
    points: np.ndarray       # (N, 3) on the tree's boundary
    normals: np.ndarray      # (N, 3) outward unit normals
    leaf: np.ndarray         # (N,) index of the producing (= deciding) leaf
    owner: np.ndarray        # (N,) its owner (operation index; -1 for a truth tree)
    density: float           # samples per unit area
    candidates: int          # candidate points tested

    @property
    def area(self) -> float:
        return len(self.points) / self.density

    @property
    def spacing(self) -> float:
        return float(1.0 / np.sqrt(self.density))


def boundary_samples(tree: CompiledTree, n_target: int = 200_000, seed: int = 0, clip=None,
                     delta_rel: float = 1e-6, max_rounds: int = 3) -> BoundarySamples:
    """Area-uniform samples of the tree's boundary (see the module docstring for the three tests). The density is set
    from the candidate area and raised when too few samples survive (up to `max_rounds` passes)."""
    if clip is None:
        clip = finite_bounds(tree.leaves, roles=("add", "clip")) if any(
            l.role in ("add", "clip") and not isinstance(l.prim, sp.HalfSpace) for l in tree.leaves) else None
    lo, hi = clip
    diag = float(np.linalg.norm(hi - lo))
    delta = delta_rel * diag
    total = sum(leaf_candidate_area(leaf, clip) for leaf in tree.leaves)
    density = n_target / max(total, 1e-12)
    density_total = 0.0
    keep_p, keep_n, keep_l, tested = [], [], [], 0
    for rnd in range(max_rounds):
        # passes at densities d1, d2, ... are independent uniform samplings: together uniform at d1 + d2 + ...
        rng = np.random.default_rng(seed + rnd)
        for leaf in tree.leaves:
            P, Nc = leaf_candidates(leaf, density, rng, clip)
            if not len(P):
                continue
            tested += len(P)
            Nc = Nc / np.maximum(np.linalg.norm(Nc, axis=1, keepdims=True), 1e-300)
            a, b = leaf.sdf(P + delta * Nc), leaf.sdf(P - delta * Nc)
            on_leaf = a * b < 0
            P, Nc = P[on_leaf], Nc[on_leaf]
            if not len(P):
                continue
            sp_, sm_ = tree.sdf(P + delta * Nc), tree.sdf(P - delta * Nc)
            on_tree = sp_ * sm_ < 0
            P, Nc, sp_ = P[on_tree], Nc[on_tree], sp_[on_tree]
            if not len(P):
                continue
            _, active = tree.evaluate(P)
            mine = active == leaf.index
            keep_p.append(P[mine])
            keep_n.append(np.where((sp_[mine] > 0)[:, None], Nc[mine], -Nc[mine]))
            keep_l.append(np.full(int(mine.sum()), leaf.index, np.int64))
        density_total += density
        n_kept = sum(len(k) for k in keep_p)
        if n_kept >= 0.6 * n_target or n_kept == 0:
            break
        density = density_total * (min(20.0, n_target / n_kept) - 1.0)
    pts = np.concatenate(keep_p) if keep_p else np.zeros((0, 3))
    nrm = np.concatenate(keep_n) if keep_n else np.zeros((0, 3))
    lf = np.concatenate(keep_l) if keep_l else np.zeros(0, np.int64)
    return BoundarySamples(pts, nrm, lf, tree.owners()[lf] if len(lf) else lf, float(density_total), tested)


# --------------------------------------------------------------------------------------------------- distances

def gradient(f, P: np.ndarray, h: float) -> np.ndarray:
    g = np.empty_like(P)
    for ax in range(3):
        e = np.zeros(3)
        e[ax] = h
        g[:, ax] = (f(P + e) - f(P - e)) / (2 * h)
    return g


def project(tree: CompiledTree, P: np.ndarray, h: float, iters: int = 8, tol: float = 1e-9) -> Tuple[np.ndarray, np.ndarray]:
    """Newton steps onto the tree's zero set; `ok` where the end point is on the boundary (sign changes across it)."""
    X = np.array(P, float)
    for _ in range(iters):
        s = tree.sdf(X)
        g = gradient(tree.sdf, X, h)
        gg = np.maximum((g * g).sum(1), 1e-24)
        X = X - (s / gg)[:, None] * g
    g = gradient(tree.sdf, X, h)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-300)
    a, b = tree.sdf(X + h * g), tree.sdf(X - h * g)
    ok = (a * b <= 0) & (np.abs(tree.sdf(X)) <= max(tol, 1e-3 * h))
    return X, ok


def surface_distances(P: np.ndarray, N: np.ndarray, other: CompiledTree, other_samples: BoundarySamples,
                      h: Optional[float] = None, newton: bool = True, open_tol: float = 0.0,
                      near: float = np.inf) -> Dict[str, np.ndarray]:
    """Distances from the points P (normals N) of one boundary to the OTHER tree's boundary:
    lo (|sdf|, a lower bound), hi (an upper bound: the nearest sample, improved by a Newton projection where the bracket
    is open by more than `open_tol` and lo <= `near`), signed (+: P lies outside the other solid), angle_deg between N and
    the other tree's outward normal at P (its gradient: the normal of the surface that decides there, not of a sample
    that may lie across an edge), and the owner of the nearest sample."""
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    scale = float(np.linalg.norm(np.ptp(P, axis=0))) if len(P) else 1.0
    h = h if h is not None else 1e-5 * (scale or 1.0)
    s = other.sdf(P)
    lo = np.abs(s)
    if len(other_samples.points):
        dn, idx = cKDTree(other_samples.points).query(P)
        owner = other_samples.owner[idx]
    else:
        dn, owner = np.full(len(P), np.inf), np.full(len(P), -1)
    hi = dn.copy()
    if newton and len(P):
        todo = np.flatnonzero((dn - lo > open_tol) & (lo <= near))
        if len(todo):
            X, ok = project(other, P[todo], h)
            dp = np.linalg.norm(X - P[todo], axis=1)
            better = ok & (dp < hi[todo])
            hi[todo[better]] = dp[better]
    hi = np.maximum(hi, lo)                                               # an upper bound is never below the lower one
    angle = np.full(len(P), 180.0)
    if len(P):
        g = gradient(other.sdf, P, h)
        gn = np.linalg.norm(g, axis=1)
        cosang = np.clip((N * g).sum(1) / np.maximum(gn, 1e-300), -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))
    return dict(lo=lo, hi=hi, signed=np.sign(s) * hi, angle_deg=angle, owner=owner)
