#!/usr/bin/env python3
"""
The starting shape of a reconstruction, chosen before anything else (the user's rule, 2026-09-26): a shape that is at
least partly EXACT - some of the part's real surfaces lie on it - and holds the whole part; excess is tolerated (cuts
remove it). Among the candidates that hold the part (contains >= MIN_CONTAINS, every composite operand justified, an
exact excess), `choose_start` mode "product" (the default, 2026-09-26) takes the highest score = covered share of the
generated volume x surface coverage; mode "band" (the rule before) the least excess within COVERAGE_BAND of the best
coverage. The excess needs reco_gui.model_mesh (manifold3d): without it every candidate's excess is None, nothing
passes, and residual_loop ends a run with start_shape != "legacy" with an empty program (no fallback; review R-01).

    surfaces = surface_inventory(path, field)           # the part's exact surfaces: B-rep faces (STEP) or RANSAC (STL)
    cands = start_candidates(surfaces, field, t_in, spec)   # [(name, Primitive)]
    graded = [grade_start(p, field, mesh) for ...]       # coverage, excess, contains, operands
    name, prim, table = choose_start(...)                 # or a named type (the user's input)

Coverage = the area share of the part's surface triangles whose centre lies on the candidate's surface; excess =
volume(candidate) - volume(part), exact (manifold3d solids); contains = share of the part's surface samples inside.
The "dominating cover" is the intersection of the solids of every DOMINATING surface - one with the whole part on its
material side (Shapiro-Vossler / Buchele-Crawford dominating half-spaces, the guru's L02 [3]): its boundary lies on
those surfaces by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco_engine import cascade
from reco_engine.result_schema import Primitive, StageSource
from reco_engine.result_to_program import primitive_to_sp, quaternion_of

START_TYPES = ("auto", "legacy", "dominating", "dominating major", "cascade", "oriented box", "world box", "cylinder",
               "cylinder sector", "sphere", "box & cylinder",
               "cubes",              # the voxel-cube cover (residual_loop._cube_cover_add), not a candidate of this stage
               "compartments",       # the mass compartments (residual_loop._compartment_add), not a candidate either
               "leaves")             # the recognized leaves (residual_loop._leaves_add, reco_engine/leaf_recognition.py)
# not offered in the window since the method grading of 2026-10-30 (D: cubes 1 rescued / 8 broken, compartments good
# 29 -> 1 on the gate); run_part --start still takes them
DROPPED_STARTS = ("cubes", "compartments")
COVERAGE_BAND = 0.05          # candidates within this of the best coverage compete on excess
MIN_CONTAINS = 0.995          # a starting shape holds the part
MIN_OPERAND_SHARE = 0.01      # every operand of a composite carries at least this share of the part's surface
EXACT_LABEL = "dominating surface"   # an operand made from an exact surface of the inventory: never varied


@dataclass
class Surface:
    """One exact surface of the part (several B-rep faces may lie on it)."""
    kind: str                                   # "plane" | "cylinder" | "sphere" | "cone" | "torus" | "other"
    area_share: float
    dominating: bool
    params: Dict = dc_field(default_factory=dict)   # plane: normal (outward), point; cylinder: axis, center, radius, convex


def _frame(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float) / np.linalg.norm(z)
    u = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(z, u), z], axis=1)


# ------------------------------------------------------------------------------------------------ the inventory

def surface_inventory(path, field, tol_rel: float = 1e-3, channels=None, spec=None) -> Tuple[List[Surface], Dict]:
    """(surfaces, mesh): the part's exact surfaces and its surface triangles (vertices, triangles, face_id or None).
    STEP: the B-rep faces (reco_engine.brep.step_faces), grouped by the surface they lie on. STL: with the whole part's
    voxel channels and grid (`channels`, `spec`) first the cylinders and spheres of the strict RANSAC
    (reco_engine.cascade.strict_cylinders / strict_spheres), each taking the triangles on it, then planes by RANSAC over
    the rest; without them planes only. Area shares from the mesh."""
    from reco_engine.brep import is_step, step_faces
    V, F = np.asarray(field.vertices, float), np.asarray(field.faces)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = tol_rel * diag
    if path is not None and is_step(path):
        inv = step_faces(path)
        total = float(sum(f["area"] for f in inv["faces"])) or 1.0
        groups: Dict[tuple, Surface] = {}
        for f in inv["faces"]:
            t = f["type"]
            if t == "Plane":
                n = np.asarray(f["normal"], float)
                key = ("plane", tuple(np.round(n, 3)), round(float(n @ np.asarray(f["point"], float)) / tol))
                params = dict(normal=n.tolist(), point=f["point"])
                kind = "plane"
            elif t in ("Cylinder", "Sphere"):
                kind = t.lower()
                c = np.asarray(f["center"], float)
                if kind == "cylinder":
                    a = np.asarray(f["axis"], float)
                    a = a if a[np.argmax(np.abs(a))] > 0 else -a
                    foot = c - (c @ a) * a                                   # the axis line's point nearest the origin
                    key = ("cylinder", tuple(np.round(a, 3)), tuple(np.round(foot / tol)), round(f["radius"] / tol))
                    params = dict(axis=a.tolist(), center=c.tolist(), radius=f["radius"], convex=f.get("convex"))
                else:
                    key = ("sphere", tuple(np.round(c / tol)), round(f["radius"] / tol))
                    params = dict(center=c.tolist(), radius=f["radius"], convex=f.get("convex"))
            else:
                kind = {"Cone": "cone", "Toroid": "torus"}.get(t, "other")
                key = (kind, f["id"])
                params = {k: f[k] for k in ("axis", "center", "radius", "apex", "semi_angle", "major_radius",
                                              "minor_radius") if k in f}
            g = groups.get(key)
            if g is None:
                groups[key] = Surface(kind, f["area"] / total, bool(f["dominating"]), params)
            else:
                g.area_share += f["area"] / total
                g.dominating = g.dominating or bool(f["dominating"])
        surfaces = sorted(groups.values(), key=lambda s: -s.area_share)
        mesh = dict(vertices=inv["vertices"], triangles=inv["triangles"], face_id=inv["face_id"], volume=inv["volume"])
        return surfaces, mesh
    # STL: strict RANSAC on the whole surface
    surfaces = []
    samples = cascade._field_samples(field)
    P, N = np.asarray(samples.surface, float), np.asarray(samples.normals, float)
    tri = V[F]
    raw_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(raw_n, axis=1)
    tri_n = raw_n / np.maximum(area[:, None] * 2.0, 1e-12)
    cent = tri.mean(axis=1)
    taken = np.zeros(len(F), bool)                                          # triangles on a cylinder / sphere
    if channels is not None and spec is not None:
        from reco_engine.result_to_program import rotation_matrix
        found = [("cylinder", p) for p in cascade.strict_cylinders(channels, spec, field, top=6, n_hyp=3000)]
        found += [("sphere", p) for p in cascade.strict_spheres(channels, spec, field)]
        for kind, p in found:
            c = np.asarray(p.center_mm, float)
            r = float(p.dims_mm["radius"])
            if kind == "cylinder":
                a = rotation_matrix(p)[:, 2]
                rel = cent - c
                radial = rel - np.outer(rel @ a, a)
                rho = np.linalg.norm(radial, axis=1)
                on = (np.abs(rho - r) < tol) & (np.abs(tri_n @ a) < 0.3) & ~taken
                # a real cylinder wraps around its axis: its triangles cover several azimuths; a plane tangent to a
                # spurious RANSAC cylinder touches it along one line only (a box face gave r = 8.4 along x, 14 % area)
                u = _frame(a)[:, 0]
                v = np.cross(a, u)
                az = np.arctan2(radial[on] @ v, radial[on] @ u)
                hist = np.bincount(((az + np.pi) / (2 * np.pi / 12)).astype(int) % 12, weights=area[on], minlength=12)
                if (hist > 0.005 * max(area[on].sum(), 1e-12)).sum() < 4:
                    continue
                outward = np.einsum("ij,ij->i", tri_n[on], radial[on] / np.maximum(rho[on, None], 1e-12))
                rel_v = V - c
                rho_v = np.linalg.norm(rel_v - np.outer(rel_v @ a, a), axis=1)
                params = dict(axis=a.tolist(), center=c.tolist(), radius=r)
            else:
                rel = cent - c
                rho = np.linalg.norm(rel, axis=1)
                on = (np.abs(rho - r) < tol) & ~taken
                outward = np.einsum("ij,ij->i", tri_n[on], rel[on] / np.maximum(rho[on, None], 1e-12))
                rho_v = np.linalg.norm(V - c, axis=1)
                params = dict(center=c.tolist(), radius=r)
            share = float(area[on].sum() / area.sum())
            if share < 0.005:
                continue
            convex = bool(np.median(outward) > 0)                            # material towards the axis / centre
            dom = bool(convex and (rho_v <= r + 2 * tol).all())               # the whole part inside it
            params["convex"] = convex
            taken |= on
            surfaces.append(Surface(kind, share, dom, params))
    rng = np.random.default_rng(0)
    for i in rng.choice(len(P), size=min(400, len(P)), replace=False):     # planes: point + normal hypotheses
        n = N[i] / max(np.linalg.norm(N[i]), 1e-12)
        on = np.abs((cent - P[i]) @ n) < tol
        on &= (raw_n @ n) > 0
        on &= ~taken
        share = float(area[on].sum() / area.sum())
        if share < 0.005 or any(s.kind == "plane" and abs(np.asarray(s.params["normal"]) @ n) > 0.999
                                and abs((np.asarray(s.params["point"]) - P[i]) @ n) < 2 * tol for s in surfaces):
            continue
        dom = bool(((V - P[i]) @ n).max() <= tol)
        surfaces.append(Surface("plane", share, dom, dict(normal=n.tolist(), point=P[i].tolist())))
    surfaces.sort(key=lambda s: -s.area_share)
    mesh = dict(vertices=V, triangles=F, face_id=None, volume=abs(float(np.einsum(
        "ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0))
    return surfaces, mesh


# ------------------------------------------------------------------------------------------------ candidates

def _extent_along(V: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
    t = V @ axis
    return float(t.min()), float(t.max())


def _surface_solid(s: Surface, V: np.ndarray, pad: float) -> Optional[Primitive]:
    """The solid side of one dominating surface as a Primitive holding the part: a plane -> a half-space; a convex
    cylinder -> the cylinder spanning the part along its axis; a convex sphere -> the sphere."""
    if s.kind == "plane":
        n = np.asarray(s.params["normal"], float)
        return Primitive(kind="halfspace", quaternion=quaternion_of(_frame(n)), center_mm=tuple(s.params["point"]), dims_mm={},
                         label=EXACT_LABEL)
    if s.kind == "cylinder" and s.params.get("convex"):
        a = np.asarray(s.params["axis"], float); a /= np.linalg.norm(a)
        c = np.asarray(s.params["center"], float)
        t0, t1 = _extent_along(V - c, a)
        centre = c + a * (t0 + t1) / 2.0
        return Primitive(kind="cylinder", quaternion=quaternion_of(_frame(a)), center_mm=tuple(centre),
                         dims_mm=dict(radius=float(s.params["radius"]), half_length=(t1 - t0) / 2.0 + pad), label=EXACT_LABEL)
    if s.kind == "sphere" and s.params.get("convex"):
        return Primitive(kind="sphere", center_mm=tuple(s.params["center"]), dims_mm=dict(radius=float(s.params["radius"])),
                         label=EXACT_LABEL)
    return None


def _chain(parts: Sequence[Primitive], label: str) -> Optional[Primitive]:
    """parts[0] & parts[1] & ... as nested composites."""
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = Primitive(kind="composite", center_mm=out.center_mm, dims_mm={}, parts=[out, p], combine="common")
    out.label = label
    return out


def dominating_cover(surfaces: Sequence[Surface], field, obb: Optional[Primitive], max_operands: int = 12,
                     min_share: float = 0.0) -> Optional[Primitive]:
    """The intersection of the solid sides of the dominating surfaces (largest area first, at most `max_operands`, each
    with at least `min_share` of the part's surface), closed by the oriented box where the surfaces leave it open."""
    V = np.asarray(field.vertices, float)
    pad = 1e-3 * float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    parts = [s for s in surfaces if s.dominating and s.area_share >= min_share]
    solids = [p for p in (_surface_solid(s, V, pad) for s in parts[:max_operands]) if p is not None]
    if not solids:
        return None
    if obb is not None:
        import copy
        closing = copy.copy(obb)
        closing.label = "closing box"                                # bounds whatever the surfaces leave open; exempt
        solids.append(closing)                                       # from the operand justification
    return _chain(solids, "dominating cover")


def start_candidates(surfaces, field, t_in, spec, registry=None, first=None) -> List[Tuple[str, Primitive]]:
    """Every starting shape: the dominating cover (all dominating surfaces, and only those with >= 5 % of the area),
    the covering modes of cascade.bounding_candidates, and the cascade's answer when given."""
    out: List[Tuple[str, Primitive]] = []
    base = cascade.bounding_candidates(t_in, spec, field=field)
    obb = next((p for n, p in base if n == "oriented box"), None)
    dom = dominating_cover(surfaces, field, obb)
    if dom is not None:
        out.append(("dominating", dom))
    dom_major = dominating_cover(surfaces, field, obb, min_share=0.05)
    if dom_major is not None:
        dom_major.label = "dominating cover (surfaces >= 5 %)"
        out.append(("dominating major", dom_major))
    out += [({"oriented box": "oriented box", "world-axis box": "world box", "bounding cylinder": "cylinder",
              "bounding cylinder sector": "cylinder sector", "bounding sphere": "sphere",
              "box & cylinder": "box & cylinder"}.get(n, n), p) for n, p in base]
    if first is not None and not first.is_unknown:
        out.append(("cascade", first))
    return out


# ------------------------------------------------------------------------------------------------ grading

def _operands(p: Primitive) -> int:
    return sum(_operands(q) for q in p.parts) if p.kind == "composite" else 1


def _leaves(p: Primitive) -> List[Primitive]:
    return [x for q in p.parts for x in _leaves(q)] if p.kind == "composite" else [p]


def _sdf_mesh(q: Primitive):
    """A primitive manifold3d cannot build analytically (a cylinder sector, a torus) from its own SDF."""
    try:
        from reco_field import peel
        from reco_gui.field_session import MeshData
        v, f = peel.mesh_from_sdf(primitive_to_sp(q), 96)
        return MeshData(v, f) if len(f) else None
    except Exception:
        return None


def grade_start(p: Primitive, field, mesh: Dict, tol_rel: float = 2e-3, h: Optional[float] = None,
                fast: bool = False, max_triangles: int = 20000) -> Dict:
    """coverage (area share of the part's surface on p's surface), excess (volume(p) - volume(part), exact),
    contains (share of the part's surface samples inside p), covered (|P & M| / |M|), score = covered x coverage, operands.
    With a scale `h` also the SOFT coverage of the band B_h (docs/thesis/chapters/10b_solution_spaces.tex, eq. sobolev):
    every triangle weighted by (1 - (d / 2h)^2)+ x (n_tri . n_p)+ - position and orientation - and score_soft; smooth in the
    parameters where the strict coverage is a step function (an oversized cylinder is rewarded for shrinking)."""
    V, T = np.asarray(mesh["vertices"], float), np.asarray(mesh["triangles"])
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    tol = tol_rel * diag
    sdf = primitive_to_sp(p)
    if fast and len(T) > max_triangles:                                     # the variation's trials: a fixed random subset
        T = T[np.random.default_rng(0).choice(len(T), size=max_triangles, replace=False)]
    tri = V[T]
    raw_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(raw_n, axis=1)
    cent = tri.mean(axis=1)
    d = sdf.sdf(cent)
    on = np.abs(d) <= tol
    coverage = float(area[on].sum() / area.sum())
    coverage_soft = None
    if h is not None and h > 0:
        eps = 1e-3 * diag
        grad = np.stack([(sdf.sdf(cent + eps * e) - sdf.sdf(cent - eps * e)) / (2 * eps)
                         for e in np.eye(3)], axis=1)
        n_p = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)
        n_t = raw_n / np.maximum(2.0 * area[:, None], 1e-12)
        # quadratic kernel of width 2h (the L2 position term): strictly concave, so a shape centred on the surface
        # (uniform small d) beats one touching it on one side (d spread 0..2h), which a linear kernel scores equally
        w = np.clip(1.0 - (d / (2.0 * h)) ** 2, 0.0, None) * np.clip(np.einsum("ij,ij->i", n_t, n_p), 0.0, None)
        coverage_soft = float((area * w).sum() / area.sum())
    # every operand of a composite must carry part of the part's surface itself (its own surface there, and the whole
    # candidate's too): box & cylinder on a stepped box trims excess with a cylinder the part never had (2026-09-26)
    unjustified = []
    if p.kind == "composite":
        centres = tri.mean(axis=1)
        leaves = _leaves(p)
        on_leaf = [np.abs(primitive_to_sp(q).sdf(centres)) <= tol for q in leaves]
        for k, leaf in enumerate(leaves):
            if leaf.label == "closing box":
                continue
            # its UNIQUE share: on it and on no other operand (a cylinder's end caps in the box's face planes do not count)
            others = np.zeros_like(on)
            for m, q in enumerate(leaves):
                if m != k and q.label != "closing box":
                    others |= on_leaf[m]
            share = float(area[on & on_leaf[k] & ~others].sum() / area.sum())
            if share < MIN_OPERAND_SHARE:
                unjustified.append(leaf.kind)
    contains = float(np.mean(sdf.sdf(V) <= tol))
    excess, covered, solid = None, None, None
    try:
        from reco_gui.model_mesh import primitive_solid
        lo, hi = V.min(axis=0) - 0.5 * diag, V.max(axis=0) + 0.5 * diag
        solid = primitive_solid(p, sdf_mesh=_sdf_mesh, crop=(lo, hi))
        if solid is not None:
            excess = float(solid.volume() - mesh["volume"])
    except ImportError:
        pass
    # covered: the share of the candidate's volume that is the part's, |P & M| / |M| - exact where the part's mesh is a
    # manifold (the examples' STLs, the STEP tessellation), else contains x |P| / |M|
    if solid is not None and solid.volume() > 0 and not fast:                # fast: the proxy below (no exact intersection)
        try:
            import manifold3d as m3d
            pm = m3d.Mesh(vert_properties=V.astype(np.float32), tri_verts=T.astype(np.uint32))
            pm.merge()
            part = m3d.Manifold(pm)
            if part.volume() > 0.5 * mesh["volume"]:
                covered = float((solid ^ part).volume() / solid.volume())
        except Exception:
            covered = None
    if covered is None and mesh["volume"] and solid is not None:               # the proxy (fast, or no manifold part)
        covered = float(contains * mesh["volume"] / (mesh["volume"] + max(excess or 0.0, 0.0)))
    # the objective of 2026-09-26 (the user's rule): the covered share of the generated volume times the surface coverage;
    # 1 for the part itself, 1 / (1 + excess share) x coverage for a start that holds the part
    score = float((covered or 0.0) * coverage)
    return dict(coverage=coverage, contains=contains, excess=excess, unjustified=unjustified,
                excess_share=(excess / mesh["volume"]) if excess is not None and mesh["volume"] else None,
                covered=covered, score=score, coverage_soft=coverage_soft,
                score_soft=(float((covered or 0.0) * coverage_soft) if coverage_soft is not None else None),
                operands=_operands(p))


def choose_start(graded: List[Tuple[str, Primitive, Dict]], band: float = COVERAGE_BAND,
                 min_contains: float = MIN_CONTAINS, mode: str = "product") -> Optional[Tuple[str, Primitive, Dict]]:
    """Among the candidates holding the part (every operand justified): mode "product" (2026-09-26) - the highest
    score = covered share of the generated volume x surface coverage, ties to fewer operands; mode "band" (the rule
    before) - those within `band` of the best coverage, of them the least excess, ties to fewer operands."""
    ok = [g for g in graded if g[2]["contains"] >= min_contains and g[2]["excess"] is not None
          and not g[2].get("unjustified")]
    if not ok:
        return None
    if mode == "product":
        return max(ok, key=lambda g: (round(g[2]["score"], 6), -g[2]["operands"]))
    best = max(g[2]["coverage"] for g in ok)
    top = [g for g in ok if g[2]["coverage"] >= best - band]
    return min(top, key=lambda g: (round(g[2]["excess"], 9), g[2]["operands"]))


# ------------------------------------------------------------------------------------------------ variation in the space

def _perturbations(p: Primitive, h: float, diag: float):
    """The neighbours of a parametric shape at scale h (docs/thesis/chapters/10b_solution_spaces.tex, def. guess): the
    centre moved by h along each world axis, every size changed by h, the frame turned by h / (its largest size) about each
    world axis; a half-space moved by h along its normal; a composite: each operand varied in turn. (label, primitive)."""
    import copy
    from reco_engine.result_to_program import quaternion_of, rotation_matrix
    from scipy.spatial.transform import Rotation
    if p.label == EXACT_LABEL:                                             # exact by construction: nothing to vary
        return
    if p.kind == "composite":
        for i, part in enumerate(p.parts):
            for label, q in _perturbations(part, h, diag):
                new = copy.copy(p)
                new.parts = list(p.parts)
                new.parts[i] = q
                yield f"operand {i} {label}", new
        return
    c = np.asarray(p.center_mm, float)
    if p.kind == "halfspace":
        n = rotation_matrix(p)[:, 2]
        for sign in (-1.0, 1.0):
            q = copy.copy(p); q.center_mm = tuple(float(x) for x in c + sign * h * n)
            yield f"offset {sign * h:+.3g}", q
    else:
        for ax in range(3):
            for sign in (-1.0, 1.0):
                q = copy.copy(p); cc = c.copy(); cc[ax] += sign * h
                q.center_mm = tuple(float(x) for x in cc)
                yield f"centre[{ax}] {sign * h:+.3g}", q
        for key, val in p.dims_mm.items():
            if not isinstance(val, (int, float)):
                continue
            for sign in (-1.0, 1.0):
                if val + sign * h <= 0.25 * h:
                    continue
                q = copy.copy(p); q.dims_mm = dict(p.dims_mm); q.dims_mm[key] = float(val + sign * h)
                yield f"{key} {sign * h:+.3g}", q
    if p.kind == "sphere":
        return
    sizes = [v for v in p.dims_mm.values() if isinstance(v, (int, float))]
    length = max(sizes) if sizes else diag
    angle = h / max(length, h)
    R = rotation_matrix(p)
    for ax in range(3):
        for sign in (-1.0, 1.0):
            e = np.zeros(3); e[ax] = 1.0
            q = copy.copy(p)
            q.quaternion = quaternion_of(Rotation.from_rotvec(sign * angle * e).as_matrix() @ R)
            yield f"turn[{ax}] {np.degrees(sign * angle):+.2f}deg", q


def vary_start(p: Primitive, field, mesh: Dict, h: float, rounds: int = 3, min_contains: float = MIN_CONTAINS,
               min_gain: float = 0.005) -> Tuple[Primitive, Dict, List[str]]:
    """Coordinate search in the shape's parameter space at scale h on the SOFT score of the band B_h (grade_start with
    h, fast: volume proxy, subsampled triangles): per round every neighbour of _perturbations is graded and the best one
    taken if its soft score is at least `min_gain` (relative) higher; operands made from an exact surface of the
    inventory (EXACT_LABEL) are never varied - tilting the exact planes of the liner's cover by 0.3-0.7 degrees for
    +0.006 of soft score broke the final solid (2026-09-27); at most `rounds` rounds; a start that held the part must
    keep holding it. (best, grade of the best re-computed exactly, steps)."""
    V = np.asarray(mesh["vertices"], float)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    best, g, steps = p, grade_start(p, field, mesh, h=h, fast=True), []
    for _ in range(rounds):
        cand = None
        for label, q in _perturbations(best, h, diag):
            gq = grade_start(q, field, mesh, h=h, fast=True)
            if g["contains"] >= min_contains and gq["contains"] < min_contains:
                continue
            if gq.get("unjustified") and not g.get("unjustified"):
                continue
            if gq["score_soft"] >= g["score_soft"] * (1.0 + min_gain) + 1e-6 and \
                    (cand is None or gq["score_soft"] > cand[1]["score_soft"]):
                cand = (q, gq, label)
        if cand is None:
            break
        best, g = cand[0], cand[1]
        steps.append(f"{cand[2]}: {g['score_soft']:.4f}")
    if steps:
        g = grade_start(best, field, mesh, h=h)
    return best, g, steps
