#!/usr/bin/env python3
"""
Grouping the pieces of a difference by TOOL TYPE (the user's request 2026-09-28: "the feature mode should have a
FeatureNet which groups the found features, and based on the grouping it uses a tool, creates one by itself, or ...
recycles an old tool of the session"). Per piece: which tool type (reco_engine.tools.CLASSES), how sure, and which
other pieces share it - same type and congruent: one GROUP (one guess, the others by pattern transfer); same type,
another size: one FAMILY.

    types = classify_piece(piece, V, op)                 # [(tool type, confidence)] ranked, source "analytic"/"fused"
    out = group_pieces(pieces, meshes, ops, net=None)     # dict(groups, ungrouped, per_piece)

Two sources:

  A, analytic (works today): the piece's exact faces -> a descriptor (`describe`: faces by kind and convexity, the
    coaxial stack of its cylinders and cones, its planes' directions, extents) -> a RULE TABLE (`RULES`, each rule
    documented where it is written) that says which tool families the geometry allows -> the catalogue's fitters
    (tools.fit_piece) rank the types they can read from those faces. A piece no rule admits is "none" (ungrouped).
  B, the network ("FeatureNet"): `FeatureNet.classify(piece_tensor) -> [(type, probability)]` over tools.CLASSES on the
    piece's standard input (`piece_tensor`). The stub `NoOpinion` answers nothing; another agent trains the real one.

Fusion (`fuse`): B only RE-RANKS A's candidates - a type that A's geometry excludes is dropped whatever B says (a
network that hallucinates on an out-of-scope piece must not cost a wrong cut; the exact FreeCAD check stays the judge
of every applied tool anyway).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco_engine import tools as T

REAL = T.REAL


# ------------------------------------------------------------------------------------------------ source A

def describe(piece: Dict, V: Optional[np.ndarray] = None) -> Dict:
    """The analytic descriptor of a piece from its exact faces: counts of real faces by kind and convexity, the coaxial
    stack (tools.revolved_segments: its segments, inner / outer, the planes square to its axis), the number of
    distinct plane directions and of parallel plane pairs, the real cylinders parallel to a plane direction, planes
    inclined to the others, and the extents' ratios (sorted, from the vertices' principal axes)."""
    faces = piece.get("faces") or []
    real = [f for f in faces if f["origin"] in REAL]
    counts: Dict[str, int] = {}
    for f in real:
        key = f["type"] + ("" if f["type"] == "plane" else ("+" if f.get("convex", True) else "-"))
        counts[key] = counts.get(key, 0) + 1
    stack = T.revolved_segments(faces)
    coaxial = 0
    if stack is not None:                                       # on the stack's axis LINE, not only parallel to it
        a, b = stack["axis"], stack["base"]

        def on_line(f):
            off = np.asarray(f["center"], float) - b
            off = off - a * (off @ a)
            return abs(T._unit(f["axis"]) @ a) >= T.AX_COS and np.linalg.norm(off) <= 1e-3 * max(float(f.get("radius") or 1.0), 1.0) + 1e-6
        coaxial = sum(1 for f in real if f["type"] in ("cylinder", "cone") and on_line(f))
    planes = [f for f in faces if f["type"] == "plane"]
    dirs: List[np.ndarray] = []
    for f in planes:
        n = T._unit(f["normal"])
        if not any(abs(n @ d) >= T.AX_COS for d in dirs):
            dirs.append(n)
    pairs = sum(1 for i, a in enumerate(dirs) for b in dirs[i + 1:] if abs(a @ b) >= T.AX_COS)
    orth = [d for d in dirs if any(abs(d @ e) < 0.02 for e in dirs if e is not d)]
    inclined = sum(1 for f in real if f["type"] == "plane" and any(0.2 < abs(T._unit(f["normal"]) @ d) < 0.98 for d in dirs))
    cyl_par = 0
    for f in real:
        if f["type"] == "cylinder" and any(abs(T._unit(f["axis"]) @ d) >= T.AX_COS for d in dirs):
            cyl_par += 1
    ext = None
    if V is not None and len(V) >= 4:
        X = np.asarray(V, float) - np.mean(V, axis=0)
        _, vec = np.linalg.eigh(X.T @ X)
        e = np.sort(np.ptp(X @ vec, axis=0))[::-1]
        ext = (e / max(e[0], 1e-12)).tolist()
    curved_real = [f for f in real if f["type"] in ("cylinder", "cone", "sphere", "torus", "other")]
    return dict(counts=counts, n_real=len(real), n_faces=len(faces), stack=stack, coaxial=coaxial,
                all_curved_coaxial=bool(curved_real) and coaxial == len([f for f in curved_real if f["type"] in ("cylinder", "cone")])
                and all(f["type"] in ("cylinder", "cone") for f in curved_real),
                plane_dirs=len(dirs), parallel_pairs=pairs, orthogonal_dirs=len(orth), inclined_planes=inclined,
                cylinders_parallel_to_planes=cyl_par, extents=ext,
                other_surfaces=sum(1 for f in real if f["type"] in ("sphere", "torus", "other")))


# The rule table: descriptor -> the tool families its geometry allows. Every rule is a necessary condition read off
# the tools' own geometry (tools.py); the fitters then decide the type within the family.
RULES: Tuple[Tuple[str, str, object], ...] = (
    # a revolved tool: every real curved face is a cylinder or cone on ONE axis, no sphere / torus / free-form face
    ("revolved", "hole, circular_pocket, taper, groove, relief",
     lambda d: d["all_curved_coaxial"] and d["other_surfaces"] == 0),
    # a pocket / slot: planes in at least two directions square to each other (walls and floor), real cylinders only
    # parallel to a plane direction (the rounded depth-parallel edges), no inclined plane
    ("prismatic", "rect_pocket, slot, through_slot, open_pocket",
     lambda d: d["orthogonal_dirs"] >= 2 and d["inclined_planes"] == 0 and d["other_surfaces"] == 0
     and d["counts"].get("cone+", 0) + d["counts"].get("cone-", 0) == 0),
    # an edge tool: at most one real cylinder (a fillet / round) or one inclined real plane (a chamfer), long along the
    # edge (the largest extent at least twice the second)
    ("edge", "fillet, round, chamfer",
     lambda d: d["other_surfaces"] == 0 and (d["counts"].get("cylinder+", 0) + d["counts"].get("cylinder-", 0) <= 2)
     and (d["extents"] is None or d["extents"][1] <= 0.5)),
    # a box with finished edges: three orthogonal plane directions with parallel pairs, and rounds (cylinders parallel
    # to a plane direction) or chamfers (inclined planes)
    ("box edges", "chamfered_box, filleted_box",
     lambda d: d["orthogonal_dirs"] >= 3 and d["parallel_pairs"] >= 2 and (d["cylinders_parallel_to_planes"] > 0
                                                                           or d["inclined_planes"] > 0)),
)

FAMILY_TYPES = {
    "revolved": ("through_hole", "blind_hole", "countersunk_hole", "counterbored_hole", "stepped_hole",
                 "circular_pocket", "taper", "groove", "relief"),
    "prismatic": ("slot", "through_slot", "rect_pocket", "open_pocket"),
    "edge": ("fillet", "round", "chamfer"),
    "box edges": ("filleted_box", "chamfered_box"),
}


def allowed_types(desc: Dict) -> List[str]:
    """The tool types the rule table admits for this descriptor, in the table's order."""
    out: List[str] = []
    for name, _, rule in RULES:
        try:
            ok = bool(rule(desc))
        except (KeyError, TypeError, IndexError):
            ok = False
        if ok:
            out += [t for t in FAMILY_TYPES[name] if t not in out]
    return out


def classify_analytic(piece: Dict, V: np.ndarray, op: str = "cut") -> Tuple[List[Tuple[str, float]], List[T.Fit]]:
    """Source A: [(type, confidence)] ranked, and the fits behind them. Confidence 1.0 for a type the fitters read
    from the faces (in their order), 0.5 for a type the rules allow but no fitter read, none else."""
    desc = describe(piece, V)
    allowed = allowed_types(desc)
    fits = T.fit_piece(piece, V, op, types=allowed) if allowed else []
    ranked: List[Tuple[str, float]] = []
    for f in fits:
        if f.tool not in [t for t, _ in ranked]:
            ranked.append((f.tool, 1.0))
    ranked += [(t, 0.5) for t in allowed if t not in [r for r, _ in ranked]]
    return ranked, fits


# ------------------------------------------------------------------------------------------------ source B

PIECE_GRID = 64                     # the network's grid: 64^3 cells, as the Type nets' in-scale input


def piece_tensor(piece: Dict, inside_region, lo, hi, V: Optional[np.ndarray] = None, F: Optional[np.ndarray] = None):
    """The FeatureNet's input for one piece - the project's standard "voxfield" (training/inputs.py on branch
    combined-input, imported lazily; None while that module is not on this branch): (6, 64, 64, 64) float32 in the
    type nets' array frame [z, -y, x], channels material, unit normal (3), d_log, inside (the module's docstring).
    Grid: 64^3 cells, a cube centred on the piece's box, pitch = 1.1 x its longest side / 64 (the piece fills the
    grid as the in-scale samples fill theirs); the material and inside bits are the piece's region (the excess
    predicate, not the piece's own mesh: a thin piece's tessellation can be open); the field channels come from the
    fallback path (voxels only) - the exact path needs a closed piece mesh. The pitch the piece's thinnest wall
    licenses (tau / 3, D-005) is recorded next to it by the caller; a piece thinner than that is out of the net's
    scope."""
    try:
        from training.inputs import voxfield_from_world
    except ImportError:
        return None
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    side = 1.1 * float((hi - lo).max())
    pitch = side / PIECE_GRID
    c = (lo + hi) / 2
    ax = [c[d] - side / 2 + (np.arange(PIECE_GRID) + 0.5) * pitch for d in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    occ = np.asarray(inside_region(P), bool).reshape((PIECE_GRID,) * 3)
    return voxfield_from_world(occ, occ, None, None, pitch)


class FeatureNet:
    """The network's interface: classify(tensor) -> [(tool type, probability)] over tools.CLASSES (probabilities sum
    to 1 over the classes it names; "none" = none of the catalogue)."""
    classes: Tuple[str, ...] = T.CLASSES

    def classify(self, tensor) -> List[Tuple[str, float]]:
        raise NotImplementedError


class NoOpinion(FeatureNet):
    """The stub: no opinion on any piece (the analytic source decides alone)."""

    def classify(self, tensor) -> List[Tuple[str, float]]:
        return []


def fuse(analytic: List[Tuple[str, float]], net: List[Tuple[str, float]]) -> Tuple[List[Tuple[str, float]], str]:
    """B re-ranks A: among the types a fitter READ from the piece's faces (A's confidence 1.0) B's probability decides
    the order; the types the rules only admit (0.5) follow in A's order; a type A excludes is dropped whatever B says -
    and B cannot lift a merely admitted type above a read one (2026-09-28: a net naming "groove" for a countersunk
    hole - admitted by the revolved rule, read by no fitter - must not come first). (ranked, "analytic" | "fused")."""
    probs = {t: p for t, p in net if t in T.CLASSES and t != "none"}
    if not probs or not analytic:
        return list(analytic), "analytic"
    read = [(t, c) for t, c in analytic if c >= 1.0]
    rest = [(t, c) for t, c in analytic if c < 1.0]
    read.sort(key=lambda tc: -probs.get(tc[0], -1.0))
    return [(t, float(probs.get(t, c))) for t, c in read] + rest, "fused"


# ------------------------------------------------------------------------------------------------ the groups

@dataclass
class PieceType:
    id: int
    ranked: List[Tuple[str, float]]
    fits: List[T.Fit]
    source: str

    @property
    def tool_type(self) -> Optional[str]:
        """The best-ranked type a fitter READ from the piece's faces; None when the rules only admit types (a piece
        is grouped by what its faces show, not by what they allow)."""
        read = {f.tool for f in self.fits}
        return next((t for t, _ in self.ranked if t in read), None)

    @property
    def confidence(self) -> float:
        t = self.tool_type
        return float(dict(self.ranked).get(t, 0.0)) if t else 0.0


def group_pieces(entries: Sequence[Dict], net: Optional[FeatureNet] = None, tensors: Optional[Dict] = None) -> Dict:
    """entries: [dict(id, piece, V, op, descriptor)] (descriptor: feature_stage.piece_descriptor or None). Returns
    dict(per_piece {id: PieceType}, groups [dict(id, tool_type, family, members, representative, confidence, source)],
    ungrouped [ids]). A group: the same type and congruent (feature_stage.congruent); a family: the same type, the
    groups of other sizes (named by the type: all groups of a type form its family)."""
    from reco_engine.feature_stage import congruent
    net = net or NoOpinion()
    per: Dict[int, PieceType] = {}
    for e in entries:
        ranked, fits = classify_analytic(e["piece"], e["V"], e["op"])
        opinion = []
        t = (tensors or {}).get(e["id"])
        if t is not None:
            try:
                opinion = net.classify(t)
            except Exception:                                    # the network is advisory
                opinion = []
        ranked, source = fuse(ranked, opinion)
        per[e["id"]] = PieceType(e["id"], ranked, fits, source)
    groups: List[Dict] = []
    ungrouped: List[int] = []
    for e in entries:
        pt = per[e["id"]]
        if pt.tool_type is None:
            ungrouped.append(e["id"])
            continue
        for g in groups:
            rep = next(x for x in entries if x["id"] == g["representative"])
            if g["tool_type"] == pt.tool_type and congruent(rep.get("descriptor"), e.get("descriptor")):
                g["members"].append(e["id"])
                break
        else:
            groups.append(dict(id=len(groups) + 1, tool_type=pt.tool_type, family=pt.tool_type, members=[e["id"]],
                               representative=e["id"], confidence=pt.confidence, source=pt.source))
    return dict(per_piece=per, groups=groups, ungrouped=ungrouped)
