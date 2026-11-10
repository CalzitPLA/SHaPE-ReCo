#!/usr/bin/env python3
"""
The difference grade of a shape guess: not how MUCH of the input the guess misses, but WHAT the miss consists of.

Two guesses with the same IoU can be in very different states. One is off by a fraction of a millimetre everywhere
(its difference to the input is a thin skin: the parameters need refining, the shape is right). The other is exact
except for four bolt holes it does not have (its difference is four compact pieces, each a cylinder: four operations
are missing). A third lacks the fillets no primitive describes. The IoU cannot tell them apart; the pieces can.

Input: the pieces of the exact difference (reco_field.exact_check.exact_residual, pieces_dir), i.e. the solids of
D+ = input - guess (missed) and D- = guess - input (invented). Per piece C:

    volume  v_C
    depth   delta_C   the largest distance of a point of the piece from the OTHER solid's surface: a piece of D+
                      from the guess's surface, a piece of D- from the input's. A skin of an offset eps has depth
                      eps, however large its area; a bore of radius r has depth r (on its axis)
    score   sigma_C   the volume IoU of the best primitive for the piece (the feature stage's), if known

and its class, with the tolerance delta_0. Default: the smaller of 1e-3 of the input's diagonal and a tenth of its
thinnest wall tau (5th percentile, dataset_gen.src.local_grids.wall_thickness). The diagonal alone is wrong for a
large slender part (measured on the frame, 8.4 m diagonal: delta_0 = 8.4 mm made all 60 root fillets, 7.9 mm deep,
"skin", and the grade promised IoU 1.0 for a part that is not solved):

    skin     delta_C <= delta_0                          a parameter error: refine
    feature  delta_C >  delta_0 and sigma_C >= 0.9       a missing operation the vocabulary has: apply
    open     else                                        no primitive describes it: the vocabulary's gap

Because the pieces are disjoint, the IoU's defect splits exactly over the classes (U = the union of input and guess):

    1 - IoU = e_skin + e_feature + e_open,        e_x = sum of v_C over class x / vol(U)

The grade is the vector (IoU, e_skin, e_feature, e_open, n_feature) and two numbers derived from it:

    reach   the IoU the guess is predicted to reach when its skins are closed and its features applied. With
            I = vol(input and guess), U = vol(U), a+ the repaired volume of D+ and a- that of D- (a skin counts with
            its volume, a feature with sigma_C v_C):
                reach = (I + a+) / (U - a-)
            Missed material that is added grows the intersection; invented material that is removed shrinks the
            UNION. (A first version added both to the numerator; on s04_tube it predicted 0.785 where the build
            measured 0.399.)
    effort  = n_feature (+ 1 if there is a skin)      the operations that takes

`reach` is a prediction, not a measurement; `apply_features` builds the program and measures it.

The grade takes the exact residual's `iou` as given: it inherits every limit of reco_field.exact_check (its module
docstring: the sample check cannot refuse a failed boolean on a sparse part). The class shares e_* sum over the listed
pieces only; slivers under the volume floor and pieces past `max_pieces` are reported as `left_out`, not classed.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

SKIN_TOLERANCE = 1e-3               # delta_0 is at most this share of the input's bounding-box diagonal ...
SKIN_WALL_SHARE = 0.1               # ... and at most this share of the input's thinnest wall
GOOD_SCORE = 0.9                    # a piece with a primitive this good is a feature
CLASSES = ("skin", "feature", "open")
DEPTH_SURFACE = 4000                # points on a piece's surface for its depth
DEPTH_VOLUME = 20000                # points in a piece's box; those inside it count


@dataclass
class PieceGrade:
    id: str
    sign: str                       # "plus" (missed) | "minus" (invented)
    volume: float
    depth: float                    # delta_C
    thickness: float                # 2 v / area: the thickness of a slab of this volume and area
    score: Optional[float]          # sigma_C, None when no feature was guessed
    kind: Optional[str]
    klass: str                      # one of CLASSES


@dataclass
class DifferenceGrade:
    iou: float
    e_skin: float
    e_feature: float
    e_open: float
    n_skin: int
    n_feature: int
    n_open: int
    reach: float
    effort: int
    delta0: float
    left_out: float                 # share of vol(U) in slivers below the judge's volume floor (in no class)
    pieces: List[PieceGrade] = dc_field(default_factory=list)

    def as_dict(self) -> Dict:
        return asdict(self)


def piece_depth(piece_stl, other, n_surface: int = DEPTH_SURFACE, n_volume: int = DEPTH_VOLUME, seed: int = 0) -> Dict:
    """depth and thickness of one piece: `other` is the SurfaceField of the solid whose surface the piece is
    measured from. The depth is the largest distance from that surface over points OF the piece: its vertices,
    `n_surface` seeded points on its surface and the points inside it among `n_volume` seeded points of its box.
    The inner points matter: a bore's deepest points lie on its axis, and a closed cavity has its whole surface on
    the other solid (measured on the surface alone its depth would be 0)."""
    import trimesh
    from reco_engine.feature_stage import _inside_mesh
    m = trimesh.load(str(piece_stl), force="mesh", process=True)
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces)
    rng = np.random.default_rng(seed)
    on, _ = trimesh.sample.sample_surface(m, n_surface, seed=seed)
    box = V.min(axis=0) + rng.random((n_volume, 3)) * (V.max(axis=0) - V.min(axis=0))
    pts = np.vstack([V, np.asarray(on, float), box[_inside_mesh(V, F, box)]])
    d = np.asarray(other.query(pts).distance, float)
    area = float(m.area)
    return dict(depth=float(d.max()) if len(d) else 0.0,
                thickness=float(2.0 * abs(m.volume) / area) if area > 0 else 0.0)


def default_delta0(input_field) -> float:
    """min(SKIN_TOLERANCE x diagonal, SKIN_WALL_SHARE x thinnest wall) of the input."""
    from src.local_grids import wall_thickness
    lo, hi = np.asarray(input_field.bounds[0], float), np.asarray(input_field.bounds[1], float)
    delta0 = SKIN_TOLERANCE * float(np.linalg.norm(hi - lo))
    _, _, tau = wall_thickness(input_field.vertices, input_field.faces, n=2000)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    return min(delta0, SKIN_WALL_SHARE * float(np.percentile(tau, 5))) if len(tau) else delta0


def classify(depth: float, score: Optional[float], delta0: float, good: float = GOOD_SCORE) -> str:
    if depth <= delta0:
        return "skin"
    return "feature" if score is not None and score >= good else "open"


def grade_difference(residual: Dict, input_field, guess_field, features: Optional[Sequence[Dict]] = None,
                     delta0: Optional[float] = None, good: float = GOOD_SCORE) -> DifferenceGrade:
    """The difference grade from an exact residual with pieces (`exact_residual(..., pieces_dir=...)`).
    `input_field`, `guess_field`: SurfaceFields of the input and of the guess (their meshes); `features`: the
    feature stage's list (dicts with id, score, kind), matched to the pieces by id."""
    lo, hi = np.asarray(input_field.bounds[0], float), np.asarray(input_field.bounds[1], float)
    if delta0 is None:
        delta0 = default_delta0(input_field)
    union = float(residual["input_volume"] + residual["r_minus"])
    by_id = {f["id"]: f for f in (features or [])}
    out: List[PieceGrade] = []
    for p in residual.get("pieces", []):
        geo = piece_depth(p["stl"], guess_field if p["sign"] == "plus" else input_field)
        f = by_id.get(p["id"])
        score = float(f["score"]) if f is not None and f.get("primitive") else None
        out.append(PieceGrade(p["id"], p["sign"], float(p["volume"]), geo["depth"], geo["thickness"], score,
                              f["kind"] if f is not None else None, classify(geo["depth"], score, delta0, good)))
    e = {c: sum(g.volume for g in out if g.klass == c) / union for c in CLASSES}
    n = {c: sum(g.klass == c for g in out) for c in CLASSES}
    repaired = {sign: sum(g.volume if g.klass == "skin" else g.score * g.volume
                          for g in out if g.sign == sign and g.klass in ("skin", "feature"))
                for sign in ("plus", "minus")}
    shared = float(residual["input_volume"]) - sum(g.volume for g in out if g.sign == "plus") \
        - float((residual.get("left_out_volume") or {}).get("plus", 0.0))
    reach = (shared + repaired["plus"]) / max(union - repaired["minus"], 1e-300)
    left = sum((residual.get("left_out_volume") or {}).values()) / union
    iou = float(residual["iou"])
    return DifferenceGrade(iou=iou, e_skin=e["skin"], e_feature=e["feature"], e_open=e["open"], n_skin=n["skin"],
                           n_feature=n["feature"], n_open=n["open"], reach=float(reach),
                           effort=n["feature"] + (1 if n["skin"] else 0), delta0=float(delta0), left_out=left,
                           pieces=out)


def format_grade(g: DifferenceGrade) -> str:
    lines = [f"IoU {g.iou:.4f}   defect {1 - g.iou:.4f} = skin {g.e_skin:.4f} + feature {g.e_feature:.4f} + "
             f"open {g.e_open:.4f}" + (f" (+ {g.left_out:.2g} in slivers)" if g.left_out > 0 else ""),
             f"pieces: {g.n_skin} skin, {g.n_feature} feature, {g.n_open} open   (delta_0 = {g.delta0:.4g})",
             f"reach {g.reach:.4f} with {g.effort} operations"]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the check

def apply_features(run_dir, good: float = GOOD_SCORE, name: str = "applied") -> Dict:
    """Build the rough program of a finished feature-stage run PLUS its features of score >= `good` (cuts and adds,
    in the list's order) in FreeCAD and measure the exact residual of the result: the measurement `reach` predicts.
    Returns dict(operations, applied, valid, iou, r_plus, r_minus, step) or dict(error). `iou` is returned also when
    `valid` is False (the build is not one valid solid): read `valid` first."""
    import json
    from reco_engine.result_schema import Operation, ReconstructionResult
    from reco_engine.result_to_program import program_model
    from reco_field.exact_check import exact_residual
    from reco_field.export import build_in_freecad
    run = Path(run_dir)
    out = json.loads((run / "features.json").read_text())
    rough = ReconstructionResult.from_dict(json.loads((run / "rough_parameters.json").read_text()))
    wrapper = dict(input_path=rough.input_path, operations=[dict(op=f["op"], primitive=f["primitive"])
                                                           for f in out["features"]
                                                           if f.get("primitive") and f["score"] >= good])
    extra = ReconstructionResult.from_dict(wrapper).operations
    rough.operations = [o for o in rough.operations if o.accepted] + [Operation(o.op, o.primitive) for o in extra]
    try:
        ex = build_in_freecad(program_model(rough), run / name, name)
    except Exception as exc:
        return dict(error=str(exc)[-300:], applied=len(extra))
    r = exact_residual(out["summary"]["input"], ex.step)
    if "error" in r:
        return dict(error=r["error"], applied=len(extra), step=str(ex.step))
    return dict(operations=len(rough.operations), applied=len(extra), valid=bool(ex.valid and ex.n_solids == 1),
                iou=r["iou"], r_plus=r["r_plus"], r_minus=r["r_minus"], step=str(ex.step))
