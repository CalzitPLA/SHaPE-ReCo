#!/usr/bin/env python3
"""
The feature stage (the user's idea of 2026-09-27): a rough guess first, then the features from what it leaves.

    1. rough_stage        the first-add phase of the add-cut loop only (residual_loop.residual_steps up to its first
                          "residual" step): the member sub-loop (a beam, a tube), the hollow sub-loop or the first add
    2. saved              rough.step / rough.stl / rough.json (the builder's program) and rough_parameters.json (the
                          ReconstructionResult: every operation's kind, pose, dimensions, label, sources)
    3. difference         input - rough (r+, missed) and rough - input (r-, invented) by OCCT booleans
    4. pieces             every solid of the difference on its own (reco_field.exact_check.exact_residual, pieces_dir)
    5. guess_feature      one primitive per piece, proposed by three sources and scored the same way

    out = feature_stage("beam.step", "out_dir")        # out["summary"], out["features"]; writes features.json

The sources of a guess: the piece's exact faces from the STEP ("step faces": a cylinder face on the input's surface
gives axis and radius exactly, the piece's extent along the axis the length), the surface fitters
(reco_field.fit.fit_best on the piece's mesh) and the cascade with its networks (cascade.run_cascade on the piece's
ray-parity voxels, pitch <= tau / 3 of the piece, D-005). A piece is a single body - the Type CNN's training scope.
The score is the volume IoU of the primitive against the piece, sampled (seeded points in the piece's box: inside the
piece by the winding number of its mesh, inside the primitive by its SDF).

r- pieces are cuts, r+ pieces are adds. Here the features are LISTED, not applied (this script's view); the loop's
feature phase (residual_loop, ResidualConfig.feature_phase, 2026-09-27) applies them: it uses `guess_candidates` and
`extend_ends` on the difference of its own model, and validates every applied feature by a FreeCAD build. This
stage: STEP input only (the pieces of an STL have facets, no exact faces). No profile semantics (D-004): a feature is its operation, its
kind and, for a cut along an axis, whether it is open at both ends ("through") or closed by the input ("blind").
"""

from __future__ import annotations

import math
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field as dc_field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import cascade  # noqa: E402
from reco_engine import residual_loop as rl  # noqa: E402
from reco_engine.result_schema import UNKNOWN, Primitive, ReconstructionResult, StageSource  # noqa: E402
from reco_engine.result_to_program import primitive_from_fit, primitive_to_sp, program_model, quaternion_of  # noqa: E402
from reco_engine.timing import timed  # noqa: E402
from reco_engine import torus_guard  # noqa: E402

# the GUI's cascade settings as measured for this stage (2026-09-27); grid and start named, since the GUI's own defaults
# moved on (grid "tau", B-090) and the rough stage stops at the first "residual" step whichever start ran
ROUGH = dict(resolution=96, max_resolution=192, top_k=2, field_backend="igl", grid="cube", start_shape="legacy")
MODE_FACES = "analytic:step_faces"
REAL = ("input", "both")            # origins of a face that is a surface of the part itself
GOOD_SCORE = 0.9
SCORE_POINTS = 100000
STOP_AT = 0.99                      # a source reaching this score ends the search: the later, dearer ones are not asked
SCORE_TIE = 0.01                    # scores this close are equal (the sampling error of SCORE_POINTS points)
SNAP_COS = float(np.cos(np.radians(1.0)))   # parallel within 1 degree
SNAP_THICKNESS = 0.1                # a face moves at most this share of the primitive's size along its normal ...
SNAP_SIZE = 0.005                   # ... and at most this share of the primitive's largest size
SNAP_RADIUS = 0.05                  # a cylinder's radius and axis together move at most this share of its radius
SNAP_SECTOR = float(np.radians(3.0))  # a sector's end plane turns at most this far onto a plane of the input
PIECE_RESOLUTION = (32, 96)         # the cascade's grid on a piece: between these, by tau / 3


# ------------------------------------------------------------------------------------------------ 1 + 2: rough

def _length_key(p: Primitive) -> str:
    return "z_half" if p.kind == "box" else "half_length"


def _axis_reach(p: Primitive, inside, reach: float) -> Optional[tuple]:
    """(t_minus, t_plus) along the primitive's local z, from its centre: how far the input's material goes on in the
    primitive's own section. Probes: a 7 x 7 grid in the section, those that are material at mid-length (a tube's or
    an I section's void drops out); the primitive goes on while at least half of them are inside the input, and the
    end is bisected. An end that is not material now stays where it is. None without material at mid-length."""
    from reco_engine.result_to_program import rotation_matrix
    R, c, d = rotation_matrix(p), np.asarray(p.center_mm, float), p.dims_mm
    hx, hy = (d["x_half"], d["y_half"]) if p.kind == "box" else (d["radius"], d["radius"])
    h = float(d[_length_key(p)])
    g = np.linspace(-0.9, 0.9, 7)
    u, v = (a.ravel() for a in np.meshgrid(g, g, indexing="ij"))
    if p.kind != "box":
        keep = u ** 2 + v ** 2 <= 0.81
        u, v = u[keep], v[keep]
    base = c + np.outer(u * hx, R[:, 0]) + np.outer(v * hy, R[:, 1])
    base = base[inside(base)]
    if len(base) < 3:
        return None
    a = R[:, 2]

    def ok(t: float) -> bool:
        return float(inside(base + t * a).mean()) >= 0.5

    step, ends = 0.5 * max(hx, hy), []
    for sign in (-1.0, 1.0):
        t = sign * h * (1.0 - 1e-6)
        if ok(t):
            while abs(t) < reach:
                nxt = t + sign * step
                if ok(nxt):
                    t = nxt
                    continue
                for _ in range(30):                          # the end lies between t (material) and nxt (none)
                    mid = 0.5 * (t + nxt)
                    t, nxt = (mid, nxt) if ok(mid) else (t, mid)
                break
        ends.append(max(abs(t), h) * sign)
    return ends[0], ends[1]


def extend_members(result: ReconstructionResult, fld) -> List[Dict]:
    """The member sub-loop ends a member where its section changes - at a bolt hole, a notch, an end plate
    (measured on m02_bolt_holes: the 2000 mm beam came out 1778 long, between the holes). For the ROUGH model a
    member runs as far as the input's material goes on in its section (`_axis_reach`); what interrupts it is then a
    piece of the difference, i.e. a feature. A member's cut (a tube's bore) keeps its overhang over the add before
    it. Changes the accepted member operations of `result` in place; returns what changed."""
    from reco_engine.ibeam import MODE_IBEAM, MODE_MEMBER
    V, F = np.ascontiguousarray(fld.vertices, np.float64), np.ascontiguousarray(fld.faces, np.int64)
    reach = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    changed, last_add = [], None
    for o in (o for o in result.operations if o.accepted):
        p = o.primitive
        if p.kind not in ("box", "cylinder") or not any(s.mode in (MODE_IBEAM, MODE_MEMBER) for s in p.sources):
            last_add = None if o.op == "add" else last_add
            continue
        key = _length_key(p)
        h, c = float(p.dims_mm[key]), np.asarray(p.center_mm, float)
        from reco_engine.result_to_program import rotation_matrix
        a = rotation_matrix(p)[:, 2]
        if o.op == "add":
            ends = _axis_reach(p, lambda pts: _inside_mesh(V, F, pts), reach)
            last_add = None if ends is None else (a, c, h, ends)
        elif last_add is not None and abs(a @ last_add[0]) > 0.999:
            a0, c0, h0, e0 = last_add                       # the cut's overhang past the add's old ends, kept
            s = float(np.sign(a @ a0))
            old = sorted(((c - c0) @ a0 - h, (c - c0) @ a0 + h))
            ends_w = (e0[0] + (old[0] + h0), e0[1] + (old[1] - h0))
            ends = tuple(sorted(s * (e - (c - c0) @ a0) for e in ends_w))
        else:
            ends = None
        if ends is None or (abs(ends[0] + h) < 1e-9 * reach and abs(ends[1] - h) < 1e-9 * reach):
            continue
        p.center_mm = tuple(float(x) for x in c + a * 0.5 * (ends[0] + ends[1]))
        p.dims_mm[key] = 0.5 * (ends[1] - ends[0])
        p.sources.append(StageSource("dimensions", "analytic:axis_reach", 1.0,
                                     detail=dict(half_length_before=h, centre_before=[float(x) for x in c])))
        changed.append(dict(label=p.label, op=o.op, length_before=2 * h, length=2 * p.dims_mm[key]))
    return changed


def _snap_position(pos: float, along: np.ndarray, want: np.ndarray, corners: np.ndarray, planes: Sequence[Dict],
                   tol: float) -> float:
    """The position along `along` of the input's plane nearest to `pos` (within `tol`) whose outward normal is
    `want` and whose face reaches the rectangle `corners`; `pos` itself without one."""
    lo, hi = corners.min(axis=0) - tol, corners.max(axis=0) + tol
    best = None
    for f in planes:
        if np.asarray(f["normal"], float) @ want < SNAP_COS:
            continue
        q = float(np.asarray(f["point"], float) @ along)
        b = np.asarray(f["bbox"], float)
        if abs(q - pos) > tol or (b[0] > hi).any() or (b[1] < lo).any():
            continue
        if best is None or abs(q - pos) < abs(best - pos):
            best = q
    return pos if best is None else best


def _snap_frame(R: np.ndarray, planes: Sequence[Dict]) -> np.ndarray:
    """The box's frame with its axes on the input's plane normals: per axis the area-weighted mean of the normals
    within 1 degree of it (either sign), the axis with the most area first, the others made square to it."""
    found = []
    for i in range(3):
        n = np.array([np.asarray(f["normal"], float) * np.sign(np.asarray(f["normal"], float) @ R[:, i]) * f["area"]
                      for f in planes if abs(np.asarray(f["normal"], float) @ R[:, i]) >= SNAP_COS])
        if len(n):
            found.append((float(np.linalg.norm(n.sum(axis=0))), i, n.sum(axis=0) / np.linalg.norm(n.sum(axis=0))))
    if not found:
        return R
    found.sort(key=lambda t: -t[0])
    out = R.copy()
    out[:, found[0][1]] = found[0][2]
    done = [found[0][1]]
    for _, i, n in found[1:2]:
        n = n - sum((n @ out[:, j]) * out[:, j] for j in done)
        out[:, i] = n / np.linalg.norm(n)
        done.append(i)
    if len(done) == 1:                                       # one direction known: the next axis squared to it
        i = (done[0] + 1) % 3
        n = out[:, i] - (out[:, i] @ out[:, done[0]]) * out[:, done[0]]
        out[:, i] = n / np.linalg.norm(n)
        done.append(i)
    k = ({0, 1, 2} - set(done)).pop()
    i, j = (k + 1) % 3, (k + 2) % 3
    out[:, k] = np.cross(out[:, i], out[:, j])
    return out


def _keep_x(R_new: np.ndarray, R_old: np.ndarray) -> np.ndarray:
    """The frame about R_new's z whose x is R_old's as nearly as it can be: a sector's angles count from x, a frame
    made from the axis alone (`_frame`) would turn the sector about it."""
    z = R_new[:, 2]
    x = R_old[:, 0] - z * (R_old[:, 0] @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def _snap_sector(u, R: np.ndarray, c: np.ndarray, radius: float, half_length: float, planes: Sequence[Dict],
                 side: float, tol: float) -> tuple:
    """The sector's two end angles on the input's planes that contain the axis. An end plane of the sector at the
    angle u0 is the half plane from the axis in the direction e(u0), its outward normal -t(u0) (t: the direction of
    growing angle), at u1 it is +t(u1) (a cut: the opposite ones). A plane of the input counts when it is parallel to
    the axis (within 1 degree), passes the axis within `tol`, looks the same way within SNAP_SECTOR and its face lies
    on the end's side of the axis, within the cylinder's length; of those the orientation with the most area wins.
    Measured 2026-09-29 on the liner: the bounding sector's angles were counted about the fitted axis, 1.26 mm
    beside the true one (at y = +1.26), so the split plane y = 0 was seen 0.58 degrees below the horizontal at both
    ends - a sector of 181.16 degrees; the axis then moved onto the input's, the angles stayed: two wedges of
    341 810 mm3 below the split plane that joined every piece touching it."""
    a, out = R[:, 2], []
    for angle, sign in ((float(u[0]), -1.0), (float(u[1]), 1.0)):
        e = np.cos(angle) * R[:, 0] + np.sin(angle) * R[:, 1]
        want = side * sign * (-np.sin(angle) * R[:, 0] + np.cos(angle) * R[:, 1])
        found: List[list] = []                                   # [area, turn]
        for f in planes:
            n = np.asarray(f["normal"], float)
            n = n / np.linalg.norm(n)
            q = np.asarray(f["point"], float) - c
            if abs(n @ a) > np.sin(np.radians(1.0)) or abs(q @ n) > tol or abs(q @ a) > half_length + tol:
                continue
            n = n - a * (n @ a)
            n /= np.linalg.norm(n)
            if n @ want < np.cos(SNAP_SECTOR) or not 0.0 < q @ e <= radius + tol:
                continue
            turn = float(np.arctan2(side * sign * (n @ -e), n @ want))    # the end plane's angle grows by this
            for g in found:
                if abs(g[1] - turn) <= 1e-9:
                    g[0] += float(f["area"])
                    break
            else:
                found.append([float(f["area"]), turn])
        out.append(angle + max(found)[1] if found else angle)
    if not 0.0 < out[1] - out[0] < 2 * np.pi - 1e-9:
        return tuple(float(x) for x in u)
    return out[0], out[1]


def snap_to_faces(result: ReconstructionResult, faces: Sequence[Dict]) -> List[Dict]:
    """The rough primitives put onto the input's exact faces (STEP). The member detection and the fitters place a
    primitive to a fraction of a millimetre (measured on s01_ipe200: the web 0.26 mm aside); the exact difference
    then is a thin skin around the whole part that joins the real pieces into one. Every face of a box and every
    end of a cylinder moves to the input's nearest parallel plane with the same outward normal (a cut: the
    opposite one), within SNAP_THICKNESS of the primitive's size along that axis and SNAP_SIZE of its largest size;
    a cylinder takes the axis and radius of the input's nearest cylinder of the same side (within SNAP_RADIUS of
    its radius). A box's axes turn onto the input's plane normals, a cylinder's axis onto the input cylinder's
    (within 1 degree). A cylinder sector's two end planes turn onto the input's planes that contain the axis
    (`_snap_sector`, within SNAP_SECTOR). Changes the accepted operations in place; returns what moved."""
    from reco_engine.result_to_program import rotation_matrix
    planes = [f for f in faces if f["type"] == "plane"]
    cylinders = [f for f in faces if f["type"] == "cylinder"]
    moved = []
    for o in (o for o in result.operations if o.accepted):
        p = o.primitive
        if p.kind not in ("box", "cylinder") or p.v or (p.u and p.kind != "cylinder"):
            continue                                                # a cylinder sector: ends, axis, radius, end planes
        R, c, d = rotation_matrix(p), np.asarray(p.center_mm, float), p.dims_mm
        before = (tuple(c), dict(d), R.copy(), tuple(p.u) if p.u else None)
        side = 1.0 if o.op == "add" else -1.0
        if p.kind == "box":                                     # the frame: the input's plane normals, largest first
            R = _snap_frame(R, planes)
        if p.kind == "cylinder":
            r, a = float(d["radius"]), R[:, 2]
            best = None
            for f in cylinders:
                fa = np.asarray(f["axis"], float)
                off = np.asarray(f["center"], float) - c
                off = off - fa * (off @ fa)                       # from the centre to the input's axis, squarely
                if abs(fa @ a) < SNAP_COS or bool(f.get("convex")) != (o.op == "add"):
                    continue
                miss = abs(f["radius"] - r) + float(np.linalg.norm(off))
                if miss <= SNAP_RADIUS * r and (best is None or miss < best[0]):
                    best = (miss, float(f["radius"]), off, fa)
            if best is not None:
                d["radius"], c = best[1], c + best[2]
                R = _frame(best[3] * float(np.sign(best[3] @ a) or 1.0))
                if p.u:
                    R = _keep_x(R, before[2])
            half = np.array([d["radius"], d["radius"], d["half_length"]])
        else:
            half = np.array([d["x_half"], d["y_half"], d["z_half"]])
        axes = (2,) if p.kind == "cylinder" else (0, 1, 2)
        for i in axes:
            tol = min(SNAP_THICKNESS * 2 * half[i], SNAP_SIZE * 2 * half.max())
            j, k = [n for n in range(3) if n != i]
            ends = []
            for s in (-1.0, 1.0):
                centre = c + s * half[i] * R[:, i]
                corners = np.array([centre + u * half[j] * R[:, j] + v * half[k] * R[:, k]
                                    for u in (-1, 1) for v in (-1, 1)])
                ends.append(_snap_position(float(centre @ R[:, i]), R[:, i], side * s * R[:, i], corners, planes, tol))
            if ends[1] - ends[0] <= 0:
                continue
            c = c + R[:, i] * (0.5 * (ends[0] + ends[1]) - c @ R[:, i])
            half[i] = 0.5 * (ends[1] - ends[0])
        if p.kind == "cylinder":
            d["half_length"] = float(half[2])
        else:
            d["x_half"], d["y_half"], d["z_half"] = (float(h) for h in half)
        u = before[3]
        if u:                                                   # after the axis: the angles count about the snapped one
            u = _snap_sector(u, R, c, float(d["radius"]), float(d["half_length"]), planes, side,
                             SNAP_SIZE * 2 * float(half.max()))
        shift = float(np.linalg.norm(c - np.asarray(before[0])))
        change = max(abs(d[k] - before[1][k]) for k in d)
        turn = float(np.degrees(np.arccos(np.clip(abs(R[:, 2] @ before[2][:, 2]), -1.0, 1.0))))
        if shift > 0 or change > 0 or turn > 0 or u != before[3]:
            p.u = u
            p.center_mm = tuple(float(x) for x in c)
            p.quaternion = quaternion_of(R)
            p.sources.append(StageSource("position", MODE_FACES, 1.0,
                                         detail=dict(centre_before=list(before[0]), dims_before=before[1])))
            moved.append(dict(label=p.label or p.kind, op=o.op, shift=shift, size_change=change, turn_deg=turn))
            if u != before[3]:
                moved[-1].update(sector_deg_before=float(np.degrees(before[3][1] - before[3][0])),
                                 sector_deg=float(np.degrees(u[1] - u[0])))
    return moved


def rough_stage(input_path, out_dir, config: Optional[rl.ResidualConfig] = None, extend: bool = True,
                snap: bool = True, contour: bool = True) -> Dict:
    """The rough model of `input_path`, built in FreeCAD and saved with its parameter set. Returns dict(result, step,
    stl, program, parameters, grade, stopped_at, snapped, extended, contour). `snap`: the primitives are put onto the
    input's exact faces (`snap_to_faces`); `contour`: a rough cylinder takes the input's turned contour
    (reco_engine.turned_contour; with `snap` only, it reads the same faces); `extend`: members run as far as the
    input's material does (`extend_members`)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(config or rl.ResidualConfig(**ROUGH), freecad=False, voxel_fallback=False, exact_grade=False,
                  feature_phase=False)
    last = None
    with timed("rough stage"):
        for step in rl.residual_steps(input_path, cfg):
            last = step
            if step.kind in ("residual", "done"):            # "residual": the first-add phase is over
                break
    result: ReconstructionResult = last.result
    if not [o for o in result.operations if o.accepted]:
        raise RuntimeError(f"no rough model: {last.info.get('stopped') or last.info.get('message') or last.kind}")
    fld, _ = rl._load(input_path)
    snapped, turned = [], []
    if snap:
        from reco_field.exact_check import exact_faces
        with timed("snap to faces"):
            inv = exact_faces(input_path)
            if "error" in inv:
                raise RuntimeError(inv["error"])
            snapped = snap_to_faces(result, inv["faces"])
        if contour:
            from reco_engine.turned_contour import turned_contour
            with timed("turned contour"):
                turned = turned_contour(result, inv["faces"], input_inside(fld.vertices, fld.faces))
    with timed("extend members"):
        extended = extend_members(result, fld) if extend else []
    with rl._field_backend(cfg.field_backend):
        g = rl.freecad_grade(program_model(result), out_dir, "rough", fld)
    result.step_path, result.stl_path, result.chamfer_mm = g["step"], g["stl"], g["chamfer"]
    parameters = out_dir / "rough_parameters.json"
    parameters.write_text(result.to_json(indent=1))
    return dict(result=result, step=g["step"], stl=g["stl"], program=str(out_dir / "rough.json"),
                parameters=str(parameters), grade=g, stopped_at=last.kind, snapped=snapped, extended=extended,
                contour=turned)


# ------------------------------------------------------------------------------------------------ 3 + 4: pieces

def difference_pieces(input_path, model_step, out_dir, max_pieces: int = 500, timeout: int = 1800) -> Dict:
    """The exact residual of the rough model with its pieces (see exact_check.exact_residual); raises on an error."""
    from reco_field.exact_check import exact_residual
    with timed("exact difference"):
        r = exact_residual(input_path, model_step, pieces_dir=Path(out_dir) / "pieces", max_pieces=max_pieces,
                           timeout=timeout)
    if "error" in r:
        raise RuntimeError(r["error"])
    return r


# ------------------------------------------------------------------------------------------------ 5: the guess

@dataclass
class Candidate:
    source: str                     # "step faces" | "fit" | "cascade"
    primitive: Primitive
    score: float = 0.0              # volume IoU against the piece (sampled), or min(precision, recall)
    n_inside: int = 0               # PartScorer: the sampled points inside the primitive (the precision's sample)
    seconds: float = 0.0            # its source's time (proposing all its candidates) plus its own scoring
    precision: Optional[float] = None   # PartScorer: share of the primitive that harms no input material
    recall: Optional[float] = None      # PartScorer: share of the piece (segment) it covers


@dataclass
class Feature:
    id: str
    op: str                         # "cut" (a piece of r-) | "add" (a piece of r+)
    kind: str                       # the best candidate's kind, UNKNOWN without any candidate
    primitive: Optional[Dict]       # result_schema.Primitive as a dict
    score: float
    source: str
    depth: Optional[str]            # a cut along an axis: "through" | "blind"
    host: Optional[str]             # the rough operation the piece sits in or on
    volume: float
    sources_agree: bool             # every source's best candidate has this kind
    faces: Dict[str, int] = dc_field(default_factory=dict)       # the piece's real faces by type
    alternatives: List[Dict] = dc_field(default_factory=list)    # [{source, kind, score, primitive}], best first
    step: Optional[str] = None
    stl: Optional[str] = None
    thickness: Optional[Dict] = None    # piece_thickness: the percentiles, the thin share, the threshold
    suppressed: Optional[str] = None    # the reason when the piece was not guessed (super-thin material)


def _frame(z) -> np.ndarray:
    z = np.asarray(z, float) / np.linalg.norm(z)
    u = np.cross(z, [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    return np.stack([u, np.cross(z, u), z], axis=1)


def _faces_source() -> List[StageSource]:
    return [StageSource(s, MODE_FACES, 1.0) for s in ("type", "rotation", "dimensions", "position")]


def _box_of_planes(planes: Sequence[Dict], V: np.ndarray) -> Optional[Primitive]:
    """The box in the frame of the piece's plane normals (the largest plane first, then the largest one square to
    it), as large as the piece in that frame."""
    if not planes:
        return None
    planes = sorted(planes, key=lambda f: (f["origin"] not in REAL, -f["area"]))
    x = np.asarray(planes[0]["normal"], float)
    x /= np.linalg.norm(x)
    y = None
    for f in planes[1:]:
        n = np.asarray(f["normal"], float)
        n = n - x * (n @ x)
        if np.linalg.norm(n) > 0.99:                        # square to the first within 8 degrees
            y = n / np.linalg.norm(n)
            break
    R = np.stack([x, y, np.cross(x, y)], axis=1) if y is not None else _frame(x)[:, [2, 0, 1]]
    local = V @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    # a side bounded by a real plane of the input ends at that plane (the outermost one, a step has several); the
    # vertices bound only the open sides. 2026-09-28, the liner's pocket pieces: drilled holes below the pocket floor
    # made the box 37 mm deep instead of 21.6 (precision 0.596; a fitted box 0.994).
    for i in range(3):
        for sign in (-1.0, 1.0):
            pos = [float(np.asarray(f["point"], float) @ R[:, i]) for f in planes
                   if f["origin"] in REAL and float(np.asarray(f["normal"], float) @ R[:, i]) * sign >= SNAP_COS]
            if pos:
                if sign > 0:
                    hi[i] = min(hi[i], max(pos))
                else:
                    lo[i] = max(lo[i], min(pos))
    half = 0.5 * (hi - lo)
    if not (half > 0).all():
        return None
    return Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in R @ (0.5 * (lo + hi))),
                     dims_mm=dict(x_half=float(half[0]), y_half=float(half[1]), z_half=float(half[2])),
                     confidence=1.0, sources=_faces_source())


def metadata_candidates(piece: Dict, V: np.ndarray) -> List[Primitive]:
    """Primitives read from the piece's exact faces: per curved surface type the largest real face gives the
    surface (axis, centre, radii), the piece's vertices `V` its extent; the planes give a box."""
    faces = piece.get("faces", [])
    real = [f for f in faces if f["origin"] in REAL] or faces
    out: List[Primitive] = []
    for kind in ("cylinder", "cone", "sphere", "torus"):
        same = sorted((f for f in real if f["type"] == kind), key=lambda f: -f["area"])
        if not same:
            continue
        f = same[0]
        c = np.asarray(f["center"], float)
        if kind == "sphere":
            out.append(Primitive(kind="sphere", center_mm=tuple(c), dims_mm=dict(radius=float(f["radius"])),
                                 confidence=1.0, sources=_faces_source()))
            continue
        a = np.asarray(f["axis"], float)
        a /= np.linalg.norm(a)
        q = quaternion_of(_frame(a))
        if kind == "torus":
            out.append(Primitive(kind="torus", quaternion=q, center_mm=tuple(c), confidence=1.0, sources=_faces_source(),
                                 dims_mm=dict(major_radius=float(f["major_radius"]), minor_radius=float(f["minor_radius"]))))
            continue
        t = (V - c) @ a
        lo, hi = float(t.min()), float(t.max())
        if hi - lo <= 0:
            continue
        centre = tuple(float(v) for v in c + a * 0.5 * (lo + hi))
        if kind == "cylinder":
            dims = dict(radius=float(f["radius"]), half_length=0.5 * (hi - lo))
        else:                                                # the cone's radius grows by tan(semi angle) along its axis
            k = float(np.tan(f["semi_angle"]))
            dims = dict(radius1=max(0.0, float(f["radius"]) + k * lo), radius2=max(0.0, float(f["radius"]) + k * hi),
                        half_length=0.5 * (hi - lo))
        out.append(Primitive(kind=kind, quaternion=q, center_mm=centre, dims_mm=dims, confidence=1.0,
                             sources=_faces_source()))
    box = _box_of_planes([f for f in faces if f["type"] == "plane"], V)
    if box is not None:
        out.append(box)
    return out


def fit_candidates(fld, top: int = 2) -> List[Primitive]:
    from reco_field.fit import fit_best
    out = []
    for fit in fit_best(fld, stop_at=1.0)[:top]:
        try:
            out.append(primitive_from_fit(fit, confidence=float(fit.explained),
                                          sources=[StageSource("type", "field_fit", float(fit.explained))]))
        except ValueError:                                   # a kind the schema does not have (a plane)
            continue
    return out


# Super-thin material in the feature input (the user's rule, 2026-09-28: "add super thin wall suppression to the
# feature input"): skins and slivers of the exact difference are no features. Material thinner than the threshold is
# suppressed - a piece that is thin as a whole is not guessed, gridded or split (it stays in the output with its
# exact volume and the reason), and the thin part of a thick piece no longer dictates its grid's pitch.
# The threshold is a share of the INPUT's diagonal. 2e-3 from the liner's 40 pieces (diagonal 1648 mm: 3.3 mm,
# measured 2026-09-28, wall thickness at 1000 surface samples per piece): its real pieces' thinnest walls (5th
# percentile) are 4.0 - 7.6 mm, the skin that joins pieces over the whole part has 5th / 25th / 50th percentiles of
# 1.30 / 1.74 / 2.97 mm. 0 switches the suppression off.
THIN_WALL = rl.THIN_WALL            # 2e-3
THIN_PERCENTILE = 95.0              # a piece is thin as a whole when this percentile of its thickness is below


def thin_wall_threshold(diagonal: float, share: float = THIN_WALL, mm: Optional[float] = None) -> float:
    """The threshold in the part's unit: `mm` when given (absolute), else `share` of the input's diagonal."""
    return float(mm) if mm is not None else float(share) * float(diagonal)


def piece_thickness(fld, thin: float = 0.0, n: int = 1000) -> Dict:
    """The thickness distribution of a piece's mesh (src.local_grids.wall_thickness at `n` surface samples):
    dict(p5, p25, p50, p75, p95, samples, threshold, thin_share (samples below `thin`), suppressed (thin as a whole:
    p95 < thin), tau (the 5th percentile of the samples at or above `thin`: what licenses the grid's pitch; of all
    samples when none is, or without a threshold))."""
    from src.local_grids import wall_thickness
    _, _, tau = wall_thickness(fld.vertices, fld.faces, n=n)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    out = dict(samples=int(len(tau)), threshold=float(thin), thin_share=0.0, suppressed=False, tau=None)
    if not len(tau):
        return out
    out.update({f"p{q}": float(np.percentile(tau, q)) for q in (5, 25, 50, 75, 95)})
    thick = tau[tau >= thin] if thin > 0 else tau
    out["thin_share"] = float(1.0 - len(thick) / len(tau))
    out["suppressed"] = bool(thin > 0 and float(np.percentile(tau, THIN_PERCENTILE)) < thin)
    out["tau"] = float(np.percentile(thick if len(thick) else tau, 5))
    return out


def suppressed_reason(th: Dict) -> str:
    return (f"suppressed: super-thin material (wall thickness p50 {th['p50']:.3g}, p95 {th['p95']:.3g} < "
            f"{th['threshold']:.3g})")


def open_thin(occ: np.ndarray, pitch: float, thin: float) -> np.ndarray:
    """`occ` without what is thinner than `thin`: a morphological opening by a ball of radius thin / 2. A grid whose
    pitch does not resolve that ball (radius under one cell), and an opening that leaves nothing, return `occ`."""
    r = 0.5 * float(thin) / float(pitch)
    if thin <= 0 or r < 1.0 or not occ.any():
        return occ
    from scipy import ndimage
    k = int(np.floor(r))
    g = np.arange(-k, k + 1)
    ball = (g[:, None, None] ** 2 + g[None, :, None] ** 2 + g[None, None, :] ** 2) <= r * r
    opened = ndimage.binary_opening(occ, structure=ball)
    return opened if opened.any() else occ


def piece_resolution(fld, thin: float = 0.0) -> int:
    """Grid cells per side for a piece: pitch <= tau / 3 of its thinnest wall (5th percentile; with `thin` of the
    walls at or above it, piece_thickness), within limits."""
    lo, hi = np.asarray(fld.bounds[0], float), np.asarray(fld.bounds[1], float)
    side = float((hi - lo).max()) * (1.0 + 2.0 * rl.ResidualConfig.pad_fraction)
    tau = piece_thickness(fld, thin)["tau"]
    want = int(np.ceil(side / (tau / 3.0))) if tau else PIECE_RESOLUTION[0]
    return int(min(PIECE_RESOLUTION[1], max(PIECE_RESOLUTION[0], want)))


def cascade_candidates(fld, registry=None, models=None, top_k: int = 2, thin: float = 0.0,
                       part_diagonal: Optional[float] = None) -> List[Primitive]:
    """The cascade's answer (type stage with its networks, analytic pose and size, field refinement) for the piece.
    `part_diagonal`: the whole part's, for the cascade's torus guard (default: the piece's)."""
    cfg = rl.ResidualConfig(resolution=piece_resolution(fld, thin), field_backend="igl")
    spec = rl._spec_for(fld, cfg)
    t_in = rl._channels(fld, spec, False, cfg.field_backend)
    prim = cascade.run_cascade(t_in, spec, registry, field=fld, top_k=top_k, models=models, part_diagonal=part_diagonal)
    return [p for p in [prim] + list(prim.alternatives) if p.kind != UNKNOWN]


def _inside_mesh(V: np.ndarray, F: np.ndarray, pts: np.ndarray) -> np.ndarray:
    try:
        import igl
        return np.abs(igl.winding_number(np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64),
                                         np.ascontiguousarray(pts, np.float64))) > 0.5
    except ImportError:
        import trimesh
        return trimesh.Trimesh(V, F, process=False).contains(pts)


def sampled_score(p: Primitive, V: np.ndarray, F: np.ndarray, n: int = SCORE_POINTS, seed: int = 0) -> float:
    """Volume IoU of the primitive against the piece, from `n` seeded points in the union of both boxes (the
    primitive's clipped to three times the piece's)."""
    try:
        solid = primitive_to_sp(p)
        plo, phi = (np.asarray(b, float) for b in solid.bounds())
        solid.sdf(np.zeros((1, 3)))                          # a sweep the SDF refuses (u1 <= u0) is no candidate
    except (ValueError, KeyError):
        return 0.0
    lo, hi = V.min(axis=0), V.max(axis=0)
    size = hi - lo
    lo2, hi2 = np.minimum(lo, np.maximum(plo, lo - size)), np.maximum(hi, np.minimum(phi, hi + size))
    pts = lo2 + np.random.default_rng(seed).random((n, 3)) * (hi2 - lo2)
    a = _inside_mesh(V, F, pts)
    b = np.asarray(solid.sdf(pts)) <= 0
    either = int((a | b).sum())
    return float((a & b).sum() / either) if either else 0.0


def _depth(piece: Dict, p: Primitive) -> Optional[str]:
    if piece["sign"] != "minus" or p.kind not in ("cylinder", "cone"):
        return None
    from reco_engine.result_to_program import rotation_matrix
    axis = rotation_matrix(p)[:, 2]
    caps = [f for f in piece["faces"] if f["type"] == "plane" and abs(np.asarray(f["normal"], float) @ axis) > 0.99]
    return "blind" if any(f["origin"] in REAL for f in caps) else "through"


def _host(piece: Dict, rough: Optional[ReconstructionResult]) -> Optional[str]:
    if rough is None:
        return None
    c = np.asarray(piece["centre"], float)[None]
    best = None
    for i, o in enumerate(o for o in rough.operations if o.accepted):
        if o.op != "add":
            continue
        try:
            d = float(np.asarray(primitive_to_sp(o.primitive).sdf(c))[0])
        except (ValueError, KeyError):
            continue
        if best is None or d < best[0]:
            best = (d, o.primitive.label or f"{o.primitive.kind} #{i + 1}")
    return best[1] if best else None


def _prim_dict(p: Primitive) -> Dict:
    d = asdict(p)
    d["alternatives"] = []                                   # the candidates are listed by the feature itself
    return d


def torus_gate(part_diagonal: Optional[float], failed: Dict):
    """admit(primitive, source) -> bool for one piece's sources: False when the torus guard (reco_engine/torus_guard.py)
    rejects the primitive - it is then not offered at all, as if its source had not proposed it - and the count and
    the first reason in `failed` under "torus guard (<source>)"."""
    seen: Dict[str, list] = {}

    def admit(p: Primitive, source: str) -> bool:
        found = torus_guard.primitive_defect(p, part_diagonal)
        if found is None:
            return True
        n_first = seen.setdefault(source, [0, found.text])
        n_first[0] += 1
        count = f"{n_first[0]} rejected, the first: " if n_first[0] > 1 else ""
        failed[f"torus guard ({source})"] = count + n_first[1]
        return False
    return admit


def guess_candidates(piece: Dict, registry=None, models=None, sources: Sequence[str] = ("step faces", "fit", "cascade"),
                     stop_at: Optional[float] = STOP_AT, deadline: Optional[float] = None, op: Optional[str] = None,
                     inside_input=None, inside_region=None, tools: Optional[Dict] = None,
                     typed: Optional[Sequence] = None, contact: Optional[Dict] = None,
                     contact_faces: bool = False, thin: float = 0.0, part_diagonal: Optional[float] = None) -> Dict:
    """Every candidate primitive for one piece of the difference (a dict of exact_residual's `pieces`), scored, best
    first: dict(candidates=[Candidate], failed={source: error}, timed_out=bool). The sources are asked in the given
    order (cheapest first: 0.2 s, 3 s, 9 s per piece on m02_bolt_holes) until one reaches `stop_at` (None: all of
    them). `deadline` (time.time() value): no further source is asked after it (timed_out) - a source already running
    is not interrupted (the fitters and the cascade run in this process). Within SCORE_TIE the exact faces win over a
    fit, a fit over the grid's answer.

    With `op` and `inside_input` (the loop's feature phase) every candidate is scored as the operation it is
    (PartScorer): precision (no input material harmed) and recall (share of the piece covered), score = the lower
    of the two. The volume IoU punished the exact STEP-faces primitives for the part of them in AIR beyond the model's
    face that closes the piece - removing nothing, and exactly the overshoot a cutter needs (measured 2026-09-27 on the
    liner: the pocket's STEP-faces box 0.580 against a fitted box's 0.963, the bores' exact R 6.92 x 54 cylinders
    0.950 against a shorter fit's 0.981). `inside_region`: where the piece's material is, by a predicate instead of
    the piece's own mesh - a thin piece's tessellation can be open (2026-09-28, the liner's largest invented piece:
    12 open bodies, mesh volume 3.55e7 for an OCCT volume of 9.37e5; scored against it a R 125 cylinder had
    precision and recall 1.000 and made the exact residual worse).

    `contact_faces` (with `op` and `inside_input`, the source "step faces" among `sources`): the tool of the piece's
    contact faces (`contact_tool`, source "tool (contact faces)") is scored after the typed candidates and BEFORE
    any whole-piece source; passing (`tools`' limits) it ends the search, failing it stays a candidate among the
    others. `contact["rank"]`: among the passing candidates the contact grade (contact recall x contact precision,
    in steps of 0.01) ranks before the number of operands. `thin`: the super-thin threshold for the cascade's
    grid (piece_resolution). `part_diagonal`: the input's bounding-box diagonal, the yardstick of the torus guard
    (reco_engine/torus_guard.py) that every candidate of every source passes before it is scored; without it the
    guard's size rule is not applied (the cascade's own guard then measures against the piece)."""
    from src.closest_point_field import load_field
    fld = load_field(piece["stl"])
    V, F = np.asarray(fld.vertices, float), np.asarray(fld.faces)
    proposers = {"step faces": lambda: metadata_candidates(piece, V), "fit": lambda: fit_candidates(fld),
                 "cascade": lambda: cascade_candidates(fld, registry, models, thin=thin, part_diagonal=part_diagonal)}
    cands: List[Candidate] = []
    failed, timed_out = {}, False
    admit = torus_gate(part_diagonal, failed)
    scorer = None
    if op is not None and inside_input is not None:
        lo_b, hi_b = (np.asarray(b, float) for b in (piece.get("bbox") or (V.min(axis=0), V.max(axis=0))))
        region = inside_region or (lambda q: _inside_mesh(V, F, q))
        scorer = PartScorer(op, region, lo_b, hi_b, float(piece.get("volume") or 0.0) or 1e-30, inside_input)
    limits = tools or dict(min_score=GOOD_SCORE, min_precision=0.999)

    def passes_typed(c):
        return c.score >= limits["min_score"] and precise_enough(c, limits["min_precision"])
    # typed candidates first (2026-09-28): the session's tools, then the catalogue's fits of the piece's tool type;
    # the first passing one ends the search - no primitive source, no composite search
    for t in typed or []:
        if deadline is not None and time.time() > deadline:
            timed_out = True
            break
        if not admit(t["primitive"], t["source"]):
            continue
        t1 = time.time()
        c = Candidate(t["source"], t["primitive"])
        if scorer is not None:
            c.precision, c.recall = scorer.score(c.primitive)
            c.score, c.n_inside = min(c.precision, c.recall), scorer.last_n
        else:
            c.score = sampled_score(c.primitive, V, F)
        c.seconds = time.time() - t1
        c.tool_type, c.tool_source, c.tool_params = t.get("tool_type"), t.get("tool_source"), t.get("params")
        cands.append(c)
        if contact and contact.get("score"):                          # the contact grade decides too
            _contact_graded([c], contact, V, F, op)
        if passes_typed(c):
            cands = _contact_graded(cands, contact, V, F, op)
            cands.sort(key=lambda q: (not passes_typed(q), -q.score))
            return dict(candidates=cands, failed=failed, timed_out=timed_out)
    if contact_faces and scorer is not None and "step faces" in sources and not timed_out:
        # hole stacks (2026-09-28): per hole axis the cylinders and cones the piece's real faces name
        try:
            with timed("guess/hole stacks"):
                stacked = hole_tools(piece, V)
        except Exception as exc:
            stacked, failed["hole stacks"] = [], str(exc)[-200:]
        try:                                                         # a union of prismatic features (2026-09-29)
            from reco_engine.prismatic_groups import candidate_tools
            with timed("guess/prismatic groups"):
                stacked = stacked + candidate_tools(piece, V, scorer, stacked)
        except Exception as exc:
            failed["prismatic groups"] = str(exc)[-200:]
        for source, t, partial in stacked:
            if not admit(t, source):
                continue
            t1 = time.time()
            c = Candidate(source, t)
            c.partial = bool(partial)
            c.precision, c.recall = scorer.score(t)
            c.n_inside = scorer.last_n
            # a partial tool is judged by what it harms: it need not cover the piece, the rest comes back as a piece
            c.score = c.precision if partial and c.recall >= HOLE_PARTIAL else min(c.precision, c.recall)
            c.seconds = time.time() - t1
            cands.append(c)
            if contact and contact.get("score"):
                _contact_graded([c], contact, V, F, op)
            if passes_typed(c) and not partial:
                cands = _contact_graded(cands, contact, V, F, op)
                cands.sort(key=lambda q: (not passes_typed(q), -q.score))
                return dict(candidates=cands, failed=failed, timed_out=timed_out)
    if contact_faces and scorer is not None and "step faces" in sources and not timed_out:
        t1 = time.time()
        try:
            with timed("guess/contact faces"):
                t = contact_tool(piece, V)
        except Exception as exc:
            t, failed["contact faces"] = None, str(exc)[-200:]
        if t is not None and admit(t, "tool (contact faces)"):
            c = Candidate("tool (contact faces)", t)
            c.precision, c.recall = scorer.score(t)
            c.score, c.n_inside = min(c.precision, c.recall), scorer.last_n
            c.seconds = time.time() - t1
            cands.append(c)
            if contact and contact.get("score"):
                _contact_graded([c], contact, V, F, op)
            if passes_typed(c):
                cands = _contact_graded(cands, contact, V, F, op)
                cands.sort(key=lambda q: (not passes_typed(q), -q.score))
                return dict(candidates=cands, failed=failed, timed_out=timed_out)
    for name in sources:
        if deadline is not None and time.time() > deadline:
            timed_out = True
            break
        t0 = time.time()
        try:
            with timed(f"guess/{name}"):
                found = proposers[name]()
        except Exception as exc:                             # one source failing is a result, the others still answer
            failed[name] = str(exc)[-200:]
            continue
        t_source = time.time() - t0
        new = [Candidate(name, p) for p in found if admit(p, name)]
        with timed("guess/score"):
            for c in new:
                t1 = time.time()
                if scorer is not None:
                    c.precision, c.recall = scorer.score(c.primitive)
                    c.score, c.n_inside = min(c.precision, c.recall), scorer.last_n
                else:
                    c.score = sampled_score(c.primitive, V, F)
                c.seconds = t_source + time.time() - t1
        cands += new
        if stop_at is not None and any(c.score >= stop_at for c in new):
            break
    order = {"step faces": 0, "fit": 1, "cascade": 2}

    def passes(c):
        return c.score >= tools["min_score"] and precise_enough(c, tools["min_precision"])
    if tools and scorer is not None and not any(passes(c) for c in cands) and not timed_out:
        # boolean tools (2026-09-28): composites of the piece's own primitives, fewest operands first
        singles = [c.primitive for c in sorted(cands, key=lambda c: -c.score)]
        t0 = time.time()
        for p in tool_candidates(piece, V, singles, tools.get("max_operands", 9), tools.get("max_candidates", 12)):
            if deadline is not None and time.time() > deadline:
                timed_out = True
                break
            if not admit(p, f"tool ({p.label or p.combine})"):
                continue
            t1 = time.time()
            c = Candidate(f"tool ({p.label or p.combine})", p)
            c.precision, c.recall = scorer.score(p)
            c.score, c.n_inside = min(c.precision, c.recall), scorer.last_n
            c.seconds = time.time() - t1
            cands.append(c)
            if passes(c):                                            # the simplest passing tool: no dearer one
                break
    cands = _contact_graded(cands, contact, V, F, op)
    top = max((c.score for c in cands), default=0.0)
    def grade(c):                                                    # the contact grade as a rank, 0 when not asked
        g = getattr(c, "contact", None)
        if not (contact and contact.get("rank")) or not g or not passes(c):
            return 0.0
        return -round(g["contact_recall"] * g["contact_precision"], 2)
    if tools:                                                        # a passing candidate first, fewest operands
        cands.sort(key=lambda c: (not passes(c), grade(c), len(leaves(c.primitive)) if passes(c) else 0,
                                  c.score < top - SCORE_TIE, order.get(c.source, 3), -c.score))
    else:
        cands.sort(key=lambda c: (c.score < top - SCORE_TIE, order.get(c.source, 3) if c.score >= top - SCORE_TIE else 0, -c.score))
    return dict(candidates=cands, failed=failed, timed_out=timed_out)


OVERSHOOT = 0.05                    # extend_ends: an end moves out by this share of the primitive's size along its axis
OVERSHOOT_SAMPLES = 400             # ... when this many points of the added slab all leave the input as it is


def _ends(p: Primitive):
    """[(local axis, sign, half size along it, section half sizes)] of the ends extend_ends may move."""
    d = p.dims_mm
    if p.kind == "box":
        h = (d["x_half"], d["y_half"], d["z_half"])
        return [(i, s, h[i], tuple(h[j] for j in range(3) if j != i)) for i in range(3) for s in (-1.0, 1.0)]
    if p.kind == "cylinder":
        return [(2, s, d["half_length"], (d["radius"], d["radius"])) for s in (-1.0, 1.0)]
    if p.kind == "cone":
        return [(2, -1.0, d["half_length"], (d["radius1"],) * 2), (2, 1.0, d["half_length"], (d["radius2"],) * 2)]
    return []


def extend_ends(p: Primitive, op: str, inside_input, overshoot: float = OVERSHOOT, samples: int = OVERSHOOT_SAMPLES,
                seed: int = 0) -> Primitive:
    """A feature's primitive with every end (box face, cylinder / cone cap) moved out by `overshoot` x its size along
    that axis where the slab it gains changes nothing of the part: for a cut, none of `samples` seeded points in the
    slab is inside the input (`inside_input(points) -> bool array`); for an add, all of them are. A cutter ending
    exactly on the model's face leaves a coplanar-face boolean or a skin in FreeCAD (the voxel loop's
    _extend_open_faces does the same on its grid); an end on a real face of the input (a blind bore's floor) keeps
    its place because its slab would remove input material. Sectors and other kinds are returned unchanged. A cone's
    radii go on linearly with its length."""
    if p.kind not in ("box", "cylinder", "cone") or p.u or p.v or inside_input is None:
        return p
    from reco_engine.result_to_program import rotation_matrix
    R, rng = rotation_matrix(p), np.random.default_rng(seed)
    c, dims = np.asarray(p.center_mm, float), dict(p.dims_mm)
    grow = {}
    for axis, sign, h, sec in _ends(p):
        delta = overshoot * 2.0 * h
        t = sign * (h + delta * rng.random(samples))
        if p.kind == "box":
            others = [j for j in range(3) if j != axis]
            local = np.zeros((samples, 3))
            local[:, axis] = t
            for j, s in zip(others, sec):
                local[:, j] = (2.0 * rng.random(samples) - 1.0) * s
        else:
            r = sec[0] * np.sqrt(rng.random(samples))
            a = 2.0 * np.pi * rng.random(samples)
            local = np.stack([r * np.cos(a), r * np.sin(a), t], axis=1)
        inside = np.asarray(inside_input(c + local @ R.T), bool)
        if (not inside.any()) if op == "cut" else inside.all():
            grow[(axis, sign)] = delta
    if not grow:
        return p
    for (axis, sign), delta in grow.items():
        c = c + R[:, axis] * sign * delta / 2.0
    q = replace(p, center_mm=tuple(float(x) for x in c), alternatives=[], sources=list(p.sources))
    if p.kind == "box":
        for axis, key in enumerate(("x_half", "y_half", "z_half")):
            dims[key] += sum(grow.get((axis, s), 0.0) for s in (-1.0, 1.0)) / 2.0
    else:
        h = dims["half_length"]
        lo, hi = grow.get((2, -1.0), 0.0), grow.get((2, 1.0), 0.0)
        if p.kind == "cone":
            k = (dims["radius2"] - dims["radius1"]) / (2.0 * h)
            dims["radius1"], dims["radius2"] = max(0.0, dims["radius1"] - k * lo), max(0.0, dims["radius2"] + k * hi)
        dims["half_length"] = h + (lo + hi) / 2.0
    q.dims_mm = dims
    return q


# ------------------------------------------------------------------------------------------------ congruent pieces

CONGRUENT_VOLUME = 1e-3             # relative: the exact (OCCT) volumes of two congruent pieces
CONGRUENT_MESH = 2e-2               # relative: area and principal moments of their tessellations


def piece_descriptor(piece: Dict) -> Optional[Dict]:
    """What makes two pieces congruent, from the piece's mesh (its STL) and exact data: the volume (OCCT when the piece
    has one, else the mesh's), the surface area, the sorted principal moments of inertia, the centroid, the principal
    axes (columns, right-handed) and the count of exact faces by type. None without a readable closed mesh. Liner run
    2026-09-27: nine pieces of 86483.945 mm^3 and three of 7716.676 mm^3 (the same to 9 digits) - one pattern each."""
    stl = piece.get("stl")
    if not stl or not Path(str(stl)).is_file():
        return None
    try:
        import trimesh
        m = trimesh.load(str(stl), process=True, force="mesh")
        if len(m.faces) < 4 or abs(m.volume) <= 0:
            return None
        moments = np.asarray(m.principal_inertia_components, float)
        axes = np.asarray(m.principal_inertia_vectors, float).T          # columns
        centre = np.asarray(m.center_mass, float)
    except Exception:
        return None
    order = np.argsort(moments)
    moments, axes = moments[order], axes[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]
    counts: Dict[str, int] = {}
    for f in piece.get("faces") or []:
        counts[f["type"]] = counts.get(f["type"], 0) + 1
    return dict(volume=float(piece.get("volume") or abs(m.volume)), area=float(m.area), moments=moments, axes=axes,
                centre=centre, faces=counts, sign=piece.get("sign"))


def feature_radius(piece: Dict) -> Optional[float]:
    """The smallest radius among the piece's real curved faces (cylinder, sphere; a torus's minor radius) - faces on
    the input's surface (origin input / both); None without one. The size a small piece is measured by: a boolean
    sliver has no real curved face, a bore has its wall."""
    radii = []
    for f in piece.get("faces") or []:
        if f.get("origin") not in REAL:
            continue
        r = f.get("minor_radius") if f.get("type") == "torus" else f.get("radius") if f.get("type") in ("cylinder", "sphere") else None
        if r is not None and float(r) > 0:
            radii.append(float(r))
    return min(radii) if radii else None


def feature_sized(piece: Dict, depth: float) -> Optional[str]:
    """Why a piece under the volume share is a feature by its absolute size: its volume is at least `depth` x pi r^3,
    r its `feature_radius` - a bore at least `depth` radii deep. None when it is not (or `depth` <= 0)."""
    r = feature_radius(piece)
    if depth <= 0 or r is None or float(piece["volume"]) < depth * np.pi * r ** 3:
        return None
    return f"a real face of radius {r:.4g} and {float(piece['volume']) / (np.pi * r ** 3):.3g} x pi r^3 of volume"


def congruent(a: Optional[Dict], b: Optional[Dict], volume_tol: float = CONGRUENT_VOLUME,
              mesh_tol: float = CONGRUENT_MESH) -> bool:
    """Two piece descriptors equal within the tolerances (same side of the difference, same face counts)."""
    if a is None or b is None or a["sign"] != b["sign"] or a["faces"] != b["faces"]:
        return False
    if abs(a["volume"] - b["volume"]) > volume_tol * max(a["volume"], b["volume"]):
        return False
    if abs(a["area"] - b["area"]) > mesh_tol * max(a["area"], b["area"]):
        return False
    scale = float(max(a["moments"].max(), b["moments"].max()))
    return bool(np.all(np.abs(a["moments"] - b["moments"]) <= mesh_tol * scale))


def group_congruent(descriptors: Sequence[Optional[Dict]]) -> List[List[int]]:
    """Indices grouped by congruence, in the given order (each group led by its first member, the representative);
    a piece without a descriptor is a group of its own."""
    groups: List[List[int]] = []
    for i, d in enumerate(descriptors):
        for g in groups:
            if congruent(descriptors[g[0]], d):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


PROPER_FLIPS = (np.diag([1.0, 1.0, 1.0]), np.diag([1.0, -1.0, -1.0]), np.diag([-1.0, 1.0, -1.0]),
                np.diag([-1.0, -1.0, 1.0]))


def degenerate_axis(d: Dict, tol: float = CONGRUENT_MESH) -> Optional[int]:
    """The index of the principal axis whose moment differs from the two others when those two are equal (within
    `tol` of the largest moment): the pair's axes are then free in their plane. None when all three differ or all
    three are equal."""
    m = d["moments"]
    eq = [abs(m[i] - m[j]) <= tol * float(m.max()) for i, j in ((1, 2), (0, 2), (0, 1))]
    if sum(eq) != 1:
        return None
    return eq.index(True)


def _spin(k: int, angle: float) -> np.ndarray:
    c, s_ = np.cos(angle), np.sin(angle)
    i, j = [n for n in range(3) if n != k]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s_, s_, c
    return R


def transfer_candidates(p: Primitive, source: Dict, target: Dict, spin: float = 0.0) -> List[Primitive]:
    """The primitive `p` fitted on the piece `source` moved onto the congruent piece `target` (descriptors): the rigid
    motions x -> R x + t that map the source's centroid and principal axes onto the target's, R = A_t Z S A_s^T for the
    four proper sign choices S (the axes' signs are not defined by the moments) and a turn Z by `spin` about the
    distinct axis when two moments are equal (`degenerate_axis`; the caller searches the angle by the score - measured
    on a 5 x 2 x 2 box: 0.850 without the search). The caller keeps the one that scores best on the target."""
    import copy
    from reco_engine.result_to_program import transform_primitive
    k = degenerate_axis(target)
    Z = _spin(k, spin) if k is not None and spin else np.eye(3)
    out = []
    for S in PROPER_FLIPS:
        R = target["axes"] @ Z @ S @ source["axes"].T
        t = target["centre"] - R @ source["centre"]
        q = transform_primitive(copy.deepcopy(p), R, t)
        q.alternatives = []
        out.append(q)
    return out


def best_transfer(p: Primitive, source: Dict, target: Dict, score) -> tuple:
    """(score, primitive) of the best transfer of `p` onto `target`, `score(primitive) -> float`: the four sign
    choices, and with two equal moments the turn about the distinct axis searched at 15 degrees, then refined by
    halving the step three times around the best."""
    def best_of(spin):
        return max(((score(q), q) for q in transfer_candidates(p, source, target, spin)), key=lambda t: t[0])
    if degenerate_axis(target) is None:
        return best_of(0.0)
    tried = {a: best_of(np.radians(a)) for a in range(0, 180, 15)}
    a = max(tried, key=lambda x: tried[x][0])
    step = 7.5
    for _ in range(3):
        for b in (a - step, a + step):
            if b not in tried:
                tried[b] = best_of(np.radians(b))
        a = max(tried, key=lambda x: tried[x][0])
        step /= 2.0
    return tried[a]


# ------------------------------------------------------------------------------------------------ voxel fallback
# The user's rule 2026-09-29 ("when reconstruction fails, always use voxels as compound, for stability"; "before
# going into fallback check other solution"): a piece no tool explains - after every other solution was tried - gets
# its own material as a compound of voxel cubes merged into few boxes. Precision first: a cell counts only when its
# centre and its eight corners are inside the piece, so the compound lies inside the piece and what it leaves is said.

FALLBACK_MAX_CELLS = 64_000         # the fallback's grid: at most this many cells
FALLBACK_REFINE = 2.0               # its pitch: the licensed one (tau / 3, D-005) over this, where the cells allow it
FALLBACK_MARGIN = 0.02              # the grid keeps this share of a pitch off the piece's box (no coplanar faces)


def voxel_tool(V: np.ndarray, F: np.ndarray, tau: Optional[float] = None, frames: Optional[Dict] = None,
               max_boxes: int = 16, max_cells: int = FALLBACK_MAX_CELLS, refine: float = FALLBACK_REFINE,
               min_voxels: int = 1, max_side: int = 96) -> Optional[Dict]:
    """The piece's mesh as boxes of voxel cubes (rl.largest_boxes: the largest completely filled boxes first, at
    most `max_boxes`), fused to one composite. The grid: cells of the licensed pitch tau / 3 (`tau`: the piece's
    thinnest wall; at most 1 / PIECE_RESOLUTION[0] of its longest side) over `refine`, coarsened by 1.25 as far as
    `max_cells` requires (`relaxed` when that ends above the licensed pitch) and then shrunk until whole cells fill
    the thinnest extent of the piece's box, centred in the piece's box of each of `frames` ({name: 3x3, columns the axes}; default the world's) - the frame that covers more is taken. None when
    no cell lies inside. Returns dict(primitive, boxes, frame, pitch, licensed_pitch, relaxed, tau, cells, voxels,
    voxel_volume (the cells inside), covered_volume (the boxes'), box_voxels)."""
    V, F = np.asarray(V, float), np.asarray(F)
    best = None
    for name, R in (frames or dict(world=np.eye(3))).items():
        R = np.asarray(R, float)
        Q = V @ R                                                    # the frame's coordinates (R's columns: its axes)
        lo, hi = Q.min(axis=0), Q.max(axis=0)
        t = float(tau) if tau else float((hi - lo).min())
        licensed = min(t / 3.0, float((hi - lo).max()) / PIECE_RESOLUTION[0])
        pitch = licensed / max(1.0, float(refine))
        while np.prod(np.maximum(1.0, np.floor((hi - lo) / pitch))) > max_cells:
            pitch *= 1.25
        thin = float((hi - lo).min())                                # whole cells across the thinnest extent: a plate
        pitch = thin / (np.ceil(thin / pitch - 1e-9) + 2.0 * FALLBACK_MARGIN)   # of 6 cells keeps 6, not 5 and a rest
        n = np.floor((hi - lo) / pitch - 2.0 * FALLBACK_MARGIN + 1e-9).astype(int)
        if np.any(n < 1):
            continue
        start = (lo + hi) / 2.0 - n * pitch / 2.0
        ax = [start[k] + pitch * np.arange(n[k] + 1) for k in range(3)]
        corners = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
        inside_c = _inside_mesh(V, F, corners @ R.T).reshape(tuple(n + 1))
        mask = np.ones(tuple(n), bool)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    mask &= inside_c[di:di + n[0], dj:dj + n[1], dk:dk + n[2]]
        if mask.any():
            idx = np.argwhere(mask)
            centres = start + (idx + 0.5) * pitch
            mask[tuple(idx.T)] = _inside_mesh(V, F, centres @ R.T)
        if not mask.any():
            continue
        boxes = rl.largest_boxes(mask, count=max_boxes, max_side=int(max_side), min_voxels=min_voxels)
        covered = int(sum(np.prod(b - a + 1) for a, b in boxes))
        if boxes and (best is None or covered * pitch ** 3 > best["covered_volume"]):
            best = dict(frame=name, R=R, start=start, pitch=float(pitch), licensed_pitch=float(licensed),
                        relaxed=bool(pitch > licensed * 1.0001), tau=t, cells=int(np.prod(n)), voxels=int(mask.sum()),
                        voxel_volume=float(mask.sum()) * pitch ** 3, covered_volume=covered * pitch ** 3,
                        box_voxels=covered, index_boxes=boxes)
    if best is None:
        return None
    R, prims = best.pop("R"), []
    start = best.pop("start")
    for a, b in best.pop("index_boxes"):
        c = start + (a + b + 1) / 2.0 * best["pitch"]
        h = (b - a + 1) / 2.0 * best["pitch"]
        prims.append(Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(float(x) for x in R @ c),
                               dims_mm=dict(x_half=float(h[0]), y_half=float(h[1]), z_half=float(h[2])), confidence=0.0,
                               sources=[StageSource(s, "voxel_fallback", 0.0)
                                        for s in ("type", "rotation", "dimensions", "position")]))
    label = (f"voxel fallback, {len(prims)} box(es) of {best['box_voxels']} cube(s) at pitch {best['pitch']:.3g}")
    tool = prims[0]
    for q in prims[1:]:
        tool = compose(tool, q, "fuse")
    tool.label = label
    return dict(best, primitive=tool, boxes=len(prims))


def subpart_tool(piece: Dict, V: np.ndarray, F: np.ndarray, max_parts: int = 6, min_gain: float = 0.03,
                 min_saving: float = 0.1, n: int = 20000, seed: int = 0,
                 part_diagonal: Optional[float] = None) -> Optional[Dict]:
    """The piece as several sub-parts fused to one tool (the user's rule 2026-09-29: "for not reconstructable parts of
    features ... assume we add more subparts"): the cover of reco_engine/subpart_cover.py on the piece's own mesh and
    faces - per sub-part the primitive, searched in the space of every primitive type, that holds it and reproduces
    the most of the piece's surface exactly. Every primitive BOUNDS its sub-part, so the tool holds the piece
    (recall about 1) and what decides is its precision: the share of the tool that lies inside the piece, sampled.
    None when the piece has no mesh to cover. Returns dict(primitive, parts, kinds, exact, containment, precision,
    recall, tool_volume, history). `part_diagonal`: the input's, the yardstick of the cover's torus guard (default:
    the piece's)."""
    from reco_engine import subpart_cover as sc
    V, F = np.asarray(V, float), np.asarray(F)
    if len(V) < 4 or len(F) < 4:
        return None
    c = sc.cover(V, F, list(piece.get("faces") or []), max_parts=max_parts, min_gain=min_gain, min_saving=min_saving,
                 part_diagonal=part_diagonal)
    prims = list(c["primitives"])
    if not prims:
        return None
    tool = prims[0]
    for q in prims[1:]:
        tool = compose(tool, q, "fuse")
    tool.label = f"sub-parts, {len(prims)}: " + " | ".join(q.kind for q in prims)
    sdf = primitive_to_sp(tool)
    lo, hi = (np.asarray(b, float) for b in sdf.bounds())
    lo, hi = np.minimum(lo, V.min(axis=0)), np.maximum(hi, V.max(axis=0))
    pts = lo + np.random.default_rng(seed).random((int(n), 3)) * (hi - lo)
    in_tool = np.asarray(sdf.sdf(pts)) <= 0
    in_piece = _inside_mesh(V, F, pts)
    both = float((in_tool & in_piece).sum())
    return dict(primitive=tool, parts=len(prims), kinds=[q.kind for q in prims], exact=float(c["exact"]),
                containment=float(c["containment"]), history=c["history"],
                precision=both / max(1.0, float(in_tool.sum())), recall=both / max(1.0, float(in_piece.sum())),
                tool_volume=float(in_tool.mean() * np.prod(hi - lo)))


def guess_feature(piece: Dict, rough: Optional[ReconstructionResult] = None, registry=None, models=None,
                  sources: Sequence[str] = ("step faces", "fit", "cascade"), stop_at: Optional[float] = STOP_AT,
                  thin: float = 0.0) -> Feature:
    """The feature of one piece of the difference (a dict of exact_residual's `pieces`): the best of
    `guess_candidates`; `sources_agree` speaks for the sources that were asked. `thin` (thin_wall_threshold): a
    piece that is super thin as a whole is not guessed - its Feature has no primitive, its exact volume and
    `suppressed` (the reason)."""
    op = "cut" if piece["sign"] == "minus" else "add"
    real = [f for f in piece["faces"] if f["origin"] in REAL]
    counts: Dict[str, int] = {}
    for f in real:
        counts[f["type"]] = counts.get(f["type"], 0) + 1
    th = None
    if thin > 0 and piece.get("stl"):
        from src.closest_point_field import load_field
        th = piece_thickness(load_field(piece["stl"]), thin)
    common = dict(id=piece["id"], op=op, host=_host(piece, rough), volume=float(piece["volume"]), faces=counts,
                  step=piece.get("step"), stl=piece.get("stl"), thickness=th)
    if th is not None and th["suppressed"]:
        return Feature(kind=UNKNOWN, primitive=None, score=0.0, source="suppressed", depth=None, sources_agree=False,
                       suppressed=suppressed_reason(th), **common)
    g = guess_candidates(piece, registry, models, sources, stop_at, thin=thin)
    cands, failed = g["candidates"], g["failed"]
    if not cands:
        return Feature(kind=UNKNOWN, primitive=None, score=0.0, source="; ".join(f"{k}: {v}" for k, v in failed.items()),
                       depth=None, sources_agree=False, **common)
    best = cands[0]
    firsts = {}
    for c in cands:
        firsts.setdefault(c.source, c.primitive.kind)
    return Feature(kind=best.primitive.kind, primitive=_prim_dict(best.primitive), score=best.score, source=best.source,
                   depth=_depth(piece, best.primitive), sources_agree=len(firsts) > 1 and set(firsts.values()) == {best.primitive.kind},
                   alternatives=[dict(source=c.source, kind=c.primitive.kind, score=c.score, primitive=_prim_dict(c.primitive))
                                 for c in cands[1:]], **common)


# ------------------------------------------------------------------------------------------------ the whole stage

def input_box(input_path) -> Optional[tuple]:
    """(lo, hi) of the input from its mesh (the field loader tessellates a STEP), None when it cannot be loaded."""
    try:
        from src.closest_point_field import load_field
        b = load_field(str(input_path)).bounds
        return np.asarray(b[0], float), np.asarray(b[1], float)
    except Exception:
        return None


def feature_stage(input_path, out_dir, config: Optional[rl.ResidualConfig] = None, max_pieces: int = 500,
                  sources: Sequence[str] = ("step faces", "fit", "cascade"), rough_step=None,
                  stop_at: Optional[float] = STOP_AT, thin_wall: float = THIN_WALL,
                  thin_wall_mm: Optional[float] = None) -> Dict:
    """Rough model, its exact difference to the input, one feature per piece. Writes `features.json` into `out_dir`
    and returns its content: dict(summary, features). `rough_step`: a STEP to take as the rough model instead of
    running the rough stage (then no host is assigned). `thin_wall` (share of the input's diagonal) / `thin_wall_mm`
    (absolute): the super-thin threshold, 0 = no suppression; the suppressed pieces stay in `features` and their
    volume in r_plus / r_minus, the summary counts them apart (suppressed_pieces, suppressed_volume)."""
    if Path(str(input_path)).suffix.lower() not in (".step", ".stp"):
        raise ValueError("the feature stage needs a STEP input (the pieces of an STL have no exact faces)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if rough_step is None:
        rough = rough_stage(input_path, out_dir, config)
        result, rough_step = rough["result"], rough["step"]
    else:
        rough, result = None, None
    t1 = time.time()
    r = difference_pieces(input_path, rough_step, out_dir, max_pieces)
    t2 = time.time()
    registry = (config.registry if config is not None and config.registry is not None
                else cascade.load_default_registry()) if "cascade" in sources else None
    box = input_box(input_path) if r["pieces"] and thin_wall_mm is None and thin_wall > 0 else None
    diagonal = float(np.linalg.norm(box[1] - box[0])) if box is not None else 0.0
    thin = thin_wall_threshold(diagonal, thin_wall, thin_wall_mm)
    features = [guess_feature(p, result, registry, config.models if config is not None else None, sources, stop_at, thin)
                for p in r["pieces"]]
    total = r["r_plus"] + r["r_minus"]
    good = sum(f.volume for f in features if f.score >= GOOD_SCORE)
    summary = dict(input=str(input_path), rough_step=str(rough_step),
                   rough_parameters=rough["parameters"] if rough else None,
                   rough_operations=len([o for o in result.operations if o.accepted]) if result else None,
                   rough_valid=bool(rough["grade"]["valid"]) if rough else None,
                   rough_snapped=rough["snapped"] if rough else None,
                   rough_extended=rough["extended"] if rough else None,
                   rough_iou=r["iou"], input_volume=r["input_volume"], rough_volume=r["model_volume"],
                   r_plus=r["r_plus"], r_minus=r["r_minus"], fuzzy=r["fuzzy"], pieces=len(features),
                   cuts=sum(f.op == "cut" for f in features), adds=sum(f.op == "add" for f in features),
                   left_out_volume=r["left_out_volume"], good_score=GOOD_SCORE,
                   good_features=sum(f.score >= GOOD_SCORE for f in features),
                   good_share=(good / total) if total > 0 else None,
                   thin_wall=dict(share=thin_wall, mm=thin_wall_mm, threshold=thin, input_diagonal=diagonal),
                   suppressed_pieces=sum(f.suppressed is not None for f in features),
                   suppressed_volume=float(sum(f.volume for f in features if f.suppressed is not None)),
                   seconds=dict(rough=t1 - t0, difference=t2 - t1, guess=time.time() - t2))
    out = dict(summary=summary, features=[asdict(f) for f in features])
    (out_dir / "features.json").write_text(json.dumps(out, indent=1))
    return out


def format_features(out: Dict) -> str:
    s = out["summary"]
    lines = [f"rough model: {s['rough_operations']} operations, exact IoU {s['rough_iou']:.4f}, "
             f"missed (r+) {s['r_plus']:.6g}, invented (r-) {s['r_minus']:.6g}",
             f"{s['pieces']} pieces ({s['cuts']} cuts, {s['adds']} adds); {s['good_features']} with score >= "
             f"{s['good_score']}" + ("" if s["good_share"] is None else f", holding {100 * s['good_share']:.1f} % of the difference"),
             *[f"snapped:  {e['label']} moved {e['shift']:.4g}, size changed by {e['size_change']:.4g}, "
               f"turned {e['turn_deg']:.4g} deg"
               for e in s.get("rough_snapped") or []],
             *[f"extended: {e['label'] or e['op']} {e['length_before']:.6g} -> {e['length']:.6g}"
               for e in s.get("rough_extended") or []],
             "", f"{'id':<16} {'op':<4} {'kind':<9} {'depth':<8} {'score':>6} {'source':<11} {'agree':<6} "
                 f"{'volume':>12}  {'dimensions':<42} host"]
    for f in out["features"]:
        dims = ", ".join(f"{k} {v:.4g}" for k, v in (f["primitive"] or {}).get("dims_mm", {}).items())
        lines.append(f"{f['id']:<16} {f['op']:<4} {f['kind']:<9} {f['depth'] or '-':<8} {f['score']:>6.3f} "
                     f"{f['source'][:11]:<11} {'yes' if f['sources_agree'] else 'no':<6} {f['volume']:>12.6g}  "
                     f"{dims[:42]:<42} {f['host'] or '-'}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ preview

FCSTD_SCRIPT = r'''
import json, os, FreeCAD, Part
job = json.load(open(os.environ["FEATURES_JOB"]))
doc = FreeCAD.newDocument("features")
def add(name, path, label):
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = Part.read(path)
    obj.Label = label
add("input", job["input"], "input")
add("rough", job["rough"], "rough model")
for i, f in enumerate(job["features"]):
    add(f"f{i:03d}", f["step"], f["label"])
doc.recompute()
doc.saveAs(job["out"])
'''


def write_fcstd(out: Dict, path, freecad: Optional[str] = None, timeout: int = 900) -> Path:
    """A FreeCAD document with the input, the rough model and every piece as its own labelled object."""
    from reco_field.export import find_freecadcmd
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        raise RuntimeError("freecadcmd not found")
    path = Path(path).resolve()

    def label(f):
        dims = ", ".join(f"{k} {v:.4g}" for k, v in (f["primitive"] or {}).get("dims_mm", {}).items())
        return f"{f['op']} {f['kind']}" + (f" {f['depth']}" if f["depth"] else "") + f" ({dims}) score {f['score']:.2f}"

    with tempfile.TemporaryDirectory() as tmp:
        script, job = Path(tmp) / "fcstd.py", Path(tmp) / "job.json"
        script.write_text(FCSTD_SCRIPT)
        job.write_text(json.dumps(dict(input=str(Path(out["summary"]["input"]).resolve()), rough=out["summary"]["rough_step"],
                                       out=str(path), features=[dict(step=f["step"], label=label(f))
                                                                for f in out["features"] if f.get("step")])))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["FEATURES_JOB"] = str(job)
        proc = subprocess.run([freecad, str(script)], env=env, capture_output=True, text=True, timeout=timeout)
    if not path.is_file():
        raise RuntimeError(f"FreeCAD wrote no document: {(proc.stderr or proc.stdout)[-400:]}")
    return path


# ------------------------------------------------------------------------------------------------ splitting a piece
# The user's decision 2026-09-27 (the live liner run): "the feature phase should split the excess and guess shapes on
# the segments of the voxel field". A piece no single primitive explains (the liner's four largest invented pieces:
# best volume IoU 0.15-0.59) is put on its own grid - ray-parity voxels at a pitch licensed by its thinnest wall
# (tau / 3, D-005) - with its field channels (occupancy, signed distance, vector to the closest surface point) computed
# ONCE, and cut into segments by one of three rules; every segment is guessed and scored for what it is: a cut that
# removes PART of the excess (precision, recall), not a replacement of the whole piece (volume IoU).

SPLIT_METHODS = ("necks", "surfaces", "axis")


PIECE_MAX_CELLS = 1_000_000        # a piece's split grid: at most this many cells (4e6 on the liner: 644 s of grids)


def piece_grid(fld, backend: Optional[str] = "igl", max_cells: int = PIECE_MAX_CELLS, region=None, bbox=None,
               chunk: int = 500_000, thin: float = 0.0):
    """(spec, channels, info) of a piece on its own grid: a box of cells at the pitch its thinnest wall licenses
    (tau / 3, tau the 5th percentile of the wall thickness of its mesh; D-005), plus two empty cells. When that needs
    more than `max_cells`, the pitch is relaxed to what the cap allows and info["relaxed"] says so - a split at a
    documented coarser pitch rather than none (2026-09-28: on the liner no rejected piece was split, their walls of
    0.37 / 0.65 / 1.6 mm needed 1.3e9 / 9.8e7 / 5.3e6 cells); the exact FreeCAD check stays the judge of every guess.
    `region` (inside(points) -> bool, e.g. excess_region): the voxels are the region's, not the ray parity of the
    piece's mesh (a thin piece's tessellation can be open), with voxel channels; `bbox` then bounds the grid.
    `thin` (the super-thin threshold, thin_wall_threshold): tau is that of the walls at or above it, and the
    occupancy is opened at it (`open_thin`; the channels are then the voxel channels of what is left).
    info: dict(tau, licensed_pitch, pitch, cells, relaxed, licensed, thin_threshold, thin_share, opened_cells)."""
    if bbox is not None:
        lo, hi = (np.asarray(b, float) for b in bbox)
    else:
        lo, hi = np.asarray(fld.bounds[0], float), np.asarray(fld.bounds[1], float)
    th = piece_thickness(fld, thin)
    t5 = th["tau"] if th["tau"] else float((hi - lo).min())
    licensed = min(t5 / 3.0, float((hi - lo).max()) / PIECE_RESOLUTION[0])
    pitch = licensed
    while np.prod(np.ceil((hi - lo) / pitch) + 4) > max_cells:
        pitch *= 1.25
    info = dict(tau=t5, licensed_pitch=licensed, pitch=pitch, cells=int(np.prod(np.ceil((hi - lo) / pitch) + 4)),
                relaxed=bool(pitch > licensed * 1.0001), licensed=True, thin_threshold=float(thin),
                thin_share=th["thin_share"], opened_cells=0)
    spec = cascade.box_grid_spec(lo, hi, pitch)
    if region is None:
        ch = rl._channels(fld, spec, False, backend)
        occ = np.asarray(ch[0]) > 0.5
        opened = open_thin(occ, pitch, thin)
        info["opened_cells"] = int(occ.sum() - opened.sum())
        return spec, (rl._voxel_channels(opened, spec) if info["opened_cells"] else ch), info
    c = spec.centres().reshape(-1, 3)
    occ = np.concatenate([np.asarray(region(c[i:i + chunk]), bool) for i in range(0, len(c), chunk)]).reshape(spec.shape)
    opened = open_thin(occ, pitch, thin)
    info["opened_cells"] = int(occ.sum() - opened.sum())
    return spec, rl._voxel_channels(opened, spec), info


def _labels_of(parts: List[np.ndarray], shape) -> np.ndarray:
    out = np.zeros(shape, np.int32)
    for i, m in enumerate(parts):
        out[m] = i + 1
    return out


def _adjacency(labels: np.ndarray, value: Optional[np.ndarray] = None) -> Dict[tuple, float]:
    """{(a, b): strength} for face-adjacent labels a < b (both > 0): the number of touching voxel pairs, or with
    `value` the highest min(value_u, value_v) over them (the saddle between two basins)."""
    out: Dict[tuple, float] = {}
    for ax in range(3):
        a = np.moveaxis(labels, ax, 0)
        u, v = a[:-1].ravel(), a[1:].ravel()
        sel = (u > 0) & (v > 0) & (u != v)
        if not sel.any():
            continue
        lo_, hi_ = np.minimum(u[sel], v[sel]), np.maximum(u[sel], v[sel])
        if value is None:
            w = np.ones(int(sel.sum()))
        else:
            val = np.moveaxis(value, ax, 0)
            w = np.minimum(val[:-1].ravel()[sel], val[1:].ravel()[sel])
        key = lo_.astype(np.int64) * (int(labels.max()) + 1) + hi_
        order = np.argsort(key)
        key, w = key[order], w[order]
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        agg = np.add.reduceat(w, starts) if value is None else np.maximum.reduceat(w, starts)
        n = int(labels.max()) + 1
        for k, x in zip(key[starts], agg):
            pair = (int(k // n), int(k % n))
            out[pair] = out.get(pair, 0.0) + float(x) if value is None else max(out.get(pair, -np.inf), float(x))
    return out


def _relabel(labels: np.ndarray) -> np.ndarray:
    ids = np.unique(labels[labels > 0])
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    return lut[labels]


def merge_small(labels: np.ndarray, min_voxels: int, max_segments: int) -> np.ndarray:
    """Segments under `min_voxels` join the neighbour they share most faces with (the largest one when several), and
    while there are more than `max_segments`, the smallest joins its neighbour the same way. Labels 1..n, sizes
    descending."""
    labels = _relabel(labels)
    while True:
        ids, sizes = np.unique(labels[labels > 0], return_counts=True)
        if len(ids) <= 1:
            break
        small = [(s, i) for i, s in zip(ids, sizes) if s < min_voxels]
        if not small and len(ids) <= max_segments:
            break
        _, victim = min(small) if small else min(zip(sizes, ids))
        adj = _adjacency(labels)
        nb = [(w, b if a == victim else a) for (a, b), w in adj.items() if victim in (a, b)]
        if not nb:
            if small and len(small) == len(ids):
                break
            labels[labels == victim] = 0                            # an island: no neighbour to join, dropped
            continue
        size = dict(zip(ids, sizes))
        _, target = max(nb, key=lambda t: (t[0], size[t[1]]))
        labels[labels == victim] = target
    labels = _relabel(labels)
    ids, sizes = np.unique(labels[labels > 0], return_counts=True)
    order = ids[np.argsort(-sizes)]
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[order] = np.arange(1, len(order) + 1)
    return lut[labels]


def segment_necks(occ: np.ndarray, depth: np.ndarray, seed_share: float = 0.25, neck_ratio: float = 0.7,
                  levels: int = 16) -> np.ndarray:
    """Necks: a marker-controlled watershed on the inside distance `depth` (voxels of `occ`). Seeds are the local
    maxima of the depth above `seed_share` of its maximum; the basins are flooded from the top down, level by level
    (geodesic growth inside {depth >= level}); two basins whose saddle (the highest depth at which they touch) reaches
    `neck_ratio` x the lower of their peaks are one blob and merged - only a real narrowing separates them. Voxels no
    basin reached take the nearest label. Returns labels (0 outside `occ`)."""
    from scipy import ndimage
    d = np.where(occ, depth, 0.0)
    top = float(d.max())
    if top <= 0:
        return occ.astype(np.int32)
    peaks = occ & (d >= seed_share * top) & (ndimage.maximum_filter(d, size=3) <= d)
    labels, n = ndimage.label(peaks, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return occ.astype(np.int32)
    for t in np.linspace(top, 0.0, levels + 1)[1:]:
        allowed = occ & (d >= t)
        while True:
            grown = ndimage.grey_dilation(labels, footprint=ndimage.generate_binary_structure(3, 1))
            new = (labels == 0) & allowed & (grown > 0)
            if not new.any():
                break
            labels[new] = grown[new]
    rest = occ & (labels == 0)
    if rest.any():
        idx = ndimage.distance_transform_edt(labels == 0, return_distances=False, return_indices=True)
        labels[rest] = labels[tuple(i[rest] for i in idx)]
    peak = ndimage.maximum(d, labels, index=np.arange(n + 1))
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    top_of = {i: float(peak[i]) for i in range(1, n + 1)}
    for (a, b), saddle in sorted(_adjacency(labels, d).items(), key=lambda t: -t[1]):
        ra, rb = find(a), find(b)
        if ra != rb and saddle >= neck_ratio * min(top_of[ra], top_of[rb]):
            parent[rb] = ra
            top_of[ra] = max(top_of[ra], top_of[rb])
    lut = np.array([find(i) for i in range(n + 1)], np.int32)
    return np.where(occ, lut[labels], 0)


def _normals(occ: np.ndarray, sdf: np.ndarray, vec: np.ndarray, pitch: float) -> np.ndarray:
    """(3, ...) unit outward normals from the field: V (towards the closest surface point, which lies outwards for a
    voxel inside) where it is longer than a quarter pitch, else minus the gradient of the signed distance."""
    g = np.stack(np.gradient(sdf), axis=0)
    n = -g
    lv = np.linalg.norm(vec, axis=0)
    use = lv > 0.25 * pitch
    n[:, use] = vec[:, use]
    return n / np.maximum(np.linalg.norm(n, axis=0), 1e-12)


def fill_proxy(mask: np.ndarray) -> float:
    """How well one box or one cylinder holds a voxel set (cheap stand-in for a guess's score, voxel units): the
    voxels over the volume of their box in their principal frame, or of the cylinder along one of the three
    principal axes (radius and length from the 99th percentile), the best of the four."""
    pts = np.argwhere(mask).astype(float)
    if len(pts) < 8:
        return 0.0
    c = pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((pts - c).T))
    loc = (pts - c) @ vecs
    ext = np.percentile(loc, 99.5, axis=0) - np.percentile(loc, 0.5, axis=0) + 1.0
    best = len(pts) / float(np.prod(ext))
    for k in range(3):
        o = [j for j in range(3) if j != k]
        r = float(np.percentile(np.linalg.norm(loc[:, o], axis=1), 99)) + 0.5
        best = max(best, len(pts) / (np.pi * r * r * ext[k]))
    return float(min(best, 1.0))


_HALF_NEIGHBOURS = [o for o in ((i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
                    if o > (0, 0, 0)]                                  # 13 offsets: each neighbour pair once


def segment_surfaces(occ: np.ndarray, sdf: np.ndarray, vec: np.ndarray, pitch: float, crease_deg: float = 25.0,
                     plane_spread: float = 0.98, near: int = 2) -> np.ndarray:
    """Surfaces: the boundary voxels of `occ` fall into patches - neighbours (26-neighbourhood: a voxelised circle's
    boundary is connected only diagonally) whose field normals turn by less than `crease_deg`. Only significant
    patches (at least 3 x their longest extent in voxels: not the one-voxel strips along a box's edges) seed segments;
    each is planar (mean resultant length of its normals >= `plane_spread`) or curved. Patches of one kind within
    `near` voxels of each other form a group (a box's faces; a cylinder's side), a single-plane group whose normal runs
    along an adjacent curved group's axis (the direction its normals avoid) joins it (the cylinder's cap), and every
    other voxel takes the group of its nearest seed voxel. Separates a box-like part from a cylinder-like part that
    meet without a neck (measured on the synthetic compound of tests/test_feature_phase.py)."""
    from scipy import ndimage
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    B = occ & ~ndimage.binary_erosion(occ)
    if B.sum() < 8:
        return occ.astype(np.int32)
    n = _normals(occ, sdf, vec, pitch)
    index = -np.ones(occ.shape, np.int64)
    index[B] = np.arange(int(B.sum()))
    nb = n[:, B].T
    cos = np.cos(np.radians(crease_deg))
    rows, cols = [], []
    X, Y, Z = index.shape
    for dx, dy, dz in _HALF_NEIGHBOURS:
        a = index[max(0, -dx):X - max(0, dx), max(0, -dy):Y - max(0, dy), max(0, -dz):Z - max(0, dz)]
        b = index[max(0, dx):X - max(0, -dx), max(0, dy):Y - max(0, -dy), max(0, dz):Z - max(0, -dz)]
        u, v = a.ravel(), b.ravel()
        sel = (u >= 0) & (v >= 0)
        u, v = u[sel], v[sel]
        smooth = np.einsum("ij,ij->i", nb[u], nb[v]) >= cos
        rows.append(u[smooth])
        cols.append(v[smooth])
    m = len(nb)
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, patch = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(m, m)), directed=False)
    pos = np.argwhere(B)
    info = []                                                       # (patch id, planar, mean normal, normals)
    for pid in range(int(patch.max()) + 1):
        sel = patch == pid
        cnt = int(sel.sum())
        ext = int((pos[sel].max(axis=0) - pos[sel].min(axis=0)).max()) + 1
        if cnt < 8 or cnt < 3 * ext:
            continue
        mean = nb[sel].mean(axis=0)
        info.append((pid, bool(np.linalg.norm(mean) >= plane_spread), mean / max(np.linalg.norm(mean), 1e-12), nb[sel]))
    if len(info) < 2:
        return occ.astype(np.int32)
    k = len(info)
    grid_of = np.full(occ.shape, -1, np.int64)                       # significant patch index on the grid
    for i, (pid, _, _, _) in enumerate(info):
        grid_of[tuple(pos[patch == pid].T)] = i
    near_pairs = set()
    for i in range(k):
        grown = ndimage.binary_dilation(grid_of == i, iterations=near)
        for j in np.unique(grid_of[grown]):
            if j > i:
                near_pairs.add((i, int(j)))
    pr = [(i, j) for i, j in near_pairs if info[i][1] == info[j][1]]
    _, group = connected_components(coo_matrix((np.ones(len(pr)), ([a for a, _ in pr], [b for _, b in pr])),
                                               shape=(k, k)), directed=False)
    members = {g: [i for i in range(k) if group[i] == g] for g in set(group.tolist())}

    def axis_of(g):                                                 # the direction the group's normals avoid
        N = np.concatenate([info[i][3] for i in members[g]])
        w, v = np.linalg.eigh(N.T @ N)
        return v[:, 0]
    for g in list(members):
        idx = members.get(g)
        if not idx or not info[idx[0]][1]:
            continue
        normals = [info[i][2] for i in idx]
        if min(abs(float(a @ b)) for a in normals for b in normals) < 0.98:
            continue                                                # several plane directions: a box, not a cap
        for g2 in list(members):
            if g2 == g or not members.get(g2) or info[members[g2][0]][1]:
                continue
            touching = any((min(i, j), max(i, j)) in near_pairs for i in idx for j in members[g2])
            if touching and abs(float(normals[0] @ axis_of(g2))) > 0.95:
                members[g2] += idx
                del members[g]
                break
    seeds = np.zeros(occ.shape, np.int32)
    for lab, (g, idx) in enumerate(members.items(), start=1):
        for i in idx:
            seeds[grid_of == i] = lab
    if len(members) < 2:
        return occ.astype(np.int32)
    # every voxel held by exactly one group's shape model takes that group; the rest the nearest seed
    cells = np.argwhere(occ).astype(float)
    held = np.zeros((len(members), len(cells)), bool)
    for lab, (g, idx) in enumerate(members.items()):
        held[lab] = _group_model(cells, [pos[patch == info[i][0]] for i in idx], [info[i] for i in idx])
    ind = ndimage.distance_transform_edt(seeds == 0, return_distances=False, return_indices=True)
    nearest = seeds[tuple(ind)][occ]
    one = held.sum(axis=0) == 1
    label = np.where(one, held.argmax(axis=0) + 1, nearest)
    out = np.zeros(occ.shape, np.int32)
    out[occ] = label
    return _relabel(out)


def _group_model(cells: np.ndarray, patch_cells: List[np.ndarray], patches: List[tuple]) -> np.ndarray:
    """Which `cells` (voxel indices) a group's shape holds, within one cell: a planar group - the intersection of its
    faces' inner half-spaces (a box's faces bound a box); a curved group - the cylinder fitted to it (axis: the
    direction its normals avoid; centre and radius: a circle fitted to its cells across that axis; the axial extent
    of its cells)."""
    if patches[0][1]:
        inside = np.ones(len(cells), bool)
        for pc, (_, _, n, _) in zip(patch_cells, patches):
            inside &= cells @ n <= float((pc @ n).mean()) + 1.0
        return inside
    P = np.concatenate(patch_cells).astype(float)
    N = np.concatenate([pt[3] for pt in patches])
    a = np.linalg.eigh(N.T @ N)[1][:, 0]
    u = np.cross(a, [1.0, 0.0, 0.0] if abs(a[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    x, y = P @ u, P @ v
    A = np.stack([x, y, np.ones_like(x)], axis=1)                    # x^2 + y^2 = 2 cx x + 2 cy y + c
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    radius = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    t = P @ a
    rho = np.hypot(cells @ u - cx, cells @ v - cy)
    tc = cells @ a
    return (rho <= radius + 1.0) & (tc >= t.min() - 1.0) & (tc <= t.max() + 1.0)


def segment_axis(occ: np.ndarray, positions=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) -> np.ndarray:
    """The plain fallback: one cut square to the longest principal axis, at the quantile of the voxels along it where
    the two halves together are held best by one primitive each (the sum of their fill_proxy - a cheap stand-in for
    the halves' best scores)."""
    pts = np.argwhere(occ)
    if len(pts) < 16:
        return occ.astype(np.int32)
    c = pts.mean(axis=0)
    w, vecs = np.linalg.eigh(np.cov((pts - c).T))
    t = (pts - c) @ vecs[:, -1]
    best = None
    for q in positions:
        cut = float(np.quantile(t, q))
        a = np.zeros(occ.shape, bool)
        low = pts[t <= cut]
        a[tuple(low.T)] = True
        b = occ & ~a
        s = fill_proxy(a) + fill_proxy(b)
        if best is None or s > best[0]:
            best = (s, a, b)
    return _labels_of([best[1], best[2]], occ.shape)


def segment_piece(occ: np.ndarray, sdf: np.ndarray, vec: np.ndarray, pitch: float, method: str = "auto",
                  min_voxels: int = 50, max_segments: int = 6) -> tuple:
    """(labels 1..n, method used) of a piece's voxels. "auto": the necks first (blobs joined by narrowings), then the
    surfaces inside every blob (a box-like and a cylinder-like part that meet without a neck), and only when neither
    splits anything the axis; small segments merged (merge_small). A named method runs alone."""
    def run(m, mask):
        if m == "necks":
            return segment_necks(mask, np.maximum(np.where(mask, sdf, 0.0), 0.0) / pitch)
        if m == "surfaces":
            return segment_surfaces(mask, np.where(mask, sdf, -pitch), vec, pitch)
        return segment_axis(mask)
    if method != "auto":
        return merge_small(run(method, occ), min_voxels, max_segments), method
    blobs = merge_small(run("necks", occ), min_voxels, max_segments)
    labels, used, n = np.zeros(occ.shape, np.int32), [], 0
    if blobs.max() >= 2:
        used.append("necks")
    for b in range(1, int(blobs.max()) + 1):
        m = blobs == b
        sub = merge_small(run("surfaces", m), min_voxels, max_segments) if m.sum() >= 2 * min_voxels else m.astype(np.int32)
        if sub.max() >= 2 and "surfaces" not in used:
            used.append("surfaces")
        labels[sub > 0] = sub[sub > 0] + n
        n += int(sub.max())
    if n < 2:
        return merge_small(run("axis", occ), min_voxels, max_segments), "axis"
    return merge_small(labels, min_voxels, max_segments), "+".join(used)


def mask_mesh(mask: np.ndarray, spec) -> tuple:
    """(vertices, faces) of a voxel set's surface: vtk flying edges at 0.5 on the lightly smoothed mask, world
    coordinates of the grid (voxel centres lo + (i + 0.5) pitch)."""
    import vtk
    from scipy import ndimage
    from vtkmodules.util import numpy_support
    f = ndimage.gaussian_filter(np.pad(mask.astype(float), 1), 0.6)
    img = vtk.vtkImageData()
    img.SetDimensions(*f.shape)
    img.SetOrigin(*(np.asarray(spec.lo, float) + (0.5 - 1.0) * spec.pitch))
    img.SetSpacing(spec.pitch, spec.pitch, spec.pitch)
    img.GetPointData().SetScalars(numpy_support.numpy_to_vtk(f.ravel(order="F"), deep=True))
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, 0.5)
    fe.ComputeNormalsOff()
    fe.Update()
    poly = fe.GetOutput()
    if poly.GetNumberOfPoints() == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    V = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    F = numpy_support.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
    return V, F


def voxel_lookup(mask: np.ndarray, spec):
    """inside(points) -> bool: the point's cell is in `mask` (outside the grid: False)."""
    lo, pitch, shape = np.asarray(spec.lo, float), float(spec.pitch), np.array(mask.shape)

    def inside(pts):
        idx = np.floor((np.asarray(pts, float) - lo) / pitch).astype(int)
        ok = np.all((idx >= 0) & (idx < shape), axis=1)
        out = np.zeros(len(idx), bool)
        out[ok] = mask[tuple(idx[ok].T)]
        return out
    return inside


# ------------------------------------------------------------------------------------------------ precision, recall

PR_POINTS = 20000


class PartScorer:
    """Precision and recall of a primitive as a cut (or add) that explains PART of an excess (missing) region:

        precision = share of the primitive's volume that does not touch the input's material wrongly - for a cut,
                    outside the input; for an add, inside it (by the input mesh's fast winding number)
        recall    = share of the region (`inside_region`, its volume `region_volume`) the primitive covers

    Deviation from "inside the model and outside the input" (the user's words, 2026-09-27): the part of a cutter in
    air (outside the model) counts as precise - it removes nothing, and it is exactly what a cutter needs to leave no
    coplanar face (measured on the liner's pocket pieces: the exact STEP-faces box overshoots into the bore's air and
    scored volume IoU 0.58 against a fitted box's 0.963). Seeded points in the primitive's box (clipped to three
    times the region's box). `shrink` moves box faces / cylinder caps and radius inward by `step` until the
    precision holds, on one fixed sample set."""

    def __init__(self, op: str, inside_region, region_lo, region_hi, region_volume: float, inside_input,
                 n: int = PR_POINTS, seed: int = 0):
        self.op, self.inside_region, self.inside_input = op, inside_region, inside_input
        self.lo, self.hi = np.asarray(region_lo, float), np.asarray(region_hi, float)
        self.volume, self.n, self.seed = float(region_volume), n, seed
        self.last_n = 0

    def _points(self, p: Primitive):
        solid = primitive_to_sp(p)
        plo, phi = (np.asarray(b, float) for b in solid.bounds())
        size = self.hi - self.lo
        lo, hi = np.maximum(plo, self.lo - size), np.minimum(phi, self.hi + size)
        if np.any(hi <= lo):
            return None, None, 0.0
        pts = lo + np.random.default_rng(self.seed).random((self.n, 3)) * (hi - lo)
        return pts, solid, float(np.prod(hi - lo))

    def _flags(self, pts):
        inp = np.asarray(self.inside_input(pts), bool)
        harm = inp if self.op == "cut" else ~inp
        return harm, np.asarray(self.inside_region(pts), bool)

    @staticmethod
    def _measure(inP, harm, reg, box_volume, n, region_volume):
        k = int(inP.sum())
        if not k:
            return 0.0, 0.0
        precision = 1.0 - float((harm & inP).sum()) / k
        recall = float((reg & inP).sum()) / n * box_volume / region_volume if region_volume > 0 else 0.0
        return precision, float(min(recall, 1.0))

    def score(self, p: Primitive) -> tuple:
        """(precision, recall)."""
        try:
            pts, solid, vol = self._points(p)
        except (ValueError, KeyError):
            return 0.0, 0.0
        if pts is None:
            return 0.0, 0.0
        inP = np.asarray(solid.sdf(pts)) <= 0
        harm, reg = self._flags(pts)
        self.last_n = int(inP.sum())                                 # the points behind the precision
        return self._measure(inP, harm, reg, vol, self.n, self.volume)

    def shrink(self, p: Primitive, min_precision: float, step: float, max_steps: int = 12) -> tuple:
        """(primitive, precision, recall) after moving faces inward, one `step` at a time, the move that raises the
        precision most first, until it reaches `min_precision` (or nothing helps). Boxes and full cylinders only."""
        from reco_engine.result_to_program import rotation_matrix
        if p.kind not in ("box", "cylinder") or p.u or p.v:
            pr, rc = self.score(p)
            return p, pr, rc
        pts, solid, vol = self._points(p)
        if pts is None:
            return p, 0.0, 0.0
        harm, reg = self._flags(pts)

        def measure(q):
            return self._measure(np.asarray(primitive_to_sp(q).sdf(pts)) <= 0, harm, reg, vol, self.n, self.volume)

        R = rotation_matrix(p)
        cur, (pr, rc) = p, measure(p)
        for _ in range(max_steps):
            if pr >= min_precision:
                break
            moves = []
            d = cur.dims_mm
            if cur.kind == "box":
                for axis, key in enumerate(("x_half", "y_half", "z_half")):
                    if d[key] > step:
                        for sgn in (-1.0, 1.0):
                            dims = dict(d, **{key: d[key] - step / 2.0})
                            c = np.asarray(cur.center_mm, float) - R[:, axis] * sgn * step / 2.0
                            moves.append(replace(cur, center_mm=tuple(c), dims_mm=dims))
            else:
                if d["half_length"] > step:
                    for sgn in (-1.0, 1.0):
                        dims = dict(d, half_length=d["half_length"] - step / 2.0)
                        c = np.asarray(cur.center_mm, float) - R[:, 2] * sgn * step / 2.0
                        moves.append(replace(cur, center_mm=tuple(c), dims_mm=dims))
                if d["radius"] > step:
                    moves.append(replace(cur, dims_mm=dict(d, radius=d["radius"] - step)))
            if not moves:
                break
            scored = [(measure(q), q) for q in moves]
            (pr2, rc2), q = max(scored, key=lambda t: (round(t[0][0], 4), t[0][1]))
            if pr2 <= pr:
                break
            cur, pr, rc = q, pr2, rc2
        return cur, pr, rc


def input_inside(V: np.ndarray, F: np.ndarray):
    """inside(points) -> bool by the fast winding number of the input mesh (igl; measured 2026-09-27 on the liner's
    251,062 triangles: 20,000 points in 0.6 s, the same answer as the exact winding number on 2,000 of them)."""
    V, F = np.ascontiguousarray(V, np.float64), np.ascontiguousarray(F, np.int64)

    def inside(pts):
        pts = np.ascontiguousarray(pts, np.float64)
        if not len(pts):
            return np.zeros(0, bool)
        try:
            import igl
            return np.abs(igl.fast_winding_number(V, F, pts)) > 0.5
        except (ImportError, AttributeError):
            return _inside_mesh(V, F, pts)
    return inside


def segment_candidates(seg: Dict, registry=None, models=None, sources: Sequence[str] = ("step faces", "fit", "cascade"),
                       deadline: Optional[float] = None) -> Dict:
    """Candidates for one segment of a split piece (`seg`: mask, spec, faces of the parent piece, op, scorer - see
    the loop's feature phase): the fitters on the segment's voxel surface mesh, the cascade on its voxel channels
    (a window of the piece grid), and the parent's exact faces whose probe point lies within two cells of the
    segment. Each scored by the PartScorer (precision, recall; score = min of both), shrunk when the precision misses
    `min_precision` by at most 0.1. dict(candidates best first, failed, timed_out) as guess_candidates."""
    from src.closest_point_field import SurfaceField
    from scipy import ndimage
    mask, spec, scorer = seg["mask"], seg["spec"], seg["scorer"]
    V, F = mask_mesh(mask, spec)
    if len(F) < 4:
        return dict(candidates=[], failed=dict(mesh="the segment has no surface"), timed_out=False)
    fld = SurfaceField(V, F)
    part_diagonal = seg.get("part_diagonal")                         # the torus guard's yardstick (the input's)

    def near_faces():
        near = voxel_lookup(ndimage.binary_dilation(mask, iterations=2), spec)
        faces = [f for f in seg.get("faces") or [] if near(np.asarray([f["point"]], float))[0]]
        return metadata_candidates(dict(faces=faces), V) if faces else []

    def cascade_on_segment():
        pts = np.argwhere(mask)
        i0, i1 = np.maximum(pts.min(axis=0) - 4, 0), np.minimum(pts.max(axis=0) + 5, mask.shape)
        sub = cascade.BoxGridSpec(tuple(float(spec.lo[d] + i0[d] * spec.pitch) for d in range(3)),
                                  tuple(int(n) for n in i1 - i0), spec.pitch)
        win = mask[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]]
        prim = cascade.run_cascade(rl._voxel_channels(win, sub), sub, registry, field=fld, top_k=2, models=models,
                                   part_diagonal=part_diagonal)
        return [p for p in [prim] + list(prim.alternatives) if p.kind != UNKNOWN]

    proposers = {"step faces": near_faces, "fit": lambda: fit_candidates(fld), "cascade": cascade_on_segment}
    cands, failed, timed_out = [], {}, False
    admit = torus_gate(part_diagonal, failed)
    for name in sources:
        if deadline is not None and time.time() > deadline:
            timed_out = True
            break
        t0 = time.time()
        try:
            found = proposers[name]()
        except Exception as exc:
            failed[name] = str(exc)[-200:]
            continue
        t_source = time.time() - t0
        for p in found:
            if not admit(p, name):
                continue
            t1 = time.time()
            pr, rc = scorer.score(p)
            if seg["min_precision"] - 0.1 <= pr < seg["min_precision"]:
                p, pr, rc = scorer.shrink(p, seg["min_precision"], spec.pitch)
                if p.label is None or "shrunk" not in (p.label or ""):
                    p.label = ((p.label + ", ") if p.label else "") + "shrunk"
                if not admit(p, name):
                    continue
            c = Candidate(name, p, min(pr, rc), t_source + time.time() - t1)
            c.precision, c.recall = pr, rc
            cands.append(c)
    order = {"step faces": 0, "fit": 1, "cascade": 2}
    ok = [c for c in cands if c.precision >= seg["min_precision"]]
    top = max((c.recall for c in ok), default=0.0)
    # the precise ones first (by recall; within SCORE_TIE the exact faces, then a fit), then the rest by score
    cands.sort(key=lambda c: (c.precision < seg["min_precision"],
                              (c.recall < top - SCORE_TIE, order.get(c.source, 3), -c.recall)
                              if c.precision >= seg["min_precision"] else (0, 0, -c.score)))
    return dict(candidates=cands, failed=failed, timed_out=timed_out)


class SampledJudge:
    """The residual of a program against the input on ONE seeded point set, for when OCCT's booleans fail on a pair
    (coincident faces: 2026-09-27 on the liner, an exact R 125 cut on its R 125 bore - common 0, even fuzzy): points
    uniform in the input's box enlarged by `pad`, the input's side by its mesh (fast winding number, computed once),
    the model's by the program's CSG signed distance. residual(model) -> (missing, invented) in volume units; the
    same points judge before and after, so their difference is the difference of the operation (its sampling error
    about box volume x sqrt(changed share / n))."""

    def __init__(self, inside_input, lo, hi, n: int = 1_000_000, pad: float = 0.05, seed: int = 0, chunk: int = 200_000):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        size = hi - lo
        self.lo, self.hi = lo - pad * size, hi + pad * size
        self.pts = self.lo + np.random.default_rng(seed).random((n, 3)) * (self.hi - self.lo)
        self.chunk = chunk
        self.inp = np.concatenate([np.asarray(inside_input(self.pts[i:i + chunk]), bool)
                                   for i in range(0, n, chunk)])
        self.cell = float(np.prod(self.hi - self.lo)) / n

    def residual(self, model) -> tuple:
        inm = np.concatenate([np.asarray(model.sdf(self.pts[i:i + self.chunk])) <= 0
                              for i in range(0, len(self.pts), self.chunk)])
        return float((self.inp & ~inm).sum()) * self.cell, float((inm & ~self.inp).sum()) * self.cell


def excess_region(op: str, inside_input, model, lo, hi, pad: float = 0.01):
    """inside(points) of the piece a feature explains, from its definition rather than its mesh: in the piece's box
    (enlarged by `pad`), inside the model and outside the input for a cut (invented), inside the input and outside
    the model for an add (missing). `model`: the program's SDF solid at the difference (negative inside)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    lo, hi = lo - pad * (hi - lo), hi + pad * (hi - lo)

    def inside(pts):
        pts = np.asarray(pts, float)
        out = np.zeros(len(pts), bool)
        box = np.all((pts >= lo) & (pts <= hi), axis=1)
        if box.any():
            q = pts[box]
            m = np.asarray(model.sdf(q)) <= 0
            i = np.asarray(inside_input(q), bool)
            out[box] = (m & ~i) if op == "cut" else (i & ~m)
        return out
    return inside


# ------------------------------------------------------------------------------------------------ boolean tools
# The user's rule 2026-09-28 ("use tools made with booleans for the features too"): a piece no single primitive
# explains precisely - the liner's pockets have 30 / 55 mm corner radii, and a plain box removed 443 mm^3 of input
# material at them (the exact check rejected all nine) - gets composites of the primitives read from its faces.

def leaves(p: Primitive) -> List[Primitive]:
    return [x for q in (p.parts or []) for x in leaves(q)] if p.kind == "composite" else [p]


def compose(a: Primitive, b: Primitive, combine: str, label: Optional[str] = None) -> Primitive:
    """a & b ("common"), a - b ("cut"), a | b ("fuse") as a schema composite (result_to_program builds it)."""
    return Primitive(kind="composite", center_mm=tuple(a.center_mm), dims_mm={}, parts=[a, b], combine=combine,
                     label=label, confidence=1.0, sources=list(a.sources))


def cylinder_faces(piece: Dict, V: np.ndarray, real_only: bool = True) -> List[Primitive]:
    """One full cylinder per cylindrical face of the piece (the input's faces first: origin input / both), as long as
    the piece along its axis. Several faces on one surface (a split seam) give one cylinder."""
    out, seen = [], []
    for f in piece.get("faces") or []:
        if f["type"] != "cylinder" or (real_only and f["origin"] not in REAL):
            continue
        a = np.asarray(f["axis"], float)
        a /= np.linalg.norm(a)
        c = np.asarray(f["center"], float)
        key = (round(float(f["radius"]), 6),)
        dup = False
        for (a2, c2, r2) in seen:
            off = c - c2
            if abs(abs(a @ a2) - 1) < 1e-6 and np.linalg.norm(off - a2 * (off @ a2)) < 1e-6 and abs(r2 - key[0]) < 1e-6:
                dup = True
        if dup:
            continue
        seen.append((a, c, key[0]))
        t = (V - c) @ a
        lo, hi = float(t.min()), float(t.max())
        if hi - lo <= 0:
            continue
        out.append(Primitive(kind="cylinder", quaternion=quaternion_of(_frame(a)),
                             center_mm=tuple(float(v) for v in c + a * 0.5 * (lo + hi)),
                             dims_mm=dict(radius=float(f["radius"]), half_length=0.5 * (hi - lo)),
                             confidence=1.0, sources=_faces_source()))
    return out


def rounded_box(box: Primitive, cylinders: Sequence[Primitive], tol: float = 0.03, over: float = 0.02) -> Optional[Primitive]:
    """The box with its rounded corners / ends: for every cylinder parallel to a box axis k whose axis lies r from
    two walls of the box (tangent to them; r its radius, within `tol` x r), the corner square between the axis and
    the corner, minus the cylinder, is cut from the box - B - (S1 - C1) - (S2 - C2) ... exact for any radii (a slot's
    half-cylinder end is two such corners). The squares and cylinders run `over` of the length past the box along k
    (no coplanar faces). None when no cylinder rounds a corner."""
    from reco_engine.result_to_program import rotation_matrix
    R, c = rotation_matrix(box), np.asarray(box.center_mm, float)
    h = np.array([box.dims_mm["x_half"], box.dims_mm["y_half"], box.dims_mm["z_half"]], float)
    result, n = box, 0
    for cyl in cylinders:
        a = rotation_matrix(cyl)[:, 2]
        k = int(np.argmax(np.abs(R.T @ a)))
        if abs(R[:, k] @ a) < 0.999:
            continue
        i, j = [m for m in range(3) if m != k]
        r = float(cyl.dims_mm["radius"])
        q = R.T @ (np.asarray(cyl.center_mm, float) - c)             # the axis in the box's frame
        for si in (-1.0, 1.0):
            for sj in (-1.0, 1.0):
                di, dj = si * h[i] - q[i], sj * h[j] - q[j]          # from the axis to the two walls, signed
                if abs(abs(di) - r) > tol * r or abs(abs(dj) - r) > tol * r or di * si < 0 or dj * sj < 0:
                    continue
                lo_l, hi_l = np.empty(3), np.empty(3)
                lo_l[i], hi_l[i] = sorted((q[i], si * h[i] + si * over * r))
                lo_l[j], hi_l[j] = sorted((q[j], sj * h[j] + sj * over * r))
                lo_l[k], hi_l[k] = -h[k] * (1 + over), h[k] * (1 + over)
                sq = Primitive(kind="box", quaternion=box.quaternion,
                               center_mm=tuple(float(v) for v in c + R @ ((lo_l + hi_l) / 2)),
                               dims_mm=dict(zip(("x_half", "y_half", "z_half"), (float(v) for v in (hi_l - lo_l) / 2))),
                               confidence=1.0, sources=_faces_source())
                ax_c = c + R @ np.array([q[m] if m != k else 0.0 for m in range(3)])
                cy = Primitive(kind="cylinder", quaternion=quaternion_of(_frame(R[:, k])), center_mm=tuple(float(v) for v in ax_c),
                               dims_mm=dict(radius=r, half_length=h[k] * (1 + 2 * over)), confidence=1.0,
                               sources=_faces_source())
                result = compose(result, compose(sq, cy, "cut"), "cut")
                n += 1
    if not n:
        return None
    result.label = f"rounded box ({n} rounded corner(s))"
    return result


def tool_candidates(piece: Dict, V: np.ndarray, singles: Sequence[Primitive], max_operands: int = 9,
                    max_candidates: int = 12) -> List[Primitive]:
    """Composite tools for one piece, fewest operands first: the rounded box of the piece's planes and cylinder faces
    (rounded_box), then the pairwise booleans of the best single candidates and of the box with each cylinder face -
    a & b (a box with a rounded end), a | b (a stepped bore), a - b. At most `max_candidates`, each with at most
    `max_operands` primitives."""
    cyls = cylinder_faces(piece, V)
    box = _box_of_planes([f for f in piece.get("faces") or [] if f["type"] == "plane"], V)
    exact: List[Primitive] = []                                      # from the piece's exact faces: always offered
    if box is not None and cyls:
        rb = rounded_box(box, cyls)
        if rb is not None:
            exact.append(rb)
    if box is not None and len(cyls) >= 2:                          # a slot as a union: box | cylinder | cylinder
        exact.append(compose(compose(box, cyls[0], "fuse"), cyls[1], "fuse", "box | cylinder | cylinder"))
    base: List[Primitive] = []
    for p in ([box] if box is not None else []) + [p for p in singles if p.kind in ("box", "cylinder", "cone", "sphere")]:
        if all(p is not q for q in base) and len(base) < 3:
            base.append(p)
    others = cyls[:4] + [p for p in singles if p.kind in ("cylinder", "sphere", "cone")][:2]
    pairs = []
    for a in base:
        for b in others:
            if a is b:
                continue
            for combine, sym in (("common", "&"), ("fuse", "|"), ("cut", "-")):
                pairs.append(compose(a, b, combine, f"{a.kind} {sym} {b.kind}"))
    exact = [p for p in exact if len(leaves(p)) <= max_operands]
    pairs = [p for p in pairs if len(leaves(p)) <= max_operands][:max(0, max_candidates - len(exact))]
    return sorted(exact + pairs, key=lambda p: len(leaves(p)))[:max_candidates]


# ------------------------------------------------------------------------------------------------ contact-face tool
# The user's rule (2026-09-28) as a guess source: a piece's faces are CONTACT faces (on the input's surface: what the
# tool must reproduce) or TOOL-ONLY faces (on the current model's surface: they only close the piece). The tool is
# built from the contact faces alone and left open - run on past the piece - wherever only tool-only faces bound it.
# Measured by hand on the liner's pocket pieces #1, #2, #4, #5 (exact, ~/.cache/shape_reco/investigate/liner_0928):
# the whole piece as a sharp box harms 11 - 12 % of the piece's volume in the input, this tool 0 mm^3 at recall 1.

CONTACT_OPEN = 0.05                 # an open side runs this share of the piece's size along that axis past the piece


def _closed_sides(contact: Sequence[Dict], R: np.ndarray, lo: np.ndarray, hi: np.ndarray, tol: float) -> Dict[tuple, float]:
    """{(axis, sign): position} of the box sides the contact faces close: a contact plane facing that way (the
    outermost one), or a contact cylinder / sphere / torus whose surface ends there (tangent to the piece's extent:
    a slot's round end) - such a side must not move, the rounding is tangent to it."""
    out: Dict[tuple, float] = {}
    size = float(np.linalg.norm(hi - lo))
    for i in range(3):
        for sign in (-1.0, 1.0):
            ext = hi[i] if sign > 0 else lo[i]
            pos = [float(np.asarray(f["point"], float) @ R[:, i]) for f in contact if f["type"] == "plane"
                   and float(np.asarray(f["normal"], float) @ R[:, i]) * sign >= SNAP_COS]
            if pos:
                out[(i, sign)] = min(ext, max(pos)) if sign > 0 else max(ext, min(pos))
                continue
            for f in contact:
                if f["type"] not in ("cylinder", "sphere", "torus") or "center" not in f:
                    continue
                if float(np.asarray(f["normal"], float) @ R[:, i]) * sign <= 0:     # the face looks the other way
                    continue
                r = float(f["radius"]) if f["type"] != "torus" else float(f["major_radius"]) + float(f["minor_radius"])
                if f["type"] != "sphere" and abs(float(np.asarray(f["axis"], float) @ R[:, i])) > 1e-3:
                    if f["type"] == "cylinder":
                        continue                                                    # along its axis a cylinder is open
                    r = float(f["minor_radius"])
                reach = float(np.asarray(f["center"], float) @ R[:, i]) + sign * r
                if abs(reach - ext) <= tol * max(r, 1e-9) + 1e-6 * size:
                    out[(i, sign)] = reach
                    break
    return out


def rounded_corners(tool: Primitive, box: Primitive, tori: Sequence[Dict], tol: float = 0.03, over: float = 0.02,
                    guard: Optional[Dict[str, List[str]]] = None) -> tuple:
    """(tool, n): the corners where two rounded edges of `box` meet. A contact torus (major M, minor m) whose axis is
    a box axis k, M + m from two walls and m (along k) from a third one, is the corner of a rounded floor: between
    the torus' plane and that wall the tool is the torus and the cylinder of radius M inside it - the corner block
    minus both is cut from the tool, T - ((S - torus) - C). `n` corners were rounded. The torus guard
    (reco_engine/torus_guard.py): a horn or near-horn face (M = m: the round end of a slot whose half width is twice
    its floor's radius) gets the thinned torus (`torus_guard.thinned`: inside the face's torus, the same outer
    equator), any other torus the guard rejects rounds no corner - the corner stays as the box made it. `guard`
    collects the reasons under "thinned" and "rejected"."""
    from reco_engine.result_to_program import rotation_matrix
    R, c = rotation_matrix(box), np.asarray(box.center_mm, float)
    h = np.array([box.dims_mm["x_half"], box.dims_mm["y_half"], box.dims_mm["z_half"]], float)
    n, seen = 0, []
    for f in tori:
        a = np.asarray(f["axis"], float)
        a /= np.linalg.norm(a)
        k = int(np.argmax(np.abs(R.T @ a)))
        if abs(R[:, k] @ a) < 0.999:
            continue
        i, j = [m for m in range(3) if m != k]
        M, m_ = float(f["major_radius"]), float(f["minor_radius"])
        dims = dict(major_radius=M, minor_radius=m_)                 # the corner torus: the face's, or its repair
        found = torus_guard.torus_defect(M, m_)
        if found is not None:
            fixed = torus_guard.thinned(M, m_)
            if guard is not None:
                guard.setdefault("rejected" if fixed is None else "thinned", []).append(found.text)
            if fixed is None:
                continue
            dims = dict(major_radius=fixed[0], minor_radius=fixed[1])
        centre = np.asarray(f["center"], float)
        q = R.T @ (centre - c)
        nf = R.T @ np.asarray(f["normal"], float)                     # the face looks out of the piece: its corner
        for sk in (-1.0, 1.0):
            dk = sk * h[k] - q[k]
            if abs(abs(dk) - m_) > tol * m_ or dk * sk < 0 or nf[k] * sk <= 0:
                continue
            for si in (-1.0, 1.0):
                for sj in (-1.0, 1.0):
                    di, dj = si * h[i] - q[i], sj * h[j] - q[j]
                    if abs(abs(di) - (M + m_)) > tol * (M + m_) or abs(abs(dj) - (M + m_)) > tol * (M + m_) \
                            or di * si < 0 or dj * sj < 0 or nf[i] * si <= 0 or nf[j] * sj <= 0:
                        continue
                    key = (k, sk, si, sj) + tuple(np.round(q, 6))
                    if key in seen:
                        continue
                    seen.append(key)
                    lo_l, hi_l = np.empty(3), np.empty(3)
                    lo_l[i], hi_l[i] = sorted((q[i], si * h[i] + si * over * m_))
                    lo_l[j], hi_l[j] = sorted((q[j], sj * h[j] + sj * over * m_))
                    lo_l[k], hi_l[k] = sorted((q[k], sk * h[k] + sk * over * m_))
                    block = Primitive(kind="box", quaternion=box.quaternion,
                                      center_mm=tuple(float(v) for v in c + R @ ((lo_l + hi_l) / 2)),
                                      dims_mm=dict(zip(("x_half", "y_half", "z_half"),
                                                       (float(v) for v in (hi_l - lo_l) / 2))),
                                      confidence=1.0, sources=_faces_source())
                    quat = quaternion_of(_frame(R[:, k]))
                    torus = Primitive(kind="torus", quaternion=quat, center_mm=tuple(float(v) for v in centre),
                                      dims_mm=dict(dims), confidence=1.0, sources=_faces_source())
                    core = Primitive(kind="cylinder", quaternion=quat,
                                     center_mm=tuple(float(v) for v in centre + R[:, k] * 0.5 * dk),
                                     dims_mm=dict(radius=M, half_length=0.5 * abs(dk) + 2 * over * m_), confidence=1.0,
                                     sources=_faces_source())
                    tool = compose(tool, compose(compose(block, torus, "cut"), core, "cut"), "cut")
                    n += 1
    return tool, n


def guard_note(guarded: Dict[str, List[str]]) -> str:
    """The label's note of what the torus guard did to a tool's corner tori (`rounded_corners`' `guard`)."""
    parts = [f"{len(v)} {k}" for k, v in sorted(guarded.items()) if v]
    return f", corner tori by the torus guard: {', '.join(parts)}" if parts else ""


def contact_tool(piece: Dict, V: np.ndarray, open_by: float = CONTACT_OPEN, tol: float = 0.03) -> Optional[Primitive]:
    """The tool of a piece from its CONTACT faces only (origin input / both), None when it has none, when it has no
    tool-only face (a closed piece: nothing to open, the whole-piece sources read it) or when no contact plane gives
    a frame. The frame is that of the contact planes (the largest first, then the largest one square to it). A side
    the contact faces close (`_closed_sides`) lies on them; every other side is bounded by tool-only faces alone and
    is OPEN: it runs `open_by` of the piece's size along that axis past the piece - into the air or the invented
    material the model's face stood in. Contact cylinders tangent to two sides round the edges (`rounded_box`),
    contact tori the corners where two rounded edges meet (`rounded_corners`). Nothing here is the judge: the tool
    is scored and checked like every other candidate."""
    faces = piece.get("faces") or []
    contact = [f for f in faces if f.get("origin") in REAL]
    if not contact or len(contact) == len(faces):
        return None
    planes = sorted((f for f in contact if f["type"] == "plane"), key=lambda f: -f["area"])
    if not planes:
        return None
    x = np.asarray(planes[0]["normal"], float)
    x /= np.linalg.norm(x)
    y = None
    for n in [f["normal"] for f in planes[1:]] + [f["axis"] for f in contact if f["type"] in ("cylinder", "torus", "cone")]:
        n = np.asarray(n, float)
        n = n - x * (n @ x)
        if np.linalg.norm(n) > 0.99:                        # square to the first within 8 degrees
            y = n / np.linalg.norm(n)
            break
    R = np.stack([x, y, np.cross(x, y)], axis=1) if y is not None else _frame(x)[:, [2, 0, 1]]
    local = np.asarray(V, float) @ R
    lo, hi = local.min(axis=0), local.max(axis=0)
    size = hi - lo
    closed = _closed_sides(contact, R, lo, hi, tol)
    opened = []
    for i in range(3):
        for sign in (-1.0, 1.0):
            if (i, sign) in closed:
                if sign > 0:
                    hi[i] = closed[(i, sign)]
                else:
                    lo[i] = closed[(i, sign)]
            else:
                opened.append((i, sign))
    for i, sign in opened:
        if sign > 0:
            hi[i] += open_by * size[i]
        else:
            lo[i] -= open_by * size[i]
    half = 0.5 * (hi - lo)
    if not (half > 0).all():
        return None
    box = Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in R @ (0.5 * (lo + hi))),
                    dims_mm=dict(x_half=float(half[0]), y_half=float(half[1]), z_half=float(half[2])),
                    confidence=1.0, sources=_faces_source())
    cyls = cylinder_faces(piece, V)
    tool = rounded_box(box, cyls, tol) or box
    n_edges = (len(leaves(tool)) - 1) // 2
    guarded: Dict[str, List[str]] = {}
    tool, n_corners = rounded_corners(tool, box, [f for f in contact if f["type"] == "torus"], tol, guard=guarded)
    names = "xyz"
    tool.label = (f"contact faces, {n_edges} rounded edge(s), {n_corners} corner(s), open "
                  + (" ".join(("+" if s > 0 else "-") + names[i] for i, s in opened) or "nowhere")
                  + guard_note(guarded))
    return tool


HOLE_PARTIAL = 0.02                 # a harmless stack tool that covers this share of its piece is worth its cut


def hole_tools(piece: Dict, V: np.ndarray, over: Optional[float] = None) -> List[tuple]:
    """The tools of a piece's hole stacks (reco_engine/hole_stack.py: per hole axis the cylinders and cones its real
    faces name, each step with its own reach), the most complete first: [(source, primitive, partial)].
    "tool (contact faces + hole stacks)": a piece with more real faces than its stacks' (a pocket with holes in its
    floor) - the contact tool of those other faces fused with the stacks. "tool (hole stack)": the stacks alone;
    `partial` when the piece has other real faces - then the stacks explain only a part of it, harm nothing, and what
    they leave is a piece of the next difference."""
    from reco_engine import hole_stack as hs
    faces = piece.get("faces") or []
    stacks = hs.hole_stacks(faces, V)
    if not stacks:
        return []
    holes = hs.stacks_tool(stacks, over)                             # None: every stack by its own length
    used = set()
    for s in stacks:
        used.update(s["faces"])
        r_max = max(max(q["r0"], q["r1"]) for q in s["segments"])
        t_lo, t_hi = min(q["t0"] for q in s["segments"]), max(q["t1"] for q in s["segments"])
        for i, f in enumerate(faces):                                # its shoulders and caps: planes square to it
            if f["type"] != "plane" or f.get("origin") not in REAL or f["area"] > 1.05 * math.pi * r_max ** 2:
                continue
            n = np.asarray(f["normal"], float)
            off = np.asarray(f["point"], float) - s["base"]
            t = float(off @ s["axis"])
            if abs(n @ s["axis"]) / np.linalg.norm(n) >= 0.9998 and t_lo - 1e-6 <= t <= t_hi + 1e-6 and \
                    np.linalg.norm(off - s["axis"] * t) <= r_max * 1.001:
                used.add(i)
    rest = [f for i, f in enumerate(faces) if i not in used]
    other = [f for f in rest if f.get("origin") in REAL]
    out = []
    if other:
        try:
            pocket = contact_tool(dict(piece, faces=rest), V)
        except Exception:                                            # the stacks alone are still a tool
            pocket = None
        if pocket is not None:
            both = compose(pocket, holes, "fuse", f"{pocket.label} + {holes.label}")
            out.append(("tool (contact faces + hole stacks)", both, False))
    out.append(("tool (hole stack)", holes, bool(other)))
    return out


def precise_enough(c, min_precision: float, z: float = 2.0) -> bool:
    """A candidate's SAMPLED precision is precise enough when it reaches `min_precision` within `z` standard errors
    of its sample (n_inside points; a precision without a sample size is taken as it is). 2026-09-28: an exact
    countersunk hole sampled 0.999 against the threshold 0.999 - inside the sampling error (about 5e-4 there) - and was
    rejected; the exact FreeCAD check (its harm volume) stays the judge of what is applied."""
    p = getattr(c, "precision", None)
    if p is None:
        return True
    n = int(getattr(c, "n_inside", 0) or 0)
    if n <= 0:
        return p >= min_precision
    se = float(np.sqrt(max(p * (1.0 - p), 1.0 / n) / n))
    return p + z * se >= min_precision


# ------------------------------------------------------------------------------------------------ contact grading
# The user's rule (2026-09-28): "for the feature we should think about surfaces which will touch the input, and
# surfaces that only exist on the tool. Keep them in mind for grading." A tool's surface is CONTACT where it lies on
# the input's surface (after the cut it is the part's real surface: the evidence for the tool, its type, position,
# axis and radius must be right there), HARM where it lies inside the input's material, FREE elsewhere (in the
# material the rough model invented, or in the air: an open end above the surface, a through tool's over-length, the
# walls of a fillet's box) - free faces leave no trace and cost nothing.

def _leaf_samples(p: Primitive, n: int, rng, clip_lo, clip_hi):
    """(points (m, 3), area per point) on the surface of one leaf primitive, area-uniform, in world mm."""
    from reco_engine.result_to_program import rotation_matrix
    R, c, d = rotation_matrix(p), np.asarray(p.center_mm, float), p.dims_mm or {}
    if p.kind == "box":
        h = np.array([d["x_half"], d["y_half"], d["z_half"]], float)
        areas = [4 * h[1] * h[2], 4 * h[0] * h[2], 4 * h[0] * h[1]]
        tot = 2 * sum(areas)
        pts = []
        for k in range(3):
            m = max(1, int(round(n * 2 * areas[k] / tot)))
            u = (2 * rng.random((m, 3)) - 1) * h
            u[:, k] = np.where(rng.random(m) < 0.5, -h[k], h[k])
            pts.append(u)
        loc = np.concatenate(pts)
        return c + loc @ R.T, tot / len(loc)
    if p.kind in ("cylinder", "cone"):
        hl = float(d["half_length"])
        r0, r1 = (d["radius"], d["radius"]) if p.kind == "cylinder" else (d["radius1"], d["radius2"])
        side = math.pi * (r0 + r1) * math.hypot(2 * hl, r1 - r0)
        caps = math.pi * (r0 * r0 + r1 * r1)
        tot = side + caps
        m_side = max(1, int(round(n * side / tot)))
        t = rng.random(m_side)
        z = -hl + 2 * hl * t
        r = r0 + (r1 - r0) * t
        a = 2 * math.pi * rng.random(m_side)
        loc = [np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)]
        for rr, zz in ((r0, -hl), (r1, hl)):
            m = max(1, int(round(n * math.pi * rr * rr / tot))) if rr > 0 else 0
            if m:
                rho = rr * np.sqrt(rng.random(m))
                b = 2 * math.pi * rng.random(m)
                loc.append(np.stack([rho * np.cos(b), rho * np.sin(b), np.full(m, zz)], axis=1))
        loc = np.concatenate(loc)
        return c + loc @ R.T, tot / len(loc)
    if p.kind == "sphere":
        r = float(d["radius"])
        v = rng.normal(size=(n, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return c + r * v, 4 * math.pi * r * r / n
    if p.kind == "halfspace":
        nrm = R[:, 2]
        lo, hi = np.asarray(clip_lo, float), np.asarray(clip_hi, float)
        q = lo + rng.random((n, 3)) * (hi - lo)
        q = q - np.outer((q - c) @ nrm, nrm)                   # onto the plane
        area = float(np.prod(np.sort(hi - lo)[1:]))            # the plane's share of the clip box (an estimate)
        return q, area / n
    return np.zeros((0, 3)), 0.0


def tool_surface_samples(p: Primitive, n: int = 6000, seed: int = 0):
    """(points, weights) on the SURFACE of a primitive or boolean tree: every leaf's surface sampled by area, kept
    where it is on the tree's own boundary (|sdf| below 1e-6 of the tree's size)."""
    from reco_engine.result_to_program import primitive_to_sp
    rng = np.random.default_rng(seed)
    sp_all = primitive_to_sp(p)
    lo, hi = (np.asarray(b, float) for b in sp_all.bounds())
    lv = leaves(p)
    if not np.all(np.isfinite(hi - lo)) or (hi - lo).max() > 1e6:
        fin = [np.asarray(b, float) for q in lv if q.kind != "halfspace" for b in primitive_to_sp(q).bounds()]
        lo, hi = np.min(fin[0::2], axis=0), np.max(fin[1::2], axis=0)
    # a leaf's sample on the tree's boundary has the tree's SDF 0 to rounding (max / min of zero and the others);
    # 1e-3 of the size kept faces a corner piece removes by its small overshoot (the SDF of a boolean is a bound)
    eps = 1e-6 * float(np.linalg.norm(hi - lo))
    P, W = [], []
    for q in lv:
        pts, w = _leaf_samples(q, max(50, n // max(1, len(lv))), rng, lo, hi)
        if len(pts):
            on = np.abs(np.asarray(sp_all.sdf(pts))) <= eps
            P.append(pts[on])
            W.append(np.full(int(on.sum()), w))
    if not P:
        return np.zeros((0, 3)), np.zeros(0)
    return np.concatenate(P), np.concatenate(W)


def contact_grade(p: Primitive, piece_V: np.ndarray, piece_F: np.ndarray, input_V: np.ndarray, input_F: np.ndarray,
                  inside_input, tol: Optional[float] = None, n: int = 6000, seed: int = 0, op: str = "cut",
                  return_points: bool = False) -> Dict:
    """Contact / free grading of a candidate against its piece (see the section's comment):
      contact_area   the tool's surface on the input's surface (within `tol`)
      harm_area      a cut: the tool's surface inside the input's material; an add: outside the input (a face of
                     an add that ends inside the part - a fillet's walls against the part - is free)
      free_area      the rest (in invented material or in the air)
      contact_precision = contact / (contact + harm)
      contact_recall     share of the piece's surface that lies on the input's surface (its real faces) and that
                         the tool's surface reproduces (within `tol`)
    `tol` default: 3e-3 of the piece's diagonal, at least 0.02 (mm) - the input mesh's chords lie that close."""
    import igl
    from reco_engine.result_to_program import primitive_to_sp
    Vi, Fi = np.ascontiguousarray(input_V, np.float64), np.ascontiguousarray(input_F, np.int64)
    Vp, Fp = np.asarray(piece_V, float), np.asarray(piece_F, np.int64)
    diag = float(np.linalg.norm(Vp.max(axis=0) - Vp.min(axis=0)))
    tol = tol if tol is not None else max(0.02, 3e-3 * diag)
    P, W = tool_surface_samples(p, n, seed)
    out = dict(contact_area=0.0, harm_area=0.0, free_area=0.0, contact_precision=0.0, contact_recall=0.0, tol=tol)
    if len(P):
        d2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(P, np.float64), Vi, Fi)
        on = np.sqrt(np.maximum(d2, 0)) <= tol
        ins = np.asarray(inside_input(P), bool)
        harm = ~on & (ins if op == "cut" else ~ins)
        if return_points:
            out.update(points=P, weights=W, on=on, harm=harm)
        out.update(contact_area=float(W[on].sum()), harm_area=float(W[harm].sum()), free_area=float(W[~on & ~harm].sum()))
        ch = out["contact_area"] + out["harm_area"]
        out["contact_precision"] = out["contact_area"] / ch if ch > 0 else 0.0
    # the piece's real surface: its triangles on the input's surface, sampled by area
    tri = Vp[Fp]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    cen = tri.mean(axis=1)
    d2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(cen, np.float64), Vi, Fi)
    real = np.sqrt(np.maximum(d2, 0)) <= tol
    if real.any() and area[real].sum() > 0:
        rng = np.random.default_rng(seed + 1)
        idx = np.flatnonzero(real)
        pick = idx[rng.choice(len(idx), size=min(n, 4 * len(idx)), p=area[idx] / area[idx].sum())]
        a, b = rng.random(len(pick)), rng.random(len(pick))
        flip = a + b > 1
        a[flip], b[flip] = 1 - a[flip], 1 - b[flip]
        T3 = tri[pick]
        S = T3[:, 0] + a[:, None] * (T3[:, 1] - T3[:, 0]) + b[:, None] * (T3[:, 2] - T3[:, 0])
        out["contact_recall"] = float(np.mean(np.abs(np.asarray(primitive_to_sp(p).sdf(S))) <= tol))
        out["piece_contact_area"] = float(area[real].sum())
    return out


CONTACT_GRADED = 6                  # candidates per piece that get a contact grade (the best ones: it costs ~0.3 s each)


def _contact_graded(cands: List, contact: Optional[Dict], V: np.ndarray, F: np.ndarray, op: Optional[str]) -> List:
    """Contact-grade the best candidates (contact_grade: contact / free / harm areas, contact recall and precision,
    kept on the candidate as `.contact`); with contact["score"] (ResidualConfig.feature_contact_grading) the score is
    also min(score, contact recall, contact precision). No-op without `contact`."""
    if not contact or op is None:
        return cands
    for c in sorted(cands, key=lambda q: -q.score)[:CONTACT_GRADED]:
        if getattr(c, "contact", None) is not None:
            continue
        try:
            c.contact = contact_grade(c.primitive, V, F, contact["V"], contact["F"], contact["inside"], op=op)
        except Exception:                                           # a grade that cannot be computed is no verdict
            c.contact = None
            continue
        if contact.get("score"):
            c.score = min(c.score, c.contact["contact_recall"], c.contact["contact_precision"])
    return cands
