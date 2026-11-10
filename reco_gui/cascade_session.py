#!/usr/bin/env python3
"""
The Qt-free core of the cascade (add-cut) mode of the GUI: `reco_engine.residual_loop.residual_steps` yields one ResidualStep per
stage (input, member, ibeam_box, member_op, hollow, add, residual, component, candidate, accepted, rejected, solution, fallback, the
feature phase's feature_residual, feature_piece, feature_split, feature_candidate, feature_accepted, feature_rejected, feature_done, grade,
done), each carrying the
ReconstructionResult so far (reco_engine/result_schema.py). `CascadeFolder` folds them into immutable CascadeStates - the input
mesh, the model of the accepted operations, the residual voxels (material still missing / material the model has in excess),
the leftover component being fitted, the candidate and its verdict, and in the feature phase the pieces of the exact
difference with their state (`FeatureView`) - and meshes everything here, in the worker thread, so the GUI thread only draws. A StepSession built with this folder (`make_cascade_session`) has the same interface as the voxel and
the vector-field ones, so the worker, the timeline, the breakpoints and the pacing are shared.

The residual loop is imported lazily: without reco_engine/residual_loop.py the GUI still starts, only Start in this mode fails.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine.timing import Recorder, timed  # noqa: E402
from reco_gui.field_session import MESH_RESOLUTION, MeshData  # noqa: E402

# member / ibeam_box / member_op / hollow: the first-add sub-loops (reco_engine/ibeam.py): a straight member found in the
# mesh's sections (drawn as the component, yellow; its profile in the detail panel), one of its primitives (an I-beam's
# three boxes are "ibeam_box"), a hollow body's outer / cavity shell
# feature_*: the feature phase (after the rough model): the exact difference between the model and the part, split into
# pieces; per piece a guessed primitive, scored, applied as a cut / an add and validated by a real FreeCAD build
# feature_split: a piece no single primitive explains is split into segments on its voxel field; every segment then
# runs through feature_piece ... feature_accepted / feature_rejected like a piece (its dict has "parent")
# solution: the choice between the first solution (a sub-loop's) and the classic one, before the fallback
# (reco_engine/solution_choice.py); the classic solution's own steps come before it with info["solution"] = "classic"
# and carry the classic program, the "solution" step carries the program that was kept
FEATURE_STAGES = ("feature_residual", "feature_piece", "feature_split", "feature_candidate", "feature_accepted",
                  "feature_rejected", "feature_done")
CASCADE_STAGES = ("input", "member", "ibeam_box", "member_op", "hollow", "add", "residual", "component", "candidate",
                  "accepted", "rejected", "solution", "fallback") + FEATURE_STAGES + (
                      "feature_confirm", "feature_confirmed", "grade", "done")
# the stages the pace is waited after (StepSession.run's `paced`): every stage but the candidates, which come by the dozen
UNPACED_STAGES = ("candidate", "feature_candidate")
CASCADE_PACED_STAGES = tuple(s for s in CASCADE_STAGES if s not in UNPACED_STAGES)
CASCADE_DEFAULT_BREAKPOINTS = ()        # none checked at start (the user's rule 2026-10-15); Step stops at every stage anyway
MEMBER_STAGES = ("member", "ibeam_box", "member_op", "hollow")
DEFAULT_REGISTRY = ROOT / "models" / "registry.json"

# info keys the residual loop may use for the two residual masks (world x, y, z axes, on the grid of info["spec"])
POS_KEYS = ("residual_pos", "pos", "pos_mask")
NEG_KEYS = ("residual_neg", "neg", "neg_mask")
ALIASES = {"primitive": "candidate", "mask": "component_mask", "gain": "net_gain"}     # residual_loop's names -> the ones used here


@dataclass(frozen=True)
class CascadeStep:
    """What the timeline, the log and the breakpoints read (the same fields as the other modes' steps)."""
    stage: str
    phase: str
    iteration: Optional[int]
    message: str
    raw: object = None                          # the ResidualStep


# "guessed": the piece has a guess that waits for its FreeCAD build (the loop validates an iteration's guesses together)
# "split": the piece was split into segments and waits for their verdicts (it is drawn as its segments)
PIECE_STATES = ("pending", "current", "guessed", "split", "accepted", "rejected", "skipped")
# segments of a split piece: a qualitative palette per segment, the state as the opacity (and as the outline's colour)
SEGMENT_PALETTE = ((0.122, 0.467, 0.706), (1.000, 0.498, 0.055), (0.173, 0.627, 0.173), (0.839, 0.153, 0.157),
                   (0.580, 0.404, 0.741), (0.549, 0.337, 0.294), (0.890, 0.467, 0.761), (0.737, 0.741, 0.133),
                   (0.090, 0.745, 0.812), (0.498, 0.498, 0.498))
SEGMENT_OPACITY = {"pending": 0.30, "current": 0.95, "guessed": 0.55, "split": 0.30, "accepted": 0.75, "rejected": 0.15,
                   "skipped": 0.10}
STATE_COLOURS = {"pending": (0.55, 0.55, 0.58), "current": (1.0, 0.8, 0.0), "guessed": (0.4, 0.7, 0.95),
                 "split": (0.55, 0.55, 0.58), "accepted": (0.106, 0.686, 0.478), "rejected": (0.95, 0.52, 0.10),
                 "skipped": (0.30, 0.30, 0.33)}


@dataclass(frozen=True)
class FeaturePiece:
    """One piece of the exact difference (a possible feature) and what became of it."""
    id: int
    side: str = ""                              # "missing" (the part has it, the model lacks it) | "invented"
    op: str = ""                                # the operation that would explain it: "add" | "cut"
    volume: float = 0.0                         # mm^3
    share: float = 0.0                          # of the difference's volume
    bbox: Optional[Tuple[np.ndarray, np.ndarray]] = None   # world mm
    stl: Optional[str] = None
    state: str = "pending"                      # one of PIECE_STATES
    mesh: Optional[MeshData] = None             # its real mesh, from its "feature_piece" step (else its box is drawn)
    mesh_truncated: bool = False
    candidates: int = 0                         # guesses tried so far
    primitive: object = None                    # the last guess
    source: Optional[str] = None
    score: Optional[float] = None
    reason: str = ""                            # of a rejection
    validation: Dict[str, object] = None        # the FreeCAD build of the last verdict
    precision: Optional[float] = None           # of the last guess against the piece (segments: score = min of both)
    recall: Optional[float] = None
    parent: Optional[int] = None                # a segment: the id of the piece it was split from
    voxels: Optional[int] = None                # a segment: its voxels on the split's grid
    split: Dict[str, object] = None             # a split piece: method, pitch, grid, segments (ids), seconds
    segments_accepted: Tuple[int, ...] = ()     # a split piece's verdict: the segments that were applied


@dataclass(frozen=True)
class ColouredMesh(MeshData):
    """A mesh with a colour per triangle (the segments of a split piece)."""
    colours: np.ndarray = None                  # (t, 4) uint8: red, green, blue, opacity


@dataclass(frozen=True)
class FeatureView:
    """The feature phase as the viewer and the report need it: the pieces of the current iteration with their state, the
    volumes and the exact IoU per iteration, the limits in force and, at the end, the summary."""
    iteration: int = 0                          # from 1
    pieces: Tuple[FeaturePiece, ...] = ()
    current: Optional[int] = None               # id of the piece being worked on
    index: Optional[int] = None                 # ... it is piece `index` of `of` (1-based)
    of: Optional[int] = None
    missing_volume: Optional[float] = None      # mm^3, of this iteration's difference
    invented_volume: Optional[float] = None
    input_volume: Optional[float] = None
    exact_iou: Optional[float] = None           # the latest known (the iteration's, then each validated build's)
    dropped_small: int = 0
    dropped_limit: int = 0
    limits: Dict[str, object] = None
    history: Tuple[Dict[str, object], ...] = ()  # per iteration: iteration, volumes, exact_iou, pieces, applied, rejected
    applied: int = 0                            # counted here, over the whole phase
    rejected: int = 0
    done: Dict[str, object] = None              # the "feature_done" step's summary (stopped, seconds, ...)
    layers: Dict[str, MeshData] = None          # what the viewer draws: PIECE_STATES -> the pieces in that state, merged;
                                                # "segments" / "segment_outline": the segments of the split pieces (coloured)

    def segments(self, piece_id) -> Tuple[FeaturePiece, ...]:
        return tuple(p for p in self.pieces if p.parent == piece_id)

    def piece(self, piece_id) -> Optional[FeaturePiece]:
        return next((p for p in self.pieces if p.id == piece_id), None)

    def counts(self) -> Dict[str, int]:
        return {state: sum(1 for p in self.pieces if p.state == state) for state in PIECE_STATES}


@dataclass(frozen=True)
class CascadeState:
    index: int
    step: CascadeStep
    input_path: str = ""
    input_mesh: Optional[MeshData] = None
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
    result: object = None                       # the ReconstructionResult after this step
    program: Tuple[object, ...] = ()            # the accepted Operations
    model_mesh: Optional[MeshData] = None       # union/subtraction of the accepted analytic primitives
    fallback_mesh: Optional[MeshData] = None    # the accepted voxel-cube fallbacks (purple)
    residual_pos_mesh: Optional[MeshData] = None
    residual_neg_mesh: Optional[MeshData] = None
    residual_pos: int = 0                       # voxels: material still missing
    residual_neg: int = 0                       # voxels: material the model has in excess
    input_voxel_mesh: Optional[MeshData] = None  # the input's voxelisation on the loop's grid (from the input step)
    field_arrows: Optional[Dict[str, np.ndarray]] = None   # sample of the closest-point field: points, vectors, sdf (mm)
    cut_once_mesh: Optional[MeshData] = None    # the cut history (residual_loop): voxels emptied by exactly one cut
    recut_mesh: Optional[MeshData] = None       # ... and by two or more cuts (cut again)
    recut_voxels: int = 0
    component_mesh: Optional[MeshData] = None
    component_voxels: int = 0
    candidate: object = None                    # the Primitive being tried
    candidate_op: Optional[str] = None          # "add" | "cut"
    candidate_mesh: Optional[MeshData] = None
    candidate_fallback: bool = False            # the candidate is a voxel-cube fallback
    member: object = None                       # the ibeam.Member being built (member sub-loop), shown as the component
    member_info: Dict[str, object] = None       # the "member" step's info (kind, profile, extent, I-section dims)
    verdict: Optional[str] = None               # "accepted" | "rejected" | None
    net_gain: Optional[float] = None
    finished: bool = False
    info: Dict[str, object] = None
    type_stage: Dict[str, object] = None        # the type stage on the whole input (input step: cascade.TypeStageResult.as_dict():
                                                # parent, subclass, subclass_source, mode, probs, entries with weights)
    wall_grid: Dict[str, object] = None         # the input's walls against the loop's pitch (`wall_grid_voxels`): tau, pitches, ...
    tau_voxel_mesh: Optional[MeshData] = None   # the input's parity voxels at pitch tau over its own box (display only)
    timings: Tuple[object, ...] = ()            # reco_engine.timing sections of this step: the loop's, then "gui fold"
    feature: Optional[FeatureView] = None       # the feature phase (kept after it ended, for its summary)

    @property
    def in_feature_phase(self) -> bool:
        return self.feature is not None and self.step.stage in FEATURE_STAGES


# ------------------------------------------------------------------------------------------------ geometry

def quaternion_matrix(q) -> np.ndarray:
    """(qx, qy, qz, qw) -> 3x3 rotation matrix (columns: the local axes in the world)."""
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _dim(d: Dict[str, float], *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return float(d[n])
    if default is None:
        raise KeyError(f"none of {names} in {sorted(d)}")
    return default


def primitive_sdf(p):
    """A result_schema.Primitive as a placed sdf_primitives solid (centre = the primitive's centre, also for the z-based
    cylinder/cone of sdf_primitives). None for an unknown kind. Uses reco_engine.result_to_program.primitive_to_sp (the
    converter the builder uses) when it is there, so the GUI draws what FreeCAD builds; the local copy below is its fallback."""
    try:
        from reco_engine.result_to_program import primitive_to_sp
    except ImportError:
        primitive_to_sp = None
    if primitive_to_sp is not None:
        try:
            return primitive_to_sp(p)
        except (ValueError, KeyError):
            return None
    from src import sdf_primitives as sp
    d = dict(p.dims_mm or {})
    u = tuple(p.u) if p.u else sp.FULL
    shift = np.zeros(3)
    if p.kind == "box":
        solid = sp.Box((_dim(d, "x_half", "hx"), _dim(d, "y_half", "hy"), _dim(d, "z_half", "hz")))
    elif p.kind == "sphere":
        solid = sp.Sphere(_dim(d, "radius"), u=u, v=tuple(p.v) if p.v else (-np.pi / 2, np.pi / 2))
    elif p.kind == "cylinder":
        h = _dim(d, "half_length", "half_height")
        solid, shift = sp.Cylinder(_dim(d, "radius"), 2 * h, u=u), np.array([0.0, 0.0, -h])
    elif p.kind == "cone":
        h = _dim(d, "half_length", "half_height")
        solid, shift = sp.Cone(_dim(d, "radius1"), _dim(d, "radius2"), 2 * h, u=u), np.array([0.0, 0.0, -h])
    elif p.kind == "torus":
        solid = sp.Torus(_dim(d, "major_radius", "major"), _dim(d, "minor_radius", "minor"), u=u,
                         v=tuple(p.v) if p.v else sp.FULL)
    else:
        return None
    R = quaternion_matrix(p.quaternion)
    return sp.Placed(solid, rotation=R, translation=np.asarray(p.center_mm, float) + R @ shift)


def is_fallback(p) -> bool:
    return p is not None and any(getattr(s, "mode", "") == "voxel_fallback" for s in (p.sources or ()))


def _spec(spec):
    """(lo, hi) of a grid spec given as a dict or an object with lo/hi (resolution comes from the mask)."""
    if spec is None:
        return None
    get = spec.get if isinstance(spec, dict) else (lambda k, default=None: getattr(spec, k, default))
    lo, hi = get("lo"), get("hi")
    if lo is None or hi is None:
        return None
    return np.asarray(lo, float), np.asarray(hi, float)


def voxel_mesh(mask: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> Optional[MeshData]:
    """The exposed faces of the filled voxels of `mask` (world x, y, z axes) on the grid [lo, hi], as triangles."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 3 or not mask.any():
        return None
    size = (np.asarray(hi, float) - np.asarray(lo, float)) / np.asarray(mask.shape, float)
    padded = np.pad(mask, 1)
    verts, tris = [], []
    corners = {                                       # the four corners (in voxel units) of the face towards +/- axis
        0: np.array([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], float),
        1: np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]], float),
        2: np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float),
    }
    n = 0
    for axis in range(3):
        for sign in (-1, 1):
            neighbour = np.roll(padded, -sign, axis=axis)[1:-1, 1:-1, 1:-1]
            idx = np.argwhere(mask & ~neighbour).astype(float)
            if not len(idx):
                continue
            offset = np.zeros(3)
            offset[axis] = 1.0 if sign > 0 else 0.0
            quad = corners[axis] + offset                                        # (4, 3)
            v = (idx[:, None, :] + quad[None, :, :]).reshape(-1, 3)
            k = np.arange(len(idx)) * 4 + n
            t = np.concatenate([np.stack([k, k + 1, k + 2], 1), np.stack([k, k + 2, k + 3], 1)])
            verts.append(v)
            tris.append(t)
            n += len(v)
    return MeshData(np.asarray(lo, float) + np.vstack(verts) * size, np.vstack(tris).astype(np.int64))


def _mask(info: dict, keys) -> Optional[np.ndarray]:
    for k in keys:
        if info.get(k) is not None:
            return np.asarray(info[k], bool)
    return None


def wall_grid_voxels(vertices: np.ndarray, faces: np.ndarray, fraction: float = 1.0, max_voxels: int = 40_000_000,
                     tau: Optional[np.ndarray] = None):
    """(mesh of the parity voxels, info) of the input on a grid whose pitch is licensed by its walls (D-005): pitch =
    `fraction` x the 5th percentile of the wall thickness tau (dataset_gen/src/local_grids.py), over the part's own box
    (any aspect ratio, not the loop's cube). On the frame tau/1 already gives the exact topology (docs/FRAME_FAILURE.md,
    "Thickness-adaptive local grids"). Coarsened (info["capped"]) when the box would exceed `max_voxels`. Display only:
    the loop keeps its own grid. `tau`: the wall-thickness samples the loop already measured (its input step's
    info["wall_tau"], grid "tau") - the rays are cast only once per run (they took 32 s of the hollow sphere's 87 s)."""
    import time
    from src import local_grids
    t0 = time.time()
    V, F = np.asarray(vertices, float), np.asarray(faces)
    reused = tau is not None
    if not reused:
        from reco_engine.cache import wall_thickness_cached
        with timed("wall thickness"):
            tau = wall_thickness_cached(V, F)                           # once per mesh, also across runs
    tau = np.asarray(tau, float)
    finite = tau[np.isfinite(tau) & (tau > 0)]
    if not len(finite):
        return None, None
    licensed = local_grids.adaptive_pitch(tau, fraction=fraction)
    lo, hi = V.min(axis=0), V.max(axis=0)
    pitch = licensed
    while np.prod(np.ceil((hi - lo) / pitch) + 4) > max_voxels:
        pitch *= 1.25
    grid = local_grids.box_grid(lo, hi, pitch)
    occ = np.concatenate(list(local_grids.iter_slabs(V, F, grid)), axis=1)
    edge_lo = np.asarray(grid.lo, float)
    mesh = voxel_mesh(occ, edge_lo, edge_lo + np.asarray(grid.shape, float) * pitch)
    tri = V[F]
    volume = abs(float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)
    info = dict(tau_p5=float(np.percentile(finite, 5)), tau_median=float(np.median(finite)), pitch=float(pitch),
                licensed_pitch=float(licensed), capped=pitch > licensed * 1.0001, shape=tuple(int(n) for n in grid.shape),
                voxels=int(occ.sum()), volume_ratio=float(occ.sum() * pitch ** 3 / volume) if volume > 0 else None,
                seconds=time.time() - t0, tau_reused=reused)
    return mesh, info


def grid_warning(wall_grid: Optional[dict]) -> str:
    """'' when the loop's pitch resolves the thin walls, else one line saying by how much it does not."""
    if not wall_grid or wall_grid.get("loop_pitch") is None:
        return ""
    loop, tau = wall_grid["loop_pitch"], wall_grid["tau_p5"]
    if loop <= tau:
        return ""
    return (f"WARNING: the loop's voxel pitch {loop:.4g} is {loop / tau:.1f} x the thin walls (tau {tau:.4g}): its voxels "
            f"alias them (D-005) - compare the 'Voxels at tau' layer")


# ------------------------------------------------------------------------------------------------ feature phase

_BOX_QUADS = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]


def bbox_mesh(lo, hi) -> Optional[MeshData]:
    """The axis-aligned box [lo, hi] as 12 triangles (a piece that has no mesh yet is drawn as its box)."""
    lo, hi = np.asarray(lo, float).reshape(3), np.asarray(hi, float).reshape(3)
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        return None
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    faces = np.array([t for a, b, c, e in _BOX_QUADS for t in ((a, b, c), (a, c, e))], np.int64)
    return MeshData(lo + corners * (hi - lo), faces)


def merge_meshes(meshes: Sequence[MeshData]) -> Optional[MeshData]:
    meshes = [m for m in meshes if m is not None and len(m.triangles)]
    if not meshes:
        return None
    if len(meshes) == 1:
        return meshes[0]
    offsets = np.cumsum([0] + [len(m.vertices) for m in meshes[:-1]])
    return MeshData(np.vstack([m.vertices for m in meshes]),
                    np.vstack([m.triangles + o for m, o in zip(meshes, offsets)]))


def _number(value, default=None):
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _whole(value, default=None):
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _piece_mesh(mesh) -> Optional[MeshData]:
    """info["mesh"] of a "feature_piece" step (dict(vertices (n, 3), faces (m, 3))) as a MeshData; None when unusable."""
    if not isinstance(mesh, dict):
        return None
    try:
        v = np.asarray(mesh.get("vertices"), float).reshape(-1, 3)
        f = np.asarray(mesh.get("faces") if mesh.get("faces") is not None else mesh.get("triangles"), np.int64).reshape(-1, 3)
    except (TypeError, ValueError):
        return None
    if not len(v) or not len(f) or f.min() < 0 or f.max() >= len(v):
        return None
    return MeshData(v, f)


def _piece(d, fallback_id: int) -> FeaturePiece:
    """A piece dict of the step contract (unknown keys ignored, missing ones defaulted)."""
    d = d if isinstance(d, dict) else {}
    box = None
    try:
        lo, hi = d["bbox"]
        box = (np.asarray(lo, float).reshape(3), np.asarray(hi, float).reshape(3))
    except (KeyError, TypeError, ValueError):
        pass
    side = str(d.get("side") or "")
    return FeaturePiece(id=_whole(d.get("id"), fallback_id), side=side,
                        op=str(d.get("op") or {"missing": "add", "invented": "cut"}.get(side, "")),
                        volume=_number(d.get("volume"), 0.0), share=_number(d.get("share"), 0.0), bbox=box,
                        stl=str(d["stl"]) if d.get("stl") else None, parent=_whole(d.get("parent")),
                        voxels=_whole(d.get("voxels")))


def feature_progress(state) -> str:
    """'feature phase: piece 7 of 40, iteration 2' while the state is in the feature phase, else ''."""
    f = getattr(state, "feature", None)
    if f is None or f.done is not None or getattr(state.step, "stage", "") not in FEATURE_STAGES:
        return ""
    current = f.piece(f.current) if f.current is not None else None
    if current is not None and current.parent is not None and f.index is not None:
        piece = f"piece #{current.parent}, segment {f.index}" + (f" of {f.of}" if f.of else "")
    elif f.index is not None:
        piece = f"piece {f.index}" + (f" of {f.of}" if f.of else "")
    else:
        piece = f"{len(f.pieces)} piece(s)"
    text = f"feature phase: {piece}, iteration {f.iteration}"
    waiting = sum(1 for p in f.pieces if p.state == "guessed" or (p.state == "current" and p.candidates))
    if waiting and f.index is not None and f.index == f.of and state.step.stage != "feature_piece":
        text += f", validating {waiting} guess(es)"            # every piece was looked at: the FreeCAD builds run
    return text


def _feature_summary(kind: str, info: dict, candidate) -> str:
    """The line of a feature step that came without info["message"]."""
    pid = info.get("piece_id")
    what = f"{info.get('op') or ''}:{candidate.kind}" if candidate is not None else ""
    score = f", score {_number(info.get('score')):.3f}" if _number(info.get("score")) is not None else ""
    source = f" from {info['source']}" if info.get("source") else ""
    if kind == "feature_residual":
        iou = f", exact IoU {_number(info.get('exact_iou')):.4f}" if _number(info.get("exact_iou")) is not None else ""
        return (f"feature iteration {info.get('iteration', '?')}: {len(info.get('pieces') or ())} piece(s), missing "
                f"{_number(info.get('missing_volume'), 0.0):.4g}, invented {_number(info.get('invented_volume'), 0.0):.4g} mm^3{iou}")
    if kind == "feature_piece":
        p = info.get("piece") if isinstance(info.get("piece"), dict) else {}
        return (f"piece {p.get('id', '?')} ({info.get('index', '?')} of {info.get('of', '?')}): {p.get('side', '')}, "
                f"{_number(p.get('volume'), 0.0):.4g} mm^3 ({_number(p.get('share'), 0.0):.1%})")
    if kind == "feature_split":
        return (f"piece {pid} split by {info.get('method', '?')} into {len(info.get('segments') or ())} segment(s)"
                + (f" (pitch {_number(info.get('pitch')):.4g} mm)" if _number(info.get("pitch")) is not None else ""))
    if kind == "feature_accepted" and candidate is None:
        return f"piece {pid}: {len(info.get('segments_accepted') or ())} segment(s) accepted"
    if kind == "feature_candidate":
        return f"piece {pid}: candidate {what}{source}{score}"
    if kind == "feature_accepted":
        return f"piece {pid}: accepted {what}{source}{score}"
    if kind == "feature_rejected":
        return f"piece {pid}: rejected {what}{source}{score}" + (f" ({info['reason']})" if info.get("reason") else "")
    if kind == "feature_done":
        return (f"feature phase done: {info.get('applied', 0)} applied, {info.get('rejected', 0)} rejected, "
                f"{info.get('skipped', 0)} skipped; stopped: {info.get('stopped', '')}")
    return kind


# ------------------------------------------------------------------------------------------------ folding

def _summary(kind: str, result, info: dict, candidate) -> str:
    if info.get("message"):
        return str(info["message"])
    if kind in FEATURE_STAGES:
        return _feature_summary(kind, info, candidate)
    pos, neg = getattr(result, "residual_voxels_pos", 0), getattr(result, "residual_voxels_neg", 0)
    recut = getattr(result, "recut_voxels", 0) or 0
    n = sum(1 for o in getattr(result, "operations", ()) if o.accepted)
    what = f"{info.get('op', '')}:{candidate.kind}" if candidate is not None else ""
    gain = f", net gain {info['net_gain']:.0f} voxels" if info.get("net_gain") is not None else ""
    comp = info.get("component_mask")
    ts = info.get("type_stage") or {}
    type_text = (f", type {ts.get('parent')}" + (f"/{ts['subclass']}" if ts.get("subclass") else "")
                 + f" ({ts.get('mode', '')})") if ts.get("parent") else ""
    return {
        "input": f"input {Path(getattr(result, 'input_path', '')).name}{type_text}",
        "member": f"member: {info.get('kind', '')} {info.get('profile', '')}",
        "ibeam_box": f"I-beam box {getattr(candidate, 'label', None) or what}{gain}",
        "member_op": f"member primitive {getattr(candidate, 'label', None) or what}{gain}",
        "hollow": f"hollow body: {info.get('shell', '')} shell",
        "add": f"first addition {what}",
        "residual": f"residual: {pos} voxels missing, {neg} in excess ({n} operation(s))"
                    + (f", {recut} voxels cut again" if recut else ""),
        "decision": f"decision {info.get('branch', '')}: {info.get('reason', '')}",          # the tree (residual_tree.py)
        "repair": f"repair: {info.get('op', '')} {what} {info.get('reason', '')}",             # reco_engine/repair.py
        "component": f"leftover component of {int(np.count_nonzero(comp)) if comp is not None else '?'} voxels",
        "candidate": f"candidate {what}{gain}",
        "accepted": f"accepted {what}{gain}",
        "rejected": f"rejected {what}{gain}",
        "solution": f"solution kept: {info.get('kept', '')}",
        "fallback": f"voxel-cube fallback {what}{gain}",
        "grade": f"grade: chamfer {result.chamfer_mm:.4g} mm" if getattr(result, "chamfer_mm", None) is not None else "grade",
        "done": f"done: {n} operation(s), residual +{pos} / -{neg} voxels",
    }.get(kind, kind)


class CascadeFolder:
    """`fold(previous, residual_step, index)` -> CascadeState. Primitives and models are meshed once (cached)."""

    def __init__(self, resolution: int = MESH_RESOLUTION, wall_grid: bool = False, voxel_meshes: bool = False):
        self.resolution = resolution
        # the user's rule 2026-10-15: no voxels on screen - the surface distance field with its inside/outside colouring
        # is the picture of the input. `voxel_meshes` builds the cube meshes of the input, the residual, the cut history
        # and the component for the debug layers (they cost a marching pass per step); `wall_grid` the tau-pitch voxels
        # of the input (a parity grid over the part's box, seconds on a big part). Both off unless asked for.
        self.wall_grid = wall_grid                  # compute the tau-pitch voxels of the input (the 'Voxels at tau' layer)
        self.voxel_meshes = voxel_meshes
        self._cache: Dict[object, Optional[MeshData]] = {}
        self._crop = None
        self._iteration = -1
        self._box = None
        self._feature_iterations: Dict[tuple, int] = {}   # (feature iteration, piece id) -> its timeline iteration
        self._feature_layers: Dict[str, tuple] = {}  # piece state -> (key, merged mesh): unchanged layers are reused
        # exact meshes by mesh booleans (reco_gui/model_mesh.py) when manifold3d is installed; else the SDF path below,
        # which loses walls thinner than a grid cell (the frame's model came out empty from the 4th operation on)
        from reco_gui import model_mesh
        self._exact = model_mesh.available()
        if self._exact:
            self._meshers = {flag: model_mesh.ModelMesher(sdf_mesh=self._primitive_sdf_mesh, union_all=flag,
                                                          crop=lambda: self._crop)
                             for flag in (False, True)}

    def _primitive_sdf_mesh(self, p) -> Optional[MeshData]:
        """A primitive without an analytic solid (torus, a u/v sweep), from its own SDF on a grid that fits it."""
        try:
            from reco_field import peel
            prim = primitive_sdf(p)
            if prim is None:
                return None
            v, f = peel.mesh_from_sdf(prim, self.resolution)
            return MeshData(v, f) if len(f) else None
        except Exception:
            return None

    def _mesh_sdf(self, key, build) -> Optional[MeshData]:
        if key not in self._cache:
            try:
                prim = build()
                if prim is None:
                    self._cache[key] = None
                else:
                    from reco_field import peel
                    from src import sdf_primitives as sp
                    if self._crop is not None:                          # a cutter reaching far out of the part: crop it
                        lo, hi = self._crop
                        prim = sp.Intersection(prim, sp.Placed(sp.Box((hi - lo) / 2), translation=(lo + hi) / 2))
                    v, f = peel.mesh_from_sdf(prim, self.resolution)
                    self._cache[key] = MeshData(v, f) if len(f) else None
            except Exception:                                           # an empty or degenerate level set: nothing to draw
                self._cache[key] = None
        return self._cache[key]

    @staticmethod
    def _prim_key(p):
        """Everything that shapes the primitive - a composite's combine and parts too (box & cylinder and box - cylinder
        share their pose and dims)."""
        return (p.kind, tuple(np.round(p.quaternion, 6)), tuple(np.round(p.center_mm, 6)),
                tuple(sorted((k, round(float(v), 6)) for k, v in (p.dims_mm or {}).items())),
                tuple(p.u) if p.u else None, tuple(p.v) if p.v else None,
                getattr(p, "combine", None), tuple(CascadeFolder._prim_key(q) for q in (getattr(p, "parts", None) or ())))

    def primitive_mesh(self, p) -> Optional[MeshData]:
        if p is None:
            return None
        if self._exact:
            key = ("exact prim",) + self._prim_key(p)
            if key not in self._cache:
                from reco_gui import model_mesh
                solid = model_mesh.primitive_solid(p, self._primitive_sdf_mesh, crop=self._crop)
                if solid is not None and self._crop is not None:        # a cutter reaching far out of the part: crop it
                    import manifold3d as m
                    lo, hi = self._crop
                    solid = solid ^ m.Manifold.cube(tuple(hi - lo)).translate(tuple(lo))
                self._cache[key] = model_mesh.to_mesh_data(solid) if solid is not None else \
                    self._mesh_sdf(("prim",) + self._prim_key(p), lambda: primitive_sdf(p))
            return self._cache[key]
        return self._mesh_sdf(("prim",) + self._prim_key(p), lambda: primitive_sdf(p))

    def model_mesh(self, ops, fallback: bool) -> Optional[MeshData]:
        """The accepted analytic operations (fallback False) or the accepted voxel-cube fallbacks (True) as one mesh."""
        chosen = [o for o in ops if is_fallback(o.primitive) == fallback]
        if not chosen:
            return None
        key = ("model", fallback) + tuple((o.op,) + self._prim_key(o.primitive) for o in chosen)
        if self._exact:
            if key not in self._cache:
                self._cache[key] = self._meshers[fallback].mesh(chosen, self._prim_key)
            return self._cache[key]

        def build():
            model = None
            for o in chosen:
                s = primitive_sdf(o.primitive)
                if s is None:
                    continue
                if model is None:
                    model = s if (o.op == "add" or fallback) else None
                elif o.op == "add" or fallback:
                    model = model | s
                else:
                    model = model - s
            return model
        return self._mesh_sdf(key, build)

    def __call__(self, prev: Optional[CascadeState], rstep, index: int) -> CascadeState:
        kind = rstep.kind
        info = dict(rstep.info or {})
        for theirs, ours in ALIASES.items():
            if info.get(ours) is None and info.get(theirs) is not None:
                info[ours] = info[theirs]
        if info.get("spec") is not None:                                     # the grid comes with the input step only
            self._box = _spec(info["spec"])
        result = rstep.result
        candidate = info.get("candidate")
        op = info.get("op") or (("cut" if kind == "cut" else "add") if candidate is not None else None)
        if kind in ("component", "add", "fallback", "member", "hollow"):     # one timeline "cand." number per component
            self._iteration += 1
        feature_piece_step = kind in FEATURE_STAGES and kind not in ("feature_residual", "feature_done")
        if feature_piece_step:                                               # ... and one per feature piece
            piece_info = info.get("piece") if isinstance(info.get("piece"), dict) else {}
            key = (info.get("iteration"), info.get("piece_id", piece_info.get("id")))
            if key not in self._feature_iterations:                          # a verdict may come long after the guess
                self._iteration += 1                                         # (validated together): its piece's number
                self._feature_iterations[key] = self._iteration
        phase = op or ("member" if kind in MEMBER_STAGES else "")
        if kind in FEATURE_STAGES:
            phase = str(info.get("phase") or "feature")
        in_iteration = (kind in ("add", "component", "candidate", "accepted", "rejected", "fallback") + MEMBER_STAGES
                        or feature_piece_step) and self._iteration >= 0
        iteration = self._feature_iterations[key] if feature_piece_step else self._iteration
        step = CascadeStep(kind, phase, iteration if in_iteration else None,
                           _summary(kind, result, info, candidate), rstep)
        s = (replace(prev, index=index, step=step, info=info) if prev is not None
             else CascadeState(index=index, step=step, info=info))

        path = getattr(result, "input_path", "") if result is not None else ""
        if not s.input_path and path:                                          # load the input once, at its first mention
            mesh, bounds = None, None
            try:
                from src.closest_point_field import load_mesh
                v, f = load_mesh(path, 0.02)
                mesh = MeshData(np.asarray(v, float), np.asarray(f, np.int64))
                bounds = (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
                pad = 0.08 * float((bounds[1] - bounds[0]).max())
                self._crop = (bounds[0] - pad, bounds[1] + pad)
            except Exception:
                mesh = None
            s = replace(s, input_path=str(path), input_mesh=mesh, bounds=bounds)

        program = tuple(o for o in getattr(result, "operations", ()) if o.accepted) if result is not None else s.program
        with timed("mesh model"):
            model_mesh, fallback_mesh = self.model_mesh(program, False), self.model_mesh(program, True)
        s = replace(s, result=result if result is not None else s.result, program=program,
                    model_mesh=model_mesh, fallback_mesh=fallback_mesh,
                    residual_pos=int(getattr(result, "residual_voxels_pos", s.residual_pos) or 0),
                    residual_neg=int(getattr(result, "residual_voxels_neg", s.residual_neg) or 0))

        box = _spec(info.get("spec")) or self._box
        pos, neg = _mask(info, POS_KEYS), _mask(info, NEG_KEYS)
        cubes = self.voxel_meshes                                          # the cube meshes only when asked for
        if box is not None and (pos is not None or neg is not None):
            s = replace(s, residual_pos_mesh=voxel_mesh(pos, *box) if pos is not None and cubes else None,
                        residual_neg_mesh=voxel_mesh(neg, *box) if neg is not None and cubes else None)
            if pos is not None and kind == "residual" and result is None:
                s = replace(s, residual_pos=int(pos.sum()))
            if neg is not None and kind == "residual" and result is None:
                s = replace(s, residual_neg=int(neg.sum()))
        if kind == "input" and box is not None:                              # voxelisation + field: once, from the input step
            occ = info.get("occ_in")
            s = replace(s, input_voxel_mesh=voxel_mesh(occ, *box) if occ is not None and cubes else None,
                        field_arrows=info.get("field_arrows"))
        if kind == "input":
            s = replace(s, type_stage=info.get("type_stage") or None)
            if self.wall_grid and s.input_mesh is not None:
                try:
                    with timed("tau voxels"):
                        mesh, wall = wall_grid_voxels(s.input_mesh.vertices, s.input_mesh.triangles,
                                                      tau=info.get("wall_tau"))
                except Exception as exc:                                    # advisory: the run goes on without it
                    mesh, wall = None, dict(error=str(exc))
                occ = info.get("occ_in")
                if wall is not None and box is not None and occ is not None:
                    wall["loop_pitch"] = float((box[1] - box[0]).max() / max(np.asarray(occ).shape))
                warning = grid_warning(wall)
                if warning:
                    s = replace(s, step=replace(s.step, message=f"{s.step.message} | {warning}"))
                s = replace(s, wall_grid=wall, tau_voxel_mesh=mesh)
        history = info.get("cut_history")
        if history is not None and box is not None:
            history = np.asarray(history)
            s = replace(s, cut_once_mesh=voxel_mesh(history == 1, *box) if cubes else None,
                        recut_mesh=voxel_mesh(history >= 2, *box) if cubes else None,
                        recut_voxels=int(np.count_nonzero(history >= 2)))
        if kind == "member":
            s = replace(s, member=info.get("member"), member_info=info)
        elif kind in ("residual", "done", "input", "component", "hollow", "solution"):
            s = replace(s, member=None, member_info=None)
        comp = info.get("component_mask")
        if comp is not None and box is not None:
            s = replace(s, component_mesh=voxel_mesh(comp, *box) if cubes else None, component_voxels=int(np.count_nonzero(comp)))
        elif info.get("member") is not None and kind in ("accepted", "rejected"):
            pass                                                              # the member stays yellow while its parts go in
        elif kind in ("residual", "done", "grade", "input", "accepted", "fallback", "add", "feature_residual", "solution"):
            s = replace(s, component_mesh=None, component_voxels=0)

        gain = info.get("net_gain")
        if candidate is not None:
            verdict = {"accepted": "accepted", "rejected": "rejected", "add": "accepted", "fallback": "accepted",
                       "feature_accepted": "accepted", "feature_rejected": "rejected"}.get(kind)
            s = replace(s, candidate=candidate, candidate_op=op, candidate_mesh=self.primitive_mesh(candidate),
                        candidate_fallback=kind == "fallback" or is_fallback(candidate), verdict=verdict,
                        net_gain=float(gain) if gain is not None else None)
        elif kind == "rejected" and s.candidate is not None:                   # no acceptable candidate for the component
            s = replace(s, verdict="rejected")
        elif kind == "feature_rejected" and s.candidate is not None and prev is not None \
                and prev.step.stage == "feature_candidate" and (prev.info or {}).get("piece_id") == info.get("piece_id"):
            s = replace(s, verdict="rejected")                                 # the piece's last guess stays in view
        elif kind in ("residual", "component", "grade", "done", "input", "member", "hollow", "solution", "feature_residual",
                      "feature_piece", "feature_split", "feature_accepted", "feature_rejected", "feature_done",
                      "feature_confirm", "feature_confirmed"):
            s = replace(s, candidate=None, candidate_op=None, candidate_mesh=None, candidate_fallback=False, verdict=None,
                        net_gain=None)
        if kind in ("accepted", "add", "feature_accepted"):                   # now part of the model: drawn blue, not orange
            s = replace(s, candidate_mesh=None)
        if kind in FEATURE_STAGES:
            with timed("feature pieces"):
                s = replace(s, feature=self._fold_feature(s.feature, kind, info, candidate))
            info.pop("mesh", None)                                            # kept as the piece's MeshData
        if kind == "done":
            s = replace(s, finished=True, component_mesh=None)
        return replace(s, info=_light(info))

    # ---------------------------------------------------------------------------------------- feature phase
    def _fold_feature(self, f: Optional[FeatureView], kind: str, info: dict, candidate) -> FeatureView:
        """The FeatureView after one feature step. Tolerant: a step about a piece the iteration did not list adds it, a
        missing key keeps what is known."""
        f = f or FeatureView()
        if kind == "feature_residual":
            history = f.history
            if f.pieces or f.iteration:                                       # close the iteration before
                history = self._close_iteration(f)
            pieces = tuple(_piece(d, i + 1) for i, d in enumerate(info.get("pieces") or ()))
            f = replace(f, iteration=_whole(info.get("iteration"), f.iteration + 1), pieces=pieces, current=None, index=None,
                        of=len(pieces) or None, missing_volume=_number(info.get("missing_volume")),
                        invented_volume=_number(info.get("invented_volume")),
                        input_volume=_number(info.get("input_volume"), f.input_volume),
                        exact_iou=_number(info.get("exact_iou"), f.exact_iou),
                        dropped_small=_whole(info.get("dropped_small"), 0), dropped_limit=_whole(info.get("dropped_limit"), 0),
                        limits=dict(info.get("limits") or {}) if isinstance(info.get("limits"), dict) else f.limits,
                        history=history, done=None)
        elif kind == "feature_done":
            pieces = tuple(replace(p, state="skipped") if p.state in ("pending", "current", "guessed", "split") else p
                           for p in f.pieces)
            f = replace(f, pieces=pieces, current=None)
            f = replace(f, history=self._close_iteration(f), done=dict(_light(info)),
                        exact_iou=_number(info.get("exact_iou"), f.exact_iou),
                        missing_volume=_number(info.get("missing_volume"), f.missing_volume),
                        invented_volume=_number(info.get("invented_volume"), f.invented_volume))
        elif kind == "feature_split":
            pid = _whole(info.get("piece_id"), f.current)
            pieces = list(f.pieces)
            at = next((i for i, p in enumerate(pieces) if p.id == pid), None)
            if at is None:
                pieces.append(FeaturePiece(id=pid if pid is not None else max([p.id for p in pieces], default=0) + 1))
                at = len(pieces) - 1
            parent = pieces[at]
            taken = {p.id for p in pieces}
            segments = []
            for d in info.get("segments") or ():
                d = d if isinstance(d, dict) else {}
                seg = _piece(dict(d, side=d.get("side") or parent.side, op=d.get("op") or parent.op, parent=parent.id),
                             max(taken, default=0) + 1)
                if seg.id in taken:                                            # never two entries of one id
                    seg = replace(seg, id=max(taken) + 1)
                taken.add(seg.id)
                segments.append(seg)
            grid = info.get("grid")
            pieces[at] = replace(parent, state="split", split=dict(
                method=str(info.get("method") or ""), pitch=_number(info.get("pitch")),
                grid=tuple(_whole(n, 0) for n in grid) if isinstance(grid, (list, tuple)) else None,
                segments=tuple(s.id for s in segments), seconds=_number(info.get("seconds"))))
            pieces[at + 1:at + 1] = segments                                   # nested: right after their piece
            f = replace(f, pieces=tuple(pieces), current=parent.id, iteration=_whole(info.get("iteration"), f.iteration))
        else:
            given = info.get("piece") if isinstance(info.get("piece"), dict) else None
            pid = _whole(info.get("piece_id"), _whole((given or {}).get("id"), f.current))
            if pid is None:
                pid = max([p.id for p in f.pieces], default=0) + 1
            pieces = list(f.pieces)
            if f.current is not None and f.current != pid:     # left without a verdict: its guess waits for the build
                pieces = [replace(p, state="guessed" if p.candidates else "skipped")
                          if p.id == f.current and p.state == "current" else p for p in pieces]
            at = next((i for i, p in enumerate(pieces) if p.id == pid), None)
            if at is None:
                pieces.append(_piece(given or dict(id=pid, op=info.get("op")), pid))
                at = len(pieces) - 1
            p = pieces[at]
            if kind == "feature_piece":
                new = _piece(given, pid) if given else p
                mesh = _piece_mesh(info.get("mesh"))
                p = replace(new, id=pid, state="current", mesh=mesh if mesh is not None else p.mesh,
                            mesh_truncated=bool(info.get("mesh_truncated")), bbox=new.bbox if new.bbox is not None else p.bbox,
                            parent=new.parent if new.parent is not None else p.parent,     # a segment inherits
                            side=(given or {}).get("side") and new.side or p.side or new.side,
                            op=(given or {}).get("op") and new.op or p.op or new.op,
                            voxels=new.voxels if new.voxels is not None else p.voxels)
                f = replace(f, index=_whole(info.get("index"), at + 1), of=_whole(info.get("of"), f.of or len(pieces)))
            else:
                known = dict(primitive=candidate if candidate is not None else p.primitive,
                             source=str(info["source"]) if info.get("source") else p.source,
                             score=_number(info.get("score"), p.score), op=str(info.get("op") or p.op))
                validation = info.get("validation") if isinstance(info.get("validation"), dict) else None
                whole = p.split is None                        # a split piece's verdict sums its segments' up
                if kind == "feature_candidate":
                    p = replace(p, state="current", candidates=p.candidates + 1, precision=_number(info.get("precision")),
                                recall=_number(info.get("recall")), **known)
                elif kind == "feature_accepted":
                    done = info.get("segments_accepted")
                    p = replace(p, state="accepted", validation=validation, reason="", **known,
                                segments_accepted=tuple(_whole(x, -1) for x in done) if isinstance(done, (list, tuple)) else ())
                    f = replace(f, applied=f.applied + (1 if whole and candidate is not None else 0))
                else:
                    p = replace(p, state="rejected", validation=validation, reason=str(info.get("reason") or ""), **known)
                    f = replace(f, rejected=f.rejected + (1 if whole else 0))
                if validation and _number(validation.get("exact_iou")) is not None and kind == "feature_accepted":
                    f = replace(f, exact_iou=_number(validation.get("exact_iou")))
            pieces[at] = p
            # a rejected piece stays the one in work: its next guess may follow
            f = replace(f, pieces=tuple(pieces), current=None if p.state == "accepted" else pid,
                        iteration=_whole(info.get("iteration"), f.iteration))
        return replace(f, layers=self._feature_meshes(f))

    @staticmethod
    def _close_iteration(f: FeatureView) -> tuple:
        """`f.history` with the iteration of `f` as its last entry (replaced when it is there already)."""
        counts = f.counts()
        row = dict(iteration=f.iteration, missing_volume=f.missing_volume, invented_volume=f.invented_volume,
                   input_volume=f.input_volume, exact_iou=f.exact_iou, pieces=len(f.pieces), applied=counts["accepted"],
                   rejected=counts["rejected"],
                   skipped=counts["skipped"] + counts["pending"] + counts["current"] + counts["guessed"])
        return tuple(h for h in f.history if h.get("iteration") != f.iteration) + (row,)

    def _feature_meshes(self, f: FeatureView) -> Dict[str, MeshData]:
        """Per piece state the pieces in it as one mesh: a piece's own mesh when it has one, else its bounding box. A
        state whose pieces did not change keeps its mesh object (the viewer redraws only what changed)."""
        out = {}
        segments = [p for p in f.pieces if p.parent is not None]
        key = (f.iteration,) + tuple((p.id, p.state, id(p.mesh) if p.mesh is not None else None) for p in segments)
        cached = self._feature_layers.get("segments")
        if cached is None or cached[0] != key:
            cached = self._feature_layers["segments"] = (key, self._segment_meshes(segments))
        out.update({name: mesh for name, mesh in cached[1].items() if mesh is not None})
        for state in PIECE_STATES:
            # a split piece is drawn as its segments, a segment in its own colour
            chosen = [p for p in f.pieces if p.state == state and p.split is None and p.parent is None]
            key = (f.iteration,) + tuple((p.id, id(p.mesh) if p.mesh is not None else None) for p in chosen)
            cached = self._feature_layers.get(state)
            if cached is None or cached[0] != key:
                mesh = merge_meshes([p.mesh if p.mesh is not None else (bbox_mesh(*p.bbox) if p.bbox is not None else None)
                                     for p in chosen])
                cached = self._feature_layers[state] = (key, mesh)
            if cached[1] is not None:
                out[state] = cached[1]
        return out

    @staticmethod
    def _segment_meshes(segments) -> Dict[str, Optional[MeshData]]:
        """"segments": the segments as one mesh, each in its colour of the palette (by its place among its parent's
        segments) with its state as the opacity - its voxel surface when it has one, else its box; "segment_outline": the
        segments' boxes in their state's colour (drawn as a wireframe)."""
        body, outline, order = [], [], {}
        for p in segments:
            k = order.setdefault(p.parent, {}).setdefault(p.id, len(order[p.parent]))
            box = bbox_mesh(*p.bbox) if p.bbox is not None else None
            mesh = p.mesh if p.mesh is not None else box
            for shown, rgb, alpha, into in ((mesh, SEGMENT_PALETTE[k % len(SEGMENT_PALETTE)], SEGMENT_OPACITY.get(p.state, 0.3), body),
                                            (box, STATE_COLOURS.get(p.state, (0.5, 0.5, 0.5)), 1.0, outline)):
                if shown is not None and len(shown.triangles):
                    colour = np.round(255 * np.array(rgb + (alpha,))).astype(np.uint8)
                    into.append((shown, np.tile(colour, (len(shown.triangles), 1))))
        out = {}
        for name, parts in (("segments", body), ("segment_outline", outline)):
            merged = merge_meshes([m for m, _ in parts])
            out[name] = ColouredMesh(merged.vertices, merged.triangles, np.vstack([c for _, c in parts])) if merged else None
        return out


BIG_ARRAY = 100_000                 # elements: arrays this large are meshed above, not kept in a state's info


def _light(info: Optional[dict]) -> dict:
    """The step's info without the grid-sized arrays (residual masks, cut history, the input occupancy) and the grid
    spec: every state is kept for the timeline, and on the wall-thickness grid (20.7M cells on the frame) those came to
    ~40-60 MB per step. The report reads only the scalars; the meshes were made from the arrays already."""
    return {k: v for k, v in (info or {}).items()
            if k != "spec" and not (isinstance(v, np.ndarray) and v.size > BIG_ARRAY)}


# ------------------------------------------------------------------------------------------------ session

# The feature phase's settings and the ResidualConfig fields they may be called (the first name the config has is used;
# a setting the loop does not know is not passed, so the GUI runs with and without the feature phase in the loop).
FEATURE_FIELDS = {
    "feature_phase": ("feature_phase",),
    "max_pieces": ("feature_max_pieces", "feature_pieces_max", "max_feature_pieces", "feature_pieces"),
    "max_total_pieces": ("feature_max_total_pieces", "feature_total_pieces", "feature_max_pieces_total"),
    "max_iterations": ("feature_max_iterations", "feature_iterations_max", "max_feature_iterations", "feature_iterations",
                       "feature_max_rounds"),
    "time_budget_s": ("feature_time_budget_s", "feature_time_budget", "feature_budget_s", "feature_max_seconds",
                      "feature_seconds", "feature_timeout"),
    "min_score": ("feature_min_score", "feature_score_min", "min_feature_score"),
    "target_iou": ("feature_target_iou", "feature_iou_target", "target_iou"),
    "rough_subparts": ("rough_subparts",),             # the rough model as several sub-parts added before any cut
    "feature_subparts": ("feature_subparts",),
    "max_applied": ("feature_max_applied",),           # features applied at most in the whole phase
    "apply_at_once": ("feature_apply_at_once",),       # an accepted feature joins the model at once, confirmed per batch         # a piece without a tool as sub-parts, before the voxel fallback
}


def config_fields() -> Optional[set]:
    """The field names of reco_engine.residual_loop.ResidualConfig (None when the loop cannot be imported)."""
    try:
        from dataclasses import fields
        from reco_engine.residual_loop import ResidualConfig
        return {f.name for f in fields(ResidualConfig)}
    except Exception:
        return None


def config_defaults() -> Dict[str, object]:
    """{setting: the loop's own default} for the FEATURE_FIELDS the config has."""
    try:
        from dataclasses import MISSING, fields
        from reco_engine.residual_loop import ResidualConfig
        by_name = {f.name: f.default for f in fields(ResidualConfig) if f.default is not MISSING}
    except Exception:
        return {}
    out = {}
    for setting, names in FEATURE_FIELDS.items():
        name = next((n for n in names if n in by_name), None)
        if name is not None and by_name[name] is not None:
            out[setting] = by_name[name]
    return out


def feature_config(settings: Dict[str, object], known: Optional[set]) -> Tuple[Dict[str, object], Tuple[str, ...]]:
    """(keyword arguments for the config, the settings it has no field for). `settings`: FEATURE_FIELDS keys -> values
    (None: the loop's default). `known`: the config's field names; None (a test's steps_fn takes a plain dict) passes
    every setting under its first name."""
    kw, missing = {}, []
    for setting, names in FEATURE_FIELDS.items():
        value = settings.get(setting)
        if value is None:
            continue
        name = names[0] if known is None else next((n for n in names if n in known), None)
        if name is None:
            missing.append(setting)
        else:
            kw[name] = value
    return kw, tuple(missing)


def make_cascade_session(path, top_k: int = 2, op_guard: bool = False, max_ops: Optional[int] = None,
                         freecad_per_step: bool = False, registry_path=None, resolution: Optional[int] = None,
                         freecad_final: bool = True,
                         max_resolution: Optional[int] = None, type_source: str = "registry", steps_fn=None,
                         field_backend: Optional[str] = None, grid: str = "cube", start_shape: str = "legacy",
                         mode: str = "loop", feature_phase: Optional[bool] = None, feature_max_pieces: Optional[int] = None,
                         feature_max_total_pieces: Optional[int] = None, feature_max_iterations: Optional[int] = None, feature_time_budget_s: Optional[float] = None,
                         feature_min_score: Optional[float] = None, feature_target_iou: Optional[float] = None,
                         rough_subparts: Optional[bool] = None, feature_subparts: Optional[bool] = None,
                         feature_apply_at_once: Optional[bool] = None, feature_max_applied: Optional[int] = None,
                         voxel_meshes: bool = False, memory_min_free_mb: Optional[float] = None, first_guess=None):
    """A StepSession over the residual add-cut loop on the STL/STEP file `path`. `type_source`: "registry" (every type
    entry, fused) | "descriptors" | "cnn" (reco_engine.cascade_node_support.residual_config filters the registry).
    `freecad_final`: the program is built and graded in FreeCAD at the end of the run (off: the GUI's button
    "FreeCAD rebuild" does it on demand). `feature_*`: the feature phase and its limits (None: the loop's own default), passed under the names the
    ResidualConfig has (FEATURE_FIELDS); what it has no field for is said on stdout (the Output panel) and left out.
    `steps_fn(path, config)` replaces reco_engine.residual_loop.residual_steps (tests); the loop runs lazily, when the
    first step is asked for."""
    from reco_gui.session import StepSession
    config = dict(top_k=top_k, op_guard=op_guard, max_ops=max_ops, freecad_per_step=freecad_per_step,
                  freecad=bool(freecad_final or freecad_per_step),
                  registry_path=str(registry_path) if registry_path else None, resolution=resolution,
                  max_resolution=max_resolution, type_source=type_source, field_backend=field_backend, grid=grid, start_shape=start_shape,
                  mode=mode, memory_min_free_mb=memory_min_free_mb, first_guess=first_guess or None)
    feature = dict(feature_phase=feature_phase, max_pieces=feature_max_pieces,
                   max_total_pieces=feature_max_total_pieces, max_iterations=feature_max_iterations,
                   time_budget_s=feature_time_budget_s, min_score=feature_min_score, target_iou=feature_target_iou,
                   rough_subparts=rough_subparts, feature_subparts=feature_subparts,
                   apply_at_once=feature_apply_at_once, max_applied=feature_max_applied)
    if steps_fn is None:
        from reco_engine.residual_loop import residual_steps as steps_fn          # ImportError here: Start reports it
        from reco_engine.cascade_node_support import residual_config
        kw, missing = feature_config(feature, config_fields())
        if feature_phase and "feature_phase" in missing:
            print("[gui] feature phase: this loop's ResidualConfig has no 'feature_phase' switch - the run has no "
                  "feature phase")
        elif feature_phase and missing:
            print(f"[gui] feature phase: the ResidualConfig has no field for {', '.join(missing)} - the loop's own "
                  f"limits apply")
        config = residual_config(**config, **kw)
    else:
        config.update(feature_config(feature, None)[0])
    session = timed_session(StepSession, steps_fn(str(path), config), CascadeFolder(wall_grid=voxel_meshes, voxel_meshes=voxel_meshes))
    session.config = config                               # every setting of the run, for the run store
    session.paced = CASCADE_PACED_STAGES                  # no pacing wait after a candidate
    return session


def timed_session(session_class, steps, folder):
    """A StepSession whose states carry the timing sections of their step (`CascadeState.timings`: the loop's own, then
    the GUI's folding as "gui fold"); `session.recorder` has the whole run (reco_engine.timing)."""
    rec = Recorder()

    def recorded():
        rec.activate()                                    # in the thread that runs the loop (the worker)
        yield from steps

    def fold(prev, step, index):
        with rec.section("gui fold"):
            state = folder(prev, step, index)
        return replace(state, timings=tuple(rec.take()))

    session = session_class(recorded(), fold=fold)
    session.recorder = rec
    return session
