#!/usr/bin/env python3
"""
The TYPE -> ROTATION -> DIMENSIONS -> POSITION cascade: one primitive for one target (the whole input, or one leftover
component of the add-cut loop, reco_engine/residual_loop.py), from whatever the stage registry (reco_engine/stage_registry.py,
models/registry.json) lists for each stage - CNNs, analytic fits and priors - tried in COST order.

Why a cascade and not one network: the inventory (docs, plan Phase 0) found that the type nets are good (0.99 on box/sphere),
while every rotation and dimension net so far loses to an analytic measurement (face-normal frame fit, measured extents,
the IoU descents of sphere_fit / cylinder_fit). So each stage runs its cheap modes first and stops as soon as one is
confident enough (`entry.gate`); only an unsure answer pays for the next mode.

    target_channels   (5, R, R, R): occupancy, SDF (positive INSIDE, mm), V = closest point - x (mm), array axes = world x, y, z
                      (dataset_gen/src/field_channels.py: mesh_channels for the input, voxel_channels for a leftover)
    spec              the grid (field_channels.GridSpec, or `LocalGridSpec` here: lo, hi, resolution; cells over [lo, hi])

    prim = run_cascade(channels, spec, registry, field=field)       # reco_engine.result_schema.Primitive

Stages:
  TYPE        every type entry in cost order; each gives probabilities over the kinds; the first whose top probability
              reaches its gate short-circuits. When several entries answered, their probabilities are FUSED: the weighted
              mean per kind, weights = each entry's registry metrics["parent_acc"] (0.5 without one); the top-k kinds
              (k = 2 by default) become branches. Every entry's vote is recorded in the type StageSource's `detail`, and the
              winner's subclass (a hierarchical CNN's or the descriptors') with it - the add-cut loop's sub-loop hint.
              Type sources read the target SCALE-NORMALISED (`inscale_target(mode="always")`: the largest extent scaled to
              0.7 x the 40 mm grid), so a 6.7 m frame and a 20 mm sample look alike to them; type is scale-free for our
              classes, pose and dimensions keep their analytic paths in mm.
  ROT/DIM/POS per branch (kind), the entries registered for that stage and kind, in cost order, same gating. The analytic
              modes answer all three at once (a fit), so they are computed once and shared; a box whose dimensions come from
              a different mode than its rotation is re-measured along the chosen axes (the axis order must agree). A "cnn"
              pose entry answers only when `models` maps its id to a callable (`_cnn_pose`): the rotation and dimension
              CNN entries of models/registry.json are never loaded from their paths, so in a normal run they do not run.
  REFINE      the exact-field fit (reco_field.fit, Tukey IRLS) on the target's surface samples - the input mesh's surface near
              the target when `field` is given ("the leftover's exposed input surface"), else points x + V(x) of the channels.
              It is what gives cones, tori and segments (u/v); it replaces the voxel answer when its IoU is not worse.
  SCORE       each branch is scored by the voxel IoU of the finished primitive with the target inside `scope` (default the
              whole grid; the add-cut loop passes "outside the model" for an add and "inside the model" for a cut, where
              material outside the scope is free). The best branch wins, the others go to `alternatives`; none reaching
              `pass_iou` -> kind "unknown" (the loop then falls back to voxel cubes).

Registry entries are duck-typed (id, stage, kinds, mode "cnn" | "descriptor" | "analytic" | "prior", path, input_kind, cost,
gate, metrics), so tests can pass plain objects. A CNN entry runs through `models`: a dict {entry id: Models | callable} or one
Models object; a callable gets the in-scale occupancy ({-1, +1}, 64^3, 40 mm, array frame [z, -y, x], what the type nets were
trained on) and returns [(kind, probability), ...] - or ([(kind, p)], {parent: {subclass: p}}) when it knows subclasses.
Without `models` the checkpoint at `entry.path` is loaded (type_cnn/ inside, or the net's own directory) and cached. A
"descriptor" entry is reco_engine/type_descriptors.py's model (the joblib file at `entry.path`, or models[entry.id]).
Type entries marked `legacy` in the registry (D-007) cast no vote.

Silent degradation (review R-01, 2026-09-30; documented, not changed): `load_default_registry` returns the network-free
DEFAULT_REGISTRY on ANY error of the registry file, and a type net or descriptor model that fails to load prints one
stderr line and is cached as None for the life of the process - the type stage then fuses fewer voters (or falls to
the uniform prior) without a warning in the step stream; `TypeStageResult.as_dict()["entries"]` lists only the entries
that voted.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from reco_engine.timing import timed_fn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine.result_schema import KINDS, UNKNOWN, Primitive, StageSource  # noqa: E402
from reco_engine.result_to_program import primitive_to_sp, primitive_from_fit, quaternion_of  # noqa: E402
from reco_engine import torus_guard  # noqa: E402

INSCALE_RESOLUTION = 64          # what the type nets were trained on (in-scale voxel mode)
INSCALE_GRID_MM = 40.0
PASS_IOU = 0.8                   # a branch below this explains its target too poorly to be called that kind
FIELD_PREFERENCE = 0.02          # the exact-field fit replaces the voxel answer unless its IoU is worse by more than this
MIN_FIELD_POINTS = 40            # fewer surface points than this: no field fit
VOXEL_UNREADABLE = ("points", "exact_field", "vf_exact", "fused")   # type-net inputs the voxel target cannot provide here


# ------------------------------------------------------------------------------------------------ grid (duck-typed)

@dataclass
class LocalGridSpec:
    """Fallback GridSpec while dataset_gen/src/field_channels.py is not there: a cube of cells over [lo, hi]."""
    lo: np.ndarray
    hi: np.ndarray
    resolution: int

    def centres(self) -> np.ndarray:
        ax = [np.linspace(self.lo[i], self.hi[i], self.resolution + 1) for i in range(3)]
        mids = [(a[1:] + a[:-1]) / 2.0 for a in ax]
        return np.stack(np.meshgrid(*mids, indexing="ij"), axis=-1)


@dataclass(frozen=True, eq=False)
class BoxGridSpec:
    """A box of cells with ONE pitch on every axis and its own count per axis (not a cube): voxel centres
    `lo + (index + 0.5) * pitch`, indexed [i, j, k] = (x, y, z), world axes. The grid a pitch licensed by the part's
    wall thickness needs (D-005; the frame at tau = 0.12: 88 x 562 x 419 cells instead of 648^3). `centres()` is computed
    once and cached (float32 would lose the thin walls' sub-voxel offsets on a 7 m part: kept float64)."""
    lo: Tuple[float, float, float]
    shape: Tuple[int, int, int]
    pitch: float

    @property
    def hi(self) -> Tuple[float, float, float]:
        return tuple(float(self.lo[d] + self.shape[d] * self.pitch) for d in range(3))

    def centres(self) -> np.ndarray:
        c = getattr(self, "_centres", None)
        if c is None:
            ax = [self.lo[d] + (np.arange(self.shape[d]) + 0.5) * self.pitch for d in range(3)]
            c = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
            c.flags.writeable = False
            object.__setattr__(self, "_centres", c)
        return c

    def to_record(self) -> dict:
        return dict(lo=[float(v) for v in self.lo], shape=[int(n) for n in self.shape], pitch=float(self.pitch),
                    axes="world_xyz", sign="positive_inside", units="mm")


def box_grid_spec(lo, hi, pitch: float, margin: int = 2) -> BoxGridSpec:
    """The BoxGridSpec over [lo, hi] at `pitch`, with `margin` empty cells on every side."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    shape = tuple(int(np.ceil((hi[d] - lo[d]) / pitch)) + 2 * margin for d in range(3))
    return BoxGridSpec(tuple(float(lo[d] - margin * pitch) for d in range(3)), shape, float(pitch))


def grid_spec(lo, hi, resolution: int):
    try:
        from src.field_channels import grid_spec as gs
        return gs(np.asarray(lo, float), np.asarray(hi, float), resolution)
    except ImportError:
        return LocalGridSpec(np.asarray(lo, float), np.asarray(hi, float), int(resolution))


def grid_centres(spec) -> np.ndarray:
    """(R, R, R, 3) world centres of the cells, whatever the GridSpec implementation calls them."""
    for name in ("centres", "centers"):
        c = getattr(spec, name, None)
        if c is not None:
            return np.asarray(c() if callable(c) else c, float)
    return LocalGridSpec(np.asarray(spec.lo, float), np.asarray(spec.hi, float), int(spec.resolution)).centres()


def grid_frame(spec) -> Tuple[np.ndarray, float]:
    """(world centre of cell [0, 0, 0], pitch in mm): index -> world is c0 + index * pitch on every axis."""
    c = grid_centres(spec)
    return c[0, 0, 0].copy(), float(c[1, 0, 0, 0] - c[0, 0, 0, 0])


def to_world(spec, index) -> np.ndarray:
    c0, pitch = grid_frame(spec)
    return c0 + np.asarray(index, float) * pitch


def primitive_occupancy(p: Primitive, spec, centres: Optional[np.ndarray] = None) -> np.ndarray:
    c = grid_centres(spec) if centres is None else centres
    return primitive_to_sp(p).sdf(c) <= 0


# ------------------------------------------------------------------------------------------------ registry access

@dataclass
class _Entry:
    id: str
    stage: str
    kinds: Tuple[str, ...]
    mode: str
    path: Optional[str] = None
    input_kind: str = "occupancy"
    cost: float = 1.0
    gate: Optional[float] = None
    metrics: Dict = dc_field(default_factory=dict)


# Used when no registry is given and reco_engine/stage_registry.py cannot provide one: network-free, the analytic modes the
# inventory found best (frame fit for boxes, the IoU descents for spheres and cylinders, the exact-field fit for the rest).
DEFAULT_REGISTRY = [
    _Entry("type-analytic", "type", KINDS, "analytic", "voxel_fit", cost=10.0, gate=0.9),
    _Entry("box-frame", "rotation", ("box",), "analytic", "frame", cost=1.0, gate=0.9),
    _Entry("box-frame-dims", "dimensions", ("box",), "analytic", "frame", cost=1.0, gate=0.9),
    _Entry("box-frame-pos", "position", ("box",), "analytic", "frame", cost=1.0, gate=0.9),
    _Entry("sphere-fit", "rotation", ("sphere",), "analytic", "sphere_fit", cost=1.0, gate=0.9),
    _Entry("sphere-fit-dims", "dimensions", ("sphere",), "analytic", "sphere_fit", cost=1.0, gate=0.9),
    _Entry("sphere-fit-pos", "position", ("sphere",), "analytic", "sphere_fit", cost=1.0, gate=0.9),
    _Entry("cylinder-fit", "rotation", ("cylinder",), "analytic", "cylinder_fit", cost=2.0, gate=0.9),
    _Entry("cylinder-fit-dims", "dimensions", ("cylinder",), "analytic", "cylinder_fit", cost=2.0, gate=0.9),
    _Entry("cylinder-fit-pos", "position", ("cylinder",), "analytic", "cylinder_fit", cost=2.0, gate=0.9),
]


def load_default_registry():
    """The registry file (reco_engine.stage_registry.load_registry) when it exists and loads, else DEFAULT_REGISTRY.
    Any exception (a missing file, a malformed entry) gives the network-free default silently: no warning, no step."""
    try:
        from reco_engine.stage_registry import load_registry
        reg = load_registry()
        if reg:
            return reg
    except Exception:                                                  # not written yet / no file: the analytic default
        pass
    return list(DEFAULT_REGISTRY)


def _entries_for(registry, stage: str, kind: Optional[str] = None) -> list:
    try:
        from reco_engine.stage_registry import entries_for
        return list(entries_for(registry, stage, kind))
    except ImportError:
        pass
    out = [e for e in registry if e.stage == stage and (kind is None or not e.kinds or kind in e.kinds)]
    return sorted(out, key=lambda e: e.cost)


def _mode_label(entry, impl: str) -> str:
    if entry.mode in ("cnn", "descriptor"):
        return f"{entry.mode}:{entry.id}"
    if impl == "field_fit":
        return "field_fit"
    return f"{entry.mode}:{impl}"


def _impl(entry, kind: Optional[str]) -> str:
    """Which analytic / prior implementation an entry means, from its path or id (the registry names them; unknown names
    fall back to the kind's best-measured default)."""
    text = f"{entry.path or ''} {entry.id}".lower()
    for key, name in (("field", "field_fit"), ("obb", "obb"), ("bounding", "obb"), ("pca", "pca"), ("frame", "frame"),
                      ("sphere_fit", "sphere_fit"), ("cylinder_fit", "cylinder_fit"), ("voxel_fit", "voxel_fit")):
        if key in text:
            return name
    if entry.stage == "type":
        return "voxel_fit" if entry.mode == "analytic" else "uniform"
    if entry.mode == "prior" and kind == "box":
        return "obb"
    return {"box": "frame", "sphere": "sphere_fit", "cylinder": "cylinder_fit"}.get(kind, "field_fit")


# ------------------------------------------------------------------------------------------------ the context

@dataclass
class _Pose:
    prim: Primitive                   # kind, quaternion, centre, dims, u, v (no sources yet)
    iou: float
    impl: str


class _Context:
    """Everything shared by the stages of one cascade call: the target, its scope, the grid, the fit cache."""

    def __init__(self, channels, spec, field, models, scope, part_diagonal: Optional[float] = None):
        self.channels = np.asarray(channels)
        self.occ = self.channels[0] > 0.5
        self.spec = spec
        self.centres = grid_centres(spec)
        self.c0, self.pitch = self.centres[0, 0, 0], float(self.centres[1, 0, 0, 0] - self.centres[0, 0, 0, 0])
        self.field = field
        self.models = models
        self.scope = np.ones_like(self.occ) if scope is None else np.asarray(scope, bool)
        self.fits: Dict[Tuple[str, str], Optional[_Pose]] = {}
        self._samples = None
        self.part_diagonal = part_diagonal                             # the torus guard's yardstick (see `diagonal`)
        self.guarded: List[str] = []                                   # the torus guard's reasons, one per rejection

    def diagonal(self) -> float:
        """The part's bounding-box diagonal for the torus guard: the caller's, else the input field's (the length scale
        of every field-fit tolerance, reco_field.fit.Samples.extent), else the grid's."""
        if self.part_diagonal:
            return float(self.part_diagonal)
        if self.field is not None:
            try:
                return float(_field_samples(self.field).extent)
            except Exception:                                          # a field without a mesh: the grid
                pass
        return float(np.linalg.norm(self.centres[-1, -1, -1] - self.c0)) + self.pitch

    def admit(self, prim: Optional[Primitive]) -> Optional[Primitive]:
        """`prim`, or None when it cannot be built (Primitive.defect: the ABC batch's sector / box / torus crashes came
        through here unchecked, cascade.py `_estimate` -> ctx.iou) or the torus guard rejects it
        (reco_engine/torus_guard.py: spindle/horn, near-horn, oversized): a mode whose answer is such a primitive has
        no answer, as a fit that failed."""
        if prim is None:
            return None
        bad = prim.defect()                      # an inverted sweep, a zero extent, a void box: no SDF, no IoU, no build
        if bad is not None:
            self.guarded.append(f"invalid primitive: {bad}")
            return None
        if not torus_guard.tori(prim):
            return prim
        found = torus_guard.primitive_defect(prim, self.diagonal())
        if found is None:
            return prim
        self.guarded.append(found.text)
        return None

    # -- scoring
    @property
    def counted(self) -> np.ndarray:
        """Where a mismatch counts: inside the scope and not within one voxel of the target's boundary. A leftover component
        is cut back by half a voxel at both surfaces it lies between (residual_loop's band), so the EXACT primitive overlaps
        it by a voxel rim; forgiving that rim (as reco_engine/fit_grade.py does) keeps the exact fit from losing to a voxel
        fit that hugs the blocky component."""
        if not hasattr(self, "_counted"):
            from scipy import ndimage
            rim = ndimage.binary_dilation(self.occ) & ~ndimage.binary_erosion(self.occ)
            self._counted = self.scope & ~rim
        return self._counted

    def iou(self, p: Primitive) -> float:
        occ = primitive_occupancy(p, self.spec, self.centres)
        t, o = self.occ & self.counted, occ & self.counted
        union = int(np.count_nonzero(t | o))
        return int(np.count_nonzero(t & o)) / union if union else 0.0

    def iou_strict(self, p: Primitive) -> float:
        """The IoU without the forgiven rim: the tie-break between kinds the tolerant IoU cannot tell apart (a thin cylinder
        and a box differ mostly in that rim - void_fit breaks its ties the same way)."""
        occ = primitive_occupancy(p, self.spec, self.centres)
        t, o = self.occ & self.scope, occ & self.scope
        union = int(np.count_nonzero(t | o))
        return int(np.count_nonzero(t & o)) / union if union else 0.0

    def unexplained(self, p: Primitive) -> float:
        occ = primitive_occupancy(p, self.spec, self.centres)
        n = int(np.count_nonzero(self.occ))
        return float(np.count_nonzero(self.occ & ~occ) / n) if n else 0.0

    def world(self, index) -> np.ndarray:
        return self.c0 + np.asarray(index, float) * self.pitch

    # -- surface samples for the exact-field fit
    def samples(self):
        if self._samples is None:
            self._samples = _target_samples(self)
        return self._samples


_FIELD_SAMPLES: Dict[int, object] = {}


def _field_samples(field):
    from reco_field.fit import make_samples
    key = id(field)
    if key not in _FIELD_SAMPLES:
        _FIELD_SAMPLES.clear()                                       # one input at a time: do not keep old meshes alive
        _FIELD_SAMPLES[key] = (field, make_samples(field, n_surface=6000, n_near=3000))
    return _FIELD_SAMPLES[key][1]


def _target_samples(ctx: _Context):
    """reco_field.fit.Samples of the target's surface. With the input field: the input mesh's own surface points that lie
    within one voxel of the target (for the whole input: all of them, with the off-surface distances that pin the size).
    Without: x + V(x) at the target's boundary voxels, normals from V."""
    from reco_field.fit import Samples
    from scipy import ndimage
    if ctx.field is not None:
        s = _field_samples(ctx.field)
        idx = np.clip(np.rint((s.surface - ctx.c0) / ctx.pitch).astype(int), 0, np.array(ctx.occ.shape) - 1)
        near = ndimage.binary_dilation(ctx.occ, iterations=1)
        keep = near[idx[:, 0], idx[:, 1], idx[:, 2]]
        if keep.mean() > 0.95:
            return s
        return Samples(s.surface[keep], s.normals[keep], np.empty((0, 3)), np.empty(0), s.extent)
    sdf, V = ctx.channels[1], np.moveaxis(ctx.channels[2:5], 0, -1)
    band = ctx.occ & (np.abs(sdf) < 1.5 * ctx.pitch)
    if not band.any():
        return None
    x, v, s_in = ctx.centres[band], V[band], sdf[band]
    n = np.linalg.norm(v, axis=1)
    ok = n > 1e-9
    normals = np.where((s_in[ok] > 0)[:, None], v[ok], -v[ok]) / n[ok][:, None]      # V points to the surface: outward iff inside
    return Samples(x[ok] + v[ok], normals, np.empty((0, 3)), np.empty(0), float(np.linalg.norm(ctx.centres[-1, -1, -1] - ctx.c0)))


# ------------------------------------------------------------------------------------------------ analytic estimators

def _box_pose(ctx: _Context, M, half_vox, centre_vox, impl) -> Primitive:
    M = np.asarray(M, float)
    h = np.asarray(half_vox, float) * ctx.pitch
    return Primitive(kind="box", quaternion=quaternion_of(M), center_mm=tuple(float(x) for x in ctx.world(centre_vox)),
                     dims_mm=dict(x_half=float(h[0]), y_half=float(h[1]), z_half=float(h[2])))


MAX_FIT_CUBE = 192                  # the voxel fitters' cube side; a bigger target is block-averaged down to this


def _fit_cube(occ: np.ndarray):
    """(cube, offset, step): the voxel fitters (BoxOrienter, SphereFitter, CylinderFitter) take an n^3 array and answer
    in its voxel indices. A cube grid is passed as it is (offset 0, step 1: bit-identical to before). A box grid (the
    wall-thickness grid, BoxGridSpec) is cropped to the target's bounding box plus one empty voxel, embedded in a cube
    and - when that cube would be larger than MAX_FIT_CUBE - block-averaged by `step` (majority). A voxel index i of the
    cube is the grid index offset + step * i + (step - 1) / 2."""
    occ = np.asarray(occ, bool)
    if len(set(occ.shape)) == 1:
        return occ, np.zeros(3), 1
    pts = np.argwhere(occ)
    lo = pts.min(axis=0) - 1
    size = pts.max(axis=0) + 2 - lo
    step = max(1, int(np.ceil(size.max() / MAX_FIT_CUBE)))
    n = int(np.ceil(size.max() / step)) * step
    lo = np.maximum(lo, 0)
    crop = occ[lo[0]:lo[0] + n, lo[1]:lo[1] + n, lo[2]:lo[2] + n]
    cube = np.zeros((n, n, n), bool)
    cube[:crop.shape[0], :crop.shape[1], :crop.shape[2]] = crop
    if step > 1:
        m = n // step
        cube = cube.reshape(m, step, m, step, m, step).mean(axis=(1, 3, 5)) >= 0.5
    return cube, lo.astype(float), step


def _from_cube(index, offset, step) -> np.ndarray:
    return np.asarray(offset, float) + step * np.asarray(index, float) + (step - 1) / 2.0


def _fit_frame_box(ctx: _Context, align: str) -> Optional[Primitive]:
    from reco_engine.box_orientation import BoxOrienter
    if np.count_nonzero(ctx.occ) < 20:
        return None
    cube, offset, step = _fit_cube(ctx.occ)
    occ = cube.astype(np.uint8)
    n = occ.shape[0]
    try:
        est = BoxOrienter.frame_only(grid_mm=n * ctx.pitch * step, resolution=n, align_method=align).estimate(occ)
    except ValueError:                  # walls about one voxel thick vanish when resampled into the frame (the frame's
        return None                     # webs at 96^3: measured_half_extents of an empty volume) - this mode has no answer
    if est is None:
        return None
    centre = est.center_vox
    if centre is None:
        pts = np.argwhere(occ)
        centre = (pts.min(axis=0) + pts.max(axis=0)) / 2.0
    return _box_pose(ctx, est.orientation, np.asarray(est.half_extents_vox, float) * step, _from_cube(centre, offset, step),
                     align)


def _obb(ctx: _Context) -> Optional[Primitive]:
    """The oriented bounding box: face-normal frame, extents that COVER every target voxel (no IoU refinement) - the safe
    first add of the loop when no primitive explains the whole input. With the input field: reco_field.fit.fit_box_bounding
    on the exact surface points (their true min/max along the frame)."""
    s = ctx.samples() if ctx.field is not None else None
    if s is not None and len(s.surface) >= MIN_FIELD_POINTS:
        from reco_field.fit import fit_box_bounding
        return primitive_from_fit(fit_box_bounding(s))
    from training import rotation_symmetry as rs
    occ = ctx.occ.astype(np.uint8)
    if occ.sum() < 8:
        return None
    _, E = rs.align_volume(occ, "frame")
    if np.linalg.det(E) < 0:
        E = E.copy(); E[:, 2] *= -1.0
    proj = np.argwhere(occ).astype(float) @ E
    lo, hi = proj.min(axis=0) - 0.5, proj.max(axis=0) + 0.5
    return _box_pose(ctx, E, (hi - lo) / 2.0, E @ ((lo + hi) / 2.0), "obb")


def _fit_sphere(ctx: _Context) -> Optional[Primitive]:
    from reco_engine.sphere_fit import SphereFitter
    cube, offset, step = _fit_cube(ctx.occ)
    n = cube.shape[0]
    est = SphereFitter(grid_mm=n * ctx.pitch * step, resolution=n).estimate(cube.astype(np.uint8))
    if est is None:
        return None
    return Primitive(kind="sphere", center_mm=tuple(float(x) for x in ctx.world(_from_cube(est.center_vox, offset, step))),
                     dims_mm=dict(radius=float(est.radius_vox * step * ctx.pitch)))


def _fit_cylinder(ctx: _Context) -> Optional[Primitive]:
    from reco_engine.cylinder_fit import CylinderFitter, _frame_with_axis
    cube, offset, step = _fit_cube(ctx.occ)
    n = cube.shape[0]
    est = CylinderFitter(grid_mm=n * ctx.pitch * step, resolution=n).estimate(cube.astype(np.uint8))
    if est is None:
        return None
    F = _frame_with_axis(est.axis)                                      # column 0 = axis; local z must be the axis
    R = np.stack([F[:, 1], F[:, 2], F[:, 0]], axis=1)
    return Primitive(kind="cylinder", quaternion=quaternion_of(R),
                     center_mm=tuple(float(x) for x in ctx.world(_from_cube(est.center_vox, offset, step))),
                     dims_mm=dict(radius=float(est.radius_vox * step * ctx.pitch),
                                  half_length=float(est.half_length_vox * step * ctx.pitch)))


def _fit_field(ctx: _Context, kind: str, hint: Optional[Primitive] = None) -> Optional[Primitive]:
    s = ctx.samples()
    if s is None or len(s.surface) < MIN_FIELD_POINTS:
        return None
    from reco_field import fit as fitmod
    try:
        if kind == "box" and hint is not None and hint.kind == "box":
            from reco_engine.result_to_program import rotation_matrix
            d = hint.dims_mm
            res = fitmod.fit_box(s, hint=(np.asarray(hint.center_mm), rotation_matrix(hint),
                                          np.array([d["x_half"], d["y_half"], d["z_half"]])))
        else:
            res = fitmod.FITTERS[kind](s)
        if res.half is None or res.centre is None or not np.all(np.isfinite(res.half)):
            return None
        prim = primitive_from_fit(res)
    except Exception:                                                  # a degenerate sample set: this mode has no answer
        return None
    return ctx.admit(prim)


def _estimate(ctx: _Context, kind: str, impl: str) -> Optional[_Pose]:
    """The pose of `kind` by the analytic implementation `impl`, cached per call (one fit answers rotation, dimensions and
    position together)."""
    key = (kind, impl)
    if key in ctx.fits:
        return ctx.fits[key]
    prim = None
    if impl == "obb" and kind == "box":
        prim = _obb(ctx)
    elif impl in ("frame", "pca") and kind == "box":
        prim = _fit_frame_box(ctx, impl)
    elif impl == "sphere_fit" and kind == "sphere":
        prim = _fit_sphere(ctx)
    elif impl == "cylinder_fit" and kind == "cylinder":
        prim = _fit_cylinder(ctx)
    elif impl == "field_fit" and kind in KINDS:
        prim = _fit_field(ctx, kind)
    elif impl in ("frame", "sphere_fit", "cylinder_fit", "voxel_fit"):          # a name for another kind: the kind's default
        default = {"box": "frame", "sphere": "sphere_fit", "cylinder": "cylinder_fit"}.get(kind, "field_fit")
        pose = _estimate(ctx, kind, default)
        ctx.fits[key] = pose
        return pose
    prim = ctx.admit(prim)
    pose = _Pose(prim, ctx.iou(prim), impl) if prim is not None else None
    ctx.fits[key] = pose
    return pose


# ------------------------------------------------------------------------------------------------ CNN helpers

_TYPE_NETS: Dict[str, object] = {}
_DESCRIPTORS: Dict[str, object] = {}
INSCALE_FILL = 0.7               # scaled modes: the target's largest extent becomes this share of the grid (the middle of the
                                 # generator's size_fraction range 0.5-0.9, dataset_gen/src/inscale.py)
INSCALE_MODES = ("fixed", "fit", "always")


@dataclass
class InscaleInput:
    """The target on the type nets' in-scale grid: `x` {-1, +1} float32 (resolution^3, array frame [z, -y, x]), `inside`
    the same occupancy as a bool grid - the base-point bit of the "dirlog_inside" input (the loop's grid is parity on the
    exact mesh, so its occupancy IS that bit), `scale` world mm per grid mm (1.0: the target's own scale; > 1: shrunk to
    fit), `extent_mm` the target's largest bounding-box side, `mode` the INSCALE_MODES entry that made it."""
    x: np.ndarray
    inside: np.ndarray
    scale: float
    extent_mm: float
    mode: str


def inscale_target(ctx: _Context, grid_mm: float = INSCALE_GRID_MM, resolution: int = INSCALE_RESOLUTION,
                   mode: str = "fit") -> Optional[InscaleInput]:
    """The target resampled (nearest neighbour) onto the in-scale grid, centred on its bounding box.
      "fixed"   at the target's own scale; None when it does not fit into `grid_mm` (the old behaviour: the frame,
                the 1.6 m targets and every real part never reached the nets)
      "fit"     its own scale when it fits, else scaled so that its largest extent is INSCALE_FILL x grid_mm
      "always"  scaled to INSCALE_FILL x grid_mm whatever its size (the type stage: type is scale-free for our classes
                and the nets saw parts of 0.5-0.9 of the grid; pose and dims keep their analytic paths in mm)"""
    from src import sdf_primitives as sp
    if mode not in INSCALE_MODES:
        raise ValueError(f"inscale mode must be one of {INSCALE_MODES}, got {mode!r}")
    pts = np.argwhere(ctx.occ)
    if not len(pts):
        return None
    lo, hi = ctx.world(pts.min(axis=0)), ctx.world(pts.max(axis=0))
    extent = float((hi - lo + ctx.pitch).max())
    scale = 1.0
    if mode == "always" or (mode == "fit" and extent > grid_mm):
        scale = extent / (INSCALE_FILL * grid_mm)
    elif extent > grid_mm:
        return None
    world = sp.voxel_centres(resolution, grid_mm) * scale + (lo + hi) / 2.0
    idx = np.rint((world - ctx.c0) / ctx.pitch).astype(int)
    inside = np.all((idx >= 0) & (idx < np.array(ctx.occ.shape)), axis=-1)
    idx = np.clip(idx, 0, np.array(ctx.occ.shape) - 1)
    occ = ctx.occ[idx[..., 0], idx[..., 1], idx[..., 2]] & inside
    return InscaleInput(occ.astype(np.float32) * 2.0 - 1.0, occ, float(scale), extent, mode)


def inscale_occupancy(ctx: _Context, grid_mm: float = INSCALE_GRID_MM, resolution: int = INSCALE_RESOLUTION,
                      mode: str = "fit") -> Optional[np.ndarray]:
    """`inscale_target(...).x` (the {-1, +1} grid alone, for the callers that need nothing else), None when there is none."""
    t = inscale_target(ctx, grid_mm, resolution, mode)
    return None if t is None else t.x


def _type_models(entry, models):
    if isinstance(models, dict):
        if entry.id in models:
            return models[entry.id]
    elif models is not None:
        return models
    if not entry.path:
        return None
    if entry.path not in _TYPE_NETS:
        try:
            _TYPE_NETS[entry.path] = _load_type_net(Path(entry.path))
        except Exception as exc:                                     # not a TypeCNN checkpoint (FusedNet, a point net): skip it
            print(f"[cascade] type entry {entry.id}: not loadable as a voxel Type CNN ({exc})", file=sys.stderr)
            _TYPE_NETS[entry.path] = None
    return _TYPE_NETS[entry.path]


def _load_type_net(path: Path):
    """A Models with only the Type CNN, from a checkpoint root (type_cnn/ inside) or the type_cnn directory itself."""
    import torch
    from reco_engine import model_registry as mr
    from training.train_type_cnn import TypeCNN
    path = path if path.is_absolute() else ROOT / path
    root = path if (path / "type_cnn").is_dir() else path.parent
    kind, trunc, channels = mr._type_cnn_input_kind(root, "best_model.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = mr._load_one(TypeCNN(in_channels=channels), root / "type_cnn" / "best_model.pth", device)
    return mr.Models(type_cnn=net, rotation_cnn=None, dimension_cnn=None, device=device, checkpoint_root=root,
                     type_input_kind=kind, type_sdf_truncation=trunc, type_resolution=mr._type_cnn_resolution(root))


def descriptor_resolution(model) -> int:
    """The grid resolution a descriptor model was fitted on: its meta "resolution", else type_descriptors.RESOLUTION
    (every fit so far). Read here, not in type_descriptors.py: that module's source is the fitted models' fingerprint."""
    from reco_engine.type_descriptors import RESOLUTION
    return int((getattr(model, "meta", None) or {}).get("resolution", RESOLUTION))


def check_descriptor_grid(model, occupancy) -> None:
    """Raise ValueError for a grid of another resolution than the descriptor model's fit: the features are not
    scale-free across resolutions (a 32^3 grid cost 0.049 parent accuracy, R-05c)."""
    expected = descriptor_resolution(model)
    if np.shape(occupancy) != (expected,) * 3:
        raise ValueError(f"descriptor model fitted on {expected}^3 grids, got shape {np.shape(occupancy)}")


def _descriptor_model(entry, models):
    """The reco_engine.type_descriptors model of a "descriptor" entry: `models[entry.id]` when given (a DescriptorModel or
    a callable on the in-scale grid, tests), else the joblib file at entry.path, loaded once; None when it is not there."""
    if isinstance(models, dict) and entry.id in models:
        return models[entry.id]
    if not entry.path:
        return None
    if entry.path not in _DESCRIPTORS:
        try:
            from reco_engine.type_descriptors import load
            path = Path(entry.path)
            _DESCRIPTORS[entry.path] = load(path if path.is_absolute() else ROOT / path)
        except Exception as exc:                                     # not fitted yet: the entry is skipped
            print(f"[cascade] type entry {entry.id}: no descriptor model ({exc})", file=sys.stderr)
            _DESCRIPTORS[entry.path] = None
    return _DESCRIPTORS[entry.path]


@dataclass
class TypeAnswer:
    """One type entry's answer: probabilities over the kinds it was asked about, and (hierarchical sources: a
    hierarchical Type CNN, the descriptors) per parent its subclasses {sub: p}."""
    probs: Dict[str, float]
    subclasses: Optional[Dict[str, Dict[str, float]]] = None

    def subclass_of(self, parent: str) -> Optional[str]:
        subs = (self.subclasses or {}).get(parent)
        return max(subs, key=subs.get) if subs else None


def _answer(raw, allowed: Sequence[str]) -> Optional[TypeAnswer]:
    """A TypeAnswer from what a type source returned: [(kind, p), ...] or {kind: p}; a (probs, subclasses) pair with
    subclasses {parent: {sub: p}} or {"parent/sub": p}; a predict.TypePrediction (top5 + subclasses)."""
    subclasses = None
    if raw is None:
        return None
    if hasattr(raw, "top5"):
        probs = dict(raw.top5)
        if getattr(raw, "subclasses", None):
            subclasses = {parent: dict(pairs) for parent, pairs in raw.subclasses.items()}
    elif isinstance(raw, tuple) and len(raw) == 2:
        probs, subs = dict(raw[0]), dict(raw[1] or {})
        if subs and all("/" in k for k in subs):
            from reco_engine.type_descriptors import DescriptorModel
            subclasses = {parent: DescriptorModel.subclasses_of(parent, subs) for parent in probs}
        elif subs:
            subclasses = {parent: dict(v) for parent, v in subs.items()}
    else:
        probs = dict(raw)
    out = {k: float(probs.get(k, 0.0)) for k in allowed}
    if subclasses:
        subclasses = {p: {s: float(v) for s, v in subs.items()} for p, subs in subclasses.items() if p in out and subs}
    return TypeAnswer(out, subclasses or None)


def _run_type_entry(entry, ctx: _Context, kinds: Sequence[str]) -> Optional[TypeAnswer]:
    allowed = [k for k in (entry.kinds or KINDS) if k in kinds]
    grid_mm = float((entry.metrics or {}).get("grid_mm", INSCALE_GRID_MM))
    inscale_mode = str((entry.metrics or {}).get("inscale_mode", "always"))         # the type stage: scale-normalised
    if entry.mode == "cnn":
        if getattr(entry, "input_kind", "occupancy") in VOXEL_UNREADABLE and not isinstance(ctx.models, dict):
            return None                                              # needs a point cloud / the mesh, not the voxel target
        net = _type_models(entry, ctx.models)
        t = inscale_target(ctx, grid_mm, getattr(net, "type_resolution", None) or INSCALE_RESOLUTION, mode=inscale_mode)
        if net is None or t is None:
            return None
        if callable(net) and not hasattr(net, "type_cnn"):
            return _answer(net(t.x), allowed)
        from reco_engine.predict import predict_type
        return _answer(predict_type(net, t.x, inside=t.inside), allowed)
    if entry.mode == "descriptor":
        model = _descriptor_model(entry, ctx.models)
        resolution = descriptor_resolution(model)
        t = inscale_target(ctx, grid_mm, resolution, mode=inscale_mode)
        if model is None or t is None:
            return None
        if callable(model) and not hasattr(model, "predict_proba"):
            return _answer(model(t.x), allowed)
        check_descriptor_grid(model, t.inside)
        return _answer(model.predict_proba(t.inside), allowed)
    if entry.mode == "prior":
        prior = (entry.metrics or {}).get("prior")
        if prior:
            return TypeAnswer({k: float(prior.get(k, 0.0)) for k in allowed})
        return TypeAnswer({k: 1.0 / len(allowed) for k in allowed}) if allowed else None
    # analytic: the kinds ranked by how well each one's own fit explains the target (IoU within the scope)
    out = {}
    for k in allowed:
        pose = _estimate(ctx, k, _impl(entry, k))                        # "voxel_fit": the kind's own default fit
        out[k] = pose.iou if pose is not None else 0.0
    return TypeAnswer(out)


# ------------------------------------------------------------------------------------------------ the stages

DEFAULT_TYPE_WEIGHT = 0.5        # the fusion weight of a type entry whose registry metrics give no parent_acc


@dataclass
class TypeVote:
    """One type entry's contribution to the fused type stage (recorded in StageSource.detail for the GUI)."""
    id: str
    mode: str                      # the StageSource mode label: "cnn:<id>", "descriptor:<id>", "analytic:...", "prior:..."
    weight: float                  # the entry's fusion weight (registry metrics parent_acc, else DEFAULT_TYPE_WEIGHT)
    answer: TypeAnswer
    seconds: float

    def as_dict(self, parent: Optional[str] = None) -> Dict:
        return dict(id=self.id, mode=self.mode, weight=float(self.weight), seconds=float(self.seconds),
                    probs={k: float(v) for k, v in self.answer.probs.items()},
                    subclass=self.answer.subclass_of(parent) if parent else None)


@dataclass
class TypeStageResult:
    """What the type stage decided for one target: `probs` (over the kinds; the fused vector when several entries
    answered), `origin` per kind the StageSource that records it, the votes, and the winner's subclass with the vote it
    came from (None when no entry knows subclasses). `channels` keeps the target the stage saw, so run_cascade can
    reuse this result (and the analytic fits it made) for the very same target."""
    probs: Dict[str, float]
    origin: Dict[str, StageSource]
    votes: List[TypeVote] = dc_field(default_factory=list)
    subclass: Optional[str] = None
    subclass_source: Optional[str] = None
    mode: str = "prior:uniform"
    channels: Optional[np.ndarray] = None
    fits: Dict = dc_field(default_factory=dict)

    @property
    def parent(self) -> Optional[str]:
        return max(self.probs, key=self.probs.get) if self.probs else None

    @property
    def hint(self) -> Optional[str]:
        """"<parent>/<subclass>" for residual_loop.residual_steps(hint_subclass=...), None without a subclass."""
        return f"{self.parent}/{self.subclass}" if self.subclass and self.parent else None

    def as_dict(self) -> Dict:
        return dict(parent=self.parent, subclass=self.subclass, subclass_source=self.subclass_source, mode=self.mode,
                    probs={k: float(v) for k, v in self.probs.items()},
                    entries=[v.as_dict(self.parent) for v in self.votes])


def type_weight(entry) -> float:
    w = (getattr(entry, "metrics", None) or {}).get("parent_acc")
    return float(w) if w is not None else DEFAULT_TYPE_WEIGHT


def _fuse(votes: List[TypeVote], kinds: Sequence[str]) -> Dict[str, float]:
    """The weighted mean of the votes' probabilities per kind, over the votes that answered that kind (an entry that
    cannot say "cylinder" casts no vote against it), the weights renormalised per kind; the result renormalised to sum 1."""
    fused = {}
    for k in kinds:
        have = [(v.weight, v.answer.probs[k]) for v in votes if k in v.answer.probs]
        total = sum(w for w, _ in have)
        if have and total > 0:
            fused[k] = sum(w * p for w, p in have) / total
    s = sum(fused.values())
    return {k: p / s for k, p in fused.items()} if s > 0 else fused


def _type_stage(ctx: _Context, registry, kinds: Sequence[str]) -> TypeStageResult:
    votes: List[TypeVote] = []
    for entry in _entries_for(registry, "type"):
        if getattr(entry, "legacy", False):                              # registry "legacy": true (D-007): not voted
            continue
        t0 = time.time()
        out = _run_type_entry(entry, ctx, kinds)
        if out is None or not out.probs:
            continue
        votes.append(TypeVote(entry.id, _mode_label(entry, _impl(entry, None)), type_weight(entry), out, time.time() - t0))
        if entry.gate is not None and max(out.probs.values()) >= entry.gate:
            break                                                     # confident: the costlier type modes are not run
    if not votes:                                                      # no type mode answered: every kind is a branch
        probs = {k: 1.0 / len(kinds) for k in kinds}
        origin = {k: StageSource("type", "prior:uniform", probs[k]) for k in kinds}
        return TypeStageResult(probs, origin, channels=ctx.channels, fits=ctx.fits)
    if len(votes) == 1:                                                # one source: its numbers, unchanged
        probs, mode, seconds = dict(votes[0].answer.probs), votes[0].mode, votes[0].seconds
    else:
        probs, mode, seconds = _fuse(votes, kinds), "fused", sum(v.seconds for v in votes)
    winner = max(probs, key=probs.get)
    # the winner's subclass: from the vote that contributed most to it and knows subclasses
    subclass, source = None, None
    for v in sorted(votes, key=lambda v: -v.weight * v.answer.probs.get(winner, 0.0)):
        subclass = v.answer.subclass_of(winner)
        if subclass:
            source = v.mode
            break
    detail = dict(entries=[v.as_dict(winner) for v in votes], subclass=subclass, subclass_source=source)
    origin = {k: StageSource("type", mode, float(p), float(seconds), detail=detail) for k, p in probs.items()}
    return TypeStageResult(probs, origin, votes, subclass, source, mode, ctx.channels, ctx.fits)


@timed_fn("type stage")
def type_stage(target_channels, spec, registry=None, *, field=None, models=None, kinds: Sequence[str] = KINDS,
               scope: Optional[np.ndarray] = None) -> TypeStageResult:
    """The type stage alone (every type entry in cost order, gated, fused): the parent probabilities and the winner's
    subclass for a target - what the add-cut loop asks before it chooses a first-add sub-loop. Pass the result to
    run_cascade(type_result=...) on the same target to skip the stage there."""
    registry = load_default_registry() if registry is None else registry
    ctx = _Context(target_channels, spec, field, models, scope)
    if not ctx.occ.any():
        return TypeStageResult({}, {}, channels=ctx.channels)
    return _type_stage(ctx, registry, kinds)


def filter_type_entries(registry, source: str = "registry") -> list:
    """The registry with its type entries restricted to one source: "registry" (all), "descriptors" (mode "descriptor"
    only), "cnn" (mode "cnn" only). Analytic and prior type entries stay in every choice (network-free fallbacks);
    the other stages are untouched."""
    keep = {"registry": None, "descriptors": ("descriptor", "analytic", "prior"), "cnn": ("cnn", "analytic", "prior")}
    if source not in keep:
        raise ValueError(f"type source must be one of {tuple(keep)}, got {source!r}")
    if keep[source] is None:
        return list(registry)
    return [e for e in registry if e.stage != "type" or e.mode in keep[source]]


def _pose_stages(ctx: _Context, registry, kind: str):
    """(Primitive with sources, iou) of one branch after ROTATION, DIMENSIONS and POSITION, or None."""
    chosen: Dict[str, Tuple[_Pose, StageSource]] = {}
    for stage in ("rotation", "dimensions", "position"):
        entries = _entries_for(registry, stage, kind) or [_Entry(f"{kind}-default", stage, (kind,), "analytic", None, cost=0.0)]
        best = None
        for entry in entries:
            t0 = time.time()
            if entry.mode == "cnn":
                pose = _cnn_pose(entry, ctx, kind, stage)
                impl = entry.id
            else:
                impl = _impl(entry, kind)
                pose = _estimate(ctx, kind, impl)
            if pose is None:
                continue
            src = StageSource(stage, _mode_label(entry, impl), float(pose.iou), float(time.time() - t0))
            if best is None or pose.iou > best[0].iou:
                best = (pose, src)
            if entry.gate is not None and pose.iou >= entry.gate:
                break
        if best is None:
            return None
        chosen[stage] = best
    rot, dim, pos = chosen["rotation"][0].prim, chosen["dimensions"][0].prim, chosen["position"][0].prim
    prim = Primitive(kind=kind, quaternion=rot.quaternion, center_mm=pos.center_mm, dims_mm=dict(dim.dims_mm), u=rot.u, v=rot.v)
    if kind == "box" and chosen["dimensions"][0].impl != chosen["rotation"][0].impl:
        prim = _remeasure_box(ctx, prim)
    if ctx.admit(prim) is None:                                        # a torus put together from several modes
        return None
    return prim, [chosen[s][1] for s in ("rotation", "dimensions", "position")]


def _cnn_pose(entry, ctx: _Context, kind: str, stage: str) -> Optional[_Pose]:
    """A CNN pose entry: `models[entry.id]` as a callable (in-scale occupancy, kind) -> Primitive. The existing rotation and
    dimension nets are not wired here: the inventory found none that beats the analytic modes (plan, Phase 0)."""
    fn = ctx.models.get(entry.id) if isinstance(ctx.models, dict) else None
    if fn is None or not callable(fn):
        return None
    x = inscale_occupancy(ctx, mode="fixed")                            # a pose net works in mm: never rescaled
    prim = ctx.admit(fn(x, kind)) if x is not None else None
    return _Pose(prim, ctx.iou(prim), entry.id) if prim is not None else None


def _remeasure_box(ctx: _Context, prim: Primitive) -> Primitive:
    from reco_engine.result_to_program import rotation_matrix
    R = rotation_matrix(prim)
    pts = ctx.centres[ctx.occ] @ R
    lo, hi = pts.min(axis=0) - ctx.pitch / 2, pts.max(axis=0) + ctx.pitch / 2
    h = (hi - lo) / 2.0
    return Primitive(kind="box", quaternion=prim.quaternion, center_mm=tuple(float(x) for x in R @ ((lo + hi) / 2.0)),
                     dims_mm=dict(x_half=float(h[0]), y_half=float(h[1]), z_half=float(h[2])))


@timed_fn("cascade")
def run_cascade(target_channels, spec, registry=None, *, field=None, top_k: int = 2, models=None,
                scope: Optional[np.ndarray] = None, pass_iou: float = PASS_IOU, kinds: Sequence[str] = KINDS,
                field_refine: bool = True, type_result: Optional[TypeStageResult] = None,
                part_diagonal: Optional[float] = None) -> Primitive:
    """One primitive for the target (see the module docstring). `scope`: where the IoU is measured (voxels outside it are
    free); `kinds`: the kinds a branch may take; `field_refine` False skips the exact-field refinement; `type_result`: a
    `type_stage` result made for this very target (same channels array) replaces the type stage here. `part_diagonal`:
    the part's bounding-box diagonal for the torus guard (default: the field's, else the grid's) - a torus the guard
    rejects (reco_engine/torus_guard.py) is no answer of its mode, so its branch falls to the next mode or kind."""
    registry = load_default_registry() if registry is None else registry
    ctx = _Context(target_channels, spec, field, models, scope, part_diagonal)
    if not ctx.occ.any():
        return Primitive(kind=UNKNOWN)
    if type_result is not None and type_result.channels is ctx.channels and type_result.probs:
        ctx.fits.update(type_result.fits)                              # the analytic fits the type stage already made
        ts = type_result
    else:
        ts = _type_stage(ctx, registry, kinds)
    probs, type_src = ts.probs, ts.origin
    ranked = sorted(probs, key=lambda k: -probs[k])[:max(1, top_k)]

    branches: List[Tuple[float, float, Primitive]] = []
    for kind in ranked:
        done = _pose_stages(ctx, registry, kind)
        if done is None:
            prim, sources, iou = None, [], 0.0
        else:
            prim, sources = done
            iou = ctx.iou(prim)
        if field_refine:
            t0 = time.time()
            fitted = _fit_field(ctx, kind, hint=prim)
            if fitted is not None:
                f_iou = ctx.iou(fitted)
                if prim is None or f_iou >= iou - FIELD_PREFERENCE:
                    secs = float(time.time() - t0)
                    prim, iou = fitted, f_iou
                    sources = [StageSource(s, "field_fit", float(f_iou), secs) for s in ("rotation", "dimensions", "position")]
        if prim is None:
            continue
        prim.sources = [type_src[kind]] + sources
        conf = 1.0
        for s in prim.sources:
            conf *= max(0.0, min(1.0, s.confidence))
        prim.confidence = float(conf)
        prim.residual = ctx.unexplained(prim)
        branches.append((iou, ctx.iou_strict(prim), prim))

    branches.sort(key=lambda t: (-round(t[0], 2), -t[1]))
    if not branches or branches[0][0] < pass_iou:
        return Primitive(kind=UNKNOWN, sources=[type_src[k] for k in ranked if k in type_src],
                         alternatives=[p for _, _, p in branches], residual=1.0)
    best = branches[0][2]
    best.alternatives = [p for _, _, p in branches[1:]]
    return best


def branch_iou(prim: Primitive, target_channels, spec, scope: Optional[np.ndarray] = None) -> float:
    """The score run_cascade ranks branches by, for a caller that wants it back. It has no caller, in the
    package or the tests (checked 2026-10-01)."""
    return _Context(target_channels, spec, None, None, scope).iou(prim)


MIN_ARC_GAP = 0.2                   # rad: an angular gap in the samples smaller than this is no gap - the cover is a full cylinder


def bounding_cylinder(target_channels, spec, field=None) -> Optional[Primitive]:
    """The covering cylinder of the input: axis and frame from the exact-field cylinder fit (reco_field.fit.fit_cylinder
    on the input's surface samples), then radius, length and sector stretched until they contain EVERY surface sample -
    the counterpart of `oriented_bounding_box` for a round part. A sector (u) when the samples leave an angular gap of
    MIN_ARC_GAP or more around the axis (the liner, a half-pipe: 0..180 deg), else a full cylinder. None without a field
    or when the fit fails. Sources "prior:bounding_cylinder"."""
    if field is None:
        return None
    from reco_field import fit as fitmod
    samples = _field_samples(field)
    # the axis from the strict RANSAC hypothesis when there is one (exact on the true cylinder's own points; the
    # least-squares fit is pulled by other faces - a flat pulled cy_flat's axis 0.054 off, 2026-09-26), else the fit
    hyps = fitmod._cylinder_ransac(samples, STRICT_TOLERANCE * float(samples.extent), n_hyp=600, top=1)
    if hyps:
        R, centre0 = np.asarray(hyps[0][0], float), np.asarray(hyps[0][1], float)
    else:
        try:
            res = fitmod.fit_cylinder(samples)
        except Exception:
            return None
        if res is None or res.frame is None or res.centre is None or res.half is None:
            return None
        R, centre0 = np.asarray(res.frame, float), np.asarray(res.centre, float)
    res = type("Fit", (), dict(frame=R, centre=centre0))
    local = (np.asarray(samples.surface, float) - np.asarray(res.centre, float)) @ R
    r = np.hypot(local[:, 0], local[:, 1])
    radius = float(r.max())
    z0, z1 = float(local[:, 2].min()), float(local[:, 2].max())
    if radius <= 0 or z1 <= z0:
        return None
    u = None
    ring = r > 0.05 * radius                                             # angles near the axis say nothing
    if ring.sum() >= 3:
        ang = np.sort(np.mod(np.arctan2(local[ring, 1], local[ring, 0]), 2 * np.pi))
        gaps = np.diff(np.concatenate([ang, ang[:1] + 2 * np.pi]))
        k = int(np.argmax(gaps))
        if gaps[k] >= MIN_ARC_GAP:
            start = float(ang[(k + 1) % len(ang)])
            u = (start, start + float(2 * np.pi - gaps[k]))
    centre = np.asarray(res.centre, float) + R @ np.array([0.0, 0.0, (z0 + z1) / 2.0])
    prim = Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(float(x) for x in centre),
                     dims_mm=dict(radius=radius, half_length=(z1 - z0) / 2.0), u=u)
    ctx = _Context(target_channels, spec, field, None, None)
    iou = ctx.iou(prim)
    prim.sources = [StageSource(st, "prior:bounding_cylinder", float(iou)) for st in ("type", "rotation", "dimensions",
                                                                                         "position")]
    prim.confidence = float(iou)
    prim.residual = ctx.unexplained(prim)
    return prim


def bounding_candidates(target_channels, spec, field=None) -> List[Tuple[str, Primitive]]:
    """Every mode of covering the whole input for the first add, each containing the input's surface samples:
    the oriented bounding box, the world-axis bounding box, the bounding cylinder (a sector when the samples leave an
    angular gap, and always also the full one), the bounding sphere, and box & full cylinder (a round part with flats, a
    half-pipe's outside). Graded by the residual loop on how their surfaces lie against the input's
    (residual_loop._grade_cover)."""
    out: List[Tuple[str, Primitive]] = []
    obb = oriented_bounding_box(target_channels, spec, field=field)
    if obb is not None:
        out.append(("oriented box", obb))
    if field is not None:
        pts = np.asarray(_field_samples(field).surface, float)
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        out.append(("world-axis box", Primitive(kind="box", center_mm=tuple(float(x) for x in (lo + hi) / 2),
                                                dims_mm=dict(zip(("x_half", "y_half", "z_half"),
                                                                 (float(v) for v in (hi - lo) / 2))))))
        cyl = bounding_cylinder(target_channels, spec, field=field)
        if cyl is not None:
            if cyl.u:
                out.append(("bounding cylinder sector", cyl))
            full = Primitive(kind="cylinder", quaternion=cyl.quaternion, center_mm=cyl.center_mm,
                             dims_mm=dict(cyl.dims_mm), u=None)
            out.append(("bounding cylinder", full))
            if obb is not None:
                out.append(("box & cylinder", Primitive(kind="composite", center_mm=obb.center_mm, dims_mm={},
                                                        parts=[obb, full], combine="common", label="box & cylinder")))
        try:
            from reco_field import fit as fitmod
            sph = fitmod.fit_sphere(_field_samples(field))
            if sph is not None and sph.centre is not None:
                c = np.asarray(sph.centre, float)
                out.append(("bounding sphere", Primitive(kind="sphere", center_mm=tuple(float(x) for x in c),
                                                         dims_mm=dict(radius=float(np.linalg.norm(pts - c, axis=1).max())))))
        except Exception:
            pass
    for name, p in out:
        p.label = p.label or name
        if not p.sources:
            p.sources = [StageSource(st, f"prior:{name.replace(' ', '_')}", 1.0) for st in ("type", "rotation",
                                                                                              "dimensions", "position")]
    return out


STRICT_TOLERANCE = 0.001            # x the samples' extent: the RANSAC tolerance of the strict cylinder candidates


def strict_cylinders(channels, spec, field, scope=None, top: int = 3, n_hyp: int = 600) -> List[Primitive]:
    """Cylinder candidates straight from the RANSAC stage of reco_field.fit on the input's surface samples near the target,
    at a STRICT tolerance (STRICT_TOLERANCE x their extent, a quarter of the fit's default) and without the least-squares
    refinement over all samples. Where one surface of a leftover is a true cylinder but the leftover also holds other
    geometry (the liner's bore with the pockets recessed into it), the refined fit is a compromise and this is the
    cylinder itself - measured on the liner: default tolerance -> R 142-151, refined R 116.6 at y -25.5; strict -> R 125.2
    at (0.6, -0.4), the drawing's R 125 on the axis. Full cylinders, length from their inliers; sources
    "analytic:strict_cylinder"."""
    if field is None:
        return []
    from reco_field import fit as fitmod
    ctx = _Context(channels, spec, field, None, scope)
    s = ctx.samples()
    if s is None or len(s.surface) < MIN_FIELD_POINTS:
        return []
    tol = STRICT_TOLERANCE * float(s.extent)
    out = []
    for R, centre, radius, inliers, half_length in fitmod._cylinder_ransac(s, tol, n_hyp=n_hyp, top=top):
        if radius <= 0 or half_length <= 0:
            continue
        p = Primitive(kind="cylinder", quaternion=quaternion_of(np.asarray(R, float)),
                      center_mm=tuple(float(x) for x in centre),
                      dims_mm=dict(radius=float(radius), half_length=float(half_length)))
        p.sources = [StageSource(st, "analytic:strict_cylinder", float(inliers) / len(s.surface))
                     for st in ("type", "rotation", "dimensions", "position")]
        p.confidence = float(inliers) / len(s.surface)
        out.append(p)
    return out


def strict_planes(channels, spec, field, scope=None, top: int = 3, n_hyp: int = 300, seed: int = 0) -> List[Primitive]:
    """Half-space cut candidates from the input's surface samples near the target: RANSAC planes (a sample point and its
    normal; inliers within STRICT_TOLERANCE x the extent AND with normals within ~18 deg), the best `top` distinct ones.
    Kind "halfspace", local +z = the surface's outward normal, so as a cut it removes everything on the outside of that
    face (a flat on a sphere or cylinder, a chamfer, a trimmed end). Sources "analytic:strict_plane"."""
    if field is None:
        return []
    ctx = _Context(channels, spec, field, None, scope)
    s = ctx.samples()
    if s is None or len(s.surface) < MIN_FIELD_POINTS:
        return []
    P, N = np.asarray(s.surface, float), np.asarray(s.normals, float)
    tol = STRICT_TOLERANCE * float(s.extent)
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(P), size=min(n_hyp, len(P)), replace=False)
    hyps = []
    for i in picks:
        n = N[i] / max(np.linalg.norm(N[i]), 1e-12)
        inl = (np.abs((P - P[i]) @ n) < tol) & (N @ n > 0.95)
        hyps.append((int(inl.sum()), n, inl))
    hyps.sort(key=lambda h: -h[0])
    out, taken = [], []
    for count, n, inl in hyps:
        if count < 12 or len(out) >= top:
            break
        centre = P[inl].mean(axis=0)
        fitted = np.linalg.svd(P[inl] - centre, full_matrices=False)[2][2]    # refit the normal on the inliers
        n = fitted if fitted @ n > 0 else -fitted
        if any(abs(n @ m) > 0.995 and abs((centre - c) @ m) < 2 * tol for m, c in taken):
            continue
        taken.append((n, centre))
        u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        R = np.stack([u, np.cross(n, u), n], axis=1)
        p = Primitive(kind="halfspace", quaternion=quaternion_of(R), center_mm=tuple(float(x) for x in centre), dims_mm={})
        p.confidence = float(count) / len(P)
        p.sources = [StageSource(st, "analytic:strict_plane", p.confidence) for st in ("type", "rotation", "dimensions",
                                                                                         "position")]
        out.append(p)
    return out


def composites(boxes: Sequence[Primitive], cylinders: Sequence[Primitive], parallel: float = 0.98) -> List[Primitive]:
    """Two-primitive candidates judged as one: for every box and every cylinder whose axis runs along one of the box's
    axes, box & cylinder (a pocket or slot with a rounded end, the liner's "Quader mit Radien") and box - cylinder (a
    fillet: the corner block without the round - D-001). The cylinder is stretched along its axis to span the box, so
    the pair meets over the box's whole length. Spheres in `cylinders` pair with every box, as they are: box & sphere,
    box - sphere (the corner region above a dome, bs_box_dome). Kind "composite"."""
    from reco_engine.result_to_program import rotation_matrix
    out = []
    for b in boxes:
        if b.kind != "box":
            continue
        Rb, cb = rotation_matrix(b), np.asarray(b.center_mm, float)
        half = np.array([b.dims_mm["x_half"], b.dims_mm["y_half"], b.dims_mm["z_half"]])
        for c in cylinders:
            if c.kind == "sphere" and not c.u and not c.v:
                for combine, name in (("common", "box & sphere"), ("cut", "box - sphere")):
                    p = Primitive(kind="composite", center_mm=b.center_mm, dims_mm={}, parts=[b, c], combine=combine,
                                  label=name)
                    p.confidence = min(b.confidence or 0.0, c.confidence or 0.0)
                    p.sources = [StageSource(st, f"analytic:composite_{combine}", p.confidence)
                                 for st in ("type", "rotation", "dimensions", "position")]
                    out.append(p)
                continue
            if c.kind != "cylinder" or c.u:
                continue
            axis = rotation_matrix(c)[:, 2]
            k = int(np.argmax(np.abs(Rb.T @ axis)))
            if abs(Rb[:, k] @ axis) < parallel:
                continue
            centre = np.asarray(c.center_mm, float)
            centre = centre + axis * float((cb - centre) @ axis)             # centred on the box along the axis
            long = Primitive(kind="cylinder", quaternion=c.quaternion, center_mm=tuple(float(x) for x in centre),
                             dims_mm=dict(radius=float(c.dims_mm["radius"]), half_length=float(half[k]) * 1.02))
            for combine, name in (("common", "box & cylinder"), ("cut", "box - cylinder")):
                p = Primitive(kind="composite", center_mm=b.center_mm, dims_mm={}, parts=[b, long], combine=combine,
                              label=name)
                p.confidence = min(b.confidence or 0.0, c.confidence or 0.0)
                p.sources = [StageSource(st, f"analytic:composite_{combine}", p.confidence)
                             for st in ("type", "rotation", "dimensions", "position")]
                out.append(p)
    return out


def strict_spheres(channels, spec, field, scope=None) -> List[Primitive]:
    """The best sphere of the input's surface samples near the target at the strict tolerance (reco_field.fit's
    RANSAC on pairs of oriented points, no refinement) - the dome of bs_box_dome, a spherical pocket. [] if none."""
    if field is None:
        return []
    from reco_field import fit as fitmod
    ctx = _Context(channels, spec, field, None, scope)
    s = ctx.samples()
    if s is None or len(s.surface) < MIN_FIELD_POINTS:
        return []
    best = fitmod._sphere_ransac(s, STRICT_TOLERANCE * float(s.extent), n_hyp=600)
    if best is None:
        return []
    centre, radius, inliers = best
    p = Primitive(kind="sphere", center_mm=tuple(float(x) for x in centre), dims_mm=dict(radius=float(radius)))
    p.confidence = float(inliers) / len(s.surface)
    p.sources = [StageSource(st, "analytic:strict_sphere", p.confidence) for st in ("type", "rotation", "dimensions",
                                                                                      "position")]
    return [p]


def near_surface_agreement(prim: Primitive, channels, spec, field, scope=None) -> Optional[float]:
    """The share of the input's surface samples NEAR the target (within a voxel of it) that lie on the primitive's surface
    (tolerance 2 x STRICT_TOLERANCE x their extent). None without a field. The liner's bore component: 44.5 % on the
    true bore, 14.4 % on the refined compromise (1 mm)."""
    if field is None:
        return None
    ctx = _Context(channels, spec, field, None, scope)
    s = ctx.samples()
    if s is None or not len(s.surface):
        return None
    d = primitive_to_sp(prim).sdf(np.asarray(s.surface, float))
    return float(np.mean(np.abs(d) <= 2 * STRICT_TOLERANCE * float(s.extent)))


def surface_agreement(prim: Primitive, field, tolerance: float) -> Tuple[float, float]:
    """(on, inside): the share of the input's surface samples lying on the primitive's own surface (|sdf| <= tolerance)
    and the share it contains (sdf <= tolerance). A cover that follows the part's surface - a half-pipe's round outside
    by a cylinder - scores high on `on`; one that only touches it along lines (its bounding box) scores low."""
    pts = np.asarray(_field_samples(field).surface, float)
    d = primitive_to_sp(prim).sdf(pts)
    return float(np.mean(np.abs(d) <= tolerance)), float(np.mean(d <= tolerance))


@timed_fn("bounding box")
def oriented_bounding_box(target_channels, spec, field=None) -> Optional[Primitive]:
    """The covering oriented bounding box of the target (face-normal frame; exact surface extents when `field` is given),
    with its sources marked "prior:obb" - the add-cut loop's first add when no primitive explains the whole input."""
    ctx = _Context(target_channels, spec, field, None, None)
    prim = _obb(ctx)
    if prim is None:
        return None
    iou = ctx.iou(prim)
    prim.sources = [StageSource(s, "prior:obb", float(iou)) for s in ("type", "rotation", "dimensions", "position")]
    prim.confidence = float(iou)
    prim.residual = ctx.unexplained(prim)
    return prim
