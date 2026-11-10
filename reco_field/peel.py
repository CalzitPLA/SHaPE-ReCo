#!/usr/bin/env python3
"""
The peeling loop: reconstruct an STL/STEP model as a CSG program of analytic primitives from its closest-point field.

    field = load_field("part.stl")
    result = peel(field)
    result.ops                       # [base, cut, cut, add, clip, ...]: what was fitted and how it enters the model
    result.model                     # the resulting solid (an SDF primitive: model.sdf(points))
    chamfer(result.model, field)     # distance of the model to the input and back, in the model's units

1. Fit the best primitive to the whole part (surface points plus the off-surface distances of the field): the BASE.
2. Take the surface points the model does not explain. Group them into connected clusters, biggest first.
3. The unexplained points on the inside of the model (its SDF is negative there) are the walls of pockets, holes and steps: group
   them and fit each cluster as a CUTTER (a flat cluster becomes a CLIP by a half space). Those outside are added features: fit
   them as ADDITIONS. The two sides are grouped separately, so a cluster never mixes them. A cluster is fitted from its own
   surface points and normals.
4. A cutter's hidden faces are not observable. A box or cylinder end with no surface data on it is OPEN: it is extended out of the
   body (cuts) or slightly into it (additions) so that the boolean is clean.
5. Accept the operation if it explains more of the input surface AND does not raise the share of the model's surface that lies off
   the input (`invented`) by more than INVENTED_SLACK; stop when enough is explained, MAX_OPS is reached or nothing helps.

The grade is geometric, in both directions and in the input's units: `chamfer` measures the model's surface against the input
mesh and back, so it also catches geometry the model invents (which a one-way explained-share cannot).
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field import fit as fitmod  # noqa: E402
from reco_field.fit import FitResult, Samples, make_samples, subset, tolerance  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402


MAX_OPS = 24                # operations at most, the base included (a part built from many features needs many)
INVENTED_SLACK = 0.02       # an operation may raise the invented share of the model by at most this much
SPLIT_RATIO = 5.0           # _split_multibody: biggest sub-body >= this x the second-biggest -> treat as one (noise), else split

# resolutions for predict_ops_budget's coarse-genus sweep: matches what docs/FRAME_FAILURE.md measured on the real
# frame part (genus 1 up to 16, a stable plateau at 9 from 24 to 96, the exact 17 only at 128)
BUDGET_RESOLUTIONS = (8, 16, 24, 32, 48, 64, 96, 128)


@dataclass
class OpsBudget:
    """What `predict_ops_budget` found, kept around so a caller (the GUI, `peel_case.py`) can show the reasoning, not
    just the final number."""
    sweep: Dict[int, float]           # resolution -> coarse genus, as measured (stops early - see `predict_ops_budget`)
    exact_genus: float                # field.genus() - cheap always, but often an OVERESTIMATE of what a practical
                                       # reconstruction should budget for (see predict_ops_budget's docstring)
    plateau_genus: float              # the number the budget is actually based on: the largest coarse genus reached
    plateau_resolution: int           # the finest resolution swept before stopping
    base_ops: int
    margin: int
    max_ops: int                      # base_ops + ceil(plateau_genus) + margin, capped at `upper_cap`
    reason: str                       # why the sweep stopped where it did


def predict_ops_budget(field: SurfaceField, base_ops: int = 1, margin: int = 4,
                       resolutions: Sequence[int] = BUDGET_RESOLUTIONS, max_seconds: float = 20.0,
                       upper_cap: int = 60) -> OpsBudget:
    """Predict a practical MAX_OPS for `field` from its genus, so the loop (and a GUI showing it beforehand) gets a
    value tailored to the part in front of it instead of the fixed default.

    Not from the EXACT genus, which on a real mechanical part is often inflated by many small real holes (bolt holes,
    rivets) that are not what "how many big independent features does this part have" usually means: budgeting one
    operation per one of those would demand far more operations than is practical, or useful, for a first reconstruction
    (`docs/FRAME_FAILURE.md`: a real part reads as genus 1 - a plain frame - at a coarse voxel scale, a stable 9 at a
    medium scale, and its exact genus of 17 only once voxels approach the smallest real feature's own size). Instead,
    this sweeps `resolutions` (coarsest first) with `SurfaceField.coarse_genus`, stopping as soon as either the running
    plateau has caught up with the cheap exact genus (nothing coarser is being missed - the common case for a part with
    few or no real tunnels, so this usually returns after just one or two resolutions) or `max_seconds` of wall clock
    have passed (a practical cap for a large or intricate mesh, where the fine end of the sweep can itself take a
    minute or more - see `coarse_genus`). `max_ops = base_ops + ceil(plateau_genus) + margin`, capped at `upper_cap`:
    one base primitive, one cut per independent feature resolved at that plateau, and a fixed margin for operations
    genus does not count at all (additions, blind cuts/pockets, plane clips - none of which change the topology; margin
    4 covers every one of this project's 41 example parts, whose "ops beyond base+genus" is at most 2 -
    `docs/data/peel_examples.json`). `upper_cap` defaults to 60, not the old fixed `MAX_OPS` (24): capping a
    GENUS-DRIVEN budget at exactly the fixed number it replaces would silently reintroduce the old ceiling for
    precisely the high-genus parts this function exists for (measured up to 60 without trouble on the frame case study,
    `docs/FRAME_FAILURE.md`) - it is still a safety cap, just one set well above where the fixed default used to bite,
    not equal to it.

    Known gap, not fixed (`coarse_genus`'s own docstring, `tests/test_closest_point_field.py`): a part whose only
    non-trivial topology is a handle sealed entirely inside a cavity never lets the sweep's `best >= exact` stopping
    condition fire, so every call for such a part pays the full `max_seconds` and still under-predicts."""
    exact = field.genus()
    if exact != exact:                                                            # NaN: not closed, genus undefined
        return OpsBudget({}, exact, float("nan"), 0, base_ops, margin, upper_cap, "not closed: falling back to the default")
    sweep, best, t0, reason, r = {}, 0.0, time.time(), "swept every resolution", resolutions[0]
    for r in resolutions:
        sweep[r] = field.coarse_genus(r)
        best = max(best, sweep[r])
        if best >= exact:
            reason = "matched the exact genus"
            break
        if time.time() - t0 > max_seconds:
            reason = f"stopped after {max_seconds:.0f}s of sweeping"
            break
    max_ops = min(base_ops + int(np.ceil(best)) + margin, upper_cap)
    return OpsBudget(sweep, exact, best, r, base_ops, margin, max_ops, reason)


@dataclass
class Op:
    op: str                          # "base" | "add" | "cut" | "clip"
    fit: FitResult
    primitive: sp.Primitive          # as entered into the model (a cutter with its open ends extended)
    n_points: int                    # surface points it was fitted to
    explained_after: float           # share of the input surface the model explains after this operation


@dataclass
class PeelResult:
    ops: List[Op]
    model: sp.Primitive
    explained: float
    history: List[float]
    tol: float
    stopped: str = ""                # why the loop ended: "target reached" | "max operations" | "nothing helps"
    budget: Optional[OpsBudget] = None  # set when max_ops was predicted rather than given explicitly (see peel_steps)


@dataclass
class PeelStep:
    """One observable stage of the loop (`peel_steps`); what a front end draws. Arrays and primitives are never mutated afterwards.

    stage: "field" (samples drawn), "base" (whole-part fit), "cluster" (unexplained surface grouped, one chosen), "candidate" (a
    fit of that cluster, folded into a trial model), "accepted" / "rejected" (the trial's verdict), "grade" (chamfer), "done"."""
    stage: str
    phase: str                       # "base" | "peel" | "done"
    iteration: int                   # index of the operation being worked on (0 = the base)
    message: str
    samples: Optional[Samples] = None
    field: Optional[SurfaceField] = None
    vector_points: Optional[np.ndarray] = None       # a subset of the field to draw: points x ...
    vectors: Optional[np.ndarray] = None             # ... and V(x) = c(x) - x at them
    tol: float = 0.0
    ops: Tuple[Op, ...] = ()                         # the operations accepted so far
    model: Optional[sp.Primitive] = None             # the model after them
    explained: float = 0.0
    unexplained: Optional[np.ndarray] = None         # indices into samples.surface the model does not explain
    cluster: Optional[np.ndarray] = None             # indices into samples.surface of the cluster being fitted
    inside: Optional[bool] = None                    # the cluster lies in the material (a cutter) rather than outside (an addition)
    candidate: Optional[Op] = None                   # the operation being tried
    trial: Optional[sp.Primitive] = None             # the model with it folded in
    trial_explained: Optional[float] = None
    invented: Optional[float] = None                 # share of the model's surface farther than 2 tol from the input (see `invented`)
    trial_invented: Optional[float] = None           # the same for the trial model
    grade: Optional[Dict[str, Dict[str, float]]] = None
    result: Optional["PeelResult"] = None
    budget: Optional[OpsBudget] = None                # set on the "field" step when max_ops was predicted, not given
    freecad: Optional[Dict] = None                    # verify_freecad only: this step's real-CSG check (see peel_steps)


def fold(model: Optional[sp.Primitive], op: str, prim: sp.Primitive) -> sp.Primitive:
    if op == "base":
        return prim
    if op == "add":
        return sp.Union(model, prim)
    if op == "cut":
        return sp.Subtraction(model, prim)
    if op == "clip":
        return sp.Intersection(model, prim)
    raise ValueError(op)


def build_model(ops: Sequence[Op]) -> sp.Primitive:
    model = None
    for o in ops:
        model = fold(model, o.op, o.primitive)
    return model


# ------------------------------------------------------------------------------------------------ open-end extension

def _extend(fit: FitResult, pts: np.ndarray, normals: np.ndarray, extent: float, mode: str) -> sp.Primitive:
    """Box, cylinder and cone: an end (a face of the box, a cap of the cylinder or cone) that carries no surface data is open.
    It is set to where the data stops and pushed from there: far out of the body for a cutter, a little into it for an addition.
    A cone keeps its surface when extended (the radius follows the slope) and stops at its apex. Sphere and torus have no ends."""
    if fit.kind not in ("box", "cylinder", "cone"):
        return fit.primitive
    tol = fit.tol
    R, c = fit.frame, fit.centre
    local = (pts - c) @ R
    if fit.kind == "box":
        lo, hi = -fit.half.copy(), fit.half.copy()
        axes = (0, 1, 2)
    else:
        lo, hi = np.array([-fit.half[0], -fit.half[1], -fit.half[2]]), fit.half.copy()
        if fit.kind == "cone":
            lo, hi = np.array([0.0, 0.0, -fit.half[2]]), np.array([0.0, 0.0, fit.half[2]])
        axes = (2,)
    far = extent if mode == "cut" else 0.05 * extent
    for a in axes:
        for end, sign in ((hi, 1), (lo, -1)):
            # a face here = points ON the end plane whose normals run ALONG the axis (lateral points near the end are not a cap)
            on_face = (np.abs(local[:, a] - end[a]) < 2 * tol) & (np.abs(normals @ R[:, a]) > 0.9)
            if on_face.mean() < 0.02:                                            # no face here: open
                # from where the DATA stops, not from the fit's end: a face without data is a free parameter of the
                # fit, and where the optimiser leaves it depends on the machine (the buried face of the add in
                # test_union_of_two_boxes: z = -1.63 on the laptop, -4.47 on workstation, same samples, same versions of
                # numpy and scipy; on workstation the add then stuck out of the base and was rejected, 2026-09-27)
                data = local[:, a].max() if sign > 0 else local[:, a].min()
                end[a] = data + sign * far
    if fit.kind == "cone":
        r1, r2, h = fit.half
        slope = (r2 - r1) / (2 * h)
        radius = lambda z: 0.5 * (r1 + r2) + slope * z                          # noqa: E731
        z0, z1 = lo[2], hi[2]
        if slope > 1e-9:                                                        # the apex is at the bottom: do not go past it
            z0 = max(z0, -0.5 * (r1 + r2) / slope)
        elif slope < -1e-9:
            z1 = min(z1, -0.5 * (r1 + r2) / slope)
        prim = sp.Cone(max(radius(z0), 0.0) + 1e-6, max(radius(z1), 0.0) + 1e-6, z1 - z0, fit.u or sp.FULL)
        return sp.Placed(prim, rotation=R, translation=c + R @ np.array([0.0, 0.0, z0]))
    centre = c + R @ ((lo + hi) / 2)
    half = (hi - lo) / 2
    if fit.kind == "box":
        return sp.Placed(sp.Box(half), rotation=R, translation=centre)
    h = half[2]
    return sp.Placed(sp.Cylinder(abs(fit.half[0]), 2 * h, fit.u or sp.FULL), rotation=R, translation=centre - R @ np.array([0.0, 0.0, h]))


# --------------------------------------------------------------------------------------------------------- the loop

def _clusters(points: np.ndarray, radius: float) -> List[np.ndarray]:
    """Index groups of points connected by chains of neighbours closer than `radius`, biggest first."""
    if len(points) == 0:
        return []
    pairs = np.array(list(cKDTree(points).query_pairs(radius)))
    n = len(points)
    if len(pairs) == 0:
        return [np.array([i]) for i in range(n)]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(g, directed=False)
    groups = [np.nonzero(labels == k)[0] for k in np.unique(labels)]
    return sorted(groups, key=len, reverse=True)


def _split_multibody(idx: np.ndarray, points: np.ndarray, tight_radius: Optional[float], ratio: float) -> List[np.ndarray]:
    """`[idx]` unchanged, or `idx` broken into its separate bodies at a TIGHTER connectivity radius than the
    cluster it came from was grouped at - the "single or multibody" split (docs/FRAME_FAILURE.md): a cluster
    is one connected region only at the coarse radius `_side_clusters` groups with; some of what it merges
    together are several real, separate small features (or a real feature plus scattered noise) that only
    happen to chain together through in-between points closer than that coarse radius.

    `tight_radius=None` (the default and the recommended setting): scaled to the CLUSTER'S OWN point density
    (a few times its median nearest-neighbour spacing) rather than a fixed value - a fixed radius tight enough
    to separate small debris fragments is, wrongly, ALSO tight enough to fragment a large but genuinely
    continuous surface into many small pieces purely from ordinary sampling irregularity (measured while
    building this: the frame's own clean, well-fit 484-point flange cluster fragmented into dozens of
    20-30-point pieces at a fixed tight radius, indistinguishable by point count alone from a real multi-body
    cluster). Scaling to the cluster's own density keeps a continuous surface connected regardless of its
    size, while a genuinely separate body (a real gap larger than the local sampling spacing) still splits.

    Decided by the VOLUMETRIC relation between the biggest and second-biggest sub-body (point count, the same
    proxy `min_fraction`'s floor already uses): biggest >> second (`ratio`, default `SPLIT_RATIO`) means the
    second (and any smaller) sub-body is noise-scale debris next to one real dominant piece - proceed as one
    cluster (a robust fit/network can be expected to shrug off a little contamination); otherwise there are
    two or more comparably-sized real pieces merged together, which is what should be split apart and fit
    separately rather than asked to look like one primitive (or one Type-network answer) at all."""
    if tight_radius is None:
        if len(idx) < 4:
            return [idx]
        own_points = points[idx]
        dists, _ = cKDTree(own_points).query(own_points, k=2)          # k=1 is the point itself, distance 0
        tight_radius = 2.5 * float(np.median(dists[:, 1]))
    groups = _clusters(points[idx], tight_radius)
    if len(groups) <= 1:
        return [idx]
    if len(groups[0]) >= ratio * len(groups[1]):
        return [idx]
    return [idx[g] for g in groups]


def _mesh_component_samples(field: SurfaceField, face_idx: np.ndarray, n_points: int, extent: float, seed: int) -> Samples:
    """A `Samples` of `n_points` drawn (barycentric, area-weighted - the same construction `fit.make_samples`
    uses for the whole mesh) from ONLY the triangles in `face_idx`, so a mesh-topology component with too few
    (or zero) of the original sparse sample points still gets a proper, dense sample of its own to fit."""
    rng = np.random.default_rng(seed)
    area = 0.5 * np.linalg.norm(np.cross(field._b[face_idx] - field._a[face_idx], field._c[face_idx] - field._a[face_idx]), axis=1)
    pick = rng.choice(len(face_idx), size=n_points, p=area / area.sum())
    f = face_idx[pick]
    r1, r2 = rng.random((n_points, 1)), rng.random((n_points, 1))
    flip = (r1 + r2) > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    surface = field._a[f] + (field._b[f] - field._a[f]) * r1 + (field._c[f] - field._a[f]) * r2
    return Samples(surface, field.face_normals[f], np.empty((0, 3)), np.empty(0), extent)


def _mesh_clusters(model: sp.Primitive, field: SurfaceField, tol: float, extent: float, n_points: int = 200,
                   min_faces: int = 3, seed: int = 0) -> List[Tuple[Samples, bool, int]]:
    """The "known CAD pipeline" alternative to `_side_clusters`: instead of Euclidean-distance-clustering a
    sparse SAMPLE of the surface (confounded by sampling density - a large but genuinely continuous region
    fragments just as easily as a real multi-feature one, `docs/FRAME_FAILURE.md`), classify every triangle of
    the FULL input mesh by its own centroid's SDF (unexplained + which side), then split that triangle subset
    into its TRUE topologically-connected components (shared edges, `trimesh.Trimesh.split` - exact, immune to
    sampling density since it uses the mesh's actual connectivity, not point proximity). Each component gets
    its own dense, freshly-drawn sample (`_mesh_component_samples`) rather than reusing whatever sparse sample
    points happened to land on it - a real small feature with zero of the original ~4000 samples on it is
    still found and fittable. Returns (Samples, inside, face count) biggest (by face count) first, `min_faces` and larger
    only (fewer than that cannot support a meaningful fit regardless of the primitive kind)."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=field.vertices, faces=field.faces, process=False)
    face_sdf = model.sdf(mesh.triangles_center)
    out = []
    for side in (True, False):
        face_mask = (np.abs(face_sdf) >= tol) & ((face_sdf < 0) == side)
        if not face_mask.any():
            continue
        candidate_faces = np.nonzero(face_mask)[0]
        sub = mesh.submesh([candidate_faces], only_watertight=False, append=False)[0]
        candidate_tree = cKDTree(mesh.triangles_center[candidate_faces])
        for comp in sub.split(only_watertight=False):
            if len(comp.faces) < min_faces:
                continue
            # map the component's own (submesh-local) faces back to ORIGINAL mesh face indices via each
            # face's centroid (submesh vertices are a copy, not shared indices with the original mesh); a tree,
            # not the full distance matrix: the liner's 147044 x 74370 faces would need 244 GiB
            orig = candidate_faces[candidate_tree.query(comp.triangles_center)[1]]
            samples = _mesh_component_samples(field, np.unique(orig), min(n_points, 50 * len(comp.faces)), extent, seed)
            out.append((samples, side, len(comp.faces)))
    return sorted(out, key=lambda t: t[2], reverse=True)


def _grid_centres(lo: np.ndarray, hi: np.ndarray, resolution) -> np.ndarray:
    """The centres of a `resolution`^3 (or `resolution[0] x resolution[1] x resolution[2]`) grid of cells over
    [lo, hi], as a (rx, ry, rz, 3) array - the same grid `input_occupancy` and `_octree_clusters` must share
    exactly (both call this, never build a grid by hand), or the XOR between input and model occupancy would
    compare cells that do not correspond to the same box in space."""
    res = np.broadcast_to(np.asarray(resolution, int), (3,))
    ax = [np.linspace(lo[i], hi[i], int(res[i]) + 1) for i in range(3)]
    step = np.array([a[1] - a[0] for a in ax])
    return np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)[:-1, :-1, :-1] + step / 2.0


def input_occupancy(field: SurfaceField, lo: np.ndarray, hi: np.ndarray, resolution: int) -> np.ndarray:
    """A `resolution`^3 occupancy grid (bool) of the TRUE input solid over the box [lo, hi]: exact point-in-mesh
    (`trimesh.Trimesh.contains`, needs a closed mesh - every part this loop runs on is, `docs/FRAME_FAILURE.md`),
    not a rasterised/filled voxelization, so it can be evaluated on any grid origin/pitch without alignment games.
    Computed once per `peel_steps` call (the input never changes) and passed into `_octree_clusters` every
    iteration; `resolution` in the 24-32 range costs well under a second even on a part the size of the frame.

    Fast and exact, but not what the rest of this project treats as the stable ground truth of "is there material
    here" - see `real_voxel_occupancy` for the project's own established voxelizer used instead, on request
    (`docs/FRAME_FAILURE.md`, "Voxels as the ground truth occupancy source")."""
    import trimesh
    mesh = trimesh.Trimesh(vertices=field.vertices, faces=field.faces, process=False)
    centres = _grid_centres(lo, hi, resolution)
    return mesh.contains(centres.reshape(-1, 3)).reshape(centres.shape[:3])


def real_voxel_occupancy(field: SurfaceField, lo: np.ndarray, hi: np.ndarray, resolution) -> np.ndarray:
    """A `resolution`^3 occupancy grid (bool) of the input solid over the box [lo, hi], the SAME grid as
    `input_occupancy` (`_grid_centres`), computed by the project's ground-truth voxelizer: scanline ray parity at
    the voxel centres (`dataset_gen.src.field_channels.parity_axes`, D-vox), not `trimesh.Trimesh.contains`.

    Replaces the earlier stltovoxel version, which built its own cube centred on the WORLD origin with a half side
    from the bbox diagonal: any part farther from the origin than that was clipped (a box at (100, 50, 20) got
    0 voxels, the frame 21 %; `.claude/guru/council/guru.md` P2), and it used the voxelizer D-vox retired. Every
    `octree_real_voxels` measurement made with that version is void (`docs/FRAME_FAILURE.md`). The grid rows are
    shifted by a tiny fraction of a voxel, as in `parity_occupancy`, so no row runs through a mesh vertex."""
    from src.field_channels import parity_axes
    c = _grid_centres(np.asarray(lo, float), np.asarray(hi, float), resolution)
    step = c[1, 1, 1] - c[0, 0, 0]
    return parity_axes(field.vertices, field.faces, c[:, 0, 0, 0],
                       c[0, :, 0, 1] + 1.2345e-4 * step[1], c[0, 0, :, 2] + 2.3456e-4 * step[2])


def _octree_leaves(occ: np.ndarray, min_voxels: int, min_fraction: float, max_depth: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """[(index_lo, index_hi)] (both int, half-open) of an occupancy octree's leaves: split `occ` into octants
    recursively, dropping an octant the moment it is empty (no material there at all - a genuinely different
    prune from `min_voxels`/`min_fraction`, which drop a THIN-but-nonempty octant only at the leaf, exactly
    `reco_engine/octree_search.py`'s `build_octree` on the `octree` branch, ported here for `reco_field`'s
    occupancy grids (input-solid XOR model, not a voxel-CNN grid) - same shape of algorithm, same pruning
    reasoning: an empty octant can never hold a feature, a small/sparse one is not a coherent feature."""
    root_lo, root_hi = np.zeros(3, int), np.array(occ.shape)
    leaves = []

    def descend(idx_lo, idx_hi, depth):
        size = idx_hi - idx_lo
        cell = occ[idx_lo[0]:idx_hi[0], idx_lo[1]:idx_hi[1], idx_lo[2]:idx_hi[2]]
        count = int(cell.sum())
        if count == 0:
            return
        is_leaf = depth >= max_depth or np.any(size <= 1) or count < min_voxels
        if is_leaf:
            fraction = count / float(np.prod(size))
            if depth == 0 or (count >= min_voxels and fraction >= min_fraction):
                leaves.append((idx_lo, idx_hi))
            return
        half = size // 2
        for dx in (0, half[0]):
            for dy in (0, half[1]):
                for dz in (0, half[2]):
                    offset = np.array([dx, dy, dz])
                    width = np.where(offset == 0, half, size - half)          # the upper octant covers the
                    lo2 = idx_lo + offset                                     # remainder, not `half` again -
                    hi2 = np.minimum(lo2 + np.maximum(width, 1), idx_hi)      # else an odd axis size loses
                    if np.any(hi2 <= lo2):                                    # its last voxel to neither child
                        continue
                    descend(lo2, hi2, depth + 1)

    descend(root_lo, root_hi, 0)
    return leaves


def _octree_clusters(model: sp.Primitive, points: np.ndarray, unexplained: np.ndarray, input_occ: np.ndarray,
                     grid_lo: np.ndarray, grid_hi: np.ndarray, max_depth: int = 1, min_voxels: int = 4,
                     min_fraction: float = 0.03) -> List[Tuple[np.ndarray, bool]]:
    """A volumetric alternative to `_side_clusters`: instead of grouping unexplained SURFACE points by Euclidean
    distance (sample-density confounded) or the mesh's own triangle connectivity (`_mesh_clusters` - exact, but
    fragments a part like the frame into two dozen slivers, most too small to pass `min_fraction`'s floor on
    their own), group by an OCCUPANCY OCTREE over the missing VOLUME - axis-aligned, coarse, and immune to both
    confounds, because a cell's occupancy count does not care how many surface points happened to land near it or
    how many disconnected triangle islands it straddles.

    The motivating observation (`docs/FRAME_FAILURE.md`): the frame's bounding box minus the frame's own material
    is still a simple, roughly rectangular volume (the central opening, plus the corners a box base always
    overshoots) - geometrically much simpler than its SURFACE, which a cutting operation chops into many small
    disconnected patches. Working on the volume's occupancy instead of the surface's connectivity is what lets a
    coarse, few-leaf octree describe it as one or two big regions rather than dozens of slivers.

    `input_occ` (see `input_occupancy`) is computed ONCE per `peel_steps` call and passed in unchanged every
    iteration; only the MODEL's occupancy is cheap to re-evaluate every time (`model.sdf` is analytic, no meshing
    or voxelizing needed). `add_occ = input_occ & ~model_occ` (material the input has that the model does not:
    additions), `cut_occ = model_occ & ~input_occ` (material the model has that the input does not: cutters) -
    the same inside/outside split `_side_clusters` makes, just on voxels instead of points. Each octree leaf
    becomes a cluster by SELECTING the existing sparse unexplained points that fall inside its box (no fresh
    sampling, unlike `_mesh_clusters` - there is already a surface sample everywhere real geometry is, since the
    octree only ever sees the volume that surface encloses or is missing from).

    `max_depth`/`min_voxels`/`min_fraction`: the same bounded-cost pruning `reco_engine/octree_search.py`'s
    `OctreeConfig` uses on the voxel-CNN route - `max_depth=2` on a `resolution`-per-axis grid means at most
    `resolution / 4` voxels per side at the finest leaf, coarse on purpose (avoiding hallucination means never
    asking about a region too small to be a real feature, not resolving every last one)."""
    grid_res = np.array(input_occ.shape)
    cell = (grid_hi - grid_lo) / grid_res
    model_occ = (model.sdf(_grid_centres(grid_lo, grid_hi, grid_res)) < 0).reshape(input_occ.shape)
    out = []
    for occ, inside in ((input_occ & ~model_occ, False), (model_occ & ~input_occ, True)):
        for idx_lo, idx_hi in _octree_leaves(occ, min_voxels, min_fraction, max_depth):
            lo = grid_lo + idx_lo * cell
            hi = grid_lo + idx_hi * cell
            p = points[unexplained]
            in_box = np.all((p >= lo) & (p <= hi), axis=1)
            side = (model.sdf(p) < 0) == inside
            sel = unexplained[in_box & side]
            if len(sel):
                out.append((sel, inside))
    return sorted(out, key=lambda c: len(c[0]), reverse=True)


def _side_clusters(model: sp.Primitive, points: np.ndarray, unexplained: np.ndarray, radius: float,
                   cuts_first: bool = False, split_multibody: bool = False, split_radius: Optional[float] = None,
                   split_ratio: float = SPLIT_RATIO) -> List[Tuple[np.ndarray, bool]]:
    """Clusters of the unexplained points as (indices into `points`, inside), biggest first (or, with
    `cuts_first`, every INSIDE/cutter cluster before any OUTSIDE/addition one, each group still biggest
    first - see `peel_steps`'s `bounding_box` docstring: a bounding-box base necessarily overshoots, and
    carving it down toward the true shape before trying to add anything avoids accepting an addition
    against a base that still has excess material an upcoming cut would have removed anyway). The points on
    each side of the model's surface are grouped on their own: those in the material (its SDF is negative)
    are the walls of pockets, holes and steps (cutters), those outside are added features. One cluster never
    mixes the two, so a part whose model is only roughly right (a frame's slab) still yields clean cutters
    and clean additions.

    `split_multibody`: additionally run `_split_multibody` on each cluster (`split_radius`, default `None` -
    scaled to each cluster's own point density, see `_split_multibody`; `split_ratio`, default `SPLIT_RATIO`)
    - a cluster connected only at the coarse `radius` above
    may still be several real, comparably-sized features (or several small features plus debris) chained
    together, which neither a geometric fit nor a Type network can be expected to read as one primitive."""
    inside = model.sdf(points[unexplained]) < 0
    out = []
    for side in (True, False):
        idx = unexplained[inside == side]
        for cl in _clusters(points[idx], radius):
            sel = idx[cl]
            pieces = _split_multibody(sel, points, split_radius, split_ratio) if split_multibody else [sel]
            out += [(piece, side) for piece in pieces]
    if cuts_first:
        return sorted(out, key=lambda c: (not c[1], -len(c[0])))
    return sorted(out, key=lambda c: len(c[0]), reverse=True)


def _explained(model: sp.Primitive, s: Samples, tol: float) -> float:
    return float(np.mean(np.abs(model.sdf(s.surface)) < tol))


def invented(model: sp.Primitive, field: SurfaceField, tol: float, extent: float, n: int = 60000, keep: int = 1500, seed: int = 0) -> float:
    """The other direction of `_explained`: the share of the MODEL's surface that is farther than 2 tol from the input surface, i.e.
    geometry the model invents. Random points around the model are moved onto its zero set (two Newton steps), a sample of them is
    checked against the field with exact queries. A cheap stand-in for the model -> input half of `chamfer` (no mesh needed)."""
    rng = np.random.default_rng(seed)
    lo, hi = model.bounds()
    pad = 0.03 * extent
    P = rng.uniform(lo - pad, hi + pad, size=(n, 3))
    P = P[np.abs(model.sdf(P)) < 0.03 * extent]
    if len(P) == 0:
        return 0.0
    eps = 1e-3 * extent
    for _ in range(2):
        grad = np.stack([(model.sdf(P + eps * e) - model.sdf(P - eps * e)) / (2 * eps) for e in np.eye(3)], axis=1)
        P = P - (model.sdf(P) / np.maximum((grad ** 2).sum(1), 1e-12))[:, None] * grad
    P = P[np.abs(model.sdf(P)) < 0.05 * tol]                                 # the points that landed on the surface
    if len(P) < 30:
        return 0.0
    if len(P) > keep:
        P = P[rng.choice(len(P), keep, replace=False)]
    return float(np.mean(field.query(P).distance > 2 * tol))


def _freecad_check(model: sp.Primitive, out_dir, name: str, input_genus: Optional[float] = None) -> dict:
    """`verify_freecad`'s own check: build `model` for real in FreeCAD (`export.build_split_in_freecad`) and boil
    the result down to what a front end shows. Never raises - a diagnostic aid, not load-bearing.

    Also the exact genus of the built solid (its STL's per-shell Euler characteristic, `SurfaceField.genus`) next to
    the input's: a topological invariant that IoU and chamfer average away (the frame's "solved" model lacked the 16
    plate loops of its genus 17). Report only: `genus_match` is True/False, or None when either genus is unknown (an
    unclosed mesh gives NaN), never a silent pass. Intermediate steps differ from the input by design."""
    try:
        from reco_field.export import build_split_in_freecad
        ex = build_split_in_freecad(model, out_dir, name)
        leftovers = [s["volume"] for s in ex.solids[1:]] if ex.solids else []
        out = dict(valid=ex.valid, n_solids=ex.n_solids, volume=ex.volume, leftover_volumes=leftovers, step=str(ex.step))
    except Exception as e:
        return dict(error=str(e))
    try:
        from src.closest_point_field import load_field
        g = float(load_field(str(ex.stl)).genus())
    except Exception:
        g = float("nan")
    known = np.isfinite(g) and input_genus is not None and np.isfinite(input_genus)
    out.update(genus=g, input_genus=input_genus, genus_match=(bool(round(g) == round(input_genus)) if known else None))
    return out


def _best(kinds, samples, bounding_box: bool):
    """`fitmod.fit_best`, with the box result's GEOMETRY (not its ranking - see the caller sites' comments)
    replaced by `fitmod.fit_box_bounding` when `bounding_box` is set and box is already the winning kind."""
    results = fitmod.fit_best(None, kinds=kinds, samples=samples)
    if bounding_box and results and results[0].kind == "box":
        results[0] = fitmod.fit_box_bounding(samples)
    return results


def peel_steps(field: SurfaceField, max_ops: Optional[int] = None, target: float = 0.985, kinds: Sequence[str] = fitmod.WHOLE_KINDS,
               n_surface: int = 4000, n_near: int = 2500, seed: int = 0, min_fraction: float = 0.01,
               min_gain: float = 0.005, n_vectors: int = 0, grade: bool = False, bounding_box: bool = False,
               invented_slack: float = INVENTED_SLACK, cuts_first: Optional[bool] = None,
               max_candidate_invented: Optional[float] = None, split_multibody: bool = False,
               split_radius: Optional[float] = None, split_ratio: float = SPLIT_RATIO,
               mesh_clusters: bool = False, verify_freecad: bool = False, octree: bool = False,
               octree_resolution: int = 32, octree_depth: int = 1, octree_real_voxels: bool = False) -> Iterator[PeelStep]:
    """The loop of `peel`, yielding a PeelStep at every stage (the work of a stage happens when the generator is advanced to it).
    `n_vectors` > 0 adds that many field vectors to the first step, `grade` adds the chamfer as a step before the last.
    `max_ops=None` (the default): predicted from the part's own genus (`predict_ops_budget`) rather than the fixed
    `MAX_OPS` - pass an int to keep the old fixed behaviour.

    `bounding_box` (default off): whenever a box already won a fit (base, cut or add), use
    `fitmod.fit_box_bounding` for its GEOMETRY instead of the usual RANSAC-start + Tukey-IRLS converged fit -
    never for choosing which kind wins, since a bounding box's own explained/cost score is structurally worse
    than a converged fit's and would corrupt that ranking. A converged box can quietly exclude real surface
    (Tukey IRLS treats far points as outliers and shrinks away from them - a real problem for a non-convex
    cluster like an I-beam flange's concave chamfer where it meets the web, `docs/FRAME_FAILURE.md`'s
    "saddle"); a bounding box mathematically cannot exclude any point of its own input. It also invents MORE
    (a box's own flat faces where the real part has none, e.g. spanning a sparse outer step), so
    `invented_slack` (default `INVENTED_SLACK`, the same 2% every other caller gets) needs to be opened wider
    for this mode to accept what it proposes - the "rethought acceptance gate" this option exists for. Applies
    uniformly to the base and every CUT candidate (never add - measured worse there, see the module's own
    comment at the call site) for the whole run. OFF changes nothing (identical to every call before this
    feature: `docs/data/peel_examples.json`'s 41 parts, checked).

    `cuts_first`: try every INSIDE/cutter cluster before any OUTSIDE/addition one (`_side_clusters`), so an
    oversized `bounding_box` base gets carved down before anything is added to it. `None` (default) follows
    `bounding_box`; pass `True`/`False` explicitly to decouple them.

    `max_candidate_invented`: reject a candidate outright if ITS OWN standalone `invented` share (not the
    whole trial's) exceeds this, regardless of `invented_slack` - the per-candidate purity check `invented_slack`
    alone cannot provide, since a global slack on the WHOLE model's invented share can still be spent entirely
    by one badly oversized candidate. `None` (default): no such check, identical to before this parameter existed.

    `split_multibody` (default off): before fitting a cluster, check whether it is really several real,
    comparably-sized bodies chained together at the coarse clustering radius (`_split_multibody`,
    `split_radius`/`split_ratio`) and fit each one separately if so - a cluster is otherwise asked to look
    like ONE primitive (or answer ONE Type-network question) even when it structurally is not one at all.
    OFF changes nothing (identical to every call before this feature existed). Superseded by `mesh_clusters`
    when that is also on (mesh components are already exact, not sample-density-confounded like
    `_side_clusters`'s point clustering is - see `mesh_clusters` below - so there is nothing left to split).

    `verify_freecad` (default off): after the base and every accepted step, actually BUILD the cumulative model
    in FreeCAD (real `Part` booleans via `freecadcmd`, not just the symbolic SDF tree every other check in this
    loop uses) and attach the result (`PeelStep.freecad`: `n_solids`, `valid`, `volume`, `leftover_volumes` - the
    volume of every solid beyond the biggest, if the real CSG actually split the model into more than one piece,
    which the symbolic SDF never notices at all) to that step. Slow (a `freecadcmd` subprocess per step, ~1-3s) -
    a verification/debugging aid (`scripts/debug_freecad_solid_loop.py`, `docs/FRAME_FAILURE.md`'s "Materializing
    every step in real FreeCAD"), never a source of candidates: it cannot reject or change a step, only report on
    the one already accepted. Needs `freecadcmd`; a failure (not found, a timeout, an unbuildable primitive) is
    caught and reported as `{"error": ...}` rather than raised, since this is diagnostic, not load-bearing for
    the reconstruction itself. OFF changes nothing (identical to every call before this feature existed).

    `mesh_clusters` (default off): cluster with `_mesh_clusters` (full-mesh triangle connectivity, "known CAD
    pipeline" component detection - `trimesh.Trimesh.split`) instead of `_side_clusters` (Euclidean-distance
    clustering of the SPARSE `n_surface`-point sample). Measured on the frame part: the sparse sample's
    distance clustering found only 8 real cut-side features (many small ones never sampled at all, or
    accidentally bridged into a neighbour) where the full mesh actually has 24 - `docs/FRAME_FAILURE.md`,
    "Full-mesh component clustering". Each mesh component gets its own freshly-drawn, dense sample
    (`_mesh_component_samples`) rather than reusing whatever of the original sparse sample happened to land on
    it, so a real small feature with zero original samples is still found and fittable. `min_fraction` still
    gates acceptance (by the component's own dense sample count, same as any other cluster); `split_multibody`
    and `cuts_first`'s cluster REORDERING still apply on top (mesh components are already resolved into true
    single bodies, so nothing gets split further, but cuts-before-additions ordering still means something).
    OFF changes nothing (identical to every call before this feature existed).

    `octree` (default off): cluster with `_octree_clusters` (an occupancy octree over the missing VOLUME - input
    solid XOR model, `input_occupancy`/`_octree_leaves`, ported from `reco_engine/octree_search.py`'s voxel-CNN
    strategy on the `octree` branch) instead of `_side_clusters`/`mesh_clusters`. Where those group by SURFACE
    proximity or connectivity, this groups by spatial locality of the missing VOLUME, coarse and axis-aligned -
    immune to both the point-clustering density confound and the mesh-topology fragmentation into dozens of
    slivers `mesh_clusters` measured on the frame (`docs/FRAME_FAILURE.md`, "Debugging the identification
    schema with the frame"): the frame's bounding box minus its own material is still a simple, roughly
    rectangular volume, so a coarse octree over it stays a handful of big leaves rather than fragmenting.
    `octree_resolution` (default 32) sets the shared grid `input_occupancy` and the model's own `sdf` are
    evaluated on; `octree_depth` (default 1) bounds how many times a leaf is still split - both exist for the
    same reason `LocalSearchConfig`/`OctreeConfig` bound their own search: predictable cost however fragmented
    the target is. `octree_real_voxels` (default off): use `real_voxel_occupancy` (the project's ground-truth
    voxelizer, scanline ray parity at the voxel centres, D-vox) instead of `input_occupancy`
    (`trimesh.Trimesh.contains`) as the occupancy source - the stable ground truth every other
    voxel-based part of this project already trusts, rather than a bespoke one built just for this feature
    (`docs/FRAME_FAILURE.md`, "Voxels as the ground truth occupancy source"). Only takes effect with
    `octree=True`; has no effect otherwise. OFF changes nothing (identical to every call before this feature
    existed).

    `min_fraction` and `min_gain` are BOTH relative to what is still unexplained, not to the whole part
    (`docs/FRAME_FAILURE.md`, Strategy item 1): a cluster needs `min_fraction` of the CURRENT unexplained points
    (`len(unexplained)`, not the fixed `n_surface`) to be tried at all, and a candidate needs to explain
    `min_gain` of the CURRENT unexplained share (`1 - explained`, not a flat percentage point of the whole part)
    to be accepted. At iteration 1 (nothing explained yet, unexplained == n_surface) this is numerically
    identical to the project's original flat floors (0.01 * n_surface = 40 points, 0.005 = half a percentage
    point); both floors shrink automatically as more of the part gets explained, so a small feature that is a
    large share of what's LEFT (many similarly-small repeated features sharing what a big part's early
    operations already covered) can pass even though its share of the WHOLE part is tiny - the concrete case
    that motivated this: ~150 bridging rods on a real frame part, each too small a share of the whole part to
    ever ATTEMPT or PASS the original flat floors (see `docs/FRAME_FAILURE.md`, "Why it fails", point 4)."""
    s = make_samples(field, n_surface=n_surface, n_near=n_near, seed=seed)
    tol = tolerance(s)
    spacing = float(np.sqrt(field.area / n_surface))
    budget = predict_ops_budget(field) if max_ops is None else None
    max_ops = budget.max_ops if budget else max_ops
    vp = vv = None
    if n_vectors > 0:
        vp = field.sample_points(n_vectors, band=0.15 * s.extent, seed=seed + 2)
        vv = field.query(vp).vector
    budget_msg = f"; budget {max_ops} ops ({budget.reason})" if budget else ""
    yield PeelStep("field", "base", 0, f"{n_surface} surface points, {len(s.near)} field points; tolerance {tol:.4g}{budget_msg}",
                   samples=s, field=field, vector_points=vp, vectors=vv, tol=tol, budget=budget)

    base = _best(kinds, s, bounding_box)[0]
    ops = [Op("base", base, base.primitive, n_surface, 0.0)]
    model = base.primitive
    explained = _explained(model, s, tol)
    ops[0].explained_after = explained
    history = [explained]

    def unexplained_of(m):
        return np.nonzero(np.abs(m.sdf(s.surface)) >= tol)[0]

    octree_grid = None
    if octree:
        pad = 0.03 * s.extent
        grid_lo, grid_hi = field.bounds[0] - pad, field.bounds[1] + pad
        occupancy = real_voxel_occupancy if octree_real_voxels else input_occupancy
        occ = occupancy(field, grid_lo, grid_hi, octree_resolution)
        octree_grid = (grid_lo, grid_hi, occ)
    verify_dir = None
    if verify_freecad:
        verify_dir = tempfile.mkdtemp(prefix="peel_verify_freecad_")
        atexit.register(shutil.rmtree, verify_dir, ignore_errors=True)   # peel_steps is a generator a caller
                                                                          # (the GUI) may never exhaust - clean
                                                                          # up at process exit, not generator close
    inv = invented(model, field, tol, s.extent, seed=seed)
    input_genus = float(field.genus()) if verify_freecad else None
    fc = _freecad_check(model, verify_dir, "step0_base", input_genus) if verify_freecad else None
    yield PeelStep("base", "base", 0, f"base: {base.kind}, {explained:.1%} explained, {inv:.1%} of its surface off the input", samples=s,
                   tol=tol, ops=tuple(ops), model=model, explained=explained, unexplained=unexplained_of(model), candidate=ops[0],
                   trial=model, trial_explained=explained, invented=inv, trial_invented=inv, freecad=fc)

    stopped = "max operations"
    for it in range(1, max_ops):
        if explained >= target:
            stopped = "target reached"
            break
        unexplained = unexplained_of(model)
        progressed = False
        cuts_first_now = bounding_box if cuts_first is None else cuts_first
        if mesh_clusters:
            # (cluster Samples, inside, size-for-the-min_fraction-floor, index-array-for-the-step-or-None):
            # a mesh component's own face count stands in for the point path's cluster-point-count (its dense
            # sample is capped near a constant regardless of the component's real size, so the SAMPLE count
            # itself is not a meaningful size proxy here - see `mesh_clusters`'s docstring above), compared
            # against the total unexplained face count instead of `len(unexplained)` points.
            components = _mesh_clusters(model, field, tol, s.extent, seed=seed)
            total_faces = sum(n for _samp, _side, n in components) or 1
            # the index array is only ever used by the GUI to highlight a step's cluster in `s.surface` - a
            # mesh component has no such indices of its own (it is a FRESH dense sample, not a subset of
            # `s.surface`), so approximate it with whichever of the existing sparse points land within one
            # clustering radius of the component's own sample; None (rather than crash) for a component the
            # sparse sample never reached at all, e.g. a small feature `_side_clusters` would have missed too.
            tree = cKDTree(s.surface)
            candidate_list = []
            for samp, side, n in components:
                near = sorted(set(np.concatenate(tree.query_ball_point(samp.surface, 3.0 * spacing))))
                sel = np.asarray(near, dtype=int) if near else None
                candidate_list.append((samp, side, n / total_faces * len(unexplained), sel))
            if cuts_first_now:
                candidate_list.sort(key=lambda c: (not c[1], -c[2]))
            else:
                candidate_list.sort(key=lambda c: -c[2])
        elif octree:
            grid_lo, grid_hi, input_occ = octree_grid
            octree_pairs = _octree_clusters(model, s.surface, unexplained, input_occ, grid_lo, grid_hi,
                                            max_depth=octree_depth)
            side_clusters = sorted(octree_pairs, key=lambda c: (not c[1], -len(c[0]))) if cuts_first_now else \
                sorted(octree_pairs, key=lambda c: len(c[0]), reverse=True)
            candidate_list = [(None, side, len(sel), sel) for sel, side in side_clusters]
        else:
            side_clusters = _side_clusters(model, s.surface, unexplained, 3.0 * spacing, cuts_first=cuts_first_now,
                                           split_multibody=split_multibody, split_radius=split_radius, split_ratio=split_ratio)
            candidate_list = [(None, side, len(sel), sel) for sel, side in side_clusters]
        for cluster_or_none, inside, size, sel in candidate_list:
            if size < min_fraction * len(unexplained):
                continue      # not `break`: with cuts first the list is cuts THEN adds, and a small cut must not hide the adds
            where = "inside the material: a cutter" if inside else "outside the material: an addition"
            yield PeelStep("cluster", "peel", it, f"cluster of {round(size)} of {len(unexplained)} unexplained points, {where}",
                           samples=s, tol=tol, ops=tuple(ops), model=model, explained=explained, unexplained=unexplained,
                           cluster=sel, inside=inside)
            if cluster_or_none is not None:
                cluster = cluster_or_none
            else:
                mask = np.zeros(n_surface, bool)
                mask[sel] = True
                cluster = subset(s, mask)
            # bounding_box only ever helps a CUTTER (carving an oversized base down is conservative - a cut
            # that undershoots just leaves material for a later cut, never invents); measured directly against
            # real addition clusters it does WORSE than the converged fit (docs/FRAME_FAILURE.md), so an
            # addition always gets the normal converged search regardless of `bounding_box`.
            candidates = _best(tuple(kinds) + (("plane",) if inside else ()), cluster, bounding_box and inside)
            # the three best fits, and the plane clip whatever its rank: it is another operation, not a fourth kind of
            # cutter. On a dense mesh-component sample a torus and a cylinder outrank it (94 % and 79 % of the cluster
            # against 76 %), all three cutters are rejected, and without the clip the run ended at the bounding box
            # (69.8 % explained, where the point clusters reach 100 %; tests/test_mesh_clusters.py, 2026-09-27)
            for fit in candidates[:3] + [f for f in candidates[3:] if f.kind == "plane"]:
                if fit.explained < 0.15:                                       # a cluster may hold several primitives: take a part
                    continue
                if fit.kind == "plane":
                    op, prim = "clip", fit.primitive
                else:
                    op = "cut" if inside else "add"
                    prim = _extend(fit, cluster.surface, cluster.normals, s.extent, op)
                trial = fold(model, op, prim)
                e = _explained(trial, s, tol)
                cand = Op(op, fit, prim, len(cluster.surface), e)
                gain = e > explained + min_gain * (1 - explained)
                inv_trial = invented(trial, field, tol, s.extent, seed=seed) if gain else float("nan")   # only what could be accepted
                # a candidate's OWN standalone invented share (not the whole trial's) - a wide `invented_slack`
                # only bounds how much the WHOLE model's invented share may rise, so one badly oversized
                # candidate can still slip through by spending that whole allowance itself, spread thin over a
                # now-much-bigger model surface (docs/FRAME_FAILURE.md, "Bounding-box fits..." - measured: a
                # 47%-of-its-own-cluster, 85.6%-of-its-own-surface-invented cylinder was accepted exactly this
                # way at invented_slack=0.2). `max_candidate_invented` judges the candidate by itself - ADD
                # only: a cutter's own surface is a TOOL, mostly never meant to coincide with real geometry (it
                # carves out a void, e.g. a hole's cylindrical wall lies entirely inside the material it
                # removes), so the same standalone-invented measure is meaningless there and would reject
                # every cut (confirmed while building this: it did, collapsing the run to the base alone).
                own_inv = (invented(prim, field, tol, s.extent, seed=seed)
                          if gain and max_candidate_invented is not None and op == "add" else None)
                yield PeelStep("candidate", "peel", it, f"{op}:{fit.kind} explains {fit.explained:.0%} of the cluster; "
                               f"the model would explain {e:.1%} (now {explained:.1%})"
                               + (f", {inv_trial:.1%} of its surface off the input (now {inv:.1%})" if gain else "")
                               + (f"; the candidate ITSELF is {own_inv:.1%} off the input" if own_inv is not None else ""),
                               samples=s, tol=tol, ops=tuple(ops), model=model, explained=explained, unexplained=unexplained,
                               cluster=sel, inside=inside, candidate=cand, trial=trial, trial_explained=e, invented=inv,
                               trial_invented=None if not gain else inv_trial)
                candidate_pure = own_inv is None or own_inv <= max_candidate_invented
                if gain and inv_trial <= inv + invented_slack and candidate_pure:
                    ops.append(cand)
                    model, explained, inv = trial, e, inv_trial
                    history.append(e)
                    progressed = True
                    fc = _freecad_check(model, verify_dir, f"step{len(ops) - 1}_{op}", input_genus) if verify_freecad else None
                    yield PeelStep("accepted", "peel", it, f"accepted {op}:{fit.kind}: {e:.1%} explained", samples=s, tol=tol,
                                   ops=tuple(ops), model=model, explained=explained, unexplained=unexplained_of(model),
                                   candidate=cand, trial=trial, trial_explained=e, invented=inv, trial_invented=inv, freecad=fc)
                    break
                why = (f"gain {e - explained:+.1%} is below {min_gain * (1 - explained):.1%} of what's left" if not gain
                       else f"the candidate itself is {own_inv:.1%} off the input, above {max_candidate_invented:.1%}" if not candidate_pure
                       else f"it invents geometry: {inv_trial:.1%} of the model's surface off the input, was {inv:.1%}")
                yield PeelStep("rejected", "peel", it, f"rejected {op}:{fit.kind}: {why}",
                               samples=s, tol=tol, ops=tuple(ops), model=model, explained=explained, unexplained=unexplained,
                               cluster=sel, inside=inside, candidate=cand, trial=trial, trial_explained=e, invented=inv,
                               trial_invented=None if not gain else inv_trial)
            if progressed:
                break
        if not progressed:
            stopped = "nothing helps"
            break
    else:
        stopped = "target reached" if explained >= target else "max operations"
    result = PeelResult(ops, model, explained, history, tol, stopped, budget)
    grades = chamfer(model, field) if grade else None
    if grade:
        yield PeelStep("grade", "done", len(ops), f"chamfer {grades['symmetric_mean']:.4g}, worst {grades['hausdorff']:.4g}",
                       samples=s, tol=tol, ops=tuple(ops), model=model, explained=explained, grade=grades)
    yield PeelStep("done", "done", len(ops), f"done: {len(ops)} operation(s), {explained:.1%} explained ({stopped})", samples=s,
                   tol=tol, ops=tuple(ops), model=model, explained=explained, grade=grades, result=result)


def peel(field: SurfaceField, max_ops: Optional[int] = None, target: float = 0.985, kinds: Sequence[str] = fitmod.WHOLE_KINDS,
         n_surface: int = 4000, n_near: int = 2500, seed: int = 0, min_fraction: float = 0.01,
         min_gain: float = 0.005, bounding_box: bool = False, invented_slack: float = INVENTED_SLACK,
         cuts_first: Optional[bool] = None, max_candidate_invented: Optional[float] = None,
         split_multibody: bool = False, split_radius: Optional[float] = None, split_ratio: float = SPLIT_RATIO,
         mesh_clusters: bool = False, verify_freecad: bool = False, octree: bool = False,
         octree_resolution: int = 32, octree_depth: int = 1, octree_real_voxels: bool = False) -> PeelResult:
    """`max_ops=None` (the default): predicted from the part's own genus, see `peel_steps`."""
    step = None
    for step in peel_steps(field, max_ops, target, kinds, n_surface, n_near, seed, min_fraction, min_gain,
                           bounding_box=bounding_box, invented_slack=invented_slack, cuts_first=cuts_first,
                           max_candidate_invented=max_candidate_invented, split_multibody=split_multibody,
                           split_radius=split_radius, split_ratio=split_ratio, mesh_clusters=mesh_clusters,
                           verify_freecad=verify_freecad, octree=octree, octree_resolution=octree_resolution,
                           octree_depth=octree_depth, octree_real_voxels=octree_real_voxels):
        pass
    return step.result


# ------------------------------------------------------------------------------------- the geometric grade (both ways)

def mesh_from_sdf(prim: sp.Primitive, resolution: int = 128, pad_fraction: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Triangle mesh of the zero level set of `prim` (vtk flying edges on a grid), vertices then projected onto the exact surface."""
    import vtk
    from vtkmodules.util import numpy_support
    lo, hi = prim.bounds()
    pad = pad_fraction * float((hi - lo).max())
    lo, hi = lo - pad, hi + pad
    ax = [np.linspace(lo[i], hi[i], resolution) for i in range(3)]
    spacing = [(hi[i] - lo[i]) / (resolution - 1) for i in range(3)]
    g = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
    d = prim.sdf(g)
    img = vtk.vtkImageData()
    img.SetDimensions(resolution, resolution, resolution)
    img.SetOrigin(*lo)
    img.SetSpacing(*spacing)
    arr = numpy_support.numpy_to_vtk(d.ravel(order="F"), deep=True)
    img.GetPointData().SetScalars(arr)
    fe = vtk.vtkFlyingEdges3D()
    fe.SetInputData(img)
    fe.SetValue(0, 0.0)
    fe.ComputeNormalsOff()
    fe.Update()
    poly = fe.GetOutput()
    verts = numpy_support.vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    tris = numpy_support.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(int)
    eps = 1e-4 * float((hi - lo).max())
    for _ in range(2):                                                       # Newton step onto the zero set
        f = prim.sdf(verts)
        grad = np.stack([(prim.sdf(verts + eps * e) - prim.sdf(verts - eps * e)) / (2 * eps) for e in np.eye(3)], axis=1)
        verts = verts - (f / np.maximum((grad ** 2).sum(1), 1e-12))[:, None] * grad
    # orient the triangles OUTWARD (their normals point to positive SDF): flying edges does not promise a winding
    n = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-30)[:, None]
    centre = verts[tris].mean(axis=1)
    step = 3.0 * float(max(spacing))
    if np.mean(prim.sdf(centre + step * n) > prim.sdf(centre - step * n)) < 0.5:
        tris = tris[:, [0, 2, 1]]
    return verts, tris


def _stats(d: np.ndarray) -> Dict[str, float]:
    return dict(mean=float(d.mean()), p95=float(np.percentile(d, 95)), max=float(d.max()))


def chamfer(model: sp.Primitive, field: SurfaceField, n: int = 20000, resolution: int = 128, seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Distance of the model's surface to the input mesh and of the input mesh to the model's surface (mean, 95th percentile,
    maximum), in the units of the geometry. `input_to_model` large: the model misses geometry; `model_to_input` large: the model
    invents geometry."""
    v, f = mesh_from_sdf(model, resolution)
    mf = SurfaceField(v, f)
    rng = np.random.default_rng(seed)

    def surface_points(fld: SurfaceField, k: int):
        area = 0.5 * np.linalg.norm(np.cross(fld._b - fld._a, fld._c - fld._a), axis=1)
        idx = rng.choice(len(fld.faces), size=k, p=area / area.sum())
        r1, r2 = rng.random((k, 1)), rng.random((k, 1))
        flip = (r1 + r2) > 1
        r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
        return fld._a[idx] + (fld._b[idx] - fld._a[idx]) * r1 + (fld._c[idx] - fld._a[idx]) * r2

    # exact queries: the approximate mode is unreliable on CAD meshes (cylinder walls are long sliver triangles)
    a = mf.query(surface_points(field, n)).distance                       # input surface -> model surface
    b = field.query(surface_points(mf, n)).distance                       # model surface -> input surface
    out = dict(input_to_model=_stats(a), model_to_input=_stats(b))
    out["symmetric_mean"] = 0.5 * (out["input_to_model"]["mean"] + out["model_to_input"]["mean"])
    out["hausdorff"] = max(out["input_to_model"]["max"], out["model_to_input"]["max"])
    return out
