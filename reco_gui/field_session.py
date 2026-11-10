#!/usr/bin/env python3
"""
The Qt-free core of the vector-field mode of the GUI: `reco_field.peel.peel_steps` yields one PeelStep per stage; `FieldFolder`
folds each into an immutable FieldState (the input geometry and its field vectors, the model so far as a mesh, the surface the
model does not explain, the cluster and candidate being tried, the accepted operations). A StepSession built with this folder
(`make_field_session`) has the same interface as the voxel one, so the worker, the timeline and the pacing are shared.

Everything is in the geometry's own units (mm), not in voxels. The meshes are made here, in the worker thread, so the GUI
thread only draws.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

FIELD_STAGES = ("field", "base", "cluster", "candidate", "accepted", "rejected", "grade", "done")
FIELD_DEFAULT_BREAKPOINTS = ()          # none checked at start (the user's rule 2026-10-15)
MESH_RESOLUTION = 72


@dataclass(frozen=True)
class MeshData:
    vertices: np.ndarray             # (v, 3)
    triangles: np.ndarray            # (t, 3)


@dataclass(frozen=True)
class OpInfo:
    """One accepted operation, for the report."""
    op: str                          # base | add | cut | clip
    kind: str                        # sphere | box | cylinder | cone | torus | plane
    params: Dict[str, float]
    explained_fit: float             # share of its own points the fit explains
    explained_after: float           # share of the whole input surface the model explains after it
    n_points: int


@dataclass(frozen=True)
class FieldState:
    index: int
    step: object                                     # the PeelStep this state follows
    input_mesh: Optional[MeshData] = None
    vector_points: Optional[np.ndarray] = None       # sample of the field: x ...
    vectors: Optional[np.ndarray] = None             # ... and V(x)
    surface: Optional[np.ndarray] = None             # the input surface points the loop works with (n, 3)
    tol: float = 0.0
    ops: Tuple[OpInfo, ...] = ()
    model_mesh: Optional[MeshData] = None            # the model after the accepted operations
    explained: float = 0.0
    history: Tuple[float, ...] = ()
    unexplained: Optional[np.ndarray] = None         # indices into `surface`
    cluster: Optional[np.ndarray] = None
    inside: Optional[bool] = None
    candidate: Optional[OpInfo] = None               # the operation being tried
    candidate_mesh: Optional[MeshData] = None        # the primitive as entered (an open cutter is cropped to the part's box)
    trial_explained: Optional[float] = None
    invented: Optional[float] = None                 # share of the model's surface farther than 2 tol from the input
    trial_invented: Optional[float] = None           # the same with the candidate folded in (None: not measured, it gained too little)
    verdict: Optional[str] = None                    # "accepted" | "rejected" | None
    grade: Optional[dict] = None
    stopped: str = ""
    finished: bool = False
    result: object = None
    freecad: Optional[dict] = None                   # verify_freecad only: this step's real-CSG check


def op_info(op) -> OpInfo:
    return OpInfo(op.op, op.fit.kind, {k: float(v) for k, v in op.fit.params.items()}, float(op.fit.explained),
                  float(op.explained_after), int(op.n_points))


class FieldFolder:
    """`fold(previous, step, index)` -> FieldState. Keeps a mesh cache so that a primitive is meshed once."""

    def __init__(self, resolution: int = MESH_RESOLUTION):
        self.resolution = resolution
        self._meshes: Dict[int, Optional[MeshData]] = {}
        self._keep = []                              # the primitives whose id() is a cache key must stay alive
        self._crop = None

    def mesh(self, prim) -> Optional[MeshData]:
        if prim is None:
            return None
        if id(prim) not in self._meshes:
            from reco_field import peel
            from src import sdf_primitives as sp
            clipped = prim
            if self._crop is not None:                                        # an open cutter reaches far out of the part: crop it
                lo, hi = self._crop
                clipped = sp.Intersection(prim, sp.Placed(sp.Box((hi - lo) / 2), translation=(lo + hi) / 2))
            try:
                v, f = peel.mesh_from_sdf(clipped, self.resolution)
                data = MeshData(v, f) if len(f) else None
            except Exception:                                                 # an empty or degenerate level set: nothing to draw
                data = None
            self._meshes[id(prim)] = data
            self._keep.append(prim)
        return self._meshes[id(prim)]

    def __call__(self, prev: Optional[FieldState], step, index: int) -> FieldState:
        s = replace(prev, index=index, step=step) if prev is not None else FieldState(index=index, step=step)
        if step.stage == "field":
            lo, hi = step.field.bounds
            pad = 0.08 * float((hi - lo).max())
            self._crop = (lo - pad, hi + pad)
            return replace(s, input_mesh=MeshData(step.field.vertices, step.field.faces), vector_points=step.vector_points,
                           vectors=step.vectors, surface=step.samples.surface, tol=step.tol)
        common = dict(ops=tuple(op_info(o) for o in step.ops), explained=step.explained,
                      history=tuple(o.explained_after for o in step.ops), unexplained=step.unexplained,
                      invented=step.invented if step.invented is not None else s.invented, trial_invented=step.trial_invented)
        if step.stage == "base":
            return replace(s, **common, model_mesh=self.mesh(step.model), cluster=None, inside=None, candidate=None,
                           candidate_mesh=None, trial_explained=None, verdict=None, freecad=step.freecad)
        if step.stage == "cluster":
            return replace(s, **common, model_mesh=self.mesh(step.model), cluster=step.cluster, inside=step.inside, candidate=None,
                           candidate_mesh=None, trial_explained=None, verdict=None)
        if step.stage == "candidate":
            return replace(s, **common, cluster=step.cluster, inside=step.inside, candidate=op_info(step.candidate),
                           candidate_mesh=self.mesh(step.candidate.primitive), trial_explained=step.trial_explained, verdict=None)
        if step.stage == "accepted":
            return replace(s, **common, model_mesh=self.mesh(step.model), cluster=None, candidate=op_info(step.candidate),
                           candidate_mesh=None, trial_explained=step.trial_explained, verdict="accepted", freecad=step.freecad)
        if step.stage == "rejected":
            return replace(s, **common, cluster=step.cluster, candidate=op_info(step.candidate),
                           trial_explained=step.trial_explained, verdict="rejected")
        if step.stage == "grade":
            return replace(s, grade=step.grade, cluster=None, candidate=None, candidate_mesh=None)
        if step.stage == "done":
            return replace(s, finished=True, result=step.result, grade=step.grade or s.grade, stopped=step.result.stopped,
                           cluster=None, candidate=None, candidate_mesh=None)
        return s


def make_field_session(path, max_ops: int = 24, target: float = 0.985, n_surface: int = 4000, n_near: int = 2500,
                       n_vectors: int = 300, grade: bool = True, deflection: float = 0.02, bounding_box: bool = False,
                       invented_slack: Optional[float] = None, cuts_first: Optional[bool] = None,
                       max_candidate_invented: Optional[float] = None, mesh_clusters: bool = False,
                       verify_freecad: bool = False, octree: bool = False, octree_resolution: int = 32,
                       octree_depth: int = 1, octree_real_voxels: bool = False):
    """A StepSession over the peeling loop of the STL/STEP file `path`. The field is loaded (a STEP is tessellated by freecadcmd)
    when the first step is asked for, so `make` itself is instant.

    `bounding_box`, `invented_slack`, `cuts_first`, `max_candidate_invented`, `mesh_clusters`, `verify_freecad`,
    `octree`/`octree_resolution`/`octree_depth`/`octree_real_voxels`: reco_field.peel.peel_steps's "rethought
    acceptance gate" options (docs/FRAME_FAILURE.md) - all off/default unless set, so the ordinary run this GUI
    has always done is unaffected. `verify_freecad` actually builds every accepted step's CSG in FreeCAD (real
    `Part` booleans via `freecadcmd`, not just the symbolic SDF) and reports whether it stayed one valid solid -
    slow (a freecadcmd subprocess per step), see `PeelStep.freecad`. `octree_real_voxels` uses the project's
    ground-truth voxelizer (ray parity at the voxel centres, D-vox) as the occupancy source instead of `trimesh.contains` -
    only takes effect when `octree` is also on."""
    from reco_gui.session import StepSession

    def steps():
        from reco_field.peel import INVENTED_SLACK, peel_steps
        from src.closest_point_field import load_field
        field = load_field(path, deflection=deflection)
        yield from peel_steps(field, max_ops=max_ops, target=target, n_surface=n_surface, n_near=n_near, n_vectors=n_vectors,
                              grade=grade, bounding_box=bounding_box,
                              invented_slack=invented_slack if invented_slack is not None else INVENTED_SLACK,
                              cuts_first=cuts_first, max_candidate_invented=max_candidate_invented,
                              mesh_clusters=mesh_clusters, verify_freecad=verify_freecad, octree=octree,
                              octree_resolution=octree_resolution, octree_depth=octree_depth,
                              octree_real_voxels=octree_real_voxels)

    return StepSession(steps(), fold=FieldFolder())
