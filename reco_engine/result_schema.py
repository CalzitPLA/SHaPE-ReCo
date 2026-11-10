#!/usr/bin/env python3
"""
One result schema for the TYPE -> ROTATION -> DIMENSIONS -> POSITION cascade (reco_engine/cascade.py) and the
residual add-cut loop (reco_engine/residual_loop.py), shared by the GUI, the FreeCAD nodes and the JSON output, so
the three consumers cannot drift apart.

Units: millimetres in the input's own world frame. `quaternion` is (qx, qy, qz, qw), the same order the dataset
labels and reco_engine.placement use. `dims_mm` is keyed by the parameter names of reco_field / sdf_primitives
(box: x_half, y_half, z_half; sphere: radius; cylinder: radius, half_length; cone: radius1, radius2, half_length;
torus: major_radius, minor_radius), so a result can be built with reco_field.export without renaming.

A stage records WHICH mode answered it (`StageSource.mode`: "cnn:<registry id>", "analytic:<name>",
"prior:<name>", "field_fit", "voxel_fallback") and with what confidence, so the GUI can show the provenance of every
number and an evaluation can count how often each mode short-circuited the cascade.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

KINDS = ("box", "sphere", "cylinder", "cone", "torus")
UNKNOWN = "unknown"
OPS = ("add", "cut", "intersect")    # model | P, model - P, model & P
# Beyond KINDS: "halfspace" (a plane: local +z is the side a cut removes; the centre lies on the plane; no dims) and
# "composite" (two `parts` joined by `combine` = "common" | "cut" | "fuse", built and judged as one primitive:
# box - cylinder for a fillet, box & cylinder for a pocket with a rounded end).
COMBINES = ("common", "cut", "fuse")
STAGES = ("type", "rotation", "dimensions", "position")


@dataclass
class StageSource:
    stage: str                    # one of STAGES
    mode: str                     # "cnn:<id>", "analytic:<name>", "prior:<name>", "field_fit", "voxel_fallback"
    confidence: float = 1.0       # softmax probability for type; 1.0 for an analytic answer; fit-derived otherwise
    seconds: float = 0.0
    # The type stage's provenance when several registry entries were fused (mode "fused", reco_engine/cascade.py):
    # {"entries": [{"id", "mode", "weight", "probs", "subclass", "seconds"}], "subclass", "subclass_source"}; None otherwise.
    detail: Optional[Dict] = None


@dataclass
class Primitive:
    """One primitive of the cascade: what it is, where it is, how sure, how well it fits."""
    kind: str                                          # one of KINDS, or UNKNOWN
    quaternion: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    center_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    dims_mm: Dict[str, float] = field(default_factory=dict)
    u: Optional[Tuple[float, float]] = None            # sweep about the axis (rad), None = full body
    v: Optional[Tuple[float, float]] = None            # sphere latitude / torus minor range (rad)
    confidence: float = 0.0                            # product of the stage confidences
    residual: Optional[float] = None                   # share of the target left unexplained after this primitive
    grade: Optional[float] = None                      # two-way chamfer mean (mm), when graded in FreeCAD
    sources: List[StageSource] = field(default_factory=list)
    alternatives: List["Primitive"] = field(default_factory=list)   # the pruned top-k branches, best first
    label: Optional[str] = None                        # a name for the GUI / JSON, e.g. "HEA 500 top flange" (member sub-loop)
    parts: List["Primitive"] = field(default_factory=list)   # kind "composite": its two operands
    combine: Optional[str] = None                      # kind "composite": one of COMBINES

    @property
    def is_unknown(self) -> bool:
        return self.kind == UNKNOWN

    def defect(self, min_extent: float = 1e-9) -> Optional[str]:
        """Why this primitive cannot be built, or None. Checked before any SDF, IoU or FreeCAD build: the ABC batch of
        2026-09-30 crashed 104 runs on parameters that reached the builders unchecked - an inverted sweep (`sector needs
        u1 > u0`, 53 runs, a torus v range the fit left unclipped), box extents of zero ("height of box too small", 23),
        a void bounding box (10), a zero-norm axis (6); the torus radii are the torus guard's. Every kind's dims must be finite
        and above `min_extent`; a sweep u or v must be a proper interval (u at most 2 pi); the quaternion must be finite
        with a norm; a composite needs two buildable parts and a combine."""
        import math
        kind = self.kind
        if kind == "composite":
            if len(self.parts) != 2 or self.combine not in COMBINES:
                return f"composite needs two parts and a combine, got {len(self.parts)} / {self.combine!r}"
            for part in self.parts:
                d = part.defect(min_extent)
                if d:
                    return f"composite part: {d}"
            return None
        if kind not in KINDS and kind not in ("halfspace", UNKNOWN):
            return f"unknown kind {kind!r}"
        if kind == UNKNOWN:
            return "kind unknown"
        q = self.quaternion
        if len(q) != 4 or not all(math.isfinite(float(x)) for x in q) or sum(float(x) ** 2 for x in q) < 1e-12:
            return f"quaternion not a rotation: {tuple(q)}"
        if len(self.center_mm) != 3 or not all(math.isfinite(float(x)) for x in self.center_mm):
            return f"centre not finite: {tuple(self.center_mm)}"
        need = dict(box=("x_half", "y_half", "z_half"), sphere=("radius",), cylinder=("radius", "half_length"),
                    cone=("radius1", "radius2", "half_length"), torus=("major_radius", "minor_radius"), halfspace=())
        for key in need[kind]:
            v = self.dims_mm.get(key)
            if v is None or not math.isfinite(float(v)):
                return f"{kind} {key} missing or not finite"
            if kind == "cone" and key in ("radius1", "radius2"):
                if float(v) < 0.0:
                    return f"cone {key} negative: {v}"
            elif float(v) <= min_extent:
                return f"{kind} {key} too small: {v}"
        if kind == "cone" and max(float(self.dims_mm["radius1"]), float(self.dims_mm["radius2"])) <= min_extent:
            return "cone with both radii zero"
        # the relation of a torus's radii (spindle, horn, near-horn, oversized) is the torus guard's domain
        # (reco_engine/torus_guard.py), which names each case with its own reason; only positivity is checked here
        for name, rng, top in (("u", self.u, 2.0 * math.pi + 1e-9), ("v", self.v, None)):
            if rng is None:
                continue
            if len(rng) != 2 or not all(math.isfinite(float(x)) for x in rng):
                return f"{name} range not finite: {tuple(rng)}"
            lo, hi = float(rng[0]), float(rng[1])
            if hi - lo <= 1e-9:
                return f"{name} range empty or inverted: ({lo}, {hi})"
            if top is not None and hi - lo > top:
                return f"{name} range wider than a turn: ({lo}, {hi})"
        return None


@dataclass
class Operation:
    """One step of the residual loop: a primitive fused (add) or subtracted (cut)."""
    op: str                                            # one of OPS
    primitive: Primitive
    component_voxels: int = 0                          # size of the leftover component it was fitted to
    net_gain: float = 0.0                              # |R| before minus |R| after, in voxels
    accepted: bool = True
    freecad_valid: Optional[bool] = None               # the FreeCAD solid after this op is one valid solid
    recut_voxels: int = 0                              # voxels this cut covers that an earlier cut had already emptied


@dataclass
class ReconstructionResult:
    input_path: str
    operations: List[Operation] = field(default_factory=list)
    residual_voxels_pos: int = 0                       # material still missing at the end
    residual_voxels_neg: int = 0                       # material the model has that the input lacks
    recut_voxels: int = 0                              # voxels cut two or more times (the cut history, residual_loop)
    chamfer_mm: Optional[float] = None
    step_path: Optional[str] = None
    stl_path: Optional[str] = None
    # The exact judge (reco_field.exact_check, OCCT booleans against the input; None when not graded or failed):
    exact_iou: Optional[float] = None
    exact_method: Optional[str] = None                 # "occt" or "sampled (pitch ...)" (see residual_loop._exact_grade)
    exact_r_plus: Optional[float] = None               # missed material, input units^3
    exact_r_minus: Optional[float] = None              # invented material
    exact_r_plus_solids: Optional[int] = None          # pieces of missed material
    genus: Optional[float] = None                      # of the built model (per-shell Euler characteristic)
    input_genus: Optional[float] = None
    grid_pitch_over_tau: Optional[float] = None        # > 1: the loop's grid is coarser than the thinnest wall

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)

    @classmethod
    def from_dict(cls, d: dict) -> "ReconstructionResult":
        def prim(p: dict) -> Primitive:
            return Primitive(
                kind=p["kind"], quaternion=tuple(p["quaternion"]), center_mm=tuple(p["center_mm"]),
                dims_mm=dict(p["dims_mm"]), u=tuple(p["u"]) if p.get("u") else None,
                v=tuple(p["v"]) if p.get("v") else None, confidence=p.get("confidence", 0.0),
                residual=p.get("residual"), grade=p.get("grade"),
                sources=[StageSource(**s) for s in p.get("sources", [])],
                alternatives=[prim(a) for a in p.get("alternatives", [])], label=p.get("label"),
                parts=[prim(a) for a in p.get("parts", [])], combine=p.get("combine"))
        ops = [Operation(op=o["op"], primitive=prim(o["primitive"]), component_voxels=o.get("component_voxels", 0),
                         net_gain=o.get("net_gain", 0.0), accepted=o.get("accepted", True),
                         freecad_valid=o.get("freecad_valid"), recut_voxels=o.get("recut_voxels", 0))
               for o in d.get("operations", [])]
        rest = {k: d.get(k) for k in ("residual_voxels_pos", "residual_voxels_neg", "recut_voxels", "chamfer_mm", "step_path",
                                      "stl_path", "exact_iou", "exact_method", "exact_r_plus", "exact_r_minus", "exact_r_plus_solids",
                                      "genus", "input_genus", "grid_pitch_over_tau") if k in d}
        return cls(input_path=d["input_path"], operations=ops, **rest)
