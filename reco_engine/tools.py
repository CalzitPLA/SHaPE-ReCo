#!/usr/bin/env python3
"""
The type set of standard tools (the user's request 2026-09-28: "let's implement the use of standard tools, since we
often have details made by machines ... a chamfered or filleted box, a standard drilling hole etc."). A tool is its
GEOMETRY - a boolean tree of Part-native primitives (result_schema Primitive / composite, built by result_to_program
and reco_field/freecad_builder.py) with named parameters - never a part-specific feature (D-004 in spirit).

    tool = TOOLS["countersunk_hole"]
    prim = tool.build(dict(d=6.6, d_c=12.4, angle=90.0, length=10.0), R, origin)   # composite, world mm
    tool.volume(params)                                                          # the closed formula
    candidates = fit_piece(piece, V)            # [(tool name, params, R, origin)] from the piece's exact STEP faces

Every tool has a CANONICAL LOCAL FRAME: origin on the tool's axis at its ENTRY (where it is open to the outside / to
the model's surface), local +z pointing from the entry INTO the material (a hole's depth direction, a pocket's floor
direction); x the long side of a prismatic tool. `symmetry` names the proper rotations that map the tool onto itself
("Cinf_z": any turn about z, the revolved tools; "C2_z": a half turn about z; "C1": none) - the pattern transfer
(feature_stage.best_transfer) needs it to know when a spin is free.

The catalogue (`CLASSES` is the network's class list: the catalogue in this order plus "none"):

  holes (drilling; ISO 273 clearance diameters, DIN 74 / ISO 15065 countersinks 90 deg, DIN 974 counterbores, the
  drill point 118 deg of DIN 1412 as the default):
    through_hole        cylinder d x length, open at both ends
    blind_hole          cylinder d x depth + drill-point cone (point_angle)
    countersunk_hole    cone d_c -> d (angle) + cylinder d x length
    counterbored_hole   cylinder d_cb x depth_cb + cylinder d x length (d_cb > d)
    stepped_hole        cylinder d1 x length1 + cylinder d2 x length2 (d1 < d2: a stepped / tapped-core bore)
  milled pockets and slots:
    circular_pocket     flat-bottom cylinder d x depth
    rect_pocket         L x W x depth with corner radius r on its four depth-parallel edges
    slot                L x W x depth, both ends half cylinders (r = W / 2; DIN 6885 keyways are this shape)
    through_slot        the slot open at both depth ends
    open_pocket         L x W x depth, open at +x, corner radius r on the two closed corners only
  edges:
    chamfer             triangular prism, legs a, b along x and y, length along z
    fillet              concave edge: material ADDED in the corner - square r x r minus the quarter cylinder
    round               convex edge: material CUT from the corner - the same geometry as the fillet
    chamfered_box       a box whose chosen edges carry chamfers of leg c
    filleted_box        a box whose chosen edges carry rounds of radius r
  turned shapes:
    groove              ring of rectangular section: R_out, R_in, width (a recess on a shaft or in a bore)
    relief              an undercut at a shoulder (DIN 509 form without its radii): the groove's geometry, a
                        separate class because it sits against a shoulder face - the classifier tells them apart
    taper               cone frustum r1 (entry) -> r2, length

Measured facts and choices are in the tests (tests/test_tools.py): every tool's volume against its closed formula on
the built solid, and every fitter against a piece made by FreeCAD with known parameters, tilted and off-centre.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco_engine.result_schema import Primitive, StageSource
from reco_engine.result_to_program import quaternion_of, rotation_matrix

EDGE_IDS = [(k, si, sj) for k in range(3) for si in (-1, 1) for sj in (-1, 1)]   # a box's 12 edges: axis, corner


# ------------------------------------------------------------------------------------------------ primitives in a frame

def _src(mode: str = "tool") -> List[StageSource]:
    return [StageSource(s, mode, 1.0) for s in ("type", "rotation", "dimensions", "position")]


def _cyl(R, c, r: float, z0: float, z1: float) -> Primitive:
    """A cylinder of radius r on the frame's z axis from z0 to z1 (z0 < z1)."""
    return Primitive(kind="cylinder", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in c + R[:, 2] * (z0 + z1) / 2),
                     dims_mm=dict(radius=float(r), half_length=float((z1 - z0) / 2)), confidence=1.0, sources=_src())


def _cone(R, c, r0: float, r1: float, z0: float, z1: float) -> Primitive:
    """A cone frustum on the frame's z axis: radius r0 at z0, r1 at z1 (z0 < z1)."""
    return Primitive(kind="cone", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in c + R[:, 2] * (z0 + z1) / 2),
                     dims_mm=dict(radius1=float(max(r0, 0.0)), radius2=float(max(r1, 0.0)), half_length=float((z1 - z0) / 2)),
                     confidence=1.0, sources=_src())


def _box(R, c, lo, hi) -> Primitive:
    """The box [lo, hi] of the frame's local coordinates."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Primitive(kind="box", quaternion=quaternion_of(R), center_mm=tuple(float(v) for v in c + R @ ((lo + hi) / 2)),
                     dims_mm=dict(zip(("x_half", "y_half", "z_half"), (float(v) for v in (hi - lo) / 2))),
                     confidence=1.0, sources=_src())


def _half(R, c, n_local, p_local) -> Primitive:
    """The half space {x : n . x <= n . p} in local coordinates (primitive "halfspace": local +z is its normal)."""
    n = R @ np.asarray(n_local, float)
    n /= np.linalg.norm(n)
    u = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    F = np.stack([u, np.cross(n, u), n], axis=1)
    return Primitive(kind="halfspace", quaternion=quaternion_of(F), center_mm=tuple(float(v) for v in c + R @ np.asarray(p_local, float)),
                     dims_mm={}, confidence=1.0, sources=_src())


def _op(a: Primitive, b: Primitive, combine: str, label: Optional[str] = None) -> Primitive:
    return Primitive(kind="composite", center_mm=tuple(a.center_mm), dims_mm={}, parts=[a, b], combine=combine,
                     label=label, confidence=1.0, sources=list(a.sources))


def _fuse_all(parts: Sequence[Primitive]) -> Primitive:
    out = parts[0]
    for p in parts[1:]:
        out = _op(out, p, "fuse")
    return out


def _rot_z(R: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    Z = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
    return R @ Z


def _corner_round(R, c, lo, hi, k: int, si: int, sj: int, r: float, over: float) -> Primitive:
    """What rounds the box edge along local axis k at corner (si, sj) of the two other axes: the r x r corner square
    (running `over` past the box along k) minus the cylinder on its inner corner."""
    i, j = [m for m in range(3) if m != k]
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ei = hi[i] if si > 0 else lo[i]
    ej = hi[j] if sj > 0 else lo[j]
    slo, shi = np.empty(3), np.empty(3)
    slo[i], shi[i] = sorted((ei - si * r, ei + si * over))
    slo[j], shi[j] = sorted((ej - sj * r, ej + sj * over))
    slo[k], shi[k] = lo[k] - over, hi[k] + over
    sq = _box(R, c, slo, shi)
    axis_pt = np.zeros(3)
    axis_pt[i], axis_pt[j] = ei - si * r, ej - sj * r
    Rk = np.stack([R[:, i], R[:, j], R[:, k]], axis=1) if k != 2 else R
    if np.linalg.det(Rk) < 0:
        Rk = np.stack([R[:, j], R[:, i], R[:, k]], axis=1)
    ck = c + R @ axis_pt
    cy = _cyl(Rk, ck, r, lo[k] - 2 * over, hi[k] + 2 * over)
    return _op(sq, cy, "cut")


def _corner_chamfer(R, c, lo, hi, k: int, si: int, sj: int, leg: float, over: float) -> Primitive:
    """What chamfers the box edge along local axis k at corner (si, sj): the leg x leg corner square (running `over`
    past the box along k) intersected with the half space beyond the 45-degree chamfer plane."""
    i, j = [m for m in range(3) if m != k]
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ei = hi[i] if si > 0 else lo[i]
    ej = hi[j] if sj > 0 else lo[j]
    slo, shi = np.empty(3), np.empty(3)
    slo[i], shi[i] = sorted((ei - si * leg, ei + si * over))
    slo[j], shi[j] = sorted((ej - sj * leg, ej + sj * over))
    slo[k], shi[k] = lo[k] - over, hi[k] + over
    n = np.zeros(3)
    n[i], n[j] = -si, -sj                                        # keeps the corner side: n . x <= n . p
    p = np.zeros(3)
    p[i], p[j] = ei - si * leg, ej
    return _op(_box(R, c, slo, shi), _half(R, c, n, p), "common")


# ------------------------------------------------------------------------------------------------ the catalogue

@dataclass
class Param:
    name: str
    unit: str                           # "mm" | "deg" | "-"
    lo: float
    hi: float
    doc: str = ""


@dataclass
class Tool:
    name: str
    family: str                         # "hole" | "pocket" | "edge" | "turned"
    params: List[Param]
    build_fn: Callable                  # (params, R, origin, over) -> Primitive
    volume_fn: Callable                 # (params) -> mm^3
    symmetry: str                       # "Cinf_z" | "C2_z" | "C1"
    standard: str = ""
    doc: str = ""

    def build(self, params: Dict, R=None, origin=(0.0, 0.0, 0.0), over: float = 0.0) -> Primitive:
        """The tool as a composite in world mm; `over`: how far the open ends run past the entry (a cutter needs it
        against coplanar faces; 0 for the exact solid)."""
        R = np.eye(3) if R is None else np.asarray(R, float)
        p = self.build_fn(dict(params), R, np.asarray(origin, float), float(over))
        p.label = self.name
        return p

    def volume(self, params: Dict) -> float:
        return float(self.volume_fn(dict(params)))

    def valid(self, params: Dict) -> bool:
        for q in self.params:
            v = params.get(q.name)
            if v is None:
                return False
            vals = np.atleast_1d(np.asarray(v, float))
            if len(vals) and not np.all((vals >= q.lo) & (vals <= q.hi)):
                return False
        return True


def _hole_build(segments_fn, open_bottom: bool = False):
    def build(p, R, c, over):
        segs = segments_fn(p)                                  # [(kind, z0, z1, r0, r1)] from the entry downwards
        parts = []
        for n, (kind, z0, z1, r0, r1) in enumerate(segs):
            z0e = z0 - (over if n == 0 else 0.0)                  # past the entry: a cone goes on with its slope
            z1e = z1 + (over if (n == len(segs) - 1 and (open_bottom or p.get("_open_bottom"))) else 0.0)
            if kind == "cyl":
                parts.append(_cyl(R, c, r0, z0e, z1e))
            else:
                k = (r1 - r0) / (z1 - z0)
                parts.append(_cone(R, c, r0 + k * (z0e - z0), r1 + k * (z1e - z1), z0e, z1e))
        return _fuse_all(parts)
    return build


def _through(p):
    return [("cyl", 0.0, p["length"], p["d"] / 2, p["d"] / 2)]


def _blind(p):
    r = p["d"] / 2
    h = r / math.tan(math.radians(p["point_angle"]) / 2)
    return [("cyl", 0.0, p["depth"], r, r), ("cone", p["depth"], p["depth"] + h, r, 0.0)]


def _csk(p):
    r, rc = p["d"] / 2, p["d_c"] / 2
    h = (rc - r) / math.tan(math.radians(p["angle"]) / 2)
    return [("cone", 0.0, h, rc, r), ("cyl", h, h + p["length"], r, r)]


def _two(p, a="d_cb", la="depth_cb", b="d", lb="length"):
    return [("cyl", 0.0, p[la], p[a] / 2, p[a] / 2), ("cyl", p[la], p[la] + p[lb], p[b] / 2, p[b] / 2)]


def _cbore(p):
    return _two(p)


def _stepped(p):
    return _two(p, "d1", "length1", "d2", "length2")


def _circ(p):
    return [("cyl", 0.0, p["depth"], p["d"] / 2, p["d"] / 2)]


def _taper(p):
    return [("cone", 0.0, p["length"], p["r1"], p["r2"])]


def _vol_segments(segs) -> float:
    return sum(math.pi * (z1 - z0) * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0 for _, z0, z1, r0, r1 in segs)


def _pocket_build(corners_fn, open_x: bool = False, open_bottom: bool = False):
    def build(p, R, c, over):
        L, W, D = p["L"], p["W"], p["depth"]
        lo, hi = np.array([-L / 2, -W / 2, -over]), np.array([L / 2, W / 2, D + (over if open_bottom else 0.0)])
        if open_x:
            hi[0] += over
        out = _box(R, c, lo, hi)
        blo, bhi = np.array([-L / 2, -W / 2, -over]), np.array([L / 2, W / 2, D])
        for si, sj, r in corners_fn(p):
            if r > 0:
                out = _op(out, _corner_round(R, c, blo, bhi, 2, si, sj, r, max(over, 1e-3 * r)), "cut")
        return out
    return build


def _rect_corners(p):
    return [(si, sj, p["r"]) for si in (-1, 1) for sj in (-1, 1)]


def _slot_corners(p):
    return [(si, sj, p["W"] / 2) for si in (-1, 1) for sj in (-1, 1)]


def _open_corners(p):
    return [(-1, sj, p["r"]) for sj in (-1, 1)]


def _pocket_volume(p, corners):
    return p["depth"] * (p["L"] * p["W"] - sum((4 - math.pi) / 4 * r * r for _, _, r in corners))


def _edge_build(kind):
    def build(p, R, c, over):
        if kind == "chamfer":                                    # legs along x and y from the edge, the edge on z
            a, b, L = p["a"], p["b"], p["length"]
            box = _box(R, c, (0.0, 0.0, -over), (a, b, L + over))
            return _op(box, _half(R, c, (b, a, 0.0), (a, 0.0, 0.0)), "common")
        r, L = p["r"], p["length"]                               # the corner square [0, r]^2, the cylinder at (r, r)
        box = _box(R, c, (0.0, 0.0, -over), (r, r, L + over))
        return _op(box, _cyl(R, c + R @ np.array([r, r, 0.0]), r, -2 * over - 1e-3 * r, L + 2 * over + 1e-3 * r), "cut")
    return build


def _box_edges_build(kind):
    def build(p, R, c, over):
        lo, hi = -np.asarray(p["size"], float) / 2, np.asarray(p["size"], float) / 2
        out = _box(R, c, lo, hi)
        for e in p["edges"]:
            k, si, sj = EDGE_IDS[int(e)]
            piece = (_corner_chamfer if kind == "chamfer" else _corner_round)(R, c, lo, hi, k, si, sj, p["leg"],
                                                                             max(over, 1e-3 * p["leg"]))
            out = _op(out, piece, "cut")
        return out
    return build


def _box_edges_volume(kind):
    """The box minus each finished edge's cross-section x its length - exact for edges without a common vertex (at a
    shared corner the two cuts overlap: leg^3 / 3 for two chamfers, measured 2026-09-28 in FreeCAD)."""
    def vol(p):
        size = np.asarray(p["size"], float)
        cut = (0.5 if kind == "chamfer" else (4 - math.pi) / 4) * p["leg"] ** 2
        return float(np.prod(size) - sum(cut * size[EDGE_IDS[int(e)][0]] for e in p["edges"]))
    return vol


def _ring_build(p, R, c, over):
    w = p["width"]
    return _op(_cyl(R, c, p["R_out"], 0.0, w), _cyl(R, c, p["R_in"], -over - 1e-3 * w, w + over + 1e-3 * w), "cut")


def _ring_volume(p):
    return math.pi * (p["R_out"] ** 2 - p["R_in"] ** 2) * p["width"]


MM = 1e4
TOOLS: Dict[str, Tool] = {}


def _add(t: Tool):
    TOOLS[t.name] = t


_add(Tool("through_hole", "hole", [Param("d", "mm", 0.1, MM), Param("length", "mm", 0.01, MM)],
          _hole_build(_through, open_bottom=True), lambda p: _vol_segments(_through(p)), "Cinf_z", "ISO 273 clearance diameters"))
_add(Tool("blind_hole", "hole", [Param("d", "mm", 0.1, MM), Param("depth", "mm", 0.01, MM),
                                 Param("point_angle", "deg", 60.0, 179.0, "118 (DIN 1412)")],
          _hole_build(_blind), lambda p: _vol_segments(_blind(p)), "Cinf_z", "DIN 1412 drill point 118 deg"))
_add(Tool("countersunk_hole", "hole", [Param("d", "mm", 0.1, MM), Param("d_c", "mm", 0.2, MM),
                                       Param("angle", "deg", 60.0, 179.0, "90 (ISO 15065)"), Param("length", "mm", 0.0, MM)],
          _hole_build(_csk), lambda p: _vol_segments(_csk(p)), "Cinf_z", "DIN 74 / ISO 15065 countersinks, 90 deg"))
_add(Tool("counterbored_hole", "hole", [Param("d_cb", "mm", 0.2, MM), Param("depth_cb", "mm", 0.01, MM),
                                        Param("d", "mm", 0.1, MM), Param("length", "mm", 0.0, MM)],
          _hole_build(_cbore), lambda p: _vol_segments(_cbore(p)), "Cinf_z", "DIN 974 counterbores"))
_add(Tool("stepped_hole", "hole", [Param("d1", "mm", 0.1, MM), Param("length1", "mm", 0.01, MM),
                                   Param("d2", "mm", 0.1, MM), Param("length2", "mm", 0.01, MM)],
          _hole_build(_stepped), lambda p: _vol_segments(_stepped(p)), "Cinf_z", "a stepped / tapped-core bore"))
_add(Tool("circular_pocket", "pocket", [Param("d", "mm", 0.1, MM), Param("depth", "mm", 0.01, MM)],
          _hole_build(_circ), lambda p: _vol_segments(_circ(p)), "Cinf_z", "flat-bottom (end mill)"))
_add(Tool("rect_pocket", "pocket", [Param("L", "mm", 0.1, MM), Param("W", "mm", 0.1, MM), Param("depth", "mm", 0.01, MM),
                                    Param("r", "mm", 0.0, MM)],
          _pocket_build(_rect_corners), lambda p: _pocket_volume(p, _rect_corners(p)), "C2_z", "milled, corner radius = cutter radius"))
_add(Tool("slot", "pocket", [Param("L", "mm", 0.1, MM), Param("W", "mm", 0.1, MM), Param("depth", "mm", 0.01, MM)],
          _pocket_build(_slot_corners), lambda p: _pocket_volume(p, _slot_corners(p)), "C2_z", "DIN 6885 keyway form"))
_add(Tool("through_slot", "pocket", [Param("L", "mm", 0.1, MM), Param("W", "mm", 0.1, MM), Param("depth", "mm", 0.01, MM)],
          _pocket_build(_slot_corners, open_bottom=True), lambda p: _pocket_volume(p, _slot_corners(p)), "C2_z",
          "slot open at both depth ends"))
_add(Tool("open_pocket", "pocket", [Param("L", "mm", 0.1, MM), Param("W", "mm", 0.1, MM), Param("depth", "mm", 0.01, MM),
                                    Param("r", "mm", 0.0, MM)],
          _pocket_build(_open_corners, open_x=True), lambda p: _pocket_volume(p, _open_corners(p)), "C1",
          "open at +x, two rounded corners"))
_add(Tool("chamfer", "edge", [Param("a", "mm", 0.01, MM), Param("b", "mm", 0.01, MM), Param("length", "mm", 0.01, MM)],
          _edge_build("chamfer"), lambda p: p["a"] * p["b"] * p["length"] / 2, "C1", "ISO 13715 edge, C x 45 deg when a = b"))
_add(Tool("fillet", "edge", [Param("r", "mm", 0.01, MM), Param("length", "mm", 0.01, MM)],
          _edge_build("round"), lambda p: (4 - math.pi) / 4 * p["r"] ** 2 * p["length"], "C1", "concave edge (an add)"))
_add(Tool("round", "edge", [Param("r", "mm", 0.01, MM), Param("length", "mm", 0.01, MM)],
          _edge_build("round"), lambda p: (4 - math.pi) / 4 * p["r"] ** 2 * p["length"], "C1", "convex edge (a cut)"))
_add(Tool("chamfered_box", "edge", [Param("size", "mm", 0.01, MM), Param("edges", "-", 0, 11), Param("leg", "mm", 0.01, MM)],
          _box_edges_build("chamfer"), _box_edges_volume("chamfer"), "C1", "box, chosen edges C x 45 deg"))
_add(Tool("filleted_box", "edge", [Param("size", "mm", 0.01, MM), Param("edges", "-", 0, 11), Param("leg", "mm", 0.01, MM)],
          _box_edges_build("round"), _box_edges_volume("round"), "C1", "box, chosen edges rounded"))
_add(Tool("groove", "turned", [Param("R_out", "mm", 0.1, MM), Param("R_in", "mm", 0.0, MM), Param("width", "mm", 0.01, MM)],
          _ring_build, _ring_volume, "Cinf_z", "a recess of rectangular section"))
_add(Tool("relief", "turned", [Param("R_out", "mm", 0.1, MM), Param("R_in", "mm", 0.0, MM), Param("width", "mm", 0.01, MM)],
          _ring_build, _ring_volume, "Cinf_z", "DIN 509 undercut at a shoulder (without its radii)"))
_add(Tool("taper", "turned", [Param("r1", "mm", 0.0, MM), Param("r2", "mm", 0.0, MM), Param("length", "mm", 0.01, MM)],
          _hole_build(_taper), lambda p: _vol_segments(_taper(p)), "Cinf_z", "a cone frustum"))

CLASSES: Tuple[str, ...] = tuple(TOOLS) + ("none",)


# ------------------------------------------------------------------------------------------------ fitting from exact faces
# The piece's exact faces (reco_field.exact_check face records: type, area, normal (out of the piece), point (inside the
# face), axis / center / apex / radius / semi_angle of its surface, convex, origin "input" | "model" | "both" | "none").
# A face of origin input / both is a real surface of the part; a face of origin model only closes the piece (an open end).

REAL = ("input", "both")
AX_COS = math.cos(math.radians(1.0))


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _frame_z(z: np.ndarray, x_hint: Optional[np.ndarray] = None) -> np.ndarray:
    z = _unit(z)
    x = x_hint if x_hint is not None else (np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0]))
    x = np.asarray(x, float) - z * (np.asarray(x, float) @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


def revolved_segments(faces: Sequence[Dict], tol: float = 1e-3) -> Optional[Dict]:
    """The coaxial stack of a revolved piece: dict(axis, base (a point on the axis), segments [dict(kind, t0, t1, r0,
    r1, inner)] sorted along the axis, caps [dict(t, origin, sign, area)] the planes square to the axis). The largest
    real cylinder or cone sets the axis; a face joins when its axis is parallel within 1 degree and on the same line
    (within `tol` x its radius). Faces of one surface (a seam splits a full cylinder in two) are merged by summing
    their areas. A cylinder's height is area / (2 pi r), its middle the face's mid point; a cone's height is
    area cos(a) / (2 pi rho_m), rho_m its radius at the mid point, its radii rho_m -+ h tan(a) / 2, growing away from
    the apex. `inner`: the piece's material lies outside the surface (a ring's bore). None without a real cylinder or
    cone."""
    curved = [f for f in faces if f["type"] in ("cylinder", "cone") and "axis" in f]
    real = [f for f in curved if f["origin"] in REAL]
    if not real:
        return None
    ref = max(real, key=lambda f: f["area"])
    a = _unit(ref["axis"])
    base = np.asarray(ref["center"], float)

    def on_axis(f):
        if abs(_unit(f["axis"]) @ a) < AX_COS:
            return False
        off = np.asarray(f["center"], float) - base
        off = off - a * (off @ a)
        return np.linalg.norm(off) <= tol * max(float(f.get("radius") or 1.0), 1.0) + 1e-6
    groups: List[Dict] = []
    for f in curved:
        if not on_axis(f):
            continue
        pm = np.asarray(f["point"], float)
        tm = float((pm - base) @ a)
        rho = float(np.linalg.norm((pm - base) - a * tm))
        key = dict(kind="cyl" if f["type"] == "cylinder" else "cone", tm=tm, rho=rho, inner=not bool(f.get("convex", True)),
                   real=f["origin"] in REAL)
        for g in groups:
            if g["kind"] == key["kind"] and abs(g["tm"] - tm) <= 1e-6 * (1 + abs(tm)) + 1e-6 and \
                    abs(g["rho"] - rho) <= 1e-6 * (1 + rho) + 1e-6 and g["inner"] == key["inner"] and g["real"] == key["real"]:
                g["area"] += f["area"]
                break
        else:
            groups.append(dict(key, area=f["area"], face=f))
    segs = []
    for g in groups:
        f = g["face"]
        if g["kind"] == "cyl":
            r = float(f["radius"])
            h = g["area"] / (2 * math.pi * r)
            r0 = r1 = r
        else:
            al = float(f["semi_angle"])
            h = g["area"] * math.cos(al) / (2 * math.pi * max(g["rho"], 1e-12))
            s = 1.0 if (np.asarray(f["point"], float) - np.asarray(f["apex"], float)) @ a > 0 else -1.0
            dr = s * h * math.tan(al) / 2
            r0, r1 = g["rho"] - dr, g["rho"] + dr
        segs.append(dict(kind=g["kind"], t0=g["tm"] - h / 2, t1=g["tm"] + h / 2, r0=r0, r1=r1, inner=g["inner"],
                         real=g["real"]))
    caps = []
    for f in faces:
        if f["type"] == "plane" and abs(_unit(f["normal"]) @ a) >= AX_COS:
            caps.append(dict(t=float((np.asarray(f["point"], float) - base) @ a), origin=f["origin"],
                             sign=float(np.sign(_unit(f["normal"]) @ a)), area=f["area"]))
    return dict(axis=a, base=base, segments=sorted(segs, key=lambda q: q["t0"]), caps=caps)


@dataclass
class Fit:
    """One interpretation of a piece as a catalogue tool: its name, parameters, canonical frame (columns x, y, z in the
    world) and origin (the entry point on the tool's axis), and how it was read ("step faces" | "fit")."""
    tool: str
    params: Dict
    R: np.ndarray
    origin: np.ndarray
    how: str = "step faces"

    def build(self, over: float = 0.0) -> Primitive:
        return TOOLS[self.tool].build(self.params, self.R, self.origin, over)


def _end_closed(stack: Dict, t_end: float, sign: float, scale: float) -> Optional[bool]:
    """Is the stack's end at t_end (sign: +1 the upper end, -1 the lower) closed by a real plane of the input? True /
    False (open: a model face or nothing), judged by the caps within 1e-3 x the piece's length of that end."""
    at = [c for c in stack["caps"] if abs(c["t"] - t_end) <= 1e-3 * scale + 1e-6]
    if any(c["origin"] in REAL for c in at):
        return True
    return False


def fit_revolved(faces: Sequence[Dict], op: str = "cut") -> List[Fit]:
    """Holes, circular pockets, tapers, grooves and reliefs from the coaxial stack (revolved_segments): the ends are
    open (entry) or closed (by a real plane, or by a cone that runs to a point); the tool's z runs from the open end
    into the material. Ranked interpretations, the most specific first; [] when the stack fits none."""
    st = revolved_segments(faces)
    if st is None or not st["segments"]:
        return []
    a, base = st["axis"], st["base"]
    segs = st["segments"]
    t_lo, t_hi = min(q["t0"] for q in segs), max(q["t1"] for q in segs)
    scale = t_hi - t_lo
    outer = [q for q in segs if not q["inner"]]
    inner = [q for q in segs if q["inner"] and q["real"]]
    out: List[Fit] = []
    if inner and outer:                                          # a ring: groove / relief (its outside may be the model's)
        o, i = max(outer, key=lambda q: q["r0"]), max(inner, key=lambda q: q["r0"])
        if o["kind"] == i["kind"] == "cyl":
            R = _frame_z(a)
            origin = base + a * min(o["t0"], i["t0"])
            prm = dict(R_out=o["r0"], R_in=i["r0"], width=max(o["t1"], i["t1"]) - min(o["t0"], i["t0"]))
            ring_end = math.pi * (o["r0"] ** 2 - i["r0"] ** 2)
            shoulder = any(c["origin"] in REAL and c["area"] > 1.2 * ring_end for c in st["caps"])
            names = ("relief", "groove") if shoulder else ("groove", "relief")
            out += [Fit(n, dict(prm), R, origin) for n in names]
        return out
    outer = [q for q in outer if q["real"]]                     # a hole's stack: real surfaces only
    if not outer:
        return []
    closed_lo = _end_closed(st, t_lo, -1, scale) or (outer[0]["kind"] == "cone" and min(outer[0]["r0"], outer[0]["r1"]) < 1e-3 * scale
                                                     and outer[0]["r0"] < outer[0]["r1"])
    closed_hi = _end_closed(st, t_hi, 1, scale) or (outer[-1]["kind"] == "cone" and outer[-1]["r1"] < 1e-3 * scale)
    if closed_lo and not closed_hi:
        flip = True                                              # the entry at the upper end
    elif closed_hi and not closed_lo:
        flip = False
    else:                                                        # both open (through) or both closed: the wider end
        r_lo, r_hi = outer[0]["r0"], outer[-1]["r1"]
        flip = r_hi > r_lo + 1e-9
    z = -a if flip else a
    t_entry = t_hi if flip else t_lo
    origin = base + a * t_entry
    R = _frame_z(z)
    # the segments in the tool's z (from the entry downwards)
    zs = []
    for q in (reversed(outer) if flip else outer):
        z0, z1 = (t_entry - q["t1"], t_entry - q["t0"]) if flip else (q["t0"] - t_entry, q["t1"] - t_entry)
        r0, r1 = (q["r1"], q["r0"]) if flip else (q["r0"], q["r1"])
        zs.append((q["kind"], z0, z1, r0, r1))
    bottom_closed = closed_lo if flip else closed_hi
    through = not closed_lo and not closed_hi
    kinds = [q[0] for q in zs]
    L = zs[-1][2] - zs[0][1]
    if kinds == ["cyl"]:
        d = 2 * zs[0][3]
        if through:
            out.append(Fit("through_hole", dict(d=d, length=L), R, origin))
        out.append(Fit("circular_pocket", dict(d=d, depth=L), R, origin))
    elif kinds == ["cone"]:
        out.append(Fit("taper", dict(r1=zs[0][3], r2=zs[0][4], length=L), R, origin))
    elif kinds == ["cyl", "cone"] and zs[1][4] < 1e-3 * max(L, 1e-9) + 1e-6:
        r, h = zs[0][3], zs[1][2] - zs[1][1]
        out.append(Fit("blind_hole", dict(d=2 * r, depth=zs[0][2] - zs[0][1],
                                          point_angle=2 * math.degrees(math.atan2(r, h))), R, origin))
    elif kinds[:2] == ["cone", "cyl"] and zs[0][3] > zs[0][4]:
        rc, r, h = zs[0][3], zs[1][3], zs[0][2] - zs[0][1]
        out.append(Fit("countersunk_hole", dict(d=2 * r, d_c=2 * rc, angle=2 * math.degrees(math.atan2(rc - r, h)),
                                                length=zs[1][2] - zs[1][1]), R, origin))
    elif kinds == ["cyl", "cyl"]:
        (_, a0, a1, ra, _), (_, b0, b1, rb, _) = zs
        if ra > rb:
            out.append(Fit("counterbored_hole", dict(d_cb=2 * ra, depth_cb=a1 - a0, d=2 * rb, length=b1 - b0), R, origin))
        out.append(Fit("stepped_hole", dict(d1=2 * ra, length1=a1 - a0, d2=2 * rb, length2=b1 - b0), R, origin))
    for f in out:
        f.params["_through"] = bool(through)
        f.params["_bottom_closed"] = bool(bottom_closed)
    return out


def _plane_box(faces: Sequence[Dict], V: np.ndarray):
    """(R, centre, half) of the piece's planes' box (feature_stage._box_of_planes: the frame of the largest real planes,
    sides at the input's real planes, open sides by the vertices); None without planes."""
    from reco_engine.feature_stage import _box_of_planes
    b = _box_of_planes([f for f in faces if f["type"] == "plane"], V)
    if b is None:
        return None
    return rotation_matrix(b), np.asarray(b.center_mm, float), np.array([b.dims_mm["x_half"], b.dims_mm["y_half"], b.dims_mm["z_half"]])


def _box_ends(faces, R, c, half, scale):
    """closed[axis][0 / 1]: the box's -/+ side along each local axis is a real plane of the input."""
    closed = [[False, False] for _ in range(3)]
    for f in faces:
        if f["type"] != "plane" or f["origin"] not in REAL:
            continue
        n = _unit(f["normal"])
        loc = R.T @ n
        k = int(np.argmax(np.abs(loc)))
        if abs(loc[k]) < AX_COS:
            continue
        t = float(R[:, k] @ (np.asarray(f["point"], float) - c))
        side = 1 if loc[k] > 0 else 0
        if abs(abs(t) - half[k]) <= 1e-3 * scale + 1e-6:
            closed[k][side] = True
    return closed


def _rounded_corners(faces, R, c, half, k, tol=0.03):
    """{(si, sj): r} of the box's edges along local axis k rounded by a real cylinder tangent to both walls."""
    i, j = [m for m in range(3) if m != k]
    out = {}
    for f in faces:
        if f["type"] != "cylinder" or f["origin"] not in REAL or abs(_unit(f["axis"]) @ R[:, k]) < AX_COS:
            continue
        r = float(f["radius"])
        q = R.T @ (np.asarray(f["center"], float) - c)
        for si in (-1, 1):
            for sj in (-1, 1):
                di, dj = si * half[i] - q[i], sj * half[j] - q[j]
                if abs(abs(di) - r) <= tol * r and abs(abs(dj) - r) <= tol * r and di * si > 0 and dj * sj > 0:
                    out[(si, sj)] = r
    return out


def fit_prismatic(faces: Sequence[Dict], V: np.ndarray, op: str = "cut") -> List[Fit]:
    """Pockets and slots from the planes' box and the cylinders that round its depth-parallel edges: the depth axis is
    the one with an open end (a pocket's mouth) - both ends open: a through slot / pocket - and the tool's z runs from
    the mouth to the floor, x along the longer side. Ranked interpretations."""
    pb = _plane_box(faces, V)
    if pb is None:
        return []
    R0, c, half = pb
    scale = 2 * float(half.max())
    closed = _box_ends(faces, R0, c, half, scale)
    open_axes = [k for k in range(3) if not all(closed[k])]
    if not open_axes:
        return []

    def depth_rank(k):                                          # a mouth opposite a floor first, the floor the biggest
        return (all(not s for s in closed[k]), -np.prod([2 * half[m] for m in range(3) if m != k]))
    out: List[Fit] = []
    for k in sorted(open_axes, key=depth_rank)[:2]:
        i, j = [m for m in range(3) if m != k]
        through = not closed[k][0] and not closed[k][1]
        mouth = 0 if not closed[k][0] else 1                     # the open side along k
        z = R0[:, k] * (1.0 if mouth == 0 else -1.0)
        x_ax, y_ax = (i, j) if half[i] >= half[j] else (j, i)
        x = R0[:, x_ax]
        corners = _rounded_corners(faces, R0, c, half, k)
        # the open side along x (an open pocket's end)
        x_open = [s for s, side in ((-1, 0), (1, 1)) if not closed[x_ax][side]]
        if len(x_open) == 1 and x_open[0] < 0:
            x = -x
        R = _frame_z(z, x)
        origin = c - z * half[k]
        L, W, D = 2 * half[x_ax], 2 * half[y_ax], 2 * half[k]
        radii = sorted(corners.values())
        # the corners in the tool's frame: which x end they sit on
        loc = {}
        for (si, sj), r in corners.items():
            e = np.zeros(3)
            e[i], e[j] = si * half[i], sj * half[j]
            loc[(int(np.sign((R0 @ e) @ R[:, 0])), int(np.sign((R0 @ e) @ R[:, 1])))] = r
        if len(radii) == 4 and max(radii) - min(radii) <= 0.03 * max(radii):
            r = float(np.mean(radii))
            if abs(r - W / 2) <= 0.03 * W / 2:
                out.append(Fit("through_slot" if through else "slot", dict(L=L, W=W, depth=D), R, origin))
            out.append(Fit("rect_pocket", dict(L=L, W=W, depth=D, r=r), R, origin))
        elif len(radii) == 2 and len(x_open) == 1 and all(xs < 0 for xs, _ in loc):
            out.append(Fit("open_pocket", dict(L=L, W=W, depth=D, r=float(np.mean(radii))), R, origin))
        elif not radii:
            if len(x_open) == 1:
                out.append(Fit("open_pocket", dict(L=L, W=W, depth=D, r=0.0), R, origin))
            out.append(Fit("rect_pocket", dict(L=L, W=W, depth=D, r=0.0), R, origin))
    return out


def fit_edge(faces: Sequence[Dict], V: np.ndarray, op: str = "cut") -> List[Fit]:
    """Fillet (an add) / round (a cut): the piece is the r x r corner square minus a quarter cylinder - one real
    cylinder, its axis the edge; chamfer: a triangular prism - one real plane inclined to the two model planes that
    meet at the edge. Frame: z along the edge, x and y along the two walls from the edge line, origin on the edge at
    the piece's lower end."""
    V = np.asarray(V, float)
    out: List[Fit] = []
    cyls = [f for f in faces if f["type"] == "cylinder" and f["origin"] in REAL]
    if len(cyls) >= 1 and all(abs(_unit(f["axis"]) @ _unit(cyls[0]["axis"])) >= AX_COS for f in cyls):
        f = max(cyls, key=lambda q: q["area"])
        a, q, r = _unit(f["axis"]), np.asarray(f["center"], float), float(f["radius"])
        rel = V - q
        t = rel @ a
        radial = rel - np.outer(t, a)
        far = radial[np.argmax(np.linalg.norm(radial, axis=1))]      # the edge line lies r sqrt 2 from the axis
        if abs(np.linalg.norm(far) - r * math.sqrt(2)) <= 0.05 * r:
            d = _unit(far)
            # the walls: the two directions at 45 degrees to d in the plane square to a
            w1 = _unit(math.cos(math.pi / 4) * d + math.sin(math.pi / 4) * np.cross(a, d))
            w2 = _unit(math.cos(math.pi / 4) * d - math.sin(math.pi / 4) * np.cross(a, d))
            x, y = -w1, -w2                                           # from the edge towards the cylinder's side
            if np.linalg.det(np.stack([x, y, a], axis=1)) < 0:
                x, y = y, x
            R = np.stack([x, y, a], axis=1)
            edge = q + far
            origin = edge + a * float(t.min())
            name = "fillet" if op == "add" else "round"
            out.append(Fit(name, dict(r=r, length=float(t.max() - t.min())), R, origin))
    planes = [f for f in faces if f["type"] == "plane"]
    model = [f for f in planes if f["origin"] not in REAL]
    real = [f for f in planes if f["origin"] in REAL]
    inclined = [f for f in real if all(abs(_unit(f["normal"]) @ _unit(m["normal"])) < AX_COS for m in model)]
    walls = [m for m in model if inclined and abs(_unit(m["normal"]) @ _unit(inclined[0]["normal"])) > 0.2]
    if not cyls and len(walls) >= 2 and inclined:
        model = walls                                           # the two walls that meet at the edge (not the end caps)
        n1, n2 = _unit(model[0]["normal"]), _unit(model[1]["normal"])
        for m in model[1:]:
            if abs(_unit(m["normal"]) @ n1) < 0.1:
                n2 = _unit(m["normal"])
                model = [model[0], m]
                break
        e = np.cross(n1, n2)
        if np.linalg.norm(e) > 0.9:
            e = _unit(e)
            inc = [f for f in real if abs(_unit(f["normal"]) @ e) < 0.1]
            if inc:
                x, y = -n1, -n2                                  # into the piece from the two walls... the walls' normals
                # the model planes close the piece on the part's outside: the legs run along -normal of the other wall
                P1, P2 = np.asarray(model[0]["point"], float), np.asarray(model[1]["point"], float)
                # the edge line: the intersection of the two wall planes
                A = np.stack([n1, n2, e])
                b = np.array([n1 @ P1, n2 @ P2, float(e @ V.mean(axis=0))])
                p0 = np.linalg.solve(A, b)
                a_leg = float(((V - p0) @ (-n2)).max())          # along wall 1 (normal n1), away from wall 2
                b_leg = float(((V - p0) @ (-n1)).max())
                x, y = -n2, -n1
                if np.linalg.det(np.stack([x, y, e], axis=1)) < 0:
                    e = -e
                t = (V - p0) @ e
                R = np.stack([x, y, e], axis=1)
                out.append(Fit("chamfer", dict(a=a_leg, b=b_leg, length=float(t.max() - t.min())), R, p0 + e * float(t.min())))
    return out


def fit_box_edges(faces: Sequence[Dict], V: np.ndarray, op: str = "add") -> List[Fit]:
    """Chamfered / filleted box: the planes' box; an edge carries a round when a real cylinder parallel to it is
    tangent to both its walls, a chamfer when a real plane's normal bisects its two walls. Equal legs / radii assumed
    (their median is taken); the frame is the box's, centred."""
    pb = _plane_box(faces, V)
    if pb is None:
        return []
    R, c, half = pb
    rounds, chamfers = {}, {}
    for k in range(3):
        for (si, sj), r in _rounded_corners(faces, R, c, half, k).items():
            rounds[EDGE_IDS.index((k, si, sj))] = r
    for f in faces:
        if f["type"] != "plane" or f["origin"] not in REAL:
            continue
        loc = R.T @ _unit(f["normal"])
        k = int(np.argmin(np.abs(loc)))
        i, j = [m for m in range(3) if m != k]
        if abs(loc[k]) > 0.05 or abs(abs(loc[i]) - abs(loc[j])) > 0.05:
            continue
        si, sj = int(np.sign(loc[i])), int(np.sign(loc[j]))
        p = R.T @ (np.asarray(f["point"], float) - c)
        leg = float(half[i] + half[j] - (si * p[i] + sj * p[j]))   # the 45-degree plane cuts both walls at `leg`
        chamfers[EDGE_IDS.index((k, si, sj))] = leg
    out = []
    if rounds:
        out.append(Fit("filleted_box", dict(size=list(2 * half), edges=sorted(rounds), leg=float(np.median(list(rounds.values())))), R, c))
    if chamfers:
        out.append(Fit("chamfered_box", dict(size=list(2 * half), edges=sorted(chamfers), leg=float(np.median(list(chamfers.values())))), R, c))
    return out


def fit_piece(piece: Dict, V: np.ndarray, op: str = "cut", types: Optional[Sequence[str]] = None) -> List[Fit]:
    """Every catalogue interpretation of a piece from its exact faces (the fitters' interpretations are the loop's other
    sources): revolved, prismatic, edge and box-edge families, filtered to `types` (in their order) when given."""
    faces = piece.get("faces") or []
    fits: List[Fit] = []
    for fn in (fit_revolved, fit_prismatic, fit_edge, fit_box_edges):
        try:
            fits += fn(faces, V, op) if fn is not fit_revolved else fn(faces, op)
        except (ValueError, np.linalg.LinAlgError, KeyError):
            continue
    fits = [f for f in fits if TOOLS[f.tool].valid({k: v for k, v in f.params.items() if not k.startswith("_")})]
    if types is not None:
        order = {t: n for n, t in enumerate(types)}
        fits = sorted((f for f in fits if f.tool in order), key=lambda f: order[f.tool])
    return fits


# ------------------------------------------------------------------------------------------------ the session's store

def tree_signature(p: Primitive):
    """The shape of a boolean tree: kinds and combines, no sizes - "a created tool is an instance of a catalogue type"
    when the signatures agree (the operands of a union in either order)."""
    if p.kind != "composite":
        return p.kind
    a, b = (tree_signature(q) for q in p.parts)
    if p.combine in ("fuse", "common"):
        a, b = sorted((a, b), key=repr)
    return (p.combine, a, b)


def _catalogue_signatures() -> Dict:
    samples = dict(through_hole=dict(d=2, length=3), blind_hole=dict(d=2, depth=3, point_angle=118),
                   countersunk_hole=dict(d=2, d_c=4, angle=90, length=3), counterbored_hole=dict(d_cb=4, depth_cb=1, d=2, length=3),
                   stepped_hole=dict(d1=2, length1=1, d2=3, length2=2), circular_pocket=dict(d=2, depth=1),
                   rect_pocket=dict(L=6, W=4, depth=1, r=0.5), slot=dict(L=6, W=2, depth=1),
                   through_slot=dict(L=6, W=2, depth=1), open_pocket=dict(L=6, W=4, depth=1, r=0.5),
                   chamfer=dict(a=1, b=1, length=3), fillet=dict(r=1, length=3), round=dict(r=1, length=3),
                   groove=dict(R_out=3, R_in=2, width=1), relief=dict(R_out=3, R_in=2, width=1), taper=dict(r1=2, r2=1, length=3))
    out = {}
    for name, prm in samples.items():
        out.setdefault(repr(tree_signature(TOOLS[name].build(prm))), name)   # the first name for a shared shape
    return out


def catalogue_type_of(p: Primitive) -> Optional[str]:
    """The catalogue type whose tree has the shape of `p` (a created tool that turned out to be a standard one)."""
    global _SIGS
    try:
        _SIGS
    except NameError:
        _SIGS = _catalogue_signatures()
    return _SIGS.get(repr(tree_signature(p)))


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


@dataclass
class ToolRecord:
    id: str
    tool: str                                   # a catalogue name, or "created:<n>"
    params: Optional[Dict] = None               # catalogue tools: the parameters (a standard size of the session)
    construction: Optional[Dict] = None         # created tools: the primitive tree in the piece's principal frame
    descriptor: Optional[Dict] = None           # created tools: the piece descriptor it was fitted to (congruence)
    uses: List[Dict] = dc_field(default_factory=list)      # [dict(part, piece, score, source, time)]

    @property
    def count(self) -> int:
        return len(self.uses)


class ToolStore:
    """The tools of a session (the user's rule 2026-09-28: "recycle an old tool of the session"): every tool that was
    accepted - catalogue tools with their parameters, created tools with their construction - with the parts and
    pieces it was used on and its scores. Lives in the loop's state for a run and is handed from run to run by the
    caller (the GUI: one per window session; scripts/time_cascade.py --tool-store <file>)."""

    PARAM_TOL = 0.02                            # a stored size is the piece's size when every length agrees to 2 %

    def __init__(self, records: Optional[List[ToolRecord]] = None):
        self.records: List[ToolRecord] = list(records or [])

    def __len__(self):
        return len(self.records)

    def of_type(self, tool: Optional[str]) -> List[ToolRecord]:
        return [r for r in self.records if tool is None or r.tool == tool]

    @staticmethod
    def same_params(a: Dict, b: Dict, tol: float = PARAM_TOL) -> bool:
        keys = [k for k in a if not k.startswith("_")]
        if set(keys) != {k for k in b if not k.startswith("_")}:
            return False
        for k in keys:
            x, y = np.atleast_1d(np.asarray(a[k], float)), np.atleast_1d(np.asarray(b[k], float))
            if x.shape != y.shape or np.any(np.abs(x - y) > tol * np.maximum(np.abs(x), np.abs(y)) + 1e-9):
                return False
        return True

    def match(self, tool: str, params: Dict) -> Optional[ToolRecord]:
        """The stored catalogue tool of this type whose parameters are the piece's (within PARAM_TOL)."""
        for r in self.of_type(tool):
            if r.params is not None and self.same_params(r.params, params):
                return r
        return None

    def add_use(self, tool: str, part: str, piece, score: float, source: str, params: Optional[Dict] = None,
                construction: Optional[Primitive] = None, descriptor: Optional[Dict] = None) -> ToolRecord:
        rec = self.match(tool, params) if params is not None else None
        if rec is None and construction is not None:
            for r in self.records:
                if r.construction is not None and r.tool == tool and r.descriptor is not None and descriptor is not None:
                    from reco_engine.feature_stage import congruent
                    if congruent(_descriptor_from_json(r.descriptor), descriptor):
                        rec = r
                        break
        if rec is None:
            n = sum(1 for r in self.records if r.tool == tool) + 1
            rec = ToolRecord(id=f"{tool}#{n}", tool=tool, params=None if params is None else
                             {k: v for k, v in params.items() if not k.startswith("_")},
                             construction=None if construction is None else _jsonable(_prim_to_dict(construction)),
                             descriptor=None if descriptor is None else _jsonable(descriptor))
            self.records.append(rec)
        rec.uses.append(dict(part=str(part), piece=piece, score=float(score), source=source, time=time.time()))
        return rec

    def to_json(self) -> str:
        return json.dumps([_jsonable(r.__dict__) for r in self.records], indent=1)

    @classmethod
    def from_json(cls, text: str) -> "ToolStore":
        return cls([ToolRecord(**d) for d in json.loads(text)])

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path) -> "ToolStore":
        path = Path(path)
        return cls.from_json(path.read_text()) if path.is_file() else cls()


def _prim_to_dict(p: Primitive) -> Dict:
    from dataclasses import asdict
    d = asdict(p)
    d["alternatives"] = []
    return d


def _descriptor_from_json(d: Dict) -> Dict:
    out = dict(d)
    for k in ("moments", "axes", "centre"):
        if k in out and out[k] is not None:
            out[k] = np.asarray(out[k], float)
    return out


def construction_primitive(rec: ToolRecord) -> Primitive:
    """A created tool's stored tree as a Primitive (result_schema's JSON form)."""
    from reco_engine.result_schema import ReconstructionResult
    r = ReconstructionResult.from_dict(dict(input_path="", operations=[dict(op="cut", primitive=copy.deepcopy(rec.construction))]))
    return r.operations[0].primitive
