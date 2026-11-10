#!/usr/bin/env python3
"""FreeCAD side of the I-beam data: the DIN I-section as a real Part solid, plus the plate boxes and envelopes built from it.

Two uses:
  * as a library INSIDE FreeCAD (freecadcmd): `make_ibeam_solid(profile_dims, length, fillet=True)` returns the beam
    segment as a Part solid in its local frame (axis = local z, centred, depth h along y, width b along x). The frame
    generator reuses it for struts with I-beam members.
  * as a batch script run by freecadcmd (dataset_gen/src/ibeam_generator.py does this): with the environment variable
    IBEAM_JOB_JSON pointing at a JSON list of jobs, every job is built, rotated, exported to STL and measured, and the
    results go to IBEAM_RESULT_JSON. One freecadcmd start per batch, not per solid.

Section: a closed 12-corner polygon (Part.makePolygon -> Part.Face), extruded along z, then the four web-flange root edges
filleted with radius r (Shape.makeFillet on the straight edges parallel to z at the root corners). The DIN 1025-1 "I"
series has tapered flanges: the inner flange faces slope at 14 %, the tabulated tf is the thickness at a distance b/4
from the flange tip (TAPER_MEASURE_FROM_TIP), and the toes carry a second radius r2 = 0.6 tf. The polygon then has the
same 12 corners, with sloped inner flange faces.

The pure-geometry helpers (section_points, root/toe corner positions) do not import FreeCAD, so the host can test them.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Mapping, Tuple

TAPER_MEASURE_FROM_TIP = 0.25          # tf of the tapered series is measured b/4 from the flange tip (DIN 1025-1)


def _dims(profile_dims) -> Dict[str, float]:
    """IProfile or mapping -> {h, b, tw, tf, r, slope, r2}. Idempotent (a result of _dims passes through)."""
    if isinstance(profile_dims, Mapping) and "slope" in profile_dims and "r2" in profile_dims:
        return dict(profile_dims)
    get = (lambda k, d=None: profile_dims.get(k, d)) if isinstance(profile_dims, Mapping) \
        else (lambda k, d=None: getattr(profile_dims, k, d))
    slope = float(get("flange_slope", 0.0) or 0.0)
    tf = float(get("tf"))
    r2 = get("toe_radius", None)
    return dict(h=float(get("h")), b=float(get("b")), tw=float(get("tw")), tf=tf, r=float(get("r", 0.0) or 0.0),
                slope=slope, r2=float(r2) if r2 is not None else (0.6 * tf if slope > 0 else 0.0))


def flange_thickness_tip_root(d: Dict[str, float]) -> Tuple[float, float]:
    """(thickness at the flange tip, thickness where the flange meets the web face). Equal to tf for parallel flanges."""
    if d["slope"] <= 0:
        return d["tf"], d["tf"]
    t_tip = d["tf"] - d["slope"] * TAPER_MEASURE_FROM_TIP * d["b"]
    t_root = t_tip + d["slope"] * (d["b"] - d["tw"]) / 2.0
    return t_tip, t_root


def section_points(profile_dims) -> List[Tuple[float, float]]:
    """The 12 corners of the unfilleted section (x across the flanges, y along the depth), counter-clockwise."""
    d = _dims(profile_dims)
    h2, b2, w2 = d["h"] / 2, d["b"] / 2, d["tw"] / 2
    t_tip, t_root = flange_thickness_tip_root(d)
    return [(-b2, -h2), (b2, -h2), (b2, -h2 + t_tip), (w2, -h2 + t_root), (w2, h2 - t_root), (b2, h2 - t_tip),
            (b2, h2), (-b2, h2), (-b2, h2 - t_tip), (-w2, h2 - t_root), (-w2, -h2 + t_root), (-b2, -h2 + t_tip)]


def root_corners(profile_dims) -> List[Tuple[float, float]]:
    d = _dims(profile_dims)
    _, t_root = flange_thickness_tip_root(d)
    y = d["h"] / 2 - t_root
    return [(sx * d["tw"] / 2, sy * y) for sx in (-1, 1) for sy in (-1, 1)]


def toe_corners(profile_dims) -> List[Tuple[float, float]]:
    d = _dims(profile_dims)
    t_tip, _ = flange_thickness_tip_root(d)
    y = d["h"] / 2 - t_tip
    return [(sx * d["b"] / 2, sy * y) for sx in (-1, 1) for sy in (-1, 1)]


def polygon_area(points) -> float:
    return 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])))


# ------------------------------------------------------------------------------------------------ FreeCAD builders

def _edges_at(shape, corners, tol=1e-6):
    """The straight edges parallel to z whose (x, y) is one of `corners`."""
    out = []
    for e in shape.Edges:
        v0, v1 = e.Vertexes[0].Point, e.Vertexes[-1].Point
        if abs(v0.x - v1.x) > tol or abs(v0.y - v1.y) > tol:
            continue
        if any(abs(v0.x - cx) < 1e-4 and abs(v0.y - cy) < 1e-4 for cx, cy in corners):
            out.append(e)
    return out


def make_ibeam_solid(profile_dims, length, fillet=True):
    """The I-beam segment as a Part solid, LOCAL frame: axis z from -length/2 to +length/2, depth h along y, flange width
    b along x. `fillet`: the four root fillets (radius r) and, for the tapered series, the four toe radii r2."""
    import FreeCAD as App
    import Part
    d = _dims(profile_dims)
    L = float(length)
    pts = [App.Vector(x, y, -L / 2) for x, y in section_points(d)]
    face = Part.Face(Part.makePolygon(pts + [pts[0]]))
    solid = face.extrude(App.Vector(0, 0, L))
    if fillet and d["r"] > 0:
        solid = solid.makeFillet(d["r"], _edges_at(solid, root_corners(d)))
    if fillet and d["r2"] > 0:
        solid = solid.makeFillet(d["r2"], _edges_at(solid, toe_corners(d)))
    solids = solid.Solids
    if len(solids) != 1:
        raise RuntimeError(f"I-beam build gave {len(solids)} solids")
    return solids[0]


def mitre_cut(solid, profile_dims, length, axis: str, angle_deg: float):
    """Cut the +z end at a mitre: the plane through the end's edge at the low side of `axis`'s partner direction
    (axis "x": the cut runs across the depth, u = y, extent h; axis "y": across the width, u = x, extent b). The long side
    keeps `length`; the short side is length - extent * tan(angle)."""
    import FreeCAD as App
    d = _dims(profile_dims)
    a = math.radians(angle_deg)
    L = float(length)
    big = 4.0 * max(d["h"], d["b"], L)
    if axis == "x":
        normal, p0 = App.Vector(0, math.sin(a), math.cos(a)), App.Vector(0, -d["h"] / 2, L / 2)
    else:
        normal, p0 = App.Vector(math.sin(a), 0, math.cos(a)), App.Vector(-d["b"] / 2, 0, L / 2)
    import Part
    cutter = Part.makeBox(big, big, big, App.Vector(-big / 2, -big / 2, 0))
    cutter.Placement = App.Placement(p0, App.Rotation(App.Vector(0, 0, 1), normal))
    out = solid.cut(cutter)
    if len(out.Solids) != 1:
        raise RuntimeError(f"mitre gave {len(out.Solids)} solids")
    return out.Solids[0]


def box_solid(center, half_extents):
    """Part.makeBox with the given local centre and half extents (axis-aligned in the beam's local frame)."""
    import FreeCAD as App
    import Part
    c, hx = center, half_extents
    return Part.makeBox(2 * hx[0], 2 * hx[1], 2 * hx[2], App.Vector(c[0] - hx[0], c[1] - hx[1], c[2] - hx[2]))


def _rotate(shape, quat):
    import FreeCAD as App
    qx, qy, qz, qw = quat
    shape = shape.copy()
    shape.transformShape(App.Placement(App.Vector(), App.Rotation(qx, qy, qz, qw)).toMatrix())
    return shape


def export_stl(shape, path: str, deflection: float) -> int:
    import MeshPart
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=float(deflection),
                                  AngularDeflection=math.radians(5.0), Relative=False)
    mesh.write(path)
    return int(mesh.CountFacets)


def build_job(job: Dict) -> Dict:
    """One job -> a PLACED STL and its measurements.

    job: id, kind ("ibeam" | "box"), profile (dims), length, fillet, mitre ({axis, angle} or None), center/half_extents
    (box, beam-local mm), quaternion [qx, qy, qz, qw] (local -> world), stl, deflection (mm, placed), and the in-scale
    draw: grid_mm, size_fraction (largest world extent / grid) and center_unit (3 numbers in [-1, 1]).

    World = k * R * local + T: the solid is built in real millimetres in its local frame, rotated, scaled about the origin
    by k so its largest extent is size_fraction * grid_mm, and translated so its bounding-box centre sits at
    center_unit * (grid_mm / 2 - extent / 2) - inside the grid centred on the world origin, which is the frame
    src/sample_writer.finish_sample voxelises in."""
    import FreeCAD as App
    if job["kind"] == "ibeam":
        shape = make_ibeam_solid(job["profile"], job["length"], fillet=job.get("fillet", True))
        if job.get("mitre"):
            shape = mitre_cut(shape, job["profile"], job["length"], job["mitre"]["axis"], job["mitre"]["angle"])
    elif job["kind"] == "box":
        shape = box_solid(job["center"], job["half_extents"])
    else:
        raise ValueError(f"unknown job kind {job['kind']!r}")
    local_volume = float(shape.Volume)
    shape = _rotate(shape, job["quaternion"])
    bb = shape.optimalBoundingBox() if hasattr(shape, "optimalBoundingBox") else shape.BoundBox
    extent = [bb.XLength, bb.YLength, bb.ZLength]
    grid = float(job.get("grid_mm", 40.0))
    k = float(job.get("size_fraction", 1.0)) * grid / max(extent)
    shape.scale(k, App.Vector(0, 0, 0))
    ext_k = [e * k for e in extent]
    u = job.get("center_unit", [0.0, 0.0, 0.0])
    centre = [ui * (grid / 2 - e / 2) for ui, e in zip(u, ext_k)]
    bbc = [k * bb.Center.x, k * bb.Center.y, k * bb.Center.z]
    T = [c - b for c, b in zip(centre, bbc)]
    shape.translate(App.Vector(*T))
    pb = shape.BoundBox
    facets = export_stl(shape, job["stl"], job.get("deflection", 0.01))
    return dict(id=job["id"], ok=True, volume_mm3=float(shape.Volume), local_volume_mm3=local_volume,
                valid=bool(shape.isValid()), n_solids=len(shape.Solids), facets=facets, scale_factor=k,
                translation_mm=T, center_mm=centre, extent_mm=ext_k, local_extent_mm=extent,
                bbox_min=[pb.XMin, pb.YMin, pb.ZMin], bbox_max=[pb.XMax, pb.YMax, pb.ZMax])


def run_jobs(job_path: str, result_path: str) -> None:
    with open(job_path, encoding="utf-8") as fh:
        jobs = json.load(fh)
    results = []
    for job in jobs:
        try:
            results.append(build_job(job))
        except Exception as exc:                     # reported per job; the host decides
            results.append(dict(id=job.get("id"), ok=False, error=f"{type(exc).__name__}: {exc}"))
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh)


if os.environ.get("IBEAM_JOB_JSON") and os.environ.get("IBEAM_RESULT_JSON"):
    run_jobs(os.environ["IBEAM_JOB_JSON"], os.environ["IBEAM_RESULT_JSON"])
