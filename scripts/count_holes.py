#!/usr/bin/env python3
"""
The holes of a STEP part counted from its B-rep faces (2026-10-01; thesis chapter 14 quoted "about 205 small holes" for
the liner, the face review counted 122). Run by freecadcmd from here:

    python3 scripts/count_holes.py <part.step> [--json out.json] [--small 7.0]

A hole is a set of coaxial faces - cylinders and cones on one axis line (direction and position within `--tol`) -
whose extents along the axis touch, with at least one CONCAVE cylinder (the part's outward normal points to the axis:
material around a void) that closes the full circle: its faces at one radius sweep 360 degrees together. Partial
cylinders (fillets, pocket corners, a groove's round ends, the half bore of a half pipe) are no holes; two holes on
one axis line with a gap between them are two. Its radius is the largest of its cylinders; "small": radius <= --small.
Also listed: the cylindrical faces by radius (concave, convex), the raw material for any other count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

SCRIPT = r'''
import json, math, os, Part, FreeCAD
job = json.load(open(os.environ["HOLES_JOB"]))
shape = Part.read(job["step"])
tol = float(job["tol"])
diag = shape.BoundBox.DiagonalLength

def canon(d):
    d = FreeCAD.Vector(d); d.normalize()
    for c in (d.x, d.y, d.z):
        if abs(c) > 1e-9:
            return d if c > 0 else d.negative()
    return d

rows = []
for i, f in enumerate(shape.Faces):
    s = f.Surface
    kind = type(s).__name__
    if kind not in ("Cylinder", "Cone"):
        continue
    d = canon(s.Axis)
    c = FreeCAD.Vector(s.Center) if kind == "Cylinder" else FreeCAD.Vector(s.Apex)
    foot = c - d * c.dot(d)                                 # the axis line's point closest to the origin
    u0, u1, v0, v1 = f.ParameterRange
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    p, n = f.valueAt(um, vm), f.normalAt(um, vm)           # the part's outward normal
    r = p - c; r = r - d * r.dot(d)
    concave = n.dot(r) < 0                                  # the normal points to the axis: material around a void
    ts = [(v.Point - c).dot(d) for v in f.Vertexes]
    pts = f.tessellate(1e-3 * diag)[0]
    ts += [(q - c).dot(d) for q in pts]
    t0 = min(ts) + c.dot(d); t1 = max(ts) + c.dot(d)        # the extent along the axis, from the line's foot
    row = dict(face=i, kind=kind.lower(), axis=[d.x, d.y, d.z], foot=[foot.x, foot.y, foot.z], concave=bool(concave),
               span=float(u1 - u0), t=[float(t0), float(t1)], area=float(f.Area))
    if kind == "Cylinder":
        row["radius"] = float(s.Radius)
    else:
        rs = [((q - c) - d * (q - c).dot(d)).Length for q in pts] or [0.0]
        row["radius"] = float(max(rs)); row["radius_min"] = float(min(rs)); row["semi_angle"] = float(s.SemiAngle)
    rows.append(row)
json.dump(dict(rows=rows, faces=len(shape.Faces), volume=shape.Volume, diag=diag), open(job["out"], "w"))
'''


def same_line(a, b, tol):
    import numpy as np
    da, db = np.asarray(a["axis"]), np.asarray(b["axis"])
    if np.linalg.norm(da - db) > 1e-4:
        return False
    return float(np.linalg.norm(np.asarray(a["foot"]) - np.asarray(b["foot"]))) <= tol


def holes_of(rows, tol, gap):
    """Faces on one axis line, split where their extents along it do not touch; each such set with a concave full
    cylinder is a hole."""
    import math
    lines = []
    for r in rows:
        for L in lines:
            if same_line(L[0], r, tol):
                L.append(r)
                break
        else:
            lines.append([r])
    holes, partial = [], 0
    for L in lines:
        L = sorted(L, key=lambda r: r["t"][0])
        segs, cur, hi = [], [], -math.inf
        for r in L:
            if cur and r["t"][0] > hi + gap:
                segs.append(cur)
                cur, hi = [], -math.inf
            cur.append(r)
            hi = max(hi, r["t"][1])
        segs.append(cur)
        for S in segs:
            full = []
            by_radius = {}
            for r in S:
                if r["kind"] == "cylinder" and r["concave"]:
                    key = round(r["radius"] / tol)
                    by_radius[key] = by_radius.get(key, 0.0) + r["span"]
            full = sorted(k * tol for k, span in by_radius.items() if span >= 2 * math.pi * (1 - 1e-3))
            if not full:
                partial += 1
                continue
            cyl = sorted({round(r["radius"], 4) for r in S if r["kind"] == "cylinder" and r["concave"]})
            holes.append(dict(axis=S[0]["axis"], foot=S[0]["foot"], t=[min(r["t"][0] for r in S), max(r["t"][1] for r in S)],
                              radius=max(full), radii=cyl, cones=sum(1 for r in S if r["kind"] == "cone"),
                              faces=[r["face"] for r in S]))
    return holes, partial


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("step")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--small", type=float, default=7.0, help="a hole of at most this radius is small (mm)")
    ap.add_argument("--tol", type=float, default=0.01, help="coaxial within this distance (mm); radii to this")
    ap.add_argument("--gap", type=float, default=0.05, help="extents along the axis closer than this touch (mm)")
    args = ap.parse_args(argv)
    from src.closest_point_field import find_freecadcmd
    freecad = find_freecadcmd()
    with tempfile.TemporaryDirectory() as tmp:
        script, job, out = Path(tmp) / "holes.py", Path(tmp) / "job.json", Path(tmp) / "out.json"
        script.write_text(SCRIPT)
        job.write_text(json.dumps(dict(step=str(Path(args.step).resolve()), tol=args.tol, out=str(out))))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["HOLES_JOB"] = str(job)
        proc = subprocess.run([freecad, str(script)], env=env, capture_output=True, text=True, timeout=1800)
        if not out.is_file():
            print((proc.stderr or proc.stdout)[-2000:], file=sys.stderr)
            return 1
        data = json.loads(out.read_text())
    rows = data["rows"]
    holes, partial = holes_of(rows, args.tol, args.gap)
    by_radius: dict = {}
    for h in holes:
        key = f"{h['radius']:.4g}"
        by_radius[key] = by_radius.get(key, 0) + 1
    faces_by_radius: dict = {}
    for r in rows:
        if r["kind"] != "cylinder":
            continue
        key = f"{r['radius']:.4g} {'concave' if r['concave'] else 'convex'}"
        faces_by_radius[key] = faces_by_radius.get(key, 0) + 1
    small = [h for h in holes if h["radius"] <= args.small]
    summary = dict(step=str(args.step), faces=data["faces"], cylinder_faces=sum(1 for r in rows if r["kind"] == "cylinder"),
                   cone_faces=sum(1 for r in rows if r["kind"] == "cone"), holes=len(holes),
                   small=dict(max_radius=args.small, holes=len(small)),
                   holes_by_radius=dict(sorted(by_radius.items(), key=lambda kv: float(kv[0]))),
                   coaxial_sets_without_full_circle=partial,
                   cylinder_faces_by_radius=dict(sorted(faces_by_radius.items(), key=lambda kv: float(kv[0].split()[0]))),
                   definition=__doc__.strip().split("\n\n")[2] if len(__doc__.strip().split("\n\n")) > 2 else "")
    print(json.dumps({k: v for k, v in summary.items() if k != "definition"}, indent=1))
    if args.json:
        args.json.write_text(json.dumps(dict(summary, holes_list=holes), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
