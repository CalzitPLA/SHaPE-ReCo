"""Exact Betti numbers of a STEP solid intersected with boxes (freecadcmd; the reference for scripts/local_grid_frame.py).

    freecadcmd scripts/freecad_crop_topology.py   (reads the job from $CROP_TOPOLOGY_JOB)

The job JSON: {"step": path, "boxes": [[xlo, ylo, zlo, xhi, yhi, zhi], ...] (null = the whole solid), "out": path}.
Per box: b0 = solids of the common, b2 = extra shells (cavities), chi = chi(boundary) / 2 from a tessellation of each
solid with coincident vertices merged, b1 = b0 + b2 - chi.
"""

import json
import os

import FreeCAD
import Part

job = json.load(open(os.environ["CROP_TOPOLOGY_JOB"]))
shape = Part.read(job["step"])
out = []


def boundary_chi(solid, tol):
    pts, tris = solid.tessellate(tol)
    key = {}
    idx = []
    for p in pts:
        k = (round(p.x, 5), round(p.y, 5), round(p.z, 5))
        idx.append(key.setdefault(k, len(key)))
    faces = set()
    edges = set()
    for t in tris:
        a, b, c = idx[t[0]], idx[t[1]], idx[t[2]]
        if len({a, b, c}) < 3:
            continue
        faces.add(tuple(sorted((a, b, c))))
        for e in ((a, b), (b, c), (a, c)):
            edges.add(tuple(sorted(e)))
    used = {v for f in faces for v in f}
    return len(used) - len(edges) + len(faces), len(edges), len(faces)


for box in job["boxes"]:
    if box is None:
        part = shape
    else:
        lo, hi = box[:3], box[3:]
        cutter = Part.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], FreeCAD.Vector(*lo))
        part = shape.common(cutter)
    solids = part.Solids
    b0 = len(solids)
    b2 = sum(len(s.Shells) - 1 for s in solids)
    chi2 = 0
    closed = True
    for s in solids:
        c, e, f = boundary_chi(s, job.get("tolerance", 0.5))
        closed &= (2 * e == 3 * f)          # every edge shared by exactly two triangles
        chi2 += c
    rec = dict(box=box, b0=b0, b2=b2, chi=chi2 // 2 if chi2 % 2 == 0 else chi2 / 2.0,
               b1=(b0 + b2 - chi2 // 2) if chi2 % 2 == 0 else None, volume=part.Volume, closed=bool(closed))
    out.append(rec)
    print(rec, flush=True)

json.dump(out, open(job["out"], "w"), indent=1)
