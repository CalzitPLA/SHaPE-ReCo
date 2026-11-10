"""The exact surface inventory of a STEP part (freecadcmd): every B-rep face with its surface type and parameters, area,
bounding box, outward normal, whether the whole part lies on its material side ("dominating" - the supporting surfaces
of a tight first cover), its neighbours, and a per-face tessellation that keeps the face id of every triangle.

    BREP_JOB=job.json freecadcmd scripts/freecad_brep_faces.py

job: {"step": part.step, "out_json": faces.json, "out_npz": mesh.npz, "deflection": 0.02 (mm, tessellation),
      "tol": 1e-3 (relative to the part's diagonal: how far outside a surface still counts as on it)}
faces.json: {"volume", "area", "diag", "faces": [{"id", "type", "area", "bbox", "normal" (outward, at the face's
middle), "point", <type parameters>, "dominating"}], "edges": [[face_i, face_j], ...]}
mesh.npz: vertices (n, 3), triangles (m, 3), face_id (m,) - the part's surface, triangle by triangle, with its face.
Called by reco_engine/brep.py (cached per file); runnable by hand for a look at a part.
"""
import json
import os

import numpy as np
import Part

job = json.load(open(os.environ["BREP_JOB"]))
shape = Part.read(job["step"])
solid = shape.Solids[0] if len(shape.Solids) == 1 else shape
bb = shape.BoundBox
diag = float((bb.XLength ** 2 + bb.YLength ** 2 + bb.ZLength ** 2) ** 0.5)
tol = float(job.get("tol", 1e-3)) * diag
deflection = float(job.get("deflection", 0.02))

# the part's surface, face by face (one tessellation, the face id kept per triangle)
verts, tris, fid = [], [], []
for i, f in enumerate(shape.Faces):
    v, t = f.tessellate(deflection)
    if not t:
        continue
    base = len(verts)
    verts += [(p.x, p.y, p.z) for p in v]
    tris += [(a + base, b + base, c + base) for a, b, c in t]
    fid += [i] * len(t)
V = np.asarray(verts, float)
np.savez_compressed(job["out_npz"], vertices=V, triangles=np.asarray(tris, np.int64), face_id=np.asarray(fid, np.int64))


def vec(p):
    return [float(p.x), float(p.y), float(p.z)]


faces = []
for i, f in enumerate(shape.Faces):
    s = f.Surface
    kind = type(s).__name__
    u0, u1, v0, v1 = f.ParameterRange
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    p = f.valueAt(um, vm)
    n = f.normalAt(um, vm)                                   # respects the face's orientation: outward on a solid
    eps = 1e-3 * diag
    if solid.isInside(p + n * eps, 1e-9, True) and not solid.isInside(p - n * eps, 1e-9, True):
        n = n * -1.0                                         # a reversed face: turn it outward
    rec = dict(id=i, type=kind, area=float(f.Area), point=vec(p), normal=vec(n),
               bbox=[f.BoundBox.XMin, f.BoundBox.YMin, f.BoundBox.ZMin, f.BoundBox.XMax, f.BoundBox.YMax, f.BoundBox.ZMax])
    nn = np.array(vec(n))
    if kind == "Plane":
        # dominating: the whole part on the inner side of the plane
        rec["dominating"] = bool(((V - np.array(vec(p))) @ nn).max() <= tol) if len(V) else False
    elif kind in ("Cylinder", "Cone", "Sphere", "Toroid"):
        if kind == "Cylinder":
            rec.update(axis=vec(s.Axis), center=vec(s.Center), radius=float(s.Radius))
        elif kind == "Cone":
            rec.update(axis=vec(s.Axis), apex=vec(s.Apex), center=vec(s.Center), radius=float(s.Radius),
                       semi_angle=float(s.SemiAngle))
        elif kind == "Sphere":
            rec.update(center=vec(s.Center), radius=float(s.Radius))
        else:
            rec.update(axis=vec(s.Axis), center=vec(s.Center), major_radius=float(s.MajorRadius),
                       minor_radius=float(s.MinorRadius))
        # convex (material towards the axis / centre) or concave (a bore): which way does the outward normal point
        if kind in ("Cylinder", "Cone"):
            a, c = np.array(rec["axis"]), np.array(rec["center"])
            radial = (np.array(vec(p)) - c) - ((np.array(vec(p)) - c) @ a) * a
            rec["convex"] = bool(radial @ nn > 0)
        elif kind == "Sphere":
            rec["convex"] = bool((np.array(vec(p)) - np.array(rec["center"])) @ nn > 0)
        else:
            rec["convex"] = None
        # dominating: a convex cylinder / sphere whose solid holds the whole part (every surface point inside it)
        dom = False
        if rec["convex"] and len(V):
            if kind == "Cylinder":
                a, c = np.array(rec["axis"]), np.array(rec["center"])
                rel = V - c
                dom = bool((np.linalg.norm(rel - np.outer(rel @ a, a), axis=1) <= rec["radius"] + tol).all())
            elif kind == "Sphere":
                dom = bool((np.linalg.norm(V - np.array(rec["center"]), axis=1) <= rec["radius"] + tol).all())
        rec["dominating"] = dom
    else:
        rec["dominating"] = False
    faces.append(rec)

# neighbours: faces sharing an edge
owner = {}
for i, f in enumerate(shape.Faces):
    for e in f.Edges:
        owner.setdefault(e.hashCode(), set()).add(i)
edges = sorted({tuple(sorted(s)) for s in owner.values() if len(s) == 2})

json.dump(dict(volume=float(shape.Volume), area=float(shape.Area), diag=diag, n_solids=len(shape.Solids),
               faces=faces, edges=[list(e) for e in edges]), open(job["out_json"], "w"))
print("BREP_DONE", len(faces), "faces")
