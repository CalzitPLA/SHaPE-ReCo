"""
The exact judge: the residual of a built model against its input by OpenCASCADE booleans (freecadcmd), not on a
voxel grid. A grid only sees what its pitch allows: on the I-beam frame the cascade's 48^3 grid saw 0 missing voxels
while the model missed 9 % of the part (8 plates of 20 mm and the root fillets; docs/FRAME_FAILURE.md). The
networks may propose on grids; acceptance and "done" are judged here.

    r = exact_residual(input_path, model_step)          # STEP input: its B-rep; STL input: the mesh as a solid
    r["iou"], r["r_plus"], r["r_minus"], r["r_plus_solids"]

r_plus = input - model (missed material), r_minus = model - input (invented material), both mm^3 (input units);
iou = (V_in - r_plus) / (V_in + r_minus). Two methods, in this order (`method` in the result): the general fuse
("general fuse": one partition of both solids, the pieces by their origin; tolerances none, 6e-8 and 6e-7 of the
diagonal) and the separate booleans ("occt": common, then both cuts; tolerances none, 1e-7, 1e-6, 1e-5 of the
diagonal; `fuzzy` in the result is the one used). Self-checks, both methods: the balances V_in - r_plus and
V_model - r_minus must agree (within 1e-5 V_in for the fuse, 1e-4 V_in and also with the intersection's volume for
the separate booleans), every piece of the fuse must be valid, and the shared volume must agree with a sample of
`check_points` (1500) seeded points in the JOINT BOUNDING BOX (`sample_common`: tolerance 4 sampling errors + 0.2 %
of V_in). A try that fails a check goes to the next tolerance / method; when every one fails the result is
{"error": ...}. Never raises.

What the checks do not catch (review R-02, 2026-09-30): the sample's tolerance grows with the bounding box, not with
V_in, so on a part that fills a small share of its box it cannot refuse anything. Review part r13 (2026-09-29,
~/shape_reco_results/compare_server_a_20260929/review__r13_rotated_rect/): V_in 7.53e7 mm^3 in a joint box of
3.8e10 mm^3, 3 of 1500 points inside both, tolerance about 1.75e8 mm^3 (computed) - more than V_in; the separate
booleans returned r_plus = V_in and r_minus = V_model (a total failure, which also satisfies both balances) as exact
IoU 1.0e-7 without an error, against a sampled IoU of 0.9366 +- 0.001 (compare.json). That run predates the general
fuse; whether the fuse path refuses r13 was not measured, by reading its balance and sample checks cannot. Nothing
here bounds the shared volume by min(V_in, V_model) or compares with `sampled_iou`: the loop's grade does
(reco_engine.residual_loop._exact_grade: a disagreement over 0.05 reports the sampled IoU instead); the feature
phase's differences and the standalone scripts do not. `solid_of` builds an STL operand without isValid / sign / unit
checks (the input is never scaled, only the model by `model_scale`).
"""

from __future__ import annotations

import json
import numpy as np
import time
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

HELPERS = r'''
import json, os, time, Part, Mesh, FreeCAD
job = json.load(open(os.environ["EXACT_JOB"]))

def solid_of(path, scale):
    p = path.lower()
    if p.endswith((".step", ".stp")):
        s = Part.read(path)
    else:                                                   # a closed triangle mesh -> one solid of its facets
        m = Mesh.Mesh(path)
        sh = Part.Shape()
        sh.makeShapeFromMesh(m.Topology, 1e-6 * max(m.BoundBox.DiagonalLength, 1.0))
        s = Part.makeSolid(Part.makeShell(sh.Faces))
    if scale != 1.0:
        s = s.copy(); s.scale(scale)
    return s

def betti(shape, tol, per_shell=False):
    """dict(b0, b1, b2) of a solid or a compound of solids: b0 the solids, b2 their inner shells (voids), b1 = b0 + b2 -
    chi summed over the solids, chi(solid) = chi(boundary) / 2 from the tessellation with coincident vertices merged
    (exact topology, independent of how an exported STL was stitched). per_shell: also `shells`, the genus of every
    connected piece of each solid's tessellation (one per shell unless two shells touch), largest first; b1 is their sum
    (half lives, half dies). None if a tessellation is not closed."""
    b0 = b1 = b2 = 0
    shells = []
    q = 1e-6 * max(shape.BoundBox.DiagonalLength, 1e-9)       # merge quantum relative to the size (not 7 decimals:
    for sol in shape.Solids:                                   # at the frame's 8 m those left edges unmatched)
        pts, tris = sol.tessellate(tol)
        key, idx = {}, []
        for p in pts:
            idx.append(key.setdefault((round(p.x / q), round(p.y / q), round(p.z / q)), len(key)))
        faces, edges = set(), set()
        for t in tris:
            a, b, c = idx[t[0]], idx[t[1]], idx[t[2]]
            if len({a, b, c}) < 3:
                continue
            faces.add(tuple(sorted((a, b, c))))
            edges.update((tuple(sorted(e)) for e in ((a, b), (b, c), (a, c))))
        if 2 * len(edges) != 3 * len(faces):
            return None
        chi2 = len({v for f in faces for v in f}) - len(edges) + len(faces)
        b0 += 1
        b2 += len(sol.Shells) - 1
        b1 += 1 + (len(sol.Shells) - 1) - chi2 // 2
        if per_shell:                                          # the pieces by union-find over the edges
            parent = {}
            def root(x):
                while parent.setdefault(x, x) != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for a, b in edges:
                parent[root(a)] = root(b)
            chi = {}
            for f in faces:
                r = root(f[0])
                chi[r] = chi.get(r, 0) + 1
            for a, b in edges:
                r = root(a)
                chi[r] = chi.get(r, 0) - 1
            for v in {v for f in faces for v in f}:
                r = root(v)
                chi[r] = chi.get(r, 0) + 1
            shells += [1 - c // 2 for c in chi.values()]
    out = dict(b0=b0, b1=b1, b2=b2)
    if per_shell:
        out["shells"] = sorted(shells, reverse=True)
    return out

def genus(shape, tol):
    """b1 of `betti` (the sum over the solids). None if not closed."""
    b = betti(shape, tol)
    return None if b is None else b["b1"]

# ---- the exact surface of a face, and where the face comes from
KINDS = dict(Plane="plane", Cylinder="cylinder", Cone="cone", Sphere="sphere", Toroid="torus")

def vec(v):
    return [float(v.x), float(v.y), float(v.z)]

class FaceIndex:
    """The faces of a shape with their bounding boxes (enlarged by eps): a point is measured only against the faces
    whose box holds it. distToShape against the compound of all faces made the liner's difference with 40 pieces
    time out at 300 s (2026-09-27; its input has 1322 faces)."""
    def __init__(self, faces, eps):
        self.items = []
        for f in faces:
            bb = f.BoundBox
            bb.enlarge(eps)
            self.items.append((f, bb))

    def on(self, pts, eps):
        for p in pts:
            v = Part.Vertex(p)
            if not any(bb.isInside(p) and v.distToShape(f)[0] <= eps for f, bb in self.items):
                return False
        return True

def face_info(f, in_faces, mod_faces, eps, defl):
    s = f.Surface
    d = dict(type=KINDS.get(type(s).__name__, "other"), area=float(f.Area))
    u0, u1, v0, v1 = f.ParameterRange
    um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    n, p = f.normalAt(um, vm), f.valueAt(um, vm)             # the normal points out of the piece
    d["normal"], d["point"] = vec(n), vec(p)
    bb = f.BoundBox                                          # the face's own reach (reco_engine/prismatic_groups.py)
    d["bbox"] = [[bb.XMin, bb.YMin, bb.ZMin], [bb.XMax, bb.YMax, bb.ZMax]]
    outline = []                                             # points on its edges: its reach along any direction
    for e in f.Edges:
        try:
            outline += [vec(q) for q in e.discretize(9 if len(f.Edges) <= 8 else 3)]
        except Exception:
            outline += [vec(v.Point) for v in e.Vertexes]
    d["outline"] = outline[::max(1, len(outline) // 96)]
    for key in ("Axis", "Center", "Apex", "Position"):
        if hasattr(s, key):
            d[key.lower()] = vec(getattr(s, key))
    for key, name in (("Radius", "radius"), ("MajorRadius", "major_radius"), ("MinorRadius", "minor_radius"),
                      ("SemiAngle", "semi_angle")):
        if hasattr(s, key):
            d[name] = float(getattr(s, key))
    if d["type"] in ("cylinder", "cone", "sphere") and "center" in d:
        r = p.sub(s.Center)
        if "axis" in d:
            r = r.sub(s.Axis.multiply(r.dot(s.Axis)))
        d["convex"] = bool(n.dot(r) > 0)                    # True: the piece's material is inside the surface
    pts, tris = f.tessellate(defl)                          # points INSIDE the trimmed face (valueAt may lie in a hole):
    pick = [tris[i] for i in sorted({0, len(tris) // 2, len(tris) - 1})] if tris else []
    probe = [(pts[a] + pts[b] + pts[c]).multiply(1.0 / 3.0) for a, b, c in pick] or [p]
    probe = [s.value(*s.parameter(q)) for q in probe]      # triangle centres, put back onto the curved surface
    if in_faces is not None:
        a, b = in_faces.on(probe, eps), mod_faces.on(probe, eps)
        d["origin"] = "both" if a and b else "input" if a else "model" if b else "none"
    return d


# ---- the pieces, one solid each (opt-in: "pieces_dir"), by the compound's own explosion
def explode(shape):
    """The solids of a boolean result as FreeCAD's Part > Compound > Explode compound gives them: the children of a
    compound (nested compounds opened), a compsolid's solids, a lone solid as a compound of one. Checked 2026-09-27 on
    the generated box with two bores: CompoundTools.Explode, childShapes() and .Solids gave the same 3 + 1 pieces."""
    if shape.ShapeType == "Compound":
        out = []
        for c in shape.childShapes():
            out += explode(c)
        return out
    if shape.ShapeType == "CompSolid":
        return list(shape.Solids)
    return [shape] if shape.ShapeType == "Solid" else list(shape.Solids)

def piece_info(sol, sign, i, folder, in_faces, mod_faces, eps, diag):
    import MeshPart
    name = f"piece_{sign}_{i:03d}"
    bb = sol.BoundBox
    defl = max(1e-3 * bb.DiagonalLength, 1e-5 * diag)
    step, stl = os.path.join(folder, name + ".step"), os.path.join(folder, name + ".stl")
    sol.exportStep(step)
    MeshPart.meshFromShape(Shape=sol, LinearDeflection=defl, AngularDeflection=0.1, Relative=False).write(stl)
    return dict(id=name, sign=sign, volume=float(sol.Volume), valid=bool(sol.isValid()), step=step, stl=stl,
                bbox=[[bb.XMin, bb.YMin, bb.ZMin], [bb.XMax, bb.YMax, bb.ZMax]], centre=vec(sol.CenterOfMass),
                faces=[face_info(f, in_faces, mod_faces, eps, defl) for f in sol.Faces])

def write_pieces(out, plus, minus, inp, mod, fuzzy, floor, diag, method):
    """The solids of r_plus and r_minus (lists) into job["pieces_dir"], the biggest first; `out` gets pieces,
    left_out_volume, left_out_count and pieces_method."""
    folder = job["pieces_dir"]
    os.makedirs(folder, exist_ok=True)
    eps = max(1e-6 * diag, 10.0 * (fuzzy or 0.0))
    in_faces, mod_faces = FaceIndex(inp.Faces, eps), FaceIndex(mod.Faces, eps)
    pieces, slivers = [], dict(plus=0.0, minus=0.0)
    small, capped = dict(plus=0, minus=0), dict(plus=0, minus=0)
    for sign, solids in (("plus", plus), ("minus", minus)):
        solids = sorted(solids, key=lambda s: s.Volume, reverse=True)
        above = [s for s in solids if s.Volume > floor]
        kept = above[:int(job.get("max_pieces", 500))]
        small[sign], capped[sign] = len(solids) - len(above), len(above) - len(kept)
        slivers[sign] = float(sum(s.Volume for s in solids) - sum(s.Volume for s in kept))
        pieces += [piece_info(s, sign, i, folder, in_faces, mod_faces, eps, diag) for i, s in enumerate(kept)]
    out["pieces"], out["left_out_volume"] = pieces, slivers
    out["left_out_count"] = dict(small=small, capped=capped)    # solids under the volume floor / past max_pieces
    out["pieces_method"] = method
    json.dump(dict(input=job["input"], model=job["model"], pieces=pieces, left_out_volume=slivers),
              open(os.path.join(folder, "pieces.json"), "w"), indent=1)

# ---- the state of a difference: ALL its solids (slivers too), for the incremental difference of the next model
def write_state(folder, plus, minus, info):
    os.makedirs(folder, exist_ok=True)
    for sign, solids in (("plus", plus), ("minus", minus)):
        path = os.path.join(folder, sign + ".brep")
        if os.path.exists(path):
            os.remove(path)
        if solids:                                              # an empty compound does not read back
            Part.makeCompound(list(solids)).exportBrep(path)
    json.dump(dict(info, r_plus=float(sum(s.Volume for s in plus)), r_minus=float(sum(s.Volume for s in minus)),
                   n_plus=len(plus), n_minus=len(minus)), open(os.path.join(folder, "state.json"), "w"))

def read_state(folder):
    info = json.load(open(os.path.join(folder, "state.json")))
    out = []
    for sign in ("plus", "minus"):
        path = os.path.join(folder, sign + ".brep")
        out.append(explode(Part.read(path)) if os.path.exists(path) else [])
    return info, out[0], out[1]

def sample_common(inp, mod, diag, v_in, n):
    """(the volume inside both, its error) from n seeded points in the box of both, by isInside: a check of the
    booleans that is none itself. The error: 4 sampling errors + 0.2 % of the input's volume. The sampling error
    scales with the BOX, not with the input: on a part that fills a small share of its box (frame review r13: 0.2 %)
    the error exceeds the input's volume and the check cannot refuse a failed boolean (module docstring)."""
    import random
    rng = random.Random(0)
    box = inp.BoundBox
    box.add(mod.BoundBox)
    pts = [FreeCAD.Vector(box.XMin + rng.random() * box.XLength, box.YMin + rng.random() * box.YLength,
                          box.ZMin + rng.random() * box.ZLength) for _ in range(n)]
    eps = 1e-6 * diag
    both = sum(1 for q in pts if inp.isInside(q, eps, True) and mod.isInside(q, eps, True))
    vbox = box.XLength * box.YLength * box.ZLength
    p = both / len(pts)
    return p * vbox, 4.0 * vbox * (max(p * (1 - p), 1.0 / len(pts)) / len(pts)) ** 0.5 + 2e-3 * v_in
'''

SCRIPT = HELPERS + r'''
def input_faces(shape, defl):
    faces = []
    for f in shape.Faces:
        d = face_info(f, None, None, 0.0, defl)             # with its bbox and outline
        if "axis" in d and "center" in d:                   # the trimmed face's reach along its axis, from its centre
            # (the input's faces only: a piece's faces keep the reach hole_stack was measured with, 2026-09-29)
            t = [q.sub(f.Surface.Center).dot(f.Surface.Axis) for q in f.tessellate(defl)[0]]
            if t:
                d["along"] = [float(min(t)), float(max(t))]
        faces.append(d)
    return faces

if job.get("faces_of"):                                     # only the faces of one shape (exact_faces)
    shape = solid_of(job["faces_of"], 1.0)
    faces = input_faces(shape, 1e-3 * shape.BoundBox.DiagonalLength)
    json.dump(dict(faces=faces, volume=shape.Volume), open(job["out"], "w"))
    raise SystemExit(0)

if job.get("facts_of"):                                     # faces, topology and counts of one shape (exact_facts)
    shape = solid_of(job["facts_of"], 1.0)
    bb = shape.BoundBox
    defl = 1e-3 * bb.DiagonalLength
    # the Betti numbers first: face_info tessellates every face on its own, and the solid's tessellation reuses those
    # triangulations, whose edges no longer match (measured 2026-10-29: aegir n4_S_000230 and n16_S_000045 not closed)
    b = betti(shape, defl, per_shell=True)
    out = dict(faces=input_faces(shape, defl), volume=float(shape.Volume), area=float(shape.Area),
               valid=bool(shape.isValid()), bbox=[[bb.XMin, bb.YMin, bb.ZMin], [bb.XMax, bb.YMax, bb.ZMax]],
               counts=dict(solids=len(shape.Solids), shells=len(shape.Shells), faces=len(shape.Faces),
                           wires=len(shape.Wires), edges=len(shape.Edges), vertexes=len(shape.Vertexes)),
               betti=b)
    if job.get("stl_out"):
        import MeshPart
        MeshPart.meshFromShape(Shape=shape, LinearDeflection=defl, AngularDeflection=0.1, Relative=False).write(job["stl_out"])
    json.dump(out, open(job["out"], "w"))
    raise SystemExit(0)




mod = solid_of(job["model"], job.get("model_scale", 1.0))
tol = 1e-3 * mod.BoundBox.DiagonalLength
if not job.get("booleans", True):
    json.dump(dict(model_volume=mod.Volume, model_genus=genus(mod, tol)), open(job["out"], "w"))
    raise SystemExit(0)
inp = solid_of(job["input"], 1.0)
v_in, v_mod = inp.Volume, mod.Volume
floor = job.get("min_solid", 1e-6 * v_in)                   # boolean slivers of zero volume are not pieces
diag = max(inp.BoundBox.DiagonalLength, mod.BoundBox.DiagonalLength)

def consistent(plus, minus, common):
    # the shared volume three ways: V_in - r+, V_model - r- and the intersection itself must agree, else a boolean
    # failed silently. The first two alone agree on garbage too (measured on s03_rod, model = input up to 1e-4 mm:
    # r+ = r- = 0.83 V_in, IoU 0.09 for a model that is the part)
    rel = job.get("rel_check", 1e-4) * v_in
    return abs((v_in - plus.Volume) - (v_mod - minus.Volume)) <= rel and abs((v_in - plus.Volume) - common.Volume) <= rel

# Plain booleans first; where faces of model and input coincide (adds snapped onto the input's planes) OCCT can fail
# silently (measured: the frame STL's model, common = 0 with identical bounding boxes). Then fuzzy booleans with a
# tolerance of 1e-7 to 1e-5 of the size; if the volumes still disagree, the result is an error, never a number.
# The three volumes also agree when all booleans fail the same way: common = 0, r+ = V_in, r- = V_model (measured
# 2026-09-27 on the liner: its rough model minus an exact R 125 cylinder on the liner's R 125 bore - IoU 3e-9 for a
# model that is 0.95 of the part). So the common volume is also checked against a sample: seeded points in the box of
# both, inside both by isInside; a disagreement beyond 4 sampling errors + 0.2 % of V_in is a failed boolean.
sampled_common, sampled_error = sample_common(inp, mod, diag, v_in, int(job.get("check_points", 1500)))

def plausible(common):
    return abs(common.Volume - sampled_common) <= sampled_error

# The general fuse first (the user's rule 2026-09-29: "use instead of boolean difference more stable functions like
# xor"): ONE run of OCCT's general fuse partitions both solids, and its map says which solid a piece came from - of the
# input only: missing, of the model only: invented. FreeCAD's boolean XOR and boolean fragments are this function.
# Measured 2026-09-29 on the institute server (FreeCAD 1.0.0), seven pairs:
#   liner + 39 features  r- 1 224 974.95 in 31 s  (incremental difference: 1 224 974.39 in 172 s; the cuts: no answer)
#   liner + 52 features  r- 1 088 794.47 in 40 s  (incremental: 1 088 794.72 in 193 s; the cuts: "Null shape")
#   rough liner, frame   right in 5 s and 1 s
#   frame with a fuzzy tolerance, liner + 60 features, review r13 and r08: WRONG (nothing fused, or pieces lost)
# So it is one more method, not the answer: faster, and right on the liner where the cuts fail, but no more stable in
# general. Two findings decide how it is used. (1) The piece both solids share is often lost from the result or has no
# volume, while the pieces of the difference are right: the shared volume is therefore taken from the two balances
# V_in - r+ and V_model - r-, which must agree within 1e-5 V_in (the liner + 60 case disagrees by 8e-5 V_in: refused)
# and with the sample. (2) Plain and fuzzy fail on different parts: every tolerance is tried.
def partition(a, b, fz):
    res, mp = a.generalFuse([b], fz or 0.0)
    def solids(lst):
        out = []
        for x in lst:
            out += [x] if x.ShapeType == "Solid" else list(x.Solids)
        return out
    sa, sb = solids(mp[0]), solids(mp[1])
    only_a = [x for x in sa if not any(x.isSame(y) for y in sb)]
    only_b = [x for x in sb if not any(x.isSame(y) for y in sa)]
    return only_a, only_b

def by_general_fuse():
    notes = []
    if not job.get("general_fuse", True):
        return None, notes
    for fz in (None, 6e-8 * diag, 6e-7 * diag):
        t_f = time.time()
        try:
            only_in, only_mod = partition(inp, mod, fz)
        except Exception as exc:
            notes.append(f"general fuse {fz}: {str(exc)[-80:]}")
            continue
        rp, rm = sum(x.Volume for x in only_in), sum(x.Volume for x in only_mod)
        c_in, c_mod = v_in - rp, v_mod - rm
        if rp < -1e-9 * v_in or rm < -1e-9 * v_in or any(not x.isValid() for x in only_in + only_mod):
            notes.append(f"general fuse {fz}: an invalid piece (r+ {rp:.4g}, r- {rm:.4g})")
        elif abs(c_in - c_mod) > job.get("fuse_check", 1e-5) * v_in:
            notes.append(f"general fuse {fz}: the balances disagree, {c_in:.6g} and {c_mod:.6g}")
        elif abs(0.5 * (c_in + c_mod) - sampled_common) > sampled_error:
            notes.append(f"general fuse {fz}: shared {0.5 * (c_in + c_mod):.4g}, the sample says {sampled_common:.4g}")
        else:
            return dict(plus=only_in, minus=only_mod, r_plus=rp, r_minus=rm, fuzzy=fz, seconds=time.time() - t_f,
                        balance_error=c_in - c_mod), notes
    return None, notes

import time
fused, fuse_notes = by_general_fuse()
if fused is not None:
    out = dict(input_volume=v_in, model_volume=v_mod, r_plus=fused["r_plus"], r_minus=fused["r_minus"],
               fuzzy=fused["fuzzy"], sampled_common=sampled_common, balance_error=fused["balance_error"],
               r_plus_solids=sum(1 for x in fused["plus"] if x.Volume > floor),
               r_minus_solids=sum(1 for x in fused["minus"] if x.Volume > floor),
               method="general fuse", method_seconds=fused["seconds"], tried=fuse_notes)
    out["iou"] = (v_in - out["r_plus"]) / (v_in + out["r_minus"])
    out["model_genus"] = genus(mod, tol)
    if job.get("state_dir"):
        write_state(job["state_dir"], fused["plus"], fused["minus"], dict(input_volume=v_in, model=job["model"]))
    if job.get("pieces_dir"):
        write_pieces(out, fused["plus"], fused["minus"], inp, mod, fused["fuzzy"], floor, diag, "general fuse")
    json.dump(out, open(job["out"], "w"))
    raise SystemExit(0)

# The intersection first: a try whose intersection is not plausible computes no cuts (measured 2026-09-29 on the
# liner's model with 39 features: every intersection came back empty after 15 to 24 s, the plain mod.cut(inp) had no
# answer after 275 s), and a boolean that raises ("Null shape" at fuzzy 1e-2 there) is a failed try, not a crash.
tries = [None, 1e-7 * diag, 1e-6 * diag, 1e-5 * diag]
failed = list(fuse_notes)
for fuzzy in tries:
    try:
        common = inp.common(mod) if fuzzy is None else inp.common([mod], fuzzy)
        if not plausible(common):
            failed.append(f"{fuzzy}: intersection {common.Volume:.4g}")
            continue
        if fuzzy is None:
            plus, minus = inp.cut(mod), mod.cut(inp)
        else:
            plus, minus = inp.cut([mod], fuzzy), mod.cut([inp], fuzzy)
    except Exception as exc:
        failed.append(f"{fuzzy}: {str(exc)[-80:]}")
        continue
    if consistent(plus, minus, common):
        break
    failed.append(f"{fuzzy}: r+ {plus.Volume:.4g}, r- {minus.Volume:.4g}, intersection {common.Volume:.4g}")
else:
    json.dump(dict(error="exact residual: booleans disagree (V_in - r+, V_model - r-, the intersection and its sample "
                         f"{sampled_common:.4g} +- {sampled_error:.3g}) even with fuzzy tolerance {tries[-1]:.3g}"
                         f" [{'; '.join(failed)}]",
                   input_volume=v_in, model_volume=v_mod), open(job["out"], "w"))
    raise SystemExit(0)
out = dict(input_volume=v_in, model_volume=v_mod, r_plus=plus.Volume, r_minus=minus.Volume, fuzzy=fuzzy,
           sampled_common=sampled_common,
           r_plus_solids=sum(1 for s in plus.Solids if s.Volume > floor),
           r_minus_solids=sum(1 for s in minus.Solids if s.Volume > floor))
out["iou"] = (v_in - out["r_plus"]) / (v_in + out["r_minus"])
out["model_genus"] = genus(mod, tol)
out["method"] = "occt"

if job.get("state_dir"):
    write_state(job["state_dir"], explode(plus), explode(minus), dict(input_volume=v_in, model=job["model"]))
if job.get("pieces_dir"):
    write_pieces(out, explode(plus), explode(minus), inp, mod, fuzzy, floor, diag, "childShapes")
json.dump(out, open(job["out"], "w"))
'''


def exact_residual(input_path, model_path, model_scale: float = 1.0, freecad: Optional[str] = None,
                   timeout: int = 900, min_solid: Optional[float] = None, booleans: bool = True,
                   pieces_dir=None, max_pieces: int = 500, state_dir=None, general_fuse: bool = True) -> dict:
    """The exact residual of `model_path` (a STEP or STL of the built model) against `input_path` (STEP or STL), and
    the model's genus from its B-rep. `booleans=False`: only the model's volume and genus (for inputs too large to
    turn into a solid in time; `sampled_iou` then stands in for the IoU).

    `general_fuse`: the general fuse is tried first (method "general fuse"; see the script), the separate booleans
    after it (method "occt"); False: the separate booleans only.

    `pieces_dir`: every solid of r_plus and r_minus above the volume floor (the `max_pieces` biggest per sign) is
    written there as `piece_<sign>_<i>.step` / `.stl`, and the result gets `pieces` (also `pieces.json`): id, sign,
    volume, bbox, centre and the piece's faces - type (plane, cylinder, cone, sphere, torus, other), area, outward
    normal, the surface's parameters, `convex` (the piece's material inside the surface) and `origin`: "input" for a
    face on the input's surface (a real surface of the feature), "model" for one on the model's (it only closes the
    piece), "both", "none". `left_out_count`: dict(small, capped) per sign, the solids under the volume floor
    (`min_solid`, default 1e-6 of the input's volume) and past `max_pieces`; `left_out_volume` their volume.

    `state_dir`: every solid of the difference (slivers too) is kept there as plus.brep / minus.brep with a
    state.json, for `incremental_residual` of the model's successors."""
    from reco_field.export import find_freecadcmd
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        return dict(error="freecadcmd not found")
    with tempfile.TemporaryDirectory() as tmp:
        script, job, out = Path(tmp) / "exact.py", Path(tmp) / "job.json", Path(tmp) / "out.json"
        script.write_text(SCRIPT)
        job.write_text(json.dumps(dict(input=str(input_path), model=str(model_path), model_scale=float(model_scale),
                                       general_fuse=bool(general_fuse),
                                       out=str(out), booleans=bool(booleans),
                                       **({} if min_solid is None else dict(min_solid=min_solid)),
                                       **({} if state_dir is None else dict(state_dir=str(state_dir))),
                                       **({} if pieces_dir is None else dict(pieces_dir=str(pieces_dir),
                                                                             max_pieces=int(max_pieces))))))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["EXACT_JOB"] = str(job)
        try:
            proc = subprocess.run([freecad, str(script)], env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return dict(error=f"exact residual: no answer within {timeout} s")
        if not out.is_file():
            return dict(error=f"exact residual failed: {(proc.stderr or proc.stdout)[-400:]}")
        return json.loads(out.read_text())


def mesh_residual(input_path, model_path, pieces_dir=None, max_pieces: int = 500, min_solid: Optional[float] = None,
                  input_mesh=None) -> dict:
    """The residual of the MESHES (the plan of 2026-10-15, step 3; the middle rung of the ladder exact -> mesh ->
    voxels): r+ = input minus model and r- = model minus input as exact mesh booleans (manifold3d), for the cases where
    OCCT's booleans disagree or give no answer in time (7.1 % of the ABC batch, 11 of 60 gate-bare runs). The same
    keys as `exact_residual` (input_volume, model_volume, r_plus, r_minus, iou, r_plus_solids, r_minus_solids,
    model_genus, method "mesh") and, with `pieces_dir`, the pieces of the difference as STL files with the schema of
    `write_pieces` - without a STEP and with an empty `faces` list: the feature phase's fit and cascade sources work
    on them, the face-typed sources (hole stacks, step faces) skip them. `input_mesh`: (vertices, triangles) instead
    of reading `input_path` (an STL, or a STEP tessellated by freecadcmd). The volumes are the meshes' own: a STEP
    input's tessellation differs from its exact solid by the deflection (0.02 mm here, below the sampled
    comparison's resolution). An error dict when manifold3d is missing or a mesh is not a closed solid."""
    try:
        import manifold3d as m
    except ImportError:
        return dict(error="mesh residual: manifold3d is not installed (pip install manifold3d)")
    from reco_gui.model_mesh import _from_triangles
    t0 = time.time()

    def solid_of_mesh(path, mesh):
        if mesh is None and str(path).lower().endswith(".stl"):
            import trimesh
            tm = trimesh.load(str(path), force="mesh", process=False)
            mesh = (np.asarray(tm.vertices, float), np.asarray(tm.faces))
        elif mesh is None:                                        # a STEP: tessellated by freecadcmd (load_field)
            import sys
            root = Path(__file__).resolve().parents[1]
            if str(root / "dataset_gen") not in sys.path:
                sys.path.insert(0, str(root / "dataset_gen"))
            from src.closest_point_field import load_field
            f = load_field(str(path))
            mesh = (np.asarray(f.vertices, float), np.asarray(f.faces))
        return _from_triangles(np.asarray(mesh[0], float), np.asarray(mesh[1], np.int64))
    try:
        inp = solid_of_mesh(input_path, input_mesh)
        mod = solid_of_mesh(model_path, None)
    except Exception as exc:
        return dict(error=f"mesh residual: {type(exc).__name__}: {str(exc)[-200:]}")
    if inp is None or mod is None:
        return dict(error=f"mesh residual: {'the input' if inp is None else 'the model'} mesh is not one closed solid")
    v_in, v_mod = float(inp.volume()), float(mod.volume())
    plus, minus = inp - mod, mod - inp
    r_plus, r_minus = (0.0 if plus.is_empty() else float(plus.volume())), (0.0 if minus.is_empty() else float(minus.volume()))
    floor = float(min_solid) if min_solid is not None else 1e-6 * v_in
    pieces_of = {"plus": [] if plus.is_empty() else plus.decompose(), "minus": [] if minus.is_empty() else minus.decompose()}
    out = dict(input_volume=v_in, model_volume=v_mod, r_plus=r_plus, r_minus=r_minus, fuzzy=None, method="mesh",
               iou=(v_in - r_plus) / (v_in + r_minus) if v_in + r_minus > 0 else 0.0,
               r_plus_solids=sum(1 for x in pieces_of["plus"] if x.volume() > floor),
               r_minus_solids=sum(1 for x in pieces_of["minus"] if x.volume() > floor),
               model_genus=int(mod.genus()) if mod.num_vert() else None, method_seconds=time.time() - t0)
    if pieces_dir is not None:
        folder = Path(pieces_dir)
        folder.mkdir(parents=True, exist_ok=True)
        pieces, slivers, small, capped = [], dict(plus=0.0, minus=0.0), dict(plus=0, minus=0), dict(plus=0, minus=0)
        for sign in ("plus", "minus"):
            solids = sorted(pieces_of[sign], key=lambda x: -x.volume())
            above = [x for x in solids if x.volume() > floor]
            kept = above[:max(1, int(max_pieces))]
            small[sign], capped[sign] = len(solids) - len(above), len(above) - len(kept)
            slivers[sign] = float(sum(x.volume() for x in solids) - sum(x.volume() for x in kept))
            for i, sol in enumerate(kept):
                mesh = sol.to_mesh()
                v = np.asarray(mesh.vert_properties, float)[:, :3]
                f = np.asarray(mesh.tri_verts, np.int64)
                name = f"piece_{sign}_{i:03d}"
                stl = folder / (name + ".stl")
                import trimesh
                trimesh.Trimesh(v, f, process=False).export(str(stl))
                tri = v[f]
                vol6 = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
                centre = (tri.sum(axis=1) * vol6[:, None]).sum(axis=0) / (4.0 * vol6.sum()) if abs(vol6.sum()) > 0 else v.mean(axis=0)
                bb = sol.bounding_box()
                pieces.append(dict(id=name, sign=sign, volume=float(sol.volume()), valid=True, step=None, stl=str(stl),
                                   bbox=[[float(bb[0]), float(bb[1]), float(bb[2])], [float(bb[3]), float(bb[4]), float(bb[5])]],
                                   centre=[float(x) for x in centre], faces=[]))
        out.update(pieces=pieces, left_out_volume=slivers, left_out_count=dict(small=small, capped=capped), pieces_method="mesh")
        json.dump(dict(input=str(input_path), model=str(model_path), pieces=pieces, left_out_volume=slivers),
                  open(folder / "pieces.json", "w"), indent=1)
    return out


POINT_DISTANCE_SCRIPT = r'''
import json, os
import numpy as np
import FreeCAD, Part, Mesh
job = json.load(open(os.environ["POINT_DISTANCE_JOB"]))
path = job["shape"]
if path.lower().endswith((".step", ".stp")):
    shape = Part.read(path)
else:                                                       # an STL: one solid of its facets
    m = Mesh.Mesh(path)
    sh = Part.Shape()
    sh.makeShapeFromMesh(m.Topology, 1e-6 * max(m.BoundBox.DiagonalLength, 1.0))
    shape = Part.makeSolid(Part.makeShell(sh.Faces))
solids = shape.Solids or [shape]
pts = np.load(job["points"])
dist = np.empty(len(pts), float)
inside = np.zeros(len(pts), bool)
tol = float(job.get("tolerance", 1e-6))
for i, p in enumerate(pts):
    v = FreeCAD.Vector(float(p[0]), float(p[1]), float(p[2]))
    best = None
    for sol in solids:
        d = sol.distToShape(Part.Vertex(v))[0]
        best = d if best is None or d < best else best
        if job.get("signed") and not inside[i] and sol.isInside(v, tol, True):
            inside[i] = True
    dist[i] = best if best is not None else float("nan")
np.save(job["dist_out"], dist)
np.save(job["inside_out"], inside)
json.dump(dict(n=int(len(pts)), solids=len(solids), faces=sum(len(s.Faces) for s in solids)), open(job["out"], "w"))
'''


def point_distances(shape_path, points, signed: bool = True, timeout: int = 3600, freecad: Optional[str] = None) -> dict:
    """The EXACT distance of every point to the B-rep of `shape_path` (a STEP; an STL becomes the solid of its facets),
    by OCCT's BRepExtrema (Part.Shape.distToShape) in freecadcmd, and with `signed` the material flag of every point
    (Part.Shape.isInside: the point lies in the solid). Returns dict(distance (n,), inside (n,) bool, n, solids,
    faces) or dict(error). Measured 2026-10-15 on cy_t's model (5 faces): 2.8 ms per point for the distance and 2 ms
    for the flag - a minute per 12 000 points. The user's rule of 2026-10-15: the nephogram between STEP files, not
    meshes - the points are the tessellations' vertices (on the surfaces), the target is the exact B-rep."""
    pts = np.asarray(points, float).reshape(-1, 3)
    if not len(pts):
        return dict(distance=np.zeros(0), inside=np.zeros(0, bool), n=0, solids=0, faces=0)
    with tempfile.TemporaryDirectory() as tmp:
        p_in, d_out, i_out = Path(tmp) / "points.npy", Path(tmp) / "dist.npy", Path(tmp) / "inside.npy"
        np.save(p_in, pts)
        r = _run_job(POINT_DISTANCE_SCRIPT, "POINT_DISTANCE_JOB",
                     dict(shape=str(shape_path), points=str(p_in), dist_out=str(d_out), inside_out=str(i_out), signed=bool(signed)),
                     timeout=timeout, freecad=freecad)
        if "error" in r:
            return dict(error=f"point distances: {r['error']}")
        return dict(r["result"], distance=np.load(d_out), inside=np.load(i_out))


def exact_faces(path, freecad: Optional[str] = None, timeout: int = 900) -> dict:
    """The faces of a STEP part (or of an STL's facets): dict(faces, volume), every face as in `exact_residual`'s
    pieces (type, area, outward normal, the surface's parameters, convex) plus its bbox and, for a surface with an axis,
    `along` (the trimmed face's reach along its axis, from `center`), without an origin.
    Returns {"error": ...} instead of raising."""
    from reco_field.export import find_freecadcmd
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        return dict(error="freecadcmd not found")
    with tempfile.TemporaryDirectory() as tmp:
        script, job, out = Path(tmp) / "exact.py", Path(tmp) / "job.json", Path(tmp) / "out.json"
        script.write_text(SCRIPT)
        job.write_text(json.dumps(dict(faces_of=str(path), out=str(out))))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["EXACT_JOB"] = str(job)
        try:
            proc = subprocess.run([freecad, str(script)], env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return dict(error=f"exact faces: no answer within {timeout} s")
        if not out.is_file():
            return dict(error=f"exact faces failed: {(proc.stderr or proc.stdout)[-400:]}")
        return json.loads(out.read_text())


def exact_facts(path, stl_out=None, freecad: Optional[str] = None, timeout: int = 900) -> dict:
    """The faces of a STEP part (or an STL's facets) as `exact_faces`, plus its topology: dict(faces, volume, area,
    valid, bbox, counts (solids, shells, faces, wires, edges, vertexes), betti (b0, b1, b2, shells: the genus per shell;
    None if a tessellation is not closed)); `stl_out`: also its tessellation (deflection 1e-3 of the diagonal) there.
    Returns {"error": ...} instead of raising."""
    from reco_field.export import find_freecadcmd
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        return dict(error="freecadcmd not found")
    with tempfile.TemporaryDirectory() as tmp:
        script, job, out = Path(tmp) / "exact.py", Path(tmp) / "job.json", Path(tmp) / "out.json"
        script.write_text(SCRIPT)
        job.write_text(json.dumps(dict(facts_of=str(path), out=str(out), stl_out=str(stl_out) if stl_out else None)))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["EXACT_JOB"] = str(job)
        try:
            proc = subprocess.run([freecad, str(script)], env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return dict(error=f"exact facts: no answer within {timeout} s")
        if not out.is_file():
            return dict(error=f"exact facts failed: {(proc.stderr or proc.stdout)[-400:]}")
        return json.loads(out.read_text())


def sampled_iou(field, model, max_cells: float = 8e6) -> dict:
    """An IoU measured independently of OCCT: input inside/outside by ray parity of the input mesh, model
    inside/outside by the program's exact CSG signed distance (`model.sdf <= 0`), both at the centres of one box grid
    (dataset_gen.src.local_grids, in slabs). Pitch: the finer of tau / 3 (tau the thinnest wall, 5th percentile) and
    1/64 of the part's longest side, coarsened only as far as `max_cells` requires over the union's bounding box. Accurate to a few percent on thin-walled parts (the
    frame at 12-26 mm: 0.99-0.87 of the volume, docs/FRAME_FAILURE.md); a cross-check that catches gross boolean
    failures (IoU 0 vs 1), not a replacement for the exact number. `exact_residual` does not call it; the loop's
    grade does (reco_engine.residual_loop._exact_grade, max_disagreement 0.05)."""
    import numpy as np
    from src.local_grids import box_grid, iter_slabs, wall_thickness
    lo_in, hi_in = np.asarray(field.bounds[0], float), np.asarray(field.bounds[1], float)
    lo_m, hi_m = (np.asarray(b, float) for b in model.bounds())
    lo, hi = np.minimum(lo_in, np.maximum(lo_m, lo_in - (hi_in - lo_in))), \
        np.maximum(hi_in, np.minimum(hi_m, hi_in + (hi_in - lo_in)))     # a model far outside is clipped to 3x the input
    _, _, tau = wall_thickness(field.vertices, field.faces, n=2000)
    tau = tau[np.isfinite(tau) & (tau > 0)]
    pitch = float((hi_in - lo_in).max()) / 64                   # resolve the part itself ...
    if len(tau):
        pitch = min(pitch, float(np.percentile(tau, 5)) / 3.0)  # ... and its thinnest wall at the licensed tau / 3 (D-005)
    pitch = max(pitch, float(np.prod(hi - lo) / max_cells) ** (1.0 / 3.0))     # coarsened only as far as the cap needs
    grid = box_grid(lo, hi, pitch)
    ys = grid.axis(1)
    xs, zs = grid.axis(0), grid.axis(2)
    both = either = 0
    j = 0
    for occ in iter_slabs(field.vertices, field.faces, grid):
        yy = ys[j:j + occ.shape[1]]
        j += occ.shape[1]
        P = np.stack(np.meshgrid(xs, yy, zs, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = np.asarray(model.sdf(P)).reshape(occ.shape) <= 0
        both += int((occ & inside).sum())
        either += int((occ | inside).sum())
    return dict(iou=both / either if either else float("nan"), pitch=pitch, cells_union=either)


# ------------------------------------------------------------------------------------------------ feature checks
# The user's request 2026-09-27 ("use compound explode to check with FreeCAD our results for features"): per applied
# feature, three exact shapes show whether it is right - common(feature, piece), feature - piece (what it takes beyond
# its piece: must be excess or air) and piece - feature (what it leaves of the piece) - and the INPUT material inside
# feature - piece (a cut's harm; for an add, feature - piece outside the input). The shapes are built by the export
# builder's own `build` (reco_field/freecad_builder.py, its Part conventions), from the feature's program JSON.

_BUILDER_PRELUDE = r"""
import json, os
import ast
# the builder's functions and constants, not the statements that run it: cutting the text at its last "main()" left
# the builder's closing "if GROUPS_JSON: main_groups() / else: main()" as an "else:" without a body - a syntax error
# that failed every check and the document (2026-09-29), silently: the checks are advisory
_tree = ast.parse(open(os.environ["BUILDER_PY"]).read())
_tree.body = [n for n in _tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
              or not any(isinstance(c, ast.Call) and getattr(c.func, "id", "").startswith("main") for c in ast.walk(n))]
exec(compile(_tree, "freecad_builder", "exec"))
"""

CHECKS_SCRIPT = _BUILDER_PRELUDE + r"""
job = json.load(open(os.environ["CHECKS_JOB"]))
inp = Part.read(job["input"])
out = []
for it in job["items"]:
    try:
        F = build(it["program"]).removeSplitter()
        P = Part.read(it["piece_step"])
        # the builder's checked boolean (plain and fuzzy, points judge where they differ), not OCCT's plain one: on
        # the liner's transferred pocket tools (run 20260928-220854, pieces #5, #6, #7, #9, #10) the plain common came
        # back empty and the plain cuts as their first operand - common 0, left = the whole piece, for tools that
        # cover 0.987 of their pieces (the end planes of a moved tool lie 1e-6 mm off the piece's)
        n_log = len(BOOLEAN_LOG)
        def checked(a, b, kind):                                   # an empty operand (a tool inside its piece leaves
            if a.isNull() or not a.BoundBox.isValid():             # nothing beyond it) has no box to compare
                return a
            return boolean(a, b, kind)
        common, beyond, left = checked(F, P, "Intersection"), checked(F, P, "Subtraction"), checked(P, F, "Subtraction")
        harm = checked(beyond, inp, "Intersection" if it["op"] == "cut" else "Subtraction")
        row = dict(id=it["id"], feature_volume=F.Volume, piece_volume=P.Volume, common_volume=common.Volume,
                   beyond_piece_volume=beyond.Volume, left_of_piece_volume=left.Volume, harm_volume=harm.Volume,
                   fuzzy_booleans=sum(1 for b in BOOLEAN_LOG[n_log:] if b["taken"] == "fuzzy"))
        # the balance: common + beyond = the feature, common + left = the piece; else a boolean failed - no numbers
        tol = float(job.get("rel_check", 1e-4))
        if abs(common.Volume + beyond.Volume - F.Volume) > tol * F.Volume or \
                abs(common.Volume + left.Volume - P.Volume) > tol * P.Volume:
            row = dict(id=it["id"], error="feature check: the booleans disagree (feature %.6g = common %.6g + beyond "
                       "%.6g, piece %.6g = common + left %.6g)" % (F.Volume, common.Volume, beyond.Volume, P.Volume,
                                                                   left.Volume))
        out.append(row)
    except Exception as exc:
        out.append(dict(id=it["id"], error=str(exc)[-300:]))
json.dump(out, open(job["out"], "w"))
"""


def _run_job(script_text: str, env_name: str, job: dict, timeout: int, freecad: Optional[str] = None) -> dict:
    """Run a FreeCAD script with a JSON job; returns the JSON it writes to job['out'] or {"error": ...}."""
    from reco_field.export import BUILDER, find_freecadcmd
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        return dict(error="freecadcmd not found")
    with tempfile.TemporaryDirectory() as tmp:
        script, jpath, out = Path(tmp) / "job.py", Path(tmp) / "job.json", Path(tmp) / "out.json"
        script.write_text(script_text)
        jpath.write_text(json.dumps(dict(job, out=str(out))))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env[env_name], env["BUILDER_PY"] = str(jpath), str(BUILDER)
        try:
            proc = subprocess.run([freecad, str(script)], env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return dict(error=f"no answer within {timeout} s")
        if not out.is_file():
            return dict(error=f"failed: {(proc.stderr or proc.stdout)[-400:]}")
        return dict(result=json.loads(out.read_text()), log=(proc.stderr or "") + (proc.stdout or ""))


def feature_checks(input_path, items, timeout: int = 300, freecad: Optional[str] = None) -> dict:
    """The exact checks of features against their pieces: `items` [dict(id, op, program (export.program_to_dict of
    the primitive's SDF), piece_step)] -> {id: dict(feature_volume, piece_volume, common_volume, beyond_piece_volume,
    left_of_piece_volume, harm_volume, fuzzy_booleans) or dict(error)}; {"error": ...} when FreeCAD failed as a
    whole. Every boolean is the builder's checked one; an item whose volumes do not balance is an error, not a row."""
    r = _run_job(CHECKS_SCRIPT, "CHECKS_JOB", dict(input=str(input_path), items=list(items)), timeout, freecad)
    if "error" in r:
        return r
    return {d["id"]: d for d in r["result"]}


# ------------------------------------------------------------------------------------------------ the incremental difference
# The difference of the NEXT model from the difference of the model before and the tools applied since (2026-09-29).
# The global boolean of input against model fails once the features are in the model (liner, 39 features: every
# intersection empty, plain and fuzzy); but the model changed only inside its new tools T:
#
#     cut T:  r-' = r- - T            r+' = r+ + (T & input) - r+      (T & input & model: what the cut harms)
#     add T:  r+' = r+ - T            r-' = r- + (T - input) - r-
#
# so every boolean is one of a SMALL solid: a piece against the tools that touch its bounding box, a tool against the
# input. Checked by the volume budget (V_in - r+' = V_model' - r-', the new model's volume from its own STEP) and by
# the same sample of the intersection as the global difference.

INCREMENTAL_SCRIPT = _BUILDER_PRELUDE + HELPERS + r'''
t0 = time.time()
seconds = {}
info, plus, minus = read_state(job["state"])
inp = solid_of(job["input"], 1.0)
mod = solid_of(job["model"], 1.0)
v_in, v_mod = inp.Volume, mod.Volume
floor = job.get("min_solid", 1e-6 * v_in)
diag = max(inp.BoundBox.DiagonalLength, mod.BoundBox.DiagonalLength)
tiny = 1e-12 * v_in                                         # a solid of no volume is no solid
seconds["load"] = time.time() - t0
n_bool = [0]
spent = dict(build=0.0, pieces=0.0, input=0.0, points=0.0)
repaired = []

_face_boxes = {}

def face_boxes(a):
    """The boxes of a solid's faces, kept per solid (hashCode: the pieces live as long as the run)."""
    key = a.hashCode()
    if key not in _face_boxes or not _face_boxes[key][0].isSame(a):
        _face_boxes[key] = (a, [f.BoundBox for f in a.Faces])
    return _face_boxes[key][1]

def touches(a, b, eps=1e-6):
    """Can a and b share volume? Their boxes meet, and a face of one lies in the other's box or one holds a vertex
    of the other. The boxes alone are no filter where a piece is a skin around the part (the liner's rough model
    left one of 676 faces with the part's own box: every tool met it, 2 booleans each)."""
    ba, bb = a.BoundBox, b.BoundBox
    ba.enlarge(eps * diag)
    if not ba.intersect(bb):
        return False
    bb.enlarge(eps * diag)
    if any(ba.intersect(f) for f in face_boxes(b)) and any(bb.intersect(f) for f in face_boxes(a)):
        return True
    return a.isInside(b.Vertexes[0].Point, eps * diag, True) or b.isInside(a.Vertexes[0].Point, eps * diag, True)

floor_rest = float(job.get("min_rest", 1e-9 * v_in))
import random
_rng = random.Random(0)
unsure = []
TOLERANCES = (FUZZY, None, 1e-5, 1e-3, 1e-2)               # fuzzy first: of 387 plain booleans on the liner 54 were wrong

def volume(solids):
    return float(sum(s.Volume for s in solids))

def solids_of(a, b, kind, fz):
    n_bool[0] += 1
    if fz is None:
        r = a.common(b) if kind == "Intersection" else a.cut(b)
    else:
        r = a.common([b], fz) if kind == "Intersection" else a.cut([b], fz)
    return [s for s in explode(r) if s.Volume > tiny]

def shared_points(a, b, n):
    """How many of n seeded points in the overlap of the boxes of a and b lie in both (well inside)."""
    ba, bb = a.BoundBox, b.BoundBox
    lo = [max(ba.XMin, bb.XMin), max(ba.YMin, bb.YMin), max(ba.ZMin, bb.ZMin)]
    hi = [min(ba.XMax, bb.XMax), min(ba.YMax, bb.YMax), min(ba.ZMax, bb.ZMax)]
    t_p, hit = time.time(), 0
    for _ in range(n):
        q = FreeCAD.Vector(*[l + _rng.random() * (h - l) for l, h in zip(lo, hi)])
        hit += a.isInside(q, 1e-6, False) and b.isInside(q, 1e-6, False)     # on a face: not inside
    spent["points"] += time.time() - t_p
    return hit

def split(a, b, what):
    """(a - b, a & b) as lists of solids, checked: OCCT fails silently where faces coincide - a tool's faces lie ON
    the input's, a piece's are the input's - plain and fuzzy, and not always alike (liner, 39 features: one bore
    piece of 2907.0 mm^3 lost 627.4 to its tool, 16 seeded points saw nothing wrong). So both results are computed
    and their volumes must add up to a's; an empty intersection of two solids that touch is believed only when
    no seeded point lies in both. The first tolerance of TOLERANCES that passes is taken; none: the one with the
    smallest error, listed in `unsure`."""
    va = a.Volume
    tol = 3e-4 * min(va, b.Volume) + 1e-9 * v_in            # measured: 0.1 to 0.5 mm^3 on bores of 2907 mm^3
    n = int(job.get("boolean_points", 16))
    best, tries, shared = None, [], None
    for fz in TOLERANCES:
        try:
            rest, both = solids_of(a, b, "Subtraction", fz), solids_of(a, b, "Intersection", fz)
        except Exception as exc:
            tries.append(dict(fuzzy=fz, error=str(exc)[-60:]))
            continue
        err = abs(va - volume(rest) - volume(both))
        tries.append(dict(fuzzy=fz, error=err, shared=volume(both)))
        if volume(both) <= tol and n > 0:                   # "nothing shared": not with points in both
            shared = shared_points(a, b, n) if shared is None else shared
            if shared:
                tries[-1]["points_in_both"] = shared
                continue
        if best is None or err < best[0]:
            best = (err, rest, both, fz)
        if err <= tol:
            break
    if best is None:
        unsure.append(dict(what=what, error="no boolean answered", volume=va, tries=tries))
        return [a], []
    if best[0] > tol:
        unsure.append(dict(what=what, error=best[0], tolerance=tol, fuzzy=best[3], volume=va, tries=tries))
    elif best[3] != TOLERANCES[0]:
        repaired.append(dict(what=what, fuzzy=best[3]))
    return best[1], best[2]

t = time.time()
tool_log = []
for k, it in enumerate(job["tools"]):
    t1 = time.time()
    T = build(it["program"])
    spent["build"] += time.time() - t1
    cut = it["op"] == "cut"
    mine, other = (minus, plus) if cut else (plus, minus)
    # The tool takes what it covers of the pieces of its sign. What of it lies beyond them (`rest`) is a cut's part
    # in the input (its harm) or in the air, an add's part outside the input or in the model already: only that part
    # meets the input - nothing, for a tool that stays inside its pieces.
    rest, left, taken = [T], [], 0.0
    for q in mine:
        if not touches(q, T):
            left.append(q)
            continue
        keep, both = split(q, T, f"tool {k}: piece - tool")
        left += keep
        taken += volume(both)
        if both:                                            # the tool without this piece: the pieces are disjoint
            more = []
            for s in rest:
                more += split(s, q, f"tool {k}: tool - piece")[0] if touches(s, q) else [s]
            rest = more
    mine = left
    t2 = time.time()
    spent["pieces"] += t2 - t1
    if abs(T.Volume - taken - volume(rest)) > 3e-4 * T.Volume + 1e-9 * v_in:
        unsure.append(dict(what=f"tool {k}: its volume {T.Volume:.6g} is not what it took {taken:.6g} + its rest "
                                f"{volume(rest):.6g}", error=T.Volume - taken - volume(rest)))
    new = []
    for s in rest:
        if s.Volume > floor_rest:
            outside, inside = split(s, inp, f"tool {k}: rest against the input")
            new += inside if cut else outside
    for q in other:
        if any(touches(s, q) for s in new):
            more = []
            for s in new:
                more += split(s, q, f"tool {k}: beyond - piece")[0] if touches(s, q) else [s]
            new = more
    other = other + new
    spent["input"] += time.time() - t2
    plus, minus = (other, mine) if cut else (mine, other)
    tool_log.append(dict(index=k, op=it["op"], volume=float(T.Volume), taken=taken, rest=volume(rest),
                         beyond=volume(new), seconds=time.time() - t1, seconds_input=time.time() - t2,
                         booleans=n_bool[0]))
    if job.get("progress"):                                 # what a run that hits its timeout got done
        open(job["progress"], "a").write(json.dumps(tool_log[-1]) + "\n")
seconds["booleans"] = time.time() - t
r_plus, r_minus = sum(s.Volume for s in plus), sum(s.Volume for s in minus)
rel = job.get("rel_check", 1e-5) * v_in                     # measured on the liner, 52 tools: 1.5 mm^3 = 3e-8
budget = (v_mod - v_in) - (r_minus - r_plus)               # 0 when the pieces explain the model's volume
t = time.time()
n_check = int(job.get("check_points", 1500)) if job["tools"] else 0      # no tools: the state was checked
sampled_common, sampled_error = sample_common(inp, mod, diag, v_in, n_check) if n_check > 0 else (None, None)
seconds["sample"] = time.time() - t
out = dict(input_volume=v_in, model_volume=v_mod, r_plus=r_plus, r_minus=r_minus, fuzzy=None, method="incremental",
           budget_error=budget, sampled_common=sampled_common, tools=len(job["tools"]), booleans=n_bool[0],
           repairs=len(repaired),
           r_plus_solids=sum(1 for s in plus if s.Volume > floor),
           r_minus_solids=sum(1 for s in minus if s.Volume > floor))
if abs(budget) > rel:
    out = dict(error=f"incremental difference: the pieces do not explain the model's volume (V_model - V_in = "
                     f"{v_mod - v_in:.6g}, r- - r+ = {r_minus - r_plus:.6g}, tolerance {rel:.3g})",
               input_volume=v_in, model_volume=v_mod)
elif sampled_common is not None and abs((v_in - r_plus) - sampled_common) > sampled_error:
    out = dict(error=f"incremental difference: the intersection {v_in - r_plus:.6g} is not its sample "
                     f"{sampled_common:.4g} +- {sampled_error:.3g}", input_volume=v_in, model_volume=v_mod)
else:
    out["iou"] = (v_in - r_plus) / (v_in + r_minus)
    out["model_genus"] = genus(mod, 1e-3 * mod.BoundBox.DiagonalLength)
    t = time.time()
    if job.get("state_dir"):
        write_state(job["state_dir"], plus, minus, dict(input_volume=v_in, model=job["model"]))
    if job.get("pieces_dir"):
        write_pieces(out, plus, minus, inp, mod, FUZZY, floor, diag, "incremental")
    seconds["pieces"] = time.time() - t
out["seconds"] = dict(seconds, total=time.time() - t0)
out["tool_log"], out["unsure"], out["repaired"], out["spent"] = tool_log, unsure, repaired, spent
json.dump(out, open(job["out"], "w"))
'''


def incremental_residual(input_path, model_path, state, tools, state_dir=None, freecad: Optional[str] = None,
                         timeout: int = 900, min_solid: Optional[float] = None, pieces_dir=None,
                         max_pieces: int = 500, check_points: int = 200, options: Optional[dict] = None) -> dict:
    """The exact residual of `model_path` against `input_path` WITHOUT the boolean of the two: from `state` (the
    folder an earlier `exact_residual(..., state_dir=)` or `incremental_residual` wrote: every solid of the difference
    of the model BEFORE) and `tools`, [dict(op "cut" / "add", program (export.program_to_dict))] in the order they
    were applied since. No tools: the state's own pieces. The result is `exact_residual`'s, with method
    "incremental", `budget_error` (V_model - V_in - (r_minus - r_plus), the volume the pieces do not explain),
    `booleans` (how many were computed), `repaired` (pairs whose first tolerance failed the volume check and another
    passed it), `unsure` (pairs no tolerance got right, tools whose volume is not what they took plus their rest;
    the budget decides whether that is an error), `tool_log` (per tool:
    its volume, what it took of the pieces, what of it lies beyond them, seconds) and `seconds` (load, booleans,
    sample, pieces, total). `model_path` is only read: its
    volume, its genus, its faces for the pieces' origin. {"error": ...} when the budget or the sample of the
    intersection disagree, or FreeCAD failed."""
    job = dict(input=str(input_path), model=str(model_path), state=str(state), tools=list(tools),
               check_points=int(check_points), **(options or {}))
    if min_solid is not None:
        job["min_solid"] = min_solid
    if state_dir is not None:
        job["state_dir"] = str(state_dir)
    if pieces_dir is not None:
        job.update(pieces_dir=str(pieces_dir), max_pieces=int(max_pieces))
    r = _run_job(INCREMENTAL_SCRIPT, "EXACT_JOB", job, timeout, freecad)
    if "error" in r:
        return dict(error=f"incremental difference: {r['error']}")
    return r["result"]


DOCUMENT_SCRIPT = _BUILDER_PRELUDE + r"""
import FreeCAD as App
import CompoundTools.Explode as Explode
job = json.load(open(os.environ["DOC_JOB"]))
doc = App.newDocument("features")
for o in list(doc.Objects):                                         # the builder prelude's own document is another one
    pass

def group(name, label, parent=None):
    g = doc.addObject("App::DocumentObjectGroup", name)
    g.Label = label
    if parent is not None:
        parent.addObject(g)
    return g

def solid(name, shape, label, grp):
    o = doc.addObject("Part::Feature", name)
    o.Shape, o.Label = shape, label
    grp.addObject(o)
    return o

def part_object(name, prim, program, label, grp):
    # the real Part::Box / Cylinder / Cone / Sphere for a full primitive (its Placement), else the built shape
    shape = build(program).removeSplitter()
    kind, d = prim["kind"], prim["dims_mm"]
    q = prim["quaternion"]
    rot = App.Rotation(q[0], q[1], q[2], q[3])
    c = App.Vector(*prim["center_mm"])
    o = None
    if not prim.get("u") and not prim.get("v"):
        if kind == "box":
            o = doc.addObject("Part::Box", name)
            o.Length, o.Width, o.Height = 2 * d["x_half"], 2 * d["y_half"], 2 * d["z_half"]
            o.Placement = App.Placement(c - rot.multVec(App.Vector(d["x_half"], d["y_half"], d["z_half"])), rot)
        elif kind == "cylinder":
            o = doc.addObject("Part::Cylinder", name)
            o.Radius, o.Height = d["radius"], 2 * d["half_length"]
            o.Placement = App.Placement(c - rot.multVec(App.Vector(0, 0, d["half_length"])), rot)
        elif kind == "cone":
            o = doc.addObject("Part::Cone", name)
            o.Radius1, o.Radius2, o.Height = max(d["radius1"], 0.0), max(d["radius2"], 0.0), 2 * d["half_length"]
            o.Placement = App.Placement(c - rot.multVec(App.Vector(0, 0, d["half_length"])), rot)
        elif kind == "sphere":
            o = doc.addObject("Part::Sphere", name)
            o.Radius = d["radius"]
            o.Placement = App.Placement(c, rot)
    if o is not None:
        doc.recompute()
        if abs(o.Shape.Volume - shape.Volume) > 1e-6 * max(shape.Volume, 1.0):
            doc.removeObject(o.Name)                                 # our reading of Part's convention failed: the shape
            o = None
    if o is None:
        o = doc.addObject("Part::Feature", name)
        o.Shape = shape
    o.Label = label
    grp.addObject(o)
    return o, shape

colours = {"accepted": (0.1, 0.7, 0.2), "rejected": (0.85, 0.15, 0.1), "skipped": (0.6, 0.6, 0.6), "": (0.5, 0.5, 0.8)}
inp = Part.read(job["input"])
solid("Input", inp, "input", group("G_input", "Input"))
rough = Part.read(job["rough"])
solid("Rough", rough, "rough model", group("G_rough", "Rough model"))
final = Part.read(job["final"])
solid("Final", final, "final model", group("G_final", "Final model"))
counts = []
for itn in job["iterations"]:
    g = group(f"G_diff{itn['iteration']}", f"Difference, iteration {itn['iteration']}")
    model = Part.read(itn["model_step"])
    fz = itn.get("fuzzy")
    for sign, shape, label in (("minus", model.cut([inp], fz) if fz else model.cut(inp), "invented (model - input)"),
                               ("plus", inp.cut([model], fz) if fz else inp.cut(model), "missing (input - model)")):
        comp = doc.addObject("Part::Feature", f"D{itn['iteration']}_{sign}")
        comp.Shape, comp.Label = shape, label
        g.addObject(comp)
        doc.recompute()
        res = Explode.explodeCompound(comp)
        kids = res[1] if isinstance(res, tuple) else []
        doc.recompute()
        pieces = [p for p in itn["pieces"] if p["sign"] == sign]
        used = set()
        for k in kids:
            v = k.Shape.Volume
            cm = k.Shape.CenterOfMass
            best = None
            for i, p in enumerate(pieces):
                if i in used or abs(p["volume"] - v) > 1e-4 * max(v, 1e-9) + 1e-9:
                    continue
                dist = (App.Vector(*p["centre"]) - cm).Length
                if best is None or dist < best[0]:
                    best = (dist, i)
            if best is not None:
                used.add(best[1])
                p = pieces[best[1]]
                k.Label = p["label"]
                k.addProperty("App::PropertyString", "Verdict", "Feature").Verdict = p.get("verdict") or ""
            else:
                k.Label = f"piece, not taken, {v:.4g} mm3"
                k.addProperty("App::PropertyString", "Verdict", "Feature").Verdict = "not taken"
        counts.append(dict(iteration=itn["iteration"], sign=sign, exploded=len(kids), matched=len(used)))
fg = group("G_features", "Features")
rg = group("G_rejected", "Rejected guesses", fg)
cg = group("G_checks", "Checks")
def tree_object(name, prim, label, grp):
    # a composite as its boolean tree: Part::MultiCommon / Part::MultiFuse / Part::Cut of its operands' objects
    if prim["kind"] != "composite":
        return part_object(name, prim, prim["program"], label, grp)
    sub = group(name + "_ops", label + " (operands)", grp)
    a, _ = tree_object(name + "a", prim["parts"][0], prim["parts"][0].get("label") or prim["parts"][0]["kind"], sub)
    b, _ = tree_object(name + "b", prim["parts"][1], prim["parts"][1].get("label") or prim["parts"][1]["kind"], sub)
    if prim["combine"] == "cut":
        o = doc.addObject("Part::Cut", name)
        o.Base, o.Tool = a, b
    else:
        o = doc.addObject("Part::MultiCommon" if prim["combine"] == "common" else "Part::MultiFuse", name)
        o.Shapes = [a, b]
    doc.recompute()
    o.Label = label
    grp.addObject(o)
    return o, o.Shape

for i, f in enumerate(job["features"]):
    o, F = tree_object(f"F{i:03d}", f["primitive"], f["label"], fg)
    if f.get("piece_step"):
        P = Part.read(f["piece_step"])
        for tag, sh in (("common", F.common(P)), ("feature - piece", F.cut(P)), ("piece - feature", P.cut(F))):
            solid(f"C{i:03d}_{tag.replace(' ', '').replace('-', '_')}", sh,
                  f"piece {f['id']:03d}: {tag}, {sh.Volume:.4g} mm3", cg)
for i, f in enumerate(job["rejected"]):
    try:
        tree_object(f"R{i:03d}", f["primitive"], f["label"], rg)
    except Exception:
        pass
sheet = doc.addObject("Spreadsheet::Sheet", "Summary")
sheet.set("A1", "exact IoU before")
sheet.set("B1", str(job["iou_before"]))
sheet.set("C1", "exact IoU after")
sheet.set("D1", str(job["iou_after"]))
cols = job["columns"]
for j, name in enumerate(cols):
    sheet.set(f"{chr(65 + j)}3", name)
for r, row in enumerate(job["rows"], start=4):
    for j, name in enumerate(cols):
        v = row.get(name)
        sheet.set(f"{chr(65 + j)}{r}", "" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v)))
doc.recompute()
doc.saveAs(job["path"])
macro = ["import FreeCAD as App, FreeCADGui as Gui", "doc = App.ActiveDocument",
         "C = " + repr(colours),
         "for o in doc.Objects:",
         "    v = getattr(o, 'Verdict', None)",
         "    if v is not None and hasattr(o, 'ViewObject') and o.ViewObject is not None:",
         "        o.ViewObject.ShapeColor = C.get(v, C[''])"]
open(job["macro"], "w").write("\n".join(macro) + "\n")
json.dump(dict(path=job["path"], macro=job["macro"], exploded=counts), open(job["out"], "w"))
"""


def write_features_document(job: dict, timeout: int = 900, freecad: Optional[str] = None) -> dict:
    """features.FCStd (headless, freecadcmd) from a job the loop's feature phase writes: input, rough and final model,
    per iteration the difference's two compounds exploded by CompoundTools.Explode with every piece labelled and its
    `Verdict` property, the applied features as Part objects with their checks, the rejected guesses, a Summary
    spreadsheet, and features_colours.FCMacro (colours by verdict - headless there is no view provider). Returns
    dict(path, macro, exploded) or {"error": ...}; never raises."""
    Path(job["path"]).with_name("features_job.json").write_text(json.dumps(job, indent=1, default=str))
    r = _run_job(DOCUMENT_SCRIPT, "DOC_JOB", job, timeout, freecad)
    if "error" not in r and not Path(job["path"]).is_file():
        return dict(error=f"FreeCAD reported no error but wrote no {job['path']}: {r.get('log', '')[-300:]}")
    return r.get("result", r) if "error" not in r else r
