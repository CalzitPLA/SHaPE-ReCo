# -*- coding: utf-8 -*-
"""
FreeCAD side of the special-shape subclasses (hollow, tube, rhs, frame). Run INSIDE FreeCAD:

    SPECIAL_JOB_JSON=jobs.json SPECIAL_RESULT_JSON=results.json freecadcmd dataset_gen/src/special_freecad.py

`jobs.json` is a list of jobs written by dataset_gen/src/special_generator.py (the host draws every random number; this
script only builds). Per job one Part solid is built, checked, rotated, scaled and placed in the in-scale grid, exported
to STL, and described in the result list (same order, one dict per job, errors reported per job).

Local frame of every parent: centred on the origin (a cylinder / cone runs from z = -h/2 to +h/2, radius1 at -h/2), axis
along z. The placement is x_world = k (R x_local - b) + c: R the sample rotation, b the centre of the rotated shape's
bounding box, k = size_fraction * grid_mm / (largest rotated extent), c = the grid position (pos_unit in [0, 1]^3 over
the range that keeps the box inside the grid, as src/inscale.sample_placement draws it).

Subclasses:
  hollow  the parent minus an inner copy shrunk by the wall w (box: half - w; sphere: r - w; cylinder: r - w, h - 2w;
          cone: both flanks moved in by w perpendicular to the slant, caps by w; torus: minor - w). Part.cut. A closed
          cavity: one solid with TWO shells.
  tube    cylinder minus a coaxial cylinder of radius r - w extended through both ends. One solid, one shell.
  rhs     box (long axis z) minus a box of half (hx - w, hy - w) extended through both ends along z.
  frame   the parent tessellated coarsely (MeshPart.meshFromShape, LinearDeflection = tol, AngularDeflection raised with
          tol until the unique mesh edges are under the cap - Shape.tessellate(tol) alone keeps OCC's fine angular
          deflection and never gets below ~70 edges on a sphere, see the report), one strut per unique edge of the
          sample's member kind (rod, bar, tube, rhs, ibeam), optional node spheres, fused with multiFuse.
The wall w is `wall_frac` of the body's thickness (box: smallest edge; sphere: diameter; cylinder: min(diameter, height);
cone: min(larger diameter, height); torus: tube diameter), raised to `min_wall_mm` in the PLACED frame (so a wall is at
least that many mm after scaling; the host passes 1.5 voxels).
"""
import json
import math
import os
import sys
import time

import FreeCAD as App
import Part

V = App.Vector
ORIGIN = V(0, 0, 0)
Z = V(0, 0, 1)
HERE = os.environ.get("SPECIAL_SRC_DIR") or (os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:                                                  # the I-beam agent's section builder (DIN polygon + fillets)
    from ibeam_freecad import make_ibeam_solid as _external_ibeam
except Exception:                                     # pragma: no cover - fallback below
    _external_ibeam = None


# --------------------------------------------------------------------------------------------------- parents

def make_parent(kind, d):
    """The parent solid in its centred local frame (dims in mm)."""
    if kind == "box":
        hx, hy, hz = d["half"]
        return Part.makeBox(2 * hx, 2 * hy, 2 * hz, V(-hx, -hy, -hz))
    if kind == "sphere":
        return Part.makeSphere(d["radius"])
    if kind == "cylinder":
        return Part.makeCylinder(d["radius"], d["height"], V(0, 0, -d["height"] / 2), Z)
    if kind == "cone":
        return Part.makeCone(d["radius1"], d["radius2"], d["height"], V(0, 0, -d["height"] / 2), Z)
    if kind == "torus":
        return Part.makeTorus(d["major"], d["minor"])
    raise ValueError("unknown parent " + kind)


def thickness(kind, d):
    if kind == "box":
        return 2 * min(d["half"])
    if kind == "sphere":
        return 2 * d["radius"]
    if kind == "cylinder":
        return min(2 * d["radius"], d["height"])
    if kind == "cone":
        return min(2 * max(d["radius1"], d["radius2"]), d["height"])
    if kind == "torus":
        return 2 * d["minor"]
    raise ValueError(kind)


def inner_body(kind, d, w, through=False):
    """The cutter of hollow / tube / rhs: the parent shrunk by the wall w (through=True: extended through both z ends)."""
    if kind == "box":
        hx, hy, hz = [h - w for h in d["half"]]
        if through:
            hz = d["half"][2] + max(1.0, w)
        if min(hx, hy, hz) <= 0:
            raise ValueError("wall eats the box")
        return Part.makeBox(2 * hx, 2 * hy, 2 * hz, V(-hx, -hy, -hz))
    if kind == "sphere":
        if d["radius"] - w <= 0:
            raise ValueError("wall eats the sphere")
        return Part.makeSphere(d["radius"] - w)
    if kind == "cylinder":
        r = d["radius"] - w
        h = d["height"] + 2 * max(1.0, w) if through else d["height"] - 2 * w
        if r <= 0 or h <= 0:
            raise ValueError("wall eats the cylinder")
        return Part.makeCylinder(r, h, V(0, 0, -h / 2), Z)
    if kind == "cone":
        r1, r2, h = d["radius1"], d["radius2"], d["height"]
        slope = (r2 - r1) / h
        radial = w * math.sqrt(1.0 + slope * slope)          # perpendicular wall w on the slanted flank
        ri1 = r1 + slope * w - radial
        ri2 = r1 + slope * (h - w) - radial
        hi = h - 2 * w
        if hi <= 0 or max(ri1, ri2) <= 0:
            raise ValueError("wall eats the cone")
        if min(ri1, ri2) < 0:                                 # the inner flank reaches the axis before the cap: stop there
            zc = (0 - ri1) / (ri2 - ri1) * hi
            if ri1 < 0:
                return Part.makeCone(0.0, ri2, hi - zc, V(0, 0, -h / 2 + w + zc), Z)
            return Part.makeCone(ri1, 0.0, zc, V(0, 0, -h / 2 + w), Z)
        return Part.makeCone(ri1, ri2, hi, V(0, 0, -h / 2 + w), Z)
    if kind == "torus":
        if d["minor"] - w <= 0:
            raise ValueError("wall eats the torus")
        return Part.makeTorus(d["major"], d["minor"] - w)
    raise ValueError(kind)


# --------------------------------------------------------------------------------------------------- placement

def rot_matrix(R, t=(0.0, 0.0, 0.0)):
    return App.Matrix(R[0][0], R[0][1], R[0][2], t[0], R[1][0], R[1][1], R[1][2], t[1], R[2][0], R[2][1], R[2][2], t[2],
                      0, 0, 0, 1)


def rotated(shape, R):
    out = shape.copy()
    out.transformShape(rot_matrix(R))
    return out


def placement_of(bb, pl):
    """k, b, c (see module docstring) from the ROTATED shape's bounding box and the host's draw."""
    ext = [bb.XLength, bb.YLength, bb.ZLength]
    largest = max(ext)
    g = float(pl["grid_mm"])
    k = float(pl["size_fraction"]) * g / largest
    ek = [e * k for e in ext]
    c = [(2 * u - 1) * (g / 2 - e / 2) for u, e in zip(pl["pos_unit"], ek)]
    b = [bb.Center.x, bb.Center.y, bb.Center.z]
    return k, b, c, ek


def place(shape, k, b, c):
    out = shape.copy()
    out.scale(k, V(*b))
    out.translate(V(c[0] - b[0], c[1] - b[1], c[2] - b[2]))
    return out


def to_world(p, R, k, b, c):
    x = [sum(R[i][j] * p[j] for j in range(3)) for i in range(3)]
    return [k * (x[i] - b[i]) + c[i] for i in range(3)]


def shell_count(shape):
    return sum(len(s.Shells) for s in shape.Solids)


# --------------------------------------------------------------------------------------------------- frame

def coarse_mesh(parent, size, tol_frac, angular, max_edges, grow=1.25, max_iter=12):
    """Unique vertices and edges of a coarse tessellation of `parent`: MeshPart with LinearDeflection = tol_frac*size and
    AngularDeflection `angular`, both raised by `grow` until the unique edge count is <= max_edges. Returns
    (verts, edges, tol_frac, angular, iterations); raises if the cap is not reached."""
    import MeshPart
    best = None
    for it in range(max_iter):
        mesh = MeshPart.meshFromShape(Shape=parent.copy(), LinearDeflection=tol_frac * size,
                                      AngularDeflection=min(angular, 3.1), Relative=False)
        pts, tris = mesh.Topology
        key, idx, verts = {}, [], []
        q = 1e-5 * size
        for p in pts:
            kk = (round(p[0] / q), round(p[1] / q), round(p[2] / q))
            if kk not in key:
                key[kk] = len(verts)
                verts.append([float(p[0]), float(p[1]), float(p[2])])
            idx.append(key[kk])
        edges = set()
        for t in tris:
            a, bq, cq = idx[t[0]], idx[t[1]], idx[t[2]]
            for u, v in ((a, bq), (bq, cq), (cq, a)):
                if u != v:
                    edges.add((min(u, v), max(u, v)))
        best = (verts, sorted(edges), tol_frac, angular, it + 1)
        if len(edges) <= max_edges:
            return best
        tol_frac *= grow
        angular *= grow
    raise RuntimeError("tessellation cap %d not reached (%d edges at tol %.3f, angular %.2f)"
                       % (max_edges, len(best[1]), best[2], best[3]))


def torus_grid(parent, size, tol_frac, max_edges):
    """Coarse torus tessellation on FreeCAD's own surface (Face.valueAt on an n_u x n_v parameter grid, each quad split
    into two triangles: 3 n_u n_v unique edges). OCC's mesher cannot go below 25 vertices (75 edges) on a torus, so the
    torus uses this grid: n per direction from the chord sagitta <= tol (n >= pi / acos(1 - tol / r)), at least 5 x 3,
    then shrunk until 3 n_u n_v <= max_edges."""
    face = parent.Faces[0]
    u0, u1, v0, v1 = face.ParameterRange
    surf = face.Surface
    R, r = surf.MajorRadius, surf.MinorRadius
    tol = tol_frac * size

    def n_for(radius):
        ratio = max(-1.0, min(1.0, 1.0 - tol / radius))
        return max(3, int(math.ceil(math.pi / max(math.acos(ratio), 1e-6))))
    nu, nv = max(5, n_for(R + r)), max(3, n_for(r))
    while 3 * nu * nv > max_edges and (nu > 5 or nv > 3):
        if nu > 5 and (nu >= nv * 1.5 or nv <= 3):
            nu -= 1
        else:
            nv -= 1
    verts = []
    for i in range(nu):
        for j in range(nv):
            p = face.valueAt(u0 + (u1 - u0) * i / nu, v0 + (v1 - v0) * j / nv)
            verts.append([p.x, p.y, p.z])
    edges = set()
    for i in range(nu):
        for j in range(nv):
            a, b = i * nv + j, ((i + 1) % nu) * nv + j
            c, d = i * nv + (j + 1) % nv, ((i + 1) % nu) * nv + (j + 1) % nv
            for e in ((a, b), (a, c), (a, d)):
                edges.add((min(e), max(e)))
    return verts, sorted(edges), nu, nv


def frame_basis(p, q, centre):
    """(e1, e2, d): d along the strut, e2 = the outward direction (strut midpoint - part centre) made perpendicular to d
    (the section's 'height' / an I-beam's web plane points outwards, like a real frame), e1 = e2 x d."""
    d = V(q[0] - p[0], q[1] - p[1], q[2] - p[2])
    d.normalize()
    m = V((p[0] + q[0]) / 2 - centre[0], (p[1] + q[1]) / 2 - centre[1], (p[2] + q[2]) / 2 - centre[2])
    e2 = m - d * m.dot(d)
    if e2.Length < 1e-9:
        e2 = V(1, 0, 0) if abs(d.x) < 0.9 else V(0, 1, 0)
        e2 = e2 - d * e2.dot(d)
    e2.normalize()
    e1 = e2.cross(d)
    e1.normalize()
    return e1, e2, d


def local_ibeam(dims, length):
    """Minimal I-section (12-corner polygon, no fillets), axis z centred, h along y, b along x - used only when
    ibeam_freecad.make_ibeam_solid is not importable."""
    h, b, tw, tf = dims["h"], dims["b"], dims["tw"], dims["tf"]
    xy = [(-b / 2, -h / 2), (b / 2, -h / 2), (b / 2, -h / 2 + tf), (tw / 2, -h / 2 + tf), (tw / 2, h / 2 - tf),
          (b / 2, h / 2 - tf), (b / 2, h / 2), (-b / 2, h / 2), (-b / 2, h / 2 - tf), (-tw / 2, h / 2 - tf),
          (-tw / 2, -h / 2 + tf), (-b / 2, -h / 2 + tf)]
    pts = [V(x, y, -length / 2) for x, y in xy]
    return Part.Face(Part.makePolygon(pts + [pts[0]])).extrude(V(0, 0, length))


def section_solid(kind, sec, length):
    """One member in ITS local frame: axis z centred (-length/2 .. length/2), section 'height' along y."""
    L = length
    if kind == "rod":
        r = sec["d"] / 2
        return Part.makeCylinder(r, L, V(0, 0, -L / 2), Z)
    if kind == "tube":
        r, ri = sec["d"] / 2, sec["d"] / 2 - sec["wall"]
        return Part.makeCylinder(r, L, V(0, 0, -L / 2), Z).cut(Part.makeCylinder(ri, L + 2, V(0, 0, -L / 2 - 1), Z))
    if kind == "bar":
        w, h = sec["w"], sec["h"]
        return Part.makeBox(w, h, L, V(-w / 2, -h / 2, -L / 2))
    if kind == "rhs":
        w, h, t = sec["w"], sec["h"], sec["wall"]
        outer = Part.makeBox(w, h, L, V(-w / 2, -h / 2, -L / 2))
        return outer.cut(Part.makeBox(w - 2 * t, h - 2 * t, L + 2, V(-w / 2 + t, -h / 2 + t, -L / 2 - 1)))
    if kind == "ibeam":
        if _external_ibeam is not None:
            return _external_ibeam(sec, L, fillet=bool(sec.get("fillet", True)))
        return local_ibeam(sec, L)
    raise ValueError("unknown member kind " + kind)


def strut(kind, sec, p, q, centre, extend):
    e1, e2, d = frame_basis(p, q, centre)
    L = math.dist(p, q) + 2 * extend
    mid = V((p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2)
    solid = section_solid(kind, sec, L)
    m = App.Matrix(e1.x, e2.x, d.x, mid.x, e1.y, e2.y, d.y, mid.y, e1.z, e2.z, d.z, mid.z, 0, 0, 0, 1)
    solid.transformShape(m)
    return solid


def section_dims(kind, t, fr, scale_ibeam, min_wall=0.0):
    """Section dims of a member of nominal size t: rod/tube diameter t; bar/rhs t x t*aspect; ibeam h = t, the DIN
    profile's b, tw, tf, r scaled by t / h_din."""
    if kind == "rod":
        return dict(d=t)
    if kind == "tube":
        return dict(d=t, wall=min(max(fr["member_wall_frac"] * t, 0.12 * t, min_wall), 0.35 * t))
    if kind == "bar":
        return dict(w=t, h=t * fr["aspect"])
    if kind == "rhs":
        return dict(w=t, h=t * fr["aspect"], wall=min(max(fr["member_wall_frac"] * t, 0.12 * t, min_wall), 0.35 * t))
    if kind == "ibeam":
        p = fr["ibeam_profile"]
        s = t / p["h"]
        out = dict(name=p.get("name"), h=t, b=p["b"] * s, tw=p["tw"] * s, tf=p["tf"] * s, r=p.get("r", 0.0) * s,
                   flange_slope=p.get("flange_slope", 0.0), fillet=bool(fr.get("fillet", True)), scale=s)
        return out
    raise ValueError(kind)


def build_frame(kind, d, fr, attempt, k_hint=None):
    """The fused frame in the parent's local frame + its description. `attempt` > 0 perturbs the build (retries):
      1  I-beam members without fillets (the filleted DIN section is the usual cause of an invalid fuse); other kinds:
         strut extensions jittered by +-15 % (breaks coincident end faces at the joints);
      2  as 1, plus jitter, plus the node spheres toggled;
      3  as 2, plus a fuzzy fuse (1e-4 of the size), struts 8 % thicker and the tessellation 15 % coarser."""
    import random
    parent = make_parent(kind, d)
    bb = parent.BoundBox
    size = max(bb.XLength, bb.YLength, bb.ZLength)
    centre = [bb.Center.x, bb.Center.y, bb.Center.z]
    tol, ang = fr["tess_tol_frac"], fr["tess_angular"]
    if attempt >= 3:
        tol, ang = tol * 1.15, ang * 1.15
    grid = None
    if kind == "torus":
        verts, edges, nu, nv = torus_grid(parent, size, tol, fr["max_edges"])
        iters, grid = 1, [nu, nv]
    else:
        verts, edges, tol, ang, iters = coarse_mesh(parent, size, tol, ang, fr["max_edges"])
    mk = fr["member_kind"]
    # thickness: >= thickness_frac of the size AND >= min_strut_voxels in the placed grid (k_hint = the placement scale of
    # the parent; the struts widen the bounding box, so the real k is up to ~10 % smaller - hence the 1.12 margin)
    vox_local = fr.get("voxel_mm", 0.0) / k_hint if k_hint else 0.0
    t_min_vox = fr.get("min_strut_voxels", {}).get(mk, 0.0) * vox_local * 1.12        # 1.12: struts widen the bbox
    t = max(fr["thickness_frac"] * size, t_min_vox) * (1.08 if attempt >= 3 else 1.0)
    fr = dict(fr)
    if attempt >= 1 and mk == "ibeam":
        fr["fillet"] = False
    sec = section_dims(mk, t, fr, True, min_wall=fr.get("min_member_wall_voxels", 0.0) * vox_local)
    nodes = bool(fr["node_spheres"]) != (attempt >= 2)
    jitter = random.Random(1000 + attempt) if (attempt >= 2 or (attempt == 1 and mk != "ibeam")) else None
    # joint model (measured on a 42-member sphere frame, see special_generator's docstring):
    #   rod, bar, ibeam  the member solids overlap at the joints (extension 0.5 t; I-beams 0.25 t) and are fused; the small
    #                    pockets that interpenetrating sections seal off are filled (outer shell only);
    #   tube, rhs        "split bore": the OUTER bodies (rod / bar) and node spheres are fused, then every member's bore,
    #                    extended through the joint, is cut in one multi-tool cut - the bores meet at the joints like a
    #                    coped welded node. Fusing 42 already-hollow members took 2-5x longer, was invalid in 1 of 3 cases
    #                    and left 60-300 sealed pockets.
    split = mk in ("tube", "rhs")
    solid_kind = {"tube": "rod", "rhs": "bar"}.get(mk, mk)
    outer_sec = dict(d=sec["d"]) if mk == "tube" else (dict(w=sec["w"], h=sec["h"]) if mk == "rhs" else sec)
    ext_frac = 0.25 if mk == "ibeam" else 0.5
    parts, bores, members = [], [], []
    for a, b2 in edges:
        p, q = verts[a], verts[b2]
        if math.dist(p, q) < 1e-6 * size:
            continue
        extend = ext_frac * t * (jitter.uniform(0.85, 1.15) if jitter else 1.0)
        parts.append(strut(solid_kind, outer_sec, p, q, centre, extend))
        if split:
            bore = dict(d=sec["d"] - 2 * sec["wall"]) if mk == "tube" else \
                dict(w=sec["w"] - 2 * sec["wall"], h=sec["h"] - 2 * sec["wall"])
            bores.append(strut(solid_kind, bore, p, q, centre, extend + 0.02 * t))
        members.append(dict(p0=p, p1=q, kind=mk, section=sec, extension=extend))
    if nodes:
        rn = 0.5 * t * fr["node_scale"]
        used = sorted({i for e in edges for i in e})
        for i in used:
            parts.append(Part.makeSphere(rn, V(*verts[i])))
    fuzzy = 1e-4 * size if attempt >= 3 else 0.0
    fused = parts[0].multiFuse(parts[1:], fuzzy) if fuzzy > 0 else parts[0].multiFuse(parts[1:])
    filled = 0
    if split:
        fused = fused.cut(bores, fuzzy) if fuzzy > 0 else fused.cut(bores)
    elif len(fused.Solids) == 1 and len(fused.Solids[0].Shells) > 1:
        shells = fused.Solids[0].Shells
        filled = len(shells) - 1
        fused = Part.Solid(max(shells, key=lambda sh: abs(Part.Solid(sh).Volume)))
    info = dict(member_kind=mk, tessellation_tol=tol * size, tessellation_tol_frac=tol, tessellation_angular=ang,
                tessellation_iterations=iters, tessellation_method="torus_uv_grid" if grid else "meshpart",
                tessellation_grid=grid, thickness_min_voxel_mm=t_min_vox, n_vertices=len({i for e in edges for i in e}), n_members=len(members),
                members=members, thickness_mm=t, thickness_frac=t / size, node_spheres=nodes,
                node_radius=0.5 * t * fr["node_scale"] if nodes else 0.0, fillet=bool(fr.get("fillet")) if mk == "ibeam" else None,
                build_attempt=attempt, joint_model="split_bore" if split else "overlap", pockets_filled=filled,
                ibeam_builder=("ibeam_freecad" if _external_ibeam is not None else "local") if mk == "ibeam" else None)
    return fused, info


# --------------------------------------------------------------------------------------------------- one job

def build_local(job, attempt, k_hint=None):
    kind, sub, d = job["parent"], job["subclass"], job["dims"]
    if sub == "frame":
        return build_frame(kind, d, job["frame"], attempt, k_hint)
    outer = make_parent(kind, d)
    thick = thickness(kind, d)
    w = job["wall_frac"] * thick
    if k_hint:
        w = max(w, job.get("min_wall_mm", 0.0) / k_hint)
    if sub == "hollow":
        inner = inner_body(kind, d, w)
    elif sub == "tube":
        if kind != "cylinder":
            raise ValueError("tube is a cylinder subclass")
        inner = inner_body(kind, d, w, through=True)
    elif sub == "rhs":
        if kind != "box":
            raise ValueError("rhs is a box subclass")
        inner = inner_body(kind, d, w, through=True)
    else:
        raise ValueError("unknown subclass " + sub)
    info = dict(wall_mm_local=w, wall_frac_effective=w / thick, thickness_mm_local=thick,
                inner_volume_local=inner.Volume)
    return outer.cut(inner), info


def check(shape, sub):
    """(ok, reason): one valid solid; hollow has a closed cavity (2 shells), tube/rhs/frame are open (hollow members
    may enclose voids at joints, so a frame only needs one solid)."""
    if not shape.isValid():
        return False, "invalid shape"
    n = len(shape.Solids)
    if n != 1:
        return False, "%d solids" % n
    if shape.Volume <= 0:
        return False, "non-positive volume"
    shells = shell_count(shape)
    if sub == "hollow" and shells != 2:
        return False, "hollow with %d shells" % shells
    if sub in ("tube", "rhs") and shells != 1:
        return False, "%s with %d shells" % (sub, shells)
    return True, "ok"


def build_job(job):
    t0 = time.time()
    R = job["R"]
    sub = job["subclass"]
    attempts = []
    shape = info = None
    # the placement scale k depends only on the outer body for hollow / tube / rhs (the cut keeps its bounding box), so it is
    # known before the cut and min_wall_mm can be enforced in the placed frame; for a frame it is the parent's k (the
    # struts stick out a little, so the real k is a few % smaller - build_frame adds a margin)
    bb0 = rotated(make_parent(job["parent"], job["dims"]), R).BoundBox
    k_hint = placement_of(bb0, job["placement"])[0]
    for attempt in range(int(job.get("max_attempts", 4))):
        try:
            shape, info = build_local(job, attempt, k_hint)
            ok, why = check(shape, sub)
        except Exception as exc:
            ok, why = False, "%s: %s" % (type(exc).__name__, exc)
        attempts.append(why)
        if ok:
            break
        if sub != "frame":                  # hollow / tube / rhs are deterministic: a retry would repeat the failure
            break
    else:
        ok = False
    if not ok:
        return dict(id=job.get("id"), ok=False, error=attempts[-1], attempts=attempts, seconds_build=time.time() - t0)
    volume_local = shape.Volume
    rot = rotated(shape, R)
    k, b, c, ek = placement_of(rot.BoundBox, job["placement"])
    world = place(rot, k, b, c)
    t_build = time.time() - t0
    ok_w, why_w = check(world, sub)
    import MeshPart
    t1 = time.time()
    # linear deflection as every generator (0.05 mm); the angular deflection 0.1 of freecad_builder is relaxed for frames
    # (job["angular_deflection"]): on a 1-2 mm strut it asks for 60 segments per circle where the 0.05 mm chord bound
    # needs about 8, and made a 50-member frame's STL 190k facets / 2 min of meshing. A mesh with holes (tiny faces where
    # many members meet) is re-meshed finer; still open -> the sample fails (the host redraws it): the field's inside /
    # outside needs a watertight STL.
    base = float(job.get("deflection", 0.05))
    for defl in (base, 0.7 * base, 0.5 * base):
        mesh = MeshPart.meshFromShape(Shape=world, LinearDeflection=defl,
                                      AngularDeflection=float(job.get("angular_deflection", 0.1)), Relative=False)
        if mesh.isSolid():
            break
    if not mesh.isSolid():
        ok_w, why_w = False, "open STL mesh after re-meshing at %.3f mm" % defl
    mesh.write(job["stl_out"])
    wb = world.BoundBox
    out = dict(id=job.get("id"), ok=bool(ok_w), error=None if ok_w else why_w, attempts=attempts,
               n_attempts=len(attempts), valid=bool(world.isValid()), n_solids=len(world.Solids),
               n_shells=shell_count(world), volume_mm3=world.Volume, volume_local_mm3=volume_local,
               n_facets=int(mesh.CountFacets), mesh_closed=bool(mesh.isSolid()), export_deflection=defl,
               bbox=[[wb.XMin, wb.YMin, wb.ZMin], [wb.XMax, wb.YMax, wb.ZMax]],
               scale_factor=k, bbox_center_rotated=b, center_mm=c, extent_mm=ek,
               parent_center_mm=to_world([0.0, 0.0, 0.0], R, k, b, c),
               seconds_build=t_build, seconds_export=time.time() - t1)
    if sub == "frame":
        info = dict(info)
        info["members"] = [dict(p0=to_world(m["p0"], R, k, b, c), p1=to_world(m["p1"], R, k, b, c), kind=m["kind"],
                                section={kk: (vv * k if isinstance(vv, float) and kk not in ("flange_slope", "scale")
                                              else vv) for kk, vv in m["section"].items()},
                                extension=m["extension"] * k)
                           for m in info["members"]]
        for key in ("tessellation_tol", "thickness_mm", "node_radius", "thickness_min_voxel_mm"):
            info[key] = info[key] * k
    else:
        info = dict(info, wall_mm=info["wall_mm_local"] * k)
    out["info"] = info
    return out


def run_jobs(job_path, result_path):
    with open(job_path, encoding="utf-8") as fh:
        jobs = json.load(fh)
    results = []
    for job in jobs:
        try:
            results.append(build_job(job))
        except Exception as exc:                          # reported per job; the host decides
            results.append(dict(id=job.get("id"), ok=False, error="%s: %s" % (type(exc).__name__, exc)))
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh)


if os.environ.get("SPECIAL_JOB_JSON") and os.environ.get("SPECIAL_RESULT_JSON"):
    run_jobs(os.environ["SPECIAL_JOB_JSON"], os.environ["SPECIAL_RESULT_JSON"])
