"""SHaPe ReCo logo, realistic 3D version (Blender / Cycles).

The sea, the crow's nest (Ausguck: a stave barrel with iron hoops on the mast top), a brass spyglass with a leather
grip, and a poison dart frog sitting in the barrel: its head peeks over the rim with the left eye at the eyepiece and
the right eye on the viewer, one hand on the rim, one holding the spyglass. Skin: blue with dark spots; the fingers
shade into orange and end in pink toe pads.

Run (headless):
  blender -b -P packaging/logo/blender_logo.py -- --out logo3d.png --res 1024 --samples 256 --device GPU
  [--blend scene.blend]   also save the scene for opening in the Blender GUI
  [--no-render]           build the scene and write <out>.json only
Next to the render it writes <out>.json: the badge disc (smallest circle around the glass's objective end, the toe
pads and the eyes) and keep-out circles (glass, frog, eyes, mast) for overlays; make_logo.py --3d reads both.
The frog is one metaball family (head, body, arms, palms, fingers, toe pads fuse into one skin).
"""
import argparse
import json
import math
import os
import random
import sys
import time

import bmesh
import bpy
from mathutils import Matrix, Vector, kdtree
from bpy_extras.object_utils import world_to_camera_view
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/shape_reco_logo3d.png")
ap.add_argument("--res", type=int, default=1024)
ap.add_argument("--samples", type=int, default=256)
ap.add_argument("--device", default="GPU", choices=("GPU", "CPU"))
ap.add_argument("--blend", default="")
ap.add_argument("--no-render", action="store_true", help="build the scene, write <out>.json, skip the render")
args = ap.parse_args(argv)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
COLL = scene.collection


# ---------------------------------------------------------------- helpers
def lin(hexstr, a=1.0):
    h = hexstr.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92 for x in c) + (a,)


def link(ob):
    COLL.objects.link(ob)
    return ob


def mesh_object(name, bm, mats, smooth=True, edge_split=None):
    me = bpy.data.meshes.new(name)
    if smooth:
        for fc in bm.faces:
            fc.smooth = True
    bm.to_mesh(me)
    bm.free()
    for m in mats:
        me.materials.append(m)
    ob = link(bpy.data.objects.new(name, me))
    if edge_split is not None:
        mod = ob.modifiers.new("split", "EDGE_SPLIT")
        mod.split_angle = math.radians(edge_split)
    return ob


def lathe(name, profile, mats, segs=96, closed=False, edge_split=32):
    """Revolve a (r, z, material_index_of_next_segment) polyline around the local Z axis."""
    bm = bmesh.new()
    rings = []
    for r, z, _ in profile:
        rings.append([bm.verts.new((r * math.cos(2 * math.pi * i / segs), r * math.sin(2 * math.pi * i / segs), z))
                      for i in range(segs)])
    n = len(profile)
    for j in range(n if closed else n - 1):
        a, b = rings[j], rings[(j + 1) % n]
        for i in range(segs):
            i2 = (i + 1) % segs
            fc = bm.faces.new((a[i], a[i2], b[i2], b[i]))
            fc.material_index = profile[j][2]
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return mesh_object(name, bm, mats, edge_split=edge_split)


def ellipsoids(name, items, mat, u=40, v=20):
    """items: (centre, radii(x,y,z), z_axis_direction) - one mesh of UV spheres."""
    bm = bmesh.new()
    for c, radii, zdir in items:
        q = Vector((0, 0, 1)).rotation_difference(Vector(zdir).normalized())
        mtx = Matrix.Translation(Vector(c)) @ q.to_matrix().to_4x4() @ Matrix.Diagonal((*radii, 1.0))
        bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=1.0, matrix=mtx)
    return mesh_object(name, bm, [mat])


def catmull(ctrl, per=12):
    p = [Vector(c) for c in ctrl]
    ext = [p[0] + (p[0] - p[1])] + p + [p[-1] + (p[-1] - p[-2])]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for k in range(per):
            t = k / per
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t * t * t))
    out.append(p[-1])
    return out


def tube(name, pts, radii, parts, mats):
    """A bevelled poly curve through pts; parts = [(i0, i1, material_index)] split it into coloured spans."""
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 1.0
    cu.bevel_resolution = 6
    cu.use_fill_caps = True
    for m in mats:
        cu.materials.append(m)
    for i0, i1, mi in parts:
        sp = cu.splines.new("POLY")
        sp.points.add(i1 - i0)
        for j, idx in enumerate(range(i0, i1 + 1)):
            q = pts[idx]
            sp.points[j].co = (q.x, q.y, q.z, 1.0)
            sp.points[j].radius = radii[idx]
        sp.material_index = mi
        sp.use_smooth = True
    return link(bpy.data.objects.new(name, cu))


def principled(name, color, rough=0.5, metal=0.0, **inputs):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = color
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    for k, v in inputs.items():
        b.inputs[k].default_value = v
    return m


def ramp(nt, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    el = r.color_ramp.elements
    el[0].position, el[0].color = stops[0]
    el[1].position, el[1].color = stops[-1]
    for pos, col in stops[1:-1]:
        e = el.new(pos)
        e.color = col
    return r


def mix_rgb(nt, fac, a, b, blend="MIX"):
    """A colour Mix node; sockets picked by type (ShaderNodeMix carries float/vector/colour sockets of one name)."""
    mx = nt.nodes.new("ShaderNodeMix")
    mx.data_type = "RGBA"
    mx.blend_type = blend
    sa = [x for x in mx.inputs if x.name == "A" and x.type == "RGBA"][0]
    sb = [x for x in mx.inputs if x.name == "B" and x.type == "RGBA"][0]
    for sock, val in ((mx.inputs["Factor"], fac), (sa, a), (sb, b)):
        if isinstance(val, bpy.types.NodeSocket):
            nt.links.new(val, sock)
        else:
            sock.default_value = val
    return [x for x in mx.outputs if x.name == "Result" and x.type == "RGBA"][0]


def math_node(nt, op, a, b=None):
    n = nt.nodes.new("ShaderNodeMath")
    n.operation = op
    for sock, val in zip(n.inputs, (a, b)):
        if isinstance(val, bpy.types.NodeSocket):
            nt.links.new(val, sock)
        elif val is not None:
            sock.default_value = val
    return n.outputs[0]


def bump(nt, height_socket, strength, distance=0.02):
    bn = nt.nodes.new("ShaderNodeBump")
    bn.inputs["Strength"].default_value = strength
    bn.inputs["Distance"].default_value = distance
    nt.links.new(height_socket, bn.inputs["Height"])
    nt.links.new(bn.outputs["Normal"], nt.nodes["Principled BSDF"].inputs["Normal"])
    return bn


# ---------------------------------------------------------------- materials
def skin(name, base, spots=False, gradient=False):
    m = principled(name, lin(base), rough=0.34, **{"Coat Weight": 1.0, "Coat Roughness": 0.05,
                                                   "Subsurface Weight": 0.12, "Subsurface Scale": 0.004})
    nt, L = m.node_tree, m.node_tree.links
    b = nt.nodes["Principled BSDF"]
    b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.3) if not spots else (0.3, 0.5, 1.0)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    fine = nt.nodes.new("ShaderNodeTexNoise")
    fine.inputs["Scale"].default_value = 900.0
    fine.inputs["Detail"].default_value = 4.0
    L.new(tc.outputs["Object"], fine.inputs["Vector"])
    bump(nt, fine.outputs["Fac"], 0.08, 0.002)
    if spots:
        var = nt.nodes.new("ShaderNodeTexNoise")
        var.inputs["Scale"].default_value = 45.0
        L.new(tc.outputs["Object"], var.inputs["Vector"])
        tone = ramp(nt, [(0.35, lin("#1556D6")), (0.65, lin("#3FA0FF"))])
        L.new(var.outputs["Fac"], tone.inputs["Fac"])
        vor = nt.nodes.new("ShaderNodeTexVoronoi")
        vor.inputs["Scale"].default_value = 46.0
        L.new(tc.outputs["Object"], vor.inputs["Vector"])
        spot = ramp(nt, [(0.2, (1, 1, 1, 1)), (0.25, (0, 0, 0, 1))])
        L.new(vor.outputs["Distance"], spot.inputs["Fac"])
        col = mix_rgb(nt, spot.outputs["Color"], tone.outputs["Color"], lin("#07102A"))
        if gradient:  # along the finger (attribute t, 0 at the palm, 1 at the tip): blue -> orange -> pink
            at = nt.nodes.new("ShaderNodeAttribute")
            at.attribute_name = "t"

            def smooth(lo, hi):
                m_r = nt.nodes.new("ShaderNodeMapRange")
                m_r.interpolation_type = "SMOOTHSTEP"
                m_r.inputs["From Min"].default_value = lo
                m_r.inputs["From Max"].default_value = hi
                L.new(at.outputs["Fac"], m_r.inputs["Value"])
                return m_r.outputs["Result"]
            col = mix_rgb(nt, smooth(0.70, 0.80), col, lin("#FF7A12"))
            col = mix_rgb(nt, smooth(0.86, 0.97), col, lin("#FF4FA3"))
        L.new(col, b.inputs["Base Color"])
    return m


FROG_SKIN = skin("frog_skin", "#2A7FEA", spots=True, gradient=True)  # blue, spots; fingers -> pads


def wood_material(name, light, mid, dark, stretch=(70, 70, 1.6)):
    m = principled(name, lin(mid), rough=0.68, **{"Specular IOR Level": 0.35})
    nt, L = m.node_tree, m.node_tree.links
    b = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = stretch
    L.new(tc.outputs["Object"], mp.inputs["Vector"])
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    add = nt.nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    L.new(mp.outputs["Vector"], add.inputs[0])
    L.new(oi.outputs["Random"], add.inputs[1])
    grain = nt.nodes.new("ShaderNodeTexNoise")
    grain.inputs["Scale"].default_value = 1.0
    grain.inputs["Detail"].default_value = 8.0
    grain.inputs["Distortion"].default_value = 0.6
    L.new(add.outputs["Vector"], grain.inputs["Vector"])
    col = ramp(nt, [(0.3, lin(dark)), (0.5, lin(mid)), (0.72, lin(light))])
    L.new(grain.outputs["Fac"], col.inputs["Fac"])
    shade = ramp(nt, [(0.0, (0.78, 0.74, 0.7, 1)), (1.0, (1.0, 0.98, 0.96, 1))])
    L.new(oi.outputs["Random"], shade.inputs["Fac"])
    L.new(mix_rgb(nt, 1.0, col.outputs["Color"], shade.outputs["Color"], "MULTIPLY"), b.inputs["Base Color"])
    bump(nt, grain.outputs["Fac"], 0.25, 0.004)
    return m


WOOD = wood_material("wood", "#B68556", "#8A5A35", "#4E3019")
MAST_WOOD = wood_material("mast_wood", "#C39260", "#956239", "#5A371C", stretch=(55, 55, 0.8))

IRON = principled("iron", lin("#1C1D22"), rough=0.5, metal=0.85)
_nt = IRON.node_tree
_n = _nt.nodes.new("ShaderNodeTexNoise")
_n.inputs["Scale"].default_value = 30.0
_r = ramp(_nt, [(0.45, lin("#1C1D22")), (0.7, lin("#5B3720"))])
_nt.links.new(_n.outputs["Fac"], _r.inputs["Fac"])
_nt.links.new(_r.outputs["Color"], _nt.nodes["Principled BSDF"].inputs["Base Color"])
RIVET = principled("rivet", lin("#C9A15A"), rough=0.35, metal=1.0)

BRASS = principled("brass", lin("#E8BC66"), rough=0.24, metal=1.0)
_nt = BRASS.node_tree
_n = _nt.nodes.new("ShaderNodeTexNoise")
_n.inputs["Scale"].default_value = 60.0
_r = ramp(_nt, [(0.3, (0.12, 0.12, 0.12, 1)), (0.75, (0.34, 0.34, 0.34, 1))])
_nt.links.new(_n.outputs["Fac"], _r.inputs["Fac"])
_nt.links.new(_r.outputs["Color"], _nt.nodes["Principled BSDF"].inputs["Roughness"])
LEATHER = principled("leather", lin("#1E3E80"), rough=0.55, **{"Sheen Weight": 0.25})
_n = LEATHER.node_tree.nodes.new("ShaderNodeTexNoise")
_n.inputs["Scale"].default_value = 500.0
_n.inputs["Detail"].default_value = 6.0
bump(LEATHER.node_tree, _n.outputs["Fac"], 0.35, 0.001)
RUBBER = principled("rubber", lin("#111216"), rough=0.6)
LENS = principled("lens", (0.004, 0.006, 0.012, 1), rough=0.02, **{"Thin Film Thickness": 380.0,
                                                                   "Thin Film IOR": 1.38})
TUBE_INSIDE = principled("tube_inside", (0.01, 0.01, 0.01, 1), rough=0.8)
MOUTH = principled("mouth", lin("#0A1230"), rough=0.45)

# a frog's eye: wet black with a faint bronze iris on the front cap and a horizontal pupil (local +Z = gaze)
EYE = principled("eye", lin("#040508"), rough=0.1, **{"Coat Weight": 1.0, "Coat Roughness": 0.0})
_nt = EYE.node_tree
_tc = _nt.nodes.new("ShaderNodeTexCoord")
_nv = _nt.nodes.new("ShaderNodeVectorMath")
_nv.operation = "NORMALIZE"
_nt.links.new(_tc.outputs["Object"], _nv.inputs[0])
_sp = _nt.nodes.new("ShaderNodeSeparateXYZ")
_nt.links.new(_nv.outputs["Vector"], _sp.inputs["Vector"])
_ir = _nt.nodes.new("ShaderNodeMapRange")
_ir.interpolation_type = "SMOOTHSTEP"
_ir.inputs["From Min"].default_value, _ir.inputs["From Max"].default_value = 0.60, 0.70
_nt.links.new(_sp.outputs["Z"], _ir.inputs["Value"])
_n = _nt.nodes.new("ShaderNodeTexNoise")
_n.inputs["Scale"].default_value = 160.0
_n.inputs["Detail"].default_value = 6.0
_nt.links.new(_tc.outputs["Object"], _n.inputs["Vector"])
_r = ramp(_nt, [(0.35, lin("#140B05")), (0.7, lin("#5A3A18"))])
_nt.links.new(_n.outputs["Fac"], _r.inputs["Fac"])
_px = math_node(_nt, "DIVIDE", _sp.outputs["X"], 0.42)
_py = math_node(_nt, "DIVIDE", _sp.outputs["Y"], 0.22)
_d = math_node(_nt, "ADD", math_node(_nt, "MULTIPLY", _px, _px), math_node(_nt, "MULTIPLY", _py, _py))
_pupil = math_node(_nt, "MULTIPLY", math_node(_nt, "LESS_THAN", _d, 1.0),
                   math_node(_nt, "GREATER_THAN", _sp.outputs["Z"], 0.0))
_col = mix_rgb(_nt, _ir.outputs["Result"], lin("#040508"), _r.outputs["Color"])
_col = mix_rgb(_nt, _pupil, _col, (0.0, 0.0, 0.0, 1.0))
_nt.links.new(_col, _nt.nodes["Principled BSDF"].inputs["Base Color"])

SEA = principled("sea", lin("#0A3A78"), rough=0.05, **{"IOR": 1.33, "Specular IOR Level": 0.28})
_nt = SEA.node_tree
_tc = _nt.nodes.new("ShaderNodeTexCoord")
_n1 = _nt.nodes.new("ShaderNodeTexNoise")
_n1.inputs["Scale"].default_value = 0.35
_n1.inputs["Detail"].default_value = 6.0
_mp = _nt.nodes.new("ShaderNodeMapping")
_mp.inputs["Scale"].default_value = (1.0, 2.2, 1.0)
_nt.links.new(_tc.outputs["Object"], _mp.inputs["Vector"])
_nt.links.new(_mp.outputs["Vector"], _n1.inputs["Vector"])
bump(_nt, _n1.outputs["Fac"], 0.9, 1.2)


# ---------------------------------------------------------------- the sea and the sky
bpy.ops.mesh.primitive_plane_add(size=40000.0, location=(0, 0, -25.0))
bpy.context.active_object.data.materials.append(SEA)

world = bpy.data.worlds.new("sky")
scene.world = world
world.use_nodes = True
wn, wl = world.node_tree.nodes, world.node_tree.links
bg = wn["Background"]
wtc = wn.new("ShaderNodeTexCoord")
sep = wn.new("ShaderNodeSeparateXYZ")
wl.new(wtc.outputs["Generated"], sep.inputs["Vector"])
mr = wn.new("ShaderNodeMapRange")
mr.inputs["From Min"].default_value = -0.02
mr.inputs["From Max"].default_value = 0.24
wl.new(sep.outputs["Z"], mr.inputs["Value"])
sky = ramp(world.node_tree, [(0.0, lin("#FFC795")), (0.07, lin("#FFA9C6")), (0.25, lin("#BFA6E6")),
                             (0.55, lin("#5E8FE6")), (1.0, lin("#20469E"))])
wl.new(mr.outputs["Result"], sky.inputs["Fac"])
wl.new(sky.outputs["Color"], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 0.9

# ---------------------------------------------------------------- the crow's nest
H, R_END, R_MID, STAVE_T = 0.62, 0.34, 0.37, 0.026
R_IN = R_END - STAVE_T
HOOPS, HOOP_HH, HOOP_T = [f * H for f in (0.1, 0.3, 0.7, 0.88)], 0.017, 0.005


def r_at(z):
    t = 2 * z / H - 1
    return R_END + (R_MID - R_END) * (1 - t * t)


N_STAVES, GAP = 18, 0.003
for i in range(N_STAVES):
    th0 = 2 * math.pi * i / N_STAVES + GAP / (2 * R_MID)
    th1 = 2 * math.pi * (i + 1) / N_STAVES - GAP / (2 * R_MID)
    bm = bmesh.new()
    rows, cols = 24, 5
    grid = [[bm.verts.new((r_at(H * a / rows) * math.cos(th0 + (th1 - th0) * c / cols),
                           r_at(H * a / rows) * math.sin(th0 + (th1 - th0) * c / cols), H * a / rows))
             for c in range(cols + 1)] for a in range(rows + 1)]
    for a in range(rows):
        for c in range(cols):
            bm.faces.new((grid[a][c], grid[a][c + 1], grid[a + 1][c + 1], grid[a + 1][c]))
    ob = mesh_object(f"stave_{i:02d}", bm, [WOOD])
    so = ob.modifiers.new("thick", "SOLIDIFY")
    so.thickness = STAVE_T
    so.offset = -1.0
    so.use_even_offset = True
    bv = ob.modifiers.new("bevel", "BEVEL")
    bv.width = 0.003
    bv.segments = 2
    bv.limit_method = "ANGLE"

rivets = []
for zc in HOOPS:
    hh, t = HOOP_HH, HOOP_T
    za, zb = zc - hh, zc + hh
    prof = [(r_at(za) - 0.004, za, 0), (r_at(za) + t - 0.0015, za, 0), (r_at(za) + t, za + 0.0015, 0),
            (r_at(zb) + t, zb - 0.0015, 0), (r_at(zb) + t - 0.0015, zb, 0), (r_at(zb) - 0.004, zb, 0)]
    lathe(f"hoop_{zc}", prof, [IRON], segs=128, closed=True)
    for k in range(14):
        ph = 2 * math.pi * (k + 0.5) / 14
        rr = r_at(zc) + t
        rivets.append(((rr * math.cos(ph), rr * math.sin(ph), zc), (0.0045, 0.0045, 0.0028),
                       (math.cos(ph), math.sin(ph), 0)))
ellipsoids("rivets", rivets, RIVET, u=12, v=8)

# the mast, tapering, through the nest; four struts under the barrel
bpy.ops.mesh.primitive_cone_add(vertices=64, radius1=0.12, radius2=0.07, depth=34.0, location=(0, 0, -25.0 + 17.0))
mast = bpy.context.active_object
mast.data.materials.append(MAST_WOOD)
bpy.ops.object.shade_smooth()
for ang in (45, 135, 225, 315):
    a = math.radians(ang)
    p0 = Vector((0.1 * math.cos(a), 0.1 * math.sin(a), -0.4))
    p1 = Vector((0.26 * math.cos(a), 0.26 * math.sin(a), 0.02))
    d = p1 - p0
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(p0 + p1) / 2)
    st = bpy.context.active_object
    st.scale = (0.05, 0.05, d.length)
    st.rotation_mode = "QUATERNION"
    st.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    st.data.materials.append(WOOD)
    bv = st.modifiers.new("bevel", "BEVEL")
    bv.width = 0.008
    bv.segments = 3
# the floor of the nest (seen only through the stave gaps)
bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=r_at(0.04) - 0.02, depth=0.04, location=(0, 0, 0.04))
bpy.context.active_object.data.materials.append(WOOD)

# ---------------------------------------------------------------- the spyglass
# sections along the local axis: (z_start, z_end, radius, material) ; 0 brass 1 leather 2 rubber 3 lens 4 inside
SECTIONS = [(0.000, 0.040, 0.021, 2), (0.040, 0.052, 0.0265, 0), (0.052, 0.165, 0.025, 0),
            (0.165, 0.182, 0.030, 0), (0.182, 0.330, 0.0335, 1), (0.330, 0.350, 0.0385, 0),
            (0.350, 0.365, 0.0405, 0), (0.365, 0.600, 0.0425, 0), (0.600, 0.612, 0.0465, 0),
            (0.612, 0.668, 0.0455, 0)]
CH = 0.0015


def spyglass_profile():
    z0, _, r0, m0 = SECTIONS[0]
    pts = [(0.0005, z0, 3), (0.006, z0, 2), (r0 - 0.004, z0, m0), (r0, z0 + 0.004, m0)]
    for i, (za, zb, r, m) in enumerate(SECTIONS):
        if i + 1 == len(SECTIONS):
            rin, depth = 0.040, 0.028
            pts += [(r, zb - CH, m), (r - CH, zb, m), (rin, zb, 4), (rin, zb - depth, 3),
                    (rin * 0.6, zb - depth + 0.002, 3), (0.0005, zb - depth + 0.003, 3)]
            break
        rn, mn = SECTIONS[i + 1][2], SECTIONS[i + 1][3]
        if rn > r:
            pts += [(r, zb, mn), (rn - CH, zb, mn), (rn, zb + CH, mn)]
        else:
            pts += [(r, zb - CH, m), (r - CH, zb, m), (rn, zb, mn)]
    return pts


# ---------------------------------------------------------------- where the frog's head is
# Local head frame: x forward (snout), y left, z up. The head peeks over the front rim, chin just above the staves.
HEAD_YAW, HEAD_PITCH = math.radians(-28.0), math.radians(8.0)
HEAD_AT = Vector((0.18 * math.cos(math.radians(-50.0)), 0.18 * math.sin(math.radians(-50.0)), H + 0.065))
HEAD_S = 1.2                                          # the head is modelled at 1/1.2 size, then scaled
HEAD_M = (Matrix.Translation(HEAD_AT) @ Matrix.Rotation(HEAD_YAW, 4, "Z") @ Matrix.Rotation(-HEAD_PITCH, 4, "Y")
          @ Matrix.Scale(HEAD_S, 4))
HF, HL, HU = ((HEAD_M.to_3x3() @ Vector(v)).normalized() for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
EYE_R, EYE_C = 0.052, Vector((0.03, 0.098, 0.07))   # eyeball radius, left eye centre (local); right one mirrored
EYE_L = HEAD_M @ EYE_C
EYE_RT = HEAD_M @ Vector((EYE_C.x, -EYE_C.y, EYE_C.z))

# the spyglass: eyecup pressed to the frog's left (far) eye
TILT, AZ = math.radians(17.0), math.radians(-16.0)
AXIS = Vector((math.cos(TILT) * math.cos(AZ), math.cos(TILT) * math.sin(AZ), math.sin(TILT)))
UP = Vector((0, 0, 1))
U_AX = (UP - AXIS * UP.dot(AXIS)).normalized()
ORIGIN = EYE_L + AXIS * (EYE_R * HEAD_S - 0.004)
glass = lathe("spyglass", spyglass_profile(), [BRASS, LEATHER, RUBBER, LENS, TUBE_INSIDE], segs=128)
glass.rotation_mode = "QUATERNION"
glass.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(AXIS)
glass.location = ORIGIN

# ---------------------------------------------------------------- camera
TARGET = Vector((0.30, -0.12, H + 0.16))
CAM_POS = TARGET + Vector((0.469 * 3.1, -0.883 * 3.1, -0.18))   # eye just below the rim: no view into the barrel
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 85.0
cam = link(bpy.data.objects.new("cam", cam_data))
cam.location = CAM_POS
cam.rotation_mode = "QUATERNION"
cam.rotation_quaternion = (TARGET - CAM_POS).to_track_quat("-Z", "Y")
scene.camera = cam

# ---------------------------------------------------------------- the frog: one metaball family
# Every body part is a metaball element: ellipsoids for head, body, palms and toe pads, ball chains for arms and
# fingers. Elements within each other's reach fuse into one smooth skin, so the arms grow out of the shoulders,
# the fingers out of the palms and into the toe pads.
K_MB = 0.575                    # iso radius / metaball radius of a lone element: 2 (1 - u)^3 = 0.6 (stiffness 2)
K_CHAIN, Q_CHAIN = 0.696, 0.3   # iso radius / metaball radius of a ball chain spaced Q_CHAIN x the metaball radius
FROG_MB = bpy.data.metaballs.new("frog")
FROG_MB.resolution = FROG_MB.render_resolution = 0.0025
FROG_MB.threshold = 0.6
SKELETON = []                   # (point, t): t = colour coordinate, 0 on the body, along a finger to 1 on its toe pad


def mb_ellipsoid(centre, semi, rot=None, t=0.0):
    r = max(semi)
    e = FROG_MB.elements.new(type="ELLIPSOID")
    e.co, e.radius = centre, r / K_MB
    e.size_x, e.size_y, e.size_z = (x / r for x in semi)
    if rot is not None:
        e.rotation = rot
    SKELETON.append((Vector(centre), t))


def mb_chain(pts, radii, ts):
    """Balls along a polyline (iso radius radii[i] at pts[i]), resampled at Q_CHAIN x the local metaball radius."""
    acc = [0.0]
    for a, b in zip(pts, pts[1:]):
        acc.append(acc[-1] + (b - a).length)

    def at(s):
        i = max(j for j in range(len(acc) - 1) if acc[j] <= s) if s < acc[-1] else len(acc) - 2
        f = min(max((s - acc[i]) / max(acc[i + 1] - acc[i], 1e-9), 0.0), 1.0)
        return pts[i].lerp(pts[i + 1], f), radii[i] + (radii[i + 1] - radii[i]) * f, ts[i] + (ts[i + 1] - ts[i]) * f

    s = 0.0
    while True:
        p, r, t = at(min(s, acc[-1]))
        e = FROG_MB.elements.new(type="BALL")
        e.co, e.radius = p, r / K_CHAIN
        SKELETON.append((p.copy(), t))
        if s >= acc[-1]:
            break
        s += Q_CHAIN * r / K_CHAIN


HEAD_Q = HEAD_M.to_3x3().normalized().to_quaternion()


def head_el(co, semi):
    mb_ellipsoid(HEAD_M @ Vector(co), tuple(x * HEAD_S for x in semi), HEAD_Q)


head_el((-0.02, 0.0, 0.01), (0.12, 0.13, 0.07))         # cranium
head_el((0.08, 0.0, 0.0), (0.075, 0.09, 0.05))          # snout
head_el((0.0, 0.0, -0.03), (0.12, 0.125, 0.035))        # lower jaw
for _side in (1, -1):                                   # the skin turrets the eyeballs sit in
    head_el((EYE_C.x - 0.006, _side * (EYE_C.y - 0.008), EYE_C.z - 0.014), (0.05, 0.05, 0.05))
head_el((-0.09, 0.0, -0.14), (0.13, 0.14, 0.12))        # neck and shoulders, below the rim
head_el((-0.13, 0.0, -0.30), (0.11, 0.12, 0.15))        # body, inside the barrel
SHOULDER_L = HEAD_M @ Vector((-0.05, 0.12, -0.16))
SHOULDER_R = HEAD_M @ Vector((-0.05, -0.12, -0.16))


def eyeball(name, centre, gaze):
    z = gaze.normalized()
    x = z.cross(UP).normalized()
    y = z.cross(x)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=EYE_R * HEAD_S)
    ob = bpy.context.active_object
    ob.name = name
    ob.matrix_world = Matrix.Translation(centre) @ Matrix((x, y, z)).transposed().to_4x4()
    ob.data.materials.append(EYE)
    bpy.ops.object.shade_smooth()
    return ob


eyeball("eye_left", EYE_L, AXIS)                                           # at the eyepiece
eyeball("eye_right", EYE_RT, (CAM_POS - EYE_RT).normalized() + 0.25 * HF)  # on the viewer

# ---------------------------------------------------------------- the frog's fingers
FR, PAD_R = 0.016, 0.025           # finger radius, toe-pad radius (a frog's discs are wider than the finger)
PAD_H = PAD_R * 0.30               # half thickness of a toe-pad disc


def finger(ctrl, pad_c, pad_n, r0=0.017, r1=0.0135, pad_r=PAD_R):
    pts = catmull(ctrl, per=14)
    n = len(pts)
    ts = [i / (n - 1) for i in range(n)]
    radii = [r0 + (r1 - r0) * t + 0.0016 * (math.exp(-((t - 0.38) / 0.06) ** 2) + math.exp(-((t - 0.66) / 0.06) ** 2))
             for t in ts]
    mb_chain(pts, radii, ts)
    mb_ellipsoid(pad_c, (pad_r, pad_r, pad_r * 0.3), Vector((0, 0, 1)).rotation_difference(Vector(pad_n)), t=1.0)


# hand A (right): on the rim, front left, fingers over the lip and down the outside of the staves
PHI = math.radians(-100.0)


def rim_pt(rho, tau, z):
    a = PHI + tau / rho
    return Vector((rho * math.cos(a), rho * math.sin(a), z))


def surface_r(z):
    """Outer radius of the nest at height z, iron hoops included."""
    return r_at(z) + (HOOP_T if any(abs(z - zc) < HOOP_HH + 0.6 * FR for zc in HOOPS) else 0.0)


def out_r(z):
    return surface_r(z) + FR + 0.0015


for k, (s, ln) in enumerate(((-0.085, 0.5), (-0.032, 1.0), (0.029, 1.1), (0.082, 0.7))):
    z_end = H - 0.07 - 0.05 * ln
    ctrl = [rim_pt(R_END - 0.11, 0.25 * s, H - 0.03), rim_pt(R_IN - 0.02, 0.55 * s, H + FR),
            rim_pt(R_END - STAVE_T / 2, 0.75 * s, H + FR + 0.001), rim_pt(R_END + 0.5 * FR, 0.85 * s, H + 0.7 * FR)]
    zs = [z for z in (H - 0.03 - 0.022 * i for i in range(30)) if z > z_end + 0.01] + [z_end]
    for j, z in enumerate(zs):
        ctrl.append(rim_pt(out_r(z), (0.85 + 0.15 * (j + 1) / len(zs)) * s, z))
    zp = z_end - 0.02
    a = PHI + 1.03 * s / out_r(zp)
    nrm = Vector((math.cos(a), math.sin(a), -0.2)).normalized()
    pc = Vector((math.cos(a), math.sin(a), 0)) * (surface_r(zp) + PAD_H + 0.0015) + UP * zp
    finger(ctrl, pc, nrm)
PALM_A = rim_pt(R_END - 0.13, 0.0, H - 0.04)
mb_ellipsoid(PALM_A, (0.04, 0.09, 0.03))
# the right arm, inside the barrel: shoulder -> elbow -> wrist under the palm
_arm = catmull([SHOULDER_R, SHOULDER_R.lerp(PALM_A, 0.5) - UP * 0.05, PALM_A - UP * 0.02], per=10)
mb_chain(_arm, [0.03 - 0.008 * i / (len(_arm) - 1) for i in range(len(_arm))], [0.0] * len(_arm))

# hand B (left): around the leather grip, from behind, over the top, pads on the front face
S_H = 0.2
u_ax = U_AX
to_cam = (CAM_POS - (ORIGIN + AXIS * S_H)).normalized()
f_ax = (to_cam - AXIS * to_cam.dot(AXIS) - u_ax * to_cam.dot(u_ax)).normalized()
R_GRIP = 0.0425


def grip_pt(ds, alpha_deg, rho):
    a = math.radians(alpha_deg)
    return ORIGIN + AXIS * (S_H + ds) + (f_ax * math.cos(a) + u_ax * math.sin(a)) * rho


for k, (ds, end) in enumerate(((-0.074, 6), (-0.025, -10), (0.024, -8), (0.071, 2))):
    rho = R_GRIP + FR + 0.0015
    ctrl = [grip_pt(ds * 0.6, 205, rho + 0.03), grip_pt(ds * 0.72, 165, rho + 0.006), grip_pt(ds * 0.84, 125, rho),
            grip_pt(ds * 0.95, 88, rho), grip_pt(ds * 1.05, 52, rho), grip_pt(ds * 1.12, end + 12, rho)]
    a = math.radians(end)
    n = f_ax * math.cos(a) + u_ax * math.sin(a)
    # pads 0.06 apart and 0.021 wide: closer or wider, neighbouring pads would fuse into one band
    finger(ctrl, ORIGIN + AXIS * (S_H + ds * 1.22) + n * (R_GRIP + 0.3 * 0.021 + 0.0015), n, pad_r=0.021)
PALM_B = ORIGIN + AXIS * S_H - f_ax * (R_GRIP + 0.04) + u_ax * 0.006
mb_ellipsoid(PALM_B, (0.04, 0.045, 0.1), Vector((0, 0, 1)).rotation_difference(AXIS))
# the left arm: the elbow rests on the rim below the glass, the forearm rises to the palm, the upper arm goes
# down into the barrel to the shoulder
wrist = PALM_B - u_ax * 0.05 - f_ax * 0.01
_a = math.atan2(wrist.y, wrist.x)
elbow = Vector(((R_END - STAVE_T / 2) * math.cos(_a), (R_END - STAVE_T / 2) * math.sin(_a), H + 0.026))
_arm = catmull([wrist, wrist.lerp(elbow, 0.5) - f_ax * 0.01, elbow, elbow.lerp(SHOULDER_L, 0.5) - UP * 0.03,
                SHOULDER_L], per=12)
mb_chain(_arm, [0.023 + 0.007 * i / (len(_arm) - 1) for i in range(len(_arm))], [0.0] * len(_arm))

# ---------------------------------------------------------------- mesh the frog, colour coordinate, face details
_ob = link(bpy.data.objects.new("frog_mb", FROG_MB))
bpy.context.view_layer.update()
_t0 = time.time()
FROG_ME = bpy.data.meshes.new_from_object(_ob.evaluated_get(bpy.context.evaluated_depsgraph_get()))
bpy.data.objects.remove(_ob)
FROG_ME.name = "frog"
FROG_ME.materials.append(FROG_SKIN)
FROG_ME.shade_smooth()
_kd = kdtree.KDTree(len(SKELETON))
for i, (q, _) in enumerate(SKELETON):
    _kd.insert(q, i)
_kd.balance()
_att = FROG_ME.attributes.new("t", "FLOAT", "POINT")
for v, a in zip(FROG_ME.vertices, _att.data):
    a.value = SKELETON[_kd.find(v.co)[1]][1]
frog = link(bpy.data.objects.new("frog", FROG_ME))
_sub = frog.modifiers.new("subsurf", "SUBSURF")
_sub.levels = _sub.render_levels = 1
print(f"frog: {len(FROG_MB.elements)} metaball elements -> {len(FROG_ME.vertices)} vertices "
      f"in {time.time() - _t0:.1f} s")
FROG_BVH = BVHTree.FromPolygons([v.co.copy() for v in FROG_ME.vertices], [tuple(p.vertices) for p in FROG_ME.polygons])
HEAD_R3 = HEAD_M.to_3x3()

# the mouth: a dark crease around the snout, rising a little toward the corners
mouth = []
for i in range(49):
    ph = math.radians(-118.0 + 236.0 * i / 48)
    z = -0.017 + 0.011 * (abs(ph) / math.radians(118.0)) ** 2.5
    loc, nrm, _, _ = FROG_BVH.ray_cast(HEAD_M @ Vector((-0.02, 0.0, z)),
                                       (HEAD_R3 @ Vector((math.cos(ph), math.sin(ph), 0.0))).normalized())
    if loc is not None:
        mouth.append(loc - nrm.normalized() * 0.0012)
tube("mouth", mouth, [0.0035] * len(mouth), [(0, len(mouth) - 1, 0)], [MOUTH])
nostrils = []
for side in (1, -1):
    loc, nrm, _, _ = FROG_BVH.ray_cast(HEAD_M @ Vector((0.09, side * 0.022, 0.0)),
                                       (HEAD_R3 @ Vector((0.55, 0.0, 0.83))).normalized())
    if loc is not None:
        nostrils.append((loc - nrm.normalized() * 0.0015, (0.006, 0.0035, 0.003), nrm))
ellipsoids("nostrils", nostrils, MOUTH, u=16, v=8)

# ---------------------------------------------------------------- lights
sun = bpy.data.lights.new("sun", "SUN")
sun.energy = 4.2
sun.angle = math.radians(1.5)
sun.color = (1.0, 0.72, 0.5)
sun_ob = link(bpy.data.objects.new("sun", sun))
S_DIR = Vector((math.cos(math.radians(9)) * math.cos(math.radians(-112)),
                math.cos(math.radians(9)) * math.sin(math.radians(-112)), math.sin(math.radians(9))))
sun_ob.rotation_mode = "QUATERNION"
sun_ob.rotation_quaternion = UP.rotation_difference(S_DIR)


def area(name, pos, energy, size, color, aim=Vector((0.1, -0.1, 0.67))):
    li = bpy.data.lights.new(name, "AREA")
    li.energy, li.size, li.color = energy, size, color
    ob = link(bpy.data.objects.new(name, li))
    ob.location = pos
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = Vector((0, 0, -1)).rotation_difference(aim - Vector(pos))
    return ob


area("fill", (4.0, -3.2, 1.6), 70.0, 2.5, (0.82, 0.87, 1.0))
area("rim", (1.8, 3.6, 2.6), 160.0, 1.5, (1.0, 0.62, 0.82))
area("key", (0.6, -1.9, 2.2), 90.0, 1.6, (1.0, 0.86, 0.7), aim=Vector((0.4, -0.2, 0.72)))  # gold band on the brass

# ---------------------------------------------------------------- render
cam_data.dof.use_dof = True
cam_data.dof.focus_distance = (ORIGIN + AXIS * S_H - CAM_POS).length
cam_data.dof.aperture_fstop = 8.0

scene.render.engine = "CYCLES"
cy = scene.cycles
cy.samples = args.samples
cy.use_adaptive_sampling = True
cy.use_denoising = True
cy.max_bounces = 10
cy.caustics_reflective = False
cy.caustics_refractive = False
if args.device == "GPU":
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for dev_type in ("OPTIX", "CUDA"):
        try:
            prefs.compute_device_type = dev_type
        except TypeError:
            continue
        prefs.get_devices()
        if any(d.type == dev_type for d in prefs.devices):
            for d in prefs.devices:
                d.use = d.type == dev_type
            cy.device = "GPU"
            break
    print("devices:", [(d.name, d.type, d.use) for d in prefs.devices])
scene.render.resolution_x = scene.render.resolution_y = args.res
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "16"
scene.view_settings.view_transform = "AgX"
for look in ("AgX - Punchy", "Punchy"):
    try:
        scene.view_settings.look = look
        break
    except TypeError:
        pass
scene.view_settings.exposure = 0.0
scene.render.filepath = args.out
scene.render.use_stamp_filename = False      # the PNG metadata would carry the .blend path of the render machine

# ---------------------------------------------------------------- the badge disc
# The smallest circle (in the frame, as fractions of the width, y down) around what must stay in the round badge:
# the objective end of the glass, the toe pads and the eyes. make_logo.py --3d reads it from <out>.json.


def enclosing_circle(pts):
    """Welzl's smallest enclosing circle, randomised incremental form."""
    pts = pts[:]
    random.Random(0).shuffle(pts)

    def inside(c, p):
        return c is not None and math.dist(c[0], p) <= c[1] + 1e-9

    def through3(a, b, c):
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        ux = ((a[0] ** 2 + a[1] ** 2) * (b[1] - c[1]) + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
              + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])) / d
        uy = ((a[0] ** 2 + a[1] ** 2) * (c[0] - b[0]) + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
              + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])) / d
        return (ux, uy), math.dist((ux, uy), a)

    c = None
    for i, p in enumerate(pts):
        if inside(c, p):
            continue
        c = (p, 0.0)
        for j, q in enumerate(pts[:i]):
            if inside(c, q):
                continue
            c = (((p[0] + q[0]) / 2, (p[1] + q[1]) / 2), math.dist(p, q) / 2)
            for k in pts[:j]:
                if not inside(c, k):
                    c = through3(p, q, k)
    return c


_side = AXIS.cross(U_AX)
_key = [ORIGIN + AXIS * SECTIONS[-1][1] + (U_AX * math.cos(a) + _side * math.sin(a)) * 0.0465
        for a in (2 * math.pi * i / 24 for i in range(24))]
_key += [q for q, t in SKELETON if t >= 1.0] + [e + UP * EYE_R * HEAD_S for e in (EYE_L, EYE_RT)]
_uv = [(v.x, 1.0 - v.y) for v in (world_to_camera_view(scene, cam, q) for q in _key)]
(_cx, _cy), _r = enclosing_circle(_uv)
_r += 0.035                                       # margin: pad and rim radii, air
_cx, _cy = min(max(_cx, _r), 1 - _r), min(max(_cy, _r), 1 - _r)
# what overlays in the badge must keep clear of: (u, v, radius) circles along the glass, on the frog's skin
# skeleton above the rim and on the eyes, and the mast as a vertical band
_cam_right = cam.matrix_world.to_3x3() @ Vector((1, 0, 0))


def _proj(q, rad):
    a, b = world_to_camera_view(scene, cam, q), world_to_camera_view(scene, cam, q + _cam_right * rad)
    return [round(a.x, 4), round(1.0 - a.y, 4), round(math.hypot(b.x - a.x, b.y - a.y), 4)]


_keep = [_proj(ORIGIN + AXIS * (SECTIONS[-1][1] * i / 40), 0.05) for i in range(41)]
_keep += [_proj(q, 0.03) for q, _ in SKELETON if q.z > H - 0.02]
_keep += [_proj(e, EYE_R * HEAD_S) for e in (EYE_L, EYE_RT)]
_keep.append(_proj(HEAD_M @ Vector((0.0, 0.0, 0.02)), 0.16 * HEAD_S))          # the head as a whole
_mast = [_proj(Vector((0, 0, z)), 0.075) for z in (H, H + 1.2)]
with open(os.path.splitext(args.out)[0] + ".json", "w") as fh:
    json.dump({"cx": round(_cx, 4), "cy": round(_cy, 4), "r": round(_r, 4), "keepout": _keep, "mast": _mast,
               "glass": [_proj(ORIGIN, 0)[:2], _proj(ORIGIN + AXIS * SECTIONS[-1][1], 0)[:2]]}, fh)
print(f"badge disc: centre ({_cx:.3f}, {_cy:.3f}), radius {_r:.3f} of the frame width"
      + ("  -- WARNING: larger than the frame" if _r > 0.5 else ""))

if args.blend:
    bpy.ops.wm.save_as_mainfile(filepath=args.blend)
if args.no_render:
    sys.exit(0)
t0 = time.time()
bpy.ops.render.render(write_still=True)
print(f"rendered {args.out} in {time.time() - t0:.1f} s")
