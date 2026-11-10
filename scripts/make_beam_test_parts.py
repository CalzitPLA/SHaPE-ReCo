"""
Build the beam-compound test suite in dataset_gen/examples/beams/ (run with freecadcmd, which provides FreeCAD/Part):

    freecadcmd scripts/make_beam_test_parts.py
    BEAM_PARTS_ONLY=j01_l_butt BEAM_PARTS_OUT=/tmp/x freecadcmd scripts/make_beam_test_parts.py    # one part, elsewhere

Every part is a real-size steel structure in mm, built from FreeCAD Part solids and booleans only: I-sections with
dataset_gen/src/ibeam_freecad.make_ibeam_solid (DIN dims from dataset_gen/src/din_profiles.py, root fillets on), rods
and tubes (CHS) as cylinders, flat bars and RHS as boxes. Per part three files: <name>.stl, <name>.step and <name>.json,
the ground truth: the members (kind, profile or section dims, nominal start and end points of the centre line in mm,
the "up" vector = the direction of the section depth h), the joints, extra features (plates, holes) and the expected
minimal program of the member sub-loop (reco_engine/ibeam.py): I-beam = 3 box adds, rod = 1 cylinder add, bar = 1 box
add, tube = cylinder add + cylinder cut, rhs = box add + box cut, plate = box add, bolt hole = cylinder cut.

Members that meet are extended 1 mm into the member they butt against (`ext`), so no two solids only touch; the
nominal start/end in the JSON do not include that overlap. Mitred ends overlap by 2 mm across the mitre plane.

Parts, from easy to hard (s = single member, j = joint, f = frame, t = truss / space frame, m = mixed features):
  s01_ipe200            IPE 200 x 2 m, axis x
  s02_hea500_rot        HEA 500 x 3 m, rotated 30 deg about x, then 30 deg about y
  s03_rod               round rod d 60 x 1.5 m, oblique in the xz plane
  s04_tube              CHS 168.3 x 8 x 2 m, axis y
  s05_rhs               RHS 200 x 100 x 8 x 2 m, axis x
  s06_flatbar           flat bar 100 x 20 x 1.5 m, axis z
  j01_l_butt            L-corner, two HEA 300: post under the end of a girder (butt joint)
  j02_l_mitre           L-corner, two HEA 300, 45 deg mitre
  j03_t_joint           HEA 300 stem standing on the mid-span of an HEA 300 beam
  j04_x_crossing        HEB 200 passing through the web of an HEA 400 (between its flanges)
  f01_portal            portal frame: 2 HEA 360 posts (4 m) + HEA 400 girder (6 m span) on top
  f02_rect_ipe300       rectangular frame of 4 IPE 300 in one plane (4 x 3 m), posts between the chords
  f03_two_bay           2-bay frame: 3 HEA 300 posts, 2 IPE 400 girders framing into the post flanges
  f04_frame_quarter     the benchmark Frame_4_x_6's layout, spans at 1/4 (1675 x 1245 mm, layers 700 mm apart), real
                        profiles: HEA 500 chords, posts 350 x 300 x 12 x 17.5 (HEA 360 envelope, 12 mm web), HEA 500
                        cross members between the two layers (12 members)
  t01_warren            planar Warren truss, RHS 100 x 100 x 6 chords, 12 rod diagonals d 40, 6 m span
  t02_space_frame       box edge frame 2 x 1.5 x 1 m of 12 CHS tubes, one size per direction (88.9 / 76.1 /
                        101.6 x 5; one closed inner cavity network)
  m01_endplate          IPE 300 x 2 m with a 400 x 200 x 20 end plate welded on
  m02_bolt_holes        HEA 300 x 2 m with 4 bolt holes d 22 through both flanges
  m03_mixed             HEB 200 + HEB 300 posts under an IPE 300 girder

Used by scripts/eval_beam_parts.py and tests/test_beam_parts.py.
"""
import json
import math
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
for p in (os.path.join(ROOT, "dataset_gen"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from src import din_profiles as dp  # noqa: E402

OUT = os.environ.get("BEAM_PARTS_OUT") or os.path.join(ROOT, "dataset_gen", "examples", "beams")
STL_DEFLECTION = 0.5          # mm
OVERLAP = 1.0                 # mm a butting member runs into the other one

EXPECTED_OPS = {"ibeam": ("add", "add", "add"), "rod": ("add",), "bar": ("add",), "tube": ("add", "cut"),
                "rhs": ("add", "cut")}


# ------------------------------------------------------------------------------------------------ small vector helpers

def _v(a):
    return [float(x) for x in a]


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _add(a, b):
    return [x + y for x, y in zip(a, b)]


def _mul(a, s):
    return [x * s for x in a]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _unit(a):
    n = math.sqrt(_dot(a, a))
    return [x / n for x in a]


def _rot(axis, deg, v):
    """Rodrigues rotation of v about a unit axis."""
    k, t = _unit(axis), math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return _add(_add(_mul(v, c), _mul(_cross(k, v), s)), _mul(k, _dot(k, v) * (1 - c)))


# ------------------------------------------------------------------------------------------------ member specs

def ibeam(profile, start, end, up, ext=(0.0, 0.0), dims=None):
    """An I-section member. `profile`: DIN name; `dims` (h, b, tw, tf, r) overrides the table (non-standard web)."""
    p = dp.PROFILES[profile]
    d = dict(h=p.h, b=p.b, tw=p.tw, tf=p.tf, r=p.r)
    if dims:
        d.update(dims)
    snapped = not dims
    return dict(kind="ibeam", profile=profile if snapped else None, envelope=profile, section=d, start=_v(start),
                end=_v(end), up=_v(up), ext=_v(ext))


def rod(d, start, end, ext=(0.0, 0.0)):
    return dict(kind="rod", profile=None, section=dict(d=d), start=_v(start), end=_v(end), up=None, ext=_v(ext))


def tube(d, t, start, end, ext=(0.0, 0.0), inner_ext=None):
    return dict(kind="tube", profile=f"CHS {d:g}x{t:g}", section=dict(d=d, t=t), start=_v(start), end=_v(end), up=None,
                ext=_v(ext), inner_ext=_v(inner_ext) if inner_ext is not None else None)


def bar(w, t, start, end, up, ext=(0.0, 0.0)):
    """Flat bar: thickness t along `up`, width w across."""
    return dict(kind="bar", profile=f"FL {w:g}x{t:g}", section=dict(w=w, t=t), start=_v(start), end=_v(end), up=_v(up),
                ext=_v(ext))


def rhs(h, b, t, start, end, up, ext=(0.0, 0.0)):
    """Rectangular hollow section: depth h along `up`, width b, wall t."""
    return dict(kind="rhs", profile=f"RHS {h:g}x{b:g}x{t:g}", section=dict(h=h, b=b, t=t), start=_v(start),
                end=_v(end), up=_v(up), ext=_v(ext))


def spec_parts():
    """{name: part spec}: description, members, joints, extra features (plates / holes / mitre planes)."""
    P = {}
    X, Y, Z = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    O = [0, 0, 0]

    P["s01_ipe200"] = dict(description="IPE 200 x 2 m along x", members=[ibeam("IPE 200", O, [2000, 0, 0], Z)])

    def r2(v):
        return _rot(Y, 30, _rot(X, 30, v))
    P["s02_hea500_rot"] = dict(description="HEA 500 x 3 m, rotated 30 deg about x then 30 deg about y",
                               members=[ibeam("HEA 500", r2([-1500, 0, 0]), r2([1500, 0, 0]), r2(Z))])
    P["s03_rod"] = dict(description="round rod d 60 x 1.5 m, oblique in the xz plane", members=[rod(60, O, [1200, 0, 900])])
    P["s04_tube"] = dict(description="CHS 168.3 x 8 x 2 m along y", members=[tube(168.3, 8, O, [0, 2000, 0])])
    P["s05_rhs"] = dict(description="RHS 200 x 100 x 8 x 2 m along x", members=[rhs(200, 100, 8, O, [2000, 0, 0], Z)])
    P["s06_flatbar"] = dict(description="flat bar 100 x 20 x 1.5 m along z", members=[bar(100, 20, O, [0, 0, 1500], X)])

    h3 = dp.PROFILES["HEA 300"].h
    P["j01_l_butt"] = dict(
        description="L-corner of two HEA 300: post under the end of the girder (butt joint)",
        members=[ibeam("HEA 300", O, [0, 0, 2000], X, ext=(0, OVERLAP)),
                 ibeam("HEA 300", [-h3 / 2, 0, 2000 + h3 / 2], [2000, 0, 2000 + h3 / 2], Z)],
        joints=[dict(type="butt", members=[0, 1])])
    P["j02_l_mitre"] = dict(
        description="L-corner of two HEA 300 with a 45 deg mitre",
        members=[ibeam("HEA 300", O, [2000, 0, 0], Z, ext=(h3 / 2 + 10, 0)),
                 ibeam("HEA 300", O, [0, 0, 2000], X, ext=(h3 / 2 + 10, 0))],
        joints=[dict(type="mitre", members=[0, 1], angle_deg=45)],
        # member i keeps the side of its mitre plane opposite to `normal` (the plane overlaps the other member by 1 mm)
        mitres=[dict(member=0, point=[-0.5, 0, 0.5], normal=_unit([-1, 0, 1])),
                dict(member=1, point=[0.5, 0, -0.5], normal=_unit([1, 0, -1]))])
    P["j03_t_joint"] = dict(
        description="T-joint: HEA 300 stem standing on the mid-span of an HEA 300 beam",
        members=[ibeam("HEA 300", O, [3000, 0, 0], Z),
                 ibeam("HEA 300", [1500, 0, h3 / 2], [1500, 0, h3 / 2 + 2000], X, ext=(OVERLAP, 0))],
        joints=[dict(type="butt", members=[1, 0])])
    P["j04_x_crossing"] = dict(
        description="X-crossing: HEB 200 passing through the web of an HEA 400, between its flanges",
        members=[ibeam("HEA 400", O, [3000, 0, 0], Z), ibeam("HEB 200", [1500, -1000, 0], [1500, 1000, 0], Z)],
        joints=[dict(type="through", members=[1, 0])])

    h36, h40 = dp.PROFILES["HEA 360"].h, dp.PROFILES["HEA 400"].h
    P["f01_portal"] = dict(
        description="portal frame: 2 HEA 360 posts (4 m) and an HEA 400 girder (6 m span) on top",
        members=[ibeam("HEA 360", O, [0, 0, 4000], X, ext=(0, OVERLAP)),
                 ibeam("HEA 360", [6000, 0, 0], [6000, 0, 4000], X, ext=(0, OVERLAP)),
                 ibeam("HEA 400", [-h36 / 2, 0, 4000 + h40 / 2], [6000 + h36 / 2, 0, 4000 + h40 / 2], Z)],
        joints=[dict(type="butt", members=[0, 2]), dict(type="butt", members=[1, 2])])

    hi3 = dp.PROFILES["IPE 300"].h
    W, H = 4000.0, 3000.0
    P["f02_rect_ipe300"] = dict(
        description="rectangular frame of 4 IPE 300 in the xz plane (4 x 3 m), posts between the chords",
        members=[ibeam("IPE 300", [-hi3 / 2, 0, 0], [W + hi3 / 2, 0, 0], Z),
                 ibeam("IPE 300", [-hi3 / 2, 0, H], [W + hi3 / 2, 0, H], Z),
                 ibeam("IPE 300", [0, 0, hi3 / 2], [0, 0, H - hi3 / 2], X, ext=(OVERLAP, OVERLAP)),
                 ibeam("IPE 300", [W, 0, hi3 / 2], [W, 0, H - hi3 / 2], X, ext=(OVERLAP, OVERLAP))],
        joints=[dict(type="butt", members=[2, 0]), dict(type="butt", members=[2, 1]),
                dict(type="butt", members=[3, 0]), dict(type="butt", members=[3, 1])])

    hi4 = dp.PROFILES["IPE 400"].h
    zg = 4000 - 20 - hi4 / 2
    P["f03_two_bay"] = dict(
        description="2-bay frame: 3 HEA 300 posts (4 m, bays 5 m), 2 IPE 400 girders framing into the post flanges",
        members=[ibeam("HEA 300", [x, 0, 0], [x, 0, 4000], X) for x in (0, 5000, 10000)]
        + [ibeam("IPE 400", [x0 + h3 / 2, 0, zg], [x0 + 5000 - h3 / 2, 0, zg], Z, ext=(OVERLAP, OVERLAP))
           for x0 in (0, 5000)],
        joints=[dict(type="butt", members=[3, 0]), dict(type="butt", members=[3, 1]),
                dict(type="butt", members=[4, 1]), dict(type="butt", members=[4, 2])])

    # the benchmark frame (Frame_4_x_6.stp: 1000 x 6700 x 4980 mm, two planar layers with their webs 700 mm apart, HEA 500
    # chords, posts 350 x 300 x 12 x 17.5, HEA 500 cross members between the layers), spans at 1/4
    h5 = dp.PROFILES["HEA 500"].h
    post = dict(tw=12.0)
    LY, LZ, GAP = 6700 / 4, 4980 / 4, 700.0
    yp = (h36 / 2, LY - h36 / 2)
    zc = (h5 / 2, LZ - h5 / 2)
    mem, joints = [], []
    for xl in (0.0, GAP):
        i0 = len(mem)
        mem += [ibeam("HEA 360", [xl, y, 0], [xl, y, LZ], Y, dims=post) for y in yp]
        mem += [ibeam("HEA 500", [xl, h36, z], [xl, LY - h36, z], Z, ext=(OVERLAP, OVERLAP)) for z in zc]
        joints += [dict(type="butt", members=[i0 + 2 + k, i0 + j]) for k in (0, 1) for j in (0, 1)]
    tw_post = post["tw"]
    for y in yp:
        for z in zc:
            mem.append(ibeam("HEA 500", [tw_post / 2, y, z], [GAP - tw_post / 2, y, z], Z, ext=(OVERLAP, OVERLAP)))
            joints.append(dict(type="butt", members=[len(mem) - 1, 0 if y == yp[0] else 1]))
            joints.append(dict(type="butt", members=[len(mem) - 1, 4 if y == yp[0] else 5]))
    P["f04_frame_quarter"] = dict(
        description="Frame_4_x_6 layout at 1/4 span (1675 x 1245 mm, 2 layers 700 mm apart): HEA 500 chords, posts "
                    "350x300x12x17.5 (HEA 360 envelope, 12 mm web), 4 HEA 500 cross members", members=mem, joints=joints)

    # Warren truss: RHS chords along x (bottom z = 0, top z = 1000), rod diagonals node to node (chord centre lines);
    # the rods are cut back to the chords' outer walls, so the chords stay hollow
    hc = 100.0
    mem = [rhs(hc, hc, 6, [-50, 0, 0], [6050, 0, 0], Z), rhs(hc, hc, 6, [-50, 0, 1000], [6050, 0, 1000], Z)]
    joints = []
    for k in range(6):
        x = 1000.0 * k
        mem.append(rod(40, [x, 0, 0], [x + 500, 0, 1000]))
        mem.append(rod(40, [x + 500, 0, 1000], [x + 1000, 0, 0]))
        joints += [dict(type="welded", members=[len(mem) - 2, 0]), dict(type="welded", members=[len(mem) - 2, 1]),
                   dict(type="welded", members=[len(mem) - 1, 0]), dict(type="welded", members=[len(mem) - 1, 1])]
    P["t01_warren"] = dict(description="planar Warren truss: RHS 100x100x6 chords (6.1 m), 12 rod diagonals d 40",
                           members=mem, joints=joints, rods_trimmed_to_chords=True)

    # space frame: the 12 edges of a 2000 x 1500 x 1000 box, one CHS size per direction (x: 88.9 x 5, y: 76.1 x 5,
    # z: 101.6 x 5 - equal radii on perpendicular intersecting axes give the degenerate Steinmetz curve OCC cannot cut
    # through); the outer cylinders run r past every node, the bores 15 mm past the nodes (inside the outer extension;
    # bores ending exactly at the node make coincident caps) - one closed cavity network
    nodes = [[x, y, z] for x in (0, 2000) for y in (0, 1500) for z in (0, 1000)]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8)
             if sum(1 for i in range(3) if nodes[a][i] != nodes[b][i]) == 1]
    chs = (88.9, 76.1, 101.6)
    mem = []
    for a, b in edges:
        d = chs[next(i for i in range(3) if nodes[a][i] != nodes[b][i])]
        mem.append(tube(d, 5, nodes[a], nodes[b], ext=(d / 2, d / 2), inner_ext=(15, 15)))
    joints = []
    for n in range(8):
        at = [i for i, (a, b) in enumerate(edges) if n in (a, b)]
        joints.append(dict(type="node", members=at))
    P["t02_space_frame"] = dict(description="space frame: the 12 edges of a 2 x 1.5 x 1 m box, CHS tubes (x 88.9x5, y 76.1x5, z 101.6x5)",
                                members=mem, joints=joints)

    P["m01_endplate"] = dict(
        description="IPE 300 x 2 m with a 400 x 200 x 20 end plate welded on",
        members=[ibeam("IPE 300", O, [2000, 0, 0], Z, ext=(0, OVERLAP))],
        features=[dict(op="add", kind="box", name="end plate", min=[2000, -100, -200], max=[2020, 100, 200])],
        joints=[dict(type="welded_plate", members=[0], feature=0)])
    feats = [dict(op="cut", kind="cylinder", name="bolt hole", d=22.0, start=[x, y, -200], end=[x, y, 200])
             for x in (100, 1900) for y in (-80, 80)]
    P["m02_bolt_holes"] = dict(description="HEA 300 x 2 m with 4 bolt holes d 22 through both flanges",
                               members=[ibeam("HEA 300", O, [2000, 0, 0], Z)], features=feats)

    hb2, hb3 = dp.PROFILES["HEB 200"].h, dp.PROFILES["HEB 300"].h
    zt = 2500 + hi3 / 2
    P["m03_mixed"] = dict(
        description="mixed profiles: HEB 200 and HEB 300 posts (2.5 m) under an IPE 300 girder",
        members=[ibeam("HEB 200", O, [0, 0, 2500], X, ext=(0, OVERLAP)),
                 ibeam("HEB 300", [3000, 0, 0], [3000, 0, 2500], X, ext=(0, OVERLAP)),
                 ibeam("IPE 300", [-hb2 / 2, 0, zt], [3000 + hb3 / 2, 0, zt], Z)],
        joints=[dict(type="butt", members=[0, 2]), dict(type="butt", members=[1, 2])])
    for spec in P.values():
        spec.setdefault("joints", [])
        spec.setdefault("features", [])
    return P


def expected_program(spec):
    """The minimal program the member sub-loop should find: per member its ops, plus the features."""
    ops = []
    for i, m in enumerate(spec["members"]):
        for op in EXPECTED_OPS[m["kind"]]:
            kind = "cylinder" if m["kind"] in ("rod", "tube") else "box"
            ops.append(dict(op=op, kind=kind, member=i))
    for j, f in enumerate(spec["features"]):
        ops.append(dict(op=f["op"], kind=f["kind"], feature=j))
    adds = sum(o["op"] == "add" for o in ops)
    notes = []
    if spec.get("mitres"):
        notes.append("mitre: the 3-box I-beams reproduce it up to tf-sized triangles at the flange ends (a rotated box cut "
                     "per member would be exact)")
    if spec.get("rods_trimmed_to_chords"):
        notes.append("the rods end inside the chord walls: the chord cuts must come after the rod adds")
    return dict(ops=ops, adds=adds, cuts=len(ops) - adds, total=len(ops), notes=notes)


# ------------------------------------------------------------------------------------------------ FreeCAD build

def _place(shape, start, end, up):
    """Local frame (axis z centred, depth y, width x) -> world: z along start->end, y along `up`, centred at the middle."""
    import FreeCAD as App
    a = _unit(_sub(end, start))
    u = up if up is not None else ([0, 0, 1] if abs(a[2]) < 0.9 else [1, 0, 0])
    u = _unit(_sub(u, _mul(a, _dot(u, a))))
    x = _cross(u, a)
    mid = _mul(_add(start, end), 0.5)
    m = App.Matrix(x[0], u[0], a[0], mid[0], x[1], u[1], a[1], mid[1], x[2], u[2], a[2], mid[2], 0, 0, 0, 1)
    shape = shape.copy()
    shape.transformShape(m)
    return shape


def _extended(m, ext):
    a = _unit(_sub(m["end"], m["start"]))
    return _sub(m["start"], _mul(a, ext[0])), _add(m["end"], _mul(a, ext[1]))


def _box_local(sx, sy, L):
    import FreeCAD as App
    import Part
    return Part.makeBox(sx, sy, L, App.Vector(-sx / 2, -sy / 2, -L / 2))


def _cyl_local(r, L):
    import FreeCAD as App
    import Part
    return Part.makeCylinder(r, L, App.Vector(0, 0, -L / 2), App.Vector(0, 0, 1))


def member_solids(m):
    """(outer solid, inner cutter or None) of one member, placed in world mm."""
    from src.ibeam_freecad import make_ibeam_solid
    s0, s1 = _extended(m, m["ext"])
    L = math.dist(s0, s1)
    sec, k = m["section"], m["kind"]
    inner = None
    if k == "ibeam":
        outer = make_ibeam_solid(sec, L, fillet=True)
    elif k == "rod":
        outer = _cyl_local(sec["d"] / 2, L)
    elif k == "bar":
        outer = _box_local(sec["w"], sec["t"], L)
    elif k == "tube":
        outer = _cyl_local(sec["d"] / 2, L)
        ie = m.get("inner_ext")
        if ie is None:                                # through bore: 1 mm past both ends
            ie = [m["ext"][0] + 1, m["ext"][1] + 1]
        i0, i1 = _extended(m, ie)
        inner = _place(_cyl_local(sec["d"] / 2 - sec["t"], math.dist(i0, i1)), i0, i1, m["up"])
    elif k == "rhs":
        outer = _box_local(sec["b"], sec["h"], L)
        inner = _place(_box_local(sec["b"] - 2 * sec["t"], sec["h"] - 2 * sec["t"], L + 2), s0, s1, m["up"])
    else:
        raise ValueError(k)
    return _place(outer, s0, s1, m["up"]), inner


def _halfspace(point, normal, size=1e5):
    """A big box filling the side of the plane (point, normal) that `normal` points into."""
    import FreeCAD as App
    import Part
    b = Part.makeBox(size, size, size, App.Vector(-size / 2, -size / 2, 0))
    b.Placement = App.Placement(App.Vector(*point), App.Rotation(App.Vector(0, 0, 1), App.Vector(*normal)))
    return b


def build_part(spec):
    """The part as one Part solid: adds (members, plate features) fused, then all cutters (hollows, holes)."""
    import FreeCAD as App
    import Part
    outers, inners = [], []
    for i, m in enumerate(spec["members"]):
        o, c = member_solids(m)
        for mt in spec.get("mitres", []):
            if mt["member"] == i:
                o = o.cut(_halfspace(mt["point"], mt["normal"])).Solids[0]
        outers.append(o)
        if c is not None:
            inners.append(c)
    if spec.get("rods_trimmed_to_chords"):           # rods must not fill the chord hollows: cut them back per rod
        outers = [o.cut(Part.makeCompound(inners)) if m["kind"] == "rod" else o
                  for o, m in zip(outers, spec["members"])]
    for f in spec["features"]:
        if f["kind"] == "box":
            lo, hi = f["min"], f["max"]
            s = Part.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], App.Vector(*lo))
        else:
            s = _place(_cyl_local(f["d"] / 2, math.dist(f["start"], f["end"])), f["start"], f["end"], None)
        (outers if f["op"] == "add" else inners).append(s)
    shape = outers[0].multiFuse(outers[1:]) if len(outers) > 1 else outers[0]
    shape = shape.removeSplitter()
    if inners:
        shape = shape.cut(inners[0].multiFuse(inners[1:]) if len(inners) > 1 else inners[0]).removeSplitter()
    if len(shape.Solids) == 1 and shape.ShapeType != "Solid":
        shape = shape.Solids[0]
    return shape


def main():
    import FreeCAD as App  # noqa: F401
    from src.ibeam_freecad import export_stl
    os.makedirs(OUT, exist_ok=True)
    only = [s for s in os.environ.get("BEAM_PARTS_ONLY", "").split(",") if s]
    specs = spec_parts()
    status = {}
    for name, spec in specs.items():
        if only and name not in only:
            continue
        try:
            shape = build_part(spec)
        except Exception as exc:
            status[name] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
            print("%-18s BUILD FAILED: %s" % (name, exc))
            continue
        valid, n_solids = bool(shape.isValid()), len(shape.Solids)
        stl = os.path.join(OUT, name + ".stl")
        facets = export_stl(shape, stl, STL_DEFLECTION)
        shape.exportStep(os.path.join(OUT, name + ".step"))
        bb = shape.BoundBox
        gt = dict(name=name, description=spec["description"], units="mm", members=spec["members"],
                  joints=spec["joints"], features=spec["features"], mitres=spec.get("mitres", []),
                  expected_program=expected_program(spec), volume_mm3=float(shape.Volume), valid=valid,
                  n_solids=n_solids, stl_facets=facets, bbox_min=[bb.XMin, bb.YMin, bb.ZMin],
                  bbox_max=[bb.XMax, bb.YMax, bb.ZMax], diagonal_mm=float(bb.DiagonalLength),
                  stl_deflection_mm=STL_DEFLECTION)
        with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(gt, fh, indent=1)
        status[name] = dict(ok=valid and n_solids == 1, volume_mm3=float(shape.Volume), valid=valid, n_solids=n_solids)
        print("%-18s %2d members, volume %.4g mm^3, %d solid(s), %s, %d facets, expected %d ops"
              % (name, len(spec["members"]), shape.Volume, n_solids, "valid" if valid else "INVALID", facets,
                 gt["expected_program"]["total"]))
    with open(os.path.join(OUT, "_build_status.json"), "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=1)


try:
    import FreeCAD  # noqa: F401
    _IN_FREECAD = True
except ImportError:
    _IN_FREECAD = False

if _IN_FREECAD and __name__ in ("__main__", "make_beam_test_parts") and not os.environ.get("BEAM_PARTS_NO_MAIN"):
    main()
