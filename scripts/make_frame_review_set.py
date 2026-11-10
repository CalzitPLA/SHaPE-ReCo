"""
The frame review set: 15 assemblies of frames, each as ONE body (the union of its members) and as a MULTI-BODY
compound (one solid per member), with the ground truth of every member. Run with freecadcmd:

    freecadcmd scripts/make_frame_review_set.py
    FRAME_REVIEW_ONLY=r03_rect_heb_plates FRAME_REVIEW_OUT=/tmp/x freecadcmd scripts/make_frame_review_set.py

Built with the member builders of scripts/make_beam_test_parts.py (real-size steel in mm, FreeCAD Part solids and
booleans only, DIN sections with their root fillets, members that meet run 1 mm into each other). Per assembly, in
dataset_gen/examples/frame_review/:

    <name>_union.step / .stl        the fused single solid - what the reconstruction gets first
    <name>_multibody.step / .stl    a compound of the members' solids, hollow members already hollow, in member order
    <name>.json                     ground truth: members (kind, section, start, end, up), joints, the expected minimal
                                    program, volume and validity of the union, volume and validity of every body

The set, from plain to hard (what each one asks of the reconstruction):
  r01_portal_hea          3 members, two sections                       the smallest frame
  r02_rect_ipe            4 IPE 300, closed rectangle                   genus 1
  r03_rect_heb_plates     4 HEB 240 + 4 corner stiffener plates         thin plates next to thick members
  r04_two_bay             3 posts, 2 girders                            girders framing into post flanges
  r05_grid_2x2            3 posts, 2 storeys, 4 girders                 several cells, genus 4
  r06_ladder_rhs          2 RHS chords, 5 RHS rungs                     hollow members, many equal ones
  r07_rect_rhs            4 RHS 200 x 100 x 8                           hollow rectangle, a closed cavity per member
  r08_rect_chs            4 CHS tubes, two sizes                        round hollow members
  r09_mixed_sections      HEA posts, IPE girder, flat-bar diagonals     three kinds of section in one frame
  r10_braced_portal       portal with two rod braces                    oblique members crossing in the panel
  r11_double_layer        two rectangles of IPE 240, 4 cross members    a 3-D frame of two planar layers
  r12_benchmark_like      the benchmark's layout: HEA 500 chords, posts with a 12 mm web, 4 + 8 plates
  r13_rotated_rect        r02 turned 30 deg about x and 20 deg about y  no member along a world axis
  r14_mitred_rect         4 IPE 300 with 45 deg mitres                  no butt joints, corners shared
  r15_space_frame_hea     the 12 edges of a box, HEA 200                three member directions, genus 5

This is a REVIEW set (evaluation), not training data: the project's rule "single uniform manifolds" holds for training
samples; the multi-body files here are inputs to study, and the Type CNN is out of its scope on them.
"""
import json
import math
import os
import sys

os.environ["BEAM_PARTS_NO_MAIN"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
for p in (HERE, os.path.join(ROOT, "dataset_gen"), ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import make_beam_test_parts as mb  # noqa: E402
from src import din_profiles as dp  # noqa: E402

OUT = os.environ.get("FRAME_REVIEW_OUT") or os.path.join(ROOT, "dataset_gen", "examples", "frame_review")
STL_DEFLECTION = 0.5
OV = mb.OVERLAP
X, Y, Z, O = [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]
ibeam, rod, tube, bar, rhs = mb.ibeam, mb.rod, mb.tube, mb.bar, mb.rhs


def h(profile):
    return dp.PROFILES[profile].h


def rectangle(chord, post, W, H, y=0.0):
    """Two chords along x (z = 0 and H) over the posts' outer faces, two posts between them: members 0..3."""
    hp, hc = h(post), h(chord)
    return [ibeam(chord, [-hp / 2, y, 0], [W + hp / 2, y, 0], Z), ibeam(chord, [-hp / 2, y, H], [W + hp / 2, y, H], Z),
            ibeam(post, [0, y, hc / 2], [0, y, H - hc / 2], X, ext=(OV, OV)),
            ibeam(post, [W, y, hc / 2], [W, y, H - hc / 2], X, ext=(OV, OV))]


def butt(pairs):
    return [dict(type="butt", members=list(p)) for p in pairs]


def turned(spec, steps):
    """The spec with every member turned by the rotations `steps` = [(axis, degrees), ...] about the origin."""
    def r(v):
        for axis, deg in steps:
            v = mb._rot(axis, deg, v)
        return v
    out = dict(spec)
    out["members"] = [dict(m, start=mb._v(r(m["start"])), end=mb._v(r(m["end"])), up=None if m["up"] is None else mb._v(r(m["up"])))
                      for m in spec["members"]]
    return out


def specs():
    P = {}
    h30, h36, i30, i40 = h("HEA 300"), h("HEA 360"), h("IPE 300"), h("IPE 400")

    P["r01_portal_hea"] = dict(
        description="portal frame: 2 HEA 300 posts (3.5 m) under an HEA 360 girder (5 m span)",
        members=[ibeam("HEA 300", O, [0, 0, 3500], X, ext=(0, OV)), ibeam("HEA 300", [5000, 0, 0], [5000, 0, 3500], X, ext=(0, OV)),
                 ibeam("HEA 360", [-h30 / 2, 0, 3500 + h36 / 2], [5000 + h30 / 2, 0, 3500 + h36 / 2], Z)],
        joints=butt([(0, 2), (1, 2)]))

    P["r02_rect_ipe"] = dict(description="closed rectangle of 4 IPE 300 in the xz plane, 4 x 3 m, posts between the chords",
                             members=rectangle("IPE 300", "IPE 300", 4000.0, 3000.0), joints=butt([(2, 0), (2, 1), (3, 0), (3, 1)]))

    hb = h("HEB 240")
    tw = dp.PROFILES["HEB 240"].tw
    mem = rectangle("HEB 240", "HEB 240", 3600.0, 2400.0)
    # one stiffener plate per corner: in the post's web plane region, between the post's flanges, standing on the chord
    for x in (0.0, 3600.0):
        for z0, z1 in ((hb / 2 - OV, hb / 2 + 300.0), (2400.0 - hb / 2 - 300.0, 2400.0 - hb / 2 + OV)):
            for side in (-1.0,):
                yc = side * (tw / 2 + 7.5 - OV)
                mem.append(bar(200.0, 15.0, [x, yc, z0], [x, yc, z1], Y))
    P["r03_rect_heb_plates"] = dict(
        description="rectangle of 4 HEB 240 (3.6 x 2.4 m) with 4 stiffener plates 200 x 15 x 300 on the post webs at the corners",
        members=mem, joints=butt([(2, 0), (2, 1), (3, 0), (3, 1)]))

    zg = 3800 - 20 - i40 / 2
    P["r04_two_bay"] = dict(
        description="2-bay frame: 3 HEA 300 posts (3.8 m, bays 4.5 m), 2 IPE 400 girders framing into the post flanges",
        members=[ibeam("HEA 300", [x, 0, 0], [x, 0, 3800], X) for x in (0, 4500, 9000)]
        + [ibeam("IPE 400", [x0 + h30 / 2, 0, zg], [x0 + 4500 - h30 / 2, 0, zg], Z, ext=(OV, OV)) for x0 in (0, 4500)],
        joints=butt([(3, 0), (3, 1), (4, 1), (4, 2)]))

    hp, hg = h("HEB 200"), h("IPE 270")
    mem = [ibeam("HEB 200", [x, 0, 0], [x, 0, 6000], X) for x in (0, 4000, 8000)]
    joints = []
    for z in (3000 - hg / 2, 6000 - hg / 2 - 10):
        for k, x0 in enumerate((0, 4000)):
            mem.append(ibeam("IPE 270", [x0 + hp / 2, 0, z], [x0 + 4000 - hp / 2, 0, z], Z, ext=(OV, OV)))
            joints += butt([(len(mem) - 1, k), (len(mem) - 1, k + 1)])
    P["r05_grid_2x2"] = dict(description="2 bays x 2 storeys: 3 HEB 200 posts (6 m), 4 IPE 270 girders (bays 4 m, storeys 3 m)",
                             members=mem, joints=joints)

    mem = [rhs(120, 80, 6, [0, 0, -60], [0, 0, 4060], X), rhs(120, 80, 6, [900, 0, -60], [900, 0, 4060], X)]
    joints = []
    for k in range(5):
        z = 400.0 + 800.0 * k
        mem.append(rhs(80, 60, 5, [60, 0, z], [840, 0, z], Z, ext=(OV, OV)))
        joints += [dict(type="welded", members=[len(mem) - 1, 0]), dict(type="welded", members=[len(mem) - 1, 1])]
    P["r06_ladder_rhs"] = dict(description="ladder: 2 RHS 120 x 80 x 6 chords (4.1 m, 0.9 m apart), 5 RHS 80 x 60 x 5 rungs",
                               members=mem, joints=joints)

    W, H = 3000.0, 2000.0
    P["r07_rect_rhs"] = dict(
        description="rectangle of 4 RHS 200 x 100 x 8 (3 x 2 m): chords over the posts, every member with its own closed cavity",
        members=[rhs(200, 100, 8, [-100, 0, 0], [W + 100, 0, 0], Z), rhs(200, 100, 8, [-100, 0, H], [W + 100, 0, H], Z),
                 rhs(200, 100, 8, [0, 0, 100], [0, 0, H - 100], X, ext=(OV, OV)),
                 rhs(200, 100, 8, [W, 0, 100], [W, 0, H - 100], X, ext=(OV, OV))],
        joints=butt([(2, 0), (2, 1), (3, 0), (3, 1)]))

    P["r08_rect_chs"] = dict(
        description="rectangle of 4 CHS tubes (3 x 2 m): chords 139.7 x 6 along x, posts 114.3 x 5 between them, bores closed at the joints",
        members=[tube(139.7, 6, [-57.15, 0, 0], [W + 57.15, 0, 0], inner_ext=(-10, -10)),
                 tube(139.7, 6, [-57.15, 0, H], [W + 57.15, 0, H], inner_ext=(-10, -10)),
                 tube(114.3, 5, [0, 0, 69.85], [0, 0, H - 69.85], ext=(20, 20), inner_ext=(-10, -10)),
                 tube(114.3, 5, [W, 0, 69.85], [W, 0, H - 69.85], ext=(20, 20), inner_ext=(-10, -10))],
        joints=[dict(type="welded", members=[2, 0]), dict(type="welded", members=[2, 1]),
                dict(type="welded", members=[3, 0]), dict(type="welded", members=[3, 1])])

    zt = 3000 + i30 / 2
    hb2 = h("HEA 240")
    P["r09_mixed_sections"] = dict(
        description="HEA 240 posts (3 m, 4 m apart) under an IPE 300 girder, two flat bars 100 x 12 as a V bracing in the panel",
        members=[ibeam("HEA 240", O, [0, 0, 3000], X, ext=(0, OV)), ibeam("HEA 240", [4000, 0, 0], [4000, 0, 3000], X, ext=(0, OV)),
                 ibeam("IPE 300", [-hb2 / 2, 0, zt], [4000 + hb2 / 2, 0, zt], Z),
                 bar(100, 12, [hb2 / 2, 0, 150], [2000, 0, 3000], Y, ext=(OV + 20, OV + 20)),
                 bar(100, 12, [4000 - hb2 / 2, 0, 150], [2000, 0, 3000], Y, ext=(OV + 20, OV + 20))],
        joints=butt([(0, 2), (1, 2)]) + [dict(type="welded", members=[3, 0]), dict(type="welded", members=[3, 2]),
                                          dict(type="welded", members=[4, 1]), dict(type="welded", members=[4, 2])])

    P["r10_braced_portal"] = dict(
        description="portal of HEA 300 posts (3.5 m, 5 m apart) and an HEA 300 girder with two crossing rod braces d 30 in parallel planes",
        members=[ibeam("HEA 300", O, [0, 0, 3500], X, ext=(0, OV)), ibeam("HEA 300", [5000, 0, 0], [5000, 0, 3500], X, ext=(0, OV)),
                 ibeam("HEA 300", [-h30 / 2, 0, 3500 + h30 / 2], [5000 + h30 / 2, 0, 3500 + h30 / 2], Z),
                 rod(30, [0, -40, 200], [5000, -40, 3300], ext=(20, 20)), rod(30, [0, 40, 3300], [5000, 40, 200], ext=(20, 20))],
        joints=butt([(0, 2), (1, 2)]) + [dict(type="welded", members=[3, 0]), dict(type="welded", members=[3, 1]),
                                          dict(type="welded", members=[4, 0]), dict(type="welded", members=[4, 1])])

    i24 = h("IPE 240")
    twi = dp.PROFILES["IPE 240"].tw
    mem, joints = [], []
    for yl in (0.0, 800.0):
        i0 = len(mem)
        mem += rectangle("IPE 240", "IPE 240", 3000.0, 2000.0, y=yl)
        joints += butt([(i0 + 2, i0), (i0 + 2, i0 + 1), (i0 + 3, i0), (i0 + 3, i0 + 1)])
    for x, post in ((0.0, 2), (3000.0, 3)):
        for z in (500.0, 1500.0):
            mem.append(ibeam("IPE 160", [x, twi / 2, z], [x, 800.0 - twi / 2, z], Z, ext=(OV, OV)))
            joints += butt([(len(mem) - 1, post), (len(mem) - 1, 4 + post)])
    P["r11_double_layer"] = dict(
        description="two rectangles of IPE 240 (3 x 2 m) 0.8 m apart, joined by 4 IPE 160 cross members between the post webs",
        members=mem, joints=joints)

    # the benchmark's layout at full size: two planar layers 700 mm apart, HEA 500 chords along y, posts along z with a
    # 12 mm web; between the layers 4 plates 23 x 350 between the posts and 8 plates 20 x 444 between the chord flanges
    h5 = h("HEA 500")
    tf5 = dp.PROFILES["HEA 500"].tf
    post = dict(tw=12.0)
    LY, LZ, GAP = 6700.0, 4980.0, 700.0
    yp, zc = (h36 / 2, LY - h36 / 2), (h5 / 2, LZ - h5 / 2)
    mem, joints = [], []
    for xl in (0.0, GAP):
        i0 = len(mem)
        mem += [ibeam("HEA 360", [xl, y, h5 - OV], [xl, y, LZ - h5 + OV], Y, dims=post) for y in yp]
        mem += [ibeam("HEA 500", [xl, 0, z], [xl, LY, z], Z) for z in zc]
        joints += butt([(i0 + k, i0 + 2 + j) for k in (0, 1) for j in (0, 1)])
    for y in yp:
        for z in (h5 + 600.0, LZ - h5 - 600.0):
            mem.append(bar(350.0, 23.0, [6.0 - OV, y, z], [GAP - 6.0 + OV, y, z], Z))
    hw = h5 - 2 * tf5
    for z in zc:
        for y in (1700.0, 2800.0, 3900.0, 5000.0):
            mem.append(bar(hw + 2 * OV, 20.0, [6.0 - OV, y, z], [GAP - 6.0 + OV, y, z], Y))
    P["r12_benchmark_like"] = dict(
        description="the benchmark's layout at full size (6.7 x 4.98 m, two layers 0.7 m apart): 4 HEA 500 chords, 4 posts "
                    "350 x 300 x 12 x 17.5, 4 plates 350 x 23 and 8 plates 444 x 20 between the layers",
        members=mem, joints=joints)

    P["r13_rotated_rect"] = dict(turned(dict(members=rectangle("IPE 300", "IPE 300", 4000.0, 3000.0)), [(X, 30), (Y, 20)]),
                                 description="r02 turned 30 deg about x, then 20 deg about y: no member along a world axis",
                                 joints=butt([(2, 0), (2, 1), (3, 0), (3, 1)]))

    e = i30 / 2 + 10
    n = mb._unit
    P["r14_mitred_rect"] = dict(
        description="rectangle of 4 IPE 300 (4 x 3 m, centre lines) with 45 deg mitres at all four corners",
        members=[ibeam("IPE 300", [0, 0, 0], [4000, 0, 0], Z, ext=(e, e)), ibeam("IPE 300", [0, 0, 3000], [4000, 0, 3000], Z, ext=(e, e)),
                 ibeam("IPE 300", [0, 0, 0], [0, 0, 3000], X, ext=(e, e)), ibeam("IPE 300", [4000, 0, 0], [4000, 0, 3000], X, ext=(e, e))],
        joints=[dict(type="mitre", members=[0, 2], angle_deg=45), dict(type="mitre", members=[0, 3], angle_deg=45),
                dict(type="mitre", members=[1, 2], angle_deg=45), dict(type="mitre", members=[1, 3], angle_deg=45)],
        # member i keeps the side of each of its planes opposite to `normal`; the planes overlap the neighbour by 1 mm
        mitres=[dict(member=0, point=[-0.5, 0, 0.5], normal=n([-1, 0, 1])), dict(member=0, point=[4000.5, 0, 0.5], normal=n([1, 0, 1])),
                dict(member=1, point=[-0.5, 0, 2999.5], normal=n([-1, 0, -1])), dict(member=1, point=[4000.5, 0, 2999.5], normal=n([1, 0, -1])),
                dict(member=2, point=[0.5, 0, -0.5], normal=n([1, 0, -1])), dict(member=2, point=[0.5, 0, 3000.5], normal=n([1, 0, 1])),
                dict(member=3, point=[3999.5, 0, -0.5], normal=n([-1, 0, -1])), dict(member=3, point=[3999.5, 0, 3000.5], normal=n([-1, 0, 1]))])

    h2 = h("HEA 200")
    nodes = [[x, y, z] for x in (0, 3000) for y in (0, 2000) for z in (0, 2500)]
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if sum(1 for i in range(3) if nodes[a][i] != nodes[b][i]) == 1]
    mem = []
    for a, b in edges:
        k = next(i for i in range(3) if nodes[a][i] != nodes[b][i])
        if k == 2:                                                   # posts run through, node to node
            mem.append(ibeam("HEA 200", nodes[a], nodes[b], X, ext=(h2 / 2, h2 / 2)))
        else:                                                        # girders between the posts' faces
            s, t = list(nodes[a]), list(nodes[b])
            s[k] += h2 / 2
            t[k] -= h2 / 2
            mem.append(ibeam("HEA 200", s, t, Z, ext=(OV, OV)))
    P["r15_space_frame_hea"] = dict(
        description="the 12 edges of a 3 x 2 x 2.5 m box: 4 HEA 200 posts, 8 HEA 200 girders between them",
        members=mem, joints=[dict(type="node", members=[i for i, (a, b) in enumerate(edges) if nn in (a, b)]) for nn in range(8)])

    for s in P.values():
        s.setdefault("joints", [])
        s.setdefault("features", [])
    return P


def bodies(spec):
    """One solid per member, as it stands in the assembly: mitred where it is mitred, hollow where it is hollow."""
    out = []
    for i, m in enumerate(spec["members"]):
        o, c = mb.member_solids(m)
        for mt in spec.get("mitres", []):
            if mt["member"] == i:
                o = o.cut(mb._halfspace(mt["point"], mt["normal"])).Solids[0]
        if c is not None:
            o = o.cut(c)
            o = o.Solids[0] if len(o.Solids) == 1 else o
        out.append(o)
    return out


def main():
    import Part
    from src.ibeam_freecad import export_stl
    os.makedirs(OUT, exist_ok=True)
    only = [s for s in os.environ.get("FRAME_REVIEW_ONLY", "").split(",") if s]
    status = {}
    for name, spec in specs().items():
        if only and name not in only:
            continue
        try:
            union = mb.build_part(spec)
            parts = bodies(spec)
        except Exception as exc:
            status[name] = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
            print("%-22s BUILD FAILED: %s" % (name, exc))
            continue
        compound = Part.makeCompound(parts)
        union.exportStep(os.path.join(OUT, name + "_union.step"))
        facets = export_stl(union, os.path.join(OUT, name + "_union.stl"), STL_DEFLECTION)
        compound.exportStep(os.path.join(OUT, name + "_multibody.step"))
        facets_mb = export_stl(compound, os.path.join(OUT, name + "_multibody.stl"), STL_DEFLECTION)
        bb = union.BoundBox
        kinds = {}
        for m in spec["members"]:
            kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
        gt = dict(name=name, description=spec["description"], units="mm", members=spec["members"], joints=spec["joints"],
                  features=spec["features"], mitres=spec.get("mitres", []), expected_program=mb.expected_program(spec),
                  member_kinds=kinds,
                  union=dict(file=name + "_union.step", stl=name + "_union.stl", volume_mm3=float(union.Volume), area_mm2=float(union.Area),
                             valid=bool(union.isValid()), n_solids=len(union.Solids), faces=len(union.Faces), stl_facets=facets),
                  multibody=dict(file=name + "_multibody.step", stl=name + "_multibody.stl", n_bodies=len(parts), stl_facets=facets_mb,
                                 bodies=[dict(member=i, volume_mm3=float(p.Volume), valid=bool(p.isValid()), n_solids=len(p.Solids),
                                              faces=len(p.Faces)) for i, p in enumerate(parts)],
                                 sum_volume_mm3=float(sum(p.Volume for p in parts))),
                  bbox_min=[bb.XMin, bb.YMin, bb.ZMin], bbox_max=[bb.XMax, bb.YMax, bb.ZMax], diagonal_mm=float(bb.DiagonalLength),
                  stl_deflection_mm=STL_DEFLECTION)
        with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(gt, fh, indent=1)
        ok = gt["union"]["valid"] and gt["union"]["n_solids"] == 1 and all(b["valid"] and b["n_solids"] == 1 for b in gt["multibody"]["bodies"])
        status[name] = dict(ok=ok, members=len(parts), kinds=kinds, volume_mm3=gt["union"]["volume_mm3"], union_solids=gt["union"]["n_solids"],
                            union_valid=gt["union"]["valid"], union_faces=gt["union"]["faces"], expected_ops=gt["expected_program"]["total"],
                            diagonal_mm=gt["diagonal_mm"])
        print("%-22s %2d members %-28s union: %d solid(s), %s, %d faces, %.4g mm^3; bodies valid: %s; expected %d ops"
              % (name, len(parts), str(kinds), gt["union"]["n_solids"], "valid" if gt["union"]["valid"] else "INVALID", gt["union"]["faces"],
                 union.Volume, all(b["valid"] for b in gt["multibody"]["bodies"]), gt["expected_program"]["total"]))
    with open(os.path.join(OUT, "_build_status.json"), "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=1)


try:
    import FreeCAD  # noqa: F401
    if __name__ in ("__main__", "make_frame_review_set"):
        main()
except ImportError:
    pass
