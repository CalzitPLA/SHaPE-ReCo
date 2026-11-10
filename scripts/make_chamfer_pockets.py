"""Blocks with a box-like cut-out whose edges carry chamfers or fillets, for the chamfer study (freecadcmd).

    POCKETS_OUT=<dir> freecadcmd scripts/make_chamfer_pockets.py

Block 60 x 40 x 30 mm, pocket 30 x 20 x 15 mm cut from the top face, centred. Variants (size s in mm):
  plain            no finished edge
  rim_chamfer_s    the 4 edges of the pocket's mouth chamfered (convex edges: material is removed)
  corner_chamfer_s the 4 vertical edges of the pocket chamfered (concave edges: material is left standing)
  corner_fillet_s  the same 4 edges rounded with radius s
Writes <name>.step, <name>.stl and inventory.json: exact volume, faces by surface type, edges, vertices, and the exact
volumes of the common and the union with the plain part (the model that ignores the finished edges).
"""
import collections
import json
import os

import FreeCAD as App
import Mesh
import MeshPart
import Part

OUT = os.environ["POCKETS_OUT"]
L, W, H = 60.0, 40.0, 30.0
l, w, d = 30.0, 20.0, 15.0
SIZES = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
x0, y0, z0 = (L - l) / 2, (W - w) / 2, H - d


def pocket_tool(kind, s):
    tool = Part.makeBox(l, w, d + 5.0, App.Vector(x0, y0, z0))          # 5 mm past the top face
    vertical = [e for e in tool.Edges if abs(e.Vertexes[0].Point.z - e.Vertexes[1].Point.z) > 1.0]
    if kind == "corner_chamfer":
        tool = tool.makeChamfer(s, vertical)
    elif kind == "corner_fillet":
        tool = tool.makeFillet(s, vertical)
    return tool


def part(kind, s):
    block = Part.makeBox(L, W, H)
    if kind in ("plain", "corner_chamfer", "corner_fillet"):
        return block.cut(pocket_tool(kind, s))
    body = block.cut(pocket_tool("plain", 0.0))
    rim = [e for e in body.Edges if all(abs(v.Point.z - H) < 1e-6 for v in e.Vertexes)
           and all(x0 - 1e-6 <= v.Point.x <= x0 + l + 1e-6 and y0 - 1e-6 <= v.Point.y <= y0 + w + 1e-6 for v in e.Vertexes)]
    assert len(rim) == 4
    return body.makeChamfer(s, rim)


plain = part("plain", 0.0)
rows = []
for kind, s in [("plain", 0.0)] + [(k, s) for k in ("rim_chamfer", "corner_chamfer", "corner_fillet") for s in SIZES]:
    name = kind if kind == "plain" else "%s_%g" % (kind, s)
    shape = part(kind, s)
    assert shape.isValid() and len(shape.Solids) == 1
    shape.exportStep(os.path.join(OUT, name + ".step"))
    mesh = MeshPart.meshFromShape(Shape=shape, LinearDeflection=0.01, AngularDeflection=0.05)
    mesh.write(os.path.join(OUT, name + ".stl"))
    common, union = shape.common(plain).Volume, shape.fuse(plain).Volume
    rows.append(dict(name=name, kind=kind, size=s, volume=shape.Volume, faces=len(shape.Faces), edges=len(shape.Edges),
                     vertices=len(shape.Vertexes),
                     surface_types=dict(collections.Counter(type(f.Surface).__name__ for f in shape.Faces)),
                     common_with_plain=common, union_with_plain=union, iou_with_plain=common / union))
json.dump(dict(block=[L, W, H], pocket=[l, w, d], sizes=SIZES, parts=rows), open(os.path.join(OUT, "inventory.json"), "w"), indent=1)
