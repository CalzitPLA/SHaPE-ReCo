#!/usr/bin/env python3
"""
The exact B-rep face inventory of a STEP file: surface kinds, radii and axis directions of the cylindrical faces, and which
faces carry an inner wire (a hole punched in that face - a bolt hole through a plate, or the entry/exit of a bridging rod).
Needs `freecadcmd` (uses FreeCAD's `Part` module directly, no fitting). Case study: `docs/FRAME_FAILURE.md`, where this
distinguished "160 identical bolt holes" (wrong) from "160 solid round rods bridging the frame" (right, confirmed by the
outward-normal check below).

    freecadcmd scripts/inspect_step_faces.py part.stp
"""
import collections
import sys

import Part


def material_is_outward(face) -> bool:
    """True if material lies on the +normal side of `face` (a hole/cavity wall, seen from inside it); False if material
    lies on the -normal side (the outer skin of a solid protrusion - a rod, a boss). Steps a tiny distance along the
    face's own normal, at its mid-parameter point, and asks the whole shape whether that point is still inside it."""
    u0, u1, v0, v1 = face.ParameterRange
    um, vm = (u0 + u1) / 2, (v0 + v1) / 2
    p, n = face.valueAt(um, vm), face.normalAt(um, vm)
    return bool(_shape.isInside(p + n * 0.5, 1e-3, True))


def main(path: str):
    global _shape
    _shape = shape = Part.read(path)
    print(f"Solids: {len(shape.Solids)}  Faces: {len(shape.Faces)}  Edges: {len(shape.Edges)}  Vertexes: {len(shape.Vertexes)}")
    kinds = collections.Counter(f.Surface.__class__.__name__ for f in shape.Faces)
    print("Face surface kinds:", dict(kinds))

    cyl = [f for f in shape.Faces if f.Surface.__class__.__name__ == "Cylinder"]
    radii = collections.Counter(round(f.Surface.Radius, 1) for f in cyl)
    print(f"Cylinder faces: {len(cyl)}, radius histogram (top 15): {dict(sorted(radii.items(), key=lambda kv: -kv[1])[:15])}")
    axes = collections.Counter((round(abs(f.Surface.Axis.x), 2), round(abs(f.Surface.Axis.y), 2),
                                round(abs(f.Surface.Axis.z), 2)) for f in cyl)
    print("Cylinder axis-direction histogram (|x|,|y|,|z|):", dict(axes))
    lengths = [max(f.BoundBox.XLength, f.BoundBox.YLength, f.BoundBox.ZLength) for f in cyl]
    if lengths:
        lengths.sort()
        print(f"Cylinder face length along its own extent: min {lengths[0]:.0f}  median {lengths[len(lengths) // 2]:.0f}"
             f"  max {lengths[-1]:.0f} mm")

    holes = [len(f.Wires) - 1 for f in shape.Faces if len(f.Wires) > 1]
    print(f"Faces with >=1 inner wire (a hole punched in that face): {len(holes)}, total inner wires: {sum(holes)}")

    sample = cyl[::max(len(cyl) // 40, 1)][:40]                                           # a spread-out sample is enough
    outward = collections.Counter(material_is_outward(f) for f in sample)
    print(f"Sampled {len(sample)} cylinder faces: material on the +normal side "
         "(True = a hole/cavity wall, False = a solid rod/boss):", dict(outward))


# freecadcmd does not run this module as "__main__" (its __name__ is the script's own basename), so - matching this
# project's other freecadcmd-executed scripts (reco_field/freecad_builder.py, scripts/sdf_reference_freecad.py) - call
# main() unconditionally rather than guarding it. sys.argv under freecadcmd is
# ["freecadcmd", "inspect_step_faces.py", <this script's own args>...].
_args = [a for a in sys.argv[2:] if not a.startswith("-")]
main(_args[0])
