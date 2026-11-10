"""
Regenerate the small test parts in dataset_gen/examples/ (run with
freecadcmd, which provides FreeCAD/Part):

    freecadcmd scripts/make_test_parts.py

Test parts for the cut phase and the octree strategy, all built from a 20 mm
cube or a 20 mm cylinder with material taken out (box_with_bore.stl, an 8 mm
through-bore, predates this script and is kept as committed):

  two_bores.stl      6 mm bore along Z and a 5 mm bore along X (crossing)
  corner_pocket.stl  9 x 9 x 12 mm blind pocket in one corner
  blind_hole.stl     8 mm blind hole, 12 mm deep, from the top face
  side_slot.stl      4 x 12 mm slot through the full height, open to one side
  l_bracket.stl      quarter of the cube removed over the full height (L profile)
  tube.stl           cylinder with a concentric 10 mm bore
  cross_hole.stl     cylinder with a 6 mm transverse hole
  three_holes.stl    three parallel bores of different diameters
  cavity.stl         closed 8 mm cube cavity inside the cube
  plate_4holes.stl   20 x 20 x 6 plate with four 4 mm holes

Used by scripts/compare_strategies.py.

Cylinder parts (prefix cy_): cy_cyl (R10 x 20), cy_cyl_long (R6 x 30), cy_on_box (cylinder on a slab), cy_box_on_cyl (box on a cylinder),
cy_flat (cylinder with a flat side), cy_t (cylinder with a sideways stub), cy_box_bore (cube with a 12 mm through-bore).

Trained-shapes-only parts (prefix bs_): built ONLY from boxes and spheres - the
two topologies the networks were actually trained on - as unions and
differences, so the true decomposition is known and exactly representable.
When a reconstruction of one of these goes wrong, the pipeline is at fault, not
a missing primitive:

  bs_box              20 x 16 x 12 mm box (control)
  bs_sphere           20 mm sphere (control)
  bs_box_dome         box with a sphere sitting on its top face (union)
  bs_box_step         box with a smaller box on top (union)
  bs_box_sphere_pocket  box with a spherical pocket in the top face (difference)
  bs_box_two_pockets  box with two spherical pockets in the top face
  bs_box_sphere_cavity  box with a closed spherical cavity inside
  bs_sphere_flat      sphere with the top cut flat by a box

(corner_pocket, side_slot, l_bracket and cavity above are box-only as well.)

Cone, torus and segment parts (prefixes co_, to_, cs_): the primitives of the vector-field reconstruction that have a circle, with
the sweep angle u (and the latitude / minor-circle range v) as a parameter:

  co_cone / co_cone_apex   frustum R10 -> R3 and a full cone (R10 -> 0), 20 mm high
  co_countersink           cube with a 4 mm bore and a conical countersink (R2 -> R8.4 over the top 6 mm)
  to_torus                 torus R8 / r3
  to_ring_groove           cylinder R10 with a circular groove (torus R10 / r2.5) around its waist
  to_elbow                 a 90 degree segment of a torus R10 / r3 (a bent pipe: u = (0, pi/2), planar annulus ends)
  cs_cyl_sector            cylinder R10, 20 mm high, sector u = 120 degrees
  cs_cone_sector           frustum R10 -> R4, sector u = 200 degrees
  cs_sphere_sector         sphere R10, u = 270 degrees
  cs_sphere_band           sphere R10 between the latitudes -20 and 90 degrees (FreeCAD's latitude = planes)
  cs_torus_half            half a torus R9 / r3 (u = 180 degrees)
"""
import os

import FreeCAD as App
import Part

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_gen", "examples")

box = Part.makeBox(20, 20, 20, App.Vector(-10, -10, -10))

two_bores = (box.cut(Part.makeCylinder(3.0, 30, App.Vector(0, 0, -15), App.Vector(0, 0, 1)))
                .cut(Part.makeCylinder(2.5, 30, App.Vector(-15, 0, 0), App.Vector(1, 0, 0))))

corner_pocket = box.cut(Part.makeBox(9, 9, 12, App.Vector(1, 1, -2)))

parts = {"two_bores": two_bores, "corner_pocket": corner_pocket}


def cyl(radius, height, base, direction=App.Vector(0, 0, 1)):
    return Part.makeCylinder(radius, height, base, direction)


parts["blind_hole"] = box.cut(cyl(4.0, 12, App.Vector(0, 0, -2)))
parts["side_slot"] = box.cut(Part.makeBox(4, 12, 30, App.Vector(-2, -2, -15)))
parts["l_bracket"] = box.cut(Part.makeBox(10, 10, 30, App.Vector(0, 0, -15)))
cylinder = cyl(10.0, 20, App.Vector(0, 0, -10))
parts["tube"] = cylinder.cut(cyl(5.0, 30, App.Vector(0, 0, -15)))
parts["cross_hole"] = cylinder.cut(cyl(3.0, 30, App.Vector(-15, 0, 0), App.Vector(1, 0, 0)))
three = box
for x, r in ((-6, 2.0), (0, 3.0), (6, 2.0)):
    three = three.cut(cyl(r, 30, App.Vector(x, 0, -15)))
parts["three_holes"] = three
parts["cavity"] = box.cut(Part.makeBox(8, 8, 8, App.Vector(-4, -4, -4)))
plate = Part.makeBox(20, 20, 6, App.Vector(-10, -10, -3))
for x in (-5, 5):
    for y in (-5, 5):
        plate = plate.cut(cyl(2.0, 10, App.Vector(x, y, -5)))
parts["plate_4holes"] = plate

def sphere(radius, x, y, z):
    return Part.makeSphere(radius, App.Vector(x, y, z))


tall = Part.makeBox(20, 16, 12, App.Vector(-10, -8, -6))
parts["bs_box"] = tall
parts["bs_sphere"] = sphere(10.0, 0, 0, 0)
parts["bs_box_dome"] = tall.fuse(sphere(6.0, 0, 0, 6)).removeSplitter()
parts["bs_box_step"] = tall.fuse(Part.makeBox(10, 8, 8, App.Vector(-5, -4, 6))).removeSplitter()
parts["bs_box_sphere_pocket"] = box.cut(sphere(6.0, 0, 0, 10))
parts["bs_box_two_pockets"] = box.cut(sphere(4.0, -5, 0, 10)).cut(sphere(4.0, 5, 0, 10))
parts["bs_box_sphere_cavity"] = box.cut(sphere(5.5, 0, 0, 0))
parts["bs_sphere_flat"] = sphere(10.0, 0, 0, 0).cut(Part.makeBox(30, 30, 10, App.Vector(-15, -15, 5)))

# cylinder parts (prefix cy_): a cylinder as base or as add-on, with a flat, and a bore through a box - built from boxes,
# spheres and cylinders only, so the true decomposition is known
parts["cy_cyl"] = cyl(10.0, 20, App.Vector(0, 0, -10))
parts["cy_cyl_long"] = cyl(6.0, 30, App.Vector(0, 0, -15))
parts["cy_on_box"] = Part.makeBox(24, 24, 10, App.Vector(-12, -12, -15)).fuse(cyl(7.0, 14, App.Vector(0, 0, -5))).removeSplitter()
parts["cy_box_on_cyl"] = cyl(9.0, 16, App.Vector(0, 0, -14)).fuse(Part.makeBox(10, 10, 8, App.Vector(-5, -5, 2))).removeSplitter()
parts["cy_flat"] = cyl(10.0, 20, App.Vector(0, 0, -10)).cut(Part.makeBox(10, 30, 30, App.Vector(6, -15, -15)))
parts["cy_t"] = cyl(10.0, 20, App.Vector(0, 0, -10)).fuse(cyl(4.0, 14, App.Vector(0, 0, 0), App.Vector(1, 0, 0))).removeSplitter()
parts["cy_box_bore"] = box.cut(cyl(6.0, 30, App.Vector(0, 0, -15)))

# cone, torus and segment parts (co_, to_, cs_)
Z = App.Vector(0, 0, 1)
parts["co_cone"] = Part.makeCone(10.0, 3.0, 20, App.Vector(0, 0, -10), Z)
parts["co_cone_apex"] = Part.makeCone(10.0, 0.0, 20, App.Vector(0, 0, -10), Z)
parts["co_countersink"] = box.cut(cyl(2.0, 30, App.Vector(0, 0, -15))).cut(Part.makeCone(2.0, 9.4, 7, App.Vector(0, 0, 4), Z))
parts["to_torus"] = Part.makeTorus(8.0, 3.0, App.Vector(0, 0, 0), Z)
parts["to_ring_groove"] = cyl(10.0, 20, App.Vector(0, 0, -10)).cut(Part.makeTorus(10.0, 2.5, App.Vector(0, 0, 0), Z))
parts["to_elbow"] = Part.makeTorus(10.0, 3.0, App.Vector(0, 0, 0), Z, 0, 360, 90)
parts["cs_cyl_sector"] = Part.makeCylinder(10.0, 20, App.Vector(0, 0, -10), Z, 120)
parts["cs_cone_sector"] = Part.makeCone(10.0, 4.0, 20, App.Vector(0, 0, -10), Z, 200)
parts["cs_sphere_sector"] = Part.makeSphere(10.0, App.Vector(0, 0, 0), Z, -90, 90, 270)
parts["cs_sphere_band"] = Part.makeSphere(10.0, App.Vector(0, 0, 0), Z, -20, 90, 360)
parts["cs_torus_half"] = Part.makeTorus(9.0, 3.0, App.Vector(0, 0, 0), Z, 0, 360, 180)

# hollow (shell) parts: a box or sphere minus a smaller, CONCENTRIC copy of itself (a closed cavity, not a bite off one side).
# The right decomposition is base = the outer shape (recall 1.0: it CONTAINS the whole shell, at the cost of a lot of spurious
# interior "material" - a bad material budget) then cut = the inner shape (the cavity). Used to test that acceptance.
parts["bs_hollow_box"] = tall.cut(Part.makeBox(16, 12, 8, App.Vector(-8, -6, -4)))                        # 20x16x12 shell, wall 2 mm
parts["bs_hollow_sphere"] = sphere(10.0, 0, 0, 0).cut(sphere(7.0, 0, 0, 0))                                # R10 shell, wall 3 mm
parts["bs_hollow_box_thin"] = tall.cut(Part.makeBox(19, 15, 11, App.Vector(-9.5, -7.5, -5.5)))             # wall 0.5 mm

for name, shape in parts.items():
    shape.exportStl(os.path.join(OUT, name + ".stl"))
    print("%-14s volume %.1f mm^3, %s" % (name, shape.Volume, "solid" if shape.isValid() else "INVALID"))
