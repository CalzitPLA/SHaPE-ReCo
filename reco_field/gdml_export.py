#!/usr/bin/env python3
"""
GDML export (the user's request 2026-09-29: "create a gdml-export function"): the reconstruction - a tree of primitives
and booleans - written as a GDML file for Geant4 and ROOT. The tree maps one to one: GDML's solids are the same
constructive solid geometry.

    from reco_field.gdml_export import export_gdml
    info = export_gdml(model, "part.gdml", name="liner", material=STEEL)      # model: an SDF tree (src.sdf_primitives)
    info = export_result(result, "part.gdml")                               # a ReconstructionResult of the loop

The mapping (lengths mm, angles rad):

  Box(half)                           box x y z = 2 half, centred
  Cylinder(radius, height, u, inner)  tube rmin rmax z startphi deltaphi; ours stands on z = 0, GDML's is centred:
                                      its position is moved by height / 2 along its axis
  Cone(radius1, radius2, height, u)   cone rmax1 (at -z/2) rmax2 (at +z/2), centred: moved by height / 2 as the tube
  Sphere, full                        orb r (hollow: sphere rmin rmax); a segment with latitude "cone" is GDML's sphere
                                      with starttheta / deltatheta (theta the polar angle: theta = pi/2 - latitude);
                                      latitude "plane" (FreeCAD's makeSphere) is the ball intersected with a slab
  Torus(major, minor, u)              torus rtor rmax startphi deltaphi; a segment of the tube (v) has no GDML solid
  HalfSpace(normal, offset)           a box of the size `big` on the -normal side of its plane
  Union / Intersection / Subtraction  union / intersection / subtraction of two solids, the second one placed in the
                                      frame of the first by position and rotation
  a union of coaxial cylinders/cones  ONE polycone (z planes with their radius): a hole stack, a turned contour. The
                                      union of solids of revolution about one axis is the solid of their largest
                                      radius at every z. 2026-09-29: FreeCAD's GDML workbench (1.9b) rebuilt the
                                      liner's file with OCCT's plain booleans and failed on 60 of its unions and
                                      cuts - coaxial solids fused end to end share a circle; Geant4 and ROOT do not
                                      build a boundary and read both forms alike. `polycones=False` writes the
                                      unions as they are.
  Placed(primitive, R, t)             no solid of its own: the pose goes into the boolean that uses the solid, the
                                      pose of the whole tree into the physical volume

TREE SHAPES (`tree=`; 2026-10-01, T-30). The same solid can be written as differently shaped trees, and Geant4's
navigation time depends on the shape. In aegir's straight-line scan (T-28: 593 reconstructed programs in 8 shapes,
docs/reports/real_trees/real_trees.tex) grouped cuts with balanced unions and grouped cuts with multiUnions both took a
median 0.58 of the navigation time of the program's own tree, and were faster on 79 % and 78 % of the programs.

  "grouped-balanced" (default)  every chain of cuts ((A - B) - C) - ... as A - (B u C u ...), and every tree of unions
                                rebuilt from its operands, halved recursively (depth log2 n instead of n)
  "grouped-multi"               the cuts grouped as above, and a tree of more than two unions whose operands are all
                                primitives as ONE multiUnion (G4MultiUnion; a node is placed like a boolean's second
                                solid: measured in aegir, docs/sandbox_geometries.md there); a union with a composite
                                operand is balanced. Geant4 only: ROOT 6.40 (TGDMLParse) does not read multiUnion
  "as-is"                       the program's own tree, node for node (the export before 2026-10-01)

Grouping and regrouping change the tree, not the solid: a cut by B, then by C removes B u C, and a union does not mind
the order. Measured 2026-10-01 in aegir's scan (Geant4, server_b) on the liner's 64-operation program (416 primitives):
200 000 lines enter the three shapes in the same events at the same points (86 918 entries, largest distance
7.5e-5 mm), and no entry lies off the program's exact surface by more than 1 um; ROOT 6.40 judges 20 000 points of
the as-is and the grouped-balanced file as the program does. The cuts are grouped before the coaxial unions become
polycones, so that a hole stack cut step by step is one polycone tool.

ROTATIONS. Geant4 reads a GDML rotation (x, y, z) as the matrix G = Rz(z) Ry(y) Rx(x) (G4GDMLRead::GetRotationMatrix)
and uses it in two different ways, read in its source on 2026-09-29 and confirmed with ROOT 6.38 and its TGeoManager
on the files this module writes:

  in a BOOLEAN solid   the second solid is placed by G itself: G4Transform3D(GetRotationMatrix(rotation), position)
                       (G4GDMLReadSolids::BooleanRead)
  in a PHYSICAL volume the volume is placed by the INVERSE of G: G4Transform3D(GetRotationMatrix(rotation).inverse(),
                       position) (G4GDMLReadStructure::PhysvolRead)
  in a MULTIUNION node like a boolean's second solid, by G: measured with Geant4 11.4.2 in aegir on an asymmetric
                       solid at generic angles (docs/sandbox_geometries.md there, "Placement conventions")

For a body whose axes in the frame of its parent are the columns of R, a boolean therefore holds the angles of R and a
physical volume those of R^T (`gdml_angles(R, placed=...)`, back: `matrix_of_angles`). ROOT's reader (TGDMLParse) does
the same: it reads every rotation with negated angles and inverts it once more for a boolean's operands.
tests/test_gdml_export.py reads the written file back with this convention and compares the solid point by point
with the tree it came from; the check against ROOT itself is scripts/gdml_check_root.py.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
if str(SANDBOX_ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402

TWO_PI = 2.0 * math.pi
SPIN = 0.7                          # rad: _Writer.spun
UNBOUNDED = 1e8                     # mm: a model larger than this is a half space alone, not a part
SCHEMA = "http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd"


@dataclass
class Material:
    """A material of one element: enough for a geometry check; a real alloy is defined in the application."""
    name: str
    Z: float
    A: float                            # g / mole
    density: float                      # g / cm3


STEEL = Material("Iron", 26.0, 55.845, 7.874)
ALUMINIUM = Material("Aluminium", 13.0, 26.982, 2.699)
VACUUM = Material("Vacuum", 1.0, 1.008, 1e-25)


def gdml_angles(R, placed: str = "physvol") -> Tuple[float, float, float]:
    """The GDML rotation (x, y, z) in rad of a body whose axes are the columns of R in the frame of its parent, for
    the place it is written to: "boolean" (the second solid of a boolean: Rz(z) Ry(y) Rx(x) = R) or "physvol" (a
    physical volume: Rz(z) Ry(y) Rx(x) = R^T); see the module docstring."""
    from scipy.spatial.transform import Rotation
    if placed not in ("boolean", "physvol"):
        raise ValueError(f"placed: 'boolean' or 'physvol', not {placed!r}")
    M = np.asarray(R, float)
    M = M.T if placed == "physvol" else M
    U, _, Vt = np.linalg.svd(M)                                       # the nearest rotation: sums of products drift
    M = U @ Vt
    if np.linalg.det(M) < 0:
        raise ValueError("a reflection is no rotation")
    import warnings
    with warnings.catch_warnings():                                   # gimbal lock: one of the solutions is as good
        warnings.simplefilter("ignore")
        x, y, z = Rotation.from_matrix(M).as_euler("xyz")             # extrinsic x, then y, then z: Rz Ry Rx
    return float(x), float(y), float(z)


def matrix_of_angles(x: float, y: float, z: float, placed: str = "physvol") -> np.ndarray:
    """R (the body's axes as columns) of a GDML rotation written to a "boolean" or a "physvol"."""
    from scipy.spatial.transform import Rotation
    G = Rotation.from_euler("xyz", [x, y, z]).as_matrix()              # Rz(z) Ry(y) Rx(x)
    return G.T if placed == "physvol" else G


def _is_full(rng, full) -> bool:
    return rng[0] <= full[0] + 1e-9 and rng[1] >= full[1] - 1e-9


FULL_TURN = repr(TWO_PI)            # "6.283185307179586": the double 2 pi itself, see _Writer.sector
HALF_TURN = repr(math.pi)           # "3.141592653589793": the double pi itself


def _span(v: float, most: float) -> float:
    """An angle span for the file: never above `most` after rounding to 9 digits (pi rounds UP to 3.141592654, which
    ROOT's TGeoSphere refuses: "invalid parameters theta1/theta2")."""
    return min(float(v), math.floor(most * 1e9) / 1e9)


def _f(v: float) -> str:
    v = float(v)
    if abs(v) < 1e-12:
        v = 0.0
    return repr(round(v, 9))


# ------------------------------------------------------------------------------------------------ coaxial unions

@dataclass
class Polycone(sp.Primitive):
    """A solid of revolution about z: the radius is linear between the z planes [(z, radius)], a step is two planes
    at one z. Only what the export needs: it is written as GDML's polycone, its SDF is the union of its pieces."""
    planes: tuple
    u: Tuple[float, float] = sp.FULL
    inner: Optional[tuple] = None       # the inner radius at every plane (a ring: what a coaxial cut leaves), None: solid

    def pieces(self) -> sp.Primitive:
        def run(z0, r0, z1, r1, grow=0.0):
            return sp.Placed(sp.Cone(float(r0), float(r1), float(z1 - z0) + 2 * grow, self.u) if abs(r1 - r0) > 1e-12
                             else sp.Cylinder(float(r0), float(z1 - z0) + 2 * grow, self.u), np.eye(3),
                             np.array([0.0, 0.0, z0 - grow]))
        out = None
        inner = self.inner or tuple(0.0 for _ in self.planes)
        for ((z0, r0), (z1, r1)), (i0, i1) in zip(zip(self.planes, self.planes[1:]), zip(inner, inner[1:])):
            if z1 - z0 <= 1e-12:
                continue
            piece = run(z0, r0, z1, r1)
            if max(i0, i1) > 0:
                piece = sp.Subtraction(piece, run(z0, i0, z1, i1))
            out = piece if out is None else sp.Union(out, piece)
        return out

    def sdf(self, p):
        return self.pieces().sdf(p)

    def bounds(self):
        return self.pieces().bounds()

    @property
    def exact(self):
        return False


def _union_leaves(p, R, t, out):
    """The operands of a tree of unions, each with its pose: whatever is no union is an operand."""
    if isinstance(p, sp.Placed):
        return _union_leaves(p.primitive, R @ np.asarray(p.rotation, float), R @ np.asarray(p.translation, float) + t, out)
    if isinstance(p, sp.Union):
        _union_leaves(p.a, R, t, out)
        return _union_leaves(p.b, R, t, out)
    out.append((p, R, t))
    return out


def _profile(members, tol: float):
    """[(z, r)] of the union of coaxial solids of revolution [(z0, z1, r0, r1)]: at every z the largest radius. None when
    they leave a gap along the axis."""
    zs = sorted({z for m in members for z in m[:2]})
    merged = [zs[0]]
    for z in zs[1:]:
        if z - merged[-1] > tol:
            merged.append(z)
    planes = []
    for za, zb in zip(merged, merged[1:]):
        live = [m for m in members if m[0] <= za + tol and m[1] >= zb - tol]
        if not live:
            return None

        def r_of(m, z):
            z0, z1, r0, r1 = m
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
        cuts = [za, zb]
        for i, a in enumerate(live):                                   # where two radii cross, the larger one changes
            for b in live[i + 1:]:
                da, db = r_of(a, zb) - r_of(a, za), r_of(b, zb) - r_of(b, za)
                if abs(da - db) > 1e-12:
                    x = za + (r_of(b, za) - r_of(a, za)) / (da - db) * (zb - za)
                    if za + tol < x < zb - tol:
                        cuts.append(x)
        cuts = sorted(cuts)
        for ca, cb in zip(cuts, cuts[1:]):
            mid = 0.5 * (ca + cb)
            top = max(live, key=lambda m: r_of(m, mid))
            planes += [(ca, max(r_of(top, ca), 0.0)), (cb, max(r_of(top, cb), 0.0))]
    slim = []
    for z, r in planes:                                                # a plane said twice, a straight run in three
        near = 1e-6 * max(1.0, abs(r))                                 # radii that differ by rounding are one radius
        if slim and abs(slim[-1][0] - z) <= 1e-12 and abs(slim[-1][1] - r) <= near:
            continue
        if len(slim) >= 2 and abs(slim[-1][1] - slim[-2][1]) <= near and abs(r - slim[-1][1]) <= near \
                and slim[-1][0] - slim[-2][0] > 1e-12 and z - slim[-1][0] > 1e-12:
            slim[-1] = (z, r)
            continue
        slim.append((float(z), float(r)))
    return slim


def _revolved(p, R=None, t=None):
    """(planes [(z, r)], R, t) of a full, solid cylinder, cone or polycone with its pose; None for anything else."""
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else t
    if isinstance(p, sp.Placed):
        return _revolved(p.primitive, R @ np.asarray(p.rotation, float), R @ np.asarray(p.translation, float) + t)
    if isinstance(p, Polycone) and p.inner is None and _is_full(p.u, (0.0, TWO_PI)):
        return list(p.planes), R, t
    if isinstance(p, sp.Cylinder) and p.inner_radius <= 0 and _is_full(p.u, (0.0, TWO_PI)):
        return [(0.0, float(p.radius)), (float(p.height), float(p.radius))], R, t
    if isinstance(p, sp.Cone) and _is_full(p.u, (0.0, TWO_PI)):
        return [(0.0, float(p.radius1)), (float(p.height), float(p.radius2))], R, t
    return None


def _radius(planes, z: float, side: int) -> Optional[float]:
    """The profile's radius just below (side -1) or above (side +1) z; None outside its reach."""
    for (z0, r0), (z1, r1) in zip(planes, planes[1:]):
        if z1 - z0 > 1e-12 and ((side > 0 and z0 - 1e-9 <= z < z1 - 1e-9) or (side < 0 and z0 + 1e-9 < z <= z1 + 1e-9)):
            return r0 + (r1 - r0) * (z - z0) / (z1 - z0)
    return None


def _ring(a, b, tol: float) -> Optional[sp.Primitive]:
    """a - b for two coaxial solids of revolution as ONE hollow Polycone: the outer radius is a's, the inner one b's
    where b reaches (a turned contour's cut ring: a cylinder minus the contour). None when they are not coaxial, or
    when b reaches a's radius somewhere (then nothing of a is left there, which a polycone cannot say)."""
    A, B = _revolved(a), _revolved(b)
    if A is None or B is None:
        return None
    (pa, Ra, ta), (pb, Rb, tb) = A, B
    axis = Ra[:, 2]
    off = tb - ta
    size = max(r for _, r in pa)
    if abs(abs(axis @ Rb[:, 2]) - 1.0) > 1e-9 or np.linalg.norm(off - axis * (off @ axis)) > tol * max(1.0, size):
        return None
    z_b = float(off @ axis)
    pb = [(z_b + z, r) for z, r in pb] if axis @ Rb[:, 2] > 0 else [(z_b - z, r) for z, r in reversed(pb)]
    lo, hi = pa[0][0], pa[-1][0]
    zs = sorted({z for z, _ in pa} | {min(max(z, lo), hi) for z, _ in pb})
    planes, inner = [], []
    for za, zb in zip(zs, zs[1:]):
        if zb - za <= 1e-9:
            continue
        for z, side in ((za, 1), (zb, -1)):
            ro, ri = _radius(pa, z, side), _radius(pb, z, side)
            if ro is None:
                return None
            ri = 0.0 if ri is None else ri
            if ri >= ro - 1e-9:
                return None
            if planes and planes[-1] == (float(z), float(ro)) and inner[-1] == float(ri):
                continue                            # the same section twice: ROOT's TGeoPcon refuses it ("Duplicated section")
            planes.append((float(z), float(ro)))
            inner.append(float(ri))
    if len(planes) < 2:
        return None
    return sp.Placed(Polycone(tuple(planes), sp.FULL, tuple(inner)), Ra, ta)


def merge_coaxial(p: sp.Primitive, tol: float = 1e-6) -> sp.Primitive:
    """The tree with every union of coaxial full cylinders and cones (a hole stack, a turned contour) as ONE Polycone.
    The operands of a tree of unions are regrouped by their axis line - a union does not mind the order; operands
    about one axis that leave no gap become a polycone, the others stay. Everything else is kept as it is."""
    if isinstance(p, sp.Placed):
        return sp.Placed(merge_coaxial(p.primitive, tol), p.rotation, p.translation)
    if isinstance(p, sp.Subtraction):
        a, b = merge_coaxial(p.a, tol), merge_coaxial(p.b, tol)
        return _ring(a, b, tol) or sp.Subtraction(a, b)
    if isinstance(p, sp.Intersection):
        return sp.Intersection(merge_coaxial(p.a, tol), merge_coaxial(p.b, tol))
    if not isinstance(p, sp.Union):
        return p
    leaves = _union_leaves(p, np.eye(3), np.zeros(3), [])
    groups, rest = [], []
    for q, R, t in leaves:
        round_ = isinstance(q, (sp.Cylinder, sp.Cone)) and (not isinstance(q, sp.Cylinder) or q.inner_radius <= 0) \
            and float(q.height) > tol
        if not round_:
            rest.append(sp.Placed(merge_coaxial(q, tol), R, t))
            continue
        a, base = R[:, 2], np.asarray(t, float)
        r0, r1 = (float(q.radius), float(q.radius)) if isinstance(q, sp.Cylinder) else (float(q.radius1), float(q.radius2))
        for g in groups:
            off = base - g["base"]
            same_u = _is_full(q.u, (0.0, TWO_PI)) and g["full"]
            if same_u and abs(abs(a @ g["axis"]) - 1.0) <= 1e-9 and \
                    np.linalg.norm(off - g["axis"] * (off @ g["axis"])) <= tol * max(1.0, r0, r1):
                z0 = float(off @ g["axis"])
                if a @ g["axis"] > 0:
                    g["members"].append((z0, z0 + float(q.height), r0, r1))
                else:
                    g["members"].append((z0 - float(q.height), z0, r1, r0))
                g["leaves"].append((q, R, t))
                break
        else:
            groups.append(dict(axis=a, base=base, R=R, full=_is_full(q.u, (0.0, TWO_PI)), u=q.u,
                               members=[(0.0, float(q.height), r0, r1)], leaves=[(q, R, t)]))
    parts = []
    for g in groups:
        planes = _profile(g["members"], tol) if len(g["members"]) > 1 and g["full"] else None
        if planes is None or len(planes) < 2:
            parts += [sp.Placed(q, R, t) for q, R, t in g["leaves"]]
        else:
            parts.append(sp.Placed(Polycone(tuple(planes)), g["R"], g["base"]))
    parts += rest
    out = parts[0]
    for q in parts[1:]:
        out = sp.Union(out, q)
    return out


# ------------------------------------------------------------------------------------------------ tree shapes

TREE_SHAPES = ("grouped-balanced", "grouped-multi", "as-is")
_BOOLEANS = (sp.Union, sp.Subtraction, sp.Intersection)


class MultiUnion(sp.Primitive):
    """The union of several solids as ONE node: GDML's multiUnion (Geant4's G4MultiUnion), every part with its own pose.
    Only what the export needs: its SDF is the least of its parts'."""
    exact = False

    def __init__(self, parts):
        self.parts = tuple(parts)

    def sdf(self, p):
        return np.min(np.stack([np.asarray(q.sdf(p), float) for q in self.parts]), axis=0)

    def bounds(self):
        b = [q.bounds() for q in self.parts]
        return np.min([lo for lo, _ in b], axis=0), np.max([hi for _, hi in b], axis=0)


def _bare(p):
    while isinstance(p, sp.Placed):
        p = p.primitive
    return p


def _poses_down(p, R=None, t=None):
    """The tree with every pose moved down to the leaves: a Placed over a boolean is folded into its operands, so that a
    chain of cuts or a tree of unions is not hidden behind a pose."""
    R = np.eye(3) if R is None else R
    t = np.zeros(3) if t is None else t
    if isinstance(p, sp.Placed):
        return _poses_down(p.primitive, R @ np.asarray(p.rotation, float), R @ np.asarray(p.translation, float) + t)
    if isinstance(p, _BOOLEANS):
        return type(p)(_poses_down(p.a, R, t), _poses_down(p.b, R, t))
    if np.array_equal(R, np.eye(3)) and not np.any(t):
        return p
    return sp.Placed(p, R, t)


def _group_cuts(p):
    """Every chain of cuts ((A - B) - C) - ... as A - (B u C u ...). Poses at the leaves (_poses_down)."""
    if isinstance(p, sp.Subtraction):
        tools, base = [], p
        while isinstance(base, sp.Subtraction):
            tools.insert(0, base.b)
            base = base.a
        tools = [_group_cuts(q) for q in tools]
        union = tools[0]
        for q in tools[1:]:
            union = sp.Union(union, q)
        return sp.Subtraction(_group_cuts(base), union)
    if isinstance(p, (sp.Union, sp.Intersection)):
        return type(p)(_group_cuts(p.a), _group_cuts(p.b))
    return p


def _union_operands(p, R, t, out):
    """The operands of a tree of unions (through poses), each as a posed solid."""
    if isinstance(p, sp.Placed):
        return _union_operands(p.primitive, R @ np.asarray(p.rotation, float), R @ np.asarray(p.translation, float) + t, out)
    if isinstance(p, sp.Union):
        _union_operands(p.a, R, t, out)
        return _union_operands(p.b, R, t, out)
    out.append(p if np.array_equal(R, np.eye(3)) and not np.any(t) else sp.Placed(p, R, t))
    return out


def _shape_unions(p, multi: bool):
    """Every tree of unions rebuilt from its operands: balanced (halved recursively), or - `multi` - one MultiUnion when
    there are more than two operands and all of them are primitives (else balanced), as aegir's converter does."""
    if isinstance(p, sp.Placed):
        return sp.Placed(_shape_unions(p.primitive, multi), p.rotation, p.translation)
    if isinstance(p, (sp.Subtraction, sp.Intersection)):
        return type(p)(_shape_unions(p.a, multi), _shape_unions(p.b, multi))
    if not isinstance(p, sp.Union):
        return p
    ops = [_shape_unions(q, multi) for q in _union_operands(p, np.eye(3), np.zeros(3), [])]

    def build(ops):
        if len(ops) == 1:
            return ops[0]
        if multi and len(ops) > 2 and not any(isinstance(_bare(q), _BOOLEANS + (MultiUnion,)) for q in ops):
            return MultiUnion(ops)
        half = (len(ops) + 1) // 2
        return sp.Union(build(ops[:half]), build(ops[half:]))
    return build(ops)


def shape_tree(model: sp.Primitive, tree: str = "grouped-balanced", polycones: bool = True) -> sp.Primitive:
    """The same solid in the tree shape `tree` (TREE_SHAPES; module docstring), with the coaxial unions as polycones
    (`polycones`). The cuts are grouped before the polycones are found, so that a hole stack cut step by step becomes
    one polycone tool; the unions are shaped after, because the polycone merge rebuilds a union as a chain."""
    if tree not in TREE_SHAPES:
        raise ValueError(f"tree: one of {', '.join(TREE_SHAPES)}, not {tree!r}")
    if tree == "as-is":
        return merge_coaxial(model) if polycones else model
    model = _group_cuts(_poses_down(model))
    if polycones:
        model = merge_coaxial(model)
    return _shape_unions(model, multi=tree == "grouped-multi")


class _Writer:
    def __init__(self, name: str, big: float):
        self.solids = ET.Element("solids")
        self.n = 0
        self.name = name
        self.big = big
        self.counts: Dict[str, int] = {}

    def new(self, tag: str, kind: str, **attrs) -> str:
        self.n += 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        ident = f"{self.name}_{kind}_{self.n:04d}"
        e = ET.SubElement(self.solids, tag, name=ident)
        for k, v in attrs.items():
            e.set(k, v if isinstance(v, str) else _f(v))
        if tag not in ("union", "subtraction", "intersection"):
            e.set("lunit", "mm")
            if tag != "box":
                e.set("aunit", "rad")
        return ident

    def sector(self, u) -> Dict[str, object]:
        """startphi, deltaphi. A full turn is written as the double 2 pi to the last digit (FULL_TURN): 2 pi rounded to
        9 digits (6.283185307, 1.8e-10 below 2 pi) makes Geant4's G4Torus (`pDPhi >= twopi`) and G4Polycone
        (`phiTotal >= twopi*(1-DBL_EPSILON)`) an OPEN solid with a hairline slit at phi = 0: a line that crosses the slit
        plane inside the solid stops there and is entered again (2026-10-01, T-30: aegir's scan of the liner, 200 000
        lines: 1815 of 88 490 entries off the program's solid by up to 13 mm, inside its holes and torus fillets at phi = 0
        of each polycone and torus; 0 after). G4Tubs, G4Cons and G4Sphere accept 1e-9 rad less than a full turn."""
        if _is_full(u, (0.0, TWO_PI)):
            return dict(startphi=0.0, deltaphi=FULL_TURN)
        if u[1] - u[0] <= 1e-9:
            raise ValueError(f"empty sector: u = ({u[0]}, {u[1]})")
        return dict(startphi=float(u[0]), deltaphi=float(u[1] - u[0]))

    def spun(self, u, R):
        """The frame of a FULL solid of revolution, turned about its own axis - which changes nothing of the solid. A
        modeller that builds the boundary (FreeCAD's GDML workbench) has the solid's seam at angle 0, and a box face
        through the seam made the cut invalid (2026-09-29: the rounded corners of the liner's pockets, a torus and a
        cylinder on the corner edge of a box)."""
        if not _is_full(u, (0.0, TWO_PI)):
            return R
        c, s_ = math.cos(SPIN), math.sin(SPIN)
        return R @ np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])

    def boolean(self, tag: str, a, b) -> tuple:
        (na, Ra, ta), (nb, Rb, tb) = a, b
        R = Ra.T @ Rb                                                  # the second solid in the frame of the first
        t = Ra.T @ (tb - ta)
        self.n += 1
        self.counts[tag] = self.counts.get(tag, 0) + 1
        ident = f"{self.name}_{tag}_{self.n:04d}"
        e = ET.SubElement(self.solids, tag, name=ident)
        ET.SubElement(e, "first", ref=na)
        ET.SubElement(e, "second", ref=nb)
        ET.SubElement(e, "position", name=ident + "_pos", unit="mm", x=_f(t[0]), y=_f(t[1]), z=_f(t[2]))
        x, y, z = gdml_angles(R, "boolean")
        ET.SubElement(e, "rotation", name=ident + "_rot", unit="rad", x=_f(x), y=_f(y), z=_f(z))
        return ident, Ra, ta

    def multi_union(self, parts, R, t) -> tuple:
        """One multiUnion in the frame (R, t): every node is placed in it like a boolean's second solid."""
        self.n += 1
        self.counts["multiUnion"] = self.counts.get("multiUnion", 0) + 1
        ident = f"{self.name}_multiUnion_{self.n:04d}"
        e = ET.SubElement(self.solids, "multiUnion", name=ident)
        for i, (n, Rn, tn) in enumerate(parts):
            node = ET.SubElement(e, "multiUnionNode", name=f"{ident}_node{i:03d}")
            ET.SubElement(node, "solid", ref=n)
            q, Q = R.T @ (tn - t), R.T @ Rn
            ET.SubElement(node, "position", name=f"{ident}_node{i:03d}_pos", unit="mm", x=_f(q[0]), y=_f(q[1]), z=_f(q[2]))
            x, y, z = gdml_angles(Q, "boolean")
            ET.SubElement(node, "rotation", name=f"{ident}_node{i:03d}_rot", unit="rad", x=_f(x), y=_f(y), z=_f(z))
        return ident, R, t

    def emit(self, p: sp.Primitive, R: np.ndarray, t: np.ndarray) -> tuple:
        """(the solid's name, R, t): the solid in its own frame and that frame's pose in the world."""
        if isinstance(p, sp.Placed):
            Rp, tp = np.asarray(p.rotation, float), np.asarray(p.translation, float)
            return self.emit(p.primitive, R @ Rp, R @ tp + t)
        if isinstance(p, sp.Union):
            return self.boolean("union", self.emit(p.a, R, t), self.emit(p.b, R, t))
        if isinstance(p, sp.Subtraction):
            return self.boolean("subtraction", self.emit(p.a, R, t), self.emit(p.b, R, t))
        if isinstance(p, sp.Intersection):
            return self.boolean("intersection", self.emit(p.a, R, t), self.emit(p.b, R, t))
        if isinstance(p, MultiUnion):
            return self.multi_union([self.emit(q, R, t) for q in p.parts], R, t)
        if isinstance(p, Polycone):
            self.n += 1
            self.counts["polycone"] = self.counts.get("polycone", 0) + 1
            ident = f"{self.name}_polycone_{self.n:04d}"
            e = ET.SubElement(self.solids, "polycone", name=ident, lunit="mm", aunit="rad")
            for k, v in self.sector(p.u).items():
                e.set(k, v if isinstance(v, str) else _f(v))
            for k, (z, r) in enumerate(p.planes):
                ET.SubElement(e, "zplane", rmin=_f(p.inner[k]) if p.inner else "0.0", rmax=_f(r), z=_f(z))
            return ident, self.spun(p.u, R), t
        if isinstance(p, sp.Box):
            h = [float(x) for x in p.half]
            return self.new("box", "box", x=2 * h[0], y=2 * h[1], z=2 * h[2]), R, t
        if isinstance(p, sp.Cylinder):
            up = R @ np.array([0.0, 0.0, 0.5 * float(p.height)])
            return (self.new("tube", "tube", rmin=max(float(p.inner_radius), 0.0), rmax=float(p.radius), z=float(p.height),
                             **self.sector(p.u)), self.spun(p.u, R), t + up)
        if isinstance(p, sp.Cone):
            up = R @ np.array([0.0, 0.0, 0.5 * float(p.height)])
            return (self.new("cone", "cone", rmin1=0.0, rmax1=float(p.radius1), rmin2=0.0, rmax2=float(p.radius2),
                             z=float(p.height), **self.sector(p.u)), self.spun(p.u, R), t + up)
        if isinstance(p, sp.Sphere):
            return self.sphere(p, R, t)
        if isinstance(p, sp.Torus):
            if not _is_full(p.v, (0.0, TWO_PI)):
                raise NotImplementedError("a torus limited around its tube (v) has no GDML solid")
            R = self.spun(p.u, R)
            return (self.new("torus", "torus", rmin=max(float(p.inner_minor), 0.0), rmax=float(p.minor),
                             rtor=float(p.major), **self.sector(p.u)), R, t)
        if isinstance(p, sp.HalfSpace):
            n = np.asarray(p.normal, float)
            n = n / np.linalg.norm(n)
            a = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
            a /= np.linalg.norm(a)
            F = np.stack([a, np.cross(n, a), n], axis=1)             # the box's z is the normal
            big = self.big
            centre = n * (float(p.offset) - big)                       # the box ends at the plane
            return self.new("box", "halfspace", x=2 * big, y=2 * big, z=2 * big), R @ F, R @ centre + t
        raise NotImplementedError(f"{type(p).__name__} has no GDML solid here")

    def sphere(self, p: sp.Sphere, R, t) -> tuple:
        full_v = _is_full(p.v, (-math.pi / 2, math.pi / 2))
        full_u = _is_full(p.u, (0.0, TWO_PI))
        inner = max(float(p.inner_radius), 0.0)
        if full_u and full_v and inner <= 0:
            return self.new("orb", "orb", r=float(p.radius)), R, t
        if full_v or p.latitude == "cone":
            theta = dict(starttheta=0.0, deltatheta=HALF_TURN) if full_v else \
                dict(starttheta=max(0.0, math.pi / 2 - float(p.v[1])),
                     deltatheta=_span(float(p.v[1] - p.v[0]), math.pi - max(0.0, math.pi / 2 - float(p.v[1]))))
            return self.new("sphere", "sphere", rmin=inner, rmax=float(p.radius), **self.sector(p.u), **theta), R, t
        # latitude "plane": the ball between the two planes z = r sin(v0) and z = r sin(v1)
        r = float(p.radius)
        ball = (self.new("sphere", "sphere", rmin=inner, rmax=r, starttheta=0.0, deltatheta=HALF_TURN, **self.sector(p.u)), R, t)
        z0, z1 = r * math.sin(float(p.v[0])), r * math.sin(float(p.v[1]))
        slab = (self.new("box", "slab", x=2.2 * r, y=2.2 * r, z=z1 - z0), R, t + R @ np.array([0.0, 0.0, 0.5 * (z0 + z1)]))
        return self.boolean("intersection", ball, slab)


def _material(parent, m: Material):
    e = ET.SubElement(parent, "material", name=m.name, Z=_f(m.Z))
    ET.SubElement(e, "D", value=repr(float(m.density)), unit="g/cm3")
    ET.SubElement(e, "atom", value=_f(m.A), unit="g/mole")


BEAM_AXES = {                       # the source axis that becomes the world's z (cyclic: a proper rotation)
    "z": np.eye(3),
    "x": np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    "y": np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
}


def export_gdml(model: sp.Primitive, path, name: str = "part", material=STEEL, world=VACUUM,
                margin: float = 0.1, margin_mm: Optional[float] = None, origin=None, beam_axis: str = "z",
                volume_name: Optional[str] = None, world_name: Optional[str] = None,
                physvol_name: Optional[str] = None, polycones: bool = True, tree: str = "grouped-balanced",
                world_half=None) -> dict:
    """Write `model` as `path` (GDML): one solid, one logical volume (`volume_name`, default `<name>_volume`) of
    `material`, placed in a world box of `world` that is `margin` x the model's size (or `margin_mm`) larger on every
    side. `material` / `world`: a Material (written into the file) or the name of a Geant4 NIST material ("G4_Fe":
    only referred to; Geant4 knows it, ROOT does not). `origin` (source point, mm) becomes the world's origin and
    `beam_axis` (the source axis "x", "y" or "z") the world's z: a simulation's beam runs along z. `tree`: the shape of
    the boolean tree (TREE_SHAPES, module docstring; default grouped cuts with balanced unions). `world_half` (3, mm):
    the world box's half sizes instead of the model's extent + margin (an empty baseline in another model's world,
    reco_engine/event_time.py). Returns dict(path,
    solids by kind, primitives, booleans (multiUnions included), world half sizes (mm), bounds in the world, volume,
    material, tree)."""
    name = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name)) or "part"
    if name[0].isdigit():
        name = "p_" + name
    if beam_axis not in BEAM_AXES:
        raise ValueError(f"beam_axis: 'x', 'y' or 'z', not {beam_axis!r}")
    if origin is not None or beam_axis != "z":
        G = BEAM_AXES[beam_axis]
        o = np.zeros(3) if origin is None else np.asarray(origin, float)
        model = sp.Placed(model, G, -G @ o)                            # x_world = G (x - origin)
    model = shape_tree(model, tree, polycones)
    lo, hi = (np.asarray(b, float) for b in model.bounds())
    size = float(np.max(hi - lo)) if (np.isfinite(lo).all() and np.isfinite(hi).all()) else float("inf")
    if not size < UNBOUNDED:
        raise ValueError("the model has no finite bounds (a half space alone)")
    w = _Writer(name, big=4.0 * max(size, 1.0))
    solid, R, t = w.emit(model, np.eye(3), np.zeros(3))
    pad = float(margin_mm) if margin_mm is not None else margin * size
    half = np.maximum(np.abs(lo), np.abs(hi)) + pad                    # the world is centred on the origin
    if world_half is not None:
        half = np.asarray(world_half, float)
    volume_name = volume_name or f"{name}_volume"
    world_name = world_name or f"{name}_world"
    ET.SubElement(w.solids, "box", name=f"{name}_world_box", x=_f(2 * half[0]), y=_f(2 * half[1]), z=_f(2 * half[2]),
                  lunit="mm")

    root = ET.Element("gdml")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", SCHEMA)
    ET.SubElement(root, "define")
    materials = ET.SubElement(root, "materials")
    written = set()
    for m in (material, world):
        if isinstance(m, Material) and m.name not in written:
            _material(materials, m)
            written.add(m.name)
    ref = [m.name if isinstance(m, Material) else str(m) for m in (material, world)]
    root.append(w.solids)
    structure = ET.SubElement(root, "structure")
    vol = ET.SubElement(structure, "volume", name=volume_name)
    ET.SubElement(vol, "materialref", ref=ref[0])
    ET.SubElement(vol, "solidref", ref=solid)
    wv = ET.SubElement(structure, "volume", name=world_name)
    ET.SubElement(wv, "materialref", ref=ref[1])
    ET.SubElement(wv, "solidref", ref=f"{name}_world_box")
    pv = ET.SubElement(wv, "physvol", name=physvol_name or f"{name}_placed")
    ET.SubElement(pv, "volumeref", ref=volume_name)
    ET.SubElement(pv, "position", name=f"{name}_position", unit="mm", x=_f(t[0]), y=_f(t[1]), z=_f(t[2]))
    x, y, z = gdml_angles(R)
    ET.SubElement(pv, "rotation", name=f"{name}_rotation", unit="rad", x=_f(x), y=_f(y), z=_f(z))
    setup = ET.SubElement(root, "setup", name="Default", version="1.0")
    ET.SubElement(setup, "world", ref=world_name)

    ET.indent(root, space="  ")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n")
    booleans = sum(w.counts.get(k, 0) for k in ("union", "subtraction", "intersection", "multiUnion"))
    return dict(path=str(path), solids=dict(w.counts), primitives=sum(w.counts.values()) - booleans, booleans=booleans,
                world_half=[float(v) for v in half], bounds=[lo.tolist(), hi.tolist()], volume=volume_name,
                material=ref[0], tree=tree)


def model_of_program(d: dict) -> sp.Primitive:
    """A program tree as plain data (reco_field.export.program_to_dict; the real-tree study's programs) back as an SDF
    tree. A rotation stored rounded is replaced by the nearest rotation, as aegir's converter does (make_gdml.rotation_of)."""
    kind = d["type"]
    if kind == "Placed":
        U, _, Vt = np.linalg.svd(np.asarray(d["rotation"], float))
        R = U @ Vt
        if np.linalg.det(R) < 0 or not np.allclose(R, d["rotation"], atol=1e-3):
            raise ValueError(f"a Placed node's matrix is no rotation:\n{d['rotation']}")
        return sp.Placed(model_of_program(d["primitive"]), R, np.asarray(d["translation"], float))
    if kind in ("Union", "Intersection", "Subtraction"):
        return getattr(sp, kind)(model_of_program(d["a"]), model_of_program(d["b"]))
    fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k not in ("type", "big")}
    return getattr(sp, kind)(**fields)


def export_result(result, path, name: Optional[str] = None, **kw) -> dict:
    """The accepted operations of a ReconstructionResult (the loop's program) as GDML."""
    from reco_engine.result_to_program import program_model
    model = program_model(result)
    if model is None:
        raise ValueError("the result has no accepted operation")
    return export_gdml(model, path, name or Path(str(result.input_path)).stem, **kw)


# ------------------------------------------------------------------------------------------------ reading back
# A reader of what this module writes, by the convention of the module docstring: the solid as an SDF tree again.
# For the tests and for a check of a written file without Geant4; NOT a general GDML reader.

def read_gdml(path) -> sp.Primitive:
    root = ET.parse(str(path)).getroot()
    solids = {e.get("name"): e for e in root.find("solids")}
    world = root.find("setup").find("world").get("ref")
    volumes = {v.get("name"): v for v in root.find("structure").findall("volume")}
    pv = volumes[world].find("physvol")
    part = volumes[pv.find("volumeref").get("ref")]

    def pose(e, placed):
        p, r = e.find("position"), e.find("rotation")
        t = np.array([float(p.get(k, 0.0)) for k in "xyz"]) if p is not None else np.zeros(3)
        R = matrix_of_angles(*(float(r.get(k, 0.0)) for k in "xyz"), placed=placed) if r is not None else np.eye(3)
        return R, t

    def g(e, k, default=0.0):
        return float(e.get(k, default))

    def sector(e):
        return (g(e, "startphi"), g(e, "startphi") + g(e, "deltaphi", TWO_PI))

    def build(name) -> sp.Primitive:
        e = solids[name]
        if e.tag == "multiUnion":
            parts = [sp.Placed(build(n.find("solid").get("ref")), *pose(n, "boolean")) for n in e.findall("multiUnionNode")]
            return MultiUnion(parts)
        if e.tag in ("union", "subtraction", "intersection"):
            a, b = build(e.find("first").get("ref")), build(e.find("second").get("ref"))
            R, t = pose(e, "boolean")
            b = sp.Placed(b, R, t)
            return {"union": sp.Union, "subtraction": sp.Subtraction, "intersection": sp.Intersection}[e.tag](a, b)
        if e.tag == "box":
            return sp.Box((g(e, "x") / 2, g(e, "y") / 2, g(e, "z") / 2))
        if e.tag == "tube":
            c = sp.Cylinder(g(e, "rmax"), g(e, "z"), sector(e), g(e, "rmin"))
            return sp.Placed(c, np.eye(3), np.array([0.0, 0.0, -g(e, "z") / 2]))
        if e.tag == "cone":
            c = sp.Cone(g(e, "rmax1"), g(e, "rmax2"), g(e, "z"), sector(e))
            return sp.Placed(c, np.eye(3), np.array([0.0, 0.0, -g(e, "z") / 2]))
        if e.tag == "polycone":
            zp = e.findall("zplane")
            inner = tuple(float(z.get("rmin", 0.0)) for z in zp)
            return Polycone(tuple((float(z.get("z")), float(z.get("rmax"))) for z in zp), sector(e),
                            inner if max(inner) > 0 else None)
        if e.tag == "orb":
            return sp.Sphere(g(e, "r"))
        if e.tag == "sphere":
            th0, dth = g(e, "starttheta"), g(e, "deltatheta", math.pi)
            return sp.Sphere(g(e, "rmax"), sector(e), (math.pi / 2 - th0 - dth, math.pi / 2 - th0), g(e, "rmin"), "cone")
        if e.tag == "torus":
            return sp.Torus(g(e, "rtor"), g(e, "rmax"), sector(e), (0.0, TWO_PI), g(e, "rmin"))
        raise NotImplementedError(e.tag)

    R, t = pose(pv, "physvol")
    return sp.Placed(build(part.find("solidref").get("ref")), R, t)
