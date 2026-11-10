#!/usr/bin/env python3
"""
Analytic signed distance fields of the solids of FreeCAD's Part module, with segment parameters.

Every primitive is a solid in its own local frame with `sdf(points)`: (..., 3) -> (...), negative inside, positive outside,
in the units of the parameters. `.exact` says whether the values are true distances everywhere (True) or not (False): a
lower bound (|sdf| <= true distance) wherever two constraints are combined with max(), as for a segment, and only a
first-order approximation for the ellipsoid. The sign, hence the occupancy `sdf <= 0`, is exact for all of them.

Segments are given by two parameter ranges, in RADIANS:
    u   the sweep angle about the axis (z): sphere and torus longitude, cylinder and cone sector; (0, 2 pi) = the full body
    v   sphere: the latitude range (-pi/2, pi/2); torus: the minor-circle angle, 0 = outer equator, counter-clockwise
        towards +z; cylinder: the axial range (default (0, height)). FreeCAD's Part::Torus measures that angle the other way
        round: its Angle1/Angle2 are v = (-Angle2, -Angle1) here (validated, tests/test_sdf_primitives.py)
For a sphere the latitude limit `latitude="cone"` (default) is the true parameter region (a spherical sector: cones from the
centre); `latitude="plane"` is what FreeCAD's Part.makeSphere(r, pnt, dir, angle1, angle2, angle3) builds - the ball between the
planes z = r sin(v0) and z = r sin(v1) (validated against Part, tests/test_sdf_primitives.py).

Frames (as in FreeCAD, so a Part shape and its SDF can be compared directly): Box, Wedge and Cylinder/Cone stand on z = 0 with
the corner or base centre at the origin only where FreeCAD does (`Box` here is CENTRED: use `Placed` to shift); Sphere, Torus and
Ellipsoid are centred at the origin.

    Box(half), Sphere(radius, u, v, inner_radius, latitude), Cylinder(radius, height, u, inner_radius),
    Cone(radius1, radius2, height, u), Torus(major, minor, u, v, inner_minor), Wedge(...), Ellipsoid(a, b, c), Prism(n, R, h)
    HalfSpace(normal, offset)                         the half space normal.x <= offset (a flat cut when intersected)
    Placed(primitive, rotation, translation)          rigid placement: world point x -> local R^T (x - t)
    Union / Intersection / Subtraction                Boolean operators (bounds, as usual for SDFs)
    voxel_centres(...), occupancy_grid(...), sdf_grid(...)   sample on the project's voxel grid (array frame [z, -y, x])

Pure numpy (+ scipy.spatial for the convex polyhedra); no FreeCAD, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

TWO_PI = 2.0 * np.pi
FULL = (0.0, TWO_PI)
BIG = 1e9


def deg(*angles):
    """Degrees to radians (for readable parameters)."""
    out = tuple(np.radians(a) for a in angles)
    return out[0] if len(out) == 1 else out


# ------------------------------------------------------------------------------------------------ 2D building blocks

def _sd_box2(x, y, hx, hy):
    dx, dy = np.abs(x) - hx, np.abs(y) - hy
    return np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)) + np.minimum(np.maximum(dx, dy), 0.0)


def _convex_wedge(x, y, h):
    """Exact SDF of the 2D wedge {|angle| <= h} (apex at the origin, axis +x), 0 < h <= pi/2."""
    qy = np.abs(y)
    d = (np.cos(h), np.sin(h))
    n = (-np.sin(h), np.cos(h))
    proj = x * d[0] + qy * d[1]
    nd = x * n[0] + qy * n[1]
    return np.where(nd <= 0, nd, np.where(proj >= 0, nd, np.hypot(x, qy)))


def sector2d(x, y, u0, u1):
    """Exact SDF of the 2D angular sector {u0 <= atan2(y, x) <= u1} (apex at the origin, sweep u1 - u0 in (0, 2 pi])."""
    w = u1 - u0
    if w <= 0:
        raise ValueError(f"sector needs u1 > u0, got ({u0}, {u1})")
    if w >= TWO_PI - 1e-9:
        return np.full(np.broadcast(x, y).shape, -BIG)
    mid = 0.5 * (u0 + u1)
    c, s = np.cos(mid), np.sin(mid)
    xr, yr = c * x + s * y, -s * x + c * y                                 # rotate by -mid: the sector is centred on +x
    h = 0.5 * w
    if h <= np.pi / 2:
        return _convex_wedge(xr, yr, h)
    return -_convex_wedge(-xr, yr, np.pi - h)                             # reflex sector = complement of a convex one


def _is_full(rng: Tuple[float, float], full: Tuple[float, float]) -> bool:
    return rng[0] <= full[0] + 1e-9 and rng[1] >= full[1] - 1e-9


# ------------------------------------------------------------------------------------------------------ primitives

class Primitive:
    exact = True

    def sdf(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """(lo, hi) of an axis-aligned box that contains the solid, in the local frame."""
        raise NotImplementedError

    def contains(self, p: np.ndarray) -> np.ndarray:
        return self.sdf(p) <= 0

    def __sub__(self, other):
        return Subtraction(self, other)

    def __or__(self, other):
        return Union(self, other)

    def __and__(self, other):
        return Intersection(self, other)


@dataclass
class Box(Primitive):
    """Box with half extents `half` (3), centred at the origin. FreeCAD's Part.makeBox(l, w, h) has its corner at the origin:
    Placed(Box((l/2, w/2, h/2)), translation=(l/2, w/2, h/2))."""
    half: Sequence[float]

    def sdf(self, p):
        p = np.asarray(p, float)
        q = np.abs(p) - np.asarray(self.half, float)
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)

    def bounds(self):
        h = np.asarray(self.half, float)
        return -h, h


@dataclass
class Sphere(Primitive):
    """Ball of `radius`, optionally hollow (`inner_radius`) and limited to a segment: longitude u, latitude v (radians)."""
    radius: float
    u: Tuple[float, float] = FULL
    v: Tuple[float, float] = (-np.pi / 2, np.pi / 2)
    inner_radius: float = 0.0
    latitude: str = "cone"                       # "cone": the parameter region; "plane": as FreeCAD's makeSphere

    @property
    def exact(self):
        return _is_full(self.u, FULL) and _is_full(self.v, (-np.pi / 2, np.pi / 2)) and self.inner_radius <= 0

    def sdf(self, p):
        p = np.asarray(p, float)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        r = np.linalg.norm(p, axis=-1)
        d = r - self.radius
        if self.inner_radius > 0:
            d = np.maximum(d, self.inner_radius - r)
        if not _is_full(self.u, FULL):
            d = np.maximum(d, sector2d(x, y, *self.u))
        if not _is_full(self.v, (-np.pi / 2, np.pi / 2)):
            if self.latitude == "plane":
                d = np.maximum(d, np.maximum(z - self.radius * np.sin(self.v[1]), self.radius * np.sin(self.v[0]) - z))
            elif self.latitude == "cone":
                d = np.maximum(d, sector2d(np.hypot(x, y), z, *self.v))
            else:
                raise ValueError(f"latitude must be 'cone' or 'plane', got {self.latitude!r}")
        return d

    def bounds(self):
        R = self.radius
        return np.full(3, -R), np.full(3, R)


@dataclass
class Cylinder(Primitive):
    """Cylinder of `radius` from z = 0 to z = `height` (FreeCAD's makeCylinder), hollow with `inner_radius` (a tube), limited
    to the sector u about the z axis."""
    radius: float
    height: float
    u: Tuple[float, float] = FULL
    inner_radius: float = 0.0

    @property
    def exact(self):
        return _is_full(self.u, FULL)

    def sdf(self, p):
        p = np.asarray(p, float)
        rho = np.hypot(p[..., 0], p[..., 1])
        ri, ro = self.inner_radius, self.radius
        # meridian profile: rho in [ri, ro] (a full cylinder has no inner wall: the axis is interior), z in [0, height]
        dr = np.maximum(rho - ro, ri - rho) if ri > 0 else rho - ro
        dz = np.abs(p[..., 2] - 0.5 * self.height) - 0.5 * self.height
        d = np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0)) + np.minimum(np.maximum(dr, dz), 0.0)
        if not _is_full(self.u, FULL):
            d = np.maximum(d, sector2d(p[..., 0], p[..., 1], *self.u))
        return d

    def bounds(self):
        return np.array([-self.radius, -self.radius, 0.0]), np.array([self.radius, self.radius, self.height])


@dataclass
class Cone(Primitive):
    """Truncated cone: `radius1` at z = 0, `radius2` at z = `height` (FreeCAD's makeCone), limited to the sector u."""
    radius1: float
    radius2: float
    height: float
    u: Tuple[float, float] = FULL

    @property
    def exact(self):
        return _is_full(self.u, FULL)

    def sdf(self, p):
        p = np.asarray(p, float)
        r1, r2, hc = self.radius1, self.radius2, 0.5 * self.height
        qx = np.hypot(p[..., 0], p[..., 1])
        qy = p[..., 2] - hc                                                # centred: z in [-hc, hc]
        k1 = np.array([r2, hc])
        k2 = np.array([r2 - r1, 2.0 * hc])
        ca = np.stack([qx - np.minimum(qx, np.where(qy < 0.0, r1, r2)), np.abs(qy) - hc], axis=-1)
        t = np.clip(((k1[0] - qx) * k2[0] + (k1[1] - qy) * k2[1]) / (k2 @ k2), 0.0, 1.0)
        cb = np.stack([qx - k1[0] + k2[0] * t, qy - k1[1] + k2[1] * t], axis=-1)
        s = np.where((cb[..., 0] < 0.0) & (ca[..., 1] < 0.0), -1.0, 1.0)
        d = s * np.sqrt(np.minimum((ca ** 2).sum(-1), (cb ** 2).sum(-1)))
        if not _is_full(self.u, FULL):
            d = np.maximum(d, sector2d(p[..., 0], p[..., 1], *self.u))
        return d

    def bounds(self):
        R = max(self.radius1, self.radius2)
        return np.array([-R, -R, 0.0]), np.array([R, R, self.height])


@dataclass
class Torus(Primitive):
    """Torus with `major` radius (axis z) and `minor` radius, limited to a segment: u = sweep about the axis, v = the angle
    around the tube (0 = outer equator, counter-clockwise towards +z); `inner_minor` > 0 makes the tube hollow. Full v is
    FreeCAD's makeTorus(R, r, pnt, dir, 0, 360, angle) with angle = u1 - u0 (its minor-angle range is not a clean tube segment)."""
    major: float
    minor: float
    u: Tuple[float, float] = FULL
    v: Tuple[float, float] = FULL
    inner_minor: float = 0.0

    @property
    def exact(self):
        return _is_full(self.u, FULL) and _is_full(self.v, FULL) and self.inner_minor <= 0

    def sdf(self, p):
        p = np.asarray(p, float)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        qx, qz = np.hypot(x, y) - self.major, z
        t = np.hypot(qx, qz)
        d = t - self.minor
        if self.inner_minor > 0:
            d = np.maximum(d, self.inner_minor - t)
        if not _is_full(self.u, FULL):
            d = np.maximum(d, sector2d(x, y, *self.u))
        if not _is_full(self.v, FULL):
            d = np.maximum(d, sector2d(qx, qz, *self.v))
        return d

    def bounds(self):
        R = self.major + self.minor
        return np.array([-R, -R, -self.minor]), np.array([R, R, self.minor])


@dataclass
class Ellipsoid(Primitive):
    """Ellipsoid with semi axes (a, b, c) along x, y, z. The distance is the usual first-order approximation (it can exceed
    the true distance by a couple of voxels at strong anisotropy); the sign is exact."""
    a: float
    b: float
    c: float
    exact = False

    def sdf(self, p):
        p = np.asarray(p, float)
        r = np.array([self.a, self.b, self.c])
        k0 = np.linalg.norm(p / r, axis=-1)
        k1 = np.linalg.norm(p / (r * r), axis=-1)
        return np.where(k1 > 1e-12, k0 * (k0 - 1.0) / np.maximum(k1, 1e-12), -r.min() * (1.0 - k0))

    def bounds(self):
        r = np.array([self.a, self.b, self.c])
        return -r, r


@dataclass
class HalfSpace(Primitive):
    """The half space {x : normal . x <= offset} (`normal` a unit vector): exact SDF `normal . x - offset`. As a cutter it removes
    everything on the +normal side of the plane; intersected with a body it clips it (a flat cut)."""
    normal: Sequence[float]
    offset: float

    def __post_init__(self):
        n = np.asarray(self.normal, float)
        self.normal = n / np.linalg.norm(n)

    def sdf(self, p):
        return np.asarray(p, float) @ np.asarray(self.normal) - self.offset

    def bounds(self):
        return np.full(3, -BIG), np.full(3, BIG)


class ConvexPolyhedron(Primitive):
    """Convex hull of a point set; the SDF is the largest signed plane distance (exact inside, a bound outside)."""
    exact = False

    def __init__(self, vertices: np.ndarray):
        from scipy.spatial import ConvexHull
        self.vertices = np.asarray(vertices, float)
        hull = ConvexHull(self.vertices)
        planes = {}
        for eq in hull.equations:                                          # (nx, ny, nz, offset), outward normals
            planes[tuple(np.round(eq, 9))] = eq
        self.normals = np.array([e[:3] for e in planes.values()])
        self.offsets = np.array([e[3] for e in planes.values()])

    def sdf(self, p):
        p = np.asarray(p, float)
        return np.max(p @ self.normals.T + self.offsets, axis=-1)

    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)


def Wedge(xmin, ymin, zmin, z2min, x2min, xmax, ymax, zmax, z2max, x2max) -> ConvexPolyhedron:
    """FreeCAD's Part.makeWedge (same argument order): the rectangle [xmin,xmax] x [zmin,zmax] at y = ymin joined to the
    rectangle [x2min,x2max] x [z2min,z2max] at y = ymax."""
    return ConvexPolyhedron(np.array([
        [xmin, ymin, zmin], [xmax, ymin, zmin], [xmax, ymin, zmax], [xmin, ymin, zmax],
        [x2min, ymax, z2min], [x2max, ymax, z2min], [x2max, ymax, z2max], [x2min, ymax, z2max]]))


def Prism(sides: int, circumradius: float, height: float, first_vertex_angle: float = 0.0) -> ConvexPolyhedron:
    """Regular prism standing on z = 0 (FreeCAD's Part::Prism): a polygon of `sides` around the z axis, one vertex at the
    angle `first_vertex_angle` from +x."""
    a = first_vertex_angle + TWO_PI * np.arange(sides) / sides
    ring = np.stack([circumradius * np.cos(a), circumradius * np.sin(a)], axis=-1)
    return ConvexPolyhedron(np.concatenate([np.c_[ring, np.zeros(sides)], np.c_[ring, np.full(sides, height)]]))


# ------------------------------------------------------------------------------------------- placement and operators

@dataclass
class Placed(Primitive):
    """`primitive` rotated by R (columns = its axes in the world) and shifted by t: x_world = R x_local + t."""
    primitive: Primitive
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def exact(self):
        return self.primitive.exact

    def sdf(self, p):
        return self.primitive.sdf((np.asarray(p, float) - np.asarray(self.translation, float)) @ np.asarray(self.rotation, float))

    def bounds(self):
        lo, hi = self.primitive.bounds()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        w = corners @ np.asarray(self.rotation).T + np.asarray(self.translation)
        return w.min(axis=0), w.max(axis=0)


class _Binary(Primitive):
    exact = False

    def __init__(self, a: Primitive, b: Primitive):
        self.a, self.b = a, b


class Union(_Binary):
    def sdf(self, p):
        return np.minimum(self.a.sdf(p), self.b.sdf(p))

    def bounds(self):
        (l1, h1), (l2, h2) = self.a.bounds(), self.b.bounds()
        return np.minimum(l1, l2), np.maximum(h1, h2)


class Intersection(_Binary):
    def sdf(self, p):
        return np.maximum(self.a.sdf(p), self.b.sdf(p))

    def bounds(self):
        (l1, h1), (l2, h2) = self.a.bounds(), self.b.bounds()
        return np.maximum(l1, l2), np.minimum(h1, h2)


class Subtraction(_Binary):
    """a minus b. Exact only outside of the result, as for every SDF subtraction."""
    def sdf(self, p):
        return np.maximum(self.a.sdf(p), -self.b.sdf(p))

    def bounds(self):
        return self.a.bounds()


# ----------------------------------------------------------------------------------------------------- voxel grids

def voxel_centres(resolution: int = 64, grid_mm: float = 40.0) -> np.ndarray:
    """(R, R, R, 3) world coordinates (mm) of the voxel centres of the project's grid, in its array frame [z, -y, x]: array axis 0
    is z, axis 1 is y running BACKWARDS, axis 2 is x (reco_engine.placement); the world origin is at the grid centre."""
    v = grid_mm / resolution
    c = (np.arange(resolution) + 0.5 - resolution / 2.0) * v
    z, yr, x = np.meshgrid(c, c, c, indexing="ij")
    return np.stack([x, -yr, z], axis=-1)


def sdf_grid(primitive: Primitive, resolution: int = 64, grid_mm: float = 40.0) -> np.ndarray:
    """Signed distance (mm) at every voxel centre, array frame [z, -y, x]."""
    return primitive.sdf(voxel_centres(resolution, grid_mm))


def occupancy_grid(primitive: Primitive, resolution: int = 64, grid_mm: float = 40.0) -> np.ndarray:
    """uint8 {0, 1} occupancy at the voxel centres, array frame [z, -y, x] - what a voxelizer of the solid produces up to
    the voxelizer's own boundary rule."""
    return (sdf_grid(primitive, resolution, grid_mm) <= 0).astype(np.uint8)


PRIMITIVES = {"halfspace": HalfSpace, "box": Box, "sphere": Sphere, "cylinder": Cylinder, "cone": Cone, "torus": Torus, "ellipsoid": Ellipsoid,
              "wedge": Wedge, "prism": Prism}
