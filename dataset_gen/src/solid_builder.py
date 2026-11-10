#!/usr/bin/env python3
"""
Exact solids for the dataset generator's --with-field / --segment-prob / --emit-negative-residual (dataset_generator.py):
the sample's shape as an `sdf_primitives` program, built by FreeCAD's Part module through reco_field/export.build_program
(reco_field/freecad_builder.py, which already applies every Part convention of docs/freecad_part_shapes.md: centred box,
cylinder/cone standing on z = 0, a sector (u0, u1) built as the sweep u1 - u0 turned by u0, sphere latitude by planes,
torus v mirrored). No new FreeCAD-side code: the builder that exports reconstructions builds the training shapes too.

Why a program instead of the per-topology macros (dataset_gen/FCMacros): the in-scale modifications (carve, planar cut,
hollow) were only ever applied to the VOXELS - the STL next to a carved sample is the uncut base. A field computed from
that STL would describe a different solid than its voxels. With the shape as a program, the modification is re-expressed
exactly in world millimetres (`carve_cutter`, `cut_halfspaces`, `hollow_inner`), the MODIFIED solid is built in FreeCAD,
and both the field and (re-voxelised) .bin come from that one solid. The macros stay the path for runs without these flags
(old behaviour unchanged).

Frames: `base_program` is the unplaced shape as the macros build it (local primitive frame rotated by the sample's
quaternion about the origin). The in-scale placement scales it by k about its STL bounding-box centre b and moves it to c:
x_world = k (R x_local - b) + c = R (k x_local) + (c - k b) - `placed_program`. The voxel grid's array index a (stltovoxel
.bin order [z, -y, x], index a at world -h + (a + 0.5) v) maps to world with `array_to_world`.
"""

from __future__ import annotations

import math
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import sdf_primitives as sp

ROOT = Path(__file__).resolve().parent.parent.parent
A_WORLD_TO_ARRAY = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])   # = rotation_symmetry.A_WORLD_TO_ARRAY
KINDS = ("box", "sphere", "cylinder", "cone", "torus")
SEGMENT_KINDS = ("sphere", "cylinder", "cone", "torus")
DEFLECTION_MM = 0.05                    # the macros' LINEAR_DEFLECTION_MM


def quat_to_matrix(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat(m: np.ndarray) -> List[float]:
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(m).as_quat()
    return [float(v) for v in (q if q[3] >= 0 else -q)]


# ------------------------------------------------------------------------------------------------ shapes and segments

def sample_segment(rng, topology: str) -> Dict:
    """A random u/v segment of a revolved kind, in radians, as sdf_primitives takes them. u: a sector of 60-300 deg starting
    anywhere. Sphere: u, v or both (v = latitude limits by planes, each end moved in with probability 1/2, at least one).
    Torus: u, v or both (v = a pie of the tube of 90-300 deg). Cylinder / cone: u only. Drawn from the generator's own
    random.Random, and only when --segment-prob draws a segment (so a run without the flag keeps its random stream)."""
    def u_range():
        u0 = rng.uniform(0.0, 2 * math.pi)
        return [u0, u0 + math.radians(rng.uniform(60.0, 300.0))]
    seg = {}
    which = "u" if topology in ("cylinder", "cone") else rng.choice(("u", "v", "uv"))
    if "u" in which:
        seg["u"] = u_range()
    if "v" in which:
        if topology == "sphere":
            lo, hi = -math.pi / 2, math.pi / 2
            while lo == -math.pi / 2 and hi == math.pi / 2:
                if rng.random() < 0.5:
                    lo = math.radians(rng.uniform(-75.0, -10.0))
                if rng.random() < 0.5:
                    hi = math.radians(rng.uniform(10.0, 75.0))
            seg["v"] = [lo, hi]
        else:
            v0 = rng.uniform(-math.pi, math.pi)
            seg["v"] = [v0, v0 + math.radians(rng.uniform(90.0, 300.0))]
    return seg


def local_primitive(topology: str, params: Dict, scale_mm: float, segment: Optional[Dict] = None) -> sp.Primitive:
    """The macro's shape (dataset_utils.PARAMETER_SPECS ratios x scale_mm, the macros' conventions) as an sdf primitive in
    its local frame, optionally limited to `segment`. The torus is WHOLE unless a segment says otherwise: the macro's
    angle1..3 draws go through Part.makeTorus's minor angles, which do not give a clean tube segment
    (reco_field/freecad_builder.py), so they are not reproduced."""
    s = float(scale_mm)
    seg = segment or {}
    u = tuple(seg.get("u", sp.FULL))
    if topology == "box":
        return sp.Box([params["x_half_ratio"] * s, params["y_half_ratio"] * s, params["z_half_ratio"] * s])
    if topology == "sphere":
        return sp.Sphere(params["radius_ratio"] * s, u=u, v=tuple(seg.get("v", (-np.pi / 2, np.pi / 2))), latitude="plane")
    if topology == "cylinder":
        return sp.Cylinder(params["radius_ratio"] * s, 2.0 * params["half_length_ratio"] * s, u=u)
    if topology == "cone":
        return sp.Cone(params["radius1_ratio"] * s, params["radius2_ratio"] * s, params["height_ratio"] * s, u=u)
    if topology == "torus":
        return sp.Torus(params["major_radius_ratio"] * s, params["minor_radius_ratio"] * s, u=u, v=tuple(seg.get("v", sp.FULL)))
    raise ValueError(f"no exact primitive for topology {topology!r} (supported: {KINDS})")


def scaled(prim: sp.Primitive, k: float) -> sp.Primitive:
    """The same primitive with every length multiplied by k (angles unchanged)."""
    if isinstance(prim, sp.Box):
        return sp.Box([h * k for h in prim.half])
    if isinstance(prim, sp.Sphere):
        return sp.Sphere(prim.radius * k, u=prim.u, v=prim.v, inner_radius=prim.inner_radius * k, latitude=prim.latitude)
    if isinstance(prim, sp.Cylinder):
        return sp.Cylinder(prim.radius * k, prim.height * k, u=prim.u, inner_radius=prim.inner_radius * k)
    if isinstance(prim, sp.Cone):
        return sp.Cone(prim.radius1 * k, prim.radius2 * k, prim.height * k, u=prim.u)
    if isinstance(prim, sp.Torus):
        return sp.Torus(prim.major * k, prim.minor * k, u=prim.u, v=prim.v, inner_minor=prim.inner_minor * k)
    raise ValueError(f"cannot scale {type(prim).__name__}")


def local_centre(prim: sp.Primitive) -> np.ndarray:
    """The reference centre in the local frame: the middle of the axis for a cylinder / cone (they stand on z = 0),
    the origin otherwise (the full body's centre also for a segment - u/v are recorded separately)."""
    if isinstance(prim, (sp.Cylinder, sp.Cone)):
        return np.array([0.0, 0.0, prim.height / 2.0])
    return np.zeros(3)


def dims_mm(prim: sp.Primitive) -> Dict[str, float]:
    """Dimensions keyed as reco_engine/result_schema.py documents them (box x/y/z_half, sphere radius, cylinder radius +
    half_length, cone radius1/radius2/half_length, torus major_radius/minor_radius)."""
    if isinstance(prim, sp.Box):
        return dict(x_half=float(prim.half[0]), y_half=float(prim.half[1]), z_half=float(prim.half[2]))
    if isinstance(prim, sp.Sphere):
        return dict(radius=float(prim.radius))
    if isinstance(prim, sp.Cylinder):
        return dict(radius=float(prim.radius), half_length=float(prim.height / 2))
    if isinstance(prim, sp.Cone):
        return dict(radius1=float(prim.radius1), radius2=float(prim.radius2), half_length=float(prim.height / 2))
    if isinstance(prim, sp.Torus):
        return dict(major_radius=float(prim.major), minor_radius=float(prim.minor))
    raise ValueError(type(prim).__name__)


def base_program(prim: sp.Primitive, R: np.ndarray) -> sp.Primitive:
    return sp.Placed(prim, rotation=np.asarray(R, float), translation=np.zeros(3))


def placed_program(prim: sp.Primitive, R: np.ndarray, placement: Dict, bbox_center) -> sp.Primitive:
    """The in-scale placed shape: scaled by placement["scale_factor"] about the STL bbox centre, moved to center_mm."""
    k = float(placement["scale_factor"])
    t = np.asarray(placement["center_mm"], float) - k * np.asarray(bbox_center, float)
    return sp.Placed(scaled(prim, k), rotation=np.asarray(R, float), translation=t)


# ------------------------------------------------------------------------------------------------ labels

def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    """A sign-free axis made unique: its largest-magnitude component positive."""
    return axis if axis[int(np.argmax(np.abs(axis)))] >= 0 else -axis


def _shortest_arc(z_to: np.ndarray) -> np.ndarray:
    """The rotation taking +z to `z_to` along the great circle (a canonical choice of the free spin about an axis)."""
    z = np.array([0.0, 0.0, 1.0])
    a = np.asarray(z_to, float) / np.linalg.norm(z_to)
    v, c = np.cross(z, a), float(z @ a)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def rotation_canonical(prim: sp.Primitive, R: np.ndarray) -> Dict:
    """The sample's orientation reduced by its symmetry group (training/rotation_symmetry.py for the box):
      box       longest-first body axes (det +1), then the admissible group element (D2 generic; bigger with ties,
                rotation_symmetry.admissible_group) that brings it closest to the identity - one quaternion per body;
      sphere    SO(3): identity;
      cylinder, torus (whole)   the axis with its sign made unique (+-axis are one pose) and the shortest-arc rotation to it;
      cone (whole)              the SIGNED axis (radius1 base -> radius2 end) and the shortest-arc rotation to it;
      segment   no continuous symmetry left: R turned by u0 about the axis so the sector starts at 0 (u = (0, sweep)).
    Quaternions (qx, qy, qz, qw), world frame."""
    R = np.asarray(R, float)
    if isinstance(prim, sp.Box):
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from training import rotation_symmetry as rs
        half = [float(h) for h in prim.half]
        order, sd = rs.canonical_axes(half)
        M = rs.canonical_orientation(R, half)                               # array frame
        W = rs.A_WORLD_TO_ARRAY.T @ M                                       # world frame, longest-first columns
        ok = rs.admissible_group(sd)
        cands = [W @ g for g, a in zip(rs.GROUP, ok) if a]
        best = max(cands, key=lambda m: np.trace(m))
        return dict(group="box", quaternion=matrix_to_quat(best), dims_order=[float(d) for d in sd],
                    admissible=int(ok.sum()))
    if isinstance(prim, sp.Sphere) and _full(prim.u) and _full_lat(prim.v):
        return dict(group="SO3", quaternion=[0.0, 0.0, 0.0, 1.0])
    axis = R[:, 2]
    segment = not _full(getattr(prim, "u", sp.FULL)) or (isinstance(prim, sp.Torus) and not _full(prim.v)) or \
        (isinstance(prim, sp.Sphere) and not _full_lat(prim.v))
    if segment:
        u0 = float(getattr(prim, "u", sp.FULL)[0])
        c, s = math.cos(u0), math.sin(u0)
        Rc = R @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        u = getattr(prim, "u", sp.FULL)
        return dict(group="none", quaternion=matrix_to_quat(Rc), axis=[float(a) for a in axis],
                    u=[0.0, float(u[1] - u[0])])
    if isinstance(prim, sp.Cone):
        return dict(group="Cinf", quaternion=matrix_to_quat(_shortest_arc(axis)), axis=[float(a) for a in axis])
    ax = _canonical_axis(axis)
    return dict(group="Dinf", quaternion=matrix_to_quat(_shortest_arc(ax)), axis=[float(a) for a in ax])


def _full(u) -> bool:
    return u[0] <= 1e-12 and u[1] >= 2 * np.pi - 1e-9


def _full_lat(v) -> bool:
    return v[0] <= -np.pi / 2 + 1e-9 and v[1] >= np.pi / 2 - 1e-9


def labels(prim_local: sp.Primitive, R: np.ndarray, placement: Optional[Dict], bbox_center) -> Dict:
    """{dims_mm, center_mm, rotation_canonical, segment} of a placed sample (placement None: unplaced, k = 1, t = 0)."""
    k = float(placement["scale_factor"]) if placement else 1.0
    t = (np.asarray(placement["center_mm"], float) - k * np.asarray(bbox_center, float)) if placement else np.zeros(3)
    prim = scaled(prim_local, k)
    out = dict(dims_mm=dims_mm(prim), center_mm=[float(v) for v in np.asarray(R, float) @ local_centre(prim) + t],
               rotation_canonical=rotation_canonical(prim, R))
    seg = {}
    if hasattr(prim, "u") and not _full(prim.u):
        seg["u"] = [float(prim.u[0]), float(prim.u[1])]
    if isinstance(prim, sp.Sphere) and not _full_lat(prim.v):
        seg["v"] = [float(prim.v[0]), float(prim.v[1])]
    if isinstance(prim, sp.Torus) and not _full(prim.v):
        seg["v"] = [float(prim.v[0]), float(prim.v[1])]
    if seg:
        out["segment"] = seg
    return out


# ------------------------------------------------------------------------------------------------ voxel draws -> world

def array_to_world(a, grid_mm: float, resolution: int) -> np.ndarray:
    """Continuous .bin array index (a0, a1, a2) = [z, -y, x], index a at world -h + (a + 0.5) v, -> world (x, y, z) mm."""
    a = np.asarray(a, float)
    h, v = grid_mm / 2.0, grid_mm / resolution
    return np.array([-h + (a[2] + 0.5) * v, h - (a[1] + 0.5) * v, -h + (a[0] + 0.5) * v])


def carve_cutter(record: Dict, grid_mm: float, resolution: int) -> sp.Primitive:
    """inscale.carve's cutter (array-index centre / radius / rotation) as a world-mm primitive. The voxel test was
    |rot^T (p_a - c_a)| <= r with p_a - c_a = A (p_w - c_w) / v, so the box axes in the world are the columns of A^T rot."""
    v = grid_mm / resolution
    centre = array_to_world(record["centre"], grid_mm, resolution)
    if record["cutter"] == "sphere":
        return sp.Placed(sp.Sphere(record["radius"] * v), rotation=np.eye(3), translation=centre)
    Rw = A_WORLD_TO_ARRAY.T @ np.asarray(record["rotation"], float)
    a = record["half_edge"] * v
    return sp.Placed(sp.Box([a, a, a]), rotation=Rw, translation=centre)


def carve_cutter_labels(record: Dict, grid_mm: float, resolution: int) -> Dict:
    """Labels of the cutter itself (the negative residual's own sample): topology, dims_mm, center_mm, rotation."""
    v = grid_mm / resolution
    centre = array_to_world(record["centre"], grid_mm, resolution)
    if record["cutter"] == "sphere":
        prim, R = sp.Sphere(record["radius"] * v), np.eye(3)
    else:
        a = record["half_edge"] * v
        prim, R = sp.Box([a, a, a]), A_WORLD_TO_ARRAY.T @ np.asarray(record["rotation"], float)
    return dict(topology=record["cutter"], R=R, dims_mm=dims_mm(prim), center_mm=[float(c) for c in centre],
                rotation_canonical=rotation_canonical(prim, R))


def cut_halfspaces(base_volume: np.ndarray, applied: List[Dict], grid_mm: float) -> List[sp.HalfSpace]:
    """The half spaces KEPT by inscale.apply_cuts' applied cuts, in world mm. apply_cuts records only (normal, keep): the
    threshold along the normal depends on the body's extent at the time of each cut, so the cuts are replayed on the clean
    base volume. Kept voxels satisfy n_a . a <= thr (a = array index); with a0 = (z + h)/v - 1/2, a1 = (h - y)/v - 1/2,
    a2 = (x + h)/v - 1/2 that is n_w . x <= v (thr - (h/v - 1/2) sum(n_a)), n_w = A^T n_a."""
    from . import inscale
    r = base_volume.shape[0]
    h, v = grid_mm / 2.0, grid_mm / r
    grid = np.indices(base_volume.shape).reshape(3, -1).T.astype(np.float64)
    volume = base_volume.copy()
    out = []
    for cut in applied:
        n = np.asarray(cut["normal"], float)
        occupied = volume.reshape(-1) > 0
        width = float(np.abs(n).sum())
        along = grid @ n
        lo, hi = along[occupied].min() - width / 2, along[occupied].max() + width / 2
        thr = lo + cut["keep"] * (hi - lo) + 1e-9
        out.append(sp.HalfSpace(A_WORLD_TO_ARRAY.T @ n, v * (thr - (h / v - 0.5) * n.sum())))
        volume, _ = inscale.apply_cuts(volume, [cut], min_voxels=0, min_thickness=0.0)
    return out


def hollow_inner(prim_placed: sp.Placed, wall_mm: float) -> Optional[sp.Primitive]:
    """The cavity of a hollow box / sphere as the same body shrunk by `wall_mm` (an exact inward offset for a box's faces
    and a ball; inscale.hollow's voxel erosion is only that on axis-aligned faces, which is why a --with-field hollow
    sample is re-voxelised from this solid). None when the wall eats the body."""
    p = prim_placed.primitive
    if isinstance(p, sp.Box):
        half = [hh - wall_mm for hh in p.half]
        inner = sp.Box(half) if min(half) > 0 else None
    elif isinstance(p, sp.Sphere) and _full(p.u) and _full_lat(p.v):
        inner = sp.Sphere(p.radius - wall_mm) if p.radius > wall_mm else None
    else:
        return None
    return None if inner is None else sp.Placed(inner, rotation=prim_placed.rotation, translation=prim_placed.translation)


# ------------------------------------------------------------------------------------------------ FreeCAD

def build_stl(program: sp.Primitive, out_stl: Path, freecad: Optional[str] = None, deflection: float = DEFLECTION_MM) -> Dict:
    """Build `program` with FreeCAD Part (reco_field/export.build_program) and write its STL to `out_stl`.
    Returns {volume, valid, n_solids}; raises if FreeCAD fails or the result is invalid. With more than one solid (a
    carve or cut leaving a crumb) the builder exports the biggest; n_solids records it."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from reco_field.export import build_program, program_to_dict
    with tempfile.TemporaryDirectory(prefix="solid_") as tmp:
        res = build_program(program_to_dict(program), tmp, "solid", deflection=deflection, freecad=freecad)
        if not res.valid:
            raise RuntimeError(f"FreeCAD built an invalid solid ({res.n_solids} solid(s))")
        Path(out_stl).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(res.stl, out_stl)
    return dict(volume=float(res.volume), valid=bool(res.valid), n_solids=int(res.n_solids))
