#!/usr/bin/env python3
"""
The first guess by hand: shapes the user places roughly (coarse sliders in the GUI) before the algorithm starts.

    shapes = [UserShape("add", "box", (10, 0, 5), (0, 0, 0, 1), {"x_half": 20, "y_half": 5, "z_half": 5}, "web")]
    save_first_guess("first_guess.json", shapes)
    refined = refine(shapes, vertices, faces)          # a rough guess, not the truth: snapped and refitted
    config = ResidualConfig(first_guess=[r.shape.as_dict() for r in refined if r.passes])

No Qt here. The GUI's panel (reco_gui/first_guess_panel.py) uses the scales (`Scales`: what a slider position means),
the live hint (`inside_share`) and the pick (`ray_mesh_hit`).

THE CONTRACT WITH THE LOOP (reco_engine/residual_loop.py implements it; the GUI passes the keyword only when the
config has the field):

  ResidualConfig.first_guess: None (default: the loop as it is) | a list | a path (str / Path) to a first_guess.json.
    A list holds one entry per shape, in the user's order; an entry is a `UserShape`, or its dict
        {"op": "add" | "cut", "kind": "box" | "cylinder" | "cone" | "sphere" | "torus" | <a tool class>,
         "center_mm": [x, y, z], "quaternion": [qx, qy, qz, qw], "dims_mm": {...}, "label": str or null,
         "u": [u0, u1] or null, "v": [v0, v1] or null}
    (the keys of result_schema.Primitive, plus "op"), or a tuple (op, Primitive). `load_first_guess(path)` reads the
    file; `as_operations(first_guess)` turns any of the three forms into [(op, Primitive), ...].
  Semantics:
    1. The shapes are ROUGH. The loop calls `refine(shapes, vertices, faces, exact_faces=...)` (or takes shapes
       that the GUI refined already: an entry with "refined": true is not refined again) before it uses them.
    2. The refined shapes are the loop's FIRST operations, in the given order when "ordered" is true in the file /
       `ordered=True` in `as_operations`, else every add before every cut (the order within each kept).
       They replace the starting-shape stage and the sub-loops' first add when at least one add passes; when
       none passes the loop starts as without a first guess.
    3. Each shape is evaluated and validated like any candidate of the loop (its gain, its precision, the surface
       check of a new boolean, the FreeCAD build when `grade_each_step`): a shape that fails is REPORTED AND DROPPED,
       never forced.
    4. Steps: per shape one "candidate" step and then "accepted" or "rejected", with info["source"] = "user" and
       info["first_guess"] = the entry's index; the primitive's sources get StageSource("type", "user", 1.0), so
       the program says which operations a human placed. After the last shape the loop goes on with its residual.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine.result_schema import Primitive, StageSource  # noqa: E402

SCHEMA = 1
FILE_NAME = "first_guess.json"
KINDS = ("box", "cylinder", "cone", "sphere", "torus")           # the Part-native primitives of the palette
DIMS = {"box": ("x_half", "y_half", "z_half"), "cylinder": ("radius", "half_length"),
        "cone": ("radius1", "radius2", "half_length"), "sphere": ("radius",), "torus": ("major_radius", "minor_radius")}

# the scales of the coarse sliders
POSITION_STEPS = 50                 # a position step is 1/50 of the part's extent along that axis
POSITION_MARGIN = 0.25              # the sliders reach this share of the extent beyond the part's box
SIZE_STEPS = 40                     # a size slider: logarithmic, from diagonal / 100 to the diagonal
SIZE_MIN_SHARE = 0.01
ANGLE_STEP = 15.0                   # degrees
# the window `refine` may move a shape in, around the user's values
REFINE_POSITION_STEPS = 3.0         # position: within this many slider steps
REFINE_SIZE_FACTOR = 1.5            # dimensions: within this factor
REFINE_ANGLE = 15.0                 # angles: within this many degrees
HINT_POINTS = 2000                  # sampled points of a shape for the live hint
# the checks of `refine` (the loop applies its own again): the config's values when it has them
CHECK_DEFAULTS = dict(member_min_precision=0.95, boolean_min_surface=0.2)


@dataclass
class UserShape:
    """A shape placed by hand. The same pose and dimension conventions as result_schema.Primitive."""
    op: str                                             # "add" | "cut"
    kind: str                                           # one of KINDS (or a tool class of reco_engine.tools)
    center_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    dims_mm: Dict[str, float] = field(default_factory=dict)
    label: Optional[str] = None
    u: Optional[Tuple[float, float]] = None
    v: Optional[Tuple[float, float]] = None
    # "cover": the shape bounds the part (or the sub-part inside region_mm = (lo, hi), mm) - an over-cover by design whose
    # excess the loop cuts away; it is judged by the material it holds (residual_loop._cover_check), not by its net gain
    role: Optional[str] = None
    region_mm: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None

    def primitive(self) -> Primitive:
        return Primitive(kind=self.kind, quaternion=tuple(float(x) for x in self.quaternion),
                         center_mm=tuple(float(x) for x in self.center_mm),
                         dims_mm={k: float(v) for k, v in self.dims_mm.items()}, u=self.u, v=self.v, confidence=1.0,
                         sources=[StageSource("type", "user", 1.0)], label=self.label)

    @classmethod
    def from_primitive(cls, op: str, p: Primitive) -> "UserShape":
        return cls(op=op, kind=p.kind, center_mm=tuple(float(x) for x in p.center_mm),
                   quaternion=tuple(float(x) for x in p.quaternion), dims_mm={k: float(v) for k, v in (p.dims_mm or {}).items()},
                   label=p.label, u=tuple(p.u) if p.u else None, v=tuple(p.v) if p.v else None)

    def as_dict(self) -> dict:
        return dict(op=self.op, kind=self.kind, center_mm=[float(x) for x in self.center_mm],
                    quaternion=[float(x) for x in self.quaternion], dims_mm={k: float(v) for k, v in self.dims_mm.items()},
                    label=self.label, u=list(self.u) if self.u else None, v=list(self.v) if self.v else None,
                    role=self.role, region_mm=[list(map(float, b)) for b in self.region_mm] if self.region_mm else None)

    @classmethod
    def from_dict(cls, d: dict) -> "UserShape":
        if d.get("op") not in ("add", "cut"):
            raise ValueError(f"a shape's op must be 'add' or 'cut', got {d.get('op')!r}")
        return cls(op=d["op"], kind=str(d["kind"]), center_mm=tuple(float(x) for x in d.get("center_mm", (0, 0, 0))),
                   quaternion=tuple(float(x) for x in d.get("quaternion", (0, 0, 0, 1))),
                   dims_mm={k: float(v) for k, v in (d.get("dims_mm") or {}).items()}, label=d.get("label"),
                   u=tuple(d["u"]) if d.get("u") else None, v=tuple(d["v"]) if d.get("v") else None,
                   role=d.get("role"),
                   region_mm=tuple(tuple(float(x) for x in b) for b in d["region_mm"]) if d.get("region_mm") else None)


# ------------------------------------------------------------------------------------------------ file, loop

def save_first_guess(path, shapes: Sequence[UserShape], ordered: bool = False, input_path=None, refined: bool = False) -> Path:
    """Write the shapes as JSON (`path`: the file, or a directory that gets first_guess.json)."""
    path = Path(path)
    if path.is_dir():
        path = path / FILE_NAME
    entries = [dict(s.as_dict(), refined=bool(refined)) for s in shapes]
    path.write_text(json.dumps(dict(schema=SCHEMA, input=str(input_path) if input_path else None, ordered=bool(ordered),
                                    shapes=entries), indent=1))
    return path


def load_first_guess(path) -> Tuple[List[UserShape], dict]:
    """(shapes, the file's other fields: ordered, input, refined flags) of a first_guess.json (or its directory)."""
    path = Path(path)
    if path.is_dir():
        path = path / FILE_NAME
    d = json.loads(path.read_text())
    entries = d.get("shapes") or []
    return ([UserShape.from_dict(e) for e in entries],
            dict(ordered=bool(d.get("ordered")), input=d.get("input"), refined=[bool(e.get("refined")) for e in entries]))


def as_operations(first_guess, ordered: Optional[bool] = None) -> List[Tuple[str, Primitive]]:
    """[(op, Primitive), ...] of a first guess in any form of the contract (a path, a list of UserShape / dict /
    (op, Primitive)), in the order the loop applies them: the given one when `ordered`, else adds before cuts."""
    if first_guess is None:
        return []
    if isinstance(first_guess, (str, Path)):
        shapes, meta = load_first_guess(first_guess)
        ordered = meta["ordered"] if ordered is None else ordered
        entries = list(shapes)
    else:
        entries = list(first_guess)
    out = []
    for e in entries:
        if isinstance(e, UserShape):
            out.append((e.op, e.primitive()))
        elif isinstance(e, dict):
            s = UserShape.from_dict(e)
            out.append((s.op, s.primitive()))
        else:
            op, p = e
            out.append((str(op), p))
    if not ordered:
        out = [x for x in out if x[0] == "add"] + [x for x in out if x[0] != "add"]
    return out


# ------------------------------------------------------------------------------------------------ slider scales

def euler_quaternion(angles_deg) -> Tuple[float, float, float, float]:
    """(qx, qy, qz, qw) of the three slider angles: turns about the world x, then y, then z axis (degrees)."""
    from scipy.spatial.transform import Rotation
    return tuple(float(x) for x in Rotation.from_euler("xyz", np.asarray(angles_deg, float), degrees=True).as_quat())


def quaternion_euler(q) -> Tuple[float, float, float]:
    from scipy.spatial.transform import Rotation
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                              # gimbal lock: any of the equivalent triples
        return tuple(float(x) for x in Rotation.from_quat(np.asarray(q, float)).as_euler("xyz", degrees=True))


def matrix_quaternion(R) -> Tuple[float, float, float, float]:
    from reco_engine.result_to_program import quaternion_of
    return quaternion_of(np.asarray(R, float))


@dataclass
class Scales:
    """What a slider position means for one part: coarse on purpose (the algorithm refines)."""
    lo: np.ndarray                  # the part's bounding box
    hi: np.ndarray

    @classmethod
    def of(cls, vertices) -> "Scales":
        V = np.asarray(vertices, float).reshape(-1, 3)
        return cls(V.min(axis=0), V.max(axis=0))

    @property
    def extent(self) -> np.ndarray:
        return np.maximum(self.hi - self.lo, 1e-9)

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.extent))

    @property
    def centre(self) -> np.ndarray:
        return (self.lo + self.hi) / 2

    # position: linear, step = extent / 50, from the box minus the margin to the box plus the margin
    def position_step(self, axis: int) -> float:
        return float(self.extent[axis] / POSITION_STEPS)

    def position_count(self) -> int:
        """The slider runs over 0 .. count."""
        return int(round(POSITION_STEPS * (1 + 2 * POSITION_MARGIN)))

    def position(self, axis: int, index: int) -> float:
        index = min(max(int(index), 0), self.position_count())
        return float(self.lo[axis] - POSITION_MARGIN * self.extent[axis] + index * self.position_step(axis))

    def position_index(self, axis: int, value: float) -> int:
        i = round((float(value) - (self.lo[axis] - POSITION_MARGIN * self.extent[axis])) / self.position_step(axis))
        return int(min(max(i, 0), self.position_count()))

    # size: logarithmic, 40 steps from diagonal / 100 to the diagonal
    def size(self, index: int) -> float:
        index = min(max(int(index), 0), SIZE_STEPS)
        return float(SIZE_MIN_SHARE * self.diagonal * (1.0 / SIZE_MIN_SHARE) ** (index / SIZE_STEPS))

    def size_index(self, value: float) -> int:
        value = max(float(value), 1e-300)
        i = round(SIZE_STEPS * math.log(value / (SIZE_MIN_SHARE * self.diagonal)) / math.log(1.0 / SIZE_MIN_SHARE))
        return int(min(max(i, 0), SIZE_STEPS))

    # angle: steps of 15 degrees, -180 .. 180
    @staticmethod
    def angle_count() -> int:
        return int(round(360 / ANGLE_STEP))

    @staticmethod
    def angle(index: int) -> float:
        return float(-180.0 + min(max(int(index), 0), Scales.angle_count()) * ANGLE_STEP)

    @staticmethod
    def angle_index(value: float) -> int:
        value = (float(value) + 180.0) % 360.0 - 180.0
        return int(round((value + 180.0) / ANGLE_STEP))

    def default_shape(self, kind: str, op: str = "add", centre=None, label: Optional[str] = None) -> UserShape:
        """A shape of a fifth of the part's diagonal, at `centre` (default: the part's centre), on the slider grid."""
        if kind not in DIMS:
            raise ValueError(f"no default dimensions for a {kind!r}")
        c = self.centre if centre is None else np.asarray(centre, float)
        s = self.size(self.size_index(0.1 * self.diagonal))
        small = self.size(self.size_index(0.03 * self.diagonal))
        dims = {"box": dict(x_half=s, y_half=s, z_half=s), "cylinder": dict(radius=s / 2, half_length=s),
                "cone": dict(radius1=s, radius2=s / 2, half_length=s), "sphere": dict(radius=s),
                "torus": dict(major_radius=s, minor_radius=small)}[kind]
        return UserShape(op=op, kind=kind, center_mm=tuple(float(x) for x in c), dims_mm=dims, label=label)


def part_frames(vertices, faces=None, max_directions: int = 6, max_faces: int = 200_000) -> List[Tuple[str, np.ndarray]]:
    """The "align with the part" choices as (name, rotation matrix with the direction as the local z axis): the world
    axes, the axes of the part's oriented bounding box (principal axes of its vertices) and, when the mesh is small
    enough for it to be cheap, the member directions of reco_engine.ibeam.candidate_directions."""
    V = np.asarray(vertices, float).reshape(-1, 3)
    out = [(f"world {n}", _frame_with_z(a)) for n, a in zip("xyz", np.eye(3))]
    try:
        c = V - V.mean(axis=0)
        _, _, vt = np.linalg.svd(c[:: max(1, len(c) // 20000)], full_matrices=False)
        R = vt.T
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1
        out.append(("bounding box", R))
        out += [(f"box axis {i + 1}", _frame_with_z(R[:, i])) for i in range(3)]
    except np.linalg.LinAlgError:
        pass
    if faces is not None and len(faces) <= max_faces:
        try:
            from reco_engine.ibeam import candidate_directions
            for i, d in enumerate(candidate_directions(V, np.asarray(faces), max_dirs=max_directions)[:max_directions]):
                out.append((f"member direction {i + 1}", _frame_with_z(d)))
        except Exception:                                           # advisory: the other choices stay
            pass
    return out


def _frame_with_z(axis) -> np.ndarray:
    z = np.asarray(axis, float)
    z = z / (np.linalg.norm(z) or 1.0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], axis=1)


# ------------------------------------------------------------------------------------------------ geometry

def shape_sdf(shape):
    """The placed SDF solid of a UserShape or a Primitive (reco_engine.result_to_program.primitive_to_sp)."""
    from reco_engine.result_to_program import primitive_to_sp
    return primitive_to_sp(shape.primitive() if isinstance(shape, UserShape) else shape)


def shape_radius(shape) -> float:
    """The radius of a sphere about the shape's centre that holds it."""
    d = shape.dims_mm
    k = shape.kind
    if k == "box":
        return float(np.linalg.norm([d["x_half"], d["y_half"], d["z_half"]]))
    if k == "sphere":
        return float(d["radius"])
    if k == "cylinder":
        return float(np.hypot(d["radius"], d["half_length"]))
    if k == "cone":
        return float(np.hypot(max(d["radius1"], d["radius2"]), d["half_length"]))
    if k == "torus":
        return float(d["major_radius"] + d["minor_radius"])
    return float(max(d.values())) * 2.0 if d else 1.0


def sample_inside(shape, n: int = HINT_POINTS, seed: int = 0) -> np.ndarray:
    """About `n` points uniformly inside the shape (seeded; rejection from the cube around it)."""
    rng = np.random.default_rng(seed)
    solid, r, c = shape_sdf(shape), shape_radius(shape), np.asarray(shape.center_mm, float)
    out, got = [], 0
    for _ in range(12):
        pts = c + (rng.random((4 * n, 3)) * 2 - 1) * r
        pts = pts[solid.sdf(pts) <= 0]
        out.append(pts)
        got += len(pts)
        if got >= n:
            break
    pts = np.vstack(out) if out else np.empty((0, 3))
    return pts[:n]


def inside_mesh(vertices, faces, points) -> np.ndarray:
    """Which points lie inside the mesh, by its winding number (libigl's fast one; trimesh without libigl)."""
    V, F = np.ascontiguousarray(vertices, np.float64), np.ascontiguousarray(faces, np.int64)
    P = np.ascontiguousarray(points, np.float64).reshape(-1, 3)
    if not len(P):
        return np.zeros(0, bool)
    try:
        import igl
        fn = getattr(igl, "fast_winding_number", None) or igl.winding_number
        return np.abs(np.asarray(fn(V, F, P)).reshape(-1)) > 0.5
    except ImportError:
        import trimesh
        return np.asarray(trimesh.Trimesh(V, F, process=False).contains(P), bool)


def inside_share(shape, vertices, faces, n: int = HINT_POINTS, seed: int = 0) -> Optional[float]:
    """The live hint of a rough shape: the share of `n` sampled points of it that lie inside the input (an add) or
    outside it (a cut). None for a shape without volume."""
    pts = sample_inside(shape, n, seed)
    if not len(pts):
        return None
    inside = float(inside_mesh(vertices, faces, pts).mean())
    return inside if shape.op == "add" else 1.0 - inside


HINT_LABELS = ((0.95, "good"), (0.8, "fair"), (0.5, "rough"), (0.0, "poor"))


def hint_label(share: Optional[float]) -> str:
    if share is None:
        return "no volume"
    return next(text for limit, text in HINT_LABELS if share >= limit)


def hint_colour(share: Optional[float]) -> Tuple[float, float, float]:
    """One hue (blue), from pale (poor) to saturated dark (good): readable without telling red from green."""
    t = 0.0 if share is None else min(max(float(share), 0.0), 1.0)
    pale, dark = np.array([0.80, 0.87, 0.96]), np.array([0.03, 0.25, 0.55])
    return tuple(float(x) for x in pale + (dark - pale) * t ** 2)


def ray_mesh_hit(origin, direction, vertices, faces) -> Optional[np.ndarray]:
    """The first point of the mesh on the ray (Moeller-Trumbore over all triangles), None when it misses."""
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.0)
    tri = np.asarray(vertices, float)[np.asarray(faces)]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    p = np.cross(d, e2)
    det = np.einsum("ij,ij->i", e1, p)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = o - tri[:, 0]
    u = np.einsum("ij,ij->i", s, p) * inv
    q = np.cross(s, e1)
    v = (q @ d) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    if not hit.any():
        return None
    return o + d * float(t[hit].min())


def drop_point(origin, direction, vertices, faces, centre=None) -> np.ndarray:
    """Where a dropped shape lands: on the input's surface under the cursor; if the ray misses, on the ray at the depth
    of the part's centre."""
    hit = ray_mesh_hit(origin, direction, vertices, faces)
    if hit is not None:
        return hit
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    d = d / (np.linalg.norm(d) or 1.0)
    c = np.asarray(vertices, float).reshape(-1, 3).mean(axis=0) if centre is None else np.asarray(centre, float)
    return o + d * float((c - o) @ d)


def surface_samples(vertices, faces, n: int = 6000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """(points, outward normals) on the mesh, area-weighted and seeded."""
    V, F = np.asarray(vertices, float), np.asarray(faces)
    tri = V[F]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    rng = np.random.default_rng(seed)
    f = rng.choice(len(F), size=n, p=area / area.sum())
    r1, r2 = rng.random((n, 1)), rng.random((n, 1))
    flip = (r1 + r2) > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    pts = tri[f, 0] + (tri[f, 1] - tri[f, 0]) * r1 + (tri[f, 2] - tri[f, 0]) * r2
    return pts, cross[f] / np.maximum(area[f], 1e-300)[:, None]


# ------------------------------------------------------------------------------------------------ refine

@dataclass
class Refined:
    """One shape after `refine`: what changed, by how much, and whether it would pass."""
    shape: UserShape                                    # the refined shape
    given: UserShape                                    # what the user placed
    moved_mm: float = 0.0                               # distance of the centres
    moved_steps: float = 0.0                            # ... in slider steps (the largest axis ratio)
    turned_deg: float = 0.0
    scaled: Dict[str, float] = field(default_factory=dict)       # per dimension: refined / given
    snapped: List[dict] = field(default_factory=list)   # what feature_stage.snap_to_faces moved
    points: int = 0                                     # input surface points the refit used
    at_limit: List[str] = field(default_factory=list)   # parameters that ended on the edge of the window
    precision: Optional[float] = None                   # share of the shape inside the input (add) / outside it (cut)
    surface: Optional[float] = None                     # share of the nearby input surface on the shape
    precision_given: Optional[float] = None
    surface_given: Optional[float] = None
    passes: bool = False
    reason: str = ""
    thresholds: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(shape=self.shape.as_dict(), given=self.given.as_dict(), moved_mm=self.moved_mm,
                    moved_steps=self.moved_steps, turned_deg=self.turned_deg, scaled=dict(self.scaled),
                    snapped=list(self.snapped), points=self.points, at_limit=list(self.at_limit),
                    precision=self.precision, surface=self.surface, precision_given=self.precision_given,
                    surface_given=self.surface_given, passes=self.passes, reason=self.reason,
                    thresholds=dict(self.thresholds))

    def summary(self) -> str:
        scale = ", ".join(f"{k} x{v:.2f}" for k, v in self.scaled.items() if abs(v - 1.0) > 0.005)
        parts = [f"moved {self.moved_mm:.3g} mm ({self.moved_steps:.1f} steps)", f"turned {self.turned_deg:.1f} deg",
                 scale or "sizes kept"]
        if self.snapped:
            parts.append(f"snapped to {len(self.snapped)} exact face(s)")
        if self.at_limit:
            parts.append("at the window's edge: " + ", ".join(self.at_limit))
        text = f"{self.shape.op} {self.shape.kind}" + (f" '{self.shape.label}'" if self.shape.label else "") + ": " + "; ".join(parts)
        if self.precision is not None:
            text += f"; precision {self.precision_given:.2f} -> {self.precision:.2f}"
        if self.surface is not None:
            text += f", surface {self.surface_given:.2f} -> {self.surface:.2f}"
        return text + ("; passes" if self.passes else f"; FAILS: {self.reason}")


def check_thresholds(config=None) -> Dict[str, float]:
    """The thresholds of `refine`'s checks: the ResidualConfig's (the given one, else its defaults), else CHECK_DEFAULTS."""
    out = dict(CHECK_DEFAULTS)
    try:
        if config is None:
            from reco_engine.residual_loop import ResidualConfig
            config = ResidualConfig()
        for k in out:
            if getattr(config, k, None) is not None:
                out[k] = float(getattr(config, k))
    except Exception:
        pass
    return out


def _gradient(solid, points, h: float) -> np.ndarray:
    """The unit gradient of the solid's signed distance at the points (central differences): its outward direction."""
    g = np.stack([solid.sdf(points + h * e) - solid.sdf(points - h * e) for e in np.eye(3)], axis=1)
    return g / np.maximum(np.linalg.norm(g, axis=1), 1e-300)[:, None]


def _surface_share(solid, points, tol: float, window: float) -> Optional[float]:
    d = np.abs(solid.sdf(points))
    near = d <= window
    return float((d[near] <= tol).mean()) if near.sum() >= 12 else None


def refine(shapes: Sequence[UserShape], vertices, faces, exact_faces: Optional[Sequence[dict]] = None,
           scales: Optional[Scales] = None, config=None, n_surface: int = 6000, seed: int = 0) -> List[Refined]:
    """The hand-made shapes as the algorithm may use them: a ROUGH guess, not the truth.

    Per shape: (1) its free parameters are refitted to the input's surface near it - the residuals are the shape's
    signed distance at the input's surface points within the window that face the way the shape's surface does (an
    add: the normals agree; a cut: they are opposite, the wall of the hole), a robust fit (soft-L1, then Tukey weights, as
    reco_field.fit does) - INSIDE a window around the user's values: the centre within REFINE_POSITION_STEPS slider
    steps per axis, every dimension within a factor REFINE_SIZE_FACTOR, the frame within REFINE_ANGLE degrees;
    (2) with `exact_faces` (reco_field.exact_check.exact_faces(path)["faces"], a STEP input) it is put onto the
    input's exact faces where they are near (reco_engine.feature_stage.snap_to_faces); (3) the checks: precision
    (share of the shape inside the input for an add, outside it for a cut) >= member_min_precision and the share of
    the nearby input surface on it >= boolean_min_surface. The loop evaluates the shape again with its own checks.

    `vertices`, `faces`: the input mesh (or pass a SurfaceField's). Returns one `Refined` per shape, in order."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    V, F = np.asarray(vertices, float), np.asarray(faces)
    scales = scales or Scales.of(V)
    thresholds = check_thresholds(config)
    tol = 0.004 * scales.diagonal                                   # reco_field.fit.tolerance
    points, normals = surface_samples(V, F, n_surface, seed)
    step = np.array([scales.position_step(a) for a in range(3)])
    out = []
    for given in shapes:
        names = [k for k in DIMS.get(given.kind, tuple(given.dims_mm)) if k in given.dims_mm]
        c0 = np.asarray(given.center_mm, float)
        R0 = Rotation.from_quat(np.asarray(given.quaternion, float))
        d0 = np.array([given.dims_mm[k] for k in names], float)

        def build(th, given=given, names=names, c0=c0, R0=R0, d0=d0):
            s = copy.copy(given)
            s.center_mm = tuple(c0 + th[:3])
            s.quaternion = tuple((Rotation.from_rotvec(th[3:6]) * R0).as_quat())
            s.dims_mm = dict(given.dims_mm, **{k: float(v) for k, v in zip(names, d0 * np.exp(th[6:]))})
            return s

        x0 = np.zeros(6 + len(names))
        angle = math.radians(REFINE_ANGLE) / math.sqrt(3.0)             # the rotation vector stays within the angle
        hi = np.r_[REFINE_POSITION_STEPS * step, [angle] * 3, [math.log(REFINE_SIZE_FACTOR)] * len(names)]
        solid0 = shape_sdf(given)
        window = float(REFINE_POSITION_STEPS * step.max() + (REFINE_SIZE_FACTOR - 1.0) * (d0.max() if len(d0) else 0.0))
        near = np.abs(solid0.sdf(points)) <= window
        # ... and only surface that faces the way the shape's does: an add's surface is the input's (the normals
        # agree), a cut's is the wall of the hole it makes (they are opposite). The faces a cutter's caps stick out
        # of and the walls beside it are no evidence for it.
        side = 1.0 if given.op == "add" else -1.0
        agree = np.einsum("ij,ij->i", _gradient(solid0, points[near], 1e-4 * scales.diagonal), normals[near]) * side
        near[np.nonzero(near)[0][agree < 0.5]] = False
        P = points[near]
        th = x0
        if len(P) >= max(12, 2 * len(x0)):
            try:
                th = least_squares(lambda t: shape_sdf(build(t)).sdf(P), x0, bounds=(-hi, hi), loss="soft_l1",
                                   f_scale=2 * tol, max_nfev=80).x
                for _ in range(4):
                    r = shape_sdf(build(th)).sdf(P)
                    w = np.clip(1.0 - (r / (3.0 * tol)) ** 2, 0.0, 1.0) ** 2
                    if (w > 0).sum() < max(12, 2 * len(x0)):
                        break
                    sw = np.sqrt(w)
                    th = least_squares(lambda t: sw * shape_sdf(build(t)).sdf(P), np.clip(th, -hi * 0.999999, hi * 0.999999),
                                       bounds=(-hi, hi), max_nfev=40).x
            except Exception:                                       # a degenerate fit: the user's shape stays
                th = x0
        shape = build(th)
        shape.center_mm = tuple(float(x) for x in shape.center_mm)
        shape.quaternion = tuple(float(x) for x in shape.quaternion)
        labels = ["x", "y", "z", "turn x", "turn y", "turn z"] + names
        at_limit = [labels[i] for i in range(len(th)) if hi[i] > 0 and abs(th[i]) >= 0.98 * hi[i]]
        snapped = []
        if exact_faces:
            try:
                from reco_engine.feature_stage import snap_to_faces
                from reco_engine.result_schema import Operation, ReconstructionResult
                res = ReconstructionResult(input_path="", operations=[Operation(shape.op, shape.primitive())])
                snapped = list(snap_to_faces(res, list(exact_faces)) or [])
                if snapped:
                    shape = UserShape.from_primitive(shape.op, res.operations[0].primitive)
                    shape.label = given.label
            except Exception as exc:
                snapped = [dict(error=str(exc)[-200:])]
        c1 = np.asarray(shape.center_mm, float)
        turn = (Rotation.from_quat(np.asarray(shape.quaternion, float)) * R0.inv()).magnitude()
        r = Refined(shape=shape, given=given, moved_mm=float(np.linalg.norm(c1 - c0)),
                    moved_steps=float(np.max(np.abs(c1 - c0) / step)), turned_deg=float(np.degrees(turn)),
                    scaled={k: float(shape.dims_mm[k] / given.dims_mm[k]) for k in names if given.dims_mm[k] > 0},
                    snapped=[_plain(m) for m in snapped], points=int(len(P)), at_limit=at_limit, thresholds=thresholds)
        r.precision_given, r.precision = inside_share(given, V, F, seed=seed), inside_share(shape, V, F, seed=seed)
        r.surface_given = _surface_share(solid0, points, tol, window)
        r.surface = _surface_share(shape_sdf(shape), points, tol, window)
        reasons = []
        if r.precision is None or r.precision < thresholds["member_min_precision"]:
            what = "inside the input" if shape.op == "add" else "of the removal outside the input"
            reasons.append(f"only {0.0 if r.precision is None else r.precision:.0%} {what} "
                           f"(needs {thresholds['member_min_precision']:.0%})")
        if r.surface is None or r.surface < thresholds["boolean_min_surface"]:
            reasons.append(f"only {0.0 if r.surface is None else r.surface:.0%} of the nearby surface on it "
                           f"(needs {thresholds['boolean_min_surface']:.0%})")
        r.passes, r.reason = not reasons, "; ".join(reasons)
        out.append(r)
    return out


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------------------------------- the starting part
# The GUI's starting-part dialog (reco_gui/start_part_dialog.py; the user, 2026-10-29: "define a starting part by a
# basic shape type and the bounding part or sub-part, or by exact location and size, via a dialog"). Both give a
# UserShape "add"; the dialog passes it as ResidualConfig.first_guess (refined by the loop, or "refined": true when the
# user wants it exactly as given).
AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def _frame_z(a) -> np.ndarray:
    """A right-handed frame (columns) whose local z is the unit vector `a`."""
    a = np.asarray(a, float) / max(float(np.linalg.norm(a)), 1e-12)
    h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(h, a)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(a, x), a])


def region_points(vertices, faces=None, lo=None, hi=None, n: int = 20000, seed: int = 0) -> np.ndarray:
    """The part's points inside the box lo..hi (mm; None: the whole part - its vertices): the vertices and `n`
    area-uniform surface samples (a flat face has only its corner vertices) that lie in the box."""
    V = np.asarray(vertices, float)
    if lo is None and hi is None:
        return V
    pts = V
    if faces is not None and len(faces):
        import trimesh
        mesh = trimesh.Trimesh(V, np.asarray(faces), process=False)
        pts = np.vstack([V, trimesh.sample.sample_surface(mesh, n, seed=seed)[0]])
    lo = np.full(3, -np.inf) if lo is None else np.asarray(lo, float)
    hi = np.full(3, np.inf) if hi is None else np.asarray(hi, float)
    return pts[np.all((pts >= lo) & (pts <= hi), axis=1)]


def _axial(pts: np.ndarray, a) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(frame, centre, local points, radius about the axis line, half length) of the tightest axis line along `a`
    through the middle of the points' cross-section."""
    R = _frame_z(a)
    loc = pts @ R
    mid = (loc.min(axis=0) + loc.max(axis=0)) / 2
    rho = np.hypot(loc[:, 0] - mid[0], loc[:, 1] - mid[1])
    centre = R @ mid
    return R, centre, loc - mid, float(rho.max()), float((loc[:, 2].max() - loc[:, 2].min()) / 2)


def _principal(pts: np.ndarray) -> List[np.ndarray]:
    X = pts - pts.mean(axis=0)
    _, vecs = np.linalg.eigh(X.T @ X)
    return [vecs[:, i] for i in range(3)]


def bounding_shape(kind: str, vertices, faces=None, lo=None, hi=None, axis: str = "auto", label: Optional[str] = None
                   ) -> UserShape:
    """A shape of `kind` that bounds the part, or the sub-part inside the box lo..hi (mm), as a starting part.
    `axis`: "x" | "y" | "z" | "auto" - a box's frame ("auto": the oriented box) and the axis of a cylinder, cone or
    torus ("auto": the world and principal axes, the smallest bounding cylinder wins). A cone's radii are the bounds
    of its lower and upper quarter; a torus's major radius the middle of the radial range. ROUGH by design: the loop
    refines it unless the user takes it as given. The shape is a cover (role "cover", the region in region_mm): the loop
    keeps it when it holds the material of the part (or region) and cuts its excess away."""
    shape = _bounding_shape(kind, vertices, faces, lo, hi, axis, label)
    shape.role = "cover"
    if lo is not None or hi is not None:
        V = np.asarray(vertices, float)
        lo_r = V.min(axis=0) if lo is None else np.asarray(lo, float)
        hi_r = V.max(axis=0) if hi is None else np.asarray(hi, float)
        shape.region_mm = (tuple(map(float, lo_r)), tuple(map(float, hi_r)))
    return shape


def _bounding_shape(kind: str, vertices, faces, lo, hi, axis: str, label: Optional[str]) -> UserShape:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    pts = region_points(vertices, faces, lo, hi)
    if len(pts) < 4:
        raise ValueError("the region holds no part of the part (fewer than 4 points)")
    where = "the part" if lo is None and hi is None else "the sub-part"
    label = label or f"starting part: {kind} around {where}"
    if kind == "box":
        if axis == "auto":
            import trimesh
            T, ext = trimesh.bounds.oriented_bounds(pts)
            inv = np.linalg.inv(T)
            R, centre, half = inv[:3, :3], inv[:3, 3], np.asarray(ext, float) / 2
        else:
            lo_p, hi_p = pts.min(axis=0), pts.max(axis=0)
            R, centre, half = np.eye(3), (lo_p + hi_p) / 2, (hi_p - lo_p) / 2
        half = np.maximum(half, 1e-6)
        return UserShape("add", "box", tuple(map(float, centre)), matrix_quaternion(R),
                         dict(x_half=float(half[0]), y_half=float(half[1]), z_half=float(half[2])), label)
    if kind == "sphere":
        lo_p, hi_p = pts.min(axis=0), pts.max(axis=0)
        c = (lo_p + hi_p) / 2
        return UserShape("add", "sphere", tuple(map(float, c)), (0.0, 0.0, 0.0, 1.0),
                         dict(radius=float(np.linalg.norm(pts - c, axis=1).max())), label)
    cands = [AXES[axis]] if axis in AXES else list(AXES.values()) + _principal(pts)
    best = min((_axial(pts, a) for a in cands), key=lambda t: t[3] ** 2 * t[4])
    R, centre, loc, r, h = best
    q = matrix_quaternion(R)
    if kind == "cylinder":
        return UserShape("add", "cylinder", tuple(map(float, centre)), q, dict(radius=r, half_length=max(h, 1e-6)), label)
    rho = np.hypot(loc[:, 0], loc[:, 1])
    if kind == "cone":
        z = loc[:, 2]
        low, top = rho[z <= z.min() + 0.5 * h], rho[z >= z.max() - 0.5 * h]
        r1 = float(low.max()) if len(low) else r
        r2 = float(top.max()) if len(top) else r
        return UserShape("add", "cone", tuple(map(float, centre)), q,
                         dict(radius1=max(r1, 1e-3), radius2=max(r2, 1e-3), half_length=max(h, 1e-6)), label)
    major = float((rho.max() + rho.min()) / 2)
    minor = float(max((rho.max() - rho.min()) / 2, h, 1e-3))
    return UserShape("add", "torus", tuple(map(float, centre)), q, dict(major_radius=max(major, minor * 1.01), minor_radius=minor),
                     label)


SIZES = {"box": ("length", "width", "height"), "cylinder": ("radius", "height"), "cone": ("radius 1 (bottom)", "radius 2 (top)", "height"),
         "sphere": ("radius",), "torus": ("major radius", "minor radius")}


def exact_shape(kind: str, center, angles_deg=(0.0, 0.0, 0.0), sizes=(), label: Optional[str] = None) -> UserShape:
    """A shape at an exact location and size: `center` (mm), rotation angles about x, y, z (degrees, applied in that
    order: R = Rz Ry Rx; the shape's local z is the axis of a cylinder, cone or torus) and the full sizes of SIZES[kind]
    (a box's length / width / height along its local x / y / z; heights are full heights)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    s = [float(x) for x in sizes]
    if len(s) != len(SIZES[kind]) or min(s) <= 0:
        raise ValueError(f"{kind} needs {len(SIZES[kind])} positive sizes ({', '.join(SIZES[kind])}), got {sizes}")
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [float(a) for a in angles_deg], degrees=True).as_matrix()
    dims = {"box": lambda: dict(x_half=s[0] / 2, y_half=s[1] / 2, z_half=s[2] / 2),
            "cylinder": lambda: dict(radius=s[0], half_length=s[1] / 2),
            "cone": lambda: dict(radius1=s[0], radius2=s[1], half_length=s[2] / 2),
            "sphere": lambda: dict(radius=s[0]),
            "torus": lambda: dict(major_radius=s[0], minor_radius=s[1])}[kind]()
    return UserShape("add", kind, tuple(float(x) for x in center), matrix_quaternion(R), dims,
                     label or f"starting part: {kind} at an exact place")


def shape_sizes(shape: UserShape) -> List[float]:
    """The full sizes of SIZES[kind] of a shape (the inverse of exact_shape's `sizes`), for filling the exact form."""
    d = shape.dims_mm
    return {"box": lambda: [2 * d["x_half"], 2 * d["y_half"], 2 * d["z_half"]],
            "cylinder": lambda: [d["radius"], 2 * d["half_length"]],
            "cone": lambda: [d["radius1"], d["radius2"], 2 * d["half_length"]],
            "sphere": lambda: [d["radius"]],
            "torus": lambda: [d["major_radius"], d["minor_radius"]]}[shape.kind]()


def held_share(shape: UserShape, vertices) -> float:
    """The share of the part's vertices inside the shape (the starting part should hold the part, or its sub-part)."""
    from reco_engine.result_to_program import primitive_to_sp
    V = np.asarray(vertices, float)
    return float(np.mean(np.asarray(primitive_to_sp(shape.primitive()).sdf(V)) <= 1e-6 * max(1.0, float(np.ptp(V)))))
