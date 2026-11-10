#!/usr/bin/env python3
"""
Shapes whose CSG tree is known to the last digit - the ground truth of the validation cycle (reco_engine/validation.py).
The reconstruction only ever sees the shape's mesh or STEP; the rating compares what it built with the tree here.

    t = truth_from_aegir(".../geometry/sandbox/stack/n6_I/n6_I_000445_T2d4c.json")   # aegir's synthetic stacked set
    t = truth_from_v5(record, shard_dir)                                              # a v5 dataset sample
    g = truth_gate(t, mesh_path)        # is the tree really the solid the run reads? (shape and volume agree)

Sources (1 October 2026):
  aegir stack   ~/Documents/Sandbox_aegir/aegir/geometry/sandbox/stack/<class>/*.json, written in
                reco_field/export.program_to_dict's format (read back by program_from_dict here); 1-32 leaves, 40-100 mm.
  v5 dataset    dataset_metadata.jsonl records of the test split: a single placed primitive (full / sector / band
                segments; spheres limited by planes, as src/solid_builder.local_primitive builds them) and the carved ones
                (base minus solid_builder.carve_cutter). Records with planar cuts (the cut offsets are not stored), hollow
                and frame samples (no converter yet) and filleted I-beams are NOT exact and are refused by v5_exact.
A truth is used only after truth_gate: the tree and the mesh the run reads must be the same solid (point-in-solid
agreement outside a thin band around the surface, and the volume), so a convention mismatch shows up as an excluded
truth, never as a bad reconstruction.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "dataset_gen") not in sys.path:
    sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402

from reco_engine.csg_surface import CompiledTree, compile_tree, finite_bounds  # noqa: E402

V5_GRID_MM = 40.0
V5_RESOLUTION = 64


@dataclass
class Truth:
    name: str
    source: str                       # 'aegir' | 'v5' | 'program'
    tree: sp.Primitive
    compiled: CompiledTree
    lo: np.ndarray
    hi: np.ndarray
    input_path: Optional[str] = None  # the mesh / STEP the reconstruction reads, when it exists already
    meta: Dict = field(default_factory=dict)

    @property
    def diag(self) -> float:
        return float(np.linalg.norm(self.hi - self.lo))


def truth_from_tree(tree: sp.Primitive, name: str, source: str = "program", input_path=None, meta=None) -> Truth:
    compiled = compile_tree(tree)
    lo, hi = finite_bounds(compiled.leaves, roles=("add", "clip"))
    return Truth(name, source, tree, compiled, lo, hi, None if input_path is None else str(input_path), dict(meta or {}))


def program_from_dict(d: dict) -> sp.Primitive:
    """The inverse of reco_field.export.program_to_dict (aegir's stacked set is written in that format)."""
    t = d["type"]
    if t == "Placed":
        return sp.Placed(program_from_dict(d["primitive"]), np.asarray(d["rotation"], float), np.asarray(d["translation"], float))
    if t in ("Union", "Intersection", "Subtraction"):
        return getattr(sp, t)(program_from_dict(d["a"]), program_from_dict(d["b"]))
    if t == "Box":
        return sp.Box([float(x) for x in d["half"]])
    if t == "Sphere":
        return sp.Sphere(float(d["radius"]), u=tuple(d.get("u", sp.FULL)), v=tuple(d.get("v", (-np.pi / 2, np.pi / 2))),
                         inner_radius=float(d.get("inner_radius", 0.0)), latitude=d.get("latitude", "cone"))
    if t == "Cylinder":
        return sp.Cylinder(float(d["radius"]), float(d["height"]), u=tuple(d.get("u", sp.FULL)),
                           inner_radius=float(d.get("inner_radius", 0.0)))
    if t == "Cone":
        return sp.Cone(float(d["radius1"]), float(d["radius2"]), float(d["height"]), u=tuple(d.get("u", sp.FULL)))
    if t == "Torus":
        return sp.Torus(float(d["major"]), float(d["minor"]), u=tuple(d.get("u", sp.FULL)), v=tuple(d.get("v", sp.FULL)))
    if t == "HalfSpace":
        return sp.HalfSpace(d["normal"], float(d["offset"]))
    raise ValueError(f"unknown node type {t!r}")


def program_json(truth: Truth) -> dict:
    """The truth tree as JSON (reco_field.export.program_to_dict), stored next to the shape's mesh for the runner."""
    from reco_field.export import program_to_dict
    return dict(name=truth.name, source=truth.source, meta=truth.meta, program=program_to_dict(truth.tree))


def truth_from_json(path) -> Truth:
    """A truth written by program_json (or a bare program_to_dict tree, as aegir writes them)."""
    d = json.loads(Path(path).read_text())
    if "program" in d and "type" not in d:
        return truth_from_tree(program_from_dict(d["program"]), d.get("name", Path(path).stem), d.get("source", "program"),
                               meta=d.get("meta"))
    return truth_from_tree(program_from_dict(d), Path(path).stem, "aegir", meta=dict(json=str(path)))


def truth_from_aegir(json_path) -> Truth:
    return truth_from_json(json_path)


# ---------------------------------------------------------------------------------------------------- v5 records

def _quat_matrix(q) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat([q["qx"], q["qy"], q["qz"], q["qw"]]).as_matrix()


def v5_exact(record: dict) -> Tuple[bool, str]:
    """Whether the record's solid is exactly rebuildable (see the module docstring)."""
    if record.get("record_type") == "dataset_header":
        return False, "header"
    pl = record.get("placement") or {}
    if record.get("topology") not in ("box", "sphere", "cylinder", "cone", "torus"):
        return False, f"topology {record.get('topology')}"
    if pl.get("cuts"):
        return False, "planar cuts (offsets not stored)"
    if record.get("wall_mm") or pl.get("hollow"):
        return False, "hollow (no converter yet)"
    if record.get("members"):
        return False, "frame (no converter yet)"
    if record.get("boxes"):
        return False, "I-beam (filleted)"
    if record.get("subclass") not in (None, "full", "sector", "band"):
        return False, f"subclass {record.get('subclass')}"
    if pl.get("degenerate"):
        return False, "degenerate"
    if record.get("sample_kind") == "negres":
        return False, "negative residual (the carve cutter's voxels, not a body)"
    return True, "carved" if pl.get("carve") else "plain"


def v5_leaf(record: dict) -> sp.Primitive:
    """The placed primitive of a record: its dims_mm (already scaled), its world centre (the axis midpoint of a cylinder
    or cone), its rotation quaternion and segment - exactly src/solid_builder.labels in reverse."""
    kind, d = record["topology"], record["dims_mm"]
    seg = record.get("segment") or {}
    u = tuple(seg.get("u", sp.FULL))
    if kind == "box":
        prim = sp.Box([d["x_half"], d["y_half"], d["z_half"]])
    elif kind == "sphere":
        prim = sp.Sphere(d["radius"], u=u, v=tuple(seg.get("v", (-np.pi / 2, np.pi / 2))), latitude="plane")
    elif kind == "cylinder":
        prim = sp.Cylinder(d["radius"], 2.0 * d["half_length"], u=u)
    elif kind == "cone":
        prim = sp.Cone(d["radius1"], d["radius2"], 2.0 * d["half_length"], u=u)
    elif kind == "torus":
        prim = sp.Torus(d["major_radius"], d["minor_radius"], u=u, v=tuple(seg.get("v", sp.FULL)))
    else:
        raise ValueError(kind)
    R = _quat_matrix(record["rotation"])
    local_centre = np.array([0.0, 0.0, prim.height / 2.0]) if isinstance(prim, (sp.Cylinder, sp.Cone)) else np.zeros(3)
    t = np.asarray(record["center_mm"], float) - R @ local_centre
    return sp.Placed(prim, R, t)


def truth_from_v5(record: dict, shard_dir) -> Truth:
    ok, why = v5_exact(record)
    if not ok:
        raise ValueError(f"v5 record not exact: {why}")
    from src import solid_builder
    tree = v5_leaf(record)
    carve = (record.get("placement") or {}).get("carve")
    if carve:
        grid = float((record.get("placement") or {}).get("grid_mm", V5_GRID_MM))
        tree = sp.Subtraction(tree, solid_builder.carve_cutter(carve, grid, V5_RESOLUTION))
    stl = record.get("stl_path", "")
    stem = Path(stl).with_suffix("")
    solid = Path(shard_dir) / f"{stem}.solid.stl"
    name = f"v5_{record.get('shard', 'x')}_{stem.name}"
    meta = dict(topology=record["topology"], subclass=record.get("subclass"), kind=why, index=record.get("index"),
                freecad_volume_mm3=record.get("freecad_volume_mm3") or ((record.get("placement") or {}).get("solid") or {}).get("volume_mm3"))
    return truth_from_tree(tree, name, "v5", solid if solid.exists() else Path(shard_dir) / stl, meta)


def iter_v5_records(split_dir):
    """(record, shard_dir) for every sample record below a split directory (test/val/train), manifests per class."""
    for manifest in sorted(Path(split_dir).glob("**/dataset_metadata.jsonl")):
        shard = manifest.parent if manifest.parent.name.startswith("shard") else manifest.parent.parent
        for line in manifest.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("record_type") != "dataset_header":
                    yield rec, shard


# ---------------------------------------------------------------------------------------------------------- gate

def load_mesh(path) -> Tuple[np.ndarray, np.ndarray]:
    import trimesh
    m = trimesh.load(str(path), force="mesh", process=True)
    return np.ascontiguousarray(m.vertices, np.float64), np.ascontiguousarray(m.faces, np.int64)


def mesh_inside(V, F, P) -> np.ndarray:
    import igl
    return np.abs(igl.fast_winding_number(V, F, np.ascontiguousarray(P, np.float64))) > 0.5


def mesh_genus(V, F) -> Optional[int]:
    """Genus of a closed single-shell triangle mesh from its Euler characteristic (None if not closed or several shells)."""
    import trimesh
    m = trimesh.Trimesh(V, F, process=True)
    if not m.is_watertight or m.body_count != 1:
        return None
    return int(round((2 - m.euler_number) / 2))


def truth_gate(truth: Truth, mesh_path, n: int = 200_000, seed: int = 0, band_rel: float = 2e-3,
               max_mismatch: float = 1e-3, volume_ref: Optional[float] = None, volume_rel: float = 5e-3) -> dict:
    """Do the tree and the mesh the run reads describe the same solid? Uniform points in the padded box: inside the tree
    (exact sign) vs inside the mesh (winding number), compared outside a band of band_rel x diag around the tree's surface
    (the mesh is a tessellation); the tree's volume by the same points against `volume_ref` (e.g. FreeCAD's) and the
    mesh's own volume. ok only if both hold; reasons say what failed."""
    V, F = load_mesh(mesh_path)
    D = truth.diag
    pad = 0.02 * D
    lo, hi = np.minimum(truth.lo, V.min(0)) - pad, np.maximum(truth.hi, V.max(0)) + pad
    rng = np.random.default_rng(seed)
    P = lo + rng.random((n, 3)) * (hi - lo)
    s = truth.compiled.sdf(P)
    tin, min_ = s <= 0, mesh_inside(V, F, P)
    away = np.abs(s) > band_rel * D
    mismatch = float(np.mean(tin[away] != min_[away])) if away.any() else 1.0
    box = float(np.prod(hi - lo))
    frac = float(tin.mean())
    vol = frac * box
    se = box * np.sqrt(frac * (1 - frac) / n)
    import trimesh
    mesh = trimesh.Trimesh(V, F, process=True)
    vol_mesh = float(abs(mesh.volume))
    ref = volume_ref if volume_ref is not None else vol_mesh
    reasons = []
    if mismatch > max_mismatch:
        reasons.append(f"shape mismatch {mismatch:.2e} > {max_mismatch:.0e}")
    if abs(vol - ref) > 3 * se + volume_rel * ref:
        reasons.append(f"volume {vol:.6g} vs {ref:.6g} (3 SE {3 * se:.3g})")
    genus = mesh_genus(V, F)
    return dict(ok=not reasons, reasons=reasons, mismatch=mismatch, band=band_rel * D, volume_tree=vol, volume_se=se,
                volume_ref=ref, volume_mesh=vol_mesh, genus=genus, watertight=bool(mesh.is_watertight),
                bodies=int(mesh.body_count), n=n)
