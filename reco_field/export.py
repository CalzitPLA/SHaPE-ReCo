#!/usr/bin/env python3
"""
Export a reconstructed program (the SDF primitive tree of `peel(...).model`) as a FreeCAD solid: STEP (exact B-rep) and STL.

    res = peel(load_field("part.stl"))
    out = build_in_freecad(res.model, "out_dir", "part")            # runs freecadcmd on reco_field/freecad_builder.py
    out.step, out.stl, out.volume, out.valid
    export_check(out, load_field("part.stl"))                        # two-way chamfer of the FreeCAD solid to the input, in mm

The tree is serialised to JSON (`program_to_dict`) and rebuilt with the Part module's own shapes and booleans, applying the Part conventions
(docs/freecad_part_shapes.tex). Not exportable, with a clear error: a sphere with a cone-limited latitude (Part builds only plane-limited
latitudes), the convex polyhedron and the ellipsoid.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField, find_freecadcmd, load_field  # noqa: E402

BUILDER = Path(__file__).resolve().parent / "freecad_builder.py"


def _pair(rng) -> list:
    return [float(rng[0]), float(rng[1])]


def program_to_dict(prim: sp.Primitive, big: Optional[float] = None) -> dict:
    """The primitive tree as plain data (lists and numbers). `big`: the size of the box standing for a half space (default: 4 x the
    extent of the tree's bounds, which are finite unless the tree consists of half spaces only)."""
    if big is None:
        lo, hi = prim.bounds()
        span = float(np.max(np.minimum(hi - lo, 1e6)))
        big = 4.0 * max(span, 1.0)
    d = program_to_dict
    if isinstance(prim, sp.Placed):
        return dict(type="Placed", primitive=d(prim.primitive, big), rotation=np.asarray(prim.rotation, float).tolist(),
                    translation=np.asarray(prim.translation, float).tolist())
    if isinstance(prim, (sp.Union, sp.Intersection, sp.Subtraction)):
        return dict(type=type(prim).__name__, a=d(prim.a, big), b=d(prim.b, big))
    if isinstance(prim, sp.Box):
        return dict(type="Box", half=[float(x) for x in prim.half])
    if isinstance(prim, sp.Sphere):
        return dict(type="Sphere", radius=float(prim.radius), u=_pair(prim.u), v=_pair(prim.v), inner_radius=float(prim.inner_radius),
                    latitude=prim.latitude)
    if isinstance(prim, sp.Cylinder):
        return dict(type="Cylinder", radius=float(prim.radius), height=float(prim.height), u=_pair(prim.u), inner_radius=float(prim.inner_radius))
    if isinstance(prim, sp.Cone):
        return dict(type="Cone", radius1=float(prim.radius1), radius2=float(prim.radius2), height=float(prim.height), u=_pair(prim.u))
    if isinstance(prim, sp.Torus):
        if prim.inner_minor > 0:
            raise NotImplementedError("a hollow torus is not exported")
        return dict(type="Torus", major=float(prim.major), minor=float(prim.minor), u=_pair(prim.u), v=_pair(prim.v))
    if isinstance(prim, sp.HalfSpace):
        return dict(type="HalfSpace", normal=[float(x) for x in prim.normal], offset=float(prim.offset), big=float(big))
    raise NotImplementedError(f"{type(prim).__name__} cannot be exported to FreeCAD")


def program_from_dict(d: dict) -> sp.Primitive:
    """The inverse of `program_to_dict`: the JSON that FreeCAD builds, back as the primitive tree (the half space's
    `big` is FreeCAD's stand-in size and not part of the solid)."""
    f = program_from_dict
    kind = d["type"]
    if kind == "Placed":
        return sp.Placed(f(d["primitive"]), rotation=np.asarray(d["rotation"], float),
                         translation=np.asarray(d["translation"], float))
    if kind in ("Union", "Intersection", "Subtraction"):
        return getattr(sp, kind)(f(d["a"]), f(d["b"]))
    fields = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k not in ("type", "big")}
    if kind in ("Box", "Sphere", "Cylinder", "Cone", "Torus", "HalfSpace"):
        return getattr(sp, kind)(**fields)
    raise NotImplementedError(f"{kind} is not a program node")


@dataclass
class Exported:
    step: Path
    stl: Path
    volume: float
    valid: bool
    n_solids: int                # solids the booleans produced (the largest is exported)
    n_faces: int
    bbox: list
    program: Path                # the JSON that was built
    solids: Optional[list] = None  # build_split_in_freecad only: [{step, stl, volume, valid, n_faces, bbox}, biggest first]


def _run_builder(freecad: str, env: dict, timeout: int):
    """freecadcmd BUILDER under `env`; through the persistent worker when SHAPE_RECO_FREECAD_WORKER=1
    (reco_field/freecad_worker.py), else the plain subprocess call. Returns an object with stdout / stderr."""
    from reco_field import freecad_worker
    if freecad_worker.enabled():
        import types
        rc, out = freecad_worker.run_script(freecad, BUILDER, env, timeout=timeout)
        return types.SimpleNamespace(returncode=rc, stdout=out, stderr="")
    return subprocess.run([freecad, str(BUILDER)], capture_output=True, text=True, timeout=timeout, env=env)


def build_in_freecad(model: sp.Primitive, out_dir, name: str = "reconstruction", deflection: float = 0.02,
                     freecad: Optional[str] = None, timeout: int = 300) -> Exported:
    """Build `model` with FreeCAD's Part module and write `<name>.step`, `<name>.stl` and `<name>.json` (the program) into `out_dir`."""
    return build_program(program_to_dict(model), out_dir, name, deflection, freecad, timeout)


def build_program(program: dict, out_dir, name: str = "reconstruction", deflection: float = 0.02,
                  freecad: Optional[str] = None, timeout: int = 300) -> Exported:
    """The same from the serialised program (`program_to_dict`, or its JSON text): what the FreeCAD node passes between nodes."""
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        raise RuntimeError("export needs FreeCAD: freecadcmd was not found")
    if isinstance(program, str):
        program = json.loads(program)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(program, indent=1))
    step, stl, result = out_dir / f"{name}.step", out_dir / f"{name}.stl", out_dir / f"{name}.result.json"
    for f in (step, stl, result):
        f.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PROGRAM_JSON=str(path), STEP_OUT=str(step), STL_OUT=str(stl), RESULT_JSON=str(result), DEFLECTION=str(deflection))
    proc = _run_builder(freecad, env, timeout)
    if not result.is_file():
        raise RuntimeError(f"FreeCAD could not build the program: {(proc.stderr or proc.stdout)[-600:]}")
    r = json.loads(result.read_text())
    result.unlink()
    return Exported(step, stl, r["volume"], r["valid"], r["n_solids"], r["n_faces"], r["bbox"], path, r.get("solids"))


def build_groups(operations, out_dir, name: str = "reconstruction", tools: str = "solid",
                 freecad: Optional[str] = None, timeout: int = 300) -> dict:
    """The program in FreeCAD WITHOUT its final booleans: every operation's tool as one object of `<name>_groups.FCStd`,
    in the group of its operation ("adds", "cuts"), in the program's order. `operations`: [(op, label, primitive)],
    primitive an SDF tree (reco_field.sdf_primitives). `tools`: "solid" - a composite tool is built with its own
    booleans; "leaves" - no boolean at all, a composite is the group of its primitives' shapes; "primitives" - as
    "leaves", every primitive a parametric Part object (Part::Box, Part::Cylinder, ...) with its placement. Returns dict(fcstd,
    operations=[dict(index, op, label, parts, volume, valid, seconds | error)], groups, seconds)."""
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        raise RuntimeError("export needs FreeCAD: freecadcmd was not found")
    if tools not in ("solid", "leaves", "primitives"):
        raise ValueError(f"tools: 'solid', 'leaves' or 'primitives', not {tools!r}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fcstd, job, result = (out_dir / f"{name}_groups.FCStd", out_dir / f"{name}_groups.json",
                          out_dir / f"{name}_groups.result.json")
    for f in (fcstd, result):
        f.unlink(missing_ok=True)
    job.write_text(json.dumps(dict(fcstd=str(fcstd), tools=tools,
                                   operations=[dict(op=op, label=label, program=program_to_dict(prim))
                                               for op, label, prim in operations]), indent=1))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(GROUPS_JSON=str(job), RESULT_JSON=str(result))
    proc = _run_builder(freecad, env, timeout)
    if not result.is_file():
        raise RuntimeError(f"FreeCAD could not build the groups: {(proc.stderr or proc.stdout)[-600:]}")
    r = json.loads(result.read_text())
    result.unlink()
    return r


def build_split_in_freecad(model: sp.Primitive, out_dir, name: str = "reconstruction", deflection: float = 0.02,
                           freecad: Optional[str] = None, timeout: int = 300) -> Exported:
    """Like `build_in_freecad`, but when the booleans produce more than one solid, EVERY one is built, checked and
    exported (`<out_dir>/<name>_solids/solid_<i>.step/.stl`, biggest first, `i` starting at 1 - `Exported.solids`),
    not only the biggest. `Exported.solids[0]` is always the same solid `Exported.step`/`.stl` already are; the rest
    ("the leftovers") are real, checked B-rep solids of their own - what a boolean op split off, not points."""
    out_dir = Path(out_dir)
    split_dir = out_dir / f"{name}_solids"
    split_dir.mkdir(parents=True, exist_ok=True)
    freecad = freecad or find_freecadcmd()
    if freecad is None:
        raise RuntimeError("export needs FreeCAD: freecadcmd was not found")
    program = program_to_dict(model)
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(program, indent=1))
    step, stl, result = out_dir / f"{name}.step", out_dir / f"{name}.stl", out_dir / f"{name}.result.json"
    for f in (step, stl, result):
        f.unlink(missing_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env.update(PROGRAM_JSON=str(path), STEP_OUT=str(step), STL_OUT=str(stl), RESULT_JSON=str(result),
              DEFLECTION=str(deflection), SPLIT_OUT_DIR=str(split_dir))
    proc = _run_builder(freecad, env, timeout)
    if not result.is_file():
        raise RuntimeError(f"FreeCAD could not build the program: {(proc.stderr or proc.stdout)[-600:]}")
    r = json.loads(result.read_text())
    result.unlink()
    solids = [dict(step=Path(d["step"]), stl=Path(d["stl"]), volume=d["volume"], valid=d["valid"],
                   n_faces=d["n_faces"], bbox=d["bbox"]) for d in r.get("solids", [])] or None
    return Exported(step, stl, r["volume"], r["valid"], r["n_solids"], r["n_faces"], r["bbox"], path, solids)


def _surface_points(field: SurfaceField, k: int, rng: np.random.Generator) -> np.ndarray:
    area = 0.5 * np.linalg.norm(np.cross(field._b - field._a, field._c - field._a), axis=1)
    idx = rng.choice(len(field.faces), size=k, p=area / area.sum())
    r1, r2 = rng.random((k, 1)), rng.random((k, 1))
    flip = (r1 + r2) > 1
    r1, r2 = np.where(flip, 1 - r1, r1), np.where(flip, 1 - r2, r2)
    return field._a[idx] + (field._b[idx] - field._a[idx]) * r1 + (field._c[idx] - field._a[idx]) * r2


def mesh_chamfer(a: SurfaceField, b: SurfaceField, n: int = 20000, seed: int = 0) -> Dict[str, float]:
    """Two-way distance between the surfaces of two meshes (exact queries), in their units: mean and maximum each way."""
    rng = np.random.default_rng(seed)
    ab = a.query(_surface_points(b, n, rng)).distance          # b's surface -> a
    ba = b.query(_surface_points(a, n, rng)).distance          # a's surface -> b
    return dict(mean=float(0.5 * (ab.mean() + ba.mean())), max=float(max(ab.max(), ba.max())),
                a_to_b_mean=float(ba.mean()), b_to_a_mean=float(ab.mean()))


def export_check(exported: Exported, reference: SurfaceField, n: int = 20000) -> Dict[str, float]:
    """Two-way chamfer of the FreeCAD solid (its STL) to `reference` (the input mesh, or the mesh of the SDF model)."""
    out = mesh_chamfer(load_field(exported.stl), reference, n)
    out["volume"] = exported.volume
    return out


if __name__ == "__main__":
    import argparse
    from reco_field.peel import peel
    ap = argparse.ArgumentParser(description="Reconstruct an STL/STEP with the peeling loop and build the program in FreeCAD.")
    ap.add_argument("part")
    ap.add_argument("--out", default="reco_export")
    args = ap.parse_args()
    field = load_field(args.part)
    res = peel(field)
    print("program:", [(o.op, o.fit.kind) for o in res.ops], f"explained {res.explained:.3f}")
    ex = build_in_freecad(res.model, args.out, Path(args.part).stem)
    print(f"FreeCAD: volume {ex.volume:.1f} mm^3, valid {ex.valid}, {ex.n_faces} faces -> {ex.step}")
    print("chamfer to the input:", export_check(ex, field))
