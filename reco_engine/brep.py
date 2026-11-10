#!/usr/bin/env python3
"""
The exact surface inventory of a STEP part: its B-rep faces with type, parameters, area, outward normal and whether the
whole part lies on their material side (scripts/freecad_brep_faces.py, run in freecadcmd), plus the part's surface as
triangles that keep their face id.

    inv = step_faces("part.step")
    inv["faces"][i] -> {"type": "Plane" | "Cylinder" | "Cone" | "Sphere" | "Toroid" | ..., "area", "normal", "point",
                        "axis", "center", "radius", ..., "convex", "dominating"}
    inv["vertices"], inv["triangles"], inv["face_id"]

Cached in ~/.cache/shape_reco/brep/ ($SHAPE_RECO_BREP_CACHE, "off" disables), keyed by the file's bytes, the deflection and
the job script's source: a STEP is read by FreeCAD once. A cache entry is valid when faces.json and mesh.npz both exist;
they are not written atomically.

Unlike reco_field.exact_check (which returns {"error": ...}), `step_faces` raises: RuntimeError when the job wrote no
faces.json, subprocess.TimeoutExpired after `timeout` seconds. It resolves freecadcmd through
src.closest_point_field.find_freecadcmd (dataset_gen on the path), the judge through reco_field.export.find_freecadcmd.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JOB_SCRIPT = ROOT / "scripts" / "freecad_brep_faces.py"


def is_step(path) -> bool:
    return str(path).lower().endswith((".step", ".stp"))


def _cache_dir() -> Optional[Path]:
    env = os.environ.get("SHAPE_RECO_BREP_CACHE", "").strip()
    if env.lower() in ("0", "off", "no", "false"):
        return None
    return Path(env) if env else Path.home() / ".cache" / "shape_reco" / "brep"


def step_faces(path, deflection: float = 0.02, tol: float = 1e-3, freecad: Optional[str] = None,
               timeout: int = 600) -> Dict:
    """The inventory of a STEP file (see the module docstring)."""
    path = Path(path)
    key = hashlib.sha1(path.read_bytes() + repr((deflection, tol)).encode() + JOB_SCRIPT.read_bytes()).hexdigest()
    folder = _cache_dir()
    out = (folder / key) if folder is not None else Path(tempfile.mkdtemp(prefix="brep_"))
    js, npz = out / "faces.json", out / "mesh.npz"
    if not (js.is_file() and npz.is_file()):
        out.mkdir(parents=True, exist_ok=True)
        job = out / "job.json"
        job.write_text(json.dumps(dict(step=str(path.resolve()), out_json=str(js), out_npz=str(npz),
                                       deflection=deflection, tol=tol)))
        from src.closest_point_field import find_freecadcmd
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["BREP_JOB"] = str(job)
        run = subprocess.run([freecad or find_freecadcmd(), str(JOB_SCRIPT)], capture_output=True, text=True,
                             timeout=timeout, env=env)
        if not js.is_file():
            raise RuntimeError(f"freecad_brep_faces failed on {path}: {run.stdout[-800:]} {run.stderr[-800:]}")
    inv = json.loads(js.read_text())
    mesh = np.load(npz)
    inv.update(vertices=mesh["vertices"], triangles=mesh["triangles"], face_id=mesh["face_id"])
    return inv
