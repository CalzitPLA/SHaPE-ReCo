#!/usr/bin/env python3
"""
Logic behind the vector-field nodes of the FreeCAD Nodes workbench (freecad_nodes/shape_reco/shape_reco_field_*.py), kept here so it can be
tested without FreeCAD, like reco_engine/node_support.py:

  peel_program       "Peel (Vector Field)": an STL/STEP file -> the CSG program (JSON), the share of the surface explained, a readable summary
  export_program     "Export Program": the program (JSON) -> a FreeCAD solid (STEP + STL) in a work directory, with its volume
  field_grade        "Field Grade": two-way chamfer distance of two meshes (a reconstruction and its input), in mm

The program travels between the nodes as JSON text (`export.program_to_dict`): plain data, so a saved graph and the socket viewer can show it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.export import build_program, mesh_chamfer, program_to_dict  # noqa: E402
from reco_field.peel import peel  # noqa: E402
from src.closest_point_field import load_field  # noqa: E402


def _describe(op) -> str:
    fit = op.fit
    seg = ""
    if fit.u is not None:
        seg += f" u={fit.u[1] * 180 / 3.141592653589793:.0f}deg"
    if fit.v is not None:
        seg += f" v=[{fit.v[0] * 180 / 3.141592653589793:.0f},{fit.v[1] * 180 / 3.141592653589793:.0f}]deg"
    name = fit.kind + seg
    return name if op.op == "base" else f"{op.op}:{name}"


def peel_program(stl_path, max_ops: int = 8, target: float = 0.985) -> Tuple[str, float, str]:
    """(program JSON, explained, summary) of the peeling loop on an STL/STEP file."""
    field = load_field(stl_path)
    res = peel(field, max_ops=max_ops, target=target)
    program = json.dumps(program_to_dict(res.model))
    lines = [f"explained {100 * res.explained:.1f} % of the surface with {len(res.ops)} operation(s), tolerance {res.tol:.3f} mm"]
    for i, op in enumerate(res.ops):
        params = ", ".join(f"{k}={v:g}" for k, v in op.fit.params.items() if k in ("radius", "radius1", "radius2", "major", "minor", "hx", "hy", "hz",
                                                                                     "half_length", "half_height", "u_sweep"))
        lines.append(f"  {i}: {_describe(op)}  ({params})  -> {100 * op.explained_after:.1f} %")
    return program, float(res.explained), "\n".join(lines)


def export_program(program_json: str, workdir, name: str = "reconstruction") -> Tuple[str, str, str]:
    """(STEP path, STL path, summary): the program built with FreeCAD's Part module in `workdir`."""
    out = build_program(program_json, workdir, name or "reconstruction")
    summary = (f"FreeCAD solid: volume {out.volume:.1f} mm^3, {'valid' if out.valid else 'INVALID'}, {out.n_faces} faces, "
               f"{out.n_solids} solid(s) built; STEP {out.step.name}")
    return str(out.step), str(out.stl), summary


def field_grade(result_stl, input_stl, n: int = 3000) -> Tuple[float, str]:
    """(mean two-way chamfer in mm, summary) of a reconstruction's mesh against the input mesh."""
    c: Dict[str, float] = mesh_chamfer(load_field(result_stl), load_field(input_stl), n)
    return c["mean"], (f"chamfer mean {c['mean']:.4f} mm, max {c['max']:.3f} mm "
                       f"(input -> result {c['b_to_a_mean']:.4f}, result -> input {c['a_to_b_mean']:.4f})")
