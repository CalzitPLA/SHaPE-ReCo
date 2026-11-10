#!/usr/bin/env python3
"""
Logic behind the "Cascade Reconstruct" node of the FreeCAD Nodes workbench (freecad_nodes/shape_reco/shape_reco_cascade.py), kept
here so it can be tested without FreeCAD or Qt, like reco_field/node_support.py:

  cascade_program    an STL/STEP file -> the residual add-cut loop (reco_engine/residual_loop.py) run to the end ->
                     (program JSON = ReconstructionResult.to_json, residual voxels missing, residual voxels in excess, summary)

The program travels as JSON text in the reco_engine/result_schema.py format (ReconstructionResult.from_dict reads it back).
"""

from __future__ import annotations

import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Callable, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))


def _describe(op) -> str:
    p = op.primitive
    dims = ", ".join(f"{k}={v:.3g}" for k, v in (p.dims_mm or {}).items())
    centre = ", ".join(f"{v:.2f}" for v in p.center_mm)
    modes = "/".join(dict.fromkeys(s.mode for s in p.sources)) or "?"
    extra = "" if op.freecad_valid is None else (" FreeCAD valid" if op.freecad_valid else " FreeCAD INVALID")
    return f"{op.op}:{p.kind} ({dims}) at ({centre}) conf {p.confidence:.2f} [{modes}] gain {op.net_gain:.0f}{extra}"


def summarize(result) -> str:
    """One line for the whole run, then one per accepted operation."""
    accepted = [o for o in result.operations if o.accepted]
    head = (f"{len(accepted)} operation(s) accepted, {len(result.operations) - len(accepted)} rejected; residual "
            f"+{result.residual_voxels_pos} / -{result.residual_voxels_neg} voxels")
    if result.chamfer_mm is not None and not (isinstance(result.chamfer_mm, float) and math.isnan(result.chamfer_mm)):
        head += f", chamfer {result.chamfer_mm:.4g} mm"
    if result.step_path:
        head += f", STEP {Path(result.step_path).name}"
    return "\n".join([head] + [f"  {i}: {_describe(o)}" for i, o in enumerate(accepted)])


def residual_config(top_k: int = 2, max_ops: Optional[int] = None, registry_path=None, op_guard: bool = False,
                    freecad_per_step: bool = False, resolution: Optional[int] = None, type_source: str = "registry",
                    **extra):
    """The reco_engine.residual_loop.ResidualConfig for the GUI's / the node's settings: the registry file is loaded with
    reco_engine.stage_registry.load_registry (a missing file: the loop's default registry), "build every step in FreeCAD"
    is grade_each_step, None means the loop's own default. `type_source` ("registry" | "descriptors" | "cnn") restricts
    the registry's type entries to one source (reco_engine.cascade.filter_type_entries) - the GUI's demonstration switch."""
    from reco_engine.residual_loop import ResidualConfig
    kw = dict(top_k=int(top_k), op_guard=bool(op_guard), max_ops=max_ops, grade_each_step=bool(freecad_per_step),
              resolution=resolution, **extra)
    if registry_path and Path(registry_path).is_file():
        from reco_engine.stage_registry import load_registry
        kw["registry"] = load_registry(Path(registry_path))
    if type_source and type_source != "registry":
        from reco_engine.cascade import filter_type_entries, load_default_registry
        kw["registry"] = filter_type_entries(kw.get("registry") or load_default_registry(), type_source)
    known = {f.name for f in fields(ResidualConfig)}
    return ResidualConfig(**{k: v for k, v in kw.items() if k in known and v is not None})


def run_cascade(input_path, top_k: int = 2, max_ops: Optional[int] = None, registry_path=None, op_guard: bool = False,
                freecad_per_step: bool = False, steps_fn: Optional[Callable] = None):
    """The final ReconstructionResult of the residual loop (`steps_fn(path, config)` replaces residual_steps in tests)."""
    kw = dict(top_k=int(top_k), op_guard=bool(op_guard), max_ops=max_ops, freecad_per_step=bool(freecad_per_step),
              registry_path=str(registry_path) if registry_path else None)
    if steps_fn is None:
        from reco_engine.residual_loop import residual_steps as steps_fn
        config = residual_config(**kw)
    else:
        config = kw
    result = None
    for step in steps_fn(str(input_path), config):
        if step.result is not None:
            result = step.result
        if step.kind == "done":
            break
    if result is None:
        raise RuntimeError(f"the residual loop produced no result for {input_path}")
    return result


def cascade_program(input_path, top_k: int = 2, max_ops: Optional[int] = None, registry_path=None, op_guard: bool = False,
                    steps_fn: Optional[Callable] = None) -> Tuple[str, int, int, str]:
    """(program JSON, residual voxels missing, residual voxels in excess, summary)."""
    result = run_cascade(input_path, top_k=top_k, max_ops=max_ops, registry_path=registry_path, op_guard=op_guard,
                         steps_fn=steps_fn)
    return result.to_json(), int(result.residual_voxels_pos), int(result.residual_voxels_neg), summarize(result)
