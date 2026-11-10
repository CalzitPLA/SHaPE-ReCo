#!/usr/bin/env python3
"""
Run the peeling loop (reco_field.peel) on every example STL and write program, explained share and two-way chamfer per part to
docs/data/peel_examples.json (the table in docs/VECTORFIELD.md comes from it).

    python3 scripts/eval_peel_examples.py [part ...]        # default: every dataset_gen/examples/*.stl
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.fit import tolerance  # noqa: E402,F401
from reco_field.peel import chamfer, peel  # noqa: E402
from src.closest_point_field import load_field  # noqa: E402

EX = ROOT / "dataset_gen" / "examples"
OUT = ROOT / "docs" / "data" / "peel_examples.json"


def describe(op) -> list:
    """[op, kind] plus the segment parameters (degrees) of a body of revolution that is a sector."""
    out = [op.op, op.fit.kind]
    seg = {}
    if op.fit.u is not None:
        seg["u_sweep_deg"] = round(float(np.degrees(op.fit.u[1] - op.fit.u[0])), 1)
    if op.fit.v is not None:
        seg["v_deg"] = [round(float(np.degrees(x)), 1) for x in op.fit.v]
        if op.fit.kind == "sphere":
            seg["latitude"] = op.fit.latitude
    return out + ([seg] if seg else [])


def main(names):
    parts = []
    for path in sorted(EX.glob("*.stl")):
        if names and path.stem not in names:
            continue
        field = load_field(path)
        t = time.time()
        res = peel(field)
        seconds = time.time() - t
        c = chamfer(res.model, field)
        parts.append(dict(part=path.stem, seconds=round(seconds, 1), explained=round(res.explained, 4),
                          extent=round(float(np.linalg.norm(field.bounds[1] - field.bounds[0])), 2),
                          program=[describe(o) for o in res.ops],
                          input_to_model_mean=c["input_to_model"]["mean"], input_to_model_max=c["input_to_model"]["max"],
                          model_to_input_mean=c["model_to_input"]["mean"], model_to_input_max=c["model_to_input"]["max"]))
        print(f"{path.stem:22s} {seconds:5.1f}s  explained {res.explained:.3f}  chamfer {c['symmetric_mean']:.4f}  "
              f"max {c['hausdorff']:.3f}  {[p[:2] for p in parts[-1]['program']]}", flush=True)
    if not names:
        OUT.write_text(json.dumps(dict(tolerance_fraction=0.004, parts=parts), indent=1))


if __name__ == "__main__":
    main(set(sys.argv[1:]))
