#!/usr/bin/env python3
"""
S6 on the frame: the cascade add-cut loop (reco_engine.residual_loop) judged EXACTLY, not on its voxel grid.

For each input (STEP, STL) and member length ratio, the loop runs as usual and its FreeCAD grade now includes the
exact judge (reco_field.exact_check): the built model against the INPUT ITSELF by OCCT booleans - missed material
r+ = input - model, invented material r- = model - input, exact IoU = (V_in - r+) / (V_in + r-), the pieces of r+ -
and the genus of the result against the input's (17). Volumes are reported in mm^3 (the STL is 1/100 of the STEP).

    CUDA_VISIBLE_DEVICES="" python3 scripts/s6_frame_eval.py --ratios 2.0 1.5 --json docs/data/s6_frame_plates.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine.residual_loop import ResidualConfig, reconstruct  # noqa: E402

D = Path("~/shape_reco_parts/benchmark_parts/1.0")
INPUTS = {"STEP": (D / "Frame_4_x_6.stp", 1.0), "STL": (D / "Unbenannt-Frame_4_x_6.stl", 100.0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="*", default=list(INPUTS))
    ap.add_argument("--ratios", type=float, nargs="*", default=[2.0, 1.5])
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for name in args.inputs:
        path, scale = INPUTS[name]
        for ratio in args.ratios:
            with tempfile.TemporaryDirectory() as tmp:
                t = time.time()
                res = reconstruct(path, ResidualConfig(member_min_length_ratio=ratio, out_dir=Path(tmp) / "fc", freecad=True))
                loop_s = time.time() - t
            accepted = [o for o in res.operations if o.accepted]
            u3 = scale ** 3                                     # input units^3 -> mm^3 (the STL is 1/100 of the STEP)
            row = dict(input=name, ratio=ratio, operations=len(accepted),
                       adds=sum(o.op == "add" for o in accepted), cuts=sum(o.op == "cut" for o in accepted),
                       residual_voxels=[res.residual_voxels_pos, res.residual_voxels_neg],
                       chamfer_mm=None if res.chamfer_mm is None else res.chamfer_mm * scale,
                       exact_iou=res.exact_iou, r_plus_mm3=None if res.exact_r_plus is None else res.exact_r_plus * u3,
                       r_minus_mm3=None if res.exact_r_minus is None else res.exact_r_minus * u3,
                       r_plus_solids=res.exact_r_plus_solids, genus=res.genus, input_genus=res.input_genus,
                       grid_pitch_over_tau=res.grid_pitch_over_tau, loop_seconds=round(loop_s, 1))
            rows.append(row)
            print(f"{name} ratio {ratio}: {row['operations']} ops ({row['adds']} add, {row['cuts']} cut), exact IoU "
                  f"{row['exact_iou']}, r+ {row['r_plus_mm3']} mm3 in {row['r_plus_solids']} pieces, r- {row['r_minus_mm3']}, "
                  f"genus {row['genus']} (input {row['input_genus']}), grid residual {row['residual_voxels']}, "
                  f"pitch/tau {row['grid_pitch_over_tau']}, chamfer {row['chamfer_mm']} mm, {loop_s:.0f} s", flush=True)
    if args.json:
        args.json.write_text(json.dumps(dict(rows=rows), indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
