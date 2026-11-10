#!/usr/bin/env python3
"""
R1, second test: what does the residual of a finished reconstruction look like on a fixed grid vs a thickness-adaptive
one, against the EXACT residual? r+ = input - model (missed material), r- = model - input (invented material).

Exact: scripts/freecad_residual.py (freecadcmd booleans of the input STEP and the program's solid). Grids: parity of
the input mesh and of the model's mesh on the same box grid, slab by slab; per grid the r+/r- voxel volumes and the
number of connected r+ components (26-connected).

    python3 scripts/local_grid_residual.py --program <reco_field.freecad_builder program.json>
    python3 scripts/local_grid_residual.py --program p.json --json docs/data/local_grid_residual.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.freecad_path import freecadcmd_or_default  # noqa: E402  ($SHAPE_RECO_FREECADCMD, PATH, /usr/bin)

from dataset_gen.src.closest_point_field import load_mesh  # noqa: E402
from dataset_gen.src.local_grids import (adaptive_pitch, box_grid, iter_slabs, slab_topology,  # noqa: E402
                                          wall_thickness)

D = Path("~/shape_reco_parts/benchmark_parts/1.0")


def exact_residual(step: Path, program: Path, out_dir: Path) -> dict:
    job = out_dir / "job.json"
    job.write_text(json.dumps(dict(input=str(step), program=str(program), out_dir=str(out_dir))))
    subprocess.run([freecadcmd_or_default(), str(ROOT / "scripts" / "freecad_residual.py")], check=True, capture_output=True,
                   env=dict(os.environ, RESIDUAL_JOB=str(job)))
    return json.loads((out_dir / "residual.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", type=Path, required=True)
    ap.add_argument("--stl", type=Path, default=D / "Unbenannt-Frame_4_x_6.stl")
    ap.add_argument("--step", type=Path, default=D / "Frame_4_x_6.stp")
    ap.add_argument("--scale", type=float, default=100.0, help="input STL units -> mm (the frame's STL is 1/100)")
    ap.add_argument("--fixed", type=int, nargs="*", default=[48, 64, 256], help="fixed cube resolutions")
    ap.add_argument("--k", type=float, nargs="*", default=[1.0, 3.0], help="adaptive pitches tau/k")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        t = time.time()
        ex = exact_residual(args.step, args.program, Path(tmp))
        Vm, Fm = load_mesh(Path(tmp) / "model.stl")
        print(f"exact (freecadcmd, {time.time() - t:.0f} s): input {ex['input_volume']:.4e}, model {ex['model_volume']:.4e}"
              f" (valid {ex['model_valid']}, {ex['model_solids']} solid), r+ {ex['r_plus']['volume']:.4e} in "
              f"{sum(s['volume'] > 1 for s in ex['r_plus']['solids'])} solids, r- {ex['r_minus']['volume']:.4e}",
              flush=True)

    Vi, Fi = load_mesh(args.stl)
    Vi = Vi * args.scale
    lo, hi = Vi.min(axis=0), Vi.max(axis=0)
    L = float((hi - lo).max())
    _, _, tau = wall_thickness(Vi, Fi)
    tau_q = adaptive_pitch(tau, fraction=1.0)
    grids = [(f"R={R}", L / R) for R in args.fixed] + [(f"tau/{k:g}", tau_q / k) for k in args.k]
    rows = []
    for name, pitch in grids:
        t = time.time()
        grid = box_grid(lo, hi, pitch)
        counts = dict(inp=0, model=0, plus=0, minus=0)

        def plus_stream():
            for a, b in zip(iter_slabs(Vi, Fi, grid), iter_slabs(Vm, Fm, grid)):
                counts["inp"] += int(a.sum()); counts["model"] += int(b.sum())
                counts["minus"] += int((b & ~a).sum())
                p = a & ~b
                counts["plus"] += int(p.sum())
                yield p

        topo = slab_topology(plus_stream(), pitch)
        v = pitch ** 3
        rec = dict(grid=name, pitch=pitch, seconds=round(time.time() - t, 1), input_volume=counts["inp"] * v,
                   model_volume=counts["model"] * v, r_plus_volume=counts["plus"] * v,
                   r_minus_volume=counts["minus"] * v, r_plus_voxels=counts["plus"], r_plus_components=topo.b0)
        rows.append(rec)
        print(f"  {name:>8s} pitch {pitch:7.2f} mm: r+ {counts['plus']:>9d} vox = {rec['r_plus_volume']:.3e} mm3 "
              f"({rec['r_plus_volume'] / ex['r_plus']['volume']:.2f} x exact) in {topo.b0} components; "
              f"r- {rec['r_minus_volume']:.3e}  [{rec['seconds']} s]", flush=True)
    if args.json:
        args.json.write_text(json.dumps(dict(program=str(args.program), step=str(args.step), tau_p5=tau_q, exact=ex,
                                             rows=rows), indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
