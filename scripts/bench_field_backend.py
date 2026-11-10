#!/usr/bin/env python3
"""Time the one-pass closest-point query (SurfaceField.query, exact=False, on the refined mesh as field_channels.mesh_channels
uses it) with each backend on real v5 solids, and check that the backends agree. The evidence for the torch backend of
dataset_gen/src/closest_point_field.py (training plan v5, cost model).

    python3 scripts/bench_field_backend.py dataset_gen/local_dataset_v5/small/train --n 6 --json docs/data/field_backend_bench.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import field_channels as fc  # noqa: E402
from src.closest_point_field import load_field  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backends", nargs="+", default=["numpy", "torch", "cuda"],
                    help="numpy | torch | cuda | igl (igl: libigl's exact query on the RAW mesh, the mesh_channels path)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    stls = sorted(args.root.rglob("*.solid.stl"))
    pick = [stls[i] for i in np.random.default_rng(args.seed).choice(len(stls), min(args.n, len(stls)), replace=False)]
    spec = fc.inscale_spec(40.0, 64)
    pts = spec.centres().reshape(-1, 3)
    rows = []
    for stl in pick:
        raw = load_field(stl)
        rf = fc.refined_field(raw, fc.REFINE_EDGE_VOXELS * spec.pitch)
        row = dict(stl=str(stl.relative_to(ROOT)) if ROOT in stl.parents else str(stl), faces=int(len(rf.faces)),
                   raw_faces=int(len(raw.faces)))
        ref = None
        for b in args.backends:
            os.environ["SHAPE_RECO_FIELD_BACKEND"] = b
            t = time.time()
            q = (raw if b == "igl" else rf).query(pts, exact=False)      # igl is exact on the raw mesh (no refinement)
            row[f"{b}_s"] = time.time() - t
            if ref is None:
                ref = q.distance
            else:
                row[f"{b}_max_abs_diff_mm"] = float(np.abs(q.distance - ref).max())
        print(json.dumps(row))
        rows.append(row)
    out = dict(grid=64, points=int(len(pts)), refine_edge_voxels=fc.REFINE_EDGE_VOXELS, k=fc.REFINE_K, rows=rows)
    for b in args.backends[1:]:
        out[f"speedup_{b}_median"] = float(np.median([r[f"{args.backends[0]}_s"] / r[f"{b}_s"] for r in rows]))
    print({k: v for k, v in out.items() if k.startswith("speedup")})
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
