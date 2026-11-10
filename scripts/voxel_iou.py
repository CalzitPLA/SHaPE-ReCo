#!/usr/bin/env python3
"""IoU of two closed meshes by ray-parity voxels on one fine grid (dataset_gen/src/local_grids: slab-wise, no dense
float grid) - the check when OCCT booleans fail on coincident faces (the liner's model cylinder lies on its outer face).

    python3 scripts/voxel_iou.py <model.stl|.step> <input.stl|.step> --pitch 2.0 [--scale-model 1.0]

Prints {cells, model, input, common, union, iou, missing, invented} (volumes in input units^3). The error of a voxel
volume is about (surface area x pitch / 2) / volume - choose the pitch well below the thinnest wall.
Checked 2026-09-25 on the frame's 96^3 cube program: see docs/LINER.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("input")
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--scale-model", type=float, default=1.0)
    args = ap.parse_args(argv)
    from src.closest_point_field import load_mesh
    from src import local_grids
    Vm, Fm = load_mesh(args.model, 0.02)
    Vi, Fi = load_mesh(args.input, 0.02)
    Vm = np.asarray(Vm, float) * args.scale_model
    Vi = np.asarray(Vi, float)
    lo = np.minimum(Vm.min(0), Vi.min(0))
    hi = np.maximum(Vm.max(0), Vi.max(0))
    grid = local_grids.box_grid(lo, hi, args.pitch)
    n_m = n_i = n_c = n_u = 0
    for a, b in zip(local_grids.iter_slabs(Vm, Fm, grid), local_grids.iter_slabs(Vi, Fi, grid)):
        n_m += int(a.sum()); n_i += int(b.sum()); n_c += int((a & b).sum()); n_u += int((a | b).sum())
    v = args.pitch ** 3
    print("VOXEL_IOU " + json.dumps(dict(cells=int(np.prod(grid.shape)), pitch=args.pitch, model=n_m * v, input=n_i * v,
                                         common=n_c * v, union=n_u * v, iou=n_c / n_u if n_u else 0.0,
                                         missing=(n_i - n_c) * v, invented=(n_m - n_c) * v)))


if __name__ == "__main__":
    main()
