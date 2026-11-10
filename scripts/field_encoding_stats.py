#!/usr/bin/env python3
"""
How much of each network input kind actually carries direction and distance, and how that changes with the grid
resolution - the measurement behind the "dirlog" input (training/field_input.py).

For analytic solids (dataset_gen/src/sdf_primitives.py, off-centre and rotated, on the in-scale 40 mm grid) at 32..128^3,
per input kind ("fused": occupancy, sdf/T, V/T; "dirlog": material, unit normal, log distance):

  band_dir_weight    mean length of the direction channels on voxels within 1.5 voxels of the surface
                     (where the direction IS the surface normal - the shape information)
  far_dir_weight     the same beyond 8 voxels (towards the medial axis, where the direction is least informative)
  dist_saturated     share of the grid where the distance channel is at its clip limit (|value| >= 0.999)
  normal_err_deg     median angle between the grid-derived direction and the analytic normal in the band
                     (the same smoothed-SDF gradient feeds both kinds, so this is one number per solid/resolution)

    python3 scripts/field_encoding_stats.py                    # -> docs/data/field_encoding_stats.json + a table
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src.sdf_primitives import Box, Cone, Cylinder, Placed, Sphere, Torus, sdf_grid  # noqa: E402
from training.field_input import encode  # noqa: E402

GRID_MM = 40.0
RESOLUTIONS = (32, 64, 96, 128)


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


SOLIDS = {
    "box": Placed(Box((9.0, 5.5, 3.5)), _rot((1, 2, 0.5), 27), np.array([1.3, -2.1, 0.7])),
    "cylinder": Placed(Cylinder(5.0, 16.0), _rot((0.3, 1, 0.2), 38), np.array([-1.1, 0.9, -7.2])),
    "sphere": Placed(Sphere(8.5), np.eye(3), np.array([2.2, -1.4, 0.6])),
    "cone": Placed(Cone(7.0, 2.5, 14.0), _rot((1, 0, 0.4), 22), np.array([0.8, 1.6, -6.0])),
    "torus": Placed(Torus(9.0, 3.0), _rot((0.2, 1, 0), 31), np.array([-0.7, 1.1, 0.4])),
}


def measure(solid, resolution: int) -> dict:
    pitch = GRID_MM / resolution
    sdf = sdf_grid(solid, resolution, GRID_MM) / pitch                         # voxels, array frame, negative inside
    occ = sdf <= 0
    true_n = np.stack(np.gradient(sdf), axis=0)
    true_n /= np.maximum(np.linalg.norm(true_n, axis=0, keepdims=True), 1e-9)
    band, far = np.abs(sdf) < 1.5, np.abs(sdf) > 8.0
    out = {}
    fused = encode(occ.astype(np.float32) * 2 - 1, "fused")
    dirlog = encode(occ.astype(np.float32) * 2 - 1, "dirlog")
    for kind, dir_ch, dist_ch in (("fused", fused[2:5], fused[1]), ("dirlog", dirlog[1:4], dirlog[4])):
        length = np.linalg.norm(dir_ch, axis=0)
        out[kind] = dict(band_dir_weight=float(length[band].mean()), far_dir_weight=float(length[far].mean()) if far.any() else None,
                         dist_saturated=float((np.abs(dist_ch) >= 0.999).mean()))
    n = dirlog[1:4]
    cos = np.clip((n * true_n).sum(0), -1, 1)
    out["normal_err_deg"] = float(np.median(np.degrees(np.arccos(cos[band]))))
    out["band_voxels"] = int(band.sum())
    return out


def main():
    results = {name: {str(r): measure(s, r) for r in RESOLUTIONS} for name, s in SOLIDS.items()}
    path = ROOT / "docs/data/field_encoding_stats.json"
    path.write_text(json.dumps(dict(grid_mm=GRID_MM, resolutions=RESOLUTIONS, results=results), indent=1))
    print(f"{'solid':9s} {'R':>4s} | {'fused: band dir':>15s} {'far dir':>7s} {'dist sat':>8s} | {'dirlog: band dir':>16s} {'far dir':>7s} {'dist sat':>8s} | normal err")
    for name, per in results.items():
        for r, m in per.items():
            f, d = m["fused"], m["dirlog"]
            print(f"{name:9s} {r:>4s} | {f['band_dir_weight']:15.3f} {f['far_dir_weight'] or 0:7.3f} {f['dist_saturated']:8.1%} | "
                  f"{d['band_dir_weight']:16.3f} {d['far_dir_weight'] or 0:7.3f} {d['dist_saturated']:8.1%} | {m['normal_err_deg']:.1f} deg")
    print("wrote", path)


if __name__ == "__main__":
    main()
