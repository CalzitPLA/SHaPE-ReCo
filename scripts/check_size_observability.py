#!/usr/bin/env python3
"""
Is a shape's absolute size recoverable from its voxel grid?

The Dimension CNN is trained on absolute parameters (ratio x a fixed
scale_mm). If voxelization rescales every mesh to fill the grid, absolute
size never reaches the network and no amount of training can fix that -
the best possible output is the mean. This script measures it directly:
for a `measure`-mode dataset it reports how much the occupied-voxel
count varies across samples whose physical size varies enormously.

  preserved size  -> count scales with physical volume (corr ~ 1)
  erased size     -> count ~ constant (corr ~ 0, spread of a few percent)

Usage:
    python3 scripts/check_size_observability.py \\
        --dataset-root dataset_gen/local_dataset_bs_groups/measure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.manifest import load_manifest  # noqa: E402


def _volume_proxy(topology: str, p: dict) -> float | None:
    """Something proportional to physical volume, for the two shapes
    where that is a one-liner. Other topologies are skipped."""
    if topology == "sphere":
        return p["radius_ratio"] ** 3
    if topology == "box":
        return p["x_half_ratio"] * p["y_half_ratio"] * p["z_half_ratio"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()

    records = load_manifest(args.dataset_root, require_voxel=True)
    for topology in sorted({r.topology for r in records}):
        rows = [(_volume_proxy(r.topology, r.parameters), int(np.fromfile(r.voxel_path, dtype=np.uint8).sum()))
                for r in records if r.topology == topology]
        rows = [(v, c) for v, c in rows if v is not None]
        if len(rows) < 3:
            print(f"{topology}: skipped (no volume proxy defined or too few samples)")
            continue
        vol = np.array([v for v, _ in rows])
        cnt = np.array([c for _, c in rows])
        print(f"{topology}: n={len(rows)}")
        print(f"  physical volume varies   {vol.max() / vol.min():8.0f}x")
        print(f"  occupied voxels          min={cnt.min()}  max={cnt.max()}  "
              f"(spread {100 * (cnt.max() - cnt.min()) / cnt.mean():.1f}% of mean)")
        print(f"  corr(count, volume)      {np.corrcoef(cnt, vol)[0, 1]:.3f}"
              f"   (~1 => size preserved, ~0 => size erased by voxelization)")


if __name__ == "__main__":
    main()
