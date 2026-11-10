#!/usr/bin/env python3
"""
The exact genus of a part, and its ROUGH genus at growing voxel resolutions: a mechanical part often has many small real
holes (bolt holes, rivets) that make the exact genus large and dominated by features far smaller than what "how many big
loops does this part have" usually means (`SurfaceField.genus`/`coarse_genus`/`genus_by_resolution`,
`dataset_gen/src/closest_point_field.py`). Case study: `docs/FRAME_FAILURE.md`.

    python3 scripts/genus_sweep.py part.stl [--resolutions 8 16 32 64 128]

Read the HIGH end of the sweep, where the number settles and stays there as resolution grows further - a single mid-sweep
value can dip due to voxel-grid-alignment aliasing right at the resolution that just barely resolves a feature (see
`genus_by_resolution`'s docstring). On `Frame_4_x_6` (docs/FRAME_FAILURE.md): genus 1 up to resolution 16 (a plain frame,
the ~150 bridging rods too thin to resolve), 9 from 24 to 96 (about half the rods resolved), the full 17 only at 128 (voxel
pitch close to the rods' own 54 mm diameter) - so THIS part has no single "rough genus", it depends on the scale asked
for, which is exactly the point.

`reco_field.peel.predict_ops_budget` runs the same sweep (stopping early once it catches up with the exact genus, or a
wall-clock cap) to set `peel()`'s own operation budget by default - this script is for looking at the whole sweep by
hand, `predict_ops_budget` for a single number a caller (the GUI, the loop itself) can act on.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src.closest_point_field import load_field  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--resolutions", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    args = ap.parse_args()
    field = load_field(args.path)
    print(f"{Path(args.path).name}: {len(field.faces)} triangles, closed={field.is_closed()}, exact genus={field.genus():.0f}")
    for r in args.resolutions:
        t = time.time()
        g = field.coarse_genus(r)
        print(f"  resolution {r:4d}  ->  coarse genus {g:.0f}   ({time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
