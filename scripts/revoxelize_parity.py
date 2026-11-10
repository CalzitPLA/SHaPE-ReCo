#!/usr/bin/env python3
"""
Re-voxelise an EXISTING dataset with exact ray parity (decision D-vox, 2026-09-23): the clean .bin becomes the sample's
standard base-point bits (<stem>.inside.npy, dataset_gen/src/inside_bits.py: parity of the placed solid at the voxel centres
of the exact in-scale grid) - the same grid SHAPE_RECO_VOXELIZER=parity (voxelization._voxelize_parity) writes for new data
and the same parity the reconstruction loop uses at inference. stltovoxel's anchor rounding had displaced the stored grids
by up to half a voxel per sample (IoU 0.84 median vs the exact solid, 0.0 on a 26-voxel torus).

Per record (run scripts/add_inside_bits.py first; missing bits are computed here):
  - the old grid is kept as <stem>.stltovoxel.bin (only the first time), the .bin is overwritten with the parity grid;
  - an earlier post-hoc noise is undone: <stem>.clean.bin is removed, placement.noise / noise_posthoc / voxels_clean dropped
    (re-noise afterwards with scripts/add_voxel_noise.py);
  - placement.voxelizer = "parity", placement.voxels, voxel_volume_mm3 updated;
  - placement.degenerate = true when the solid has fewer than --min-voxels voxels (training/subclass_schema.filter_records
    skips those).

    python3 scripts/revoxelize_parity.py dataset_gen/local_dataset_v5/medium/{train,val,test} --min-voxels 200
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.inside_bits import read_inside, write_inside  # noqa: E402


def _resolve(meta: Path, split_dir: Path, rel: str) -> Path:
    for base in [meta.parent, *meta.parents]:
        p = base / rel
        if p.exists():
            return p
        if base == split_dir:
            break
    return split_dir / rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", nargs="+", type=Path)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--min-voxels", type=int, default=200)
    args = ap.parse_args()
    r = args.resolution
    for split in args.splits:
        split = split.resolve()
        st = collections.Counter()
        for meta in sorted(split.glob("shard*/*/dataset_metadata.jsonl")):
            out, changed = [], False
            for line in meta.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if "stl_path" not in rec or not rec.get("voxel_path"):
                    out.append(line)
                    continue
                stl = _resolve(meta, split, rec["stl_path"])
                bin_path = stl.with_suffix(".bin")
                pl = dict(rec.get("placement") or {})
                grid_mm = float(pl.get("grid_mm", 40.0))
                inside = read_inside(bin_path, r)
                if inside is None:
                    pl["inside"] = write_inside(stl, bin_path, grid_mm, r)
                    inside = read_inside(bin_path, r)
                    st["inside_computed"] += 1
                old = bin_path.with_name(bin_path.stem + ".stltovoxel.bin")
                current = np.fromfile(bin_path, dtype=np.uint8).reshape((r,) * 3) > 0 if bin_path.exists() else None
                already = current is not None and np.array_equal(current, inside)       # new data: parity voxelizer
                if bin_path.exists() and not old.exists() and pl.get("voxelizer") != "parity" and not already:
                    os.replace(bin_path, old)
                clean = bin_path.with_name(bin_path.stem + ".clean.bin")
                if clean.exists():
                    clean.unlink()
                inside.astype(np.uint8).tofile(bin_path)             # .bin order already
                for k in ("noise", "noise_posthoc", "voxels_clean"):
                    pl.pop(k, None)
                n = int(inside.sum())
                pl["voxelizer"], pl["voxels"] = "parity", n
                pl["degenerate"] = n < args.min_voxels
                rec["voxel_volume_mm3"] = float(n * (grid_mm / r) ** 3)
                rec["placement"] = pl
                out.append(json.dumps(rec, separators=(",", ":")))
                changed = True
                st["degenerate" if pl["degenerate"] else "ok"] += 1
                if pl["degenerate"]:
                    st[f"degenerate:{rec.get('topology')}/{rec.get('subclass', 'full')}"] += 1
            if changed:
                tmp = meta.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(out) + "\n")
                os.replace(tmp, meta)
        deg = {k.split(":", 1)[1]: v for k, v in st.items() if k.startswith("degenerate:")}
        print(f"{split}: re-voxelised {st['ok'] + st['degenerate']} ({st['inside_computed']} bits computed here), "
              f"degenerate (<{args.min_voxels} voxels) {st['degenerate']}: {deg or '-'}")


if __name__ == "__main__":
    main()
