#!/usr/bin/env python3
"""
Backfill the standard base-point bits (<stem>.inside.npy, dataset_gen/src/inside_bits.py) for existing datasets, in
parallel, idempotent (samples that already have the file are skipped unless --force). The manifest records get
placement.inside = "<stem>.inside.npy" (JSONL rewritten atomically; run when no generator appends to the split).

Check: where a sample has a .field.npz (the exact mesh field), the bits must equal its occupancy channel (world_to_bin):
the median IoU per kind is printed; anything below 0.999 is listed.

    python3 scripts/add_inside_bits.py dataset_gen/local_dataset_v5/small/{train,val,test} --workers 12
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_gen"))


def _resolve(meta: Path, split_dir: Path, rel: str) -> Path:
    for base in [meta.parent, *meta.parents]:
        p = base / rel
        if p.exists():
            return p
        if base == split_dir:
            break
    return split_dir / rel


def _one(job):
    stl, bin_path, grid_mm, res, force = job
    from src.inside_bits import inside_path, write_inside, read_inside
    try:
        if force or not inside_path(bin_path).exists():
            write_inside(stl, bin_path, grid_mm, res)
        iou = None
        field = Path(bin_path).with_suffix(".field.npz")
        if field.exists():
            from src.field_channels import world_to_bin
            z = np.load(field)
            ch = z["channels"] if "channels" in z.files else z[z.files[0]]
            ref = world_to_bin(ch[0] > 0.5)
            ins = read_inside(bin_path, res)
            iou = float((ins & ref).sum() / max((ins | ref).sum(), 1))
        return str(bin_path), inside_path(bin_path).name, iou, None
    except Exception as exc:                                     # report, never abort the whole split
        return str(bin_path), None, None, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("splits", nargs="+", type=Path)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    for split in args.splits:
        split = split.resolve()
        metas = sorted(split.glob("shard*/*/dataset_metadata.jsonl"))
        jobs, where = [], {}
        for meta in metas:
            for i, line in enumerate(meta.read_text().splitlines()):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "stl_path" not in rec or not rec.get("voxel_path"):
                    continue
                stl = _resolve(meta, split, rec["stl_path"])
                bin_path = stl.with_suffix(".bin")
                grid_mm = float((rec.get("placement") or {}).get("grid_mm", 40.0))
                jobs.append((str(stl), str(bin_path), grid_mm, args.resolution, args.force))
                where[str(bin_path)] = rec.get("topology", "?") + "/" + rec.get("subclass", "full")
        results = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for b, name, iou, err in ex.map(_one, jobs, chunksize=4):
                results[b] = (name, iou, err)
        # manifests: placement.inside
        for meta in metas:
            out, changed = [], False
            for line in meta.read_text().splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    out.append(line)
                    continue
                if "stl_path" in rec and rec.get("voxel_path"):
                    b = str(_resolve(meta, split, rec["stl_path"]).with_suffix(".bin"))
                    name = results.get(b, (None,))[0]
                    if name and (rec.get("placement") or {}).get("inside") != name:
                        rec.setdefault("placement", {})["inside"] = name
                        line, changed = json.dumps(rec, separators=(",", ":")), True
                out.append(line)
            if changed:
                tmp = meta.with_suffix(".jsonl.tmp")
                tmp.write_text("\n".join(out) + "\n")
                os.replace(tmp, meta)
        errs = [(b, e) for b, (_, _, e) in results.items() if e]
        per = collections.defaultdict(list)
        for b, (_, iou, _) in results.items():
            if iou is not None:
                per[where[b]].append(iou)
        print(f"{split}: {len(results)} samples, {len(errs)} errors")
        for k in sorted(per):
            v = np.array(per[k])
            print(f"   {k:20s} n={len(v):4d}  IoU vs .field.npz median {np.median(v):.4f}  min {v.min():.4f}  <0.999: {(v < 0.999).sum()}")
        for b, e in errs[:10]:
            print("   ERROR", Path(b).name, e)


if __name__ == "__main__":
    main()
