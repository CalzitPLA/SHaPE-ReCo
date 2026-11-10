#!/usr/bin/env python3
"""
A filtered VIEW of a generated dataset: the same samples minus those whose label the voxel grid cannot resolve. Nothing
is copied or deleted - for every dataset_metadata.jsonl below a source root, a manifest of the kept records is written
at the same relative place below the output root, with the sample paths made absolute, so training/manifest.load_manifest
reads the view like any dataset root (the sibling files <stem>.inside.npy / .field.npz are found beside the voxel file).

    python3 scripts/dataset_hygiene.py --src <root> [<root> ...] --out <view root> [--base <common parent>]
                                       [--min-cone-taper-vox 2.0] [--resolution 64] [--report report.json]

A source root <base>/<rel> is written to <view root>/<rel> (--base defaults to the roots' common parent).

Rule (measured 2026-10-10 on the clean v5 set, scripts/type_predictions.py): a cone whose two radii differ by less than
2 voxels at the placed scale is not told from a cylinder - the type nets name such test cones right in 13 of 34 cases
(dirlog_inside, 19,075 training samples), against 90 of 117 for the other plain cones. `--min-cone-taper-vox` drops
every record that has radius1 and radius2 with |radius1 - radius2| / (grid_mm / resolution) below it; 0 keeps all.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path


def resolve(rel: str, meta_file: Path, root: Path) -> Path:
    """training/manifest._resolve: the generator writes paths relative to ITS output root (the root or a shard)."""
    for base in meta_file.parents:
        if root == base or root in base.parents:
            cand = (base / rel).resolve()
            if cand.exists():
                return cand
    return (root / rel).resolve()


def cone_taper_vox(entry: dict, resolution: int):
    d = entry.get("dims_mm") or {}
    grid = (entry.get("placement") or {}).get("grid_mm")
    if "radius1" not in d or "radius2" not in d or not grid:
        return None
    return abs(float(d["radius1"]) - float(d["radius2"])) / (float(grid) / resolution)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", type=Path, default=None)
    ap.add_argument("--min-cone-taper-vox", type=float, default=2.0)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--report", type=Path, default=None)
    a = ap.parse_args(argv)

    srcs = [s.resolve() for s in a.src]
    base = a.base.resolve() if a.base else Path(os.path.commonpath([str(s) for s in srcs])).parent if len(srcs) == 1 \
        else Path(os.path.commonpath([str(s) for s in srcs]))
    report = dict(rule=dict(min_cone_taper_vox=a.min_cone_taper_vox, resolution=a.resolution), base=str(base), roots={})
    for src in srcs:
        rel = src.relative_to(base)
        kept, dropped = collections.Counter(), collections.Counter()
        for meta in sorted(src.rglob("dataset_metadata.jsonl")):
            out = a.out / rel / meta.relative_to(src)
            out.parent.mkdir(parents=True, exist_ok=True)
            with meta.open() as fh, out.open("w") as oh:
                for line in fh:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e.get("record_type") == "dataset_header" or "stl_path" not in e:
                        oh.write(line if line.endswith("\n") else line + "\n")
                        continue
                    label = f"{e.get('topology')}/{e.get('subclass') or '-'}"
                    t = cone_taper_vox(e, a.resolution)
                    if t is not None and t < a.min_cone_taper_vox:
                        dropped[label] += 1
                        continue
                    e["stl_path"] = str(resolve(e["stl_path"], meta, src))
                    if e.get("voxel_path"):
                        e["voxel_path"] = str(resolve(e["voxel_path"], meta, src))
                    oh.write(json.dumps(e) + "\n")
                    kept[label] += 1
        report["roots"][str(rel)] = dict(kept=sum(kept.values()), dropped=sum(dropped.values()),
                                         dropped_by_label=dict(sorted(dropped.items())), kept_by_label=dict(sorted(kept.items())))
        print(f"{rel}: kept {sum(kept.values())}, dropped {sum(dropped.values())} {dict(sorted(dropped.items()))}", flush=True)
    rp = a.report or (a.out / "hygiene_report.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    prev = json.loads(rp.read_text()) if rp.is_file() else None
    if prev and prev.get("rule") == report["rule"]:
        prev["roots"].update(report["roots"])
        report = prev
    rp.write_text(json.dumps(report, indent=1) + "\n")
    print(f"report -> {rp}")


if __name__ == "__main__":
    main()
