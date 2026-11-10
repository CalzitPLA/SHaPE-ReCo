#!/usr/bin/env python3
"""
How many parts do the reconstructions carry that their shape does not need? (the user, 2026-10-30: "minimize the
number of parts: simple trees for simple parts instead of complex trees for simple parts"). For every stored run of a
corpus with a known tree: the program sampled against its input (reco_gui/postprocess.py), then Minimize parts at the
given IoU tolerances; the parts before and after against the true tree's leaves, and the sampled IoU before and after.

    python3 scripts/minimize_parts.py "<runs>/*" --parts ~/shape_reco_results/validation/gate_paths \\
        --tol 0 0.002 0.005 --json docs/data/minimize_parts_gate.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402


def leaves_of(truth_json) -> int | None:
    try:
        d = json.loads(Path(truth_json).read_text())
    except (OSError, ValueError):
        return None
    n = 0
    stack = [d.get("program", d)]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            if x.get("type") == "Placed":
                n += 1
                continue
            stack.extend(v for v in x.values() if isinstance(v, (dict, list)))
        elif isinstance(x, list):
            stack.extend(x)
    return n or None


def group_of(stem: str) -> str:
    m = re.match(r"aegir_n(\d+)_([A-Z])_", stem)
    if m:
        n = int(m.group(1))
        return f"{dict(U='union', I='intersection', S='subtraction', P='one primitive')[m.group(2)]} " \
               f"{'1' if n == 1 else '2-3' if n <= 3 else '4-7' if n <= 7 else '8+'}"
    return "v5 single primitive" if stem.startswith("v5_") else "other"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", help="glob of run directories (record.json, program.json)")
    ap.add_argument("--parts", required=True, help="corpus directory with stl/<stem>.stl and truths/<stem>.json")
    ap.add_argument("--tol", type=float, nargs="+", default=[0.0, 0.002, 0.005])
    ap.add_argument("--samples", type=int, default=100_000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    from reco_engine.result_schema import ReconstructionResult
    from reco_gui import postprocess as pp
    import trimesh
    parts = Path(a.parts).expanduser()
    rows = []
    for d in sorted(glob.glob(os.path.expanduser(a.runs))):
        d = Path(d)
        stem = d.name
        stl = parts / "stl" / f"{stem}.stl"
        if not (d / "program.json").is_file() or not stl.is_file():
            continue
        t = time.time()
        result = ReconstructionResult.from_dict(json.loads((d / "program.json").read_text()))
        if not pp.accepted_ops(result):
            continue
        m = trimesh.load(stl, force="mesh")
        s = pp.sample_program(result, (np.asarray(m.vertices), np.asarray(m.faces)), n=a.samples)
        keep = np.ones(len(s.ops), bool)
        m0 = pp.metrics(s, keep)
        row = dict(part=stem, group=group_of(stem), leaves=leaves_of(parts / "truths" / f"{stem}.json"),
                   parts=len(s.ops), iou=m0["iou"], after={})
        for tol in a.tol:
            k2, log = pp.minimize(s, keep, tol=tol)
            row["after"][str(tol)] = dict(parts=int(k2.sum()), iou=pp.metrics(s, k2)["iou"])
        row["seconds"] = round(time.time() - t, 1)
        rows.append(row)
        print(f"{stem:34s} leaves {row['leaves'] or '-':>3} parts {row['parts']:3d} -> "
              + "  ".join(f"tol {k}: {v['parts']:3d} (IoU {row['iou']:.4f} -> {v['iou']:.4f})" for k, v in row["after"].items())
              + f"  [{row['seconds']} s]", flush=True)
    if not rows:
        sys.exit("no runs found")
    summary = {}
    for tol in a.tol:
        k = str(tol)
        before = [r["parts"] for r in rows]
        after = [r["after"][k]["parts"] for r in rows]
        summary[k] = dict(runs=len(rows), parts_before=sum(before), parts_after=sum(after),
                          share_dropped=1 - sum(after) / sum(before),
                          median_before=statistics.median(before), median_after=statistics.median(after),
                          iou_change_mean=statistics.mean(r["after"][k]["iou"] - r["iou"] for r in rows))
    groups = {}
    for r in rows:
        g = groups.setdefault(r["group"], dict(runs=0, leaves=0, parts=0, **{f"after_{k}": 0 for k in map(str, a.tol)}))
        g["runs"] += 1
        g["leaves"] += r["leaves"] or 0
        g["parts"] += r["parts"]
        for k in map(str, a.tol):
            g[f"after_{k}"] += r["after"][k]["parts"]
    print("\nper tolerance:")
    for k, v in summary.items():
        print(f"  tol {k}: parts {v['parts_before']} -> {v['parts_after']} ({100 * v['share_dropped']:.1f} % dropped), "
              f"median {v['median_before']} -> {v['median_after']}, mean IoU change {v['iou_change_mean']:+.4f}")
    print("\nper group (parts per run: true leaves | reconstructed | after minimize):")
    for gname, g in sorted(groups.items()):
        print(f"  {gname:24s} {g['runs']:3d} runs: {g['leaves'] / g['runs']:5.1f} | {g['parts'] / g['runs']:5.1f} | "
              + " | ".join(f"tol {k}: {g[f'after_{k}'] / g['runs']:5.1f}" for k in map(str, a.tol)))
    if a.json:
        Path(a.json).write_text(json.dumps(dict(schema="minimize_parts/1", runs=a.runs, tolerances=a.tol,
                                                samples=a.samples, summary=summary, groups=groups, rows=rows), indent=1))
        print(f"\nwritten: {a.json}")


if __name__ == "__main__":
    main()
