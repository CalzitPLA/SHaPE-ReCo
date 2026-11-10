#!/usr/bin/env python3
"""
How many of a part's true leaves do its own surfaces reveal? (plan of 2026-10-30, step 4: "measure first, then build").
For every STEP part with a known tree: reco_engine/leaf_recognition.recognize, then the true tree's leaves matched
(reco_engine.validation.match_leaves, the rating's visibility rule) against
    (a) all verified hypotheses       - the upper bound of what the surfaces reveal
    (b) the selected program          - the fewest that explain the part (its sampled IoU and parts too)
    (c) today's loop                  - its stored rating.json (vector.tree), its IoU and parts
per class (union / intersection / subtraction / single by leaf count). TEST parts must not be passed here.

    python3 scripts/leaf_recognition_study.py --parts ~/shape_reco_results/validation/gate_paths \\
        --loop "~/shape_reco_results/protocol/rough_tidy_v2/runs_rt2_dev_on" --out docs/data/leaf_recognition_study
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

import numpy as np  # noqa: E402


def group_of(stem: str) -> str:
    m = re.match(r"aegir_n(\d+)_([A-Z])_", stem)
    if not m:
        return "machined" if stem.startswith("mach_") else "other"
    n = int(m.group(1))
    return f"{dict(U='union', I='intersection', S='subtraction', P='single')[m.group(2)]} " \
           f"{'1' if n == 1 else '2-3' if n <= 3 else '4-7' if n <= 7 else '8+'}"


def match(truth, ops, n_surface: int = 20_000, model_visibility: bool = True):
    """match_leaves with the rating's visibility rule (reco_engine/validation.rate_run). `model_visibility` False: every
    leaf of the model counts (the upper bound (a): the hypotheses are not one program, their surfaces hide each other)."""
    from reco_engine.csg_surface import compile_program
    from reco_engine.result_to_program import primitive_to_sp
    from reco_engine.validation import match_leaves, surface_metrics
    if not model_visibility:
        ops = [("add" if op == "cover" else op, p) for op, p in ops]
        ops = [o for o in ops if o[0] == "add"] + [o for o in ops if o[0] != "add"]
    if not ops or ops[0][0] != "add":
        return dict(precision=0.0, recall=0.0, matched=0, truth_leaves=len(truth.compiled.leaves), model_leaves=0)
    model = compile_program([(op, primitive_to_sp(p)) for op, p in ops])
    surf = surface_metrics(truth, model, n=n_surface, seed=0)
    s = surf.pop("_samples")
    vis_t = {int(i) for i, c in zip(*np.unique(s["truth"].leaf, return_counts=True)) if c >= 1e-3 * len(s["truth"].leaf)}
    vis_m = {int(i) for i, c in zip(*np.unique(s["model"].leaf, return_counts=True)) if c >= 1e-3 * len(s["model"].leaf)}
    m = match_leaves(truth.compiled, model, truth.diag, vis_t, vis_m if model_visibility else None)
    return {k: m[k] for k in ("precision", "recall", "matched", "truth_leaves", "model_leaves")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", required=True, help="corpus directory: step/<stem>.step, truths/<stem>.json")
    ap.add_argument("--loop", default=None, help="the runs of today's loop on these parts: <dir>/<stem>/{record,rating}.json")
    ap.add_argument("--out", default="docs/data/leaf_recognition_study")
    ap.add_argument("--exclude", default=None, help="a protocol manifest whose TEST stems must not be studied")
    a = ap.parse_args(argv)
    from reco_engine import leaf_recognition as lr
    from reco_engine.truth import truth_from_json
    parts = Path(a.parts).expanduser()
    loop = Path(a.loop).expanduser() if a.loop else None
    banned = set()
    if a.exclude:
        man = json.loads(Path(a.exclude).read_text())
        banned = {x["stem"] if isinstance(x, dict) else x for x in man.get("test", [])}
    rows = []
    for step in sorted((parts / "step").glob("*.step")):
        stem = step.stem
        truth_path = parts / "truths" / f"{stem}.json"
        if stem in banned or not truth_path.is_file():
            continue
        t = time.time()
        try:
            rec = lr.recognize(step)
        except Exception as exc:
            print(f"{stem}: failed ({exc})", flush=True)
            continue
        truth = truth_from_json(truth_path)
        row = dict(part=stem, group=group_of(stem), truth_leaves=len(truth.compiled.leaves),
                   surfaces=len(rec["surfaces"]), hypotheses=len(rec["hypotheses"]), passed=len(rec["passed"]),
                   selected=rec["info"], seconds_recognize=round(time.time() - t, 1))
        row["a"] = match(truth, [(h.role, h.primitive) for h in rec["passed"]], model_visibility=False)
        row["b"] = match(truth, [(o.op, o.primitive) for o in rec["program"].operations])
        row["b_parts"] = len(rec["program"].operations)
        row["b_iou"] = rec["info"].get("iou")
        if loop is not None and (loop / stem / "rating.json").is_file():
            r = json.loads((loop / stem / "rating.json").read_text())
            rc = json.loads((loop / stem / "record.json").read_text())
            tr = (r.get("vector") or {}).get("tree") or {}
            comp = rc.get("comparison") or {}
            row["c"] = {k: tr.get(k) for k in ("precision", "recall", "matched", "truth_leaves", "model_leaves")}
            row["c_iou"] = (comp.get("exact") or {}).get("iou") if (comp.get("exact") or {}).get("iou") is not None \
                else (comp.get("sampled") or {}).get("iou")
            row["c_parts"] = (rc.get("model") or {}).get("operations")
            row["c_tier"] = (r.get("grade") or {}).get("tier")
        row["seconds"] = round(time.time() - t, 1)
        rows.append(row)
        c = row.get("c") or {}
        print(f"{stem:30s} {row['group']:16s} leaves {row['truth_leaves']:2d} | surfaces: recall {row['a']['recall']:.2f} "
              f"| selected {row['b_parts']:2d} parts recall {row['b']['recall']:.2f} IoU {row['b_iou'] or 0:.3f} "
              f"| loop {c.get('model_leaves') or '-'} parts recall {c.get('recall') if c.get('recall') is not None else '-'} "
              f"IoU {row.get('c_iou') or 0:.3f}  [{row['seconds']} s]", flush=True)
    if not rows:
        sys.exit("no parts")
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
        groups.setdefault("all", []).append(r)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else None
    summary = {}
    for g, rs in sorted(groups.items()):
        summary[g] = dict(parts=len(rs), truth_leaves=mean([r["truth_leaves"] for r in rs]),
                          a_recall=mean([r["a"]["recall"] for r in rs]),
                          b_recall=mean([r["b"]["recall"] for r in rs]), b_precision=mean([r["b"]["precision"] for r in rs]),
                          b_parts=mean([r["b_parts"] for r in rs]), b_iou=mean([r["b_iou"] for r in rs]),
                          b_good=sum(1 for r in rs if (r["b_iou"] or 0) >= 0.95),
                          c_recall=mean([(r.get("c") or {}).get("recall") for r in rs]),
                          c_parts=mean([r.get("c_parts") for r in rs]), c_iou=mean([r.get("c_iou") for r in rs]),
                          c_good=sum(1 for r in rs if (r.get("c_iou") or 0) >= 0.95))
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(dict(schema="leaf_recognition_study/1", parts=str(parts),
                                                        loop=str(loop) if loop else None, summary=summary, rows=rows), indent=1))
    L = ["# Leaves recognized from the part's own surfaces", "",
         f"`scripts/leaf_recognition_study.py --parts {a.parts}" + (f" --loop {a.loop}`" if a.loop else "`"), "",
         "Leaf recall / precision: the true tree's visible leaves matched (`validation.match_leaves`: centre and size within "
         "2 % of the diagonal, axis within 5 deg, sweep within 0.05 rad). (a) all verified hypotheses, (b) the selected "
         "program, (c) today's loop (its stored rating). IoU: (b) sampled against the input on its own, (c) the run's exact "
         "IoU. 'good' here: IoU >= 0.95 (the one-solid check of (b) is not made).", "",
         "| class | parts | true leaves | (a) recall | (b) recall | (b) precision | (b) parts | (b) IoU | (b) IoU >= 0.95 | "
         "(c) recall | (c) parts | (c) IoU | (c) IoU >= 0.95 |", "|" + "---|" * 13]

    def f(x, nd=2):
        return "-" if x is None else f"{x:.{nd}f}"
    for g, v in summary.items():
        L.append(f"| {g} | {v['parts']} | {f(v['truth_leaves'], 1)} | {f(v['a_recall'])} | {f(v['b_recall'])} | "
                 f"{f(v['b_precision'])} | {f(v['b_parts'], 1)} | {f(v['b_iou'], 3)} | {v['b_good']} | {f(v['c_recall'])} | "
                 f"{f(v['c_parts'], 1)} | {f(v['c_iou'], 3)} | {v['c_good']} |")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
