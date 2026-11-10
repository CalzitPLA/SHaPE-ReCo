#!/usr/bin/env python3
"""
The difference grade of finished feature-stage runs (reco_engine/difference_grade.py): what the rough model's miss
consists of - skin, features, open pieces - the IoU it is predicted to reach, and, with --apply, the IoU it does
reach when its features are built.

    python3 scripts/grade_difference.py <run_dir> [<run_dir> ...] [--apply] [--delta0 <length>] [--json out.json]

<run_dir>: an output directory of scripts/feature_stage.py. Writes <run_dir>/grade.json and prints one block per run
and a table of all of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine.difference_grade import apply_features, format_grade, grade_difference  # noqa: E402


def grade_run(run: Path, delta0=None, apply: bool = False) -> dict:
    from src.closest_point_field import load_field
    out = json.loads((run / "features.json").read_text())
    s = out["summary"]
    pieces = json.loads((run / "pieces" / "pieces.json").read_text())
    residual = dict(input_volume=s["input_volume"], r_minus=s["r_minus"], iou=s["rough_iou"],
                    pieces=pieces["pieces"], left_out_volume=s.get("left_out_volume"))
    g = grade_difference(residual, load_field(s["input"]), load_field(str(run / "rough.stl")), out["features"], delta0)
    d = g.as_dict()
    d["name"] = run.name
    if apply:
        d["applied"] = apply_features(run)
    (run / "grade.json").write_text(json.dumps(d, indent=1))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--apply", action="store_true", help="build rough model + features and measure the exact IoU")
    ap.add_argument("--delta0", type=float, default=None, help="skin tolerance (default: min of 1e-3 of the input's diagonal and a tenth of its thinnest wall)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    rows = []
    for run in (Path(r) for r in args.runs):
        if not (run / "features.json").is_file():
            print(f"skipped {run}: no features.json")
            continue
        d = grade_run(run, args.delta0, args.apply)
        rows.append(d)
        from reco_engine.difference_grade import DifferenceGrade, PieceGrade
        g = DifferenceGrade(**{**{k: v for k, v in d.items() if k not in ("name", "applied", "pieces")},
                               "pieces": [PieceGrade(**p) for p in d["pieces"]]})
        print(f"== {run.name}\n{format_grade(g)}")
        a = d.get("applied")
        if a:
            print("applied: " + (a["error"] if "error" in a else
                                 f"{a['applied']} features, {a['operations']} operations, valid {a['valid']}, "
                                 f"exact IoU {a['iou']:.4f} (predicted {d['reach']:.4f}, "
                                 f"difference {a['iou'] - d['reach']:+.4f})"))
    print(f"\n{'part':<16} {'IoU':>7} {'skin':>7} {'feature':>8} {'open':>7} {'n s/f/o':>9} {'reach':>7} "
          f"{'effort':>6} {'measured':>9}")
    for d in rows:
        a = d.get("applied") or {}
        print(f"{d['name']:<16} {d['iou']:>7.4f} {d['e_skin']:>7.4f} {d['e_feature']:>8.4f} {d['e_open']:>7.4f} "
              f"{d['n_skin']:>3}/{d['n_feature']}/{d['n_open']:<3} {d['reach']:>7.4f} {d['effort']:>6} "
              f"{a['iou']:>9.4f}" if "iou" in a else
              f"{d['name']:<16} {d['iou']:>7.4f} {d['e_skin']:>7.4f} {d['e_feature']:>8.4f} {d['e_open']:>7.4f} "
              f"{d['n_skin']:>3}/{d['n_feature']}/{d['n_open']:<3} {d['reach']:>7.4f} {d['effort']:>6} {'-':>9}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    main()
