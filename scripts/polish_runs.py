#!/usr/bin/env python3
"""Polish finished runs of a validation corpus after the fact and rate them again (reco_engine/polish.py): the same
programs, the same truths, before -> after - no reconstruction rerun.

    python3 scripts/polish_runs.py <corpus dir> [--stems a,b] [--workers 4] [--tier B]

Per run directory <corpus>/runs/<stem> with program.json and rating.json: the input mesh (record.json "input", STL or
STEP via closest_point_field.load_field), polish_result on the accepted operations, then validation.rate against
<corpus>/truths/<stem>.json. Writes program_polished.json, polish.json and rating_polished.json next to the originals
(never overwrites them) and <corpus>/polish_summary.json; prints a table tier/IoU before -> after.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))


def polish_one(run_dir: str, truth_json: str) -> dict:
    from src.closest_point_field import load_field
    from reco_engine.polish import polish_result
    from reco_engine.result_schema import ReconstructionResult
    from reco_engine.truth import truth_from_json
    from reco_engine.validation import _json_default, load_program, rate
    run = Path(run_dir)
    t0 = time.time()
    out = dict(stem=run.name)
    try:
        before = json.loads((run / "rating.json").read_text())
        record = json.loads((run / "record.json").read_text())
        out.update(tier_before=before["grade"]["tier"], iou_before=before["vector"]["volume"]["iou"])
        result = ReconstructionResult.from_dict(json.loads((run / "program.json").read_text()))
        _, log = load_program(run)
        field = load_field(record["input"])
        report = polish_result(result, field.vertices, field.faces)
        (run / "program_polished.json").write_text(json.dumps(result.to_dict(), default=str))
        (run / "polish.json").write_text(json.dumps(report.as_dict(), indent=1, default=_json_default))
        truth = truth_from_json(truth_json)
        if truth.input_path is None:
            truth.input_path = (record.get("comparison") or {}).get("input_stl") or None
        after = rate(truth, result, log, record, model_mesh=run / "model.stl")
        (run / "rating_polished.json").write_text(json.dumps(after, indent=1, default=_json_default))
        regions = (after.get("attribution") or {}).get("regions") or []
        out.update(tier_after=after["grade"]["tier"], iou_after=after["vector"]["volume"]["iou"], kept=report.kept,
                   curved=len(report.ops), dominant_after=max(regions, key=lambda q: q["volume"])["cls"] if regions else None,
                   ops=[dict(kind=o.kind, kept=o.kept, scaled=o.scaled, moved_mm=o.moved_mm, rms=[o.rms_before, o.rms_after],
                             reason=o.reason) for o in report.ops])
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[-300:]}"
    out["seconds"] = time.time() - t0
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--stems", default=None, help="comma-separated run names (default: every rated run)")
    ap.add_argument("--tier", default=None, help="only runs of these tiers before, e.g. B or BC")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args(argv)
    runs = sorted(p.parent for p in (args.corpus / "runs").glob("*/rating.json") if (p.parent / "program.json").is_file())
    if args.stems:
        wanted = set(args.stems.split(","))
        runs = [r for r in runs if r.name in wanted]
    if args.tier:
        runs = [r for r in runs if json.loads((r / "rating.json").read_text())["grade"]["tier"] in args.tier]
    jobs = [(str(r), str(args.corpus / "truths" / f"{r.name}.json")) for r in runs
            if (args.corpus / "truths" / f"{r.name}.json").is_file()]
    print(f"{len(jobs)} runs to polish and rate, {args.workers} workers", flush=True)
    rows = []
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(polish_one, *zip(*jobs)) if jobs else []:
            rows.append(row)
            if "error" in row:
                print(f"  {row['stem']}: ERROR {row['error']}", flush=True)
            else:
                print(f"  {row['tier_before']} -> {row['tier_after']}  {row['stem'][:46]:46s} IoU {row['iou_before']:.5f} -> "
                      f"{row['iou_after']:.5f}  kept {row['kept']}/{row['curved']}  ({row['seconds']:.0f} s)", flush=True)
    ok = [r for r in rows if "error" not in r]
    moves = {}
    for r in ok:
        key = f"{r['tier_before']}->{r['tier_after']}"
        moves[key] = moves.get(key, 0) + 1
    summary = dict(corpus=str(args.corpus), runs=len(rows), errors=len(rows) - len(ok), moves=moves,
                   tiers_before={t: sum(r["tier_before"] == t for r in ok) for t in "ABCDE"},
                   tiers_after={t: sum(r["tier_after"] == t for r in ok) for t in "ABCDE"}, rows=rows)
    (args.corpus / "polish_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("tiers before:", summary["tiers_before"], " after:", summary["tiers_after"], " moves:", moves)
    return 0


if __name__ == "__main__":
    sys.exit(main())
