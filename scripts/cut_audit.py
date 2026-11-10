#!/usr/bin/env python3
"""
How often two cuts could be one (the user, 2026-10-29: "often two cuts could be done by only one; check the sizing of
cuts"): reco_engine/cut_merge.audit over stored runs - per run the cuts, the mergeable cut pairs, the redundant cuts and
the faces of cut boxes that stop short of open space (slack), and the summary over all runs.

    python3 scripts/cut_audit.py "<runs glob>" [...] [--json out.json]

A run directory is scripts/run_part.py's (program.json, record.json with the input) or the run store's (run.json).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]


def program_and_input(d: Path):
    from reco_engine.result_schema import ReconstructionResult
    from reco_engine.validation import program_ops
    if (d / "program.json").is_file():
        prog = json.loads((d / "program.json").read_text())
        rec = json.loads((d / "record.json").read_text()) if (d / "record.json").is_file() else {}
        inp = (rec.get("comparison") or {}).get("input_stl") or rec.get("input")
    else:
        run = json.loads((d / "run.json").read_text())
        prog, inp = dict(operations=run.get("program") or []), (run.get("input") or {}).get("path")
    res = ReconstructionResult.from_dict(dict(input_path=str(inp), operations=[
        dict(o, primitive=dict(o["primitive"], parts=o["primitive"].get("parts") or [])) for o in prog.get("operations") or []
        if o.get("accepted", True)]))
    ops = [(o.op, o.primitive) for o in res.operations]
    return ops, inp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--local-prefix", nargs=2, metavar=("SERVER", "LOCAL"), action="append", default=[],
                    help="map a server path prefix of the input to a local one")
    a = ap.parse_args(argv)
    from reco_engine.cut_merge import audit
    from src.closest_point_field import load_mesh
    rows = []
    for pattern in a.runs:
        for d in sorted(glob.glob(os.path.expanduser(pattern))):
            d = Path(d)
            try:
                ops, inp = program_and_input(d)
                for s, l in a.local_prefix:
                    inp = str(inp).replace(s, l)
                if inp and not Path(inp).is_file():                    # a mirror holds the STEP, not the STL
                    for alt in (Path(inp).with_suffix(".step"), Path(inp).parent.parent / "input.step"):
                        if alt.is_file():
                            inp = str(alt)
                            break
                if not inp or not Path(inp).is_file():
                    print(f"{d.name}: input not found ({inp})")
                    continue
                V, F = load_mesh(str(inp))
                r = audit(ops, V, F)
            except Exception as exc:
                print(f"{d.name}: {exc}")
                continue
            r.update(run=str(d), part=d.name)
            rows.append(r)
            print(f"{d.name}: {r['n_ops']} ops, {r['cuts']} cuts ({r['box_cuts']} boxes); mergeable pairs "
                  f"{[(p['i'], p['j']) for p in r['mergeable_pairs']]}; redundant {[x['k'] for x in r['redundant']]}; "
                  f"faces short of open space {sum(s['short_faces'] for s in r['slack'])}")
    if rows:
        n_runs = len(rows)
        with_pair = sum(bool(r["mergeable_pairs"]) for r in rows)
        cuts = sum(r["cuts"] for r in rows)
        in_pairs = sum(len({x for p in r["mergeable_pairs"] for x in (p["i"], p["j"])}) for r in rows)
        print(f"\n{n_runs} runs, {cuts} cuts: {with_pair} runs ({with_pair / n_runs:.0%}) have a mergeable cut pair; "
              f"{in_pairs} cuts ({in_pairs / max(1, cuts):.0%}) are in one; redundant cuts "
              f"{sum(len(r['redundant']) for r in rows)}; box faces short of open space "
              f"{sum(s['short_faces'] for r in rows for s in r['slack'])} of {6 * sum(r['box_cuts'] for r in rows)}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main()
