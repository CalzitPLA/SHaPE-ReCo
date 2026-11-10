#!/usr/bin/env python3
"""Per-sample generation cost of a v5 dataset, from what the generators recorded - the cost model of the training plan
(docs/training_plan_v5.tex, decision D-N).

Sources (nothing is re-run):
  every record          placement.field.seconds         the exact-field step (sample_writer.write_field)
  special records       seconds_build / seconds_export  FreeCAD build and STL export of hollow / tube / rhs / frame
  <split>_ibeam.log     the generator's closing JSON    wall seconds / samples of the I-beam generator
  run log               "generation took N min"         wall time of the whole make_dataset_v5.sh run

    python3 scripts/timing_report.py dataset_gen/local_dataset_v5/small \\
        --run-log dataset_gen/local_dataset_v5_small_run.log --json docs/data/v5_small_timing.json

Times are wall seconds measured while several generators shared the 16 laptop cores, so they are upper bounds of the cost
on a dedicated core. CPU-seconds per sample x samples / cores gives the wall time of a larger set.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _find(d, key):
    """First value of `key` anywhere in the nested dict/list `d`."""
    if isinstance(d, dict):
        if key in d:
            return d[key]
        d = list(d.values())
    if isinstance(d, list):
        for v in d:
            got = _find(v, key)
            if got is not None:
                return got
    return None


def summarize(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return None
    return dict(n=len(v), median=statistics.median(v), p90=v[int(0.9 * (len(v) - 1))], max=v[-1], mean=statistics.fmean(v))


def collect(root: Path):
    field = collections.defaultdict(list)
    build = collections.defaultdict(list)
    for manifest in root.glob("*/shard*/**/dataset_metadata.jsonl"):
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = "%s/%s" % (r.get("topology"), r.get("subclass", "full"))
            if r.get("sample_kind") == "negres":
                key += " (negres)"
            field[key].append(_find(r.get("placement", {}), "seconds"))
            b, e = _find(r, "seconds_build"), _find(r, "seconds_export")
            if b is not None:
                build[key].append(b + (e or 0.0))
    ibeam = []
    for log in root.glob("logs/*_ibeam.log"):
        text = log.read_text()
        m = re.search(r"\{\s*\"counts\".*\}", text, re.S)
        if m:
            j = json.loads(m.group(0))
            if j.get("samples"):
                ibeam.append(dict(split=log.name.split("_")[0], samples=j["samples"], wall_s=j["wall_s"],
                                  per_sample_s=j["wall_s"] / j["samples"]))
    return field, build, ibeam


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path)
    ap.add_argument("--run-log", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    field, build, ibeam = collect(args.root)
    out = dict(root=str(args.root), field_s={k: summarize(v) for k, v in sorted(field.items())},
               freecad_special_s={k: summarize(v) for k, v in sorted(build.items())}, ibeam=ibeam)
    if args.run_log and args.run_log.exists():
        m = re.search(r"generation took (\d+) min", args.run_log.read_text())
        n = sum(len(v) for v in field.values())
        if m:
            out["run"] = dict(wall_min=int(m.group(1)), records=n, wall_s_per_record=60 * int(m.group(1)) / max(n, 1))
    print("%-28s %5s %8s %8s %8s   %s" % ("subclass", "n", "field", "p90", "max", "FreeCAD build+export (median)"))
    for k, s in out["field_s"].items():
        if s:
            fb = out["freecad_special_s"].get(k)
            print("%-28s %5d %8.1f %8.1f %8.1f   %s" % (k, s["n"], s["median"], s["p90"], s["max"],
                                                       "%.1f" % fb["median"] if fb else "-"))
    for b in ibeam:
        print("ibeam %-6s %d samples, %.1f s wall per sample" % (b["split"], b["samples"], b["per_sample_s"]))
    if "run" in out:
        print("whole run: %(records)d records in %(wall_min)d min = %(wall_s_per_record).1f s wall per record" % out["run"])
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
