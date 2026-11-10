#!/usr/bin/env python3
"""Merge the starting-shape comparison matrix (scripts/start_matrix_cell.py rows) into tables.

    python3 scripts/start_matrix_report.py [--rows DIR] [--md docs/START_SHAPES_TABLES.md] [--json docs/data/start_matrix.json]

Per part: one line per starting shape (operations, validity, voxel IoU, chamfer, time, the start's coverage/excess) and the
best start. Best = valid first, then voxel IoU (rounded to 0.001), then fewer operations, then chamfer. Overall: how often
each start is best or within 0.005 IoU and +1 operation of the best, and whether "auto" hits the best.
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROWS = Path.home() / ".cache" / "shape_reco" / "prof" / "start_matrix" / "rows"


def iou(r):
    v = (r.get("iou") or {}).get("iou")
    return float(v) if v is not None else float("nan")


def key(r):
    i = iou(r)
    return (not (r.get("rc") == 0 and r.get("valid")), -(round(i, 3) if not math.isnan(i) else -1.0),
            r.get("operations") if r.get("operations") is not None else 999, r.get("chamfer") or float("inf"))


def start_grade(r):
    """coverage/excess of the chosen start, read from the first add's message ('N% of the surface', 'excess +M%')."""
    m = re.search(r"(\d+)% of the surface on it, excess ([+-]?\d+)%", r.get("first_add") or "")
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def near_best(r, b):
    return (r.get("valid") and r.get("rc") == 0 and iou(r) >= iou(b) - 0.005
            and (r.get("operations") or 999) <= (b.get("operations") or 0) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(ROWS))
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(Path(a.rows).glob("*.json"))]
    parts = defaultdict(list)
    for r in rows:
        parts[r["part"]].append(r)
    out, summary = [], {"best": defaultdict(int), "near": defaultdict(int), "auto_hits": 0, "auto_near": 0, "parts": 0}
    for part in sorted(parts):
        rs = sorted(parts[part], key=key)
        best = rs[0]
        out += [f"\n### {part}\n", "| start | ops | valid | IoU | chamfer | s | start cov/excess % | stopped |",
                "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            cov, exc = start_grade(r)
            mark = " **(best)**" if r is best else (" (near)" if near_best(r, best) else "")
            ch = r.get("chamfer")
            out.append(f"| {r['start']}{mark} | {r.get('operations')} | {r.get('valid')} | {iou(r):.4f} | "
                       f"{'' if ch is None else f'{ch:.4f}'} | {r.get('seconds') or 0:.0f} | "
                       f"{'' if cov is None else f'{cov}/{exc:+d}'} | {(r.get('stopped') or r.get('error') or '')[:50]} |")
        summary["parts"] += 1
        summary["best"][best["start"]] += 1
        for r in rs:
            if near_best(r, best):
                summary["near"][r["start"]] += 1
        auto = next((r for r in rs if r["start"] == "auto"), None)
        if auto is not None:
            summary["auto_hits"] += auto is best
            summary["auto_near"] += bool(near_best(auto, best))
    starts = sorted(set(summary["near"]) | set(summary["best"]), key=lambda s: -summary["near"][s])
    head = [f"{len(rows)} cells over {summary['parts']} parts. auto is the best start on {summary['auto_hits']} parts "
            f"and near the best (IoU within 0.005, at most +1 operation) on {summary['auto_near']}.\n",
            "| start | best on | near best on |", "|---|---|---|"]
    head += [f"| {s} | {summary['best'][s]} | {summary['near'][s]} |" for s in starts]
    text = "\n".join(head + out) + "\n"
    print(text)
    if a.md:
        Path(a.md).write_text(text)
    if a.json:
        Path(a.json).write_text(json.dumps(dict(rows=rows, summary={k: dict(v) if isinstance(v, defaultdict) else v
                                                                     for k, v in summary.items()}), indent=1))


if __name__ == "__main__":
    main()
