#!/usr/bin/env python3
"""
Gate G3 for S6: the 60-part add-cut run of this branch (label s6) against the night's `new_fix` run
(docs/data/night_g3_eval.md, same runner and settings). Better / worse as there: the chamfer differs by more than
1.5 x + 0.05 (chamfer in the unit of reco_field.export.export_check). New columns from S6: the exact IoU against the
input itself, the genus of the result against the input's, and the grid pitch over the thinnest wall.

    python3 scripts/s6_g3_compare.py --base ~/Documents/Sandbox/docs/data/night_g3/new_fix \
        --new docs/data/night_g3/s6 --md docs/data/s6_g3_eval.md --json docs/data/s6_g3_eval.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load(d: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def verdict(a, b) -> str:
    if a is None or b is None:
        return "n/a"
    if b > 1.5 * a + 0.05:
        return "worse"
    if a > 1.5 * b + 0.05:
        return "better"
    return "same"


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    base, new = load(args.base), load(args.new)
    rows = []
    for k in sorted(set(base) | set(new)):
        a, b = base.get(k, {}), new.get(k, {})
        r = b.get("result") or {}
        rows.append(dict(part=k.split("__", 1)[-1], ops_base=a.get("n_ops"), ops_s6=b.get("n_ops"),
                         chamfer_base=a.get("chamfer_mm") if a.get("ok") else None,
                         chamfer_s6=b.get("chamfer_mm") if b.get("ok") else None,
                         valid_base=a.get("freecad_valid"), valid_s6=b.get("freecad_valid"),
                         exact_iou=r.get("exact_iou"), genus=r.get("genus"), input_genus=r.get("input_genus"),
                         pitch_over_tau=r.get("grid_pitch_over_tau"), ok_s6=b.get("ok"),
                         error=b.get("error"), seconds_base=a.get("seconds"), seconds_s6=b.get("seconds")))
        rows[-1]["verdict"] = verdict(rows[-1]["chamfer_base"], rows[-1]["chamfer_s6"])
    counts = Counter(r["verdict"] for r in rows)
    valid = (sum(bool(r["valid_base"]) for r in rows), sum(bool(r["valid_s6"]) for r in rows))
    import math
    gen = [r for r in rows if all(isinstance(r[k], (int, float)) and math.isfinite(r[k]) for k in ("genus", "input_genus"))]
    gmatch = sum(round(r["genus"]) == round(r["input_genus"]) for r in gen)
    t = (sum(r["seconds_base"] or 0 for r in rows), sum(r["seconds_s6"] or 0 for r in rows))
    lines = [f"s6 vs new_fix: {dict(counts)}; FreeCAD-valid new_fix {valid[0]}/{len(rows)}, s6 {valid[1]}/{len(rows)}; "
             f"genus of the result = input's on {gmatch}/{len(gen)} parts with both known; CPU time new_fix {t[0]:.0f} s, "
             f"s6 {t[1]:.0f} s.", "",
             "| part | ops fix / s6 | chamfer fix / s6 | verdict | exact IoU | genus (input) | pitch / tau |",
             "|---|---|---|---|---|---|---|"]
    order = {"worse": 0, "better": 1, "n/a": 2, "same": 3}
    for r in sorted(rows, key=lambda r: (order[r["verdict"]], r["part"])):
        g = "–" if r["genus"] is None else f"{fmt(r['genus'])} ({fmt(r['input_genus'])})".replace("nan", "open mesh")
        lines.append(f"| {r['part']} | {fmt(r['ops_base'])} / {fmt(r['ops_s6'])} | {fmt(r['chamfer_base'])} / "
                     f"{fmt(r['chamfer_s6'])} | {r['verdict']} | {fmt(r['exact_iou'], 4)} | {g} | {fmt(r['pitch_over_tau'])} |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        args.md.write_text("# Gate G3 for S6 (this branch) against new_fix\n\n" + text + "\n")
    if args.json:
        args.json.write_text(json.dumps(dict(counts=dict(counts), valid=valid, genus_match=[gmatch, len(gen)],
                                             cpu_seconds=t, rows=rows), indent=1))


if __name__ == "__main__":
    main()
