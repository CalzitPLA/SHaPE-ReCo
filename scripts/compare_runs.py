#!/usr/bin/env python3
"""
Several runs of ONE part side by side (scripts/run_part.py and scripts/peel_part.py directories): the result numbers
in one table, and what each run decided - the type vote at the input, the first add, the rough loop's operations, the
feature phase's end - so that two approaches can be compared step by step, not only by their IoU.

    python3 scripts/compare_runs.py <run dir> [<run dir> ...] [--md out.md] [--json out.json] [--tex-rows out.tex]

A run directory holds record.json (required), steps.jsonl, path.jsonl, program.json. The label of a run is its
directory name. --tex-rows writes the result table's rows for a LaTeX tabular (no header).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def load(d: Path) -> dict:
    rec = json.loads((d / "record.json").read_text())
    steps = []
    if (d / "steps.jsonl").is_file():
        for line in (d / "steps.jsonl").read_text().splitlines():
            try:
                steps.append(json.loads(line))
            except ValueError:
                pass
    path = json.loads((d / "record.json").read_text()).get("path") or {}
    return dict(name=d.name, rec=rec, steps=steps, path=path)


def first(steps, kind):
    return next((s for s in steps if s.get("kind") == kind), None)


def f(x, nd=4):
    return "--" if x is None else f"{x:.{nd}f}"


def summarise(run: dict) -> dict:
    rec, steps = run["rec"], run["steps"]
    c = rec.get("comparison") or {}
    ex, sm = c.get("exact") or {}, c.get("sampled") or {}
    m = rec.get("model") or {}
    r = rec.get("run") or {}
    sec = rec.get("seconds") or {}
    route = (rec.get("settings") or {}).get("route", "cascade")
    out = dict(name=run["name"], route=route, state=rec.get("state"), stage=rec.get("stage"), error=rec.get("error"),
               exact_iou=ex.get("iou"), exact_method=ex.get("method"), exact_missing=ex.get("missing"), exact_invented=ex.get("invented"),
               sampled_iou=sm.get("iou"), missing_share=sm.get("missing_share"), invented_share=sm.get("invented_share"),
               chamfer_over_diagonal=sm.get("chamfer_over_diagonal"),
               operations=m.get("operations"), by_source=m.get("by_source"), by_kind=m.get("by_kind"),
               one_valid_solid=m.get("one_valid_solid"), solids=m.get("solids"), faces=m.get("faces"), volume=m.get("volume"),
               stopped=r.get("stopped"), voxel_iou=r.get("voxel_iou"), explained=r.get("explained"),
               t_reco=sec.get("reconstruction"), t_exact=sec.get("exact difference"), t_sampled=sec.get("sampled comparison"),
               t_all=sec.get("all"), peak_gb=rec.get("peak_gb"), host=rec.get("host"),
               registry=(rec.get("settings") or {}).get("registry"), type_source=(rec.get("settings") or {}).get("type_source"),
               signature=run["path"].get("signature"))
    if route == "cascade":
        inp = first(steps, "input")
        add = first(steps, "add")
        out["input_message"] = (inp or {}).get("message")
        mt = re.search(r"type ([\w/]+) \((\w+) ([\d.]+)\)", (inp or {}).get("message") or "")
        out["type_vote"] = f"{mt.group(1)} ({mt.group(2)} {mt.group(3)})" if mt else None
        out["first_add"] = ((add or {}).get("message") or "")[:400]
        out["rough_ops"] = [s["message"][:120] for s in steps if s.get("kind") == "accepted"]
        out["members"] = sum(1 for s in steps if s.get("kind") == "member")
        out["hollow"] = sum(1 for s in steps if s.get("kind") == "hollow")
        fd = first(steps, "feature_done")
        out["feature_done"] = ((fd or {}).get("message") or "")[:500]
        out["features_applied"] = [s["message"][:160] for s in steps if s.get("kind") == "feature_accepted"]
        out["features_rejected"] = sum(1 for s in steps if s.get("kind") == "feature_rejected")
        done = first(steps, "done")
        out["done"] = ((done or {}).get("message") or "")[:300]
        out["t_rough"] = next((s.get("seconds") for s in steps if s.get("kind") == "feature_residual"), None)
    else:
        out["program"] = r.get("program")
        out["grade"] = r.get("grade")
        out["budget"] = (rec.get("input_facts") or {}).get("predicted_max_ops")
        out["rough_ops"] = [f"{o['op']} {o['kind']} -> {o['explained_after']:.3f}" for o in (r.get("program") or [])]
        out["features_applied"] = []
        out["feature_done"] = None
    return out


def table_md(rows):
    head = ["variant", "state", "exact IoU", "method", "sampled IoU", "missing", "invented", "chamfer/diag", "ops", "by source",
            "features", "one solid", "feature phase ends / stopped", "t reco [s]", "t all [s]", "peak GB"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        feat = len(r.get("features_applied") or [])
        end = (r.get("stopped") or "")
        end = end.split("feature phase: ", 1)[1][:60] if "feature phase: " in end else end[:60]
        lines.append("| " + " | ".join([r["name"], str(r["state"]), f(r["exact_iou"]), str(r["exact_method"] or "--"), f(r["sampled_iou"]),
                                         f(r["missing_share"], 4), f(r["invented_share"], 4), f(r["chamfer_over_diagonal"], 5),
                                         str(r["operations"]), json.dumps(r["by_source"]) if r["by_source"] else "--", str(feat),
                                         str(r["one_valid_solid"]), end, f(r["t_reco"], 0), f(r["t_all"], 0), f(r["peak_gb"], 2)]) + " |")
    return "\n".join(lines)


def tex_rows(rows):
    out = []
    for r in rows:
        feat = len(r.get("features_applied") or [])
        src = r.get("by_source") or {}
        end = (r.get("stopped") or "")
        end = end.split("feature phase: ", 1)[1].split(",")[0].split(":")[0][:30] if "feature phase: " in end else (end[:30] or "--")
        out.append(f"{r['name'].replace('_', chr(92) + '_')} & {f(r['exact_iou'])} & {f(r['sampled_iou'])} & {f(r['missing_share'])} & "
                   f"{f(r['invented_share'])} & {r['operations'] if r['operations'] is not None else '--'} & {src.get('loop', src.get('peel', 0))} & "
                   f"{feat} & {'yes' if r['one_valid_solid'] else 'no'} & {end} & {f(r['t_reco'], 0)} \\\\")
    return "\n".join(out) + "\n"


def details_md(rows):
    out = []
    for r in rows:
        out.append(f"\n## {r['name']} ({r['route']}, {r['host']})\n")
        if r.get("registry") or r.get("type_source"):
            out.append(f"- registry `{Path(r['registry']).name if r.get('registry') else 'default'}`, type source `{r.get('type_source')}`")
        if r.get("type_vote"):
            out.append(f"- type vote at the input: {r['type_vote']}")
        if r.get("first_add"):
            out.append(f"- first add: {r['first_add']}")
        if r.get("members") or r.get("hollow"):
            out.append(f"- member steps {r.get('members')}, hollow steps {r.get('hollow')}")
        out.append(f"- rough loop: {len(r['rough_ops'])} accepted operation(s)")
        for o in r["rough_ops"]:
            out.append(f"    - {o}")
        if r.get("feature_done"):
            out.append(f"- feature phase: {r['feature_done']}")
            out.append(f"- features applied {len(r['features_applied'])}, rejected {r.get('features_rejected')}")
            for o in r["features_applied"][:60]:
                out.append(f"    - {o}")
        if r.get("grade"):
            out.append(f"- route A grade: {json.dumps(r['grade'])[:300]}")
        if r.get("signature"):
            out.append(f"- path signature: {r['signature']}")
        if r.get("error"):
            out.append(f"- ERROR: {r['error']}")
        out.append(f"- model: {r['by_kind']}; solids {r['solids']}, faces {r['faces']}, volume {r['volume']}")
        out.append(f"- exact: IoU {f(r['exact_iou'])} ({r['exact_method']}), missing {r['exact_missing']}, invented {r['exact_invented']}; "
                   f"sampled IoU {f(r['sampled_iou'])}, missing share {f(r['missing_share'])}, invented share {f(r['invented_share'])}")
        out.append(f"- seconds: reconstruction {f(r['t_reco'], 0)}, exact {f(r['t_exact'], 0)}, sampled {f(r['t_sampled'], 0)}, all {f(r['t_all'], 0)}; peak {r['peak_gb']} GB")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--tex-rows", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = [summarise(load(d)) for d in a.dirs if (d / "record.json").is_file()]
    text = table_md(rows) + "\n" + details_md(rows) + "\n"
    print(text)
    if a.md:
        a.md.write_text(text)
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1, default=str) + "\n")
    if a.tex_rows:
        a.tex_rows.write_text(tex_rows(rows))


if __name__ == "__main__":
    main()
