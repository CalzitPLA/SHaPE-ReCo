#!/usr/bin/env python3
"""
The table of a run of parts (scripts/run_part.py's records) and its three readings (the plan of 2026-09-29):

  robustness  per part: did the run end by itself and give ONE valid solid
  quality     no pass mark: the distribution of the sampled IoU per class of parts - in scope (planes, cylinders,
              cones, spheres, tori only) and out of scope (free-form faces)
  triage      what to look at, in this order: 1. crashes, timeouts, invalid solids, grouped by their text;
              2. exact and sampled IoU disagree (a boolean failure, not a reconstruction failure); 3. out of scope:
              listed, not investigated; 4. in scope below the quality line

    python3 scripts/parts_table.py <run directory> [--line 0.95] [--invented 0.01]

Writes <run>/table.md, <run>/table.csv, <run>/summary.json. Numbers only from the records; a part without a record
(its process was ended before it wrote one) is listed as "no record".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

COLUMNS = ["name", "list", "state", "stage", "in_scope", "faces", "free_form_faces", "operations", "features", "fallbacks",
           "one_valid_solid", "solids", "sampled_iou", "iou_se", "exact_iou", "exact_method", "missing_share",
           "invented_share", "chamfer_over_diagonal", "seconds", "peak_gb", "stopped", "error"]


def row_of(rec: dict, which: str) -> dict:
    facts, run, model = rec.get("input_facts") or {}, rec.get("run") or {}, rec.get("model") or {}
    comp = rec.get("comparison") or {}
    sm, ex = comp.get("sampled") or {}, comp.get("exact") or {}
    src = model.get("by_source") or {}
    return dict(name=rec.get("name"), list=which, state=rec.get("state"), stage=rec.get("stage"),
                in_scope=facts.get("in_scope"), faces=facts.get("faces"), free_form_faces=facts.get("free_form_faces"),
                operations=model.get("operations"), features=src.get("feature", 0),
                fallbacks=src.get("fallback", 0) + src.get("voxel_fallback", 0),
                one_valid_solid=model.get("one_valid_solid"), solids=model.get("solids"),
                sampled_iou=sm.get("iou"), iou_se=sm.get("iou_se"), exact_iou=ex.get("iou"), exact_method=ex.get("method"),
                missing_share=sm.get("missing_share"), invented_share=sm.get("invented_share"),
                chamfer_over_diagonal=sm.get("chamfer_over_diagonal"), seconds=(rec.get("seconds") or {}).get("all"),
                peak_gb=rec.get("peak_gb"), stopped=(run.get("stopped") or "")[:160],
                error=rec.get("error") or ex.get("error") or sm.get("error") or facts.get("error"))


def signature(text: str) -> str:
    """An error's text without what differs from part to part: numbers, paths, names of temporary files."""
    t = re.sub(r"/[^\s'\"]+", "<path>", str(text))
    t = re.sub(r"\d+(\.\d+)?(e[+-]?\d+)?", "#", t)
    return t[:140]


def quantiles(v):
    v = sorted(v)
    if not v:
        return None
    q = statistics.quantiles(v, n=4) if len(v) >= 4 else [v[0], statistics.median(v), v[-1]]
    return dict(n=len(v), min=v[0], q1=q[0], median=statistics.median(v), q3=q[2], max=v[-1])


def read(run: Path):
    lists = {}
    for f in sorted(run.glob("*.txt")):
        if f.name.startswith("machine."):
            continue
        for line in f.read_text().splitlines():
            if line.strip():
                lists.setdefault(line.strip(), f.stem)
    rows = []
    for stem, which in lists.items():
        p = run / stem / "record.json"
        if p.is_file():
            try:
                rows.append(row_of(json.loads(p.read_text()), which))
                continue
            except ValueError:
                pass
        started = (run / stem).is_dir()
        rows.append(dict.fromkeys(COLUMNS) | dict(name=stem, list=which, state="no record" if started else "not run"))
    return rows


def summarise(rows, line: float, invented: float) -> dict:
    ran = [r for r in rows if r["state"] not in ("not run",)]
    robust = [r for r in ran if r["state"] == "ended" and r["one_valid_solid"]]
    no_model = [r for r in ran if r["state"] == "ended" and r["stage"] == "no model"]
    # 1. what did not end, or gave no valid single solid
    broken = defaultdict(list)
    for r in ran:
        if r["state"] == "killed":                                  # since 2026-10-15: TERM from outside, recorded
            broken[f"killed from outside (timeout) in stage: {r['stage']}"].append(r["name"])
        elif r["state"] == "running":                               # older records, or still going on the server
            broken[f"ended from outside (timeout) in stage: {r['stage']}"].append(r["name"])
        elif r["state"] == "no record":
            broken["no record"].append(r["name"])
        elif r["state"] == "crashed":
            broken["crashed: " + signature(r["error"])].append(r["name"])
        elif r["stage"] == "no model":
            broken["ended without any operation: " + signature(r["stopped"])].append(r["name"])
        elif not r["one_valid_solid"]:
            broken[f"model is not one valid solid ({r['solids']} solids)" if r["solids"] != 1 else "model is not valid"].append(r["name"])
    # 2. exact against sampled
    disagree, no_exact = [], []
    for r in ran:
        if r["sampled_iou"] is None:
            continue
        if r["exact_iou"] is None:
            no_exact.append(r["name"])
        elif abs(r["exact_iou"] - r["sampled_iou"]) > 4 * (r["iou_se"] or 0.0) + 0.002:
            disagree.append(dict(name=r["name"], exact=r["exact_iou"], sampled=r["sampled_iou"], method=r["exact_method"]))
    measured = [r for r in ran if r["sampled_iou"] is not None]
    scope = dict(in_scope=[r for r in measured if r["in_scope"]], out_of_scope=[r for r in measured if r["in_scope"] is False])
    quality = {k: dict(iou=quantiles([r["sampled_iou"] for r in v]),
                       invented_share=quantiles([r["invented_share"] for r in v if r["invented_share"] is not None]),
                       missing_share=quantiles([r["missing_share"] for r in v if r["missing_share"] is not None]),
                       above_line=sum(r["sampled_iou"] >= line for r in v),
                       worst=[dict(name=r["name"], iou=r["sampled_iou"]) for r in sorted(v, key=lambda r: r["sampled_iou"])[:10]])
               for k, v in scope.items()}
    below = sorted((r for r in scope["in_scope"] if r["sampled_iou"] < line or (r["invented_share"] or 0) > invented),
                   key=lambda r: r["sampled_iou"])
    secs = [r["seconds"] for r in ran if r["seconds"]]
    return dict(parts=len(rows), ran=len(ran), not_run=len(rows) - len(ran), robust=len(robust), no_model=len(no_model),
                broken={k: v for k, v in sorted(broken.items(), key=lambda kv: -len(kv[1]))},
                exact=dict(methods=dict(Counter(r["exact_method"] or "none" for r in measured)), without=no_exact,
                           disagree=disagree),
                quality=quality, line=line, invented_line=invented,
                below_line=[dict(name=r["name"], iou=r["sampled_iou"], invented_share=r["invented_share"],
                                 missing_share=r["missing_share"], faces=r["faces"], operations=r["operations"],
                                 stopped=r["stopped"]) for r in below],
                seconds=quantiles(secs), seconds_sum=sum(secs),
                peak_gb=quantiles([r["peak_gb"] for r in ran if r["peak_gb"]]))


def fmt(v, spec=".4f"):
    return "" if v is None else (format(v, spec) if isinstance(v, float) else str(v))


def markdown(run: Path, rows, s) -> str:
    out = [f"# Parts run {run.name}", ""]
    for f in sorted(run.glob("machine.*.txt")):
        out += ["```", f.read_text().strip(), "```", ""]
    out += [f"**{s['parts']} parts: {s['ran']} ran, {s['not_run']} not yet. Robust (ended by itself, one valid solid): "
            f"{s['robust']} of {s['ran']}.**", ""]
    out += ["## Quality: sampled IoU by class", "",
            "| class | parts | min | 1st quartile | median | 3rd quartile | max | at or above the line |", "|---|---|---|---|---|---|---|---|"]
    for k, q in s["quality"].items():
        i = q["iou"]
        if i:
            out.append(f"| {k.replace('_', ' ')} | {i['n']} | {i['min']:.4f} | {i['q1']:.4f} | {i['median']:.4f} | {i['q3']:.4f} | "
                       f"{i['max']:.4f} | {q['above_line']} ({s['line']}) |")
    t = s["seconds"]
    if t:
        out += ["", f"Time per part: median {t['median']:.0f} s, quartiles {t['q1']:.0f} to {t['q3']:.0f} s, longest {t['max']:.0f} s; "
                    f"all together {s['seconds_sum'] / 3600:.1f} h of process time."]
    out += ["", "## Triage", "", "### 1. Did not end, or gave no single valid solid", ""]
    if s["broken"]:
        out += ["| parts | what | examples |", "|---|---|---|"]
        out += [f"| {len(v)} | {k.replace('|', '/')} | {', '.join(x[:24] for x in v[:3])} |" for k, v in s["broken"].items()]
    else:
        out.append("None.")
    e = s["exact"]
    out += ["", "### 2. Exact against sampled IoU", "",
            "Methods of the exact difference: " + ", ".join(f"{k} {n}" for k, n in sorted(e["methods"].items(), key=lambda kv: -kv[1])) + ".",
            f"No exact value: {len(e['without'])} parts. Exact and sampled disagree (by more than 4 standard errors + 0.002): "
            f"{len(e['disagree'])} parts."]
    if e["disagree"]:
        out += ["", "| part | exact | sampled | method |", "|---|---|---|---|"]
        out += [f"| {d['name'][:40]} | {d['exact']:.4f} | {d['sampled']:.4f} | {d['method']} |" for d in e["disagree"][:30]]
    o = s["quality"]["out_of_scope"]["iou"]
    out += ["", "### 3. Out of scope (free-form faces)", "",
            f"{o['n'] if o else 0} parts; listed in table.csv, not investigated."]
    out += ["", f"### 4. In scope and below the line (sampled IoU < {s['line']} or invented share > {s['invented_line']})", "",
            f"{len(s['below_line'])} parts, the worst first:", "",
            "| part | sampled IoU | invented share | missing share | faces | operations | stopped |", "|---|---|---|---|---|---|---|"]
    out += [f"| {b['name'][:40]} | {b['iou']:.4f} | {fmt(b['invented_share'], '.3f')} | {fmt(b['missing_share'], '.3f')} | {b['faces']} | "
            f"{b['operations']} | {(b['stopped'] or '').replace('|', '/')[:90]} |" for b in s["below_line"][:60]]
    out += ["", "## All parts", "", "| part | list | state | in scope | faces | operations | one valid solid | sampled IoU | exact IoU | seconds |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["sampled_iou"] is None, r["sampled_iou"] or 0)):
        out.append(f"| {r['name'][:40]} | {r['list']} | {r['state']}{'' if r['stage'] in ('done', None) else ' (' + str(r['stage']) + ')'} | "
                   f"{fmt(r['in_scope'])} | {fmt(r['faces'])} | {fmt(r['operations'])} | {fmt(r['one_valid_solid'])} | "
                   f"{fmt(r['sampled_iou'])} | {fmt(r['exact_iou'])} | {fmt(r['seconds'], '.0f')} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--line", type=float, default=0.95)
    ap.add_argument("--invented", type=float, default=0.01)
    args = ap.parse_args(argv)
    rows = read(args.run)
    s = summarise(rows, args.line, args.invented)
    (args.run / "summary.json").write_text(json.dumps(s, indent=1))
    with open(args.run / "table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (args.run / "table.md").write_text(markdown(args.run, rows, s))
    q = s["quality"]["in_scope"]["iou"]
    print(f"{args.run / 'table.md'}: {s['ran']} of {s['parts']} ran, {s['robust']} robust"
          + (f"; in scope: {q['n']} parts, median sampled IoU {q['median']:.4f}" if q else ""))


if __name__ == "__main__":
    main()
