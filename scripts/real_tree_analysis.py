#!/usr/bin/env python3
"""
What the tree shape does to navigation on REAL trees (scripts/real_tree_study.py): aegir's scan benchmark of every
reconstructed program in every tree shape, read from aegir's own analysis (benchmarks/sandbox_geometry/analyse.py:
table.json with the navigation time per event, same_solid.md with the entry-point check).

    python3 scripts/real_tree_analysis.py <aegir>/benchmarks/realtree/runs [--out docs/reports/real_trees]
                                          [--tolerate <tolerate.tsv>]

Rules:
  reference     per program its own tree ("as-is": the loop's left-deep fold with its composites, node for node)
  counted       a shape only when the entry-point check found it the same solid as the reference (a boundary entry
                more or fewer, events_differing <= 1 of 20000, still counts and is reported), and only programs whose
                reference recorded entries (a part the gun misses measures nothing); with --tolerate, also the shapes
                whose every differing event aegir's graze check found to be a graze or a seam (verdict 'tolerate')
  ratio         navigation(shape) / navigation(reference), navigation = time per event minus the empty baseline's;
                medians over the timing rounds (aegir's analysis), then over programs
Writes <out>/real_trees.json (every number) and <out>/tables.tex / figures (fig_*.pdf), and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

SHAPES = ["as-is", "binary", "balanced", "multi", "as-is+grouped", "binary+grouped", "balanced+grouped", "multi+grouped"]
BINS = [(1, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 10**6)]


def same_solid(md: Path) -> dict:
    """{geometry: (same, events_differing, largest distance)} from aegir's same_solid.md."""
    out = {}
    for line in md.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8 or not cells[3].startswith("realtree/"):
            continue
        m = re.match(r"\d+", cells[5])  # "2" or, since aegir's per-event check, "2 (1889, 14009)"
        differing = int(m.group()) if m else None
        dist = float(cells[6]) if re.match(r"^[0-9.eE+-]+$", cells[6]) else None
        out[cells[3]] = (cells[7] == "True", differing, dist)
    return out


def load(stage: Path):
    table = json.loads((stage / "table.json").read_text())
    same = same_solid(stage / "same_solid.md")
    by_series = defaultdict(dict)
    for r in table["rows"]:
        if not str(r.get("geometry", "")).startswith("realtree/") or r.get("navigation_us") is None:
            continue
        by_series[r["series"]][r["shape"]] = r
    return table, same, by_series


def tolerated_by_check(path: Path | None) -> frozenset:
    """Geometries whose differing events are all grazes or seams (aegir's graze_check.py, verdict 'tolerate')."""
    if path is None:
        return frozenset()
    rows = [line.split("\t") for line in path.read_text().splitlines()[1:] if line.strip()]
    return frozenset(r[0] for r in rows if r[-1].strip() == "tolerate")


def analyse(stage: Path, tolerate: frozenset = frozenset()) -> dict:
    table, same, by_series = load(stage)
    progs, excluded = [], defaultdict(list)
    for series, shapes in sorted(by_series.items()):
        ref = shapes.get("as-is")
        if ref is None:
            excluded["no reference timing"].append(series)
            continue
        if not ref.get("entries"):
            excluded["no entries (the gun misses the part)"].append(series)
            continue
        if ref["navigation_us"] <= 0:
            excluded["reference navigation <= 0 (within the noise of the baseline)"].append(series)
            continue
        row = dict(series=series, leaves=ref["leaves"], depth=ref["depth"], nav_ref=ref["navigation_us"],
                   entries_per_event=ref.get("entries_per_event"), source=ref.get("source"), shapes={})
        for shape, r in shapes.items():
            ok, differing, dist = same.get(r["geometry"], (shape == "as-is", 0, 0.0))
            verified = not ok and r["geometry"] in tolerate
            tolerated = ok or verified or (differing is not None and differing <= 1 and (dist is None or dist < 1e-3))
            row["shapes"][shape] = dict(nav=r["navigation_us"], depth=r["depth"], same=ok, tolerated=tolerated,
                                        graze_verified=verified,
                                        differing=differing, distance=dist,
                                        ratio=r["navigation_us"] / ref["navigation_us"] if tolerated else None)
            if not tolerated:
                excluded[f"{shape}: not the same solid"].append(series)
        progs.append(row)

    def med(xs):
        return statistics.median(xs) if xs else None

    per_shape = {}
    for shape in SHAPES:
        ratios = [p["shapes"][shape]["ratio"] for p in progs if shape in p["shapes"] and p["shapes"][shape]["ratio"] is not None]
        if not ratios:
            continue
        q = np.percentile(ratios, [25, 75]).tolist()
        per_shape[shape] = dict(n=len(ratios), median=med(ratios), q25=q[0], q75=q[1],
                                faster=sum(r < 0.95 for r in ratios) / len(ratios),
                                slower=sum(r > 1.05 for r in ratios) / len(ratios))
    per_bin = []
    for lo, hi in BINS:
        ps = [p for p in progs if lo <= p["leaves"] <= hi]
        entry = dict(leaves=f"{lo}-{hi}" if hi < 10**6 else f">{lo - 1}", n=len(ps),
                     nav_ref=med([p["nav_ref"] for p in ps]), depth=med([p["depth"] for p in ps]))
        for shape in ("balanced", "multi", "balanced+grouped", "multi+grouped"):
            rs = [p["shapes"][shape]["ratio"] for p in ps if shape in p["shapes"] and p["shapes"][shape]["ratio"] is not None]
            entry[shape] = med(rs)
        best = [min((s["ratio"] for s in p["shapes"].values() if s["ratio"] is not None), default=None) for p in ps]
        entry["best"] = med([b for b in best if b is not None])
        per_bin.append(entry)
    best_shape = defaultdict(int)
    for p in progs:
        cand = {k: v["ratio"] for k, v in p["shapes"].items() if v["ratio"] is not None}
        best_shape[min(cand, key=cand.get)] += 1
    leaves = np.array([p["leaves"] for p in progs], float)
    depth = np.array([p["depth"] for p in progs], float)
    nav = np.array([p["nav_ref"] for p in progs], float)

    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else None

    biggest = sorted(progs, key=lambda p: -p["leaves"])[:12]
    rounds = defaultdict(int)  # timing rounds per geometry: fewer than three while a benchmark still runs
    for shapes in by_series.values():
        for r in shapes.values():
            rounds[r.get("rounds")] += 1
    return dict(baseline_us=table.get("baseline_per_event_us"), programs=len(progs), rounds=dict(rounds),
                excluded={k: sorted(set(v)) for k, v in excluded.items()}, per_shape=per_shape, per_bin=per_bin,
                best_shape=dict(best_shape), spearman_nav_depth=spearman(depth, nav),
                spearman_nav_leaves=spearman(leaves, nav), biggest=biggest, rows=progs)


def figures(res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    colours = {"balanced": "#2a78d6", "multi": "#eb6834", "balanced+grouped": "#2e8b57", "multi+grouped": "#8e44ad",
               "as-is+grouped": "#777777", "binary": "#c8322d", "binary+grouped": "#b8860b"}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for shape, c in colours.items():
        pts = [(p["leaves"], p["shapes"][shape]["ratio"]) for p in rows if shape in p["shapes"] and p["shapes"][shape]["ratio"]]
        if pts:
            x, y = zip(*pts)
            ax.scatter(x, y, s=8, alpha=0.5, color=c, label=shape)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("leaves of the program (primitives)"); ax.set_ylabel("navigation time / as-is tree")
    ax.legend(fontsize=7, markerscale=2, ncol=2); ax.grid(alpha=0.3, which="both", lw=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_ratio_vs_leaves.pdf"); fig.savefig(out / "fig_ratio_vs_leaves.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sc = ax.scatter([p["leaves"] for p in rows], [p["nav_ref"] for p in rows], c=[p["depth"] for p in rows], s=10,
                    cmap="viridis", norm=matplotlib.colors.LogNorm())
    fig.colorbar(sc, ax=ax, label="depth of the as-is tree")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("leaves"); ax.set_ylabel("navigation per event, as-is tree (µs)")
    ax.grid(alpha=0.3, which="both", lw=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_nav_vs_leaves.pdf"); fig.savefig(out / "fig_nav_vs_leaves.png", dpi=160)
    plt.close(fig)
    shapes = [s for s in ("binary", "balanced", "multi", "as-is+grouped", "balanced+grouped", "multi+grouped")
              if s in res["per_shape"]]
    data = [[p["shapes"][s]["ratio"] for p in rows if s in p["shapes"] and p["shapes"][s]["ratio"]] for s in shapes]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.boxplot(data, showfliers=False); ax.set_xticks(range(1, len(shapes) + 1), shapes)  # matplotlib < 3.9 too
    ax.axhline(1.0, color="k", lw=0.8); ax.set_yscale("log"); ax.set_ylabel("navigation / as-is")
    plt.setp(ax.get_xticklabels(), rotation=20, fontsize=8); ax.grid(alpha=0.3, axis="y", lw=0.3)
    fig.tight_layout(); fig.savefig(out / "fig_ratio_by_shape.pdf"); fig.savefig(out / "fig_ratio_by_shape.png", dpi=160)
    plt.close(fig)


def tex(res: dict, out: Path):
    """numbers.tex (macros \\rt...) and the tables, generated: the report's text uses only these."""
    def num(v, fmt="{:.3f}"):
        return "--" if v is None else fmt.format(v)
    ps = res["per_shape"]
    M = {"Programs": str(res["programs"]), "Baseline": num(res["baseline_us"], "{:.2f}"),
         "RhoDepth": num(res["spearman_nav_depth"], "{:.2f}"), "RhoLeaves": num(res["spearman_nav_leaves"], "{:.2f}"),
         "MaxLeaves": str(max(p["leaves"] for p in res["rows"])),
         "Excluded": str(sum(len(v) for k, v in res["excluded"].items() if "same solid" not in k)),
         "Geometries": str(sum(v for k, v in res["rounds"].items() if k)),
         "GrazeVerified": str(sum(v.get("graze_verified", False) for p in res["rows"] for v in p["shapes"].values()))}
    M.update({f"Rounds{w}": str(res["rounds"].get(n, 0)) for n, w in ((1, "One"), (2, "Two"), (3, "Three"))})
    for key, shape in (("Bal", "balanced"), ("Multi", "multi"), ("BalGrp", "balanced+grouped"), ("MultiGrp", "multi+grouped"),
                       ("AsisGrp", "as-is+grouped"), ("Binary", "binary")):
        d = ps.get(shape, {})
        M[f"{key}Median"] = num(d.get("median"), "{:.2f}")
        M[f"{key}Faster"] = num(100 * d["faster"], "{:.0f}") if d else "--"
        M[f"{key}Slower"] = num(100 * d["slower"], "{:.0f}") if d else "--"
        M[f"{key}N"] = str(d.get("n", 0))
    big = [b for b in res["per_bin"] if b["n"] >= 5][-1]  # the largest bin with a median worth the name
    M["BigBin"] = big["leaves"]; M["BigBest"] = num(big["best"], "{:.2f}")
    M["BigSpeedup"] = num(1 / big["best"], "{:.1f}") if big["best"] else "--"
    # grouped cuts with the tools' union left as the program's chain: the program that loses the most time per event
    grp = [p for p in res["rows"] if (p["shapes"].get("as-is+grouped") or {}).get("ratio")]
    if grp:
        w = max(grp, key=lambda p: p["shapes"]["as-is+grouped"]["nav"] - p["nav_ref"])
        sh = w["shapes"]
        bal = sh.get("balanced+grouped")
        M.update(WorstGrpSeries=w["series"].replace("_", "\\_"), WorstGrpLeaves=str(w["leaves"]),
                 WorstGrpDepth=str(sh["as-is+grouped"]["depth"]), WorstGrpDepthRef=str(w["depth"]),
                 WorstGrpNavRef=num(w["nav_ref"], "{:.1f}"),
                 WorstGrpNav=num(sh["as-is+grouped"]["nav"], "{:,.0f}").replace(",", "\\,"),
                 WorstGrpFactor=num(sh["as-is+grouped"]["ratio"], "{:.0f}"),
                 WorstGrpBalGrp=num(bal["nav"] if bal and bal["tolerated"] else None, "{:.1f}"))
    lines = [f"\\newcommand{{\\rt{k}}}{{{v}}}" for k, v in M.items()]
    (out / "numbers.tex").write_text("\n".join(lines) + "\n")
    T = ["\\begin{tabular}{lrrrrr}", "\\toprule",
         "tree shape & programs & median ratio & quartiles & faster & slower \\\\", "\\midrule"]
    for shape, d in ps.items():
        if shape == "as-is":
            continue
        T.append(f"{shape} & {d['n']} & {d['median']:.2f} & {d['q25']:.2f}--{d['q75']:.2f} & "
                 f"{100 * d['faster']:.0f}\\,\\% & {100 * d['slower']:.0f}\\,\\% \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_shapes.tex").write_text("\n".join(T) + "\n")
    T = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         "leaves & programs & nav.\\ as-is ($\\mu$s) & depth & balanced & multi & bal.+grp. & multi+grp. & best \\\\",
         "\\midrule"]
    for b in res["per_bin"]:
        if b["n"]:
            T.append(" & ".join([b["leaves"], str(b["n"]), num(b["nav_ref"], "{:.2f}"), num(b["depth"], "{:.0f}")] +
                                [num(b[k], "{:.2f}") for k in ("balanced", "multi", "balanced+grouped", "multi+grouped", "best")])
                     + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_bins.tex").write_text("\n".join(T) + "\n")
    T = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         "program & leaves & depth & as-is & balanced & multi & best grouped \\\\", "\\midrule"]
    for p in res["biggest"]:
        sh = p["shapes"]
        g = min((v["nav"] for k, v in sh.items() if k.endswith("grouped") and v["tolerated"]), default=None)
        T.append(" & ".join([p["series"].replace("_", "\\_"), str(p["leaves"]), str(p["depth"])] +
                            [num(sh[k]["nav"] if k in sh and sh[k]["tolerated"] else None, "{:.2f}")
                             for k in ("as-is", "balanced", "multi")] + [num(g, "{:.2f}")]) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "table_biggest.tex").write_text("\n".join(T) + "\n")


def summary(res: dict) -> str:
    L = [f"{res['programs']} programs counted; baseline {res['baseline_us']:.3f} us per event"]
    for k, v in res["excluded"].items():
        L.append(f"  excluded - {k}: {len(v)}")
    L.append("shape              n    median ratio  (q25-q75)     faster  slower")
    for s, d in res["per_shape"].items():
        L.append(f"  {s:16s} {d['n']:4d}  {d['median']:8.3f}   ({d['q25']:.3f}-{d['q75']:.3f})   {d['faster']:5.0%}  {d['slower']:5.0%}")
    L.append("leaves      n   nav as-is us  depth  balanced  multi  bal+grp  multi+grp  best")
    f = lambda v: f"{v:6.3f}" if v is not None else "     -"
    for b in res["per_bin"]:
        L.append(f"  {b['leaves']:9s} {b['n']:4d}  {f(b['nav_ref'])}  {f(b['depth'])}  {f(b['balanced'])} {f(b['multi'])} "
                 f"{f(b['balanced+grouped'])}  {f(b['multi+grouped'])}  {f(b['best'])}")
    L.append(f"best shape per program: {res['best_shape']}")
    L.append(f"Spearman(navigation of the as-is tree, depth) = {res['spearman_nav_depth']:.3f}, "
             f"(.., leaves) = {res['spearman_nav_leaves']:.3f}")
    L.append("the biggest trees (navigation per event in us by shape):")
    for p in res["biggest"]:
        L.append(f"  {p['series']:32s} leaves {p['leaves']:4d} depth {p['depth']:4d}  " + ", ".join(
            f"{s} {v['nav']:.2f}" + ("" if v['tolerated'] else "(!)") for s, v in sorted(p["shapes"].items())))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "reports" / "real_trees")
    ap.add_argument("--tolerate", type=Path, help="aegir graze-check verdicts (tsv: geometry ... verdict)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = analyse(a.stage, tolerated_by_check(a.tolerate))
    res["graze_verified"] = sum(v.get("graze_verified", False) for p in res["rows"] for v in p["shapes"].values())
    (a.out / "real_trees.json").write_text(json.dumps(res, indent=1, default=float))
    figures(res, a.out)
    tex(res, a.out)
    text = summary(res)
    (a.out / "summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
