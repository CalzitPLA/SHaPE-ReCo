#!/usr/bin/env python3
"""
The detail behind scripts/real_tree_analysis.py, from the same stage: which benchmark parts and which primitives carry
the navigation cost of a reconstructed program, and which property of a program decides whether a tree shape helps or
hurts it.

    python3 scripts/real_tree_detail.py <aegir>/benchmarks/realtree/runs [--out docs/reports/real_trees]

Families    the program's source: the frame (32 and 40 adds), the liner's stored runs, the review frames, the ABC known
            parts without and with sub-parts, the ABC pilots (both)
Primitives  log(navigation per hit line of the as-is tree) = b0 + b1 log(depth + 1) + b2 log(leaves) + sum_k beta_k f_k,
            f_k the share of the program's leaves that are primitive k, boxes the reference; exp(beta_k) compares a
            program made of k only with one made of boxes only, at equal depth and size. 95 % intervals from 2000
            bootstrap resamples of the programs. Fitted on all programs and on the ABC programs alone (one gun, one
            placement rule).
Shapes      per shape the ratio (navigation / as-is; only shapes that are the same solid, as in real_tree_analysis.py)
            against the share of cuts among the as-is tree's boolean nodes, the change of depth the shape makes, and
            whether the program holds a torus
Noise       navigation is the time per event minus the empty baseline's; the baseline's own range over the rounds and
            the median range of all geometries give the noise floor. Every median is repeated on the programs whose
            as-is navigation is at least CLEAN_US (five times the median range), and each primitive is also compared
            directly: a program holding it against the programs without it of the same depth class and gun (ABC or not)
Writes <out>/detail.json, <out>/detail_*.tex (tables), <out>/numbers_detail.tex (macros \\rd...) and
<out>/fig_detail_*.pdf/png, and prints a summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import real_tree_analysis as rta  # noqa: E402

SHAPES = ["balanced", "multi", "as-is+grouped", "balanced+grouped", "multi+grouped", "binary"]
FAMILIES = ["frame", "liner", "review frames", "ABC known", "ABC known, sub-parts", "ABC pilot"]
PRIMS = ["cylinder", "cone", "sphere", "torus", "halfspace"]  # box is the reference
CUT_BINS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
DEPTH_BINS = [(0, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 10**6)]
CLEAN_US = 1.0
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#d9d8d2"
BLUE, ORANGE, RED, MID = "#2a78d6", "#eb6834", "#e34948", "#f0efec"


def family(series: str) -> str:
    for prefix, name in (("frame_", "frame"), ("liner_", "liner"), ("review_", "review frames"),
                         ("known_subparts_", "ABC known, sub-parts"), ("known_", "ABC known"), ("pilot2", "ABC pilot")):
        if series.startswith(prefix):
            return name
    return "other"


def prim_class(subtype: str) -> str:
    """'cylinder/sector' -> 'cylinder'; sectors and full solids count as one primitive."""
    return subtype.split("/")[0]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def programs(stage: Path, tolerate: frozenset = frozenset()) -> list:
    """real_tree_analysis's counted programs, each with its as-is row's primitives, node kinds and per-line cost."""
    res = rta.analyse(stage, tolerate)
    table, _, by_series = rta.load(stage)
    out = []
    for p in res["rows"]:
        ref = by_series[p["series"]]["as-is"]
        kinds = ref.get("nodes_by_kind") or {}
        nodes = sum(kinds.values())
        prims = Counter()
        for sub, n in (ref.get("subtypes") or {}).items():
            prims[prim_class(sub)] += n
        out.append(dict(p, family=family(p["series"]), prims=dict(prims),
                        cut_share=kinds.get("Subtraction", 0) / nodes if nodes else 0.0,
                        intersections=kinds.get("Intersection", 0), imbalance=ref.get("imbalance"),
                        nav_line=ref.get("navigation_per_hit_line_us"), nav_entry=ref.get("navigation_per_entry_us")))
    return out


def ratios(ps, shape):
    return [p["shapes"][shape]["ratio"] for p in ps if shape in p["shapes"] and p["shapes"][shape]["ratio"] is not None]


def by_family(ps) -> list:
    rows = []
    for fam in FAMILIES:
        fp = [p for p in ps if p["family"] == fam]
        if not fp:
            continue
        best = Counter(min(((k, v["ratio"]) for k, v in p["shapes"].items() if v["ratio"] is not None),
                           key=lambda kv: kv[1])[0] for p in fp)
        rows.append(dict(family=fam, n=len(fp), leaves=med([p["leaves"] for p in fp]), depth=med([p["depth"] for p in fp]),
                         cut_share=med([p["cut_share"] for p in fp]), nav=med([p["nav_ref"] for p in fp]),
                         nav_entry=med([p["nav_entry"] for p in fp]),
                         shapes={s: dict(median=med(ratios(fp, s)), n=len(ratios(fp, s))) for s in SHAPES},
                         best=best.most_common(1)[0][0], best_share=best.most_common(1)[0][1] / len(fp)))
    return rows


def benchmark_parts(ps) -> list:
    """Every program of the frame, the liner and the review frames: navigation per event by shape."""
    keep = [p for p in ps if p["family"] in ("frame", "liner", "review frames")]
    return sorted(keep, key=lambda p: (FAMILIES.index(p["family"]), p["series"]))


def fit(ps, n_boot=2000, seed=0) -> dict:
    """log(nav per hit line) on log(depth + 1) (the levels: one primitive alone has depth 0), log leaves and the
    primitive shares; exp(beta) with bootstrap intervals."""
    ps = [p for p in ps if p["nav_line"] and p["nav_line"] > 0]
    prims = [k for k in PRIMS if sum(p["prims"].get(k, 0) > 0 for p in ps) >= 5]
    X = np.array([[1.0, np.log(p["depth"] + 1), np.log(p["leaves"])] +
                  [p["prims"].get(k, 0) / p["leaves"] for k in prims] for p in ps])
    y = np.log([p["nav_line"] for p in ps])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - np.sum((y - X @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0])
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    terms = ["intercept", "log depth", "log leaves"] + prims
    coef = {t: dict(beta=float(b), lo=float(l), hi=float(h)) for t, b, l, h in zip(terms, beta, lo, hi)}
    with_k = {k: sum(p["prims"].get(k, 0) > 0 for p in ps) for k in prims}
    return dict(n=len(ps), r2=float(r2), coef=coef, prims=prims, programs_with=with_k)


def by_cut_share(ps) -> list:
    rows = []
    for lo, hi in CUT_BINS:
        bp = [p for p in ps if lo <= p["cut_share"] < hi]
        rows.append(dict(bin=f"{lo:.2f}-{min(hi, 1.0):.2f}", n=len(bp),
                         shapes={s: dict(median=med(ratios(bp, s)), n=len(ratios(bp, s))) for s in SHAPES}))
    return rows


def depth_effect(ps) -> dict:
    """Per shape: how the ratio follows the change of depth the shape makes, and what a torus does to it."""
    out, pooled_x, pooled_y = {}, [], []
    for s in SHAPES:
        pts = [(p["shapes"][s]["depth"] / p["depth"], p["shapes"][s]["ratio"], p) for p in ps
               if s in p["shapes"] and p["shapes"][s]["ratio"] is not None and p["depth"]]
        if not pts:
            continue
        dr = np.array([d for d, _, _ in pts]); rr = np.array([r for _, r, _ in pts])  # ranks: ratios <= 0 stay in
        pooled_x += list(dr); pooled_y += list(rr)
        shallower = [r for d, r, _ in pts if d <= 0.5]
        same = [r for d, r, _ in pts if d > 0.5]
        tor = [r for _, r, p in pts if p["prims"].get("torus")]
        notor = [r for _, r, p in pts if not p["prims"].get("torus")]
        out[s] = dict(n=len(pts), rho_depth=spearman(dr, rr), depth_ratio=med(list(dr)), nonpositive=int(np.sum(rr <= 0)),
                      half_depth=dict(n=len(shallower), median=med(shallower)), rest=dict(n=len(same), median=med(same)),
                      torus=dict(n=len(tor), median=med(tor)), no_torus=dict(n=len(notor), median=med(notor)),
                      rho_cut=spearman([p["cut_share"] for _, _, p in pts], rr),
                      rho_imbalance=spearman([p["imbalance"] or 0 for _, _, p in pts], rr),
                      points=[(float(d), float(r)) for d, r, _ in pts])
    return dict(shapes=out, rho_pooled=spearman(pooled_x, pooled_y))


def noise(stage: Path) -> dict:
    """The baseline's range of time per event over its rounds, and the median range over all geometries (us)."""
    table = json.loads((stage / "table.json").read_text())
    base = next((r for r in table["rows"] if r["geometry"] == table.get("baseline")), {})
    spans = [r["per_event_max"] - r["per_event_min"] for r in table["rows"]
             if str(r.get("geometry", "")).startswith("realtree/") and r.get("per_event_max") is not None]
    return dict(baseline_range=(base["per_event_max"] - base["per_event_min"]) if base else None,
                median_range=med(spans), rows_nonpositive=sum(1 for r in table["rows"] if str(r.get("geometry", ""))
                                                             .startswith("realtree/") and (r.get("navigation_us") or 1) <= 0))


def clean(ps) -> dict:
    """The per-shape medians again, on the programs whose own navigation is at least CLEAN_US."""
    cp = [p for p in ps if p["nav_ref"] >= CLEAN_US]
    return dict(n=len(cp), families=dict(Counter(p["family"] for p in cp)),
                shapes={s: dict(all=med(ratios(ps, s)), n_all=len(ratios(ps, s)), clean=med(ratios(cp, s)),
                                n_clean=len(ratios(cp, s))) for s in SHAPES},
                cut_share=by_cut_share(cp))


def matched_prims(ps, min_nav=0.0) -> dict:
    """Per primitive: each program holding it against the median per-line cost of the programs without it in the same
    depth class and gun group (ABC or not); the median and quartiles of these factors."""
    ps = [p for p in ps if p["nav_line"] and p["nav_line"] > 0 and p["nav_ref"] >= min_nav]
    dclass = lambda p: next(i for i, (lo, hi) in enumerate(DEPTH_BINS) if lo <= p["depth"] <= hi)
    group = lambda p: (dclass(p), p["family"].startswith("ABC"))
    out = {}
    for k in ["box"] + PRIMS:
        f, top = [], (None, None)
        for p in ps:
            if not p["prims"].get(k):
                continue
            ref = [q["nav_line"] for q in ps if group(q) == group(p) and not q["prims"].get(k)]
            if len(ref) >= 3:
                f.append(p["nav_line"] / statistics.median(ref))
                if top[0] is None or f[-1] > top[0]:
                    top = (f[-1], p)
        if f:
            q = np.percentile(f, [25, 75]).tolist()
            out[k] = dict(n=len(f), median=med(f), q25=q[0], q75=q[1], max=top[0], max_series=top[1]["series"],
                          max_leaves=top[1]["leaves"])
    return out


def heat(ax, M, rows, cols, title):
    """Diverging heat table of median ratios: blue faster, red slower, gray no change; the value in every cell."""
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ratio", [BLUE, MID, RED])
    L = np.log2(np.clip(np.where(np.isnan(M), 1.0, M), 1e-3, None))
    ax.imshow(np.clip(L, -2, 2), cmap=cmap, vmin=-2, vmax=2, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=8)
            else:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(L[i, j]) > 1.1 else INK)
    ax.set_xticks(range(len(cols)), cols, rotation=25, ha="right", fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=INK)
    ax.set_xticks(np.arange(-0.5, len(cols)), minor=True); ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=9, color=INK, loc="left")


def figures(res: dict, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})

    fam = res["families"]
    M = np.array([[r["shapes"][s]["median"] if r["shapes"][s]["median"] is not None else np.nan for s in SHAPES] for r in fam])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    heat(ax, M, [f"{r['family']} ({r['n']})" for r in fam], SHAPES, "median navigation / as-is tree, per benchmark family")
    fig.tight_layout(); fig.savefig(out / "fig_detail_family.pdf"); fig.savefig(out / "fig_detail_family.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    for ax, cb, title in ((axs[0], res["cut_share"], "median navigation / as-is tree, by share of cuts: all programs"),
                          (axs[1], res["clean"]["cut_share"],
                           f"the same, programs with as-is navigation >= {CLEAN_US:g} us")):
        M = np.array([[r["shapes"][s]["median"] if r["shapes"][s]["median"] is not None else np.nan for s in SHAPES]
                      for r in cb])
        heat(ax, M, [f"cuts {r['bin']} ({r['n']})" for r in cb], SHAPES, title)
    fig.tight_layout(); fig.savefig(out / "fig_detail_cutshare.pdf"); fig.savefig(out / "fig_detail_cutshare.png", dpi=160)
    plt.close(fig)

    kinds = ["box"] + PRIMS
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for off, (key, colour, label) in zip((-0.13, 0.13), (("all", BLUE, "all programs"),
                                                         ("clean", ORANGE, f"as-is navigation >= {CLEAN_US:g} us"))):
        m = res["matched"][key]
        for i, k in enumerate([k for k in kinds if k in m]):
            d, y = m[k], kinds.index(k) + off
            ax.plot([d["q25"], d["q75"]], [y, y], color=colour, lw=2, solid_capstyle="round")
            ax.plot(d["median"], y, "o", color=colour, ms=7, mec="white", mew=1.5, label=label if i == 0 else None)
    ax.axvline(1.0, color=INK, lw=0.8)
    ax.set_xscale("log"); ax.set_yticks(range(len(kinds)), [f"{k}" for k in kinds], fontsize=8, color=INK)
    lo, hi = ax.get_xlim()
    ticks = [t for t in (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8) if lo <= t <= hi]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks], fontsize=8); ax.minorticks_off()
    ax.invert_yaxis()
    ax.set_xlabel("navigation per hit line / programs without it, same depth class and gun (median, quartiles)",
                  fontsize=8)
    ax.grid(axis="x", color=GRID, lw=0.5, which="both"); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_detail_primitives.pdf"); fig.savefig(out / "fig_detail_primitives.png", dpi=160)
    plt.close(fig)

    de = res["depth"]["shapes"]
    fig, axs = plt.subplots(2, 3, figsize=(6.4, 4.4), sharex=True, sharey=True)
    for ax, s in zip(axs.flat, SHAPES):
        if s in de:
            x, y = zip(*[(d, r) for d, r in de[s]["points"] if r > 0])
            ax.scatter(x, y, s=9, color=BLUE, alpha=0.45, lw=0)
            rho = de[s]["rho_depth"]
            ax.set_title(f"{s}\nn={de[s]['n']}, ρ={rho:.2f}" if rho is not None else s, fontsize=8, color=INK, loc="left")
        ax.axhline(1, color=INK, lw=0.6); ax.axvline(1, color=INK, lw=0.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(color=GRID, lw=0.4, which="major")
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("depth / as-is depth", fontsize=8)
    for ax in axs[:, 0]:
        ax.set_ylabel("navigation / as-is", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_detail_depth.pdf"); fig.savefig(out / "fig_detail_depth.png", dpi=160)
    plt.close(fig)


def tex(res: dict, out: Path):
    def num(v, fmt="{:.2f}"):
        return "--" if v is None else fmt.format(v)
    short = {"balanced": "bal.", "multi": "multi", "as-is+grouped": "as-is+grp.", "balanced+grouped": "bal.+grp.",
             "multi+grouped": "multi+grp.", "binary": "binary"}
    T = ["\\begin{tabular}{lrrrrrr" + "r" * len(SHAPES) + "}", "\\toprule",
         "family & progr. & leaves & depth & cuts & nav.\\ ($\\mu$s) & per entry & " +
         " & ".join(short[s] for s in SHAPES) + " \\\\", "\\midrule"]
    for r in res["families"]:
        T.append(" & ".join([r["family"], str(r["n"]), num(r["leaves"], "{:.0f}"), num(r["depth"], "{:.0f}"),
                             num(r["cut_share"]), num(r["nav"]), num(r["nav_entry"])] +
                            [num(r["shapes"][s]["median"]) for s in SHAPES]) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "detail_families.tex").write_text("\n".join(T) + "\n")

    cols = ["as-is"] + SHAPES
    T = ["\\begin{tabular}{lrrr" + "r" * len(cols) + "}", "\\toprule",
         "program & leaves & depth & cuts & " + " & ".join(["as-is"] + [short[s] for s in SHAPES]) + " \\\\", "\\midrule"]
    for p in res["benchmarks"]:
        sh = p["shapes"]
        cells = []
        for s in cols:
            v = sh.get(s)
            cells.append("--" if v is None else (f"{v['nav']:.2f}" if v["tolerated"] else f"({v['nav']:.2f})"))
        T.append(" & ".join([p["series"].replace("_", "\\_"), str(p["leaves"]), str(p["depth"]), num(p["cut_share"])]
                            + cells) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "detail_benchmarks.tex").write_text("\n".join(T) + "\n")

    fa, fb = res["primitives"]["all"], res["primitives"]["abc"]
    T = ["\\begin{tabular}{lrrrr}", "\\toprule",
         "term & factor, all & 95\\,\\% interval & factor, ABC & 95\\,\\% interval \\\\", "\\midrule"]
    for t in ["log depth", "log leaves"] + PRIMS:
        a, b = fa["coef"].get(t), fb["coef"].get(t)
        f = (lambda c: (f"{c['beta']:.2f}", f"{c['lo']:.2f}--{c['hi']:.2f}")) if t.startswith("log") else \
            (lambda c: (f"{np.exp(c['beta']):.2f}", f"{np.exp(c['lo']):.2f}--{np.exp(c['hi']):.2f}"))
        ca = f(a) if a else ("--", "--"); cb_ = f(b) if b else ("--", "--")
        name = t if t.startswith("log") else f"{t} ({fa['programs_with'].get(t, 0)})"
        T.append(" & ".join([name, *ca, *cb_]) + " \\\\")
    T += ["\\midrule", f"programs, $R^2$ & {fa['n']} & {fa['r2']:.2f} & {fb['n']} & {fb['r2']:.2f} \\\\",
          "\\bottomrule", "\\end{tabular}"]
    (out / "detail_primitives.tex").write_text("\n".join(T) + "\n")

    de = res["depth"]["shapes"]
    T = ["\\begin{tabular}{lrrrrrrrr}", "\\toprule",
         "shape & programs & depth ratio & $\\rho$ depth & depth $\\le\\frac12$ & otherwise & $\\rho$ cuts & "
         "with torus & without \\\\", "\\midrule"]
    for s in SHAPES:
        if s not in de:
            continue
        d = de[s]
        T.append(" & ".join([s, str(d["n"]), num(d["depth_ratio"]), num(d["rho_depth"]),
                             f"{num(d['half_depth']['median'])} ({d['half_depth']['n']})",
                             f"{num(d['rest']['median'])} ({d['rest']['n']})", num(d["rho_cut"]),
                             f"{num(d['torus']['median'])} ({d['torus']['n']})",
                             f"{num(d['no_torus']['median'])} ({d['no_torus']['n']})"]) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "detail_depth.tex").write_text("\n".join(T) + "\n")

    cl = res["clean"]
    T = ["\\begin{tabular}{lrrrr}", "\\toprule",
         f"shape & programs & median ratio & programs, nav.\\ $\\ge {CLEAN_US:g}\\,\\mu$s & median ratio \\\\", "\\midrule"]
    for sh in SHAPES:
        d = cl["shapes"][sh]
        T.append(f"{sh} & {d['n_all']} & {num(d['all'])} & {d['n_clean']} & {num(d['clean'])} \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "detail_clean.tex").write_text("\n".join(T) + "\n")
    ma, mc = res["matched"]["all"], res["matched"]["clean"]
    T = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
         f"primitive & programs & factor & quartiles & programs, $\\ge {CLEAN_US:g}\\,\\mu$s & factor & quartiles \\\\",
         "\\midrule"]
    for k in ["box"] + PRIMS:
        a, c = ma.get(k), mc.get(k)
        cell = lambda d: [str(d["n"]), num(d["median"]), f"{d['q25']:.2f}--{d['q75']:.2f}"] if d else ["0", "--", "--"]
        T.append(" & ".join([k] + cell(a) + cell(c)) + " \\\\")
    T += ["\\bottomrule", "\\end{tabular}"]
    (out / "detail_matched.tex").write_text("\n".join(T) + "\n")

    fam = {r["family"]: r for r in res["families"]}
    nz = res["noise"]
    M = {"RhoPooled": num(res["depth"]["rho_pooled"]), "FitN": str(fa["n"]), "FitRsq": num(fa["r2"]),
         "FitDepth": num(fa["coef"]["log depth"]["beta"]), "FitLeaves": num(fa["coef"]["log leaves"]["beta"]),
         "FitNAbc": str(fb["n"]), "FitRsqAbc": num(fb["r2"]), "NoiseBase": num(nz["baseline_range"]),
         "NoiseMedian": num(nz["median_range"]), "NoiseNonpos": str(nz["rows_nonpositive"]), "CleanUs": f"{CLEAN_US:g}",
         "CleanN": str(cl["n"])}
    for sh, sk in (("balanced", "Bal"), ("multi", "Multi"), ("balanced+grouped", "BalGrp"), ("multi+grouped", "MultiGrp"),
                   ("as-is+grouped", "AsisGrp"), ("binary", "Binary")):
        M[f"Clean{sk}"] = num(cl["shapes"][sh]["clean"])
    for k in ["box"] + PRIMS:
        for key, m in (("", ma), ("Clean", mc)):
            d = m.get(k)
            M[f"Matched{key}{k.capitalize()}"] = num(d["median"]) if d else "--"
            M[f"Matched{key}{k.capitalize()}N"] = str(d["n"]) if d else "0"
            M[f"Matched{key}{k.capitalize()}Upper"] = num(d["q75"]) if d else "--"
    tor = ma.get("torus")
    M["MatchedTorusMax"] = num(tor["max"], "{:.0f}") if tor else "--"
    M["MatchedTorusMaxSeries"] = tor["max_series"].replace("_", "\\_") if tor else "--"
    M["MatchedTorusMaxLeaves"] = str(tor["max_leaves"]) if tor else "--"
    for k in PRIMS:
        for key, f in (("", fa), ("Abc", fb)):
            c = f["coef"].get(k)
            M[f"Prim{k.capitalize()}{key}"] = num(np.exp(c["beta"]) if c else None)
    for name, key in (("frame", "Frame"), ("liner", "Liner"), ("review frames", "Review"), ("ABC known", "Known"),
                      ("ABC known, sub-parts", "Sub"), ("ABC pilot", "Pilot")):
        r = fam.get(name)
        for s, sk in (("balanced+grouped", "BalGrp"), ("multi+grouped", "MultiGrp"), ("balanced", "Bal"),
                      ("multi", "Multi"), ("as-is+grouped", "AsisGrp")):
            M[f"Fam{key}{sk}"] = num(r["shapes"][s]["median"]) if r else "--"
        M[f"Fam{key}N"] = str(r["n"]) if r else "0"
    for s, sk in (("balanced", "Bal"), ("multi", "Multi"), ("balanced+grouped", "BalGrp"), ("multi+grouped", "MultiGrp"),
                  ("as-is+grouped", "AsisGrp")):
        d = de.get(s)
        M[f"Rho{sk}"] = num(d["rho_depth"]) if d else "--"
        M[f"RhoCut{sk}"] = num(d["rho_cut"]) if d else "--"
    cb = res["cut_share"]
    for i, w in enumerate(("One", "Two", "Three", "Four")):
        for s, sk in (("balanced", "Bal"), ("multi", "Multi"), ("balanced+grouped", "BalGrp"), ("multi+grouped", "MultiGrp")):
            M[f"Cut{w}{sk}"] = num(cb[i]["shapes"][s]["median"])
        M[f"Cut{w}N"] = str(cb[i]["n"])
        cc = cl["cut_share"][i]
        for s, sk in (("balanced", "Bal"), ("multi", "Multi"), ("balanced+grouped", "BalGrp"), ("multi+grouped", "MultiGrp")):
            M[f"CleanCut{w}{sk}"] = num(cc["shapes"][s]["median"])
        M[f"CleanCut{w}N"] = str(cc["n"])
    (out / "numbers_detail.tex").write_text("\n".join(f"\\newcommand{{\\rd{k}}}{{{v}}}" for k, v in M.items()) + "\n")


def summary(res: dict) -> str:
    f = lambda v: "   -  " if v is None else f"{v:6.2f}"
    L = ["family                  n  leaves depth  cuts  nav us  /entry   " + "  ".join(f"{s[:9]:>9s}" for s in SHAPES)]
    for r in res["families"]:
        L.append(f"  {r['family']:21s} {r['n']:3d}  {r['leaves']:5.0f} {r['depth']:5.0f}  {r['cut_share']:.2f} {f(r['nav'])} "
                 f"{f(r['nav_entry'])}   " + "  ".join(f"{f(r['shapes'][s]['median']):>9s}" for s in SHAPES)
                 + f"   best: {r['best']} ({r['best_share']:.0%})")
    for key in ("all", "abc"):
        p = res["primitives"][key]
        L.append(f"primitives ({key}): n={p['n']} R2={p['r2']:.2f}  " + ", ".join(
            f"{t} {p['coef'][t]['beta']:.2f}" for t in ("log depth", "log leaves")) + "  | " + ", ".join(
            f"{k} x{np.exp(p['coef'][k]['beta']):.2f} [{np.exp(p['coef'][k]['lo']):.2f},{np.exp(p['coef'][k]['hi']):.2f}]"
            f" ({p['programs_with'][k]})" for k in p["prims"]))
    nz = res["noise"]
    L.append(f"noise: baseline range {nz['baseline_range']:.3f} us, median range of all geometries {nz['median_range']:.3f} us,"
             f" rows with navigation <= 0: {nz['rows_nonpositive']}")
    cl = res["clean"]
    L.append(f"clean subset (as-is navigation >= {CLEAN_US:g} us): {cl['n']} programs {cl['families']}")
    for sh in SHAPES:
        d = cl["shapes"][sh]
        L.append(f"  {sh:16s} all {f(d['all'])} ({d['n_all']})   clean {f(d['clean'])} ({d['n_clean']})")
    for r in cl["cut_share"]:
        L.append(f"  clean cuts {r['bin']:10s} {r['n']:4d}  " + "  ".join(f"{s[:9]:>9s} {f(r['shapes'][s]['median'])}"
                                                                        for s in SHAPES))
    for key in ("all", "clean"):
        L.append(f"matched primitives ({key}): " + ", ".join(
            f"{k} x{d['median']:.2f} [{d['q25']:.2f}-{d['q75']:.2f}] ({d['n']})" for k, d in res["matched"][key].items()))
    L.append("by share of cuts:")
    for r in res["cut_share"]:
        L.append(f"  {r['bin']:10s} {r['n']:4d}  " + "  ".join(f"{s[:9]:>9s} {f(r['shapes'][s]['median'])}" for s in SHAPES))
    L.append(f"depth: pooled rho(log depth ratio, log ratio) = {res['depth']['rho_pooled']:.2f}")
    for s, d in res["depth"]["shapes"].items():
        L.append(f"  {s:16s} n={d['n']:3d} depth ratio {f(d['depth_ratio'])} rho {f(d['rho_depth'])}  "
                 f"half-depth {f(d['half_depth']['median'])} ({d['half_depth']['n']}) else {f(d['rest']['median'])} "
                 f"({d['rest']['n']})  rho cuts {f(d['rho_cut'])}  rho imbalance {f(d['rho_imbalance'])}  "
                 f"torus {f(d['torus']['median'])} ({d['torus']['n']}) no torus {f(d['no_torus']['median'])}")
    L.append("benchmark parts (navigation per event, us; (x) not the same solid):")
    for p in res["benchmarks"]:
        L.append(f"  {p['series']:28s} leaves {p['leaves']:4d} depth {p['depth']:3d} cuts {p['cut_share']:.2f}  " + ", ".join(
            f"{s} {v['nav']:.2f}" + ("" if v["tolerated"] else "(x)") for s, v in sorted(p["shapes"].items())))
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "docs" / "reports" / "real_trees")
    ap.add_argument("--tolerate", type=Path, help="aegir graze-check verdicts, as in real_tree_analysis.py")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ps = programs(a.stage, rta.tolerated_by_check(a.tolerate))
    res = dict(families=by_family(ps), benchmarks=benchmark_parts(ps),
               primitives=dict(all=fit(ps), abc=fit([p for p in ps if p["family"].startswith("ABC")])),
               cut_share=by_cut_share(ps), depth=depth_effect(ps), noise=noise(a.stage), clean=clean(ps),
               matched=dict(all=matched_prims(ps), clean=matched_prims(ps, CLEAN_US)))
    (a.out / "detail.json").write_text(json.dumps(res, indent=1, default=float))
    figures(res, a.out)
    tex(res, a.out)
    text = summary(res)
    (a.out / "detail_summary.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
