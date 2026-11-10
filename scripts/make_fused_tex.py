#!/usr/bin/env python3
"""
Figures, tables and number macros for docs/thesis_fused_input.tex and docs/slides_fused_input.tex, generated from the
stored held-out evaluations (docs/data/type_fused_eval.json, docs/data/rotation_eval_fused.json) - so no number in
those two documents is typed by hand:

    python3 scripts/eval_type_fused.py --model occupancy=checkpoints_fused/type_occupancy/type_cnn --model fused=checkpoints_fused/type_fused/type_cnn \
        --out docs/data/type_fused_eval.json
    python3 scripts/eval_rotation.py --model occupancy=checkpoints_fused/rot_occupancy/rotation_cnn --model fused=checkpoints_fused/rot_fused/rotation_cnn \
        --out docs/data/rotation_eval_fused.json
    python3 scripts/make_fused_tex.py

writes docs/figures/fig_fused_type.pdf, fig_fused_rotation.pdf, docs/generated/table_fused_type.tex,
table_fused_rotation.tex and fused_macros.tex.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG, GEN = ROOT / "docs" / "data", ROOT / "docs" / "figures", ROOT / "docs" / "generated"
sys.path.insert(0, str(ROOT / "scripts"))
from report_type_cnn import wilson  # noqa: E402

OCC, FUS, NET = "#2c6fbb", "#c0392b", "#2e8b57"                                    # same palette as make_report_artifacts.py
CLASSES = ("box", "cylinder", "sphere")
METHODS = (("random", "random"), ("pca", "PCA (no learning)"), ("old-cnn", "old Rotation CNN (occupancy, world frame)"),
          ("occupancy", "Rotation CNN, occupancy"), ("fused", "Rotation CNN, fused"), ("fused_net", "FusedNet (fused, shared backbone)"))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def load(name):
    return json.loads((DATA / name).read_text())


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def ci(c, digits=1):
    return f"[{pct(c[0], digits)}, {pct(c[1], digits)}]"


def class_n(d):
    """n per class, backed out from the stored accuracy and confusion counts (the json stores fractions, not counts)."""
    errors = {"box": 0, "cylinder": 0, "sphere": 0}
    for pair, count in d["confusions"].items():
        errors[pair.split("->")[0]] += count
    return {c: round(errors[c] / (1 - d["per_class"][c])) for c in CLASSES}


def type_table(occ, fus, net=None):
    n_occ = class_n(occ)
    header = r"class & $n$ & occupancy & 95\,\% CI & fused & 95\,\% CI"
    cols = "lrrrrr"
    if net is not None:
        header += " & FusedNet & 95\\,\\% CI"
        cols += "rr"
    lines = [r"\begin{tabular}{@{}" + cols + r"@{}}", r"\toprule", header + r" \\", r"\midrule"]
    for c in CLASSES:
        n = n_occ[c]
        k_occ, k_fus = round(occ["per_class"][c] * n), round(fus["per_class"][c] * n)
        row = f"{c} & {n} & {pct(occ['per_class'][c])}\\,\\% & {ci(wilson(k_occ, n))} & " \
              f"{pct(fus['per_class'][c])}\\,\\% & {ci(wilson(k_fus, n))}"
        if net is not None:
            k_net = round(net["per_class"][c] * n)
            row += f" & {pct(net['per_class'][c])}\\,\\% & {ci(wilson(k_net, n))}"
        lines.append(row + r" \\")
    n = occ["n"]
    lines.append(r"\midrule")
    row = f"overall & {n} & \\textbf{{{pct(occ['accuracy'])}}}\\,\\% & {ci(wilson(round(occ['accuracy'] * n), n))} & " \
          f"\\textbf{{{pct(fus['accuracy'])}}}\\,\\% & {ci(wilson(round(fus['accuracy'] * n), n))}"
    if net is not None:
        row += f" & \\textbf{{{pct(net['accuracy'])}}}\\,\\% & {ci(wilson(round(net['accuracy'] * n), n))}"
    lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_fused_type.tex").write_text("\n".join(lines))


def _present_methods(m):
    return [(key, label) for key, label in METHODS if key in m]


def rotation_table(m):
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"method & $n$ & median (deg) & mean (deg) & within 10\,\textdegree{} \\", r"\midrule"]
    for key, label in _present_methods(m):
        a = m[key]["all"]
        lines.append(f"{label} & {a['n']} & {a['median']:.1f} & {a['mean']:.1f} & {pct(a['within10'])}\\,\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (GEN / "table_fused_rotation.tex").write_text("\n".join(lines))


def fig_type(occ, fus, net=None):
    fig, ax = plt.subplots(figsize=(3.1, 2.5))
    x = np.arange(len(CLASSES) + 1)
    vals_occ = [100 * occ["per_class"][c] for c in CLASSES] + [100 * occ["accuracy"]]
    vals_fus = [100 * fus["per_class"][c] for c in CLASSES] + [100 * fus["accuracy"]]
    series = [("occupancy", vals_occ, OCC), ("fused", vals_fus, FUS)]
    if net is not None:
        series.append(("FusedNet", [100 * net["per_class"][c] for c in CLASSES] + [100 * net["accuracy"]], NET))
    w = 0.8 / len(series)
    for i, (label, vals, color) in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, color=color, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(list(CLASSES) + ["overall"])
    ax.set_ylabel("held-out accuracy [%]")
    ax.set_ylim(70, 100)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_fused_type.pdf")
    plt.close(fig)


def fig_rotation(m):
    present = _present_methods(m)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.5))
    labels = [lbl for _, lbl in present]
    base_colors = {"random": "0.6", "pca": "0.6", "old-cnn": "0.6", "occupancy": OCC, "fused": FUS, "fused_net": NET}
    colors = [base_colors[k] for k, _ in present]
    medians = [m[k]["all"]["median"] for k, _ in present]
    within10 = [100 * m[k]["all"]["within10"] for k, _ in present]
    y = np.arange(len(present))
    ax[0].barh(y, medians, color=colors)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=6)
    ax[0].set_xlabel("median angular error (deg)")
    ax[0].invert_yaxis()
    ax[1].barh(y, within10, color=colors)
    ax[1].set_yticks(y)
    ax[1].set_yticklabels([])
    ax[1].set_xlabel("within 10 deg [%]")
    ax[1].invert_yaxis()
    fig.tight_layout()
    fig.savefig(FIG / "fig_fused_rotation.pdf")
    plt.close(fig)


def macros(occ_t, fus_t, m, net_t=None):
    n_occ = class_n(occ_t)
    lines = ["% generated by scripts/make_fused_tex.py - do not edit"]

    def add(name, value):
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    add("FusedTypeN", occ_t["n"])
    add("FusedTypeOccAcc", pct(occ_t["accuracy"]))
    add("FusedTypeFusAcc", pct(fus_t["accuracy"]))
    for c in CLASSES:
        cap = c.capitalize()
        add(f"FusedType{cap}N", n_occ[c])
        add(f"FusedType{cap}Occ", pct(occ_t["per_class"][c]))
        add(f"FusedType{cap}Fus", pct(fus_t["per_class"][c]))
    add("FusedRotN", m["occupancy"]["all"]["n"])
    add("FusedRotOccMedian", f"{m['occupancy']['all']['median']:.1f}")
    add("FusedRotFusMedian", f"{m['fused']['all']['median']:.1f}")
    add("FusedRotOccWithinTen", pct(m["occupancy"]["all"]["within10"]))
    add("FusedRotFusWithinTen", pct(m["fused"]["all"]["within10"]))
    add("FusedRotPcaMedian", f"{m['pca']['all']['median']:.1f}")
    add("FusedRotPcaWithinTen", pct(m["pca"]["all"]["within10"]))
    add("FusedRotRandomMedian", f"{m['random']['all']['median']:.1f}")
    if net_t is not None:
        add("FusedNetTypeAcc", pct(net_t["accuracy"]))
        for c in CLASSES:
            add(f"FusedNetType{c.capitalize()}", pct(net_t["per_class"][c]))
    if "fused_net" in m:
        add("FusedNetRotN", m["fused_net"]["all"]["n"])
        add("FusedNetRotMedian", f"{m['fused_net']['all']['median']:.1f}")
        add("FusedNetRotWithinTen", pct(m["fused_net"]["all"]["within10"]))
    (GEN / "fused_macros.tex").write_text("\n".join(lines) + "\n")


def main():
    t = load("type_fused_eval.json")
    r = load("rotation_eval_fused.json")
    net_t = t.get("fused_net")
    type_table(t["occupancy"], t["fused"], net_t)
    rotation_table(r["methods"])
    fig_type(t["occupancy"], t["fused"], net_t)
    fig_rotation(r["methods"])
    macros(t["occupancy"], t["fused"], r["methods"], net_t)
    print("wrote docs/generated/{table_fused_type,table_fused_rotation,fused_macros}.tex, "
          "docs/figures/fig_fused_{type,rotation}.pdf")


if __name__ == "__main__":
    main()
