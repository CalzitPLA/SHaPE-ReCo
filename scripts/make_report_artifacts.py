#!/usr/bin/env python3
"""
Figures, tables and number macros for docs/thesis_inscale_training.tex, generated from the stored results
(docs/data/*.json) - so no number in the report is typed by hand:

    python3 scripts/report_type_cnn.py --out docs/data/type_cnn_eval.json          # GPU evaluation
    python3 scripts/probe_type_cnn_slices.py ... --json docs/data/probe_<net>_<mode>.json
    python3 scripts/make_report_artifacts.py                                        # this script

writes docs/figures/*.pdf, docs/generated/tables.tex and docs/generated/macros.tex.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 8, "axes.titlesize": 8,
                     "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
NEW, OLD, FILL = "#c0392b", "#2c6fbb", "#7f7f7f"
FRACTIONS = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
MODES = (("fit", "stretched grid\n(\"fit\": old training domain)"), ("uniform", "in scale, fills the grid\n(\"uniform\")"),
         ("fixed", "in scale, partly populated\n(\"fixed\": new training domain)"))


def load(name):
    return json.loads((DATA / name).read_text())


def pct(x, digits=1):
    return f"{100 * x:.{digits}f}"


def ci(c, digits=1):
    return f"[{pct(c[0], digits)}, {pct(c[1], digits)}]"


def sci(p):
    if p == 0:
        return "0"
    mant, exp = f"{p:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


# ------------------------------------------------------------------------------------------------ figures

def fig_training():
    new, old = load("training_history_inscale.json"), load("training_history_stretched.json")
    ev = load("type_cnn_eval.json")
    sel = ev["checkpoints"]["new"]["selected_epoch"]
    floor = 1 - 0.1 + 0.1 / 17
    floor = -(floor * np.log(floor) + 0.1 * (16 / 17) * np.log(0.1 / 17))       # cross-entropy floor, alpha = 0.1, K = 17
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.5))
    e = np.arange(1, len(new["val_loss"]) + 1)
    ax[0].plot(e, new["train_loss"], color=NEW, ls="--", lw=1, label="train")
    ax[0].plot(e, new["val_loss"], color=NEW, lw=1.2, label="validation")
    ax[0].axhline(floor, color="k", lw=0.6, ls=":")
    ax[0].text(1.5, floor - 0.02, "label-smoothing floor", ha="left", va="top", fontsize=6)
    ax[0].axvline(sel, color=FILL, lw=0.8)
    ax[0].set(xlabel="epoch", ylabel="cross-entropy (smoothed)", ylim=(0.5, 3.0), xlim=(0, len(e) + 1), title="in-scale data")
    ax[0].text(sel + 0.5, 2.9, f"selected: epoch {sel}", fontsize=6, va="top", color="0.35")
    ax[0].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.85))
    ax[1].plot(e, 100 * np.array(new["val_acc"]), color=NEW, lw=1.2, label="in-scale data, validation")
    eo = np.arange(1, len(old["val_acc"]) + 1)
    ax[1].plot(eo, 100 * np.array(old["val_acc"]), color=OLD, lw=1.0, label="stretched whole shapes, validation")
    ax[1].axvline(sel, color=FILL, lw=0.8)
    ax[1].set(xlabel="epoch", ylabel="validation accuracy (%)", ylim=(40, 104), title="validation accuracy")
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig_training.pdf")
    plt.close(fig)


def fig_fill():
    ev = load("type_cnn_eval.json")
    edges = np.logspace(-4, np.log10(0.8), 30)
    fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.5), sharey=False)
    for a, key, title in ((ax[0], "stretched/train", "stretched whole shapes (old data)"),
                          (ax[1], "inscale/train", "in-scale, cut, noisy (new data)")):
        d = ev["data"][key]["fill_values"]
        a.hist(d["box"], bins=edges, alpha=0.6, color=OLD, label="box", weights=np.ones(len(d["box"])) / len(d["box"]))
        a.hist(d["sphere"], bins=edges, alpha=0.6, color=NEW, label="sphere", weights=np.ones(len(d["sphere"])) / len(d["sphere"]))
        rule = ev["fill_rule"]["stretched" if key.startswith("stretched") else "inscale"]
        a.axvline(rule["threshold"], color="k", lw=0.8, ls="--")
        a.set(xscale="log", xlabel="fraction of the $64^3$ grid that is occupied", ylabel="share of samples",
              title=f"{title}\nrule: sphere iff fill $\\geq$ {rule['threshold']:.3g}, "
                    f"train acc. {pct(rule['train_accuracy'], 0)}%")
        a.title.set_fontsize(7)
    ax[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_fill.pdf")
    plt.close(fig)


def fig_samples():
    from training.manifest import load_manifest
    from training.voxel_dataset import load_voxel
    from report_type_cnn import placements_by_voxel_path
    root = ROOT / "dataset_gen" / "local_dataset_inscale" / "type"
    stretched = load_manifest(ROOT / "dataset_gen" / "local_dataset_box_sphere" / "type")
    recs = load_manifest(root)
    place = placements_by_voxel_path(root)
    rng = np.random.default_rng(3)

    def pick(topology, ncuts):
        pool = [r for r in recs if r.topology == topology and len((place.get(str(r.voxel_path)) or {}).get("cuts", [])) == ncuts]
        return pool[int(rng.integers(len(pool)))]
    cols = [("stretched\n(old data)", lambda t: [r for r in stretched if r.topology == t][7]),
            ("in scale,\nno cut", lambda t: pick(t, 0)), ("in scale,\none cut", lambda t: pick(t, 1)),
            ("in scale,\ntwo cuts", lambda t: pick(t, 2))]
    fig, ax = plt.subplots(2, 4, figsize=(6.3, 3.3))
    for i, topology in enumerate(("box", "sphere")):
        for j, (title, chooser) in enumerate(cols):
            v = (load_voxel(chooser(topology).voxel_path, 64) > 0)
            ax[i, j].imshow(v.max(axis=0), cmap="gray_r", interpolation="nearest")   # projection along array axis 0 (world z)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            for s in ax[i, j].spines.values():
                s.set_visible(True); s.set_linewidth(0.5)
            if i == 0:
                ax[i, j].set_title(title, fontsize=7)
            if j == 0:
                ax[i, j].set_ylabel(topology)
    fig.tight_layout()
    fig.savefig(FIG / "fig_samples.pdf")
    plt.close(fig)


def probe_curves(res):
    """{fraction: (mean, min, max) of P(true class) over the six cap directions}."""
    out = {}
    for pct_ in FRACTIONS:
        ps = [v["p"] for k, v in res.items() if k.startswith("cap:") and k.endswith(f"@{pct_}")]
        out[pct_] = (float(np.mean(ps)), float(np.min(ps)), float(np.max(ps)))
    return out


def fig_probe():
    fig, ax = plt.subplots(2, 3, figsize=(6.3, 3.9), sharex=True, sharey=True)
    for i, shape in enumerate(("box", "sphere")):
        for j, (mode, title) in enumerate(MODES):
            for net, color, label in (("old", OLD, "trained on stretched whole shapes"), ("new", NEW, "trained on in-scale data")):
                res = load(f"probe_{net}_{mode}.json")["raw"][shape]
                c = probe_curves(res)
                x = np.array(FRACTIONS) / 100
                ax[i, j].plot(x, [c[p][0] for p in FRACTIONS], color=color, lw=1.2, label=label)
                ax[i, j].fill_between(x, [c[p][1] for p in FRACTIONS], [c[p][2] for p in FRACTIONS], color=color, alpha=0.18, lw=0)
            ax[i, j].axhline(0.5, color="k", lw=0.5, ls=":")
            if i == 0:
                ax[i, j].set_title(title, fontsize=7)
            if j == 0:
                ax[i, j].set_ylabel(f"$P$({shape})")
            if i == 1:
                ax[i, j].set_xlabel("fraction of the extent kept")
    ax[0, 0].set_ylim(-0.02, 1.02)
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig_probe.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ tables & macros

def tables_and_macros():
    ev = load("type_cnn_eval.json")
    R = ev["results"]
    T = []

    # dataset table
    def row(name, d, has_cuts):
        cut = f"{pct(d['cut_share'], 0)}\\,\\% / {pct(d['two_cut_share'], 0)}\\,\\%" if has_cuts else "--"
        size = f"{d['size_fraction']['min']:.2f}--{d['size_fraction']['max']:.2f}" if has_cuts else "1 (stretched)"
        fb, fs = d["fill"]["box"], d["fill"]["sphere"]
        return (f"{name} & {d['n']} & {d['n_per_class']['box']}/{d['n_per_class']['sphere']} & {cut} & {size} & "
                f"{100 * fb['median']:.1f} / {100 * fs['median']:.1f} \\\\")
    D = ev["data"]
    T.append("\\begin{tabular}{@{}lrrcccc@{}}\n\\toprule\n"
             "Set & $n$ & box/sphere & cut / two cuts & size fraction & median fill (\\%), box / sphere \\\\\n\\midrule")
    T.append(row("in-scale, train", D["inscale/train"], True))
    T.append(row("in-scale, validation", D["inscale/val"], True))
    T.append(row("in-scale, test", D["inscale/test"], True))
    T.append("\\midrule")
    T.append(row("stretched, train", D["stretched/train"], False))
    T.append(row("stretched, validation", D["stretched/val"], False))
    T.append("\\bottomrule\n\\end{tabular}")
    (GEN / "table_data.tex").write_text("\n".join(T))

    # main results
    def acc(key):
        r = R[key]
        return f"{pct(r['accuracy'])} & {ci(r['ci95'])} & {r['correct']}/{r['n']}"
    T = ["\\begin{tabular}{@{}llccc@{}}\n\\toprule\nEvaluation domain / split & Classifier & Accuracy (\\%) & 95\\,\\% CI (\\%) & correct \\\\\n\\midrule"]
    for split, label in (("inscale/val", "in-scale, validation"), ("inscale/test", "in-scale, \\emph{test}")):
        for clf, key in (("Type CNN, in-scale trained", f"new@{split}"), ("Type CNN, stretched trained", f"old@{split}"),
                         ("fill-only rule", f"fill@{split}")):
            T.append(f"{label} & {clf} & {acc(key)} \\\\")
            label = ""
        T.append("\\midrule")
    label = "stretched, validation"
    for clf, key in (("Type CNN, in-scale trained", "new@stretched/val"), ("Type CNN, stretched trained", "old@stretched/val"),
                     ("fill-only rule", "fill@stretched/val")):
        T.append(f"{label} & {clf} & {acc(key)} \\\\")
        label = ""
    T.append("\\bottomrule\n\\end{tabular}")
    (GEN / "table_main.tex").write_text("\n".join(T))

    # subgroups of the new network on the test set
    r = R["new@inscale/test"]
    T = ["\\begin{tabular}{@{}llrcc@{}}\n\\toprule\nSubgroup (test set) & Level & $n$ & Accuracy (\\%) & 95\\,\\% CI (\\%) \\\\\n\\midrule"]
    def add(title, groups):
        first = True
        for name, g in groups.items():
            if g["n"]:
                T.append(f"{title if first else ''} & {name} & {g['n']} & {pct(g['accuracy'])} & {ci(g['ci95'])} \\\\")
                first = False
    add("cuts applied", {"none": r["by_cuts"]["0"], "one": r["by_cuts"]["1"], "two": r["by_cuts"]["2"]})
    T.append("\\midrule")
    add("size fraction", r["by_size_fraction"])
    T.append("\\midrule")
    add("grid fill", {k.replace("-", "--"): v for k, v in r["by_fill"].items()})
    T.append("\\bottomrule\n\\end{tabular}")
    (GEN / "table_subgroups.tex").write_text("\n".join(T))

    # probe summary
    T = ["\\begin{tabular}{@{}lllcccc@{}}\n\\toprule\nVoxelization & Network & Shape & whole shape $P$ & caps correct & slabs correct & mean $P$ (all slices) \\\\\n\\midrule"]
    macros = {}
    for mi, (mode, title) in enumerate(MODES):
        first = True
        for net, netname in (("old", "stretched-trained"), ("new", "in-scale-trained")):
            res = load(f"probe_{net}_{mode}.json")["raw"]
            for shape in ("box", "sphere"):
                d = res[shape]
                caps = [v for k, v in d.items() if k.startswith("cap:")]
                slabs = [v for k, v in d.items() if k.startswith("slab:")]
                whole = d["cap:z-@100"]
                okc = np.mean([v["top"] == shape for v in caps]); oks = np.mean([v["top"] == shape for v in slabs])
                meanp = np.mean([v["p"] for v in caps + slabs])
                T.append(f"{title.splitlines()[0] if first else ''} & {netname if shape == 'box' else ''} & {shape} & "
                         f"{whole['p']:.2f} & {pct(okc, 0)}\\,\\% & {pct(oks, 0)}\\,\\% & {meanp:.2f} \\\\")
                first = False
                macros[f"Probe{net.title()}{mode.title()}{shape.title()}Caps"] = pct(okc, 0)
                macros[f"Probe{net.title()}{mode.title()}{shape.title()}Slabs"] = pct(oks, 0)
                macros[f"Probe{net.title()}{mode.title()}{shape.title()}Whole"] = f"{whole['p']:.2f}"
                macros[f"Probe{net.title()}{mode.title()}{shape.title()}MeanP"] = f"{meanp:.2f}"
        if mi < len(MODES) - 1:
            T.append("\\midrule")
    T.append("\\bottomrule\n\\end{tabular}")
    (GEN / "table_probe.tex").write_text("\n".join(T))

    # macros
    pt = ev["paired_tests"]
    m = {
        "NewValAcc": pct(R["new@inscale/val"]["accuracy"]), "NewValCorrect": R["new@inscale/val"]["correct"], "NewValN": R["new@inscale/val"]["n"],
        "NewValCILo": pct(R["new@inscale/val"]["ci95"][0]), "NewValCIHi": pct(R["new@inscale/val"]["ci95"][1]),
        "NewTestAcc": pct(R["new@inscale/test"]["accuracy"]), "NewTestCorrect": R["new@inscale/test"]["correct"], "NewTestN": R["new@inscale/test"]["n"],
        "NewTestCILo": pct(R["new@inscale/test"]["ci95"][0]), "NewTestCIHi": pct(R["new@inscale/test"]["ci95"][1]),
        "OldTestAcc": pct(R["old@inscale/test"]["accuracy"]), "OldValAcc": pct(R["old@inscale/val"]["accuracy"]),
        "FillTestAcc": pct(R["fill@inscale/test"]["accuracy"]), "FillValAcc": pct(R["fill@inscale/val"]["accuracy"]),
        "FillThreshold": f"{ev['fill_rule']['inscale']['threshold']:.3f}", "FillTrainAcc": pct(ev["fill_rule"]["inscale"]["train_accuracy"]),
        "FillStretchedThreshold": f"{ev['fill_rule']['stretched']['threshold']:.2f}", "FillStretchedValAcc": pct(R["fill@stretched/val"]["accuracy"]),
        "FillStretchedTrainAcc": pct(ev["fill_rule"]["stretched"]["train_accuracy"]),
        "OldStretchedValAcc": pct(R["old@stretched/val"]["accuracy"], 0), "NewStretchedValAcc": pct(R["new@stretched/val"]["accuracy"], 0),
        "NewStretchedValCorrect": R["new@stretched/val"]["correct"], "StretchedValN": R["new@stretched/val"]["n"],
        "McNemarFillNewOnly": pt["new_vs_fill"]["a_only"], "McNemarFillFillOnly": pt["new_vs_fill"]["b_only"], "McNemarFillP": f"${sci(pt['new_vs_fill']['p'])}$",
        "McNemarOldNewOnly": pt["new_vs_old"]["a_only"], "McNemarOldOldOnly": pt["new_vs_old"]["b_only"], "McNemarOldP": f"${sci(pt['new_vs_old']['p'])}$",
        "SelectedEpoch": ev["checkpoints"]["new"]["selected_epoch"], "OldSelectedEpoch": ev["checkpoints"]["old"]["selected_epoch"],
        "Parameters": f"{ev['parameters']:,}".replace(",", "\\,"), "TorchVersion": ev["torch"].split("+")[0],
        "CutShare": pct(D["inscale/train"]["cut_share"], 0), "TwoCutShare": pct(D["inscale/train"]["two_cut_share"], 0),
        "NewTestOldAcc": pct(R["old@inscale/test"]["accuracy"]),
    }
    for c in ("box", "sphere"):
        m[f"NewTestAcc{c.title()}"] = pct(R["new@inscale/test"]["per_class_accuracy"][c]["accuracy"])
        m[f"OldTestPredBox{c.title()}"] = R["old@inscale/test"]["confusion"][c].get("box", 0)
        m[f"NewConf{c.title()}Box"] = R["new@inscale/test"]["confusion"][c].get("box", 0)
        m[f"NewConf{c.title()}Sphere"] = R["new@inscale/test"]["confusion"][c].get("sphere", 0)
        m[f"NewConf{c.title()}Other"] = R["new@inscale/test"]["confusion"][c].get("other", 0)
        m[f"N{c.title()}Test"] = R["new@inscale/test"]["n_per_class"][c]
    m.update(macros)
    (GEN / "macros.tex").write_text("% generated by scripts/make_report_artifacts.py - do not edit\n" +
                                    "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()) + "\n")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    fig_training(); fig_fill(); fig_samples(); fig_probe()
    tables_and_macros()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))
    print("generated:", sorted(p.name for p in GEN.glob("*.tex")))


if __name__ == "__main__":
    main()
