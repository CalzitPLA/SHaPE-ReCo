#!/usr/bin/env python3
"""
Numbers, tables and figures of docs/vectorfield_reconstruction.tex, from the measured data:

    python3 scripts/make_vectorfield_tex.py

reads   docs/data/peel_examples.json                 (scripts/eval_peel_examples.py)
writes  docs/generated/vf_macros.tex                 counts and summary numbers as \\newcommand
        docs/generated/vf_table_examples.tex         one row per example part
        docs/figures/vf_field_slice.pdf              the closest-point field in a plane through a part (quiver + |V|)
        docs/figures/vf_results.pdf                  two-way chamfer per part
        docs/figures/vf_history.pdf                  share of the surface explained after each operation, for a few parts
and times the field queries (one 20 mm part) for the text.
"""
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.peel import peel  # noqa: E402
from src.closest_point_field import load_field  # noqa: E402

DOCS = ROOT / "docs"
EX = ROOT / "dataset_gen" / "examples"
FAMILY = {"co_": "cone", "to_": "torus", "cs_": "segment", "cy_": "cylinder", "bs_": "box / sphere"}
COLORS = {"cone": "#c0504d", "torus": "#8064a2", "segment": "#4bacc6", "cylinder": "#f79646", "box / sphere": "#4f81bd", "box with cuts": "#7f7f7f"}
HISTORY_PARTS = ["bs_box_two_pockets", "cy_t", "co_countersink", "to_ring_groove", "three_holes", "plate_4holes"]


def family(name):
    for k, v in FAMILY.items():
        if name.startswith(k):
            return v
    return "box with cuts"


def tex(s):
    return s.replace("_", r"\_")


def describe(entry):
    op, kind = entry[0], entry[1]
    seg = entry[2] if len(entry) > 2 else {}
    name = kind
    if "u_sweep_deg" in seg:
        name += rf"\,($u{{=}}{seg['u_sweep_deg']:g}^\circ$)"
    if "v_deg" in seg:
        name += rf"\,($v{{=}}[{seg['v_deg'][0]:g},{seg['v_deg'][1]:g}]^\circ$, {seg.get('latitude', 'cone')})"
    return name if op == "base" else f"{op}:{name}"


def table_and_macros(parts):
    rows = []
    for p in parts:
        mean = 0.5 * (p["input_to_model_mean"] + p["model_to_input_mean"])
        mx = max(p["input_to_model_max"], p["model_to_input_max"])
        prog = " + ".join(describe(e) for e in p["program"])
        rows.append(rf"\code{{{tex(p['part'])}}} & {prog} & {100 * p['explained']:.0f} & {mean:.4f} & {mx:.2f} & {p['seconds']:.1f}")
    (DOCS / "generated" / "vf_table_examples.tex").write_text(" \\\\\n".join(rows) + "\n")     # the document ends the last row
    means = [0.5 * (p["input_to_model_mean"] + p["model_to_input_mean"]) for p in parts]
    maxes = [max(p["input_to_model_max"], p["model_to_input_max"]) for p in parts]
    macros = {
        "vfNParts": len(parts), "vfMeanChamfer": f"{np.mean(means):.4f}", "vfWorstMax": f"{max(maxes):.2f}",
        "vfMinSeconds": f"{min(p['seconds'] for p in parts):.1f}", "vfMaxSeconds": f"{max(p['seconds'] for p in parts):.1f}",
        "vfNSegmentParts": sum(1 for p in parts if any(len(e) > 2 for e in p["program"])),
        "vfNAllExplained": sum(1 for p in parts if p["explained"] >= 0.99),
    }
    return macros, means


def timings():
    field = load_field(EX / "blind_hole.stl")
    t = time.time()
    field.grid_vectors(64, 40.0, exact=False)
    grid = time.time() - t
    pts = field.sample_points(20000, band=3.0, seed=0)
    t = time.time()
    field.query(pts)
    near = time.time() - t
    return {"vfGridSeconds": f"{grid:.1f}", "vfNearSeconds": f"{near:.1f}", "vfNearRate": f"{20000 / near / 1000:.1f}",
            "vfNTriangles": len(field.faces)}


def figure_field_slice():
    field = load_field(EX / "blind_hole.stl")
    lo, hi = field.bounds
    c = (lo + hi) / 2
    ext = 0.62 * float((hi - lo).max())
    n = 61
    xs = np.linspace(c[0] - ext, c[0] + ext, n)
    zs = np.linspace(c[2] - ext, c[2] + ext, n)
    X, Z = np.meshgrid(xs, zs)
    P = np.stack([X.ravel(), np.full(X.size, c[1] + 0.31), Z.ravel()], axis=1)      # a plane through the hole's axis
    q = field.query(P)
    d = q.distance.reshape(n, n)
    Vx, Vz = q.vector[:, 0].reshape(n, n), q.vector[:, 2].reshape(n, n)
    fine = np.linspace(0, 1, 260)
    Xf, Zf = np.meshgrid(c[0] - ext + fine * 2 * ext, c[2] - ext + fine * 2 * ext)
    qf = field.query(np.stack([Xf.ravel(), np.full(Xf.size, c[1] + 0.31), Zf.ravel()], axis=1))
    df = qf.distance.reshape(Xf.shape)

    fig, ax = plt.subplots(1, 2, figsize=(9.4, 4.4))
    im = ax[0].imshow(df, origin="lower", extent=[xs[0], xs[-1], zs[0], zs[-1]], cmap="viridis")
    ax[0].contour(Xf, Zf, df, levels=[0.15], colors="white", linewidths=1.0)
    plt.colorbar(im, ax=ax[0], fraction=0.046, label=r"$|V(x)|$ = distance to the surface [mm]")
    ax[0].set_title("distance $|V|$ (white: the surface)", fontsize=9)
    step = 2
    L = np.hypot(Vx, Vz)
    mask = L > 0.4
    U = np.where(mask, Vx, np.nan)
    W = np.where(mask, Vz, np.nan)
    ax[1].quiver(X[::step, ::step], Z[::step, ::step], U[::step, ::step], W[::step, ::step], L[::step, ::step], cmap="viridis",
                 angles="xy", scale_units="xy", scale=1.0, width=0.004)
    ax[1].contour(Xf, Zf, df, levels=[0.15], colors="k", linewidths=1.0)
    ax[1].set_title(r"$V(x)=c(x)-x$ (arrows to the nearest surface point)", fontsize=9)
    for a in ax:
        a.set_aspect("equal")
        a.set_xlabel("x [mm]")
        a.set_ylabel("z [mm]")
    fig.suptitle("Closest-point field of blind_hole.stl in a plane through the hole; the arrows flip on the medial axes", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "vf_field_slice.pdf")
    plt.close(fig)


def figure_results(parts, means):
    order = sorted(range(len(parts)), key=lambda i: (family(parts[i]["part"]), parts[i]["part"]))
    fig, ax = plt.subplots(figsize=(9.4, 3.6))
    for k, i in enumerate(order):
        f = family(parts[i]["part"])
        ax.bar(k, means[i], color=COLORS[f], width=0.75, label=f if f not in ax.get_legend_handles_labels()[1] else None)
        ax.plot([k, k], [means[i], max(parts[i]["input_to_model_max"], parts[i]["model_to_input_max"])], color="#555", lw=0.6)
    ax.set_yscale("log")
    ax.set_ylim(2e-4, 0.6)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([parts[i]["part"] for i in order], rotation=90, fontsize=6)
    ax.set_ylabel("chamfer, mean (bar) to max (line) [mm]")
    ax.axhline(0.004 * 35, color="r", lw=0.7, ls="--")
    ax.text(len(order) - 0.5, 0.004 * 35 * 1.15, "tolerance (0.4 % of a 35 mm part)", color="r", fontsize=6, ha="right")
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.set_title("Two-way distance of the reconstructed model to the input mesh, all example parts", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "vf_results.pdf")
    plt.close(fig)


def figure_history():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name in HISTORY_PARTS:
        res = peel(load_field(EX / f"{name}.stl"))
        ax.plot(range(len(res.history)), [100 * h for h in res.history], marker="o", ms=4, label=name)
    ax.set_xlabel("operations in the program (0 = the base primitive)")
    ax.set_ylabel("surface explained [%]")
    ax.set_ylim(0, 101)
    ax.set_xticks(range(0, 5))
    ax.legend(fontsize=6.5, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("What each operation of the peeling loop adds", fontsize=9)
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "vf_history.pdf")
    plt.close(fig)


def main():
    parts = json.loads((DOCS / "data" / "peel_examples.json").read_text())["parts"]
    macros, means = table_and_macros(parts)
    macros.update(timings())
    (DOCS / "generated" / "vf_macros.tex").write_text(
        "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in macros.items()))
    figure_field_slice()
    figure_results(parts, means)
    figure_history()
    print("wrote docs/generated/vf_macros.tex, vf_table_examples.tex and three figures;", macros)


if __name__ == "__main__":
    main()
