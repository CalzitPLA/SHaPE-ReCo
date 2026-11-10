#!/usr/bin/env python3
"""The chamfer study: formulas against FreeCAD's exact volumes, against voxels, and against the cascade's runs.

    python3 scripts/chamfer_study.py [--parts docs/reports/chamfer_complexity/parts]
                                     [--runs ~/.cache/shape_reco/prof/chamfer] [--out docs/reports/chamfer_complexity]

Reads parts/inventory.json (scripts/make_chamfer_pockets.py) and the cascade's JSON per part (scripts/time_cascade.py),
writes data/study.json, the tables of the report (data/*.tex) and its figures (fig/*.pdf). Every check is an assert:
the script fails when a formula and a measurement disagree.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

KINDS = ("rim_chamfer", "corner_chamfer", "corner_fillet")
LABEL = {"rim_chamfer": "rim chamfer", "corner_chamfer": "corner chamfer", "corner_fillet": "corner fillet"}
COLOR = {"rim_chamfer": "#2a78d6", "corner_chamfer": "#eb6834", "corner_fillet": "#1baf7a"}     # validated, see report
MARKER = {"rim_chamfer": "o", "corner_chamfer": "s", "corner_fillet": "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


# ------------------------------------------------------------------------------------------------ formulas

def edge_volume(kind, s, l, w, d):
    """The volume between the part with finished edges and the plain part."""
    if kind == "rim_chamfer":
        return (l + w) * s ** 2 + 4.0 * s ** 3 / 3.0            # removed: the mouth widened by (s - t) at depth t
    if kind == "corner_chamfer":
        return 4 * 0.5 * s ** 2 * d                             # left standing: 4 prisms of cross-section s^2 / 2
    if kind == "corner_fillet":
        return 4 * (1 - math.pi / 4) * s ** 2 * d               # left standing: 4 x (square - quarter disc)
    return 0.0


def iou_ignoring(kind, s, V_plain, l, w, d):
    """The exact IoU of the plain model with the part: the two are nested, so 1 - IoU = v / V of the larger."""
    v = edge_volume(kind, s, l, w, d)
    return (V_plain - v) / V_plain if kind == "rim_chamfer" else V_plain / (V_plain + v)


def p_seen(c_over_p):
    """The probability that a right triangle with legs c holds a point of a square lattice of pitch p placed at
    random: c^2 / 2 p^2 up to c = p, 1 - (2 p - c)^2 / 2 p^2 up to c = 2 p, 1 beyond."""
    x = np.asarray(c_over_p, float)
    return np.where(x <= 1, x ** 2 / 2, np.where(x <= 2, 1 - (2 - x) ** 2 / 2, 1.0))


# ------------------------------------------------------------------------------------------------ voxels

def inside(kind, s, P, block, pocket):
    """The part's material at the points P (n, 3): the analytic solid, the same parity a mesh of it would give."""
    (L, W, H), (l, w, d) = block, pocket
    x0, y0, z0 = (L - l) / 2, (W - w) / 2, H - d
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    in_block = (x >= 0) & (x <= L) & (y >= 0) & (y <= W) & (z >= 0) & (z <= H)
    u = np.minimum(x - x0, x0 + l - x)                          # distance to the nearer x wall of the pocket
    v = np.minimum(y - y0, y0 + w - y)
    removed = (u >= 0) & (v >= 0) & (z >= z0)
    if kind == "corner_chamfer":
        removed &= (u + v >= s)
    elif kind == "corner_fillet":
        removed &= ~((u < s) & (v < s) & ((s - u) ** 2 + (s - v) ** 2 > s ** 2))
    elif kind == "rim_chamfer":
        g = s - (H - z)                                         # the mouth's widening at this height
        removed |= (g >= 0) & (u >= -g) & (v >= -g)
    return in_block & ~removed


def voxel_experiment(kind, s, block, pocket, pitch, offsets, rng):
    """Voxel IoU of the plain model with the part, and the voxels of the difference, for random grid offsets."""
    ious, counts = [], []
    for _ in range(offsets):
        o = rng.uniform(0, pitch, 3)
        axes = [np.arange(o[k], block[k], pitch) for k in range(3)]
        P = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        a, b = inside(kind, s, P, block, pocket), inside("plain", 0.0, P, block, pocket)
        ious.append((a & b).sum() / (a | b).sum())
        counts.append(int((a ^ b).sum()))
    return dict(iou_mean=float(np.mean(ious)), iou_std=float(np.std(ious, ddof=1)), difference_voxels_mean=float(np.mean(counts)),
                difference_voxels_std=float(np.std(counts, ddof=1)), difference_voxels_min=int(np.min(counts)),
                difference_voxels_max=int(np.max(counts)), empty_share=float(np.mean(np.array(counts) == 0)))


def seen_simulation(ratios, trials, rng):
    """The share of random lattice placements with a point in the triangle x, y >= 0, x + y <= c (pitch 1)."""
    out = []
    for c in ratios:
        o = rng.uniform(0, 1, (trials, 2))
        k = np.arange(0, int(c) + 2)
        X = o[:, None, None, 0] + k[None, :, None]
        Y = o[:, None, None, 1] + k[None, None, :]
        out.append(float(((X + Y) <= c).any(axis=(1, 2)).mean()))
    return out


# ------------------------------------------------------------------------------------------------ the study

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="docs/reports/chamfer_complexity/parts")
    ap.add_argument("--runs", default="~/.cache/shape_reco/prof/chamfer")
    ap.add_argument("--out", default="docs/reports/chamfer_complexity")
    ap.add_argument("--offsets", type=int, default=100)
    args = ap.parse_args()
    out, runs = Path(args.out), Path(os.path.expanduser(args.runs))
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "fig").mkdir(parents=True, exist_ok=True)
    inv = json.loads((Path(args.parts) / "inventory.json").read_text())
    block, pocket = inv["block"], inv["pocket"]
    l, w, d = pocket
    parts = {r["name"]: r for r in inv["parts"]}
    V_plain = parts["plain"]["volume"]
    assert abs(V_plain - (np.prod(block) - np.prod(pocket))) < 1e-6
    rng = np.random.default_rng(20260928)
    rows = []
    for r in inv["parts"]:
        if r["kind"] == "plain":
            continue
        kind, s = r["kind"], r["size"]
        v = edge_volume(kind, s, l, w, d)
        sign = -1 if kind == "rim_chamfer" else 1
        # 1. the volume formulas against FreeCAD's exact solid
        assert abs(r["volume"] - (V_plain + sign * v)) < 1e-6 * V_plain, (r["name"], r["volume"], V_plain + sign * v)
        # 2. the IoU of nested solids against OCCT's common and union
        iou = iou_ignoring(kind, s, V_plain, l, w, d)
        assert abs(r["iou_with_plain"] - iou) < 1e-9, (r["name"], r["iou_with_plain"], iou)
        row = dict(name=r["name"], kind=kind, size=s, edge_volume=v, share=v / max(V_plain, r["volume"]),
                   iou_formula=iou, iou_occt=r["iou_with_plain"], faces=r["faces"], edges=r["edges"],
                   vertices=r["vertices"], surface_types=r["surface_types"], voxels={})
        # 3. voxels at four pitches, random offsets
        for pitch in (0.725, 1.0, 2.0, 4.0):
            row["voxels"]["%g" % pitch] = dict(voxel_experiment(kind, s, block, pocket, pitch, args.offsets, rng),
                                                expected_voxels=v / pitch ** 3)
        # 4. the cascade's run
        f = runs / (r["name"] + ".json")
        if f.is_file():
            run = json.loads(f.read_text())
            row["run"] = _run_row(run)
        rows.append(row)
    plain_run = json.loads((runs / "plain.json").read_text()) if (runs / "plain.json").is_file() else None
    ratios = np.round(np.arange(0.2, 2.61, 0.2), 2)
    sim = seen_simulation(ratios, 200_000, rng)
    assert np.max(np.abs(np.array(sim) - p_seen(ratios))) < 5e-3            # the lattice formula, by simulation
    # the voxels of the difference are v / p^3 on average. Asserted where the expectation is at least 50 voxels: within
    # 4 standard errors of the mean over the placements. Below that the count is all or nothing (a row of centres is in
    # the chamfer or it is not) and a mean over a few placements says little; those cases are listed, not asserted.
    rare = []
    for row in rows:
        for pitch, vx in row["voxels"].items():
            se = vx["difference_voxels_std"] / math.sqrt(args.offsets)
            if vx["expected_voxels"] >= 50:
                assert abs(vx["difference_voxels_mean"] - vx["expected_voxels"]) <= 4 * se + 0.5, (row["name"], pitch, vx)
                assert abs(vx["iou_mean"] - row["iou_formula"]) <= 4 * vx["iou_std"] / math.sqrt(args.offsets) + 1e-5, (row["name"], pitch, vx)
            else:
                rare.append(dict(name=row["name"], pitch=float(pitch), expected=vx["expected_voxels"],
                                 mean=vx["difference_voxels_mean"], min=vx["difference_voxels_min"],
                                 max=vx["difference_voxels_max"], empty_share=vx["empty_share"]))
    study = dict(block=block, pocket=pocket, V_plain=V_plain, rows=rows, plain_run=_run_row(plain_run) if plain_run else None,
                 seen=dict(ratio=ratios.tolist(), simulated=sim, formula=p_seen(ratios).tolist()), offsets=args.offsets, rare=rare)
    (out / "data" / "study.json").write_text(json.dumps(study, indent=1))
    tables(study, out / "data")
    figures(study, out / "fig")
    print("all checks passed: %d parts, %d with a cascade run" % (len(rows), sum("run" in r for r in rows)))


def _run_row(run):
    s, g = run["summary"], run["summary"].get("grade") or {}
    prog = run.get("program") or []
    return dict(ops=len(prog), program=["%s %s" % (p["op"], p["kind"]) for p in prog], voxel_iou=s["stats"]["iou"],
                residual=s["residual"], exact_iou=g.get("exact_iou"), missed=g.get("exact_r_plus"),
                invented=g.get("exact_r_minus"), valid=g.get("valid"), chamfer=g.get("chamfer"),
                chamfer_max=g.get("chamfer_max"), seconds=s["elapsed"]["wall"], fallback=s.get("fallback_boxes"),
                input_voxels=(prog[0].get("component_voxels") if prog else None))


def tables(study, folder):
    lines = []
    for r in study["rows"]:
        lines.append("%s & %g & %.1f & %.3f & %.6f & %.6f & %d & %d & %d" % (
            LABEL[r["kind"]], r["size"], r["edge_volume"], 100 * r["share"], r["iou_formula"], r["iou_occt"],
            r["faces"], r["edges"], r["vertices"]))
    (folder / "table_exact.tex").write_text(" \\\\\n".join(lines) + "\n")
    lines = []
    for r in study["rows"]:
        v = r["voxels"]
        lines.append("%s & %g & " % (LABEL[r["kind"]], r["size"]) + " & ".join(
            "%.0f & %d--%d & %.0f" % (v[p]["expected_voxels"], v[p]["difference_voxels_min"], v[p]["difference_voxels_max"],
                                     100 * v[p]["empty_share"]) for p in ("0.725", "2", "4")))
    (folder / "table_voxels.tex").write_text(" \\\\\n".join(lines) + "\n")
    lines = []
    for r in study["rows"]:
        if "run" not in r:
            continue
        q = r["run"]
        prog = ", ".join("%d %s" % (q["program"].count(k), k) for k in sorted(set(q["program"]))).replace("composite", "comp.")
        lines.append("%s & %g & %.2f & %d & %s & %s & %.4f & %.4f & %.1f & %.1f & %.0f" % (
            LABEL[r["kind"]], r["size"], 100 * r["share"], q["ops"], prog, "yes" if q["valid"] else "no",
            q["exact_iou"], r["iou_formula"], q["missed"], q["invented"], q["seconds"]))
    (folder / "table_runs.tex").write_text(" \\\\\n".join(lines) + "\n")


def figures(study, folder):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "legend.frameon": False, "font.family": "DejaVu Sans", "axes.titlesize": 9.5,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    rows, (l, w, d), V = study["rows"], study["pocket"], study["V_plain"]
    s = np.linspace(0.3, 8, 200)

    # figure 1: the loss of IoU against the size, formula and OCCT
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for k in KINDS:
        ax.plot(s, [1 - iou_ignoring(k, x, V, l, w, d) for x in s], color=COLOR[k], lw=2, label=LABEL[k] + ", formula")
        pts = [r for r in rows if r["kind"] == k]
        ax.plot([r["size"] for r in pts], [1 - r["iou_occt"] for r in pts], MARKER[k], color=COLOR[k], ms=6,
                mec="white", mew=1.2, ls="none", label=LABEL[k] + ", OCCT")
        ax.annotate(LABEL[k], (s[-1], 1 - iou_ignoring(k, s[-1], V, l, w, d)), xytext=(4, 0), textcoords="offset points",
                    va="center", color=INK, fontsize=8)
    ax.axhline(0.005, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("the loop's stopping tolerance 0.005", (0.31, 0.005), xytext=(0, 3), textcoords="offset points",
                color=MUTED, fontsize=8)
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.set_xlim(0.3, 14), ax.set_xticks([0.5, 1, 2, 3, 4, 6]), ax.set_xticklabels(["0.5", "1", "2", "3", "4", "6"])
    ax.set_xlabel("leg of the chamfer / radius of the fillet (mm)"), ax.set_ylabel("1 − IoU of the plain model")
    ax.set_title("Ignoring a finished edge costs IoU with the square of its size")
    ax.legend(ncol=3, fontsize=7, loc="upper left", bbox_to_anchor=(0, -0.2), handlelength=1.6, columnspacing=1.0)
    fig.savefig(folder / "iou_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # figure 2: what a finished edge adds to the description, and what it adds to the grade (two panels, own axes)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0))
    ax = axes[0]
    sizes = sorted({r["size"] for r in rows})
    plain_faces = 11
    for k in KINDS:
        pts = sorted((r for r in rows if r["kind"] == k), key=lambda r: r["size"])
        ax.plot([0] + [r["size"] for r in pts], [plain_faces] + [r["faces"] for r in pts], MARKER[k] + "-", color=COLOR[k],
                lw=2, ms=6, mec="white", mew=1.2, drawstyle="steps-pre", label=LABEL[k])
    ax.set_ylim(0, 18), ax.set_xlabel("size (mm)"), ax.set_ylabel("faces of the part")
    ax.set_title("Faces: a step at any size")
    ax.annotate("plain: 11", (0, 11), xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK)
    ax.annotate("finished: 15, all three kinds", (sizes[-1], 15), xytext=(0, 6), textcoords="offset points", fontsize=8,
                color=INK, ha="right")
    ax = axes[1]
    for k in KINDS:
        pts = sorted((r for r in rows if r["kind"] == k), key=lambda r: r["size"])
        ax.plot([0] + [r["size"] for r in pts], [0] + [100 * (1 - r["iou_occt"]) for r in pts], MARKER[k] + "-",
                color=COLOR[k], lw=2, ms=6, mec="white", mew=1.2, label=LABEL[k])
        ax.annotate(LABEL[k], (pts[-1]["size"], 100 * (1 - pts[-1]["iou_occt"])), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xlim(-0.2, 9.6), ax.set_xlabel("size (mm)"), ax.set_ylabel("IoU lost by ignoring it (%)")
    ax.set_title("Grade: grows from zero")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(folder / "complexity_vs_grade.pdf", bbox_inches="tight")
    plt.close(fig)

    # figure 3: is the chamfer in the voxels at all
    fig, ax = plt.subplots(figsize=(6.3, 3.2))
    x = np.linspace(0, 2.6, 200)
    ax.plot(x, p_seen(x), color=COLOR["rim_chamfer"], lw=2, label="formula")
    ax.plot(study["seen"]["ratio"], study["seen"]["simulated"], "o", color=COLOR["rim_chamfer"], ms=6, mec="white", mew=1.2,
            ls="none", label="200 000 random grid placements each")
    for xv, txt in ((1.0, "leg = pitch: 1 in 2"), (2.0, "leg = 2 pitches: always")):
        ax.axvline(xv, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.annotate(txt, (xv, 0.04), xytext=(4, 0), textcoords="offset points", fontsize=8, color=MUTED)
    ax.set_xlabel("leg of the chamfer / voxel pitch"), ax.set_ylabel("share of placements with a voxel centre\nin the chamfer's cross-section")
    ax.set_ylim(0, 1.05), ax.set_xlim(0, 2.6)
    ax.set_title("Below two pitches the grid may not contain the chamfer at all")
    ax.legend(fontsize=8, loc="center right")
    fig.savefig(folder / "seen.pdf", bbox_inches="tight")
    plt.close(fig)

    # figure 4: the cascade on the parts
    if any("run" in r for r in rows):
        fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.1))
        for k in KINDS:
            pts = sorted((r for r in rows if r["kind"] == k and "run" in r), key=lambda r: r["size"])
            axes[0].plot([r["size"] for r in pts], [r["run"]["ops"] for r in pts], MARKER[k] + "-", color=COLOR[k], lw=2, ms=6,
                         mec="white", mew=1.2, label=LABEL[k])
            axes[1].plot(s, [100 * (1 - iou_ignoring(k, x, V, l, w, d)) for x in s[s <= 6.2]] + [np.nan] * int((s > 6.2).sum()),
                         color=COLOR[k], lw=1, ls=(0, (4, 3)))
            axes[1].plot([r["size"] for r in pts], [100 * (1 - r["run"]["exact_iou"]) for r in pts], MARKER[k] + "-",
                         color=COLOR[k], lw=2, ms=6, mec="white", mew=1.2, label=LABEL[k])
        axes[0].axhline(2, color=MUTED, lw=1, ls=(0, (4, 3)))
        axes[0].annotate("plain part: 2 operations", (0.5, 2), xytext=(0, 4), textcoords="offset points", fontsize=8, color=MUTED)
        axes[0].set_xlabel("size (mm)"), axes[0].set_ylabel("operations of the program")
        axes[0].set_ylim(0, None), axes[0].set_title("Operations found")
        axes[1].set_xlabel("size (mm)"), axes[1].set_ylabel("1 − exact IoU of the result (%)")
        axes[1].set_title("Result (solid) and plain model (dashed)")
        axes[0].legend(fontsize=7, loc="upper left")
        fig.tight_layout()
        fig.savefig(folder / "cascade.pdf", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
