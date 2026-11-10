#!/usr/bin/env python3
"""The CSG trees of the IoU report drawn as trees, and the per-tree summary of their measured effects.

    python3 scripts/iou_tree_figures.py [--out docs/reports/iou_two_stage]

Needs the data of scripts/iou_sobolev_study.py (data/exact.json) and scripts/iou_stage_study.py (data/stages.json,
data/stages_exact.json). Writes fig/tree_*.png, fig/tree_effects.pdf and data/tree_*.tex. No new measurement: every
number is read from those files.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import iou_sobolev_study as one                                      # noqa: E402
import iou_stage_study as st                                         # noqa: E402

GREY, GHOST = "#9a9992", "#c9c8c1"
FUSE, CUT = one.SERIES[0], one.SERIES[1]
EDIT = one.SERIES[7]                                                 # the node an error edits


# ------------------------------------------------------------------------------------------------ drawing


def thumb(fig, rect, man, color, ghost=None, lo=(-2, -2, -2), hi=(62, 42, 36), rplus=None, rminus=None):
    ax = fig.add_axes(rect, projection="3d")
    if ghost is not None:
        one.draw_mesh(ax, ghost, GHOST, 0.12)
    if man is not None:
        one.draw_mesh(ax, man, color, 1.0)
    if rplus is not None:
        one.draw_mesh(ax, rplus, one.RPLUS, 0.95)
    if rminus is not None:
        one.draw_mesh(ax, rminus, one.RMINUS, 0.95)
    one.frame3d(ax, lo, hi)
    return ax


def draw_tree(fig, x0, y0, w, h, leaf1, leaf2, result, op, lab1, lab2, title, edit=None, missing=False,
              rplus=None, rminus=None, result_note="", result_ghost=None):
    """one CSG tree in the figure box (x0, y0, w, h) (figure fractions): result on top, the boolean node, two leaves.
    edit: "leaf1" | "leaf2" | "root" - the node a wrong model changes, drawn in the error colour. The 3D thumbnails are
    drawn first, the 2D overlay (edges, node, labels) last, so the labels stay on top."""
    tw, th = 0.40 * w, 0.30 * h
    R, O, L1, L2 = (0.5, 0.77), (0.5, 0.49), (0.25, 0.27), (0.75, 0.27)          # centres in the box's 0..1 coords
    box_man = st.box_manifold()
    thumb(fig, [x0 + L1[0] * w - tw / 2, y0 + L1[1] * h - th / 2, tw, th], leaf1, EDIT if edit == "leaf1" else GREY)
    if leaf2 is not None:
        thumb(fig, [x0 + L2[0] * w - tw / 2, y0 + L2[1] * h - th / 2, tw, th], leaf2,
              GHOST if missing else (EDIT if edit == "leaf2" else (FUSE if op == "fuse" else CUT)), ghost=box_man)
    thumb(fig, [x0 + R[0] * w - tw / 2, y0 + R[1] * h - th / 2, tw, th], result, GREY, ghost=result_ghost,
          rplus=rplus, rminus=rminus)
    ov = fig.add_axes([x0, y0, w, h]); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")
    ov.patch.set_alpha(0)
    dash = (0, (3, 2))
    ov.plot([O[0], O[0]], [O[1] + 0.05, R[1] - 0.17], color=one.MUTED, lw=1.0, zorder=1)
    ov.plot([O[0], L1[0] + 0.08], [O[1] - 0.04, L1[1] + 0.12], color=one.MUTED, lw=1.0, zorder=1)
    ov.plot([O[0], L2[0] - 0.08], [O[1] - 0.04, L2[1] + 0.12], color=one.MUTED, lw=1.0, zorder=1,
            ls=dash if missing else "-")
    ov.scatter([O[0]], [O[1]], s=330, facecolor="white", edgecolor=FUSE if op == "fuse" else CUT, linewidths=1.6,
               linestyle="--" if missing else "-", zorder=3)
    ov.text(O[0], O[1], "$\\cup$" if op == "fuse" else "$\\setminus$", ha="center", va="center", fontsize=10,
            color=one.INK, zorder=4)
    ov.text(O[0] + 0.07, O[1], op, ha="left", va="center", fontsize=6.5, color=one.MUTED)
    ov.text(0.5, 0.995, title, ha="center", va="top", fontsize=7.5, color=one.INK, weight="bold")
    if result_note:
        ov.text(0.5, R[1] - 0.155, result_note, ha="center", va="top", fontsize=6.0, color=one.INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    ov.text(L1[0], 0.03, lab1, ha="center", va="bottom", fontsize=5.6, color=EDIT if edit == "leaf1" else one.INK)
    ov.text(L2[0], 0.03, lab2, ha="center", va="bottom", fontsize=5.6,
            color=one.MUTED if missing else (EDIT if edit == "leaf2" else one.INK))
    ov.text(0.02, L1[1] + 0.12, "stage 1", fontsize=5.8, color=one.MUTED)
    ov.text(0.98, L2[1] + 0.12, "stage 2", fontsize=5.8, color=one.MUTED, ha="right")
    if edit == "root":
        ov.add_patch(__import__("matplotlib").patches.FancyBboxPatch((0.03, 0.08), 0.94, 0.52, boxstyle="round,pad=0.01",
                                                                      fill=False, edgecolor=EDIT, lw=1.0, ls="--"))
        ov.text(0.97, 0.57, "whole tree moved", fontsize=5.8, color=EDIT, ha="right", va="top")
    return ov


def prim_label(p):
    if p["kind"] == "box":
        s = "x".join(f"{v:g}" for v in p["size"])
        return f"box {s}\nat ({', '.join(f'{v:g}' for v in p['lo'])})"
    return f"cylinder r={p['r']:g} h={p['h']:g}\nat ({', '.join(f'{v:g}' for v in p['base'])})"


def tool_label(t):
    k, c = t["kind"], st.CANON[t["kind"]]
    s = t["scale"]
    txt = dict(box=lambda: f"box {c['a'] * s:.3g}x{c['b'] * s:.3g}x{c['h'] * s:.3g}",
               cylinder=lambda: f"cylinder r={c['r'] * s:.3g} h={c['h'] * s:.3g}",
               cone=lambda: f"cone r={c['r1'] * s:.3g}->{c['r2'] * s:.3g} h={c['h'] * s:.3g}",
               sphere=lambda: f"sphere r={c['r'] * s:.3g}",
               torus=lambda: f"torus R={c['R'] * s:.3g} r={c['r'] * s:.3g}")[k]()
    extra = []
    if t["tilt"]:
        extra.append(f"tilt {t['tilt']:g} deg")
    if t["shift"]:
        extra.append(f"+{t['shift']:g} mm x")
    return txt + "\nat (30, 20, 20)" + (", " + ", ".join(extra) if extra else "")


def prim_manifold(p):
    return one.manifold_of(dict(stage1=p, op="fuse", stage2=None))


# ------------------------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/reports/iou_two_stage"))
    out = Path(ap.parse_args().out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    one.style(plt)
    ex1 = json.loads((out / "data" / "exact.json").read_text())
    rows = json.loads((out / "data" / "stages.json").read_text())
    ex2 = json.loads((out / "data" / "stages_exact.json").read_text())

    # ---- (1) the four trees of the first study
    fig = plt.figure(figsize=(7.4, 5.6))
    for i, (name, part) in enumerate(one.PARTS.items()):
        x0, y0 = (i % 2) * 0.5, 0.5 - (i // 2) * 0.5
        e = ex1[f"{name}|exact"]
        n_sol = len(one.manifold_of(part).decompose())
        draw_tree(fig, x0, y0, 0.5, 0.5, prim_manifold(part["stage1"]), prim_manifold(part["stage2"]),
                  one.manifold_of(part), part["op"], prim_label(part["stage1"]), prim_label(part["stage2"]),
                  f"{name}", result_note=f"V = {e['input_volume']:.0f} mm$^3$, genus {ex1[f'{name}|genus']}, "
                                         f"{n_sol} solid{'s' if n_sol > 1 else ''}")
    fig.savefig(out / "fig" / "tree_part1.png"); plt.close(fig)

    # ---- (2) the ten trees of the stage study
    fig = plt.figure(figsize=(7.4, 10.4))
    for i, kind in enumerate(st.KINDS):
        for j, op in enumerate(st.OPS):
            p = f"{kind}-{op}"
            x0, y0 = j * 0.5, 1 - (i + 1) * 0.2
            e = ex2[f"{p}|exact|0"]
            piece = next(r for r in rows if r["part"] == p and r["stage"] == "missing")["piece_volume"]
            draw_tree(fig, x0, y0, 0.5, 0.2, st.box_manifold(), st.tool_manifold(st.tool(kind)),
                      st.part_manifold(op, st.tool(kind)), op, "box 60x40x20\nat (0, 0, 0)",
                      tool_label(st.tool(kind)), p,
                      result_note=f"V = {e['input_volume']:.0f} mm$^3$, piece {piece:.0f} mm$^3$, genus {e['model_genus']}")
    fig.savefig(out / "fig" / "tree_part2.png"); plt.close(fig)

    # ---- (3) where each stage's error sits in the tree: cylinder cut
    A = st.part_manifold("cut", st.tool("cylinder"))
    muts = [("exact", "the input (correct tree)", st.tool("cylinder"), None),
            ("missing", "missing stage", None, "leaf2"),
            ("type", "TYPE: box of equal volume", st.same_volume("box", "cylinder"), "leaf2"),
            ("rotation", "ROTATION: tilt 10 deg", st.tool("cylinder", tilt=10), "leaf2"),
            ("dimensions", "DIMENSIONS: x1.1", st.tool("cylinder", scale=1.1), "leaf2"),
            ("position", "POSITION: +2 mm", st.tool("cylinder", shift=2), "leaf2")]
    mags = dict(exact=0, missing=0, type="box", rotation=10, dimensions=1.1, position=2)
    fig = plt.figure(figsize=(7.4, 5.2))
    for i, (stage, title, t, edit) in enumerate(muts):
        x0, y0 = (i % 3) / 3, 0.5 - (i // 3) * 0.5
        B = st.part_manifold("cut", t)
        r = next(r for r in rows if r["part"] == "cylinder-cut" and r["stage"] == stage and str(r["mag"]) == str(mags[stage]))
        note = "" if stage == "exact" else f"IoU part {1 - r['1-IoU']:.4f}, piece {1 - r['1-IoU piece']:.3f}"
        leaf2 = st.tool_manifold(t) if t is not None else st.tool_manifold(st.tool("cylinder"))
        draw_tree(fig, x0, y0, 1 / 3, 0.5, st.box_manifold(), leaf2, B if stage == "exact" else None, "cut",
                  "box 60x40x20", tool_label(t if t is not None else st.tool("cylinder")).split("\n")[0]
                  + ("" if t is None else "\n" + tool_label(t).split("\n")[1]),
                  title, edit=None if stage in ("exact", "missing") else edit, missing=(stage == "missing"),
                  rplus=None if stage == "exact" else A - B, rminus=None if stage == "exact" else B - A,
                  result_note=note, result_ghost=None if stage == "exact" else A)
    fig.savefig(out / "fig" / "tree_mutations.png"); plt.close(fig)

    # ---- (4) the four trees of study 1 with their root-level errors: which node the error edits
    fig = plt.figure(figsize=(7.4, 2.9))
    part = one.PARTS["hole"]
    A1 = one.manifold_of(part)
    cases = [("tool moved 2 mm", "leaf2", one.models_of(part)["tool moved 2 mm"]),
             ("all moved 1 mm", "root", one.models_of(part)["all moved 1 mm"]),
             ("all turned 2 deg", "root", one.models_of(part)["all turned 2 deg"])]
    for i, (name, edit, (tree, shift, turn)) in enumerate(cases):
        B = one.manifold_of(tree, shift, turn)
        iou = ex1[f"hole|{name}"]["iou"]
        draw_tree(fig, i / 3, 0, 1 / 3, 1, prim_manifold(tree["stage1"]), prim_manifold(tree["stage2"]), None, "cut",
                  prim_label(tree["stage1"]).split("\n")[0], prim_label(tree["stage2"]).split("\n")[0]
                  + ("\n+2 mm x" if edit == "leaf2" else ""),
                  f"hole: {name}", edit=edit, rplus=A1 - B, rminus=B - A1, result_note=f"IoU {iou:.4f}", result_ghost=A1)
    fig.savefig(out / "fig" / "tree_root_vs_leaf.png"); plt.close(fig)

    # ---- (5) effects: every tree x every error, part IoU and piece IoU
    parts = [f"{k}-{o}" for k in st.KINDS for o in st.OPS]
    cols = [("missing", 0)] + [("type", k) for k in st.KINDS] + \
           [(s, v) for s in ("rotation", "dimensions", "position") for v in st.SWEEPS[s]]
    col_lab = ["missing"] + [f"T:{k[:3]}" for k in st.KINDS] + [f"R {v}°" for v in st.SWEEPS["rotation"]] + \
              [f"D x{v}" for v in st.SWEEPS["dimensions"]] + [f"P {v}" for v in st.SWEEPS["position"]]
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 6.6))
    for ax, key, name in ((axs[0], "1-IoU", "IoU of the whole part (exact)"), (axs[1], "1-IoU piece", "IoU of the piece")):
        M = np.full((len(parts), len(cols)), np.nan)
        for i, p in enumerate(parts):
            for j, (s, v) in enumerate(cols):
                r = [r for r in rows if r["part"] == p and r["stage"] == s and str(r["mag"]) == str(v)]
                if r:
                    M[i, j] = 1 - r[0][key]
        vmax = 3.0 if key == "1-IoU" else 1.5              # -log10(1 - IoU): 3 = 0.999, 1.5 = 0.968
        ax.imshow(-np.log10(np.clip(1 - M, 1e-4, 1)), cmap="Blues_r", aspect="auto", vmin=0, vmax=vmax)
        for i, p in enumerate(parts):
            miss = M[i, 0]
            for j in range(len(cols)):
                if np.isnan(M[i, j]):
                    ax.text(j, i, "·", ha="center", va="center", fontsize=6, color=one.MUTED)
                    continue
                v = M[i, j]
                dark = -np.log10(max(1 - v, 1e-4)) < 0.45 * vmax
                ax.text(j, i, f"{v:.3f}"[1:] if v < 0.9995 else "1", ha="center", va="center", fontsize=4.6,
                        color="white" if dark else one.INK)
                if j > 0 and v < miss:
                    ax.add_patch(__import__("matplotlib").patches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                                            edgecolor=one.SERIES[1], lw=1.2))
        ax.set_yticks(range(len(parts))); ax.set_yticklabels(parts, fontsize=6.5)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(col_lab, fontsize=5.8, rotation=60, ha="right")
        ax.set_title(name + ": orange frame = worse than leaving stage 2 out", fontsize=8)
        ax.grid(False)
        for b in (0.5, 5.5, 11.5, 15.5):
            ax.axvline(b, color="white", lw=2)
    fig.tight_layout()
    fig.savefig(out / "fig" / "tree_effects.pdf"); plt.close(fig)

    # ---- per-tree summary table
    L = []
    for p in parts:
        R = [r for r in rows if r["part"] == p]
        miss = next(r for r in R if r["stage"] == "missing")
        worse = [r for r in R if r["stage"] not in ("exact", "missing") and r["1-IoU"] > miss["1-IoU"]]
        pos2 = next(r for r in R if r["stage"] == "position" and r["mag"] == 2)
        rot = [r for r in R if r["stage"] == "rotation"]
        typ = [r for r in R if r["stage"] == "type"]
        fail = [f"{r['stage'][0].upper()}{r['mag']}" for r in R if r.get("exact_failed")]
        L.append(f"{p} & {ex2[f'{p}|exact|0']['input_volume']:.0f} & {miss['piece_volume']:.0f} & "
                 f"{1 - miss['1-IoU']:.4f} & {pos2['1-IoU piece'] / pos2['1-IoU']:.1f} & "
                 f"{min(1 - r['1-IoU'] for r in typ):.4f}--{max(1 - r['1-IoU'] for r in typ):.4f} & "
                 f"{min(1 - r['1-IoU'] for r in rot):.4f} & {len(worse)} & {', '.join(fail) if fail else '--'} \\\\")
    (out / "data" / "tree_summary.tex").write_text("\n".join(L + ["\\bottomrule"]) + "\n")
    print("done")


if __name__ == "__main__":
    main()
