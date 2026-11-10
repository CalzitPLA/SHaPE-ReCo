#!/usr/bin/env python3
"""
One sheet of nephograms for the frame review (scripts/review_frames.py): every assembly's INPUT surface, coloured by its
distance to the reconstructed model, all on one colour scale, so that the frames can be compared at a glance.

    python3 scripts/review_frames_sheet.py union [--out ~/shape_reco_results/frame_review] [--vmax 100]

A missing member is dark over its whole length; a missing fillet is a faint strip. Distances above `vmax` are drawn at
the dark end. A run without a model is an empty panel that says why.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
sys.path.insert(0, str(ROOT / "scripts"))
SET = ROOT / "dataset_gen" / "examples" / "frame_review"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["union", "multibody"])
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("SHAPE_RECO_RESULTS", Path.home() / "shape_reco_results")) / "frame_review")
    ap.add_argument("--vmax", type=float, default=100.0)
    ap.add_argument("--patches", type=int, default=22000)
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from frame_report import mesh_of
    from frame_report_nephogram import RAMP, bisect, distance, panel
    cmap = LinearSegmentedColormap.from_list("difference", RAMP)
    names = sorted(p.name[:-5] for p in SET.glob("r*.json"))
    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11.0, 7.4))
    gs = fig.add_gridspec(3, 5, left=0.0, right=1.0, top=0.93, bottom=0.08, wspace=-0.05, hspace=0.02)
    rows = []
    for i, name in enumerate(names):
        ax = fig.add_subplot(gs[i // 5, i % 5], projection="3d")
        ax.set_axis_off()
        run_file = a.out / a.mode / name / "run.json"
        title, sub = name, ""
        if not run_file.exists():
            failed = a.out / a.mode / name / "failed.txt"
            sub = "run failed" if failed.exists() else "not run yet"
        else:
            r = json.loads(run_file.read_text())
            g, m = r["grade"], r["match"]
            m_in = mesh_of(SET / f"{name}_union.stl")
            lo, hi = np.asarray(m_in.bounds)
            size = float((hi - lo).max())
            tri = bisect(np.asarray(m_in.triangles, float), size / 110.0, a.patches)
            if g.get("stl") and Path(g["stl"]).exists():
                d = distance(mesh_of(g["stl"]), tri.mean(axis=1))
                iou = g.get("exact_iou")
                sub = (f"IoU {iou:.3f}" if isinstance(iou, float) else "IoU -") + f", {m['operations']} of {m['expected_operations']} ops, " \
                    + ("1 valid solid" if g.get("valid") and g.get("n_solids") == 1 else f"{g.get('n_solids')} solids, not valid" if not g.get("valid") else f"{g.get('n_solids')} solids") \
                    + f"\nlargest distance {d.max():.3g} mm"
                rows.append(dict(name=name, largest=float(d.max()), mean=float(d.mean())))
            else:
                d = np.full(len(tri), a.vmax)
                sub = f"no model: {m['operations']} operations"
            view = (22, -58) if name != "r06_ladder_rhs" else (18, -40)
            panel(ax, tri, d, a.vmax, cmap, view, (lo, hi))
        ax.text2D(0.5, 0.97, title, transform=ax.transAxes, ha="center", fontsize=8, color="#1F1F1F")
        ax.text2D(0.5, 0.945, sub, transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color="#6B6B6B")
    cax = fig.add_axes([0.3, 0.045, 0.4, 0.014])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, a.vmax), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_label(f"distance from the input's surface to the model (mm); {a.vmax:g} and more at the dark end", fontsize=8, color="#1F1F1F")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=7, color="#6B6B6B", labelcolor="#6B6B6B", width=0.4)
    fig.text(0.5, 0.965, f"Frame review, {a.mode}: the input's surface coloured by its distance to the model", ha="center", fontsize=10.5, color="#1F1F1F")
    out = a.out / f"nephograms_{a.mode}.png"
    fig.savefig(out, dpi=200)
    fig.savefig(a.out / f"nephograms_{a.mode}.pdf")
    print(out)


if __name__ == "__main__":
    main()
