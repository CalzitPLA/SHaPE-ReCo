#!/usr/bin/env python3
"""
Nephograms (colour maps) of the closest-point field on a representative slice of each base shape: box, sphere, cylinder,
tube, frustum, cone (the analytic solids of scripts/energetic_descriptors.py, the dimensions of the example parts).

Slice: the plane y = 0 (it contains the axis of every body of revolution, and the box's 20 x 12 mm section). Columns:
  1  signed distance f (mm, negative inside), contour lines every 2 mm
  2  dirlog: log distance l (colour) and the unit normal n = grad f (arrows) - what the network is given
  3  mean curvature H = (k1 + k2)/2 of the surface point c(x) (1/mm), from the Jacobian of the closest-point map;
     constant along each normal, so the whole field "shows" the curvature of the face it points to; grey where c(x) is
     an edge or vertex (rank-collapsed)
  4  the class of c(x) (plane / cyl / dome / saddle / edge / vertex, scripts/field_atlas.py), with the medial axis
     drawn in black: where the unit normal grad f turns by more than 30 degrees between neighbouring samples (it jumps
     there; near the centre of a sphere it only turns fast, by pitch/(r - d) per sample, so the core is not flagged)
All from the analytic SDF (equal to the FreeCAD mesh field to 0.06 %, docs/data/energetic_descriptors.json).

    python3 scripts/field_slices.py      # -> docs/slides/generated/fa/fa_slice_<shape>.pdf, docs/thesis/generated/fa_slices.pdf/.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
sys.path.insert(0, str(ROOT / "scripts"))

import field_atlas as fa  # noqa: E402
from energetic_descriptors import SHAPES  # noqa: E402

HALF = 16.0          # mm, the slice window is [-HALF, HALF]^2 in (x, z)
N = 181              # samples per side
CLASS_COLOURS = ("#b4b3ad", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")   # fa.CLASSES order


def slice_fields(prim, n: int = N, half: float = HALF, h: float = 0.02) -> dict:
    u = np.linspace(-half, half, n)
    X, Z = np.meshgrid(u, u, indexing="xy")
    pts = np.stack([X.ravel(), np.zeros(X.size), Z.ravel()], axis=1)
    f = prim.sdf(pts)
    closest = fa.analytic_closest(prim)
    d, lam, kappa, _ = fa.spectrum(closest, pts, h)
    lab = fa.classify(lam, kappa)
    eps = 1e-5
    g = np.stack([(prim.sdf(pts + e) - prim.sdf(pts - e)) / (2 * eps) for e in np.eye(3) * eps], axis=1)
    g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
    G = g.reshape(n, n, 3)
    cos_min = np.ones((n, n))
    for axis in (0, 1):
        dot = (np.take(G, range(1, n), axis=axis) * np.take(G, range(0, n - 1), axis=axis)).sum(-1)
        sl_a = [slice(None)] * 2
        sl_b = [slice(None)] * 2
        sl_a[axis], sl_b[axis] = slice(1, None), slice(0, -1)
        cos_min[tuple(sl_a)] = np.minimum(cos_min[tuple(sl_a)], dot)
        cos_min[tuple(sl_b)] = np.minimum(cos_min[tuple(sl_b)], dot)
    medial = (cos_min < np.cos(np.radians(30.0))).ravel()
    face = (lam > fa.RANK_LAMBDA).all(axis=1) & ~medial
    H = np.where(face, 0.5 * np.nan_to_num(kappa).sum(axis=1), np.nan)
    R = 2 * half / (n - 1)                                               # slice sample pitch, the "voxel" of l
    ell = np.sign(f) * np.log1p(np.abs(f) / R) / np.log1p(n / 1.0)
    cls = np.array([fa.CLASSES.index(c) for c in lab], float)
    cls[medial] = np.nan
    shape = X.shape
    return dict(X=X, Z=Z, f=f.reshape(shape), ell=ell.reshape(shape), gx=g[:, 0].reshape(shape), gz=g[:, 2].reshape(shape),
                H=H.reshape(shape), cls=cls.reshape(shape), medial=medial.reshape(shape))


def draw(names, out_paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, TwoSlopeNorm, BoundaryNorm
    plt.rcParams.update({"font.size": 7, "axes.edgecolor": "#d9d8d4", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(len(names), 4, figsize=(8.0, 2.0 * len(names) + (0.6 if len(names) > 1 else 0.45)), squeeze=False)
    titles = ("signed distance $f$ [mm]", "dirlog: $\\ell$ and $\\hat n$", "mean curvature $H$ of $c(x)$ [1/mm]", "class of $c(x)$, medial axis")
    for i, name in enumerate(names):
        prim = SHAPES[name][1]
        s = slice_fields(prim)
        X, Z = s["X"], s["Z"]
        ext = (-HALF, HALF, -HALF, HALF)
        ax = axes[i, 0]
        lim = float(np.abs(s["f"]).max())
        im = ax.imshow(s["f"], origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -lim, lim))
        ax.contour(X, Z, s["f"], levels=np.arange(-12, 14, 2), colors="#333333", linewidths=0.35)
        ax.contour(X, Z, s["f"], levels=[0], colors="black", linewidths=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax = axes[i, 1]
        im = ax.imshow(s["ell"], origin="lower", extent=ext, cmap="RdBu_r", norm=TwoSlopeNorm(0.0, -1.0, 1.0))
        k = 12
        ax.quiver(X[::k, ::k], Z[::k, ::k], s["gx"][::k, ::k], s["gz"][::k, ::k], color="#222222", scale=22, width=0.006,
                  headwidth=3.5)
        ax.contour(X, Z, s["f"], levels=[0], colors="black", linewidths=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax = axes[i, 2]
        Hm = np.ma.masked_invalid(s["H"])
        # robust scale: the 98th percentile of |H| (finite-difference outliers next to the medial axis would wash it out)
        hl = max(float(np.nanpercentile(np.abs(s["H"]), 98)) if np.isfinite(s["H"]).any() else 0.0, 0.05)
        ax.set_facecolor("#e4e3de")
        im = ax.imshow(np.clip(Hm, -hl, hl), origin="lower", extent=ext, cmap="PuOr_r", norm=TwoSlopeNorm(0.0, -hl, hl))
        ax.contour(X, Z, s["f"], levels=[0], colors="black", linewidths=1.0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        ax = axes[i, 3]
        cmap = ListedColormap(CLASS_COLOURS)
        ax.set_facecolor("white")
        ax.imshow(np.ma.masked_invalid(s["cls"]), origin="lower", extent=ext, cmap=cmap,
                  norm=BoundaryNorm(np.arange(-0.5, 6.5, 1.0), cmap.N), interpolation="nearest")
        my, mx = np.nonzero(s["medial"])
        ax.scatter(X[my, mx], Z[my, mx], s=0.6, color="black", linewidths=0)
        ax.contour(X, Z, s["f"], levels=[0], colors="black", linewidths=1.0)
        for j in range(4):
            axes[i, j].set_xticks([-10, 0, 10])
            axes[i, j].set_yticks([-10, 0, 10])
            if i == 0:
                axes[i, j].set_title(titles[j], fontsize=7.5)
        axes[i, 0].set_ylabel(f"{name}\nz [mm]", fontsize=8)
    for j in range(4):
        axes[-1, j].set_xlabel("x [mm]")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in CLASS_COLOURS] + [plt.Line2D([], [], color="black", lw=1.5)]
    fig.legend(handles, list(fa.CLASSES) + ["medial axis / surface"], ncol=7, loc="lower center", fontsize=7,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, min(0.035 * 3 / len(names), 0.12), 1, 1))
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=170)
    plt.close(fig)


def main():
    slides = ROOT / "docs/slides/generated/fa"
    thesis = ROOT / "docs/thesis/generated"
    for name in SHAPES:                                                    # one full-width row per slide
        draw((name,), [slides / f"fa_slice_{name}.pdf"])
    draw(tuple(SHAPES), [thesis / "fa_slices.pdf", thesis / "fa_slices.png"])
    print("wrote", sorted(p.name for p in slides.glob("fa_slice_*.pdf")), thesis / "fa_slices.pdf")


if __name__ == "__main__":
    main()
