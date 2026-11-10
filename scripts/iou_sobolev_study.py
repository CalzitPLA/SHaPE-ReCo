#!/usr/bin/env python3
"""The IoU on two-stage CSG trees, and what changes when the shape function is measured in another Sobolev space.

    python3 scripts/iou_sobolev_study.py [--out docs/reports/iou_two_stage] [--pitch 0.5] [--no-occt]

Four parts, each one boolean of two Part-native primitives (stage 1: a box; stage 2: a cylinder or a box, fused or
cut), and for each part a family of wrong models (the tree cut after stage 1, the stage-2 tool moved or resized, the
whole model moved or turned). Every pair is measured

  - by the project's own judges: the exact OCCT residual (`reco_field.exact_check.exact_residual`, freecadcmd), the
    voxel IoU of `reco_engine.compare` on parity grids (`dataset_gen/src/field_channels.parity_axes`) at several
    pitches and grid offsets, and the rim-forgiving IoU of `reco_engine.cascade._Context.iou` (re-implemented here with
    the same formula, cascade.py:259-274, because the cascade's context wants CNN channels);
  - by Sobolev distances of two shape functions on one fine grid: the indicator chi (in H^s only for s < 1/2) and the
    clipped signed distance phi (in H^s only for s < 3/2), ||u||_{H^s}^2 = int (1 + l^2 |xi|^2)^s |u^(xi)|^2 dxi, l fixed.

Writes data/*.json, the report's tables (data/*.tex) and figures (fig/*.pdf, fig/*.png). Asserts the consistency
checks it relies on (exact IoU of the identity = 1, voxel IoU -> exact IoU as the pitch shrinks, D_0^2 = |A^B| / |A|).
"""
import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # dataviz palette
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
RPLUS, RMINUS = "#e34948", "#2a78d6"              # missed material (r+) / invented material (r-)
ELL = 4.0                                         # mm: the length scale of the Sobolev weight
CLIP = 4.0                                        # mm: the signed distance is clipped to +-CLIP

# ------------------------------------------------------------------------------------------------ the trees
# a primitive: dict(kind="box", lo=(x,y,z), size=(dx,dy,dz)) or dict(kind="cyl", base=(x,y,z), r=, h=) (axis +z)
# a tree: dict(stage1=prim, op="fuse"|"cut", stage2=prim); a model: a tree (possibly without stage2) + a rigid motion

BOX = dict(kind="box", lo=(0, 0, 0), size=(60, 40, 20))
CENTRE = (30.0, 20.0, 10.0)

PARTS = {
    "boss": dict(title="box $\\cup$ cylinder (boss)", stage1=BOX, op="fuse",
                 stage2=dict(kind="cyl", base=(30, 20, 20), r=8.0, h=15.0)),
    "hole": dict(title="box $\\setminus$ cylinder (through hole)", stage1=BOX, op="cut",
                 stage2=dict(kind="cyl", base=(30, 20, -1), r=8.0, h=22.0)),
    "slot": dict(title="box $\\setminus$ box (2 mm slot, splits)", stage1=BOX, op="cut",
                 stage2=dict(kind="box", lo=(29, -1, -1), size=(2, 42, 22))),
    "tray": dict(title="box $\\setminus$ box (tray, 2 mm walls)", stage1=BOX, op="cut",
                 stage2=dict(kind="box", lo=(2, 2, 2), size=(56, 36, 19))),
}


def moved(p, d):
    q = dict(p)
    key = "lo" if p["kind"] == "box" else "base"
    q[key] = tuple(float(a + b) for a, b in zip(p[key], d))
    return q


def grown(p, g):
    """stage 2 made bigger by g (a cylinder's radius; a box on every side it has in x and y)."""
    q = dict(p)
    if p["kind"] == "cyl":
        q["r"] = p["r"] + g
    else:
        q["lo"] = (p["lo"][0] - g, p["lo"][1] - g, p["lo"][2])
        q["size"] = (p["size"][0] + 2 * g, p["size"][1] + 2 * g, p["size"][2])
    return q


def models_of(part):
    """The wrong models of one part: name -> (tree, shift, turn_deg)."""
    t = dict(part)
    return {
        "exact": (t, (0, 0, 0), 0.0),
        "stage 1 only": (dict(t, stage2=None), (0, 0, 0), 0.0),
        "tool moved 2 mm": (dict(t, stage2=moved(t["stage2"], (2, 0, 0))), (0, 0, 0), 0.0),
        "tool +1 mm": (dict(t, stage2=grown(t["stage2"], 1.0)), (0, 0, 0), 0.0),
        "all moved 1 mm": (t, (1, 0, 0), 0.0),
        "all turned 2 deg": (t, (0, 0, 0), 2.0),
    }


MODEL_ORDER = ["exact", "stage 1 only", "tool moved 2 mm", "tool +1 mm", "all moved 1 mm", "all turned 2 deg"]

# ------------------------------------------------------------------------------------------------ meshes (manifold3d)


def manifold_of(tree, shift=(0, 0, 0), turn=0.0, segments=180):     # 2 deg per facet: a 2 deg turn maps the polygon onto itself
    import manifold3d as m3

    def prim(p):
        if p["kind"] == "box":
            return m3.Manifold.cube(list(map(float, p["size"]))).translate(list(map(float, p["lo"])))
        return m3.Manifold.cylinder(float(p["h"]), float(p["r"]), float(p["r"]), segments).translate(
            list(map(float, p["base"])))

    s = prim(tree["stage1"])
    if tree.get("stage2") is not None:
        s = s + prim(tree["stage2"]) if tree["op"] == "fuse" else s - prim(tree["stage2"])
    if turn:
        c = np.array(CENTRE)
        s = s.translate(list(-c)).rotate([0.0, 0.0, float(turn)]).translate(list(c))
    return s.translate(list(map(float, shift)))


def mesh_arrays(man):
    m = man.to_mesh()
    return np.asarray(m.vert_properties, float)[:, :3], np.asarray(m.tri_verts, np.int64)


# ------------------------------------------------------------------------------------------------ STEP (freecadcmd)

FC_SCRIPT = r'''
import json, os, Part, FreeCAD as App
job = json.load(open(os.environ["IOU_JOB"]))
def prim(p):
    if p["kind"] == "box":
        return Part.makeBox(*p["size"], App.Vector(*p["lo"]))
    return Part.makeCylinder(p["r"], p["h"], App.Vector(*p["base"]), App.Vector(0, 0, 1))
for item in job["items"]:
    t = item["tree"]
    s = prim(t["stage1"])
    if t.get("stage2"):
        s = s.fuse(prim(t["stage2"])) if t["op"] == "fuse" else s.cut(prim(t["stage2"]))
        s = s.removeSplitter()
    if item["turn"]:
        s.rotate(App.Vector(*job["centre"]), App.Vector(0, 0, 1), item["turn"])
    s.translate(App.Vector(*item["shift"]))
    s.exportStep(item["path"])
'''


def write_steps(items, centre):
    from reco_field.export import find_freecadcmd
    fc = find_freecadcmd()
    with tempfile.TemporaryDirectory() as tmp:
        script, job = Path(tmp) / "build.py", Path(tmp) / "job.json"
        script.write_text(FC_SCRIPT)
        job.write_text(json.dumps(dict(items=items, centre=list(centre))))
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["IOU_JOB"] = str(job)
        subprocess.run([fc, str(script)], env=env, check=True, capture_output=True, timeout=900)
    missing = [i["path"] for i in items if not Path(i["path"]).is_file()]
    assert not missing, missing


# ------------------------------------------------------------------------------------------------ grids


class Grid:
    def __init__(self, lo, hi, pitch, offset=(0.0, 0.0, 0.0)):
        self.pitch = float(pitch)
        self.lo = np.asarray(lo, float) + np.asarray(offset, float) * pitch
        n = np.ceil((np.asarray(hi, float) - self.lo) / pitch).astype(int)
        self.axes = [self.lo[i] + (np.arange(n[i]) + 0.5) * pitch for i in range(3)]
        self.shape = tuple(int(k) for k in n)


def occupancy(man, grid):
    from src.field_channels import parity_axes
    v, f = mesh_arrays(man)
    xs, ys, zs = grid.axes
    p = grid.pitch
    return parity_axes(v, f, xs, ys + 1.2345e-4 * p, zs + 2.3456e-4 * p)       # as parity_occupancy shifts the rows


def voxel_iou(a, b):
    from reco_engine.compare import compare
    return compare(a, b).iou


def rim_forgiving_iou(target, cand):
    """reco_engine/cascade.py:259-274: the mismatch within one voxel of the target's boundary does not count."""
    from scipy import ndimage
    rim = ndimage.binary_dilation(target) & ~ndimage.binary_erosion(target)
    counted = ~rim
    t, o = target & counted, cand & counted
    union = int(np.count_nonzero(t | o))
    return (int(np.count_nonzero(t & o)) / union if union else float("nan")), float(counted[target].mean())


def signed_distance(occ, pitch):
    """positive outside, negative inside, from the voxel centres (error <= pitch / 2), clipped to +-CLIP."""
    from scipy import ndimage
    out = ndimage.distance_transform_edt(~occ) * pitch
    ins = ndimage.distance_transform_edt(occ) * pitch
    phi = np.where(occ, -(ins - 0.5 * pitch), out - 0.5 * pitch)
    return np.clip(phi, -CLIP, CLIP)


class Spectrum:
    """|u^|^2 on the grid's frequencies and the weights (1 + l^2 |xi|^2)^s; Parseval: sum |u|^2 p^3 = p^3/N sum |U|^2."""

    def __init__(self, grid):
        k = [2 * np.pi * np.fft.fftfreq(n, d=grid.pitch) for n in grid.shape]
        kx, ky, kz = np.meshgrid(*k, indexing="ij", sparse=True)
        self.base = 1.0 + ELL ** 2 * (kx ** 2 + ky ** 2 + kz ** 2)
        self.scale = grid.pitch ** 3 / float(np.prod(grid.shape))

    def power(self, u):
        return np.abs(np.fft.fftn(u.astype(np.float64))) ** 2

    def norm2(self, power, s):
        return float((self.base ** s * power).sum() * self.scale)

    def density(self, u, s):
        """|(1 - l^2 Laplace)^{s/2} u|^2: where in space the H^s norm of u sits (it integrates to ||u||_s^2)."""
        return np.abs(np.fft.ifftn(self.base ** (s / 2) * np.fft.fftn(u.astype(np.float64)))) ** 2


S_CHI = np.round(np.linspace(-1.5, 0.45, 40), 4)
S_PHI = np.round(np.linspace(-0.5, 1.45, 40), 4)
S_PICK_CHI = (-1.0, 0.0, 0.45)
S_PICK_PHI = (0.0, 1.0, 1.45)

# ------------------------------------------------------------------------------------------------ plots


def style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "lines.linewidth": 2.0, "savefig.bbox": "tight", "savefig.dpi": 200})


def draw_mesh(ax, man, color, alpha, edge=None, light=(0.4, -0.6, 0.7)):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.colors import to_rgb
    if man.is_empty():
        return
    man = man.refine_to_length(4.0)                 # long thin faces make matplotlib's depth sort fail (dark patches)
    v, f = mesh_arrays(man)
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1)
    keep = area > 1e-9 * area.max()                 # sliver triangles of the booleans have no reliable normal
    tri, n, area = tri[keep], n[keep], area[keep]
    n /= area[:, None]
    L = np.asarray(light, float); L /= np.linalg.norm(L)
    shade = 0.55 + 0.45 * np.clip(n @ L, 0, 1)
    rgb = np.asarray(to_rgb(color))
    fc = np.concatenate([np.clip(rgb[None] * shade[:, None], 0, 1), np.full((len(tri), 1), alpha)], axis=1)
    opaque = alpha >= 0.9                           # opaque faces get edges of their own colour: no antialiasing seams
    pc = Poly3DCollection(tri, facecolors=fc, edgecolors=fc if opaque else "none", linewidths=0.3 if opaque else 0.0)
    ax.add_collection3d(pc)


def frame3d(ax, lo=(-4, -4, -2), hi=(64, 44, 36)):
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.asarray(hi, float) - np.asarray(lo, float))
    ax.view_init(elev=28, azim=-58)
    ax.set_axis_off()


def plot_trees(out, plt):
    """fig 1: the four trees, stage 1, stage 2 and the result."""
    fig = plt.figure(figsize=(7.2, 7.6))
    for i, (name, part) in enumerate(PARTS.items()):
        for j, what in enumerate(("stage 1", "stage 2", "result")):
            ax = fig.add_subplot(4, 3, 3 * i + j + 1, projection="3d")
            if what == "stage 1":
                draw_mesh(ax, manifold_of(dict(part, stage2=None)), "#9a9992", 1.0)
            elif what == "stage 2":
                draw_mesh(ax, manifold_of(dict(part, stage2=None)), "#9a9992", 0.12)
                draw_mesh(ax, manifold_of(dict(stage1=part["stage2"], op="fuse", stage2=None)),
                          SERIES[0] if part["op"] == "fuse" else SERIES[1], 0.95)
            else:
                draw_mesh(ax, manifold_of(part), "#9a9992", 1.0)
            frame3d(ax)
            if i == 0:
                ax.set_title({"stage 1": "stage 1: box", "stage 2": "stage 2: tool (blue fuse, orange cut)",
                              "result": "result = input"}[what], fontsize=8, color=INK)
            if j == 0:
                ax.text2D(-0.08, 0.5, name, transform=ax.transAxes, rotation=90, va="center", color=INK, fontsize=9)
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    fig.savefig(out / "fig" / "trees.png"); plt.close(fig)


def plot_errors(out, plt, table):
    """fig 2: every wrong model, the missed material r+ (red) and the invented material r- (blue) on the input."""
    fig = plt.figure(figsize=(7.4, 6.4))
    cols = MODEL_ORDER[1:]
    for i, (name, part) in enumerate(PARTS.items()):
        A = manifold_of(part)
        for j, mname in enumerate(cols):
            tree, shift, turn = models_of(part)[mname]
            B = manifold_of(tree, shift, turn)
            ax = fig.add_subplot(4, len(cols), len(cols) * i + j + 1, projection="3d")
            draw_mesh(ax, A, "#9a9992", 0.10)
            draw_mesh(ax, A - B, RPLUS, 0.95)
            draw_mesh(ax, B - A, RMINUS, 0.95)
            frame3d(ax)
            row = table[name][mname]
            ax.text2D(0.5, -0.02, f"IoU {row['iou_exact']:.3f}", transform=ax.transAxes, ha="center", fontsize=7,
                      color=INK)
            if i == 0:
                ax.set_title(mname, fontsize=7.5, color=INK)
            if j == 0:
                ax.text2D(-0.1, 0.5, name, transform=ax.transAxes, rotation=90, va="center", color=INK, fontsize=9)
    fig.subplots_adjust(wspace=0.0, hspace=0.08)
    fig.savefig(out / "fig" / "errors.png"); plt.close(fig)


def plot_s_curves(out, plt, table):
    fig, axs = plt.subplots(2, 4, figsize=(7.4, 4.6), sharex="row", sharey="row")
    for i, name in enumerate(PARTS):
        for row, (key, S, xl) in enumerate((("D_chi", S_CHI, "$s$ (indicator $\\chi$)"),
                                            ("D_phi", S_PHI, "$s$ (signed distance $\\varphi$)"))):
            ax = axs[row, i]
            for k, mname in enumerate(MODEL_ORDER[1:]):
                ax.plot(S, table[name][mname][key], color=SERIES[k], lw=1.6, label=mname,
                        marker=MARKERS[k], markevery=8, ms=4)
            ax.set_yscale("log")
            ax.set_xlabel(xl)
            if row == 0:
                ax.set_title(name, fontsize=9)
                ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
            else:
                ax.axvline(1.5, color=MUTED, lw=0.8, ls=":")
            if i == 0:
                ax.set_ylabel("$D_s = \\|u_A-u_B\\|_{H^s} / \\|u_A\\|_{H^s}$")
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "fig" / "s_curves.pdf"); plt.close(fig)


def plot_rank(out, plt, table):
    """per part: the rank of every wrong model (1 = worst) under 1-IoU and under D_s, as s moves."""
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5), sharey=True)
    for i, name in enumerate(PARTS):
        ax = axs[i]
        ms = MODEL_ORDER[1:]
        D = np.array([table[name][m]["D_chi"] for m in ms])        # (models, s)
        ranks = (-D).argsort(axis=0).argsort(axis=0) + 1
        for k, m in enumerate(ms):
            ax.plot(S_CHI, ranks[k], color=SERIES[k], lw=1.6, marker=MARKERS[k], markevery=8, ms=4, label=m)
        iou_rank = (np.argsort(np.argsort([-(1 - table[name][m]["iou_exact"]) for m in ms])) + 1)
        for k in range(len(ms)):
            ax.plot([0.55], [iou_rank[k]], marker=MARKERS[k], color=SERIES[k], ms=5, clip_on=False)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("$s$ ($\\chi$);  right dot: 1-IoU")
        ax.set_yticks(range(1, len(ms) + 1))
        ax.invert_yaxis()
    axs[0].set_ylabel("rank (1 = judged worst)")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig" / "ranks.pdf"); plt.close(fig)


def plot_density(out, plt, grid, spec, e_chi, e_phi, title, fname, z_mm):
    """mid-height slices of |(1 - l^2 Laplace)^{s/2} e|^2 for three s per shape function."""
    from matplotlib.colors import PowerNorm
    k = int(np.argmin(np.abs(grid.axes[2] - z_mm)))
    ext = (grid.axes[0][0], grid.axes[0][-1], grid.axes[1][0], grid.axes[1][-1])
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.0))
    for row, (e, picks, lab) in enumerate(((e_chi, S_PICK_CHI, "\\chi"), (e_phi, S_PICK_PHI, "\\varphi"))):
        for j, s in enumerate(picks):
            d = spec.density(e, s)
            sl = d[:, :, k].T
            ax = axs[row, j]
            ax.imshow(sl / (sl.max() + 1e-30), origin="lower", extent=ext, cmap="Blues",
                      norm=PowerNorm(0.5, vmin=0, vmax=1), interpolation="nearest")
            share = float(d[:, :, max(k - 1, 0):k + 2].sum() / d.sum())
            ax.set_title(f"${lab}$,  $s={s:g}$", fontsize=8.5)
            ax.set_xlim(-6, 66); ax.set_ylim(-6, 46)
            ax.grid(False)
            ax.tick_params(labelsize=7)
    fig.suptitle(title + f"  (slice z = {grid.axes[2][k]:.2f} mm, each panel scaled to its own maximum, square-root colour scale)",
                 fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(out / "fig" / fname); plt.close(fig)


def plot_density3d(out, plt, grid, spec, e_chi, title, fname, ghost=None):
    """the voxels that carry the top 60 % of ||e||_{H^s}^2, in 3D, for three s."""
    fig = plt.figure(figsize=(7.4, 2.8))
    X, Y, Z = np.meshgrid(*grid.axes, indexing="ij")
    for j, s in enumerate(S_PICK_CHI):
        d = spec.density(e_chi, s)
        flat = d.ravel()
        order = np.argsort(flat)[::-1]
        c = np.cumsum(flat[order]) / flat.sum()
        keep = order[: int(np.searchsorted(c, 0.6)) + 1]
        if len(keep) > 40000:
            keep = np.random.default_rng(0).choice(keep, 40000, replace=False)
        ax = fig.add_subplot(1, 3, j + 1, projection="3d")
        if ghost is not None:
            draw_mesh(ax, ghost, "#9a9992", 0.08)
        w = flat[keep] / flat[keep].max()
        ax.scatter(X.ravel()[keep], Y.ravel()[keep], Z.ravel()[keep], c=w, cmap="Blues", vmin=-0.3, vmax=1,
                   s=1.5, linewidths=0, depthshade=False)
        frame3d(ax)
        ax.set_title(f"$s = {s:g}$: {len(order[:int(np.searchsorted(c, 0.6)) + 1])} voxels hold 60 %", fontsize=8)
    fig.suptitle(title, fontsize=8.5)
    fig.subplots_adjust(wspace=0.0)
    fig.savefig(out / "fig" / fname); plt.close(fig)


# ------------------------------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/reports/iou_two_stage"))
    ap.add_argument("--pitch", type=float, default=0.5)
    ap.add_argument("--no-occt", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    for d in ("fig", "data", "parts"):
        (out / d).mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style(plt)

    # ---- STEP files and the exact residual of every pair
    items, pairs = [], []
    sweeps = dict(hole_tool_shift=[0, 0.5, 1, 2, 4, 8, 12], tray_all_shift=[0, 0.25, 0.5, 1, 1.5, 2, 3],
                  boss_all_shift=[0, 0.25, 0.5, 1, 1.5, 2, 3])
    for name, part in PARTS.items():
        items.append(dict(tree=part, shift=[0, 0, 0], turn=0.0, path=str(out / "parts" / f"{name}.step")))
        for mname, (tree, shift, turn) in models_of(part).items():
            p = out / "parts" / f"{name}__{mname.replace(' ', '_')}.step"
            items.append(dict(tree=tree, shift=list(shift), turn=turn, path=str(p)))
            pairs.append((name, mname, p))
    sweep_models = {}
    for key, ds in sweeps.items():
        name = key.split("_")[0]
        part = PARTS[name]
        for d in ds:
            if key == "hole_tool_shift":
                tree, shift = dict(part, stage2=moved(part["stage2"], (d, 0, 0))), (0, 0, 0)
            else:
                tree, shift = part, (d, 0, 0)
            p = out / "parts" / f"sweep_{key}_{d:g}.step"
            items.append(dict(tree=tree, shift=list(shift), turn=0.0, path=str(p)))
            sweep_models[(key, d)] = (tree, shift, p)

    exact = {}
    cache = out / "data" / "exact.json"
    if cache.is_file():
        exact = json.loads(cache.read_text())
    if not args.no_occt:
        todo = [(n, m, p) for n, m, p in pairs if f"{n}|{m}" not in exact] + \
               [(k, d, p) for (k, d), (_, _, p) in sweep_models.items() if f"{k}|{d:g}" not in exact]
        if todo:
            print(f"freecadcmd: building {len(items)} STEP files", flush=True)
            write_steps(items, CENTRE)
            from reco_field.exact_check import exact_residual
            for a, b, p in todo:
                name = a if a in PARTS else a.split("_")[0]
                r = exact_residual(out / "parts" / f"{name}.step", p)
                key = f"{a}|{b if isinstance(b, str) else format(b, 'g')}"
                exact[key] = {k: r.get(k) for k in ("iou", "r_plus", "r_minus", "model_genus", "input_genus",
                                                    "model_solids", "fuzzy", "error", "input_volume",
                                                    "model_volume")}
                print(key, exact[key], flush=True)
            for name in PARTS:
                r = exact_residual(out / "parts" / f"{name}.step", out / "parts" / f"{name}.step", booleans=False)
                exact[f"{name}|genus"] = r.get("model_genus")
            cache.write_text(json.dumps(exact, indent=1))
    for n in PARTS:
        assert abs(exact[f"{n}|exact"]["iou"] - 1.0) < 1e-6, exact[f"{n}|exact"]

    # ---- the fine grid: Sobolev distances of both shape functions
    lo, hi = (-16, -16, -16), (76, 56, 51)
    G = Grid(lo, hi, args.pitch)
    spec = Spectrum(G)
    print("grid", G.shape, flush=True)
    table = {}
    dens_cases = {}
    for name, part in PARTS.items():
        A = occupancy(manifold_of(part), G)
        phiA = signed_distance(A, G.pitch)
        PA_chi, PA_phi = spec.power(A), spec.power(phiA)
        nA_chi = {s: spec.norm2(PA_chi, s) for s in S_CHI}
        nA_phi = {s: spec.norm2(PA_phi, s) for s in S_PHI}
        table[name] = {}
        for mname, (tree, shift, turn) in models_of(part).items():
            B = occupancy(manifold_of(tree, shift, turn), G)
            e_chi = A.astype(np.float64) - B
            e_phi = phiA - signed_distance(B, G.pitch)
            P_chi, P_phi = spec.power(e_chi), spec.power(e_phi)
            D_chi = [math.sqrt(spec.norm2(P_chi, s) / nA_chi[s]) for s in S_CHI]
            D_phi = [math.sqrt(spec.norm2(P_phi, s) / nA_phi[s]) for s in S_PHI]
            sym = int(np.count_nonzero(A ^ B))
            vox_iou = voxel_iou(A, B)
            ex = exact.get(f"{name}|{mname}", {})
            row = dict(iou_exact=ex.get("iou"), r_plus=ex.get("r_plus"), r_minus=ex.get("r_minus"),
                       model_genus=ex.get("model_genus"), input_genus=exact.get(f"{name}|genus"),
                       iou_voxel_fine=vox_iou, D_chi=D_chi, D_phi=D_phi,
                       solids=f"{len(manifold_of(part).decompose())}/{len(manifold_of(tree, shift, turn).decompose())}",
                       D0_check=math.sqrt(sym / np.count_nonzero(A)))
            s0 = int(np.argmin(np.abs(S_CHI)))
            if abs(S_CHI[s0]) < 1e-9:
                assert abs(D_chi[s0] - row["D0_check"]) < 1e-6, (name, mname, D_chi[s0], row["D0_check"])
            table[name][mname] = row
            if (name, mname) in (("hole", "tool moved 2 mm"), ("hole", "all moved 1 mm"), ("slot", "stage 1 only"),
                                 ("tray", "all turned 2 deg")):
                dens_cases[(name, mname)] = (e_chi, e_phi)
            print(name, mname, f"IoU exact {row['iou_exact']}, voxel {vox_iou:.4f}", flush=True)
        del A, phiA

    # the pairs where the H^s distances are evaluated at fixed s (for the table)
    def at(S, vals, s):
        return vals[int(np.argmin(np.abs(np.asarray(S) - s)))]

    # ---- sweeps on the fine grid
    sweep_rows = {}
    for key, ds in sweeps.items():
        name = key.split("_")[0]
        A = occupancy(manifold_of(PARTS[name]), G)
        phiA = signed_distance(A, G.pitch)
        PA_chi, PA_phi = spec.power(A), spec.power(phiA)
        rows = []
        for d in ds:
            tree, shift, _ = sweep_models[(key, d)]
            B = occupancy(manifold_of(tree, shift), G)
            e_chi, e_phi = A.astype(np.float64) - B, phiA - signed_distance(B, G.pitch)
            Pc, Pp = spec.power(e_chi), spec.power(e_phi)
            ex = exact.get(f"{key}|{d:g}", {})
            rows.append(dict(d=d, iou_exact=ex.get("iou"), iou_voxel=voxel_iou(A, B),
                             **{f"chi_{s:g}": math.sqrt(spec.norm2(Pc, s) / spec.norm2(PA_chi, s)) for s in S_PICK_CHI},
                             **{f"phi_{s:g}": math.sqrt(spec.norm2(Pp, s) / spec.norm2(PA_phi, s)) for s in S_PICK_PHI}))
        sweep_rows[key] = rows

    # ---- the pitch study: voxel IoU (compare.py) and rim-forgiving IoU (cascade) against the exact IoU
    pitch_cases = [("slot", "stage 1 only"), ("tray", "all moved 1 mm"), ("hole", "tool moved 2 mm"),
                   ("boss", "all turned 2 deg")]
    pitches = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    rng = np.random.default_rng(20260930)
    offsets = rng.random((6, 3))
    pitch_rows = {}
    for name, mname in pitch_cases:
        tree, shift, turn = models_of(PARTS[name])[mname]
        mA, mB = manifold_of(PARTS[name]), manifold_of(tree, shift, turn)
        rows = []
        for p in pitches:
            v, r, cov = [], [], []
            for off in offsets:
                g = Grid((-8, -8, -8), (70, 50, 45), p, off)
                A, B = occupancy(mA, g), occupancy(mB, g)
                v.append(voxel_iou(A, B))
                ri, c = rim_forgiving_iou(A, B)
                r.append(ri); cov.append(c)
            rows.append(dict(pitch=p, voxel=v, rim=r, counted_share=cov))
        pitch_rows[f"{name}|{mname}"] = rows
        print("pitch", name, mname, [(x["pitch"], round(float(np.mean(x["voxel"])), 4)) for x in rows], flush=True)

    # ---- the indicator's H^s norm against the pitch: why s must stay below 1/2
    conv = {}
    for p in (2.0, 1.0, 0.5):
        g = Grid(lo, hi, p)
        sp = Spectrum(g)
        P = sp.power(occupancy(manifold_of(PARTS["hole"]), g))
        conv[p] = [sp.norm2(P, s) for s in S_CHI]

    (out / "data" / "study.json").write_text(json.dumps(dict(
        ell=ELL, clip=CLIP, pitch=G.pitch, grid=G.shape, s_chi=S_CHI.tolist(), s_phi=S_PHI.tolist(), table=table,
        sweeps=sweep_rows, pitch_study=pitch_rows, norm_vs_pitch={str(k): v for k, v in conv.items()}), indent=1))

    # ---- tables
    lines = []
    for name in PARTS:
        for mname in MODEL_ORDER[1:]:
            r = table[name][mname]
            genus = f"{r['input_genus']}/{r['model_genus']} & {r['solids']}"
            lines.append(f"{name} & {mname} & {r['iou_exact']:.4f} & {r['iou_voxel_fine']:.4f} & "
                         f"{r['r_plus']:.0f} & {r['r_minus']:.0f} & {genus} & "
                         f"{at(S_CHI, r['D_chi'], -1.0):.3f} & {at(S_CHI, r['D_chi'], 0.0):.3f} & "
                         f"{at(S_CHI, r['D_chi'], 0.45):.3f} & {at(S_PHI, r['D_phi'], 1.0):.3f} \\\\")
        lines.append("\\midrule")
    (out / "data" / "table_pairs.tex").write_text("\n".join(lines[:-1] + ["\\bottomrule"]) + "\n")

    def fmt_rim(v):
        v = np.asarray(v, float)
        n = int(np.isfinite(v).sum())
        return "undef." if n == 0 else f"{np.nanmean(v):.4f}" + ("" if n == len(v) else f" ({n}/{len(v)})")

    lines = []
    for key, rows in pitch_rows.items():
        name, mname = key.split("|")
        ex = table[name][mname]["iou_exact"]
        for r in rows:
            lines.append(f"{name}, {mname} & {r['pitch']:g} & {ex:.4f} & {np.mean(r['voxel']):.4f} & "
                         f"{np.min(r['voxel']):.4f}--{np.max(r['voxel']):.4f} & {fmt_rim(r['rim'])} & "
                         f"{np.mean(r['counted_share']):.2f} \\\\")
        lines.append("\\midrule")
    (out / "data" / "table_pitch.tex").write_text("\n".join(lines[:-1] + ["\\bottomrule"]) + "\n")

    # ---- figures
    plot_trees(out, plt)
    plot_errors(out, plt, table)
    plot_s_curves(out, plt, table)
    plot_rank(out, plt, table)
    for (name, mname), (e_chi, e_phi) in dens_cases.items():
        tag = f"{name}_{mname.replace(' ', '_')}"
        z = 10.0
        plot_density(out, plt, G, spec, e_chi, e_phi, f"{name}, {mname}", f"density_{tag}.png", z)
        plot_density3d(out, plt, G, spec, e_chi, f"{name}, {mname}: where $\\|e\\|_{{H^s}}^2$ sits ($\\chi$)",
                       f"density3d_{tag}.png", ghost=manifold_of(PARTS[name]))

    # sweeps
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7), sharey=True)
    for ax, (key, rows) in zip(axs, sweep_rows.items()):
        d = [r["d"] for r in rows][1:]
        ok = [x >= G.pitch for x in d]                  # a shift below the pitch is not resolved by the grid measures
        series = [("1 - IoU (exact)", [1 - r["iou_exact"] for r in rows][1:])] + \
                 [(f"$D_{{{s:g}}}(\\chi)$", [r[f"chi_{s:g}"] for r in rows][1:]) for s in S_PICK_CHI] + \
                 [("$D_{1}(\\varphi)$", [r["phi_1"] for r in rows][1:])]
        for k, (lab, y) in enumerate(series):
            if k:
                y = [v if o else np.nan for v, o in zip(y, ok)]
            ax.plot(d, y, color=SERIES[k] if k else INK, marker=MARKERS[k], ms=4, lw=1.6, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel({"hole_tool_shift": "hole: tool moved by $\\delta$ [mm]",
                       "tray_all_shift": "tray: model moved by $\\delta$ [mm]",
                       "boss_all_shift": "boss: model moved by $\\delta$ [mm]"}[key])
    axs[0].set_ylabel("error measure")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig" / "sweeps.pdf"); plt.close(fig)

    # pitch study
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=False)
    for ax, (key, rows) in zip(axs, pitch_rows.items()):
        name, mname = key.split("|")
        ex = table[name][mname]["iou_exact"]
        p = [r["pitch"] for r in rows]
        for k, (lab, f) in enumerate((("voxel IoU (compare.py)", "voxel"), ("rim-forgiving (cascade)", "rim"))):
            fin = [np.asarray(r[f], float)[np.isfinite(r[f])] for r in rows]
            m = [x.mean() if len(x) else np.nan for x in fin]
            lo_, hi_ = [x.min() if len(x) else np.nan for x in fin], [x.max() if len(x) else np.nan for x in fin]
            ax.fill_between(p, lo_, hi_, color=SERIES[k], alpha=0.18, lw=0)
            ax.plot(p, m, color=SERIES[k], marker=MARKERS[k], ms=4, lw=1.6, label=lab)
        ax.axhline(ex, color=INK, lw=1.0, ls="--", label="exact (OCCT)")
        ax.set_title(f"{name}: {mname}", fontsize=8)
        ax.set_xlabel("pitch [mm]")
        ax.set_ylim(-0.03, 1.03)
    axs[0].set_ylabel("IoU")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig" / "pitch.pdf"); plt.close(fig)

    # norm of chi_A against s for three pitches
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for k, (p, v) in enumerate(conv.items()):
        ax.plot(S_CHI, v, color=SERIES[k], marker=MARKERS[k], markevery=6, ms=4, lw=1.6, label=f"pitch {p:g} mm")
    ax.axvline(0.5, color=MUTED, lw=0.8, ls=":")
    ax.set_yscale("log"); ax.set_xlabel("$s$"); ax.set_ylabel("$\\|\\chi_A\\|_{H^s}^2$ [mm$^3$] (hole)")
    ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(out / "fig" / "norm_vs_pitch.pdf"); plt.close(fig)
    print("done", out)


if __name__ == "__main__":
    main()
