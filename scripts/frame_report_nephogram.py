#!/usr/bin/env python3
"""
Nephograms of the surface difference (docs/reports/frame_solution): the surfaces of input and model, coloured by their
distance to the other solid.

    python3 scripts/frame_report_nephogram.py <input.step|stl> <model.stl> <out.png> [--title ...] [--zoom x y z half]

Left column: the input's surface, coloured by its distance to the model; right column: the model's surface, coloured by
its distance to the input. Top row: the whole part; bottom row: a joint in detail (default: the corner of the bounding
box with the largest deviation nearby). Triangles are subdivided until their edges are short against the view, so a
colour belongs to a patch of surface and not to a whole wall. One sequential ramp, light = on the other surface.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
sys.path.insert(0, str(ROOT / "scripts"))

RAMP = ["#ECECEA", "#F6D3BF", "#F2A97F", "#EB6834", "#B8431A", "#6E2408"]      # one hue, light (on the surface) to dark


def distance(mesh, points):
    import igl
    d2, _, _ = igl.point_mesh_squared_distance(np.ascontiguousarray(points, dtype=np.float64), np.ascontiguousarray(mesh.vertices, dtype=np.float64),
                                               np.ascontiguousarray(mesh.faces, dtype=np.int64))
    return np.sqrt(np.maximum(d2, 0.0))


def bisect(tri, max_edge, cap):
    """Triangles (n, 3, 3) split at the midpoint of their longest edge until no edge is longer than `max_edge` (or the
    count reaches `cap`). A long, thin triangle (a fillet strip) grows with its length, not with its length squared,
    as a split into four similar triangles would make it (measured: 15 million triangles and 1.4 GiB on the frame)."""
    done = []
    while len(tri):
        e = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2)               # edge k: corner k -> corner k + 1
        k = e.argmax(axis=1)
        long = e[np.arange(len(tri)), k] > max_edge
        done.append(tri[~long])
        tri, k = tri[long], k[long]
        if not len(tri) or sum(map(len, done)) + 2 * len(tri) > cap:
            done.append(tri)
            break
        r = np.arange(len(tri))
        a, b, c = tri[r, k], tri[r, (k + 1) % 3], tri[r, (k + 2) % 3]
        m = 0.5 * (a + b)
        tri = np.concatenate([np.stack([a, m, c], axis=1), np.stack([m, b, c], axis=1)])
    return np.concatenate(done)


def refined(mesh, max_edge, box=None, cap=150_000):
    """The triangles (n, 3, 3) of `mesh` inside `box`, with edges of at most `max_edge`."""
    tri = np.asarray(mesh.triangles, float)
    if box is not None:
        lo, hi = box
        tri = tri[((tri.max(axis=1) >= lo) & (tri.min(axis=1) <= hi)).all(axis=1)]
        tri = bisect(tri, max(max_edge, (hi - lo).max() / 6.0), cap)       # roughly first, to drop what lies outside
        c = tri.mean(axis=1)
        pad = (hi - lo).max() / 6.0
        tri = tri[((c >= lo - pad) & (c <= hi + pad)).all(axis=1)]
    out = bisect(tri, max_edge, cap)
    if box is not None:
        c = out.mean(axis=1)
        out = out[((c >= lo) & (c <= hi)).all(axis=1)]
    return out, max_edge


def panel(ax, tri, dist, vmax, cmap, view, box, light=(0.35, -0.55, 0.76)):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    l = np.asarray(light) / np.linalg.norm(light)
    shade = 0.72 + 0.28 * np.abs(n @ l)
    rgb = cmap(np.clip(dist / vmax, 0, 1))[:, :3] * shade[:, None]
    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(rgb, 0, 1), edgecolors="none", antialiaseds=False))
    lo, hi = box
    c, r = 0.5 * (lo + hi), 0.5 * (hi - lo).max()
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(*view); ax.set_axis_off()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input"); ap.add_argument("model"); ap.add_argument("out")
    ap.add_argument("--title", default="")
    ap.add_argument("--zoom", type=float, nargs=4, default=None, metavar=("X", "Y", "Z", "HALF"))
    ap.add_argument("--vmax", type=float, default=None, help="upper end of the colour scale in mm (default: the largest distance)")
    ap.add_argument("--view", type=float, nargs=2, default=(22, -58))
    ap.add_argument("--zoom-view", type=float, nargs=2, default=(24, -52))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from frame_report import mesh_of
    cmap = LinearSegmentedColormap.from_list("difference", RAMP)
    m_in, m_mo = mesh_of(a.input), mesh_of(a.model)
    lo, hi = np.asarray(m_in.bounds)
    size = float((hi - lo).max())
    whole = {}
    for name, mesh, other in (("input", m_in, m_mo), ("model", m_mo, m_in)):
        tri, edge = refined(mesh, size / 260.0)
        whole[name] = (tri, distance(other, tri.mean(axis=1)), edge)
    worst = max(float(whole["input"][1].max()), float(whole["model"][1].max()))
    vmax = a.vmax or worst
    if a.zoom is None:
        tri, d, _ = whole["model"]
        far = tri.mean(axis=1)[d > 0.6 * d.max()]
        corner = lo.copy()
        k = np.argmin(np.linalg.norm(far - corner, axis=1))
        centre, half = far[k], size / 22.0
    else:
        centre, half = np.asarray(a.zoom[:3]), a.zoom[3]
    zbox = (centre - half, centre + half)
    zoom = {}
    for name, mesh, other in (("input", m_in, m_mo), ("model", m_mo, m_in)):
        tri, edge = refined(mesh, half / 70.0, zbox)
        zoom[name] = (tri, distance(other, tri.mean(axis=1)), edge)

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(7.2, 6.3))
    gs = fig.add_gridspec(2, 2, left=0.0, right=1.0, top=0.97, bottom=0.07, wspace=-0.12, hspace=-0.16)
    titles = {"input": "Input's surface: distance to the model", "model": "Model's surface: distance to the input"}
    for j, name in enumerate(("input", "model")):
        ax = fig.add_subplot(gs[0, j], projection="3d")
        tri, d, _ = whole[name]
        panel(ax, tri, d, vmax, cmap, a.view, (lo, hi))
        ax.text2D(0.5, 0.95, titles[name], transform=ax.transAxes, ha="center", fontsize=8.5, color="#1F1F1F")
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        on = 100 * area[d <= 0.01].sum() / area.sum()
        ax.text2D(0.5, 0.905, f"largest {d.max():.3g} mm; {on:.1f} % of the surface within 0.01 mm", transform=ax.transAxes,
                  ha="center", fontsize=7, color="#6B6B6B")
        ax = fig.add_subplot(gs[1, j], projection="3d")
        tri, d, _ = zoom[name]
        panel(ax, tri, d, vmax, cmap, a.zoom_view, zbox)
        ax.text2D(0.5, 0.88, f"detail, {2 * half:.0f} mm wide, at ({centre[0]:.0f}, {centre[1]:.0f}, {centre[2]:.0f})", transform=ax.transAxes,
                  ha="center", fontsize=7.5, color="#6B6B6B")
    cax = fig.add_axes([0.25, 0.075, 0.5, 0.016])
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("distance to the other surface (mm)", fontsize=8, color="#1F1F1F")
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=7, color="#6B6B6B", labelcolor="#6B6B6B", width=0.4)
    if a.title:
        fig.text(0.5, 0.978, a.title, ha="center", fontsize=9.5, color="#1F1F1F")
    fig.savefig(a.out, dpi=240)
    print(f"{a.out}: scale 0..{vmax:.3f} mm, largest input->model {whole['input'][1].max():.3f}, model->input {whole['model'][1].max():.3f}; "
          f"patches {len(whole['input'][0])} + {len(whole['model'][0])} (whole), {len(zoom['input'][0])} + {len(zoom['model'][0])} (detail)")


if __name__ == "__main__":
    main()
