#!/usr/bin/env python3
"""
Field atlas: what each FreeCAD primitive leaves in the closest-point field, read from the DERIVATIVE of the field.

The closest-point map c(x) = x + V(x) has the Jacobian J = dc/dx = I - grad f grad f^T - f H(f). Off the medial axis it
is symmetric with eigenvalue 0 along the normal V/|V| and, in the tangent plane,

    lambda_i = 1 / (1 + d kappa_i)        d = signed distance (positive OUTSIDE), kappa_i = principal curvature of the
                                          surface at c(x) (positive where the solid is convex)

so kappa_i = (1/lambda_i - 1) / d. Where c(x) is an EDGE the whole wedge of points in front of it maps onto the edge
(lambda = 0 across it), at a VERTEX onto one point (both 0). Hence one local quantity - the tangential spectrum of J -
separates face / edge / vertex and, on faces, planar / cylindrical / doubly curved / saddle surface:

    class    tangential spectrum                        e.g.
    plane    rank 2, kappa1 ~ kappa2 ~ 0                box face, cut by a plane
    cyl      rank 2, one kappa ~ 0                      cylinder / cone mantle, torus nowhere
    dome     rank 2, both kappa != 0, same sign         sphere, torus outer half
    saddle   rank 2, kappa of opposite sign             torus inner half, a fillet
    edge     rank 1                                     box edge, cylinder rim, the rim of a cut
    vertex   rank 0                                     box corner, cone apex

Sources: "analytic" (sdf_primitives, c = x - f grad f), "mesh" (the exact closest-point query on the FreeCAD STL,
dataset_gen/src/closest_point_field.py), "grid" (the 64^3 voxel-derived V of training/field_input.py). Correlation:
on example parts with a known program (dataset_gen/examples, reconstructed by reco_field.peel at 100 % explained), every
sample point is labelled by the operation/primitive that owns c(x) (smallest |sdf| of the program's leaves at c(x)).

    python3 scripts/field_atlas.py            # -> docs/data/field_atlas.json, docs/thesis/generated/field_atlas_*.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import load_field  # noqa: E402

CLASSES = ("plane", "cyl", "dome", "saddle", "edge", "vertex")
RANK_LAMBDA = 0.25          # a tangential eigenvalue below this = a collapsed direction (edge / vertex)
FLAT = 0.03                 # |kappa| below this (1/mm) = flat; the parts are 10-40 mm, radii 2-10 mm


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


ANALYTIC = {       # off-centre and rotated (never axis-aligned: an axis-aligned test hides frame bugs)
    "box": sp.Placed(sp.Box((9.0, 5.5, 3.5)), _rot((1, 2, 0.5), 27), np.array([1.3, -2.1, 0.7])),
    "cylinder": sp.Placed(sp.Cylinder(5.0, 16.0), _rot((0.3, 1, 0.2), 38), np.array([-1.1, 0.9, -7.2])),
    "sphere": sp.Placed(sp.Sphere(8.5), np.eye(3), np.array([2.2, -1.4, 0.6])),
    "cone": sp.Placed(sp.Cone(7.0, 2.5, 14.0), _rot((1, 0, 0.4), 22), np.array([0.8, 1.6, -6.0])),
    "torus": sp.Placed(sp.Torus(9.0, 3.0), _rot((0.2, 1, 0), 31), np.array([-0.7, 1.1, 0.4])),
}
EXPECTED = {"box": "plane", "cylinder": "cyl", "sphere": "dome", "cone": "cyl", "torus": "dome/saddle"}

MESH_PARTS = ("bs_box", "cy_cyl", "bs_sphere", "co_cone", "to_torus", "box_with_bore", "bs_box_dome", "bs_box_sphere_pocket",
              "co_countersink", "to_ring_groove", "cy_on_box", "cy_flat", "l_bracket", "tube")


# ------------------------------------------------------------------------------------------------ the local spectrum

def spectrum(closest_fn, pts: np.ndarray, h: float):
    """(signed distance (n,), lambda (n, 2) ascending, kappa (n, 2), tangent eigenvectors (n, 3, 2)) from central
    differences of the closest-point map. `closest_fn(points) -> (closest (n,3), signed distance positive OUTSIDE)`."""
    c0, d = closest_fn(pts)
    n = len(pts)
    J = np.empty((n, 3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, :, k] = (closest_fn(pts + e)[0] - closest_fn(pts - e)[0]) / (2 * h)
    S = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    w, vec = np.linalg.eigh(S)
    normal = c0 - pts
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    drop = np.abs(np.einsum("nij,ni->nj", vec, normal)).argmax(axis=1)            # the eigenvector along the normal
    keep = np.array([[j for j in range(3) if j != i] for i in range(3)])[drop]    # (n, 2)
    lam = np.take_along_axis(w, keep, axis=1)
    tvec = np.take_along_axis(vec, keep[:, None, :].repeat(3, axis=1), axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (1.0 / lam - 1.0) / d[:, None]
    return d, lam, kappa, tvec


def classify(lam: np.ndarray, kappa: np.ndarray, flat: float = FLAT) -> np.ndarray:
    rank = (lam > RANK_LAMBDA).sum(axis=1)
    out = np.empty(len(lam), dtype=object)
    out[rank == 0], out[rank == 1] = "vertex", "edge"
    k = np.where(np.abs(kappa) < flat, 0.0, kappa)
    face = rank == 2
    nz = (k != 0).sum(axis=1)
    out[face & (nz == 0)] = "plane"
    out[face & (nz == 1)] = "cyl"
    out[face & (nz == 2) & (np.sign(k[:, 0]) == np.sign(k[:, 1]))] = "dome"
    out[face & (nz == 2) & (np.sign(k[:, 0]) != np.sign(k[:, 1]))] = "saddle"
    return out


def _fractions(labels) -> dict:
    labels = np.asarray(labels)
    return {c: float((labels == c).mean()) for c in CLASSES}


# ------------------------------------------------------------------------------------------------ sources

def analytic_closest(prim: sp.Primitive, eps: float = 1e-5):
    def fn(p):
        f = prim.sdf(p)
        g = np.stack([(prim.sdf(p + e) - prim.sdf(p - e)) / (2 * eps) for e in np.eye(3) * eps], axis=1)
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        return p - f[:, None] * g, f
    return fn


def mesh_closest(field):
    def fn(p):
        q = field.query(p, exact=True)
        return q.closest, -q.signed_distance                                     # query: positive INSIDE
    return fn


def band_points(prim: sp.Primitive, n: int, lo_mm: float, hi_mm: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lo, hi = prim.bounds()
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    b0, b1 = corners.min(0) - hi_mm, corners.max(0) + hi_mm
    out = []
    while sum(len(o) for o in out) < n:
        p = rng.uniform(b0, b1, (20000, 3))
        a = np.abs(prim.sdf(p))
        out.append(p[(a > lo_mm) & (a < hi_mm)])
    return np.concatenate(out)[:n]


def mesh_band_points(field, n: int, lo_mm: float, hi_mm: float, seed: int = 0) -> np.ndarray:
    p = field.sample_points(n, band=hi_mm, seed=seed)
    q = field.query(p, exact=True)
    keep = q.distance > lo_mm
    return p[keep]


# ------------------------------------------------------------------------------------------------ experiments

def analytic_atlas(n: int = 3000, h: float = 0.02) -> dict:
    out = {}
    for name, prim in ANALYTIC.items():
        pts = band_points(prim, n, 0.2, 1.5)
        d, lam, kappa, _ = spectrum(analytic_closest(prim), pts, h)
        lab = classify(lam, kappa)
        face = lab != "edge"
        face &= lab != "vertex"
        ks = np.sort(np.abs(kappa[face]), axis=1)
        out[name] = dict(expected=EXPECTED[name], fractions=_fractions(lab), n=int(len(pts)),
                         kappa_abs_median=[float(np.median(ks[:, 0])), float(np.median(ks[:, 1]))] if len(ks) else None,
                         _points=(kappa, lab))
    # a ground-truth check: the sphere's |kappa| must be 1/r on both axes, the cylinder's (0, 1/r)
    return out


def mesh_atlas(examples: Path, n: int = 2500, hs=(0.05, 0.2, 0.5)) -> dict:
    out = {}
    for name in MESH_PARTS:
        path = examples / f"{name}.stl"
        if not path.is_file():
            continue
        field = load_field(path)
        pts = mesh_band_points(field, n, 0.3, 1.5)
        per_h = {}
        for h in hs:
            t = time.time()
            d, lam, kappa, _ = spectrum(mesh_closest(field), pts, h)
            lab = classify(lam, kappa)
            per_h[str(h)] = dict(fractions=_fractions(lab), seconds=round(time.time() - t, 1))
        out[name] = dict(n=int(len(pts)), by_step=per_h)
    return out


def grid_atlas(resolution: int = 64, grid_mm: float = 40.0, sigma: float = 1.0, exact: bool = False) -> dict:
    """The same spectrum on the voxel grid, J by np.gradient of c = x + V between voxel centres. exact=False: V derived
    from the occupancy grid (training/field_input.sdf_and_field, what the "fused"/"dirlog" kinds give a CNN);
    exact=True: the exact V sampled at the voxel centres (what a `.field.npz` stores, dataset_gen/src/field_channels.py)."""
    from training.field_input import sdf_and_field
    out = {}
    pitch = grid_mm / resolution
    for name, prim in ANALYTIC.items():
        sdf = sp.sdf_grid(prim, resolution, grid_mm) / pitch                    # voxels, array frame
        occ = sdf <= 0
        if exact:
            g = np.stack(np.gradient(sdf), axis=0)
            g /= np.maximum(np.linalg.norm(g, axis=0, keepdims=True), 1e-9)
            eps = 1e-4
            cen = sp.voxel_centres(resolution, grid_mm).reshape(-1, 3)
            gw = np.stack([(prim.sdf(cen + e) - prim.sdf(cen - e)) / (2 * eps) for e in np.eye(3) * eps], axis=1)
            gw /= np.maximum(np.linalg.norm(gw, axis=1, keepdims=True), 1e-12)
            vw = (-sdf.reshape(-1)[:, None] * gw)                                  # world (x, y, z) components, voxels
            field = np.stack([vw[:, 2], -vw[:, 1], vw[:, 0]]).reshape((3,) + sdf.shape)   # array axes [z, -y, x]
        else:
            _, field = sdf_and_field(occ, truncation=1e6, sigma=sigma)            # untruncated V, in voxels
            field = field * 1e6
        idx = np.stack(np.indices(occ.shape), axis=0).astype(float)
        c = idx + field                                                        # closest point, voxel units
        J = np.stack([np.stack(np.gradient(c[i]), axis=0) for i in range(3)], axis=0)   # (3, 3, R, R, R)
        band = (np.abs(sdf) > 0.5) & (np.abs(sdf) < 2.5)
        Jb = np.moveaxis(J[:, :, band], -1, 0)
        S = 0.5 * (Jb + np.transpose(Jb, (0, 2, 1)))
        w, vec = np.linalg.eigh(S)
        normal = np.moveaxis(field[:, band], 0, -1)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        drop = np.abs(np.einsum("nij,ni->nj", vec, normal)).argmax(axis=1)
        keep = np.array([[j for j in range(3) if j != i] for i in range(3)])[drop]
        lam = np.take_along_axis(w, keep, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            kappa = (1.0 / lam - 1.0) / (sdf[band][:, None] * pitch)             # 1/mm
        lab = classify(lam, kappa)
        out[name] = dict(expected=EXPECTED[name], fractions=_fractions(lab), n=int(band.sum()))
    return out


def leaves(res):
    return [(op.op, op.fit.kind, op.primitive) for op in res.ops]


def correlation(examples: Path, parts=("box_with_bore", "bs_box_dome", "bs_box_sphere_pocket", "co_countersink",
                                         "to_ring_groove", "cy_on_box", "cy_flat", "l_bracket", "tube", "bs_box_step"),
                n: int = 2500, h: float = 0.2) -> dict:
    """Owner (op:kind of the program leaf nearest to c(x)) against the local field class."""
    from reco_field.peel import peel
    table, per_part = {}, {}
    for name in parts:
        path = examples / f"{name}.stl"
        if not path.is_file():
            continue
        field = load_field(path)
        res = peel(field)
        lv = leaves(res)
        pts = mesh_band_points(field, n, 0.3, 1.5)
        d, lam, kappa, _ = spectrum(mesh_closest(field), pts, h)
        lab = classify(lam, kappa)
        c = field.query(pts, exact=True).closest
        dist = np.stack([np.abs(p.sdf(c)) for _, _, p in lv], axis=1)
        owner_i = dist.argmin(axis=1)
        # an edge belongs to two leaves (or two faces of one): record whether c(x) is ON a second leaf as well
        second = np.sort(dist, axis=1)[:, 1] if dist.shape[1] > 1 else np.full(len(c), np.inf)
        junction = second < 0.05
        owners = np.array([f"{lv[i][0]}:{lv[i][1]}" for i in owner_i], dtype=object)
        per_part[name] = dict(program=" + ".join(f"{o}:{k}" for o, k, _ in lv), explained=round(res.explained, 4),
                              junction_share_of_edges=float(junction[lab == "edge"].mean()) if (lab == "edge").any() else None,
                              junction_share_of_faces=float(junction[(lab != "edge") & (lab != "vertex")].mean()))
        for o, l, j in zip(owners, lab, junction):
            key = "junction" if (j and l in ("edge", "vertex")) else o
            table.setdefault(key, {}).setdefault(l, 0)
            table[key][l] += 1
    return dict(parts=per_part, owner_by_class=table, mutual_information_bits=_mutual_information(table))


def _mutual_information(table: dict) -> float:
    owners = sorted(table)
    M = np.array([[table[o].get(c, 0) for c in CLASSES] for o in owners], float)
    P = M / M.sum()
    po, pc = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (po @ pc)[nz])).sum())


def figure(analytic: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"plane": "#6b7280", "cyl": "#2563eb", "dome": "#16a34a", "saddle": "#d97706", "edge": "#dc2626", "vertex": "#7c3aed"}
    fig, axes = plt.subplots(1, len(analytic), figsize=(3.0 * len(analytic), 3.1), sharex=True, sharey=True)
    for ax, (name, rec) in zip(axes, analytic.items()):
        kappa, lab = rec["_points"]
        face = (lab != "edge") & (lab != "vertex")
        k = np.sort(kappa[face], axis=1)
        for cls in ("plane", "cyl", "dome", "saddle"):
            m = lab[face] == cls
            ax.scatter(k[m, 0], k[m, 1], s=3, alpha=0.5, color=colours[cls], label=cls)
        ax.set_title(f"{name}  (edge {rec['fractions']['edge']:.0%}, vertex {rec['fractions']['vertex']:.0%})", fontsize=8)
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.axhline(0, color="#bbb", lw=0.5)
        ax.axvline(0, color="#bbb", lw=0.5)
        ax.set_xlabel(r"$\kappa_1$ [1/mm]", fontsize=8)
    axes[0].set_ylabel(r"$\kappa_2$ [1/mm]", fontsize=8)
    axes[0].legend(fontsize=6, markerscale=3, loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def class_scatter(path, out_png: Path, n: int = 6000, h: float | None = None, band: float | None = None):
    """A preview: near-surface points of one STL/STEP coloured by their class (three views). Step and band default to
    a fraction of the part's extent, so a 7 m frame and a 20 mm example part are treated alike."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    field = load_field(path)
    extent = float((field.bounds[1] - field.bounds[0]).max())
    h = h or 0.004 * extent
    band = band or 0.03 * extent
    pts = mesh_band_points(field, n, 0.25 * band, band)
    d, lam, kappa, _ = spectrum(mesh_closest(field), pts, h)
    lab = classify(lam, kappa, flat=FLAT * 20.0 / extent)                     # flatness relative to a 20 mm part
    c = field.query(pts, exact=True).closest
    colours = {"plane": "#9ca3af", "cyl": "#2563eb", "dome": "#16a34a", "saddle": "#d97706", "edge": "#dc2626", "vertex": "#7c3aed"}
    fig = plt.figure(figsize=(12, 4.2))
    for i, (elev, azim) in enumerate(((25, -60), (25, 30), (80, -90))):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        for cls in CLASSES:
            m = lab == cls
            ax.scatter(c[m, 0], c[m, 1], c[m, 2], s=1.5, color=colours[cls], label=f"{cls} {m.mean():.0%}")
        ax.view_init(elev, azim)
        ax.set_box_aspect(field.bounds[1] - field.bounds[0])
        ax.set_axis_off()
    fig.axes[0].legend(fontsize=7, markerscale=5, loc="upper left")
    fig.suptitle(f"{Path(path).name}: field class of the closest point (h = {h:.3g}, band {band:.3g})", fontsize=9)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return _fractions(lab)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", default=str(ROOT / "dataset_gen/examples"))
    ap.add_argument("--skip", nargs="*", default=[], choices=["mesh", "grid", "sweep", "correlation"])
    ap.add_argument("--plot", help="only draw the class preview of this STL/STEP (to --plot-out) and exit")
    ap.add_argument("--plot-out", default=str(ROOT / "docs/thesis/generated/field_classes.png"))
    args = ap.parse_args()
    if args.plot:
        print(class_scatter(args.plot, Path(args.plot_out)))
        return
    examples = Path(args.examples)
    out = ROOT / "docs/data/field_atlas.json"
    result = json.loads(out.read_text()) if out.is_file() else {}           # sections not recomputed are kept
    t = time.time()
    analytic = analytic_atlas()
    for name, rec in analytic.items():
        print(f"analytic {name:9s} {rec['expected']:12s}", {k: f"{v:.2f}" for k, v in rec["fractions"].items()},
              "median |kappa|", rec["kappa_abs_median"])
    figure(analytic, ROOT / "docs/thesis/generated/field_atlas_analytic.png")
    result["analytic"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in analytic.items()}
    print(f"  ({time.time() - t:.0f} s)")
    if "grid" not in args.skip:
        result["grid64"] = grid_atlas()
        result["grid64_exact"] = grid_atlas(exact=True)
        for key in ("grid64", "grid64_exact"):
            for name, rec in result[key].items():
                print(f"{key:12s} {name:9s}", {k: f"{v:.2f}" for k, v in rec["fractions"].items()})
    if "sweep" not in args.skip:                                            # can smoothing or resolution rescue the voxel field?
        result["grid_sweep"] = {f"{r}_sigma{s:g}": {k: v["fractions"] for k, v in grid_atlas(r, sigma=s).items()}
                                for r in (64, 128) for s in (1.0, 2.0, 3.0)}
    if "mesh" not in args.skip:
        result["mesh"] = mesh_atlas(examples)
        for name, rec in result["mesh"].items():
            for h, r in rec["by_step"].items():
                print(f"mesh     {name:22s} h={h:4s}", {k: f"{v:.2f}" for k, v in r["fractions"].items()}, f"{r['seconds']} s")
    if "correlation" not in args.skip:
        result["correlation"] = correlation(examples)
        print(json.dumps(result["correlation"], indent=1))
    out.write_text(json.dumps(result, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
