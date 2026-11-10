#!/usr/bin/env python3
"""
Energetic (integral-geometric) descriptors of the base shapes, computed three ways and compared:

  analytic  closed forms (derived below) for the example parts' own dimensions
  mesh      the FreeCAD STL of dataset_gen/examples (polyhedral: volume, area, and M = sum over edges of
            1/2 * length * signed exterior dihedral angle, the exact integral mean curvature of a polyhedron;
            chi = V - E + F)
  voxel     the 40 mm / 64^3 occupancy grid of the analytic solid, axis-aligned and turned by a fixed rotation
            (the Minkowski counts of scripts/eval_minkowski_baseline.py; M = pi * B * pitch^2... see `voxel`)

Quantities (H = (k1 + k2)/2, convex positive; an edge contributes 1/2 * length * exterior angle to M):
  V, S, M = int H dA, chi (int K dA = 2 pi chi), W = int H^2 dA (Willmore, smooth faces only - a sharp edge makes it
  infinite, so W here is the bending energy of the curved faces), and the dimensionless ratios that are 1 exactly for a
  ball among convex bodies:
      q_iso = 36 pi V^2 / S^3      (isoperimetric)
      q_SM  = 4 pi S / M^2         (Minkowski's inequality M^2 >= 4 pi S)
      q_VM  = 3 V M / S^2          (Minkowski's S^2 >= 3 V M)
  and the normalised principal moments of inertia (sorted, sum 1).

Closed forms (derivations in docs/thesis/chapters/09b_field_analysis.tex):
  box a,b,c      V abc, S 2(ab+bc+ca), M pi(a+b+c) (edges only), chi 2, W 0
  sphere r       V 4/3 pi r^3, S 4 pi r^2, M 4 pi r, chi 2, W 4 pi
  cylinder r,h   V pi r^2 h, S 2 pi r(r+h), M pi h + pi^2 r, chi 2, W pi h / (2r)
  tube R,r,h     V pi(R^2-r^2)h, S 2 pi h(R+r) + 2 pi(R^2-r^2), M pi^2 (R+r) (the two mantles cancel: +pi h, -pi h),
                 chi 0, W pi h/(2R) + pi h/(2r)
  frustum r1,r2,h (half-angle a, tan a = (r1-r2)/h, slant s)
                 V pi h(r1^2+r1 r2+r2^2)/3, S pi(r1+r2)s + pi(r1^2+r2^2),
                 M pi h + pi r1 (pi/2 + a) + pi r2 (pi/2 - a), chi 2, W pi cos^2 a / (2 sin a) * ln(r1/r2) (inf for a
                 full cone: the apex)

    python3 scripts/energetic_descriptors.py            # -> docs/data/energetic_descriptors.json + table
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
sys.path.insert(0, str(ROOT / "scripts"))

from src import sdf_primitives as sp  # noqa: E402

PI = np.pi
SHAPES = {   # name: (example STL, analytic solid centred like the STL, closed-form parameters)
    "box": ("bs_box", sp.Box((10.0, 8.0, 6.0)), dict(kind="box", a=20.0, b=16.0, c=12.0)),
    "sphere": ("bs_sphere", sp.Sphere(10.0), dict(kind="sphere", r=10.0)),
    "cylinder": ("cy_cyl", sp.Placed(sp.Cylinder(10.0, 20.0), np.eye(3), np.array([0, 0, -10.0])), dict(kind="cylinder", r=10.0, h=20.0)),
    "tube": ("tube", sp.Placed(sp.Cylinder(10.0, 20.0, inner_radius=5.0), np.eye(3), np.array([0, 0, -10.0])),
             dict(kind="tube", R=10.0, r=5.0, h=20.0)),
    "frustum": ("co_cone", sp.Placed(sp.Cone(10.0, 3.0, 20.0), np.eye(3), np.array([0, 0, -10.0])), dict(kind="cone", r1=10.0, r2=3.0, h=20.0)),
    "cone": ("co_cone_apex", sp.Placed(sp.Cone(10.0, 0.0, 20.0), np.eye(3), np.array([0, 0, -10.0])), dict(kind="cone", r1=10.0, r2=0.0, h=20.0)),
}


def closed_form(p: dict) -> dict:
    k = p["kind"]
    if k == "box":
        a, b, c = p["a"], p["b"], p["c"]
        V, S, M, chi, W = a * b * c, 2 * (a * b + b * c + c * a), PI * (a + b + c), 2, 0.0
    elif k == "sphere":
        r = p["r"]
        V, S, M, chi, W = 4 / 3 * PI * r ** 3, 4 * PI * r ** 2, 4 * PI * r, 2, 4 * PI
    elif k == "cylinder":
        r, h = p["r"], p["h"]
        V, S, M, chi, W = PI * r * r * h, 2 * PI * r * (r + h), PI * h + PI ** 2 * r, 2, PI * h / (2 * r)
    elif k == "tube":
        R, r, h = p["R"], p["r"], p["h"]
        V = PI * (R * R - r * r) * h
        S = 2 * PI * h * (R + r) + 2 * PI * (R * R - r * r)
        M, chi, W = PI ** 2 * (R + r), 0, PI * h / (2 * R) + PI * h / (2 * r)
    elif k == "cone":
        r1, r2, h = p["r1"], p["r2"], p["h"]
        a = np.arctan((r1 - r2) / h)
        s = np.hypot(h, r1 - r2)
        V = PI * h * (r1 * r1 + r1 * r2 + r2 * r2) / 3
        S = PI * (r1 + r2) * s + PI * (r1 * r1 + r2 * r2)
        M = PI * h + PI * r1 * (PI / 2 + a) + PI * r2 * (PI / 2 - a)
        chi = 2
        W = PI * np.cos(a) ** 2 / (2 * np.sin(a)) * np.log(r1 / r2) if r2 > 0 else float("inf")
    else:
        raise ValueError(k)
    return dict(V=V, S=S, M=M, chi=chi, W=W)


def ratios(d: dict) -> dict:
    V, S, M = d["V"], d["S"], d["M"]
    return dict(q_iso=36 * PI * V * V / S ** 3, q_SM=4 * PI * S / M ** 2, q_VM=3 * V * M / S ** 2)


def mesh_values(stl: Path) -> dict:
    import trimesh
    m = trimesh.load(str(stl), force="mesh", process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    ang = m.face_adjacency_angles                                      # 0 for coplanar faces
    sign = np.where(m.face_adjacency_convex, 1.0, -1.0)
    length = np.linalg.norm(np.diff(m.vertices[m.face_adjacency_edges], axis=1)[:, 0], axis=1)
    M = float(0.5 * (length * ang * sign).sum())
    inertia = np.sort(np.linalg.eigvalsh(m.moment_inertia))
    return dict(V=float(m.volume), S=float(m.area), M=M, chi=int(m.euler_number), faces=int(len(m.faces)),
                watertight=bool(m.is_watertight), inertia=(inertia / inertia.sum()).tolist())


def voxel_values(prim: sp.Primitive, resolution: int = 64, grid_mm: float = 40.0) -> dict:
    """Minkowski functionals of the voxel grid from the cell counts (eval_minkowski_baseline.cell_counts):
    V = n3 p^3, S = (-6 n3 + 2 n2) p^2, M = pi (3 n3 - 2 n2 + n1) p (the digital mean breadth B equals M / pi for a
    box, checked in the tests of the counts), chi = -n3 + n2 - n1 + n0."""
    from eval_minkowski_baseline import cell_counts
    p = grid_mm / resolution
    occ = sp.occupancy_grid(prim, resolution, grid_mm) > 0
    n3, n2, n1, n0 = cell_counts(occ)
    # the cell-count Euler number is that of the SOLID; its boundary surface has twice it (a ball: 1 -> 2, a tube: 0)
    return dict(V=n3 * p ** 3, S=(-6 * n3 + 2 * n2) * p ** 2, M=PI * (3 * n3 - 2 * n2 + n1) * p, chi=2 * (-n3 + n2 - n1 + n0))


def field_willmore(stl: Path, prim: sp.Primitive, n: int = 6000, offset: float = 0.5, seed: int = 0) -> dict:
    """W = int H^2 dA over the curved faces by Monte Carlo from the closest-point field: area-weighted surface samples of
    the STL moved `offset` mm outward, principal curvatures from the Jacobian of the analytic closest-point map
    (scripts/field_atlas.spectrum); points whose closest point is an edge or vertex (collapsed rank) are skipped - that
    curvature is concentrated on a set of zero area and makes W infinite. Also returns M_faces = int H dA the same way."""
    import field_atlas as fa
    from src.closest_point_field import load_field
    from src.field_channels import surface_points
    field = load_field(stl)
    pts = surface_points(field, n, seed=seed)
    x = pts[:, :3].astype(float) + offset * pts[:, 3:].astype(float)
    d, lam, kappa, _ = fa.spectrum(fa.analytic_closest(prim, eps=1e-6), x, 1e-3)   # small step: thin edge-transition band
    # a point within the difference step h of an edge wedge gets a partly collapsed lambda and a spurious large kappa;
    # a true face at this offset has lambda = 1/(1 + 0.5 kappa) > 0.6 for every radius above 0.33 mm, so 0.6 separates them
    face = (lam > 0.6).all(axis=1)
    H = np.where(face, 0.5 * kappa.sum(axis=1), 0.0)
    return dict(W_field=float(field.area * np.mean(H ** 2)), M_faces_field=float(field.area * np.mean(H)),
                edge_share=float(1 - face.mean()))


def _rot(axis, deg):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def main():
    R = _rot((1, 2, 0.5), 33)
    out = {}
    for name, (stl, prim, params) in SHAPES.items():
        cf = closed_form(params)
        rec = dict(params=params, analytic={**cf, **ratios(cf)})
        mesh = mesh_values(ROOT / "dataset_gen/examples" / f"{stl}.stl")
        rec["mesh"] = {**mesh, **ratios(mesh)}
        rec["field"] = field_willmore(ROOT / "dataset_gen/examples" / f"{stl}.stl", prim)
        for tag, solid in (("voxel", prim), ("voxel_rotated", sp.Placed(prim, R, np.zeros(3)))):
            v = voxel_values(solid)
            rec[tag] = {**v, **ratios(v)}
        out[name] = rec
    path = ROOT / "docs/data/energetic_descriptors.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    cols = ("V", "S", "M", "chi", "W", "q_iso", "q_SM", "q_VM")
    print(f"{'shape':9s} {'source':14s} " + " ".join(f"{c:>9s}" for c in cols))
    for name, rec in out.items():
        for src in ("analytic", "mesh", "voxel", "voxel_rotated"):
            d = rec[src]
            print(f"{name:9s} {src:14s} " + " ".join(f"{d[c]:9.3f}" if c in d else f"{'-':>9s}" for c in cols))
    print("Willmore (curved faces): analytic vs field Monte Carlo")
    for name, rec in out.items():
        print(f"  {name:9s} W {rec['analytic']['W']:8.3f}  field {rec['field']['W_field']:8.3f}   edge share {rec['field']['edge_share']:.3f}")
    print("wrote", path)


if __name__ == "__main__":
    main()
