#!/usr/bin/env python3
"""Second-order against linear surface meshes on solids with a known surface: the error of the closest-point distance
and of the normal, by element count.

    python3 scripts/measure_second_order.py [--json docs/data/second_order_mesh.json]

Solids (built by freecadcmd): a sphere r = 10 and a cylinder r = 10, h = 30 (its lateral surface is tested). Query
points: 20000 seeded points within 2 mm of the tested surface. Truth is analytic.
"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.closest_point_field import find_freecadcmd, load_mesh  # noqa: E402
from src.quadratic_mesh import QuadraticSurface, mesh_step, signed_volume, subdivide  # noqa: E402

R, H = 10.0, 30.0


def make_steps(tmp: Path):
    script = tmp / "mk.py"
    script.write_text("import Part\nPart.makeSphere(10).exportStep('sphere.step')\nPart.makeCylinder(10, 30).exportStep('cylinder.step')\n")
    subprocess.run([find_freecadcmd(), str(script)], cwd=tmp, capture_output=True, timeout=120, check=True)
    return tmp / "sphere.step", tmp / "cylinder.step"


def points(kind, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    if kind == "sphere":
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        r = R + rng.uniform(-2, 2, n)
        return d * r[:, None], np.abs(r - R), d
    d[:, 2] = 0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    r = R + rng.uniform(-2, 2, n)
    p = d * r[:, None]
    p[:, 2] = rng.uniform(3, H - 3, n)             # away from the caps: the lateral surface is the nearest
    return p, np.abs(r - R), d


def angle(n, truth):
    return np.degrees(np.arccos(np.clip(np.abs((n * truth).sum(1)), 0, 1)))


def linear_row(name, v, f, p, dist, nrm):
    import igl
    t0 = time.time()
    d2, face, c = igl.point_mesh_squared_distance(p, np.ascontiguousarray(v, dtype=float), np.ascontiguousarray(f, dtype=np.int64))
    sec = time.time() - t0
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-300)
    e = np.abs(np.sqrt(np.maximum(d2, 0)) - dist)
    a = angle(fn[face], nrm)
    return dict(mesh=name, triangles=int(len(f)), nodes=int(len(v)), dist_err_mean=float(e.mean()), dist_err_max=float(e.max()),
                normal_err_mean_deg=float(a.mean()), normal_err_max_deg=float(a.max()), query_s=round(sec, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="so_"))
    out = {}
    for kind, step in zip(("sphere", "cylinder"), make_steps(tmp)):
        p, dist, nrm = points(kind)
        rows = []
        v, f = load_mesh(step, 0.02)
        rows.append(linear_row("FreeCAD tessellation, deflection 0.02 (today)", v, f, p, dist, nrm))
        for size in (8.0, 4.0, 2.0, 1.0):
            v1, f1 = mesh_step(step, size, order=1)
            rows.append(linear_row(f"gmsh order 1, size {size:g}", v1, f1, p, dist, nrm))
            t0 = time.time()
            n2, t6 = mesh_step(step, size, order=2)
            mesh_s = time.time() - t0
            vs, fs, _, _ = subdivide(n2, t6, 1)
            rows.append(linear_row(f"gmsh order 2, size {size:g}, as 4 flat triangles", vs, fs, p, dist, nrm))
            s = QuadraticSurface(n2, t6)
            t0 = time.time()
            q = s.query(p)
            sec = time.time() - t0
            e, a = np.abs(q.distance - dist), angle(q.normal, nrm)
            rows.append(dict(mesh=f"gmsh order 2, size {size:g}, quadratic patches", triangles=int(len(t6)), nodes=int(len(n2)),
                             dist_err_mean=float(e.mean()), dist_err_max=float(e.max()), normal_err_mean_deg=float(a.mean()),
                             normal_err_max_deg=float(a.max()), query_s=round(sec, 3), mesh_s=round(mesh_s, 3),
                             newton_mean=float(q.iterations.mean()), newton_max=int(q.iterations.max()),
                             volume_flat=signed_volume(vs, fs)))
        out[kind] = rows
        print(f"\n{kind} (r = {R:g}; 20000 points within 2 mm of the surface)")
        print(f"{'mesh':52s} {'elements':>9s} {'nodes':>7s} {'dist err mean':>14s} {'max':>10s} {'normal err mean':>16s} {'max':>8s} {'query s':>8s}")
        for r in rows:
            print(f"{r['mesh']:52s} {r['triangles']:9d} {r['nodes']:7d} {r['dist_err_mean']:14.3e} {r['dist_err_max']:10.3e} "
                  f"{r['normal_err_mean_deg']:16.4f} {r['normal_err_max_deg']:8.3f} {r['query_s']:8.3f}")
    if args.json:
        Path(args.json).write_text(json.dumps(dict(date=time.strftime("%Y-%m-%d"), radius=R, points=20000, band_mm=2.0, results=out), indent=1))
        print("written", args.json)


if __name__ == "__main__":
    main()
