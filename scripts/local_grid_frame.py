#!/usr/bin/env python3
"""
R1 on the frame: does a parity grid whose pitch comes from the part's own wall thickness see the part's topology, where
the fixed-resolution grids do not? (docs/FRAME_FAILURE.md, "Thickness-adaptive local grids").

For the whole part and for crops (the straight members `reco_engine.ibeam.detect_members` finds, and a regular tiling),
the Betti numbers (b0, b1, b2) and the volume of the parity grid are compared with the EXACT ones of the STEP B-rep
intersected with the same box (freecadcmd, scripts/freecad_crop_topology.py). Grids: the fixed cube resolutions the loops
use (pitch = longest side / R) and the adaptive pitch tau/k (tau = 5th percentile of the wall thickness), at two grid
phases.

    python3 scripts/local_grid_frame.py                   # the frame (STL x 100 = mm, STEP reference)
    python3 scripts/local_grid_frame.py --json docs/data/local_grid_frame.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.freecad_path import freecadcmd_or_default  # noqa: E402  ($SHAPE_RECO_FREECADCMD, PATH, /usr/bin)

from dataset_gen.src.closest_point_field import load_mesh  # noqa: E402
from dataset_gen.src.local_grids import adaptive_pitch, grid_topology, wall_thickness  # noqa: E402

D = Path("~/shape_reco_parts/benchmark_parts/1.0")


def exact_topology(step: Path, boxes) -> list:
    with tempfile.TemporaryDirectory() as tmp:
        job = Path(tmp) / "job.json"
        out = Path(tmp) / "out.json"
        job.write_text(json.dumps(dict(step=str(step), boxes=boxes, out=str(out))))
        subprocess.run([freecadcmd_or_default(), str(ROOT / "scripts" / "freecad_crop_topology.py")], check=True,
                       env=dict(os.environ, CROP_TOPOLOGY_JOB=str(job)), capture_output=True)
        return json.loads(out.read_text())


def member_boxes(vertices, faces, pad: float):
    """World AABBs of the detected straight members (each section's uv box swept from start to end)."""
    from reco_engine.ibeam import detect_members
    boxes, labels = [], []
    for m in detect_members((vertices, faces)):
        umin, vmin, umax, vmax = m.bbox
        corners = np.array([m.to_world((u, v), s) for u in (umin, umax) for v in (vmin, vmax) for s in (m.start, m.end)])
        lo, hi = corners.min(axis=0) - pad, corners.max(axis=0) + pad
        boxes.append([float(x) for x in (*lo, *hi)])
        labels.append(f"{m.kind} {m.section.label if m.section is not None else ''} L={m.length:.0f}")
    return boxes, labels


def tile_boxes(lo, hi, n=(1, 3, 2)):
    """A regular tiling of the bounding box, cut at irrational fractions so no cut plane lies on a model face."""
    edges = [np.linspace(lo[d], hi[d], n[d] + 1) for d in range(3)]
    for d in range(3):
        edges[d][1:-1] += 0.01234567 * (hi[d] - lo[d])
    boxes = []
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                boxes.append([float(edges[0][i]), float(edges[1][j]), float(edges[2][k]),
                              float(edges[0][i + 1]), float(edges[1][j + 1]), float(edges[2][k + 1])])
    return boxes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl", type=Path, default=D / "Unbenannt-Frame_4_x_6.stl")
    ap.add_argument("--step", type=Path, default=D / "Frame_4_x_6.stp")
    ap.add_argument("--scale", type=float, default=100.0, help="STL units -> mm (the frame's STL is 1/100 of the STEP)")
    ap.add_argument("--fixed", type=int, nargs="*", default=[64, 128, 256], help="fixed cube resolutions")
    ap.add_argument("--k", type=float, nargs="*", default=[1.0, 2.0, 3.0], help="adaptive pitches tau/k (whole part)")
    ap.add_argument("--crop-k", type=float, nargs="*", default=[3.0], help="adaptive pitches tau/k for the crops")
    ap.add_argument("--no-members", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    V, F = load_mesh(args.stl)
    V = V * args.scale
    lo, hi = V.min(axis=0), V.max(axis=0)
    L = float((hi - lo).max())
    t0 = time.time()
    _, _, tau = wall_thickness(V, F, n=4000)
    tau_q = float(np.percentile(tau[np.isfinite(tau)], 5))
    modes = np.round(np.percentile(tau[np.isfinite(tau)], [5, 25, 50, 75]), 2).tolist()
    print(f"wall thickness: 5th pct {tau_q:.2f} mm, quartiles {modes} ({time.time() - t0:.1f} s)", flush=True)

    rng = np.random.default_rng(0)
    rows = []

    def run(key, name, box, pitch, crop, phase):
        b_lo, b_hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
        shift = phase * pitch
        t = time.time()
        topo = grid_topology(V, F, b_lo + shift if not crop else b_lo, b_hi + shift if not crop else b_hi, pitch,
                             crop=crop)
        rec = dict(key=key, region=name, pitch=pitch, phase=[float(x) for x in phase], seconds=round(time.time() - t, 1),
                   **{k: v for k, v in topo.to_record().items() if k not in ("pitch",)})
        rows.append(rec)
        print(f"  {name:>28s} pitch {pitch:7.2f} mm  b=({topo.b0},{topo.b1},{topo.b2})  vox {topo.voxels:>10d}  "
              f"vol {topo.volume:.3e}  {rec['seconds']} s", flush=True)
        return rec

    # the exact references
    boxes = [None]
    names = ["whole"]
    if not args.no_members:
        mb, ml = member_boxes(V, F, pad=-1.234)   # shrunk: a crop face never lies on a member face
        print(f"members: {len(mb)}", flush=True)
        for i, (b, lab) in enumerate(zip(mb, ml)):
            boxes.append(b); names.append(f"member{i:02d} {lab}")
    for i, b in enumerate(tile_boxes(lo, hi)):
        boxes.append(b); names.append(f"tile{i:02d}")
    t = time.time()
    exact = exact_topology(args.step, boxes)
    print(f"exact (freecadcmd): {time.time() - t:.1f} s", flush=True)

    whole = [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
    print(f"whole part: exact b=({exact[0]['b0']},{exact[0]['b1']},{exact[0]['b2']}), vol {exact[0]['volume']:.3e}")
    for R in args.fixed:
        for ph in (np.zeros(3), rng.random(3)):
            run(0, f"whole R={R}", whole, L / R, False, ph)
    for k in args.k:
        for ph in (np.zeros(3), rng.random(3)):
            run(0, f"whole tau/{k:g}", whole, tau_q / k, False, ph)

    for key, (name, box, ex) in enumerate(zip(names[1:], boxes[1:], exact[1:]), start=1):
        print(f"{name}: exact b=({ex['b0']},{ex['b1']},{ex['b2']}), vol {ex['volume']:.3e}", flush=True)
        for R in args.fixed[:1]:
            run(key, f"{name.split()[0]} R={R}", box, L / R, True, np.zeros(3))
        for k in args.crop_k:
            run(key, f"{name.split()[0]} tau/{k:g}", box, tau_q / k, True, np.zeros(3))

    # verdicts
    for r in rows:
        e = exact[r["key"]]
        r["exact"] = dict(b0=e["b0"], b1=e["b1"], b2=e["b2"], volume=e["volume"])
        r["topology_ok"] = (r["b0"], r["b1"], r["b2"]) == (e["b0"], e["b1"], e["b2"])
        r["volume_ratio"] = r["volume"] / e["volume"] if e["volume"] else None
    print("\nregion / grid                          ok   b (grid)     b (exact)   vol ratio")
    for r in rows:
        e = r["exact"]
        print(f"{r['region']:<38s} {'yes' if r['topology_ok'] else 'NO ':<4s} ({r['b0']},{r['b1']},{r['b2']})"
              f"{'':<6s}({e['b0']},{e['b1']},{e['b2']})     {r['volume_ratio'] if r['volume_ratio'] is None else round(r['volume_ratio'], 3)}")
    if args.json:
        args.json.write_text(json.dumps(dict(stl=str(args.stl), step=str(args.step), scale=args.scale, regions=names,
                                             boxes=boxes,
                                             tau_p5=tau_q, tau_quartiles=modes, rows=rows), indent=1))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
