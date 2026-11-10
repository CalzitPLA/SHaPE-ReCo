#!/usr/bin/env python3
"""Observability check (training plan principle P1) and analytic baselines on a v5 dataset - no network, CPU only.

A dimension or position target is only worth training if the input carries it. On bodies whose label the grid fully
determines (a plain box), the analytic estimate must be (near) exact; where it is not, the gap is either unobservable
(a planar cut removed the evidence) or the room a network could fill (carved, noisy). Reported per stratum, so the
label rule for cut bodies (plan, section "dimensions") is decided from numbers:

  box     half extents measured after the face-normal frame fit (scripts/eval_box_dims.measured_half_extents, the
          registry's `box_dims_measured`), sorted, against the label dims_mm (world mm -> voxels of the 40 mm grid);
          worst-axis relative error max_i exp(|log h_i - log t_i|) - 1, and the volume error.
          box_dims_moments: the same from the second moments, h_i = sqrt(3 lambda_i) (the continuous box's moments; a
          grid-ALIGNED voxel box would need + 1/4 - n voxel centres have variance (n^2 - 1) / 12 - but on the rotated
          v5 bodies that made thin plates worse, median 0.003 -> 0.119, so the staircase samples the continuum; no
          rotation estimate, no whole-voxel rounding: measured_half_extents counts whole voxels after resampling and is
          biased by +0.5..1 voxel per half extent - 4x on a 0.6-voxel plate). On a cut body it is the box with the same
          second moments as what is left: a computable label for option (b) of the plan
  sphere  radius of the volume-equivalent sphere r = (3 V / 4 pi)^(1/3) against dims_mm.radius
  centre  reco_engine.placement.estimate_placement centre of gravity against placement.center_vox (all kinds)

    python3 scripts/check_observability_v5.py dataset_gen/local_dataset_v5/small --json docs/data/v5_small_observability.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from training.manifest import load_manifest  # noqa: E402

GRID_MM, RES = 40.0, 64
VOX_MM = GRID_MM / RES


def stratum(r) -> str:
    p, m = r.placement or {}, r.meta or {}
    if m.get("subclass") not in (None, "full"):
        return m["subclass"]
    if m.get("sample_kind") == "negres":
        return "negres"
    if m.get("source") in ("ibeam_plate", "ibeam_envelope"):
        return m["source"].replace("ibeam_", "")
    if p.get("source") == "ibeam_plate":
        return "plate"
    if p.get("cuts"):
        return "cut"
    if m.get("carve") or p.get("carve"):
        return "carved"
    return "plain"


def summarize(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), median=float(np.median(v)), p90=float(np.quantile(v, 0.9)), max=float(v.max())) if len(v) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", type=Path, help="a v5 dataset root holding train/ val/ test/ (or a single split)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    from eval_box_dims import measured_half_extents
    from reco_engine.placement import estimate_placement
    from training import rotation_symmetry as rs

    splits = [d for d in sorted(args.root.iterdir()) if d.is_dir() and d.name in ("train", "val", "test")] or [args.root]
    recs = [r for s in splits for r in load_manifest(s, require_voxel=True)]
    box_rel, box_vol, box_mom, sph_rel, centre = (collections.defaultdict(list) for _ in range(5))
    for r in recs:
        v = np.fromfile(r.voxel_path, dtype=np.uint8).reshape(RES, RES, RES)
        if not v.any():
            continue
        s = stratum(r)
        dims = (r.meta or {}).get("dims_mm") or {}
        if r.topology == "box" and s in ("plain", "cut", "carved", "plate", "envelope", "negres") and "x_half" in dims:
            t = np.sort([dims["x_half"], dims["y_half"], dims["z_half"]])[::-1] / VOX_MM
            h = measured_half_extents(rs.align_volume(v, "frame")[0])
            box_rel[s].append(float(np.exp(np.abs(np.log(h) - np.log(t))).max() - 1))
            box_vol[s].append(float(abs(np.prod(h) / np.prod(t) - 1)))
            lam = np.sort(np.linalg.eigvalsh(np.cov(np.argwhere(v > 0).astype(float).T, bias=True)))[::-1]
            hm = np.sqrt(3 * np.maximum(lam, 0.0))
            box_mom[s].append(float(np.exp(np.abs(np.log(hm) - np.log(t))).max() - 1))
        if r.topology == "sphere" and "radius" in dims and s in ("plain", "cut", "carved", "negres"):
            rv = (3 * v.sum() / (4 * np.pi)) ** (1 / 3)
            sph_rel[s].append(float(abs(rv / (dims["radius"] / VOX_MM) - 1)))
        cv = (r.placement or {}).get("center_vox")
        if cv is not None:
            pl = estimate_placement(v, RES, GRID_MM, center="cog")
            centre["%s/%s" % (r.topology, s)].append(float(np.linalg.norm(np.asarray(pl.center_of_gravity_vox) - np.asarray(cv))))
    out = dict(root=str(args.root), n_records=len(recs),
               box_dims_rel={k: summarize(v) for k, v in sorted(box_rel.items())},
               box_volume_rel={k: summarize(v) for k, v in sorted(box_vol.items())},
               box_dims_moments_rel={k: summarize(v) for k, v in sorted(box_mom.items())},
               sphere_radius_rel={k: summarize(v) for k, v in sorted(sph_rel.items())},
               centre_cog_vox={k: summarize(v) for k, v in sorted(centre.items())})
    for name in ("box_dims_rel", "box_dims_moments_rel", "box_volume_rel", "sphere_radius_rel", "centre_cog_vox"):
        print(name)
        for k, s in out[name].items():
            if s:
                print("  %-22s n=%4d median %.3f  p90 %.3f  max %.3f" % (k, s["n"], s["median"], s["p90"], s["max"]))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print("wrote", args.json)


if __name__ == "__main__":
    main()
