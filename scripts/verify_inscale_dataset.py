#!/usr/bin/env python3
"""
Independent verification of an in-scale dataset that was generated with --carve-prob / --require-manifold.

  1. metadata: counts per topology, carve / cut / full-field shares, every sample's recorded `manifold.ok`,
     cutter/topology pairing (box - sphere labelled box, sphere - box labelled sphere), removed fractions;
  2. reconstruction: for a random sample the CLEAN body is rebuilt from the STL + the recorded placement, hollow,
     carve or cuts (not from the stored grid, which carries noise) and re-tested with manifold_report: one piece,
     no enclosed void (one piece, one cavity for a hollow sample - that's the point, not a defect there), and the
     same voxel count as recorded;
  3. stored grids vs clean body: how much of the clean body survives the noise (recall) and how many stored grids
     have more than one piece AFTER noise (expected: the noise is speckle, not part of the part);
  4. boxes: the label orientation is right - the clean voxels lie inside the analytic base box built from the
     label (canonical orientation, dimensions, fitted size);
  5. field.npz (generated with --with-field, or by src/sample_writer.py): finite everywhere, and its occupancy channel (the
     parity sign of the exact mesh field, world axes) agrees with the voxels of the same FreeCAD solid (by the voxelizer
     the record names, placement.voxelizer; records without one are stltovoxel-era) at
     >= 99 % of all voxel centres of every sample, no disagreement farther than one voxel from the surface, and on bodies
     of >= 5000 voxels IoU >= 0.8 (see check_field: the stored grids are up to half a voxel off the world grid). Samples
     with a field are rebuilt from their own placed solid STL (<stem>.solid.stl, or the STL itself for a negative
     residual), not replayed from the base STL + recorded draw.

    python3 scripts/verify_inscale_dataset.py dataset_gen/local_dataset_inscale/carve --samples 150
Exit status 1 if any check fails.
"""
import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dataset_gen"))
sys.path.insert(0, str(ROOT))

from src import inscale  # noqa: E402
from src.voxelization import find_stltovoxel, voxelize_stl_to_bin  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402


def entries(root: Path):
    for meta in sorted(root.rglob("dataset_metadata.jsonl")):
        if meta.parent == root or meta.parent.parent == root:      # the root's own / a shard's own file duplicates the leaf ones
            continue
        base = meta.parent.parent                                   # shard dir: paths in the file are relative to it
        for line in meta.read_text().splitlines():
            e = json.loads(line)
            if e.get("record_type") != "dataset_header" and e.get("voxel_path"):
                yield base, e


def solid_stl_of(base: Path, e: dict):
    """The placed FreeCAD solid a --with-field / sample_writer record was voxelised from, or None (old-style record)."""
    pl = e.get("placement") or {}
    f = pl.get("field") or {}
    if f.get("solid_stl"):
        return (base / e["stl_path"]).with_name(f["solid_stl"])
    if pl.get("stl_placed") or e.get("sample_kind") == "negres":
        return base / e["stl_path"]
    return None


def voxelizer_of(e: dict) -> str:
    """The voxelizer that made the record's grid: placement.voxelizer, "stltovoxel" for records older than that key."""
    return (e.get("placement") or {}).get("voxelizer") or "stltovoxel"


def clean_body(base: Path, e: dict, grid_mm: float, res: int):
    """Rebuild the clean (noise-free) voxel body from the STL and the recorded draw - or, for a record with its own
    placed solid (solid_stl_of), by voxelising that solid (the modification is already in it). Voxelized by the record's
    own voxelizer, so the voxel counts compare like with like."""
    solid = solid_stl_of(base, e)
    vox = voxelizer_of(e)
    if solid is not None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "p.bin"
            if not voxelize_stl_to_bin(str(solid), str(out), resolution=res, scale_mode="fixed", scale_mm=grid_mm, anchor="origin",
                                       voxelizer=vox):
                raise RuntimeError(f"voxelization failed: {solid}")
            return (np.fromfile(out, dtype=np.uint8).reshape((res,) * 3) > 0).astype(np.uint8)
    stl = base / e["stl_path"]
    pl = e["placement"]
    center, _ = inscale.bounds_center_extents(stl)
    with tempfile.TemporaryDirectory() as tmp:
        placed, out = Path(tmp) / "p.stl", Path(tmp) / "p.bin"
        inscale.write_placed_stl(stl, placed, pl, center)
        if not voxelize_stl_to_bin(str(placed), str(out), resolution=res, scale_mode="fixed", scale_mm=grid_mm, anchor="origin",
                                   voxelizer=vox):
            raise RuntimeError(f"voxelization failed: {stl}")
        vol = np.fromfile(out, dtype=np.uint8).reshape((res,) * 3)
    if "hollow" in pl:
        from scipy import ndimage
        occ = vol > 0
        eroded = ndimage.binary_erosion(occ, iterations=pl["hollow"]["wall_voxels"])
        vol = occ & ~eroded                                                   # the same erosion inscale.hollow() drew,
                                                                                # replayed from its recorded wall_voxels
    elif "carve" in pl:
        c = pl["carve"]
        grid = np.indices(vol.shape).reshape(3, -1).T.astype(float)
        if c["cutter"] == "sphere":
            hit = (((grid - c["centre"]) ** 2).sum(axis=1) <= c["radius"] ** 2)
        else:
            hit = (np.abs((grid - c["centre"]) @ np.array(c["rotation"])) <= c["half_edge"]).all(axis=1)
        vol = (vol > 0) & ~hit.reshape(vol.shape)
    elif pl.get("cuts"):
        vol, _ = inscale.apply_cuts(vol, pl["cuts"], min_thickness=4.0)
    return (vol > 0).astype(np.uint8)


def check_field(base: Path, e: dict, clean: np.ndarray, stats: dict, min_agree: float = 0.99, max_off_voxels: float = 1.0,
                min_iou: float = 0.8, iou_min_voxels: int = 5000):
    """Finiteness + sign/occupancy agreement of one sample's field.npz with the voxels of the same clean solid.

    Parity voxels (the default since 2026-10-01) are the field's own sign, so they agree exactly. stltovoxel ones cannot:
    voxelization.py (anchor="origin") voxelises in the MESH's own bounding-box frame and
    ROUNDS its offset to whole voxels, so a stored grid sits up to half a voxel off the true world grid per axis, while the
    field is sampled at the exact world voxel centres (measured on the smoke run: symmetric disagreements, every one
    within 0.88 voxel of the surface; the best whole-grid shift differs per sample, e.g. (-0.5, 0, -0.5) or (0, -0.5,
    -0.5) voxels). So: >= min_agree of all centres agree, every disagreeing centre is within max_off_voxels of the
    surface (a sign error far from the surface is a real failure), and on bodies of at least iou_min_voxels voxels the
    material IoU is >= min_iou (a half-voxel shift alone costs 0.10-0.13 IoU on the thin-walled hollow boxes of the smoke
    run, 0.87-0.90 measured, and far more on thin tori; the distance check above is the strict one)."""
    from src.field_channels import bin_to_world, load_field_npz
    path = (base / e["stl_path"]).with_name(e["placement"]["field"]["path"])
    channels, spec, _ = load_field_npz(path)
    stats["fields"] += 1
    if not np.isfinite(channels).all():
        stats["field_nonfinite"] += 1
    vox = bin_to_world(clean > 0)
    occ = channels[1] > 0                                             # the sign itself, not only the stored occupancy
    agree = float((occ == vox).mean())
    iou = float((occ & vox).sum() / max((occ | vox).sum(), 1))
    stats["field_agree"].append(agree)
    stats["field_iou"].append(iou)
    far = float(np.abs(channels[1][occ != vox]).max() / spec.pitch) if (occ != vox).any() else 0.0
    stats.setdefault("field_far", []).append(far)
    big = int(vox.sum()) >= iou_min_voxels
    if agree < min_agree or far > max_off_voxels or (big and iou < min_iou) or not np.array_equal(occ, channels[0] > 0):
        stats["field_bad"].append(f"{e['stl_path']} agree={agree:.4f} iou={iou:.3f} farthest disagreement {far:.2f} voxel")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--grid-mm", type=float, default=40.0)
    ap.add_argument("--resolution", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = args.root.resolve()
    rows = list(entries(root))
    if find_stltovoxel() is None and any(voxelizer_of(e) == "stltovoxel" for _, e in rows):
        print("stltovoxel not found (needed for this dataset's stltovoxel-era records)"); return 2
    negres = [(b, e) for b, e in rows if e.get("sample_kind") == "negres"]
    rows_all, rows = rows, [(b, e) for b, e in rows if e.get("sample_kind") != "negres"]     # negres: own checks below
    failures = []
    print(f"{root}: {len(rows)} samples")
    by = {}
    for base, e in rows:
        pl = e["placement"]
        d = by.setdefault(e["topology"], dict(n=0, carved=0, cut=0, failed=0, bad_manifold=0, wrong_cutter=0, draws=[],
                                              removed=[], full=0, hollow=0, bad_hollow=0))
        d["n"] += 1
        d["carved"] += "carve" in pl
        d["cut"] += bool(pl.get("cuts"))
        d["failed"] += bool(pl.get("carve_failed"))
        d["full"] += bool(pl.get("full_field"))
        d["hollow"] += "hollow" in pl
        if "hollow" in pl:
            # hollow.manifold.ok is False BY DESIGN (one component, one cavity is the point - see inscale.hollow) -
            # that is not a defect, unlike every other sample kind, where manifold.ok must be True
            m = pl.get("manifold", {})
            d["bad_hollow"] += not (m.get("components") == 1 and m.get("cavities") == 1)
        else:
            d["bad_manifold"] += not pl.get("manifold", {}).get("ok", False)
        if "carve" in pl:
            want = "sphere" if e["topology"] == "box" else "box"
            d["wrong_cutter"] += pl["carve"]["cutter"] != want
            d["removed"].append(pl["carve"]["removed_fraction"])
        d["draws"].append(pl.get("draws", 1))
    # noise must not correlate with class or kind: noisy share per (topology, kind)
    groups = {}
    for base, e in rows:
        pl = e["placement"]
        kind = "full-field" if pl.get("full_field") else "carved" if "carve" in pl else "cut" if pl.get("cuts") else "plain"
        groups.setdefault((e["topology"], kind), []).append(bool(pl.get("noise", True)))
    print("  share of samples with noise, by class and kind (must be similar everywhere, and neither 0 nor 1):")
    shares = []
    for (t, kind), flags in sorted(groups.items()):
        share = float(np.mean(flags))
        print(f"    {t:6s} {kind:10s} n={len(flags):5d} noisy={share:.2f}")
        if len(flags) >= 30:
            shares.append(share)
            if share in (0.0, 1.0):
                failures.append(f"{t}/{kind}: every sample is {'noisy' if share else 'clean'}")
    if len(shares) > 1 and max(shares) - min(shares) > 0.2:
        failures.append(f"noise share differs by {max(shares) - min(shares):.2f} between groups")
    for t, d in by.items():
        r = np.array(d["removed"]) if d["removed"] else np.array([np.nan])
        print(f"  {t:6s} n={d['n']:5d} carved={d['carved']/d['n']:.2f} planar-cut={d['cut']/d['n']:.2f} carve_failed={d['failed']} "
              f"full-field={d['full']} hollow={d['hollow']} | manifold flag NOT ok (excl. hollow): {d['bad_manifold']} | "
              f"hollow not (1 piece, 1 cavity): {d['bad_hollow']} | wrong cutter for label: {d['wrong_cutter']} | "
              f"removed fraction 5/50/95%: {np.round(np.nanquantile(r, [.05, .5, .95]), 2)} | redraws: "
              f"{np.mean(np.array(d['draws']) > 1):.3f} of samples, max {max(d['draws'])}")
        if d["bad_manifold"] or d["bad_hollow"] or d["wrong_cutter"]:
            failures.append(f"{t}: {d['bad_manifold']} not manifold, {d['bad_hollow']} bad hollow, {d['wrong_cutter']} wrong cutter")

    if negres:
        bad = [e["stl_path"] for _, e in negres if not (e.get("placement") or {}).get("manifold", {}).get("ok", False)]
        print(f"  negative residuals: {len(negres)} (cutter kinds {sorted(set(e['topology'] for _, e in negres))}), "
              f"not one uniform manifold: {len(bad)}")
        # a cutter bite can legitimately fall apart into pieces only if the base is non-convex; with box/sphere bases it
        # should not - report it, but do not fail on it

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.samples, len(rows)))
    sample += rng.sample(negres, min(max(args.samples // 4, 1), len(negres))) if negres else []
    stats = dict(fields=0, field_nonfinite=0, field_agree=[], field_iou=[], field_bad=[], noised=0, noised_differs=0, clean_stored=0, clean_stored_ok=0, n=0, not_ok=0, count_mismatch=0, recall=[], noisy_multi=0, inside=[], boxes=0)
    for base, e in sample:
        pl = e["placement"]
        if pl.get("full_field"):
            continue
        clean = clean_body(base, e, args.grid_mm, args.resolution)
        if (pl.get("field") or {}).get("path"):
            check_field(base, e, clean, stats)
        if e.get("sample_kind") == "negres":
            continue
        rep = inscale.manifold_report(clean)
        stats["n"] += 1
        if "hollow" in pl:                                                    # one piece, one cavity is the POINT here
            stats["not_ok"] += not (rep["components"] == 1 and rep["cavities"] == 1)
        else:
            stats["not_ok"] += not rep["ok"]
        stats["count_mismatch"] += abs(int(clean.sum()) - pl["voxels"]) > 0.005 * pl["voxels"]
        stored = np.fromfile(base / e["voxel_path"], dtype=np.uint8).reshape(clean.shape)
        stats["recall"].append(float((stored[clean > 0] > 0).mean()))
        stats["noisy_multi"] += inscale.manifold_report(stored)["components"] > 1
        if pl.get("noise") is False:                                          # stored grid must BE the clean part
            stats["clean_stored"] += 1
            rep_s = inscale.manifold_report(stored)
            whole = (rep_s["components"] == 1 and rep_s["cavities"] == 1) if "hollow" in pl else rep_s["ok"]   # hollow: by design
            stats["clean_stored_ok"] += bool(np.array_equal(stored > 0, clean > 0)) and whole
        elif pl.get("noise") is True:                                         # a noised sample must actually differ
            stats["noised"] += 1
            stats["noised_differs"] += not np.array_equal(stored > 0, clean > 0)
        if e["topology"] == "box":
            d = [e["parameters"][k] for k in rs.DIM_KEYS]
            M, _ = rs.record_target([e["rotation"][k] for k in ("qx", "qy", "qz", "qw")], d)
            sd = rs.canonical_axes(d)[1]
            s = (pl["voxels_uncut"] / (8 * sd.prod())) ** (1 / 3)
            idx = np.argwhere(clean > 0).astype(float)
            centre = np.array(pl["center_vox"])
            inside = (np.abs((idx - centre) @ M) <= s * sd + 1.0).all(axis=1).mean()
            stats["inside"].append(float(inside)); stats["boxes"] += 1
    print(f"reconstruction of {stats['n']} random clean bodies from STL + recorded draw:")
    print(f"  one uniform manifold: {stats['n'] - stats['not_ok']}/{stats['n']}   voxel count equals the recorded one (0.5 %): "
          f"{stats['n'] - stats['count_mismatch']}/{stats['n']}")
    print(f"  clean body surviving in the stored grid (recall): median {np.median(stats['recall']):.3f}, 5th pct {np.quantile(stats['recall'], .05):.3f}, "
          f"min {min(stats['recall']):.3f}, share below 0.7: {np.mean(np.array(stats['recall']) < 0.7):.3f}; "
          f"stored grids with >1 piece after noise: {stats['noisy_multi']}/{stats['n']} (noise speckle, expected)")
    if stats["noised"]:
        print(f"  noised samples whose stored grid differs from the clean body: {stats['noised_differs']}/{stats['noised']}")
    if stats["clean_stored"]:
        print(f"  samples stored WITHOUT noise: {stats['clean_stored']}, of which the stored grid equals the "
              f"rebuilt clean body and is one uniform manifold: {stats['clean_stored_ok']}")
        if stats["clean_stored_ok"] != stats["clean_stored"]:
            failures.append(f"{stats['clean_stored'] - stats['clean_stored_ok']} un-noised stored grids differ from the clean body / are not manifold")
    if stats["boxes"]:
        print(f"  box label check: clean voxels inside the analytic base box: median {np.median(stats['inside']):.3f}, "
              f"5th pct {np.quantile(stats['inside'], .05):.3f} (n={stats['boxes']})")
        if np.quantile(stats["inside"], .05) < 0.9:
            failures.append("box orientation label: >5 % of samples have <90 % of the body inside the analytic box")
    if stats["fields"]:
        agree, iou = np.array(stats["field_agree"]), np.array(stats["field_iou"])
        print(f"  field.npz: {stats['fields']} checked, non-finite: {stats['field_nonfinite']}; occupancy vs voxels of the same "
              f"solid: agreement min {agree.min():.4f} median {np.median(agree):.4f}, material IoU min {iou.min():.3f} "
              f"median {np.median(iou):.3f}; farthest disagreement from the surface {max(stats['field_far']):.2f} voxel")
        if stats["field_nonfinite"]:
            failures.append(f"{stats['field_nonfinite']} field.npz with non-finite values")
        if stats["field_bad"]:
            failures.append(f"{len(stats['field_bad'])} fields disagree with their voxels (<99 % of centres, a disagreement > 1 voxel from the surface, or IoU < 0.8 on a big body): "
                            f"{stats['field_bad'][:3]}")
    if stats["not_ok"] or stats["count_mismatch"]:
        failures.append(f"{stats['not_ok']} rebuilt bodies not manifold, {stats['count_mismatch']} voxel-count mismatches")
    print("RESULT:", "FAIL - " + "; ".join(failures) if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
