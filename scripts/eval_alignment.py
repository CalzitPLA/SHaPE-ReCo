#!/usr/bin/env python3
"""
How well do classical alignments find a box's orientation? PCA (second moments, the pipeline's current alignment) against
the face-normal frame fit (training/box_alignment.py), on box samples of the in-scale sets, by kind, cuts and noise.

The error is the symmetry-aware angle to the label: the minimum over the box's symmetry group of the angle between the
estimated frame and the canonical label (training/rotation_symmetry.py) - the number the residual network then starts from.

    python3 scripts/eval_alignment.py --sets test test_carve test_mixed            # held-out test sets
    python3 scripts/eval_alignment.py --train-roots carve rot2 --n 300 --seed 1    # a random sample of the training pool (tuning)
Writes docs/data/alignment_eval.json (--out) unless --no-write.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from training import box_alignment as ba  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402
from training.manifest import load_manifest  # noqa: E402

DATA = ROOT / "dataset_gen" / "local_dataset_inscale"


def kind_of(p: dict) -> str:
    if p.get("carve"):
        return "carved"
    return "cut" if p.get("cuts") else "plain"


def load_boxes(roots, n=None, seed=0):
    recs = []
    for root in roots:
        recs += load_manifest(DATA / root, require_voxel=True)
    recs = [r for r in recs if r.topology == "box" and not (r.placement or {}).get("full_field")]
    if n and len(recs) > n:
        idx = np.random.default_rng(seed).choice(len(recs), n, replace=False)
        recs = [recs[i] for i in sorted(idx)]
    vols = [np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) for r in recs]
    targets = [rs.record_target([r.rotation[k] for k in ("qx", "qy", "qz", "qw")], [r.parameters[k] for k in rs.DIM_KEYS]) for r in recs]
    meta = dict(kind=np.array([kind_of(r.placement or {}) for r in recs]),
                cuts=np.array([len((r.placement or {}).get("cuts", [])) for r in recs]),
                noisy=np.array([(r.placement or {}).get("noise", True) for r in recs]))
    return recs, vols, np.stack([t[0] for t in targets]), np.stack([t[1] for t in targets]), meta


def summarize(err: np.ndarray, mask=None) -> dict:
    e = err if mask is None else err[mask]
    if len(e) == 0:
        return dict(n=0)
    return dict(n=int(len(e)), median=float(np.median(e)), mean=float(e.mean()), within5=float((e < 5).mean()),
                within10=float((e < 10).mean()), within20=float((e < 20).mean()), p90=float(np.percentile(e, 90)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=None, help="test sets (default: test test_carve test_mixed)")
    ap.add_argument("--train-roots", nargs="+", default=None, help="sample boxes from training roots instead")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--out", type=Path, default=ROOT / "docs/data/alignment_eval.json")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    roots = args.train_roots or args.sets or ["test", "test_carve", "test_mixed"]
    recs, vols, M, mask, meta = load_boxes(roots, args.n, args.seed)
    print(f"{len(recs)} boxes from {roots}; kinds { {k: int((meta['kind'] == k).sum()) for k in ('plain', 'cut', 'carved')} }")

    estimators = {"pca": lambda v: rs.pca_orientation(v),
                  "frame": lambda v: ba.frame_orientation(v, starts=args.starts)}
    result = {}
    print(f"{'method':8s} {'ms/vol':>7s} {'median':>7s} {'mean':>6s} {'<5':>5s} {'<10':>5s} {'<20':>5s} {'p90':>6s} | median plain/cut/carved | <10 plain/cut/carved | by cuts 0/1/2 | noisy/clean")
    for name, fn in estimators.items():
        t = time.time()
        est = [fn(v) for v in vols]
        ms = 1000 * (time.time() - t) / len(vols)
        E = np.stack([e if e is not None else np.eye(3) for e in est])
        err = rs.symmetric_error(E, M, mask)
        err_cube = rs.symmetric_error(E, M, np.ones_like(mask))            # modulo all 24 grid rotations: axis order ignored
        s = dict(all=summarize(err), ms_per_volume=ms, all_mod_cube_group=summarize(err_cube))
        s["kind_cut_mod_cube_group"] = summarize(err_cube, meta["kind"] == "cut")
        for k in ("plain", "cut", "carved"):
            s[f"kind_{k}"] = summarize(err, meta["kind"] == k)
        for c in (0, 1, 2):
            s[f"cuts{c}"] = summarize(err, meta["cuts"] == c)
        s["noisy"], s["clean"] = summarize(err, meta["noisy"]), summarize(err, ~meta["noisy"])
        result[name] = dict(summary=s, errors=err.round(3).tolist())
        a = s["all"]
        c24 = s["all_mod_cube_group"]
        print(f"{name:8s} {ms:7.0f} {a['median']:7.1f} {a['mean']:6.1f} {a['within5']:5.2f} {a['within10']:5.2f} {a['within20']:5.2f} {a['p90']:6.1f} | "
              + " / ".join(f"{s[f'kind_{k}'].get('median', float('nan')):.1f}" for k in ("plain", "cut", "carved")) + " | "
              + " / ".join(f"{s[f'kind_{k}'].get('within10', float('nan')):.2f}" for k in ("plain", "cut", "carved")) + " | "
              + " / ".join(f"{s[f'cuts{c}'].get('median', float('nan')):.1f}" for c in (0, 1, 2)) + " | "
              + f"{s['noisy'].get('median', float('nan')):.1f} / {s['clean'].get('median', float('nan')):.1f}"
              + f" | mod cube group: <10 {c24['within10']:.2f} (cut {s['kind_cut_mod_cube_group'].get('within10', float('nan')):.2f})")
    if not args.no_write:
        args.out.write_text(json.dumps(dict(sets=roots, n=len(recs), starts=args.starts, methods=result), indent=1))


if __name__ == "__main__":
    main()
