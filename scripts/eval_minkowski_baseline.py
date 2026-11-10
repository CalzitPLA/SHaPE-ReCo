#!/usr/bin/env python3
"""
A no-CNN baseline for the Type CNN: integral-geometric ("energy"-like) descriptors of the voxel grid plus a
gradient-boosted classifier, on the SAME training roots and held-out test sets as the Type CNN comparison in
docs/RUNNING.md ("Fused input": train carve + cyl, test test_mixed + test_cyl, 800 samples).

Descriptors of one 64^3 occupancy grid (every number rotation- and translation-invariant up to voxelisation; the code is
reco_engine/type_descriptors.py's `cell_counts` / `minkowski_descriptors`, imported here):
  Minkowski functionals of the union of closed voxel cubes, from the counts of its cubes n3, faces n2, edges n1 and
  vertices n0 (Michielsen & De Raedt 2001, the open-cell formulas):
      volume V = n3,  surface S = -6 n3 + 2 n2,  mean breadth B = 3 n3 - 2 n2 + n1,  Euler chi = -n3 + n2 - n1 + n0
  the dimensionless ratios S^3 / V^2, B^3 / V, B S / V, chi (they separate a ball from a slab from a rod whatever the size),
  and the three eigenvalues of the inertia tensor divided by their sum (the shape of the mass distribution).
  Staircase note: S and B of a voxel grid overestimate the true values by an orientation-dependent factor (a plane at
  45 degrees reads ~1.4x its area) - the price of an exact, cheap count; measured here, not corrected.

    python3 scripts/eval_minkowski_baseline.py                  # -> docs/data/type_minkowski_baseline.json
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.manifest import load_manifest  # noqa: E402

from reco_engine.type_descriptors import (MINKOWSKI_FEATURES, cell_counts, minkowski_descriptors)  # noqa: E402,F401

DATA = ROOT / "dataset_gen" / "local_dataset_inscale"
FEATURES = MINKOWSKI_FEATURES
descriptors = minkowski_descriptors          # the feature code lives in reco_engine/type_descriptors.py (imported, not copied)


def load(records):
    X = np.stack([descriptors(np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) > 0) for r in records])
    y = np.array([r.topology for r in records])
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", default=["carve", "cyl"])
    ap.add_argument("--test", nargs="+", default=["test_mixed", "test_cyl"])
    ap.add_argument("--out", type=Path, default=ROOT / "docs/data/type_minkowski_baseline.json")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    t = time.time()
    train = [r for n in args.train for r in load_manifest(DATA / n, require_voxel=True)]
    test = [r for n in args.test for r in load_manifest(DATA / n, require_voxel=True)]
    Xtr, ytr = load(train)
    Xte, yte = load(test)
    print(f"{len(train)} train, {len(test)} test samples; descriptors in {time.time() - t:.0f} s")
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, random_state=0).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float((pred == yte).mean())
    per_class = {c: float((pred[yte == c] == c).mean()) for c in sorted(set(yte))}
    confusions = collections.Counter(f"{a}->{b}" for a, b in zip(yte, pred) if a != b)
    # which descriptor carries the decision: accuracy with each one alone
    alone = {}
    for j, name in enumerate(FEATURES):
        c1 = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr[:, [j]], ytr)
        alone[name] = float((c1.predict(Xte[:, [j]]) == yte).mean())
    means = {c: dict(zip(FEATURES, np.round(Xtr[ytr == c].mean(0), 4).tolist())) for c in sorted(set(ytr))}
    result = dict(train_roots=args.train, test_roots=args.test, n_train=len(train), n_test=len(test), accuracy=acc,
                  per_class=per_class, confusions=dict(confusions), single_feature_accuracy=alone, class_means=means,
                  features=list(FEATURES))
    print(f"accuracy {100 * acc:.1f} %  per class {per_class}")
    print("single-feature accuracy", {k: round(v, 3) for k, v in alone.items()})
    args.out.write_text(json.dumps(result, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
