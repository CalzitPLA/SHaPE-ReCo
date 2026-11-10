#!/usr/bin/env python3
"""
The no-network baseline on the v5 hierarchy (5 parents, 24 subclasses): rotation-invariant global descriptors of the
64^3 occupancy grid plus gradient-boosted trees, on the SAME split the hierarchical Type CNN was trained and tested on
(dataset_gen/local_dataset_v5/small: train -> test, 267 samples; docs/data/type_extra_eval.json: CNN parent accuracy 0.41,
no cone recognised). The question it answers: do the networks learn anything a handful of invariant numbers do not?

Features (the code is reco_engine/type_descriptors.py, imported here; the cascade's "descriptor" type source uses the same
27 numbers): the Minkowski functionals and inertia ratios of scripts/eval_minkowski_baseline.py, plus features aimed at
the subclasses and at the CNN's measured confusions:
  topology    solid components b0, enclosed cavities b2 (background components not touching the border: hollow bodies)
              and tunnels b1 = b0 + b2 - chi (frames, tubes, rhs, tori);
  thickness   quantiles of the Euclidean distance transform inside the solid, over V^(1/3) (plates, walls and struts
              against full bodies);
  convexity   V / V_hull (frames, sectors, bands and carves fill their hull poorly);
  radial      an 8-bin histogram of the distance to the centroid over the RMS radius (sphere vs cube vs torus);
  taper       the cross-section profile along the longest inertia axis: |slope| of sqrt(slab count) and |skewness|
              (a cone tapers, a cylinder does not; the axis sign is unknown, hence the absolute values).

    python3 scripts/eval_descriptor_baseline_v5.py                   # -> docs/data/type_descriptor_baseline_v5.{json,md}
    python3 scripts/eval_descriptor_baseline_v5.py --root dataset_gen/local_dataset_v5/medium
    python3 scripts/eval_descriptor_baseline_v5.py --train-roots inc1 inc2 --test-root test --out ckpt/desc_k2
        (the learning-curve use: the descriptor prior fitted on the same nested increments as the CNN,
         docs/training_plan_v5.tex rule D-type)
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
from reco_engine.type_descriptors import EXTRA_FEATURES as EXTRA, FEATURES, dataset_features as load  # noqa: E402,F401
from reco_engine.type_descriptors import extra_descriptors, features  # noqa: E402,F401
from training.subclass_schema import FLAT_SUBCLASSES, PARENTS, SUBCLASSES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "dataset_gen" / "local_dataset_v5" / "small")
    ap.add_argument("--train-roots", type=Path, nargs="+", default=None,
                    help="training roots (default <root>/train); several are pooled, e.g. nested increments inc1..inck")
    ap.add_argument("--test-root", type=Path, default=None, help="test root (default <root>/test)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "data" / "type_descriptor_baseline_v5")
    args = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier

    t0 = time.time()
    train_roots = args.train_roots or [args.root / "train"]
    test_root = args.test_root or args.root / "test"
    Xtr, ptr, str_, _, sk_tr = load(train_roots)
    Xte, pte, ste, _, sk_te = load(test_root)
    print(f"{len(Xtr)} train, {len(Xte)} test; {len(FEATURES)} features in {time.time() - t0:.0f} s")
    kw = dict(max_iter=300, learning_rate=0.08, random_state=0, class_weight="balanced")
    parent_clf = HistGradientBoostingClassifier(**kw).fit(Xtr, ptr)
    sub_clf = HistGradientBoostingClassifier(**kw).fit(Xtr, str_)

    proba = parent_clf.predict_proba(Xte)
    classes = parent_clf.classes_
    order = np.argsort(-proba, axis=1)
    top1 = classes[order[:, 0]]
    top2 = np.array([pte[i] in classes[order[i, :2]] for i in range(len(pte))])
    # subclass within a parent: the subclass classifier's probabilities masked to that parent's subclasses
    sp = sub_clf.predict_proba(Xte)
    sub_classes = sub_clf.classes_
    parent_of = {j: next(p for p, name in enumerate(PARENTS) if FLAT_SUBCLASSES[j].startswith(name + "/"))
                 for j in range(len(FLAT_SUBCLASSES))}

    def sub_given(parent_idx, i):
        cand = [k for k, j in enumerate(sub_classes) if parent_of[int(j)] == parent_idx]
        return int(sub_classes[cand[int(np.argmax(sp[i, cand]))]]) if cand else -1
    sub_true_parent = np.array([sub_given(int(pte[i]), i) for i in range(len(pte))])
    sub_pred_parent = np.array([sub_given(int(top1[i]), i) for i in range(len(pte))])

    res = dict(root=str(args.root), train_roots=[str(r) for r in train_roots], test_root=str(test_root), n_train=len(Xtr), n_test=len(Xte), features=list(FEATURES),
               skipped=dict(train=dict(sk_tr), test=dict(sk_te)),
               parent_acc=float((top1 == pte).mean()), parent_top2=float(top2.mean()),
               sub_acc_true_parent=float((sub_true_parent == ste).mean()),
               sub_acc_pred_parent=float((sub_pred_parent == ste).mean()),
               chance_parent=1 / len(PARENTS), chance_sub=float(np.mean([1 / len(SUBCLASSES[PARENTS[p]]) for p in pte])),
               per_parent={PARENTS[p]: dict(n=int((pte == p).sum()), acc=float((top1[pte == p] == p).mean()),
                                            sub_acc=float((sub_true_parent[pte == p] == ste[pte == p]).mean()))
                           for p in sorted(set(pte.tolist()))},
               parent_confusion=dict(collections.Counter(f"{PARENTS[a]}->{PARENTS[b]}" for a, b in zip(pte, top1) if a != b)),
               per_subclass={FLAT_SUBCLASSES[j]: dict(n=int((ste == j).sum()), acc=float((sub_true_parent[ste == j] == j).mean()))
                             for j in sorted(set(ste.tolist()))})
    # which features carry the parent decision: accuracy with each alone
    res["single_feature_parent_acc"] = {
        name: float((HistGradientBoostingClassifier(max_iter=150, random_state=0).fit(Xtr[:, [j]], ptr).predict(Xte[:, [j]]) == pte).mean())
        for j, name in enumerate(FEATURES)}
    cnn = ROOT / "docs" / "data" / "type_extra_eval.json"
    if cnn.exists():
        res["cnn_reference"] = "docs/data/type_extra_eval.json"
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    lines = [f"# No-network descriptor baseline (5 parents, 24 subclasses)", "",
             f"train: {', '.join(str(r) for r in train_roots)}; test: {test_root}", "",
             f"{res['n_train']} train / {res['n_test']} test, {len(FEATURES)} rotation-invariant features, gradient-boosted trees "
             "(class-balanced).", "",
             "| metric | descriptors | chance |", "|---|---|---|",
             f"| parent accuracy | {res['parent_acc']:.3f} | {res['chance_parent']:.2f} |",
             f"| parent top-2 | {res['parent_top2']:.3f} | 0.40 |",
             f"| subclass given true parent | {res['sub_acc_true_parent']:.3f} | {res['chance_sub']:.2f} |",
             f"| subclass given predicted parent | {res['sub_acc_pred_parent']:.3f} | |", "",
             "| parent | n | accuracy | subclass acc. |", "|---|---|---|---|"]
    lines += [f"| {p} | {v['n']} | {v['acc']:.3f} | {v['sub_acc']:.3f} |" for p, v in res["per_parent"].items()]
    top = sorted(res["single_feature_parent_acc"].items(), key=lambda kv: -kv[1])[:8]
    lines += ["", "Strongest single features (parent accuracy alone): " + ", ".join(f"{k} {v:.2f}" for k, v in top)]
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
