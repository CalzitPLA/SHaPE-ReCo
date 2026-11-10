#!/usr/bin/env python3
"""
The network-free type source of the cascade: 27 rotation-invariant descriptors of a 64^3 occupancy grid plus
gradient-boosted trees over the v5 hierarchy (5 parents, 24 subclasses; training/subclass_schema.py).

Why it is a library and not only a baseline script: on the v5 hierarchy the descriptors reach a parent accuracy of
0.850 (small) and 0.780 (noised medium) where the hierarchical occupancy CNN reached 0.408 on small
(docs/data/type_descriptor_baseline_v5*.json), so they are a type source in their own right - registry mode "descriptor"
in reco_engine/cascade.py, fused with the CNN entries. scripts/eval_descriptor_baseline_v5.py and
scripts/eval_minkowski_baseline.py import the feature code from here.

Features of one occupancy grid (every number invariant to rotation and translation up to voxelisation):
  Minkowski   the functionals of the union of closed voxel cubes from its cube / face / edge / vertex counts
              (Michielsen & De Raedt 2001): log V, the dimensionless ratios S^3/V^2, B^3/V, BS/V, the Euler number chi,
              and the three inertia eigenvalues over their sum                                      (`minkowski_descriptors`)
  topology    solid components b0, enclosed cavities b2 (background components not touching the border) and tunnels
              b1 = b0 + b2 - chi (frames, tubes, rhs, tori), the cavity volume share
  thickness   quantiles of the Euclidean distance transform inside the solid, over V^(1/3)
  convexity   V / V_hull
  radial      an 8-bin histogram of the distance to the centroid over the RMS radius
  taper       the cross-section profile along the longest inertia axis: |slope| of sqrt(slab count) and |skewness|
                                                                                                     (`extra_descriptors`)

    python3 -m reco_engine.type_descriptors --fit dataset_gen/local_dataset_v5/medium/train \\
                                            --out models/type_descriptors_v5_medium.joblib [--eval <test root>]

    model = load(path)                                  # DescriptorModel (joblib: classifiers + feature list + class order)
    parents, subclasses = model.predict_proba(occ64)    # ({"box": p, ...}, {"box/ibeam": p, ...})
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.subclass_schema import FLAT_SUBCLASSES, PARENTS, filter_records, labels_of  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "type_descriptors_v5_medium.joblib"
RESOLUTION = 64
MINKOWSKI_FEATURES = ("log_V", "S3_V2", "B3_V", "BS_V", "chi", "inertia_0", "inertia_1", "inertia_2")
EXTRA_FEATURES = ("b0", "b2_cavities", "b1_tunnels", "cavity_frac", "thick_q10", "thick_q50", "thick_q90", "thick_max",
                  "convex_fill", *(f"radial_{i}" for i in range(8)), "taper_slope", "taper_skew")
FEATURES = MINKOWSKI_FEATURES + EXTRA_FEATURES
CLASSIFIER_KW = dict(max_iter=300, learning_rate=0.08, random_state=0, class_weight="balanced")


# ------------------------------------------------------------------------------------------------ features

def cell_counts(occ: np.ndarray):
    """(n3, n2, n1, n0): cubes, and the distinct faces, edges and vertices of the union of closed unit cubes."""
    occ = np.asarray(occ, bool)
    p = np.pad(occ, 1)
    n3 = int(occ.sum())
    # a lattice cell of dimension k is present if ANY of the 2^(3-k) cubes around it is occupied
    faces = 0
    for ax in range(3):
        sl_a = [slice(None)] * 3
        sl_b = [slice(None)] * 3
        sl_a[ax], sl_b[ax] = slice(0, -1), slice(1, None)
        faces += int((p[tuple(sl_a)] | p[tuple(sl_b)]).sum())
    edges = 0
    for ax in range(3):                                       # edges along `ax`: 4 cubes around, in the other two axes
        o = [a for a in range(3) if a != ax]
        q = p
        for a in o:
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[a], s1[a] = slice(0, -1), slice(1, None)
            q = q[tuple(s0)] | q[tuple(s1)]
        edges += int(q.sum())
    q = p
    for a in range(3):
        s0 = [slice(None)] * 3
        s1 = [slice(None)] * 3
        s0[a], s1[a] = slice(0, -1), slice(1, None)
        q = q[tuple(s0)] | q[tuple(s1)]
    return n3, faces, edges, int(q.sum())


def minkowski_descriptors(occ: np.ndarray) -> np.ndarray:
    """MINKOWSKI_FEATURES of one grid: volume V = n3, surface S = -6 n3 + 2 n2, mean breadth B = 3 n3 - 2 n2 + n1,
    Euler chi = -n3 + n2 - n1 + n0 (open-cell formulas), the scale-free ratios and the inertia shape. Staircase note:
    S and B of a voxel grid overestimate the true values by an orientation-dependent factor (measured, not corrected)."""
    n3, n2, n1, n0 = cell_counts(occ)
    V = max(n3, 1)
    S = -6 * n3 + 2 * n2
    B = 3 * n3 - 2 * n2 + n1
    chi = -n3 + n2 - n1 + n0
    idx = np.argwhere(occ).astype(float)
    if len(idx) > 1:
        d = idx - idx.mean(0)
        w = np.sort(np.linalg.eigvalsh(d.T @ d / len(d)))
        w = w / max(w.sum(), 1e-12)
    else:
        w = np.zeros(3)
    return np.array([np.log(V), S ** 3 / V ** 2, B ** 3 / V, B * S / V, chi, *w], float)


def extra_descriptors(occ: np.ndarray, chi: float) -> np.ndarray:
    """EXTRA_FEATURES of one grid (`chi` = the Euler number from minkowski_descriptors, for the tunnel count)."""
    occ = np.asarray(occ, bool)
    V = max(int(occ.sum()), 1)
    _, b0 = ndimage.label(occ)
    # the border padded with BACKGROUND (True), so that the outside is one component with the pad; the baseline script
    # padded with False, which made the pad's corner label 0 and counted the real outside as a cavity (b2 one too high,
    # cavity_frac the outside volume over V: a size proxy) - fixed here, so numbers differ slightly from the 2026-09 baselines
    bg, nbg = ndimage.label(np.pad(~occ, 1, constant_values=True))
    outside = bg[0, 0, 0]
    cav_labels = [k for k in range(1, nbg + 1) if k != outside]
    b2 = len(cav_labels)
    cavity_vol = int(np.isin(bg, cav_labels).sum()) if cav_labels else 0
    b1 = b0 + b2 - chi
    scale = V ** (1.0 / 3.0)
    edt = ndimage.distance_transform_edt(occ)[occ]
    q10, q50, q90 = (np.quantile(edt, [0.1, 0.5, 0.9]) / scale) if len(edt) else (0.0, 0.0, 0.0)
    thick_max = float(edt.max() / scale) if len(edt) else 0.0
    idx = np.argwhere(occ).astype(float)
    boundary = np.argwhere(occ & ~ndimage.binary_erosion(occ)).astype(float)
    try:
        hull = ConvexHull(boundary + 0.5 * np.sign(boundary - boundary.mean(0))) if len(boundary) > 4 else None
        convex_fill = float(V / hull.volume) if hull is not None and hull.volume > 0 else 1.0
    except Exception:
        convex_fill = 1.0
    c = idx.mean(0) if len(idx) else np.zeros(3)
    r = np.linalg.norm(idx - c, axis=1)
    rms = float(np.sqrt((r ** 2).mean())) if len(r) else 1.0
    radial = np.histogram(r / max(rms, 1e-9), bins=8, range=(0.0, 2.4))[0] / max(len(r), 1)
    d = idx - c
    w, vecs = np.linalg.eigh(d.T @ d / max(len(d), 1))
    t = d @ vecs[:, -1]                                             # position along the longest axis
    counts = np.histogram(t, bins=8)[0].astype(float)
    x = np.linspace(-1, 1, 8)
    prof = np.sqrt(counts / max(counts.max(), 1))
    slope = abs(float(np.polyfit(x, prof, 1)[0]))
    skew = abs(float(((t - t.mean()) ** 3).mean() / max(t.std(), 1e-9) ** 3)) if len(t) > 2 else 0.0
    return np.array([b0, b2, b1, cavity_vol / V, q10, q50, q90, thick_max, min(convex_fill, 1.5), *radial, slope, skew],
                    float)


def descriptors(occ: np.ndarray) -> np.ndarray:
    """All FEATURES (27) of one occupancy grid (bool or {0, 1} / {-1, +1}: anything > 0 is material)."""
    occ = np.asarray(occ) > 0
    base = minkowski_descriptors(occ)
    return np.concatenate([base, extra_descriptors(occ, base[4])])


def features(bin_path: Path, resolution: int = RESOLUTION) -> np.ndarray:
    """FEATURES of a dataset sample's `.bin` voxel file."""
    return descriptors(np.fromfile(bin_path, dtype=np.uint8).reshape(resolution, resolution, resolution))


def dataset_features(roots, limit: Optional[int] = None):
    """(X, parent labels, flat subclass labels, records, skipped counter) of every labelled sample under the dataset
    roots (training.manifest + training.subclass_schema.filter_records). `limit`: the first n records only (tests)."""
    from training.manifest import load_manifest
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    recs, skipped = filter_records([r for root in roots for r in load_manifest(Path(root), require_voxel=True)])
    if limit is not None:
        recs = recs[:limit]
    X = np.stack([features(r.voxel_path) for r in recs])
    lab = np.array([labels_of(r) for r in recs])
    return X, lab[:, 0], lab[:, 1], recs, skipped


# ------------------------------------------------------------------------------------------------ the model

def source_fingerprint() -> str:
    """sha256 of this module's source: the feature code a saved model was fitted with (`load` warns on a mismatch)."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass
class DescriptorModel:
    parent_clf: object                          # sklearn HistGradientBoostingClassifier over parent indices
    sub_clf: object                             # ... over flat subclass indices
    features: Tuple[str, ...] = FEATURES
    parents: Tuple[str, ...] = PARENTS          # class order: parent index -> name
    subclasses: Tuple[str, ...] = FLAT_SUBCLASSES   # flat subclass index -> "parent/sub"
    meta: Dict = dc_field(default_factory=dict)     # train roots, n_train, fingerprint, metrics, fitted at

    def predict_proba(self, occupancy) -> Tuple[Dict[str, float], Dict[str, float]]:
        """({parent: p}, {"parent/sub": p}) for one occupancy grid (bool, {0, 1} or the nets' {-1, +1}); a class the
        training data lacked has p = 0. The subclass probabilities are the flat classifier's, not conditioned on a parent
        (`subclasses_of` masks them to one parent)."""
        x = descriptors(np.asarray(occupancy))[None]
        parents = {name: 0.0 for name in self.parents}
        for c, p in zip(self.parent_clf.classes_, self.parent_clf.predict_proba(x)[0]):
            parents[self.parents[int(c)]] = float(p)
        subs = {name: 0.0 for name in self.subclasses}
        for c, p in zip(self.sub_clf.classes_, self.sub_clf.predict_proba(x)[0]):
            subs[self.subclasses[int(c)]] = float(p)
        return parents, subs

    @staticmethod
    def subclasses_of(parent: str, flat_probs: Dict[str, float]) -> Dict[str, float]:
        """{sub: p} of `parent`'s subclasses, renormalised within the parent ("box/ibeam" -> "ibeam")."""
        own = {name.split("/", 1)[1]: p for name, p in flat_probs.items() if name.startswith(parent + "/")}
        total = sum(own.values())
        return {k: v / total for k, v in own.items()} if total > 0 else own


def fit(roots, limit: Optional[int] = None, verbose: bool = True) -> DescriptorModel:
    """A DescriptorModel fitted on every labelled sample under the dataset roots (a list pools several roots)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    t0 = time.time()
    X, parent, sub, recs, skipped = dataset_features(roots, limit=limit)
    if verbose:
        print(f"[type_descriptors] {len(X)} samples, {len(FEATURES)} features in {time.time() - t0:.0f} s"
              + (f"; skipped {dict(skipped)}" if skipped else ""))
    parent_clf = HistGradientBoostingClassifier(**CLASSIFIER_KW).fit(X, parent)
    sub_clf = HistGradientBoostingClassifier(**CLASSIFIER_KW).fit(X, sub)
    roots = [roots] if isinstance(roots, (str, Path)) else list(roots)
    meta = dict(train_roots=[str(r) for r in roots], n_train=int(len(X)), skipped=dict(skipped),
                fingerprint=source_fingerprint(), classifier=dict(CLASSIFIER_KW), fitted_at=time.strftime("%Y-%m-%d %H:%M"))
    return DescriptorModel(parent_clf, sub_clf, FEATURES, PARENTS, FLAT_SUBCLASSES, meta)


def evaluate(model: DescriptorModel, roots, limit: Optional[int] = None) -> Dict:
    """parent_acc, parent_top2, sub_acc_true_parent (the subclass masked to the true parent) on the dataset roots."""
    X, parent, sub, _, _ = dataset_features(roots, limit=limit)
    pp = np.zeros((len(X), len(model.parents)))
    for j, c in enumerate(model.parent_clf.classes_):
        pp[:, int(c)] = model.parent_clf.predict_proba(X)[:, j]
    sp = np.zeros((len(X), len(model.subclasses)))
    for j, c in enumerate(model.sub_clf.classes_):
        sp[:, int(c)] = model.sub_clf.predict_proba(X)[:, j]
    order = np.argsort(-pp, axis=1)
    top1 = order[:, 0]
    top2 = np.array([parent[i] in order[i, :2] for i in range(len(X))])
    parent_of = np.array([model.parents.index(name.split("/")[0]) for name in model.subclasses])
    masked = np.where(parent_of[None, :] == parent[:, None], sp, -1.0)
    sub_pred = masked.argmax(axis=1)
    return dict(n=int(len(X)), parent_acc=float((top1 == parent).mean()), parent_top2=float(top2.mean()),
                sub_acc_true_parent=float((sub_pred == sub).mean()),
                per_parent={model.parents[p]: dict(n=int((parent == p).sum()), acc=float((top1[parent == p] == p).mean()))
                            for p in sorted(set(parent.tolist()))})


def save(model: DescriptorModel, path=DEFAULT_MODEL) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(parent_clf=model.parent_clf, sub_clf=model.sub_clf, features=list(model.features),
                     parents=list(model.parents), subclasses=list(model.subclasses), meta=dict(model.meta)), path)
    return path


def load(path=DEFAULT_MODEL, warn: bool = True) -> DescriptorModel:
    """The saved model. Raises FileNotFoundError when the file is not there (the cascade then skips the entry); a
    feature list or fingerprint that does not match this module is reported on stderr (the model still loads)."""
    import joblib
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"descriptor model not found: {path} (fit it: python3 -m reco_engine.type_descriptors --fit ...)")
    d = joblib.load(path)
    model = DescriptorModel(d["parent_clf"], d["sub_clf"], tuple(d["features"]), tuple(d["parents"]), tuple(d["subclasses"]),
                            dict(d.get("meta") or {}))
    if warn:
        if model.features != FEATURES:
            print(f"[type_descriptors] {path}: fitted on {len(model.features)} features, this module computes {len(FEATURES)}",
                  file=sys.stderr)
        elif model.meta.get("fingerprint") not in (None, source_fingerprint()):
            print(f"[type_descriptors] {path}: fitted with an older version of the feature code (fingerprint differs)",
                  file=sys.stderr)
    return model


# ------------------------------------------------------------------------------------------------ CLI

def main(argv: Optional[Sequence[str]] = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", type=Path, nargs="+", required=True, help="training roots (several are pooled)")
    ap.add_argument("--out", type=Path, default=DEFAULT_MODEL, help=f"the joblib file to write (default {DEFAULT_MODEL})")
    ap.add_argument("--eval", type=Path, nargs="*", default=None, help="held-out roots to score after the fit")
    ap.add_argument("--limit", type=int, default=None, help="use the first n records only (a smoke test)")
    args = ap.parse_args(argv)
    model = fit(args.fit, limit=args.limit)
    if args.eval:
        model.meta["eval"] = dict(roots=[str(r) for r in args.eval], **evaluate(model, args.eval, limit=args.limit))
        print("[type_descriptors] eval", json.dumps(model.meta["eval"]))
    out = save(model, args.out)
    print(f"[type_descriptors] wrote {out} ({model.meta['n_train']} samples)")


if __name__ == "__main__":
    main()
