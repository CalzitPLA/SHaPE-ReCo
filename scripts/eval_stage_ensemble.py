#!/usr/bin/env python3
"""
Held-out evidence for choosing, per cascade stage (type, rotation, dimensions, position) and per topology, which
source answers first: a CNN, an ensemble of CNNs, or an analytic fit. The result is what models/registry.json
(reco_engine/stage_registry.py) is filled from, so the choice is traceable to numbers instead of to whichever
checkpoint directory a script happened to hard-code.

Why a new script rather than more runs of the old ones: the earlier eval scripts (eval_type_cnn_v3.py,
eval_type_fused.py, eval_rotation.py, eval_box_dims.py, eval_cylinder_fit.py, eval_alignment.py) each compared a
few checkpoints on a few test sets and kept only aggregates. Combining nets (a top-k union, a softmax mean) and
deriving a confidence gate needs the per-sample probabilities of every net on every set, which none of them
stored. So:

  type        every type net below runs once per test set (softmax cached as .npz, keyed by the checkpoint's size
              and mtime), then single nets and ensembles are scored per test set and per class, with a
              reliability table (accuracy vs max-prob bins) and a gate: the smallest max-prob above which top-1 is
              >= 99 % right, chosen on the even samples and checked on the odd ones.
  rotation    summarised from the existing docs/data/{alignment,rotation_eval_*}.json (no new inference: the
              face-normal frame fit and the residual networks were already scored on the same 600 boxes).
  dimensions  box: docs/data/box_dims_eval.json plus ONE missing cell, measured extents after the frame fit
              (cheap, ~50 ms/box); cylinder: docs/data/cylinder_fit_eval.json; sphere: the analytic fit, run here.
  position    run here (cheap): placement.py's centre of gravity / bounding-box centre and the sphere fit's centre
              against the generator's centre_vox; cylinder from cylinder_fit_eval.json.

Test sets: dataset_gen/local_dataset_inscale/{test,test_carve,test_ff,test_mixed,test_cyl,test_hollow}, 400
samples each, never used for training or checkpoint selection.

    python3 scripts/eval_stage_ensemble.py [--cache-dir DIR] [--fit-limit 100]

Writes docs/data/ensemble_selection.json and docs/data/ensemble_selection.md.

Extra test roots and hierarchical TypeNets (training/train_type_cnn.py --hierarchical):

    python3 scripts/eval_stage_ensemble.py --extra-test dataset_gen/local_dataset_v5/small/test \
        --hier-candidate checkpoints/type_hier_occ/type_cnn checkpoints/type_hier_vf/type_cnn [--only-extra]

scores every hierarchical candidate on every extra root (parent acc / top-2, subclass acc within the true parent and
through the predicted parent, the ibeam-vs-box and sector/band-vs-full confusions) and, unless --only-extra, the
legacy TYPE_NETS on the same roots by parent accuracy (their 17-way softmax restricted to the 5 parents). Written to
--extra-out (default docs/data/type_extra_eval.{json,md}); without these flags nothing changes.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from report_type_cnn import wilson  # noqa: E402
from training.manifest import load_manifest  # noqa: E402
from training.topology_schema import TOPOLOGY_TO_IDX  # noqa: E402

DATA = ROOT / "dataset_gen" / "local_dataset_inscale"
DOCS = ROOT / "docs" / "data"
SETS = ("test", "test_carve", "test_ff", "test_mixed", "test_cyl", "test_hollow")
CLASSES = ("box", "sphere", "cylinder")          # every label in the six test sets
CLS_IDX = [TOPOLOGY_TO_IDX[c] for c in CLASSES]

# id -> (checkpoint dir relative to the repo root, loader). "type_cnn" dirs hold best_model.pth + config.json.
TYPE_NETS = {
    "type_v3_B": ("checkpoints_type_v3/carve_plus_v2data/type_cnn", "type_cnn"),
    "type_inscale_v2": ("checkpoints_inscale_v2/type_cnn", "type_cnn"),
    "type_v4": ("checkpoints_type_v4/type_cnn", "type_cnn"),
    "type_hollow": ("checkpoints_hollow/type_cnn", "type_cnn"),
    "type_fused_net": ("checkpoints_fused/net_fused", "fused_net"),
}
TWO_CLASS = ("type_v3_B", "type_inscale_v2")                       # trained on box + sphere only
THREE_CLASS = ("type_v4", "type_hollow", "type_fused_net")          # box + sphere + cylinder

# Ensembles over the restricted (box, sphere, cylinder) softmax. "mean": average of the members' softmaxes (all share
# the 17-way TOPOLOGY_TO_IDX label space, so labels map by name for free). "max": argmax of the element-wise maximum
# (a max-prob union). "hier": P(cylinder) from the 3-class members, the box/sphere split from the 2-class members -
# the union that lets the 2-class nets keep deciding box vs sphere while only the 3-class nets can say "cylinder".
ENSEMBLES = {
    "mean_B_v2": ("mean", ("type_v3_B", "type_inscale_v2")),
    "mean_v4_hollow": ("mean", ("type_v4", "type_hollow")),
    "mean_v4_hollow_fused": ("mean", ("type_v4", "type_hollow", "type_fused_net")),
    "mean_all5": ("mean", tuple(TYPE_NETS)),
    "max_B_v4": ("max", ("type_v3_B", "type_v4")),
    "max_B_hollow": ("max", ("type_v3_B", "type_hollow")),
    "hier_B_v4": ("hier", ("type_v3_B", "type_v4")),
    "hier_B_hollow": ("hier", ("type_v3_B", "type_hollow")),
    "hier_B_v4hollow": ("hier", ("type_v3_B", "type_v4", "type_hollow")),
    "hier_Bv2_v4hollow": ("hier", ("type_v3_B", "type_inscale_v2", "type_v4", "type_hollow")),
    # "cylgate@t": cylinder only when the MEAN P(cylinder) of the 3-class members reaches t, otherwise box/sphere from
    # the 2-class members. t = 0.5 / 0.7 are fixed round values, not tuned per set (a sweep on these sets showed the
    # same trade-off at every t: see "Finding" in ensemble_selection.md).
    "cylgate0.5_Bv2_v4hollow": ("cylgate0.5", ("type_v3_B", "type_inscale_v2", "type_v4", "type_hollow")),
    "cylgate0.7_Bv2_v4hollow": ("cylgate0.7", ("type_v3_B", "type_inscale_v2", "type_v4", "type_hollow")),
}
GATE_TARGET = 0.99
BINS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0001)


# ---------------------------------------------------------------------------------------------------- type stage

def _checkpoint_key(directory: Path) -> str:
    st = (directory / "best_model.pth").stat()
    return hashlib.sha1(f"{directory}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:12]


def _load_type_net(directory: Path, loader: str, device):
    """(callable volumes01 -> softmax (N, 17), input kind)."""
    import torch
    from training.field_input import CHANNELS, encode
    if loader == "fused_net":
        from reco_engine.fused_predict import load_fused_model
        handle = load_fused_model(directory, "best_model.pth", device)
        model, kind, trunc, head = handle.model, handle.input_kind, handle.sdf_truncation, "type"
    else:
        from training.train_type_cnn import TypeCNN
        cfg_path = directory / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        kind, trunc, head = cfg.get("input", "occupancy"), float(cfg.get("sdf_truncation", 8.0)), None
        model = TypeCNN(in_channels=CHANNELS[kind])
        model.load_state_dict(torch.load(directory / "best_model.pth", map_location=device, weights_only=True)["model_state_dict"])
        model = model.to(device).eval()
    batch = 64 if kind == "occupancy" else 8

    @torch.no_grad()
    def run(volumes):
        out = []
        for i in range(0, len(volumes), batch):
            chunk = [v.astype(np.float32) * 2 - 1 for v in volumes[i:i + batch]]
            x = torch.from_numpy(np.stack([encode(v, kind, trunc) for v in chunk])).to(device)
            logits = model(x)
            logits = logits[head] if head else logits
            out.append(torch.softmax(logits.float(), 1).cpu().numpy())
        return np.concatenate(out)
    return run, kind


def type_probabilities(cache_dir: Path, device):
    """{set: dict(labels, kinds, probs={net: (N,17)})}, inference only for (net, set) cells not in the cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    loaded, out = {}, {}
    for set_name in SETS:
        records = load_manifest(DATA / set_name, require_voxel=True)
        labels = np.array([TOPOLOGY_TO_IDX[r.topology] for r in records])
        kinds = np.array([_body_kind(r) for r in records])
        volumes = None
        probs = {}
        for net_id, (rel, loader) in TYPE_NETS.items():
            path = cache_dir / f"type_{net_id}_{set_name}_{_checkpoint_key(ROOT / rel)}.npz"
            if path.is_file():
                probs[net_id] = np.load(path)["probs"]
                continue
            if volumes is None:
                volumes = [np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) for r in records]
            if net_id not in loaded:
                loaded[net_id] = _load_type_net(ROOT / rel, loader, device)
            t0 = time.time()
            probs[net_id] = loaded[net_id][0](volumes)
            np.savez_compressed(path, probs=probs[net_id], labels=labels)
            print(f"  inferred {net_id:16s} on {set_name:11s} ({len(records)} samples, {time.time() - t0:5.1f} s)")
        out[set_name] = dict(labels=labels, kinds=kinds, probs=probs)
    return out


def _body_kind(record) -> str:
    p = record.placement or {}
    if p.get("hollow"):
        return "hollow"
    if p.get("full_field"):
        return "full_field"
    if p.get("carve"):
        return "carved"
    return "cut" if p.get("cuts") else "plain"


def _restrict(p17: np.ndarray) -> np.ndarray:
    p = p17[:, CLS_IDX]
    return p / np.clip(p.sum(1, keepdims=True), 1e-12, None)


def combine(mode: str, members, probs: dict) -> np.ndarray:
    """(N, 3) restricted probabilities of an ensemble."""
    ps = [_restrict(probs[m]) for m in members]
    if mode == "mean":
        return np.mean(ps, axis=0)
    if mode == "max":
        p = np.max(ps, axis=0)
        return p / p.sum(1, keepdims=True)
    if mode == "hier":
        two = [_restrict(probs[m]) for m in members if m in TWO_CLASS]
        three = [_restrict(probs[m]) for m in members if m in THREE_CLASS]
        p_cyl = np.mean([p[:, 2] for p in three], axis=0)
        bs = np.mean([p[:, :2] / np.clip(p[:, :2].sum(1, keepdims=True), 1e-12, None) for p in two], axis=0)
        return np.column_stack([bs * (1 - p_cyl)[:, None], p_cyl])
    if mode.startswith("cylgate"):
        tau = float(mode[len("cylgate"):])
        two = [_restrict(probs[m]) for m in members if m in TWO_CLASS]
        p_cyl = np.mean([_restrict(probs[m])[:, 2] for m in members if m in THREE_CLASS], axis=0)
        bs = np.mean([p[:, :2] / np.clip(p[:, :2].sum(1, keepdims=True), 1e-12, None) for p in two], axis=0)
        # confidence stays a probability: the gated class keeps its own (averaged) probability mass
        cyl = p_cyl >= tau
        return np.where(cyl[:, None], np.column_stack([bs * (1 - p_cyl)[:, None], p_cyl]),
                        np.column_stack([bs, np.zeros(len(bs))]))
    raise ValueError(mode)


def score(p3: np.ndarray, labels17: np.ndarray, kinds: np.ndarray) -> dict:
    y = np.array([CLS_IDX.index(l) for l in labels17])
    pred = p3.argmax(1)
    ok = pred == y
    n, k = len(y), int(ok.sum())
    row = dict(n=n, correct=k, accuracy=k / n, ci95=wilson(k, n),
               nll=float(-np.log(np.clip(p3[np.arange(n), y], 1e-9, 1)).mean()),
               mean_max_prob=float(p3.max(1).mean()),
               top2_accuracy=float((np.argsort(-p3, 1)[:, :2] == y[:, None]).any(1).mean()))
    row["per_class"] = {c: dict(n=int((y == i).sum()), accuracy=float(ok[y == i].mean()))
                        for i, c in enumerate(CLASSES) if (y == i).any()}
    row["by_body"] = {b: dict(n=int((kinds == b).sum()), accuracy=float(ok[kinds == b].mean())) for b in sorted(set(kinds))}
    row["confusion"] = {f"{CLASSES[a]}->{CLASSES[b]}": c for (a, b), c in Counter(zip(y.tolist(), pred.tolist())).items() if a != b}
    return row


def calibration(p3: np.ndarray, correct: np.ndarray) -> dict:
    """Reliability table and the confidence gate (the smallest max-prob t with accuracy >= GATE_TARGET on every
    sample at or above t), chosen on the even samples and verified on the odd ones so it is not scored on the same
    samples it was picked on."""
    conf = p3.max(1)
    table = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (conf >= lo) & (conf < hi)
        table.append(dict(bin=[lo, min(hi, 1.0)], n=int(m.sum()), accuracy=float(correct[m].mean()) if m.any() else None))

    def pick(idx):
        order = np.argsort(-conf[idx])
        c, ok = conf[idx][order], correct[idx][order]
        running = np.cumsum(ok) / np.arange(1, len(ok) + 1)
        good = np.nonzero(running >= GATE_TARGET)[0]
        # largest prefix (lowest threshold) whose accuracy still meets the target; needs >= 50 samples to mean anything
        good = good[good >= 49]
        return float(c[good[-1]]) if len(good) else None

    even, odd = np.arange(0, len(conf), 2), np.arange(1, len(conf), 2)
    t = pick(even)
    gate = dict(target=GATE_TARGET, threshold=t)
    if t is not None:
        m = conf[odd] >= t
        gate.update(check_n=int(m.sum()), check_coverage=float(m.mean()), check_accuracy=float(correct[odd][m].mean()) if m.any() else None)
        m_all = conf >= t
        gate.update(all_coverage=float(m_all.mean()), all_accuracy=float(correct[m_all].mean()))
    return dict(reliability=table, gate=gate)


def type_stage(cache_dir: Path, device) -> dict:
    data = type_probabilities(cache_dir, device)
    candidates = {net: ("single", (net,)) for net in TYPE_NETS}
    candidates.update(ENSEMBLES)
    per_set, pooled_p, pooled_ok = {}, {c: [] for c in candidates}, {c: [] for c in candidates}
    pooled_y, pooled_kind = [], []
    for set_name, d in data.items():
        per_set[set_name] = {}
        for cid, (mode, members) in candidates.items():
            p3 = _restrict(d["probs"][members[0]]) if mode == "single" else combine(mode, members, d["probs"])
            per_set[set_name][cid] = score(p3, d["labels"], d["kinds"])
            pooled_p[cid].append(p3)
        pooled_y.append(d["labels"])
        pooled_kind.append(d["kinds"])
    y17, kinds = np.concatenate(pooled_y), np.concatenate(pooled_kind)
    pooled, calib = {}, {}
    for cid in candidates:
        p3 = np.concatenate(pooled_p[cid])
        pooled[cid] = score(p3, y17, kinds)
        y = np.array([CLS_IDX.index(l) for l in y17])
        calib[cid] = calibration(p3, p3.argmax(1) == y)
    # G0: the best candidate by pooled accuracy must not lose to the best single net on any test set
    singles = list(TYPE_NETS)
    best = max(candidates, key=lambda c: (pooled[c]["accuracy"], -len(candidates[c][1])))
    # two readings of G0: against the per-set best single net (a specialist per set), and against the single net that
    # is best pooled over all six sets (what a one-net pipeline would ship)
    pooled_single = max(singles, key=lambda c: pooled[c]["accuracy"])
    g0 = {}
    for set_name in SETS:
        best_single = max(singles, key=lambda c: per_set[set_name][c]["accuracy"])
        a, b = per_set[set_name][best]["accuracy"], per_set[set_name][best_single]["accuracy"]
        c = per_set[set_name][pooled_single]["accuracy"]
        g0[set_name] = dict(chosen=best, chosen_acc=a, best_single=best_single, best_single_acc=b, ok=a >= b,
                            pooled_best_single=pooled_single, pooled_best_single_acc=c, ok_vs_pooled_single=a >= c)
    return dict(candidates={c: dict(mode=m, members=list(ms)) for c, (m, ms) in candidates.items()},
                per_set=per_set, pooled=pooled, calibration=calib, chosen=best, gate_g0=g0)


# ------------------------------------------------------------------------------- cheap dims / position cells

def geometry_cells(cache_dir: Path, fit_limit: int) -> dict:
    """Box dims after the frame fit, sphere fit radius/centre, and placement centres - analytic, CPU only."""
    path = cache_dir / f"geometry_cells_{fit_limit}.json"
    if path.is_file():
        return json.loads(path.read_text())
    from eval_box_dims import measured_half_extents
    from reco_engine.placement import estimate_placement
    from reco_engine.sphere_fit import SphereFitter
    from training import rotation_symmetry as rs
    from training.train_box_dims import half_extents_vox

    out = {}
    recs = [r for s in ("test_mixed", "test_carve") for r in load_manifest(DATA / s, require_voxel=True)]
    boxes = [r for r in recs if r.topology == "box" and not (r.placement or {}).get("full_field")]
    spheres = [r for r in recs if r.topology == "sphere"]
    t0 = time.time()
    abs_err, rel_err, vol_err = [], [], []
    for r in boxes:
        v = np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64)
        h = measured_half_extents(rs.align_volume(v, "frame")[0])
        t = half_extents_vox(r)
        abs_err.append(np.abs(h - t).max())
        rel_err.append(np.exp(np.abs(np.log(h) - np.log(t))).max() - 1)
        vol_err.append(abs(np.prod(h) / np.prod(t) - 1))
    out["box_dims_frame_measured"] = dict(n=len(boxes), test_set="test_mixed+test_carve (boxes, no full-field)",
                                          abs_median_vox=float(np.median(abs_err)), rel_median=float(np.median(rel_err)),
                                          volume_median=float(np.median(vol_err)), seconds=time.time() - t0)
    # centres: placement cog / bbox for every box and sphere; sphere fit on a subsample (a refined fit is ~1 s)
    for kind, rs_ in (("box", boxes), ("sphere", spheres)):
        for anchor in ("cog", "bbox"):
            e = []
            for r in rs_:
                v = np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64)
                pl = estimate_placement(v, 64, 40.0, center=anchor)
                e.append(np.linalg.norm(np.asarray(pl.center_of_gravity_vox) - np.asarray(r.placement["center_vox"])))
            out[f"{kind}_position_placement_{anchor}"] = dict(n=len(e), median_vox=float(np.median(e)), mean_vox=float(np.mean(e)),
                                                               within1_vox=float((np.asarray(e) < 1).mean()))
    fitter = SphereFitter(grid_mm=40.0)
    step = max(1, len(spheres) // fit_limit) if fit_limit else 1
    ce, re = [], []
    t0 = time.time()
    for r in spheres[::step]:
        v = np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64)
        est = fitter.estimate(v)
        if est is None:
            continue
        ce.append(np.linalg.norm(est.center_vox - np.asarray(r.placement["center_vox"])))
        re.append(abs(est.radius_vox / (float(np.mean(r.placement["extent_vox"])) / 2) - 1))
    out["sphere_fit"] = dict(n=len(ce), test_set="test_mixed+test_carve spheres", center_median_vox=float(np.median(ce)),
                             center_within1_vox=float((np.asarray(ce) < 1).mean()), radius_rel_median=float(np.median(re)),
                             seconds=time.time() - t0)
    path.write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------------------- existing JSON summaries

def _load(name):
    p = DOCS / name
    return json.loads(p.read_text()) if p.is_file() else None


def entry(id_, mode, metric, value, test_set, **extra):
    return dict(id=id_, mode=mode, metric=metric, value=value, test_set=test_set, **extra)


def rotation_rankings() -> dict:
    """Box: within-10-degree rate on the same 600 boxes (test, test_carve, test_mixed) where available."""
    box = []
    al = _load("alignment_eval.json")
    s600 = "test+test_carve+test_mixed (600 boxes)"
    if al:
        for m, mode, id_ in (("frame", "analytic", "box_rot_frame"), ("pca", "prior", "box_rot_pca")):
            a = al["methods"][m]["summary"]["all"]
            box.append(entry(id_, mode, "within10", a["within10"], s600, median_deg=a["median"], mean_deg=a["mean"]))
    ie = _load("rotation_eval_I_frame.json")
    if ie:
        for m, id_ in (("I", "box_rot_I_frame"), ("H", "box_rot_H_manifold")):
            a = ie["methods"][m]["all"]
            box.append(entry(id_, "cnn", "within10", a["within10"], s600, median_deg=a["median"], mean_deg=a["mean"]))
    fu = _load("rotation_eval_fused.json")
    if fu and "fused_net" in fu["methods"]:
        a = fu["methods"]["fused_net"]["all"]
        box.append(entry("box_rot_fused_net", "cnn", "within10", a["within10"], "test+test_ff (346 boxes)",
                         median_deg=a["median"], mean_deg=a["mean"]))
    box.sort(key=lambda e: -e["value"])
    cyl = []
    cf = _load("cylinder_fit_eval.json")
    if cf:
        cyl.append(entry("cyl_fit_analytic", "analytic", "axis_within5", cf["all"]["axis_within5"], "test_cyl (134)",
                         axis_median_deg=cf["all"]["axis_deg"]["median"]))
    cyl.append(entry("cylinder_rot_field_pertopo", "cnn", "axis_mean_deg_val", 42.0, "training val split (not held-out)"))
    return dict(box=box, sphere=[entry("sphere_rot_none", "prior", "n/a", None, "rotation-invariant")], cylinder=cyl,
                cone=[entry("field_fit", "analytic", "not scored on a voxel test set", None, "41 example parts (peel)"),
                      entry("cone_rot_field_pertopo", "cnn", "axis_mean_deg_val", 50.0, "training val split")],
                torus=[entry("field_fit", "analytic", "not scored on a voxel test set", None, "41 example parts (peel)"),
                       entry("torus_rot_field_pertopo", "cnn", "axis_mean_deg_val", 47.0, "training val split")])


def dims_rankings(geo: dict) -> dict:
    box = []
    bd = _load("box_dims_eval.json")
    s = "test_mixed+test_carve (400 boxes)"
    g = geo["box_dims_frame_measured"]
    box.append(entry("box_dims_measured", "analytic", "rel_median", g["rel_median"], s, align="frame",
                     volume_median=g["volume_median"], abs_median_vox=g["abs_median_vox"]))
    if bd:
        for m, id_, mode, note in (("measured+rot", "box_dims_measured_H", "analytic", "aligned by H_manifold"),
                                   ("dims_cnn", "box_dims_cnn", "cnn", "checkpoints_box_dims/dims_cnn"),
                                   ("measured", "box_dims_measured_pca", "analytic", "PCA-aligned")):
            a = bd["methods"][m]
            box.append(entry(id_, mode, "rel_median", a["rel"]["median"], s, volume_median=a["volume"]["median"],
                             abs_median_vox=a["abs"]["median"], note=note))
    box.sort(key=lambda e: e["value"])
    cf = _load("cylinder_fit_eval.json")
    cyl = [entry("cyl_fit_analytic", "analytic", "radius_rel_median", cf["all"]["radius_rel"]["median"], "test_cyl (134)",
                 half_length_rel_median=cf["all"]["half_rel"]["median"], iou_median=cf["all"]["iou"]["median"])] if cf else []
    sf = geo["sphere_fit"]
    sph = [entry("sphere_fit_analytic", "analytic", "radius_rel_median", sf["radius_rel_median"], sf["test_set"] + f" ({sf['n']})")]
    fld = [entry("field_fit", "analytic", "not scored on a voxel test set", None, "41 example parts (peel)")]
    return dict(box=box, sphere=sph, cylinder=cyl, cone=fld, torus=fld)


def position_rankings(geo: dict) -> dict:
    s = "test_mixed+test_carve"
    out = {}
    for kind in ("box", "sphere"):
        lst = [entry(f"position_placement_{a}", "analytic", "center_median_vox", geo[f"{kind}_position_placement_{a}"]["median_vox"],
                     s + f" ({geo[f'{kind}_position_placement_{a}']['n']})",
                     within1_vox=geo[f"{kind}_position_placement_{a}"]["within1_vox"]) for a in ("bbox", "cog")]
        if kind == "sphere":
            sf = geo["sphere_fit"]
            lst.append(entry("sphere_fit_analytic", "analytic", "center_median_vox", sf["center_median_vox"], sf["test_set"] + f" ({sf['n']})",
                             within1_vox=sf["center_within1_vox"]))
        out[kind] = sorted(lst, key=lambda e: e["value"])
    cf = _load("cylinder_fit_eval.json")
    if cf:
        out["cylinder"] = [entry("cyl_fit_analytic", "analytic", "center_median_vox", cf["all"]["center_err"]["median"], "test_cyl (134)")]
    out["cone"] = out["torus"] = [entry("field_fit", "analytic", "not scored on a voxel test set", None, "41 example parts (peel)")]
    return out


def type_rankings(ts: dict) -> dict:
    """Per class: pooled accuracy over every test set containing that class."""
    out = {}
    for c in CLASSES:
        lst = [entry(cid, "cnn" if ts["candidates"][cid]["mode"] == "single" else f"ensemble:{ts['candidates'][cid]['mode']}",
                     "accuracy", r["per_class"][c]["accuracy"], f"pooled six test sets (n={r['per_class'][c]['n']})",
                     members=ts["candidates"][cid]["members"])
               for cid, r in ts["pooled"].items() if c in r["per_class"]]
        out[c] = sorted(lst, key=lambda e: (-e["value"], len(e["members"])))
    out["all"] = sorted([entry(cid, "cnn" if ts["candidates"][cid]["mode"] == "single" else f"ensemble:{ts['candidates'][cid]['mode']}",
                               "accuracy", r["accuracy"], "pooled six test sets (n=2400)", members=ts["candidates"][cid]["members"],
                               gate=ts["calibration"][cid]["gate"]["threshold"])
                         for cid, r in ts["pooled"].items()], key=lambda e: (-e["value"], len(e["members"])))
    return out


# ------------------------------------------------------------------------------------------------- report

def write_markdown(result: dict, path: Path):
    ts = result["type"]
    lines = ["# Ensemble selection (held-out)", "",
             "Generated by `scripts/eval_stage_ensemble.py`; numbers in `ensemble_selection.json`.", "",
             "## Type: accuracy per test set (restricted to box/sphere/cylinder)", ""]
    cids = list(ts["candidates"])
    lines.append("| candidate | " + " | ".join(SETS) + " | pooled | pooled top-2 | gate (p>=) | coverage |")
    lines.append("|---|" + "---|" * (len(SETS) + 4))
    for cid in cids:
        g = ts["calibration"][cid]["gate"]
        row = [f"{ts['per_set'][s][cid]['accuracy']:.4f}" for s in SETS]
        gate = f"{g['threshold']:.3f}" if g["threshold"] is not None else "-"
        cov = f"{g.get('all_coverage', 0):.2f}" if g["threshold"] is not None else "-"
        mark = " **(chosen)**" if cid == ts["chosen"] else ""
        lines.append(f"| {cid}{mark} | " + " | ".join(row) + f" | {ts['pooled'][cid]['accuracy']:.4f} | "
                     f"{ts['pooled'][cid]['top2_accuracy']:.4f} | {gate} | {cov} |")
    lines += ["", "Gate: the smallest max-prob above which top-1 is >= 99 % right, picked on the even samples of the pooled "
              "2400 and checked on the odd ones (coverage = share of all samples above the gate). '-' = never reaches 99 % "
              "on the pooled sets (for the 2-class nets: the 400 cylinders are always wrong).", "",
              "G0, chosen vs the best single net per test set: " + ", ".join(
                  f"{s} {'ok' if v['ok'] else 'LOSES'} ({v['chosen_acc']:.4f} vs {v['best_single']} {v['best_single_acc']:.4f})"
                  for s, v in ts["gate_g0"].items()), "",
              "G0, chosen vs the best pooled single net: " + ", ".join(
                  f"{s} {'ok' if v['ok_vs_pooled_single'] else 'LOSES'} ({v['chosen_acc']:.4f} vs {v['pooled_best_single']} "
                  f"{v['pooled_best_single_acc']:.4f})" for s, v in ts["gate_g0"].items()), ""]
    h = ts["per_set"]
    lines += ["**Finding.** The 3-class nets (v4, hollow, FusedNet) were trained on carve + cyl (+ hollow) data only and "
              "never saw planar-cut boxes/spheres: on `test` they call "
              f"{h['test']['type_hollow']['confusion'].get('box->cylinder', 0)} boxes and "
              f"{h['test']['type_hollow']['confusion'].get('sphere->cylinder', 0)} spheres cylinders (hollow; cut bodies "
              f"{h['test']['type_hollow']['by_body']['cut']['accuracy']:.3f}). The 2-class nets are perfect there but "
              "cannot say cylinder. Every combination therefore trades cylinder recall against box/sphere precision "
              "(a threshold sweep on P(cylinder) moves along the same front); no ensemble beats every specialist on every "
              "set. The fix is data (3-class training with planar cuts), not ensembling; until then the cascade should "
              "take the top-2 and let the analytic fit grade decide.", ""]
    lines += ["## Best source per stage and topology", "", "| stage | kind | best | mode | metric | value | test set | runner-up |",
              "|---|---|---|---|---|---|---|---|"]
    for stage in ("type", "rotation", "dimensions", "position"):
        for kind, lst in result["stages"][stage].items():
            if not lst:
                continue
            b = lst[0]
            val = "-" if b["value"] is None else (f"{b['value']:.4g}")
            ru = lst[1]["id"] + (f" ({lst[1]['value']:.4g})" if len(lst) > 1 and lst[1]["value"] is not None else "") if len(lst) > 1 else "-"
            lines.append(f"| {stage} | {kind} | {b['id']} | {b['mode']} | {b['metric']} | {val} | {b['test_set']} | {ru} |")
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------- extra test roots (hierarchical)

def _load_hier_candidate(directory: Path, device):
    """(model, dataset kwargs) of a train_type_cnn.py checkpoint dir (config.json + best_model.pth)."""
    from training.field_input import CHANNELS
    from training.train_type_cnn import load_checkpoint_model
    cfg_path = directory / "config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    kind = cfg.get("input", "occupancy")
    model = load_checkpoint_model(directory / "best_model.pth", CHANNELS[kind], device)
    return model, dict(resolution=int(cfg.get("resolution", 64)), input_kind=kind,
                       sdf_truncation=float(cfg.get("sdf_truncation", 8.0)))


def _parent_scores(p17: np.ndarray, records) -> dict:
    """Parent accuracy / top-2 of a 17-way softmax restricted to the 5 hierarchical parents."""
    from training.subclass_schema import PARENTS, PARENT_TO_IDX
    cols = [TOPOLOGY_TO_IDX[p] for p in PARENTS]
    keep = [i for i, r in enumerate(records) if r.topology in PARENT_TO_IDX]
    if not keep:
        return dict(n=0)
    p = p17[keep][:, cols]
    y = np.array([PARENT_TO_IDX[records[i].topology] for i in keep])
    return dict(n=len(keep), parent_acc=float((p.argmax(1) == y).mean()),
                parent_top2=float((np.argsort(-p, 1)[:, :2] == y[:, None]).any(1).mean()))


def extra_test_stage(roots, hier_dirs, device, legacy: bool) -> dict:
    from training.subclass_schema import subclass_counts
    from training.train_type_cnn import evaluate_records
    out = dict(roots={}, candidates={})
    hier = {}
    for d in hier_dirs:
        cid = f"hier:{d.parent.name}/{d.name}" if d.name == "type_cnn" else f"hier:{d.name}"
        hier[cid] = _load_hier_candidate(d, device)
        out["candidates"][cid] = dict(path=str(d), input=hier[cid][1]["input_kind"],
                                      hierarchical=bool(getattr(hier[cid][0], "hierarchical", False)))
    legacy_nets = {}
    for root in roots:
        records = load_manifest(root, require_voxel=True)
        row = dict(n=len(records), subclass_counts=subclass_counts(records), metrics={})
        for cid, (model, kwargs) in hier.items():
            t0 = time.time()
            row["metrics"][cid] = evaluate_records(model, records, device, 8, **kwargs)
            print(f"  {cid:28s} on {root} ({len(records)} samples, {time.time() - t0:5.1f} s)")
        if legacy:
            volumes = [np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) for r in records]
            for net_id, (rel, loader) in TYPE_NETS.items():
                if not (ROOT / rel / "best_model.pth").is_file():
                    continue
                if net_id not in legacy_nets:
                    legacy_nets[net_id] = _load_type_net(ROOT / rel, loader, device)
                row["metrics"][net_id] = _parent_scores(legacy_nets[net_id][0](volumes), records)
        out["roots"][str(root)] = row
    return out


def write_extra_markdown(extra: dict, path: Path):
    lines = ["# Type nets on extra test roots", "", "Generated by `scripts/eval_stage_ensemble.py --extra-test`.", "",
             "| root | candidate | n | parent acc | parent top-2 | subclass acc (true parent) | subclass acc (pred. parent) |",
             "|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.4f}"                    # noqa: E731
    for root, row in extra["roots"].items():
        for cid, m in row["metrics"].items():
            lines.append(f"| {root} | {cid} | {m.get('n', 0)} | {fmt(m.get('parent_acc'))} | {fmt(m.get('parent_top2'))} | "
                         f"{fmt(m.get('sub_acc'))} | {fmt(m.get('sub_acc_pred_parent'))} |")
    for root, row in extra["roots"].items():
        for cid, m in row["metrics"].items():
            if "confusion_ibeam_vs_box" not in m:
                continue
            lines += ["", f"**{cid} on {root}**: ibeam vs box/full {json.dumps(m['confusion_ibeam_vs_box'])}; "
                          f"sector/band vs full {json.dumps(m['confusion_segment_vs_full'])}"]
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(os.environ.get("SHAPE_RECO_CACHE", Path.home() / ".cache" / "shape_reco")) / "stage_ensemble")
    ap.add_argument("--fit-limit", type=int, default=100, help="spheres to run the analytic sphere fit on (0 = all)")
    ap.add_argument("--out", type=Path, default=DOCS / "ensemble_selection.json")
    ap.add_argument("--extra-test", type=Path, nargs="+", default=[], metavar="ROOT",
                    help="more test roots (e.g. dataset_gen/local_dataset_v5/small/test), scored separately (module docstring)")
    ap.add_argument("--hier-candidate", type=Path, nargs="+", default=[], metavar="DIR",
                    help="hierarchical TypeNet checkpoint dirs (config.json + best_model.pth) to score on --extra-test")
    ap.add_argument("--only-extra", action="store_true", help="skip the legacy stages and the legacy nets; only the extra roots")
    ap.add_argument("--extra-out", type=Path, default=DOCS / "type_extra_eval.json")
    args = ap.parse_args()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.time()
    if args.hier_candidate and not args.extra_test:
        ap.error("--hier-candidate needs --extra-test ROOT ...")
    if args.extra_test:
        extra = extra_test_stage(args.extra_test, args.hier_candidate, device, legacy=not args.only_extra)
        extra["generated"] = time.strftime("%Y-%m-%d %H:%M")
        args.extra_out.write_text(json.dumps(extra, indent=1))
        write_extra_markdown(extra, args.extra_out.with_suffix(".md"))
        for root, row in extra["roots"].items():
            for cid, m in row["metrics"].items():
                print(f"{root}  {cid:28s} parent {m.get('parent_acc')}  subclass {m.get('sub_acc')}  "
                      f"subclass|pred-parent {m.get('sub_acc_pred_parent')}")
        print(f"wrote {args.extra_out} and {args.extra_out.with_suffix('.md')} ({time.time() - t0:.0f} s)")
        if args.only_extra:
            return
    elif args.only_extra:
        ap.error("--only-extra needs --extra-test ROOT ...")

    ts = type_stage(args.cache_dir, device)
    print(f"type stage done ({time.time() - t0:.0f} s)")
    geo = geometry_cells(args.cache_dir, args.fit_limit)
    print(f"geometry cells done ({time.time() - t0:.0f} s)")
    result = dict(
        generated=time.strftime("%Y-%m-%d %H:%M"), test_sets=list(SETS), type_nets={k: v[0] for k, v in TYPE_NETS.items()},
        stages=dict(type=type_rankings(ts), rotation=rotation_rankings(), dimensions=dims_rankings(geo), position=position_rankings(geo)),
        type=ts, geometry_cells=geo)
    args.out.write_text(json.dumps(result, indent=1))
    write_markdown(result, args.out.with_suffix(".md"))
    for cid, r in sorted(ts["pooled"].items(), key=lambda kv: -kv[1]["accuracy"]):
        g = ts["calibration"][cid]["gate"]
        print(f"{cid:22s} pooled {r['accuracy']:.4f}  " + "  ".join(f"{s} {ts['per_set'][s][cid]['accuracy']:.3f}" for s in SETS)
              + f"  gate {g['threshold']}")
    print("chosen:", ts["chosen"], "G0:", {s: v["ok"] for s, v in ts["gate_g0"].items()})
    print(f"wrote {args.out} and {args.out.with_suffix('.md')} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
