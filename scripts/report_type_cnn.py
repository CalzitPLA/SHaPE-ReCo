#!/usr/bin/env python3
"""
Held-out evaluation of the Type CNNs for the training report (docs/thesis_inscale_training.tex).

Evaluates, on every data domain and split, the Type CNN trained on in-scale data ("new") and the earlier
one trained on stretched whole shapes ("old"), plus a fill-only baseline, and writes ONE JSON that the
report's figures, tables and numbers are generated from (scripts/make_report_artifacts.py) - nothing in
the report is typed in by hand.

Domains
  in-scale   dataset_gen/local_dataset_inscale/{type,test}  random size, position, cuts, noise (new data)
  stretched  dataset_gen/local_dataset_box_sphere/type       whole shapes stretched onto the grid (old data)
Splits: "train"/"val" reproduce the trainer's group split (seed 42, val fraction 0.2); "test" is a separate
dataset generated with fresh seeds, never used for any decision (early stopping and checkpoint selection
used "val").

Fill-only baseline: "sphere iff the grid fill fraction >= t", t chosen on the TRAIN split of the domain
(maximising its accuracy) and scored on val/test - the fair version of the shortcut the report is about.

    python3 scripts/report_type_cnn.py --out docs/data/type_cnn_eval.json
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.manifest import _resolve, load_manifest  # noqa: E402
from training.train_common import load_split  # noqa: E402
from training.train_type_cnn import TypeCNN  # noqa: E402
from training.topology_schema import IDX_TO_TOPOLOGY, TOPOLOGY_TO_IDX  # noqa: E402
from training.sdf_input import encode
from training.voxel_dataset import load_voxel  # noqa: E402

CLASSES = ("box", "sphere")


def wilson(k: int, n: int, z: float = 1.959964):
    """95 % Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(centre - half), float(centre + half))


def placements_by_voxel_path(root: Path) -> dict:
    """{resolved voxel path: placement record} from every metadata file under `root`."""
    out = {}
    root = Path(root).resolve()                                    # _resolve compares against absolute ancestors
    for meta in root.rglob("dataset_metadata.jsonl"):
        for line in meta.read_text().splitlines():
            entry = json.loads(line)
            if entry.get("record_type") == "dataset_header" or not entry.get("voxel_path"):
                continue
            out[str(_resolve(entry["voxel_path"], meta, root))] = entry.get("placement")
    return out


def load_model(path: Path, device) -> TypeCNN:
    """The network with `input_kind` / `sdf_truncation` set from the checkpoint directory's config.json (occupancy if none)."""
    model = TypeCNN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model_state_dict"])
    config = Path(path).parent / "config.json"
    cfg = json.loads(config.read_text()) if config.is_file() else {}
    model.input_kind = cfg.get("input", "occupancy")
    model.sdf_truncation = float(cfg.get("sdf_truncation", 8.0))
    return model.to(device).eval()


@torch.no_grad()
def predict(model, records, device, resolution=64, batch=64):
    """(probabilities (N, 17), fill fractions (N,)) - fill is measured on the grid the network sees."""
    probs, fills = [], []
    for i in range(0, len(records), batch):
        volumes = [load_voxel(r.voxel_path, resolution) for r in records[i:i + batch]]
        kind, truncation = getattr(model, "input_kind", "occupancy"), getattr(model, "sdf_truncation", 8.0)
        inputs = [encode(v, kind, truncation) for v in volumes]                   # occupancy: unchanged; sdf: training/sdf_input.py
        x = torch.from_numpy(np.stack(inputs)).unsqueeze(1).to(device)
        probs.append(torch.softmax(model(x), 1).cpu().numpy())
        fills += [float((v > 0).mean()) for v in volumes]
    return np.concatenate(probs), np.asarray(fills)


def summarize(records, probs, fills, placements=None) -> dict:
    labels = np.array([TOPOLOGY_TO_IDX[r.topology] for r in records])
    pred = probs.argmax(1)
    correct = pred == labels
    n, k = len(records), int(correct.sum())
    names = [r.topology for r in records]
    out = dict(n=n, correct=k, accuracy=k / n, ci95=wilson(k, n),
               mean_p_true=float(probs[np.arange(n), labels].mean()),
               nll=float(-np.log(np.clip(probs[np.arange(n), labels], 1e-12, 1)).mean()),
               n_per_class={c: names.count(c) for c in CLASSES})
    # confusion over {box, sphere, other}
    conf = {c: Counter() for c in CLASSES}
    for r, p in zip(records, pred):
        conf[r.topology][IDX_TO_TOPOLOGY[int(p)] if IDX_TO_TOPOLOGY[int(p)] in CLASSES else "other"] += 1
    out["confusion"] = {c: dict(conf[c]) for c in CLASSES}
    out["per_class_accuracy"] = {c: (lambda kk, nn: dict(accuracy=kk / nn, ci95=wilson(kk, nn), n=nn))(
        int(sum(correct[np.array(names) == c])), names.count(c)) for c in CLASSES}
    if placements is not None:
        def group(mask, key):
            kk, nn = int(correct[mask].sum()), int(mask.sum())
            return dict(n=nn, accuracy=kk / nn if nn else None, ci95=wilson(kk, nn))
        info = [placements.get(str(r.voxel_path)) or {} for r in records]
        ncuts = np.array([len(i.get("cuts", [])) for i in info])
        size = np.array([i.get("size_fraction", np.nan) for i in info])
        out["by_cuts"] = {str(c): group(ncuts == c, c) for c in (0, 1, 2)}
        edges = np.quantile(size[~np.isnan(size)], [0, 1 / 3, 2 / 3, 1])
        out["by_size_fraction"] = {f"{edges[j]:.2f}-{edges[j + 1]:.2f}": group((size >= edges[j]) & (size <= edges[j + 1]), j)
                                   for j in range(3)}
    fill_edges = [0, 0.005, 0.02, 0.05, 0.1, 1.01]
    out["by_fill"] = {}
    for lo, hi in zip(fill_edges[:-1], fill_edges[1:]):
        m = (fills >= lo) & (fills < hi)
        out["by_fill"][f"{lo:g}-{min(hi, 1):g}"] = dict(n=int(m.sum()), accuracy=float(correct[m].mean()) if m.any() else None,
                                                        ci95=wilson(int(correct[m].sum()), int(m.sum())))
    return out


def fit_fill_rule(records, fills):
    """Threshold t maximising the train accuracy of 'sphere iff fill >= t'."""
    y = np.array([r.topology == "sphere" for r in records])
    best = max(((float(np.mean((fills >= t) == y)), float(t)) for t in np.unique(fills)), key=lambda x: (x[0], -x[1]))
    return dict(threshold=best[1], train_accuracy=best[0])


def score_fill_rule(t, records, fills):
    y = np.array([r.topology == "sphere" for r in records])
    correct = (fills >= t) == y
    k, n = int(correct.sum()), len(records)
    return dict(n=n, correct=k, accuracy=k / n, ci95=wilson(k, n)), correct


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Exact two-sided McNemar test on paired per-sample correctness."""
    from scipy.stats import binomtest
    b = int(np.sum(correct_a & ~correct_b))     # a right, b wrong
    c = int(np.sum(~correct_a & correct_b))
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    return dict(a_only=b, b_only=c, p=p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new-checkpoint", type=Path, default=ROOT / "checkpoints_inscale/type_cnn/best_model.pth")
    ap.add_argument("--old-checkpoint", type=Path, default=ROOT / "checkpoints/type_cnn/best_model.pth")
    ap.add_argument("--inscale-root", type=Path, default=ROOT / "dataset_gen/local_dataset_inscale/type")
    ap.add_argument("--test-root", type=Path, default=ROOT / "dataset_gen/local_dataset_inscale/test")
    ap.add_argument("--stretched-root", type=Path, default=ROOT / "dataset_gen/local_dataset_box_sphere/type")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = {"new": load_model(args.new_checkpoint, device), "old": load_model(args.old_checkpoint, device)}
    new_epoch = torch.load(args.new_checkpoint, map_location="cpu", weights_only=False)["epoch"]
    old_epoch = torch.load(args.old_checkpoint, map_location="cpu", weights_only=False)["epoch"]

    splits = {}
    train, val = load_split(args.inscale_root, args.val_fraction, args.seed)
    splits["inscale/train"], splits["inscale/val"] = train, val
    splits["inscale/test"] = load_manifest(args.test_root, require_voxel=True)
    s_train, s_val = load_split(args.stretched_root, args.val_fraction, args.seed)
    splits["stretched/train"], splits["stretched/val"] = s_train, s_val
    placements = {**placements_by_voxel_path(args.inscale_root), **placements_by_voxel_path(args.test_root)}

    cache, results = {}, {}
    for split, records in splits.items():
        for name, model in nets.items():
            probs, fills = predict(model, records, device)
            cache[(split, name)] = (probs, fills)
            results[f"{name}@{split}"] = summarize(records, probs, fills, placements if split.startswith("inscale") else None)

    # fill-only baseline, fitted on each domain's train split
    fill_rule = {}
    for domain in ("inscale", "stretched"):
        recs = splits[f"{domain}/train"]
        fills = cache[(f"{domain}/train", "new")][1]
        rule = fit_fill_rule(recs, fills)
        fill_rule[domain] = rule
        for split in ("val", "test"):
            if f"{domain}/{split}" in splits:
                recs_s = splits[f"{domain}/{split}"]
                results[f"fill@{domain}/{split}"], _ = score_fill_rule(rule["threshold"], recs_s, cache[(f"{domain}/{split}", "new")][1])

    # paired tests on the in-scale test split
    recs = splits["inscale/test"]
    labels = np.array([TOPOLOGY_TO_IDX[r.topology] for r in recs])
    new_ok = cache[("inscale/test", "new")][0].argmax(1) == labels
    old_ok = cache[("inscale/test", "old")][0].argmax(1) == labels
    _, fill_ok = score_fill_rule(fill_rule["inscale"]["threshold"], recs, cache[("inscale/test", "new")][1])
    paired = dict(new_vs_fill=mcnemar_exact(new_ok, fill_ok), new_vs_old=mcnemar_exact(new_ok, old_ok))

    # data description + arrays for the histograms
    def describe(domain_records, domain_placements):
        fills = {c: [] for c in CLASSES}
        for r in domain_records:
            fills[r.topology].append(float((load_voxel(r.voxel_path, 64) > 0).mean()))
        info = [domain_placements.get(str(r.voxel_path)) or {} for r in domain_records] if domain_placements else []
        d = dict(n=len(domain_records), n_per_class={c: len(v) for c, v in fills.items()},
                 fill={c: dict(min=float(np.min(v)), median=float(np.median(v)), max=float(np.max(v))) for c, v in fills.items()},
                 fill_values=fills)
        if info:
            ncuts = [len(i.get("cuts", [])) for i in info]
            d.update(cut_share=float(np.mean([c > 0 for c in ncuts])), two_cut_share=float(np.mean([c == 2 for c in ncuts])),
                     size_fraction=dict(min=min(i["size_fraction"] for i in info), max=max(i["size_fraction"] for i in info)))
        return d

    data = {"inscale/train": describe(splits["inscale/train"], placements),
            "inscale/val": describe(splits["inscale/val"], placements),
            "inscale/test": describe(splits["inscale/test"], placements),
            "stretched/train": describe(splits["stretched/train"], None),
            "stretched/val": describe(splits["stretched/val"], None)}

    out = dict(
        checkpoints=dict(new=dict(path=str(args.new_checkpoint.relative_to(ROOT)), selected_epoch=int(new_epoch)),
                         old=dict(path=str(args.old_checkpoint.relative_to(ROOT)), selected_epoch=int(old_epoch))),
        split=dict(seed=args.seed, val_fraction=args.val_fraction),
        results=results, fill_rule=fill_rule, paired_tests=paired, data=data,
        torch=torch.__version__, cuda=bool(torch.cuda.is_available()),
        parameters=int(sum(p.numel() for p in nets["new"].parameters())),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    def line(key):
        r = results[key]
        return f"{key:24} n={r['n']:4d}  acc {r['accuracy']:.3f}  (95% CI {r['ci95'][0]:.3f}-{r['ci95'][1]:.3f})"
    print("selected epochs: new", new_epoch, " old", old_epoch)
    for key in ("new@inscale/val", "new@inscale/test", "old@inscale/val", "old@inscale/test", "fill@inscale/val", "fill@inscale/test",
                "old@stretched/val", "new@stretched/val", "fill@stretched/val"):
        print(line(key))
    print("fill rule (fitted on train):", fill_rule)
    print("paired tests on the in-scale test set:", paired)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
