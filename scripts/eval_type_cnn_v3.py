#!/usr/bin/env python3
"""
Held-out evaluation of Type CNN checkpoints on four test sets, broken down by what the body is (plain, planar cut,
carved, full-field cube), by class and by noise, with Wilson intervals and exact paired tests (McNemar).

Test sets (dataset_gen/local_dataset_inscale, never used to train or to choose a checkpoint):
  test        400  plain and planar-cut bodies, the set of the first two reports (v1, v2)
  test_ff     400  the same with 30 % of the boxes the completely filled grid
  test_carve  400  bodies carved by one cutter (box - sphere, sphere - box), manifold-verified, noise on ~50 %
  test_mixed  400  the mix the new training data has: plain, cut and carved, noise on ~50 %

    python3 scripts/eval_type_cnn_v3.py --model v2=checkpoints_inscale_v2/type_cnn --model A=checkpoints_type_v3/carve_only/type_cnn
Each --model is NAME=DIR with DIR holding best_model.pth. Writes docs/data/type_cnn_v3_eval.json.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from report_type_cnn import load_model, mcnemar_exact, predict, wilson  # noqa: E402
from training.manifest import load_manifest  # noqa: E402
from training.topology_schema import TOPOLOGY_TO_IDX  # noqa: E402

DATA = ROOT / "dataset_gen" / "local_dataset_inscale"
SETS = ("test", "test_ff", "test_carve", "test_mixed")


def kind_of(record) -> str:
    p = record.placement or {}
    if p.get("full_field"):
        return "full_field"
    if p.get("carve"):
        return "carved"
    return "plain" if not p.get("cuts") else "cut"


def noise_of(record) -> str:
    """Older sets did not record the flag: every sample of them was noised."""
    p = record.placement or {}
    return "noised" if p.get("noise", True) else "clean"


def cell(mask, correct):
    n = int(mask.sum())
    k = int(correct[mask].sum())
    return dict(n=n, correct=k, accuracy=k / n if n else None, ci95=wilson(k, n) if n else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="NAME=DIR (DIR holds best_model.pth)")
    ap.add_argument("--reference", default=None, help="model name for the paired tests (default: the first --model)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/data/type_cnn_v3_eval.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = {}
    for spec in args.model:
        name, directory = spec.split("=", 1)
        paths[name] = Path(directory) / "best_model.pth"
    nets = {name: load_model(path, device) for name, path in paths.items()}
    epochs = {name: int(torch.load(path, map_location="cpu", weights_only=False)["epoch"]) for name, path in paths.items()}
    ref = args.reference or next(iter(paths))

    results, correct_by = {}, {}
    for set_name in SETS:
        records = load_manifest(DATA / set_name, require_voxel=True)
        labels = np.array([TOPOLOGY_TO_IDX[r.topology] for r in records])
        kinds = np.array([kind_of(r) for r in records])
        noise = np.array([noise_of(r) for r in records])
        classes = np.array([r.topology for r in records])
        for name, model in nets.items():
            probs, _ = predict(model, records, device)
            correct = probs.argmax(1) == labels
            correct_by[(name, set_name)] = correct
            row = dict(all=cell(np.ones(len(records), bool), correct),
                       mean_p_true=float(probs[np.arange(len(labels)), labels].mean()),
                       nll=float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-9, 1)).mean()))
            row["by_kind"] = {k: cell(kinds == k, correct) for k in sorted(set(kinds))}
            row["by_class"] = {c: cell(classes == c, correct) for c in sorted(set(classes))}
            row["by_noise"] = {n: cell(noise == n, correct) for n in sorted(set(noise))}
            row["by_kind_class"] = {f"{k}/{c}": cell((kinds == k) & (classes == c), correct)
                                    for k in sorted(set(kinds)) for c in sorted(set(classes))}
            results[f"{name}@{set_name}"] = row
    paired = {}
    for set_name in SETS:
        for name in nets:
            if name != ref:
                paired[f"{name}_vs_{ref}@{set_name}"] = mcnemar_exact(correct_by[(name, set_name)], correct_by[(ref, set_name)])
    args.out.write_text(json.dumps(dict(checkpoints={n: str(p) for n, p in paths.items()}, selected_epoch=epochs, reference=ref,
                                        results=results, paired=paired), indent=1))

    width = max(len(n) for n in nets)
    for set_name in SETS:
        print(f"\n== {set_name} ==")
        for name in nets:
            r = results[f"{name}@{set_name}"]
            a = r["all"]
            kinds = "  ".join(f"{k} {c['accuracy']:.3f}({c['n']})" for k, c in r["by_kind"].items() if c["n"])
            print(f"{name:{width}s}  acc {a['accuracy']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]  nll {r['nll']:.3f}   {kinds}")
    print("\nselected epochs", epochs)
    for key, p in paired.items():
        print(f"McNemar {key}: only-this-right {p['a_only']}, only-reference-right {p['b_only']}, p={p['p']:.3g}")


if __name__ == "__main__":
    main()
