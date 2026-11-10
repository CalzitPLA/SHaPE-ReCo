#!/usr/bin/env python3
"""
Per-sample predictions of a hierarchical Type checkpoint on dataset roots, with every sample's generator parameters:
the table that says WHERE in the parameter space a network is wrong (which cones it calls cylinders, which sectors
it calls full), so that the next data increment can be aimed, and labels the grid cannot resolve can be found.

    python3 scripts/type_predictions.py --checkpoint <dir>/type_cnn --roots <root> [<root> ...] --out pred.jsonl
                                        [--batch-size 32] [--num-workers 4] [--device cuda:0]

<dir>/type_cnn holds best_model.pth and model_card.json (training/train_type_cnn.py); the input kind, resolution and
channels are read from the card. One JSON line per sample of the hierarchical label space:
  root, voxel_path, flat (true "parent/subclass"), pred_parent, pred_flat (end to end: the subclass head read through
  the PREDICTED parent), p_parent (softmax over the 5 parents), p_true, and the generator's record: dims_mm,
  parameters, segment, sample_kind, the placement's voxels / extent_vox / scale_factor / cuts, and its other keys.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.manifest import load_manifest                                                        # noqa: E402
from training.subclass_schema import FLAT_SUBCLASSES, PARENT_MASK, PARENTS, filter_records, labels_of   # noqa: E402
from training.train_type_cnn import load_checkpoint_model, predict_records                         # noqa: E402

HEAVY = ("field", "inside", "manifold")            # placement entries that are file names or bookkeeping


def _plain(v):
    if isinstance(v, (np.generic,)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="the type_cnn directory (best_model.pth, model_card.json)")
    ap.add_argument("--roots", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)

    card = json.loads((a.checkpoint / "model_card.json").read_text())
    assert card.get("hierarchical"), "a hierarchical checkpoint is needed"
    device = torch.device(a.device)
    model = load_checkpoint_model(a.checkpoint / "best_model.pth", card["in_channels"], device)
    kwargs = dict(resolution=card["resolution"], input_kind=card["input_kind"])
    n_all = 0
    with a.out.open("w") as out:
        for root in a.roots:
            records, skipped = filter_records(load_manifest(root, require_voxel=True))
            pred = predict_records(model, records, device, a.batch_size, num_workers=a.num_workers, **kwargs)
            pl, sl = pred["parent_logits"], pred["sub_logits"]
            prob = np.exp(pl - pl.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            pp = pl.argmax(axis=1)
            ps = np.where(PARENT_MASK[pp], sl, -np.inf).argmax(axis=1)
            for i, r in enumerate(records):
                yp, ys = labels_of(r)
                meta = dict(getattr(r, "meta", None) or {})
                placement = {k: v for k, v in (r.placement or {}).items() if k not in HEAVY}
                row = dict(root=str(root), voxel_path=str(r.voxel_path), flat=FLAT_SUBCLASSES[ys],
                           pred_parent=PARENTS[pp[i]], pred_flat=FLAT_SUBCLASSES[ps[i]],
                           p_parent=[round(float(x), 4) for x in prob[i]], p_true=round(float(prob[i, yp]), 4),
                           dims_mm=meta.get("dims_mm"), parameters=meta.get("parameters"), segment=meta.get("segment"),
                           sample_kind=meta.get("sample_kind"), placement=placement,
                           meta={k: v for k, v in meta.items() if k not in ("dims_mm", "parameters", "segment", "sample_kind",
                                                                              "rotation", "rotation_canonical", "center_mm")})
                out.write(json.dumps(_plain(row)) + "\n")
            n_all += len(records)
            ok = float((pp == np.array([labels_of(r)[0] for r in records])).mean()) if records else float("nan")
            print(f"{root}: {len(records)} samples (skipped {dict(skipped)}), parent accuracy {ok:.4f}", flush=True)
    print(f"{n_all} rows -> {a.out}")


if __name__ == "__main__":
    main()
