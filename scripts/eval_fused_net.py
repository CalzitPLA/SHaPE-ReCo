#!/usr/bin/env python3
"""
Held-out evaluation of training/train_fused.py's FusedNet: one network, both tasks in the SAME forward pass
(type classification and box orientation), read off the same test sets and with the same metrics
scripts/eval_type_fused.py and scripts/eval_rotation.py already use for the per-task occupancy/fused baselines -
so a FusedNet row can sit in the same tables (docs/thesis_fused_input.tex, Section "FusedNet: status, not yet a
result").

    python3 scripts/eval_fused_net.py --model checkpoints_fused/net_fused

Writes into the SAME two JSON files the per-task eval scripts write (default: docs/data/type_fused_eval.json,
docs/data/rotation_eval_fused.json), adding a "fused_net" key rather than replacing the file - re-run
scripts/make_fused_tex.py afterwards to add it to the tables/figures/macros.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from report_type_cnn import placements_by_voxel_path  # noqa: E402
from training import rotation_symmetry as rs  # noqa: E402
from training.manifest import load_manifest  # noqa: E402
from training.topology_schema import TOPOLOGY_TO_IDX  # noqa: E402

from reco_engine.fused_predict import load_fused_model  # noqa: E402

TYPE_DATA = ROOT / "dataset_gen" / "local_dataset_inscale"
ROT_DATA = ROOT / "dataset_gen" / "local_dataset_inscale"


@torch.no_grad()
def _forward(handle, volumes, batch: int = 8):
    from training.field_input import encode
    out_type, out_pose = [], []
    for i in range(0, len(volumes), batch):
        chunk = volumes[i:i + batch]
        x = torch.from_numpy(np.stack([encode(v.astype(np.float32) * 2 - 1, handle.input_kind, handle.sdf_truncation) for v in chunk]))
        result = handle.model(x.to(handle.device))
        out_type.append(result["type"].argmax(1).cpu().numpy())
        out_pose.append(result["pose"].double().cpu().numpy())
    return np.concatenate(out_type), np.concatenate(out_pose)


def eval_type(handle, sets):
    recs = []
    for name in sets:
        recs += load_manifest(TYPE_DATA / name, require_voxel=True)
    y = np.array([TOPOLOGY_TO_IDX[r.topology] for r in recs])
    vols = [np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) for r in recs]
    pred, _pose = _forward(handle, vols)
    acc = float((pred == y).mean())
    per_class = {t: float((pred[y == i] == i).mean()) for t, i in TOPOLOGY_TO_IDX.items() if (y == i).any()}
    confusions = collections.Counter((int(a), int(b)) for a, b in zip(y, pred) if a != b)
    print(f"type: {len(recs)} samples, accuracy {100 * acc:5.1f} %  per-class {per_class}")
    return dict(input=handle.input_kind, n=len(recs), accuracy=acc, per_class=per_class,
               confusions={f"{list(TOPOLOGY_TO_IDX)[a]}->{list(TOPOLOGY_TO_IDX)[b]}": c for (a, b), c in confusions.items()})


def _block(err: np.ndarray) -> dict:
    n = len(err)
    return dict(n=n, median=float(np.median(err)), mean=float(err.mean()),
               **{f"within{t}": float((err < t).mean()) for t in (5, 10, 20)})


def eval_rotation(handle, sets):
    recs, placements = [], {}
    for name in sets:
        placements.update(placements_by_voxel_path(ROT_DATA / name))
        recs += load_manifest(ROT_DATA / name, require_voxel=True)
    recs = [r for r in recs if r.topology == "box" and not (placements.get(str(r.voxel_path)) or {}).get("full_field")]
    vols = [np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) for r in recs]
    targets = [rs.record_target([r.rotation[k] for k in ("qx", "qy", "qz", "qw")], [r.parameters[k] for k in rs.DIM_KEYS]) for r in recs]
    M = np.stack([t[0] for t in targets]); mask = np.stack([t[1] for t in targets])
    _pred_type, pose = _forward(handle, vols)
    err = rs.symmetric_error(pose, M, mask)
    print(f"rotation: {len(recs)} boxes, median {np.median(err):.1f} deg, within10 {100 * (err < 10).mean():.1f} %")
    return dict(all=_block(err))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="a training/train_fused.py --checkpoint-dir")
    ap.add_argument("--checkpoint-name", default="best_model.pth")
    ap.add_argument("--type-sets", nargs="+", default=["test_mixed", "test_cyl"])
    ap.add_argument("--rotation-sets", nargs="+", default=["test", "test_ff"])
    ap.add_argument("--type-out", type=Path, default=ROOT / "docs/data/type_fused_eval.json")
    ap.add_argument("--rotation-out", type=Path, default=ROOT / "docs/data/rotation_eval_fused.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    handle = load_fused_model(args.model, args.checkpoint_name, device)
    type_result = eval_type(handle, args.type_sets)
    rotation_result = eval_rotation(handle, args.rotation_sets)

    for out_path, key, value in ((args.type_out, "fused_net", type_result), (args.rotation_out, "fused_net", rotation_result)):
        existing = json.loads(out_path.read_text()) if out_path.is_file() else {}
        if out_path == args.type_out:
            existing[key] = value
        else:
            existing.setdefault("methods", {})[key] = value
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(existing, indent=1))
    print(f"updated {args.type_out} and {args.rotation_out} with the 'fused_net' entry")


if __name__ == "__main__":
    main()
