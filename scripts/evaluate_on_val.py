#!/usr/bin/env python3
"""
Load each of the three trained CNNs' best checkpoint and run it against
its own held-out validation split (the exact same split
training.train_common.load_split reconstructs from --seed/--val-fraction,
so this evaluates on data the model never trained on, not a re-check of
training accuracy).

Prints one row per validation sample (predicted vs. actual; suppress
with --summary-only) and, more importantly, the summaries that make the
numbers interpretable rather than just reportable:

  rotation   mean geodesic error vs. the ~126.5 deg expected from random
             guessing, and the number of DISTINCT validation groups - in
             rotation mode every sample in a group shares one
             orientation, so that count (not the sample count) is the
             effective sample size.
  dimension  network MAE vs. the MAE of always predicting the training-set
             mean, plus prediction std vs. target std (a prediction std
             far below the target std means a near-constant output that
             is not reading the input).

Usage:
    python3 scripts/evaluate_on_val.py \\
        --type-dataset-root dataset_gen/local_dataset_box_sphere/type \\
        --rotation-dataset-root dataset_gen/local_dataset_box_sphere/rotation \\
        --dimension-dataset-root dataset_gen/local_dataset_box_sphere/measure \\
        --checkpoint-root checkpoints
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
import sys  # noqa: E402
sys.path.insert(0, str(SANDBOX_ROOT))

from training.topology_schema import IDX_TO_TOPOLOGY, TOPOLOGY_TO_IDX, normalize_params, param_mask  # noqa: E402
from training.train_common import load_split, resolve_device  # noqa: E402
from training.voxel_dataset import load_voxel  # noqa: E402
from training.train_type_cnn import TypeCNN  # noqa: E402
from training.train_rotation_cnn import RotationCNN  # noqa: E402
from training.train_dimension_cnn import DimensionCNN  # noqa: E402


# Mean rotation angle between two independent uniformly random rotations:
# angle density (1 - cos t)/pi on [0, pi]  ->  E[t] = pi/2 + 2/pi rad.
CHANCE_ROTATION_ERROR_DEG = math.degrees(math.pi / 2 + 2 / math.pi)


def _load_checkpoint(model: torch.nn.Module, checkpoint_dir: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()


def evaluate_type(dataset_root: Path, checkpoint_dir: Path, resolution: int,
                   val_fraction: float, seed: int, device: torch.device) -> None:
    print("\n=== Type CNN: predictions on held-out validation set ===")
    _, val_records = load_split(dataset_root, val_fraction, seed)
    model = TypeCNN().to(device)
    _load_checkpoint(model, checkpoint_dir, device)

    correct = 0
    with torch.no_grad():
        for record in val_records:
            vol = load_voxel(record.voxel_path, resolution)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(probs.argmax())
            pred_topo = IDX_TO_TOPOLOGY[pred_idx]
            confidence = float(probs[pred_idx])
            hit = pred_topo == record.topology
            correct += int(hit)
            print(f"  {'OK ' if hit else 'ERR'}  true={record.topology:8s} "
                  f"pred={pred_topo:8s} confidence={confidence:.3f}  {record.voxel_path.name}")

    print(f"  -> accuracy: {correct}/{len(val_records)} = {100 * correct / len(val_records):.1f}%")


def evaluate_rotation(dataset_root: Path, checkpoint_dir: Path, resolution: int,
                       val_fraction: float, seed: int, device: torch.device,
                       verbose: bool = True) -> None:
    print("\n=== Rotation CNN: predictions on held-out validation set ===")
    _, val_records = load_split(dataset_root, val_fraction, seed)
    model = RotationCNN().to(device)
    _load_checkpoint(model, checkpoint_dir, device)

    errors_by_topo: dict[str, list[float]] = {}
    with torch.no_grad():
        for record in val_records:
            vol = load_voxel(record.voxel_path, resolution)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            topo_idx = torch.tensor([TOPOLOGY_TO_IDX[record.topology]], device=device)
            quat_pred = model(x, topo_idx)[0].cpu()
            quat_true = torch.tensor([record.rotation["qx"], record.rotation["qy"],
                                       record.rotation["qz"], record.rotation["qw"]])
            dot = float((quat_pred * quat_true).sum().abs().clamp(max=1.0))
            angle_error_deg = 2.0 * math.degrees(math.acos(dot))
            errors_by_topo.setdefault(record.topology, []).append(angle_error_deg)
            if verbose:
                print(f"  true_topo={record.topology:8s} angle_error={angle_error_deg:7.2f} deg  "
                      f"{record.voxel_path.name}")

    all_errors = [e for errs in errors_by_topo.values() for e in errs]
    n_groups = len({r.group_key for r in val_records})
    print("  -> mean angle error by topology:")
    for topo, errors in sorted(errors_by_topo.items()):
        print(f"       {topo:8s} n={len(errors):4d}  mean={sum(errors) / len(errors):7.2f} deg")
    print(f"  -> overall mean {sum(all_errors) / len(all_errors):.2f} deg  "
          f"(random guessing: {CHANCE_ROTATION_ERROR_DEG:.2f} deg)")
    print(f"  -> {len(val_records)} val samples from {n_groups} distinct val groups "
          f"(= distinct ground-truth orientations; the effective sample size)")


def evaluate_dimension(dataset_root: Path, checkpoint_dir: Path, resolution: int,
                        val_fraction: float, seed: int, device: torch.device,
                        verbose: bool = True) -> None:
    print("\n=== Dimension CNN: predictions on held-out validation set ===")
    train_records, val_records = load_split(dataset_root, val_fraction, seed)
    model = DimensionCNN().to(device)
    _load_checkpoint(model, checkpoint_dir, device)

    def targets(records):
        out: dict[str, list] = {}
        for r in records:
            out.setdefault(r.topology, []).append(normalize_params(r.topology, r.parameters))
        return {k: np.array(v) for k, v in out.items()}

    train_targets, val_targets = targets(train_records), targets(val_records)

    preds: dict[str, list] = {}
    with torch.no_grad():
        for record in val_records:
            vol = load_voxel(record.voxel_path, resolution)
            x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)
            topo_idx = torch.tensor([TOPOLOGY_TO_IDX[record.topology]], device=device)
            pred = model(x, topo_idx)[0].cpu()
            target = torch.tensor(normalize_params(record.topology, record.parameters))
            mask = torch.tensor(param_mask(record.topology))
            mae = float(((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0))
            preds.setdefault(record.topology, []).append(pred.numpy())
            if verbose:
                print(f"  true_topo={record.topology:8s} mae={mae:.4f}  "
                      f"pred={[round(v, 2) for v in pred[mask.bool()].tolist()]}  "
                      f"true={[round(v, 2) for v in target[mask.bool()].tolist()]}  "
                      f"{record.voxel_path.name}")

    n_groups = {t: len({r.group_key for r in val_records if r.topology == t}) for t in val_targets}
    print("  -> per topology (network vs. always predicting the training-set mean):")
    for topo in sorted(val_targets):
        idx = np.array(param_mask(topo)).astype(bool)
        pred_arr = np.array(preds[topo])[:, idx]
        true_arr = val_targets[topo][:, idx]
        net_mae = np.abs(pred_arr - true_arr).mean()
        const_mae = np.abs(true_arr - train_targets[topo][:, idx].mean(axis=0)).mean()
        print(f"       {topo:8s} n={len(true_arr):4d} ({n_groups[topo]} distinct val groups)  "
              f"network MAE={net_mae:.4f}  constant-predictor MAE={const_mae:.4f}  "
              f"({100 * (const_mae - net_mae) / const_mae:+.1f}% vs constant)")
        print(f"                pred std={[round(float(v), 3) for v in pred_arr.std(axis=0)]}  "
              f"target std={[round(float(v), 3) for v in true_arr.std(axis=0)]}  "
              f"(pred std << target std means near-constant output)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type-dataset-root", type=Path, default=None)
    parser.add_argument("--rotation-dataset-root", type=Path, default=None)
    parser.add_argument("--dimension-dataset-root", type=Path, default=None)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--summary-only", action="store_true",
                         help="Skip the per-sample rows; print only the summaries/baselines.")
    args = parser.parse_args()

    device = resolve_device(args.device)

    if args.type_dataset_root:
        evaluate_type(args.type_dataset_root, args.checkpoint_root / "type_cnn",
                      args.resolution, args.val_fraction, args.seed, device)
    if args.rotation_dataset_root:
        evaluate_rotation(args.rotation_dataset_root, args.checkpoint_root / "rotation_cnn",
                           args.resolution, args.val_fraction, args.seed, device,
                           verbose=not args.summary_only)
    if args.dimension_dataset_root:
        evaluate_dimension(args.dimension_dataset_root, args.checkpoint_root / "dimension_cnn",
                            args.resolution, args.val_fraction, args.seed, device,
                            verbose=not args.summary_only)


if __name__ == "__main__":
    main()
