#!/usr/bin/env python3
"""
Train the box Dimension CNN on PCA-aligned in-scale volumes: voxel grid -> the box's three half extents, LONGEST
FIRST, in voxels of the grid (so absolute size is a label, as the in-scale data makes it observable).

It is the counterpart of training/train_rotation_sym.py and follows the same convention (training/
rotation_symmetry.py): rotation and dimensions together describe the box only if the dimensions are listed in the
order of the canonical axes, longest first. The old Dimension CNN predicted (x, y, z) in an arbitrary order that
the voxels do not determine, and (in the stretched voxelizer mode) not even the aspect ratio was visible.

Target: log half extent in voxels, h_i = k * ratio_i * 10 mm / voxel_mm  (k = placement.scale_factor of the sample,
10 mm = the macro's realization scale), sorted descending. The head predicts l1 and two non-negative steps down, so
the output is ordered by construction. Loss: L1 on the logs (a relative error, symmetric for over/under).

    python3 -m training.train_box_dims --dataset-root dataset_gen/local_dataset_inscale/carve \
        --extra-root dataset_gen/local_dataset_inscale/rot2 --checkpoint-dir checkpoints_box_dims/dims_cnn --epochs 40
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training import rotation_symmetry as rs
from training.backbone import Voxel3DBackbone
from training.train_common import (
    check_finite, common_arg_parser, load_split, resolve_device, run_training_loop,
)
from training.train_rotation_sym import SymRotationDataset, usable_boxes

REALIZATION_SCALE_MM = 10.0


def half_extents_vox(record, grid_mm: float = 40.0, resolution: int = 64) -> np.ndarray:
    """Sorted (longest first) half extents of the record's base box, in voxels of the in-scale grid."""
    d = np.array([record.parameters[k] for k in rs.DIM_KEYS], dtype=np.float64)
    k = float(record.placement["scale_factor"])
    return np.sort(d)[::-1] * k * REALIZATION_SCALE_MM / (grid_mm / resolution)


class BoxDimsDataset(SymRotationDataset):
    """PCA-aligned volumes (optionally jittered by a few degrees) -> log sorted half extents in voxels."""

    def __init__(self, records, resolution: int = 64, jitter_deg: float = 0.0):
        super().__init__(records, resolution, augment=False, group="sym", align=True, jitter_deg=jitter_deg)
        self.logs = [np.log(half_extents_vox(r, resolution=resolution)).astype(np.float32) for r in records]

    def __getitem__(self, idx):
        volume, _, _ = super().__getitem__(idx)
        return volume, torch.from_numpy(self.logs[idx])


class BoxDimsCNN(nn.Module):
    def __init__(self, feature_dim: int = 256):
        super().__init__()
        self.backbone = Voxel3DBackbone(feature_dim=feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3) log half extents in voxels, non-increasing."""
        a, b, c = self.head(self.backbone(x)).unbind(1)
        l1 = a
        l2 = l1 - F.softplus(b)
        l3 = l2 - F.softplus(c)
        return torch.stack((l1, l2, l3), dim=1)


def train_step(model, batch, device, optimizer):
    x, y = (t.to(device) for t in batch)
    optimizer.zero_grad(set_to_none=True)
    loss = F.l1_loss(model(x), y)
    check_finite(loss, "train_box_dims train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item()}


def eval_step(model, batch, device):
    x, y = (t.to(device) for t in batch)
    pred = model(x)
    rel = (pred - y).abs().exp() - 1.0                       # |ratio - 1| up to the direction: a factor exp(|dlog|)
    return {"loss": F.l1_loss(pred, y).item(), "median_rel_err": float(rel.median()), "within10pct": float((rel < 0.1).float().mean())}


def main() -> None:
    parser = common_arg_parser("Train the sorted box Dimension CNN")
    parser.add_argument("--extra-root", type=Path, action="append", default=[])
    parser.add_argument("--jitter-deg", type=float, default=10.0)
    args = parser.parse_args()

    device = resolve_device(args.device)
    train, val = load_split(args.dataset_root, args.val_fraction, args.seed)
    for extra in args.extra_root:
        t, v = load_split(extra, args.val_fraction, args.seed)
        train, val = train + t, val + v
    train, val = usable_boxes(train), usable_boxes(val)
    print(f"train boxes: {len(train)}  val boxes: {len(val)}  jitter={args.jitter_deg}")
    train_loader = DataLoader(BoxDimsDataset(train, args.resolution, args.jitter_deg), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(BoxDimsDataset(val, args.resolution, 0.0), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    torch.manual_seed(args.seed)
    model = BoxDimsCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(
        dict(jitter_deg=args.jitter_deg, epochs=args.epochs, seed=args.seed, n_train=len(train), n_val=len(val),
             target="log half extent in voxels (64^3 grid of 40 mm), longest first, PCA-aligned input"), indent=1))
    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      scheduler=scheduler, device=device, epochs=args.epochs, patience=args.patience,
                      checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="median_rel_err")


if __name__ == "__main__":
    main()
