#!/usr/bin/env python3
"""
Train the Dimension CNN: voxel grid + known topology -> normalized
per-topology shape parameters (ratios/angles, up to MAX_PARAMS=10 slots,
masked per topology - see training/topology_schema.py).

Topology-conditioned like the Rotation CNN, for the same reason
(dataset_gen "measure" samples vary rotation while dimensions are held
fixed per group, so the network needs the topology to know which of the
output slots are meaningful and how to interpret shape cues).

Usage:
    python3 -m training.train_dimension_cnn \\
        --dataset-root dataset_gen/test_output_measure \\
        --checkpoint-dir checkpoints/dimension_cnn
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.backbone import Voxel3DBackbone
from training.topology_schema import (
    MAX_PARAMS, NUM_TOPOLOGIES, TOPOLOGY_TO_IDX, normalize_params, param_mask,
)
from training.train_common import (
    check_finite, common_arg_parser, load_split, resolve_device, run_training_loop,
)
from training.voxel_dataset import VoxelDataset


class DimensionDataset(VoxelDataset):
    def __getitem__(self, idx):
        tensor, record = self._load(idx)
        topo_idx = TOPOLOGY_TO_IDX[record.topology]
        target = torch.tensor(normalize_params(record.topology, record.parameters), dtype=torch.float32)
        mask = torch.tensor(param_mask(record.topology), dtype=torch.float32)
        return tensor, torch.tensor(topo_idx, dtype=torch.long), target, mask


class DimensionCNN(nn.Module):
    def __init__(self, num_topologies: int = NUM_TOPOLOGIES, max_params: int = MAX_PARAMS, feature_dim: int = 256):
        super().__init__()
        self.backbone = Voxel3DBackbone(feature_dim=feature_dim)
        self.topo_embed = nn.Embedding(num_topologies, 32)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, max_params),
            nn.Tanh(),  # matches the [-1, 1] normalization in topology_schema
        )

    def forward(self, x: torch.Tensor, topo_idx: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embed = self.topo_embed(topo_idx)
        return self.head(torch.cat([features, embed], dim=1))


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_elem = nn.functional.smooth_l1_loss(pred, target, reduction="none") * mask
    denom = mask.sum().clamp(min=1.0)
    return per_elem.sum() / denom


def train_step(model, batch, device, optimizer):
    voxels, topo_idx, target, mask = batch
    voxels, topo_idx, target, mask = (t.to(device) for t in (voxels, topo_idx, target, mask))
    optimizer.zero_grad(set_to_none=True)
    pred = model(voxels, topo_idx)
    loss = masked_smooth_l1(pred, target, mask)
    check_finite(loss, "train_dimension_cnn train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item()}


def eval_step(model, batch, device):
    voxels, topo_idx, target, mask = batch
    voxels, topo_idx, target, mask = (t.to(device) for t in (voxels, topo_idx, target, mask))
    pred = model(voxels, topo_idx)
    loss = masked_smooth_l1(pred, target, mask)
    mae = ((pred - target).abs() * mask).sum() / mask.sum().clamp(min=1.0)
    return {"loss": loss.item(), "mae": mae.item()}


def main() -> None:
    parser = common_arg_parser("Train the per-topology dimension-regression CNN")
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_records, val_records = load_split(args.dataset_root, args.val_fraction, args.seed)
    print(f"train samples: {len(train_records)}  val samples: {len(val_records)}")

    train_loader = DataLoader(
        DimensionDataset(train_records, args.resolution),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        DimensionDataset(val_records, args.resolution),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = DimensionCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_training_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        epochs=args.epochs, patience=args.patience, checkpoint_dir=args.checkpoint_dir,
        train_step=train_step, eval_step=eval_step, extra_metrics_key="mae",
    )


if __name__ == "__main__":
    main()
