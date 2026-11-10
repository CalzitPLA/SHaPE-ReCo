#!/usr/bin/env python3
"""
Train the Rotation CNN: voxel grid + known topology -> orientation
quaternion.

Topology-conditioned (the backbone feature is concatenated with a
one-hot topology vector before the regression head) because a
dataset_gen "rotation" sample's dimensions vary while orientation is
held fixed per group - the network needs the topology to interpret
shape cues consistently across very differently-proportioned instances
of the same class. At inference time in the reconstruction pipeline,
topology comes from the Type CNN's prediction (see reco_engine).

Loss: quaternion geodesic distance, 1 - |q_pred . q_gt| (after
normalizing q_pred), rather than a raw per-component regression loss.
Component-wise MSE/SmoothL1 on quaternions is ill-posed here because q
and -q represent the same rotation but can be numerically far apart;
the dot-product loss is invariant to that sign ambiguity, unlike the
Euler-angle + Tanh approach used in the old, never-trained
FeatureNet/ShaPe_Integra_v3.py (which also has gimbal/wraparound
issues MAX_ROTATION_DEGREES-scaled Tanh doesn't solve).

Usage:
    python3 -m training.train_rotation_cnn \\
        --dataset-root dataset_gen/test_output_rotation \\
        --checkpoint-dir checkpoints/rotation_cnn
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.backbone import Voxel3DBackbone
from training.topology_schema import NUM_TOPOLOGIES, TOPOLOGY_TO_IDX
from training.train_common import (
    check_finite, common_arg_parser, load_split, resolve_device, run_training_loop,
)
from training.voxel_dataset import VoxelDataset


class RotationDataset(VoxelDataset):
    def __getitem__(self, idx):
        tensor, record = self._load(idx)
        topo_idx = TOPOLOGY_TO_IDX[record.topology]
        quat = torch.tensor(
            [record.rotation["qx"], record.rotation["qy"],
             record.rotation["qz"], record.rotation["qw"]],
            dtype=torch.float32,
        )
        return tensor, torch.tensor(topo_idx, dtype=torch.long), quat


class RotationCNN(nn.Module):
    def __init__(self, num_topologies: int = NUM_TOPOLOGIES, feature_dim: int = 256):
        super().__init__()
        self.backbone = Voxel3DBackbone(feature_dim=feature_dim)
        self.topo_embed = nn.Embedding(num_topologies, 32)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + 32, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor, topo_idx: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embed = self.topo_embed(topo_idx)
        raw_quat = self.head(torch.cat([features, embed], dim=1))
        return F.normalize(raw_quat, dim=1, eps=1e-8)


def geodesic_quat_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - |<pred, target>|, sign-invariant (q and -q are the same rotation)."""
    dot = (pred * target).sum(dim=1).abs().clamp(max=1.0)
    return (1.0 - dot).mean()


def train_step(model, batch, device, optimizer):
    voxels, topo_idx, quat_target = batch
    voxels, topo_idx, quat_target = voxels.to(device), topo_idx.to(device), quat_target.to(device)
    optimizer.zero_grad(set_to_none=True)
    quat_pred = model(voxels, topo_idx)
    loss = geodesic_quat_loss(quat_pred, quat_target)
    check_finite(loss, "train_rotation_cnn train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item()}


def eval_step(model, batch, device):
    voxels, topo_idx, quat_target = batch
    voxels, topo_idx, quat_target = voxels.to(device), topo_idx.to(device), quat_target.to(device)
    quat_pred = model(voxels, topo_idx)
    loss = geodesic_quat_loss(quat_pred, quat_target)
    angle_error_deg = 2.0 * torch.acos((quat_pred * quat_target).sum(dim=1).abs().clamp(max=1.0)) * (180.0 / 3.14159265)
    return {"loss": loss.item(), "angle_error_deg": angle_error_deg.mean().item()}


def main() -> None:
    parser = common_arg_parser("Train the rotation-regression CNN")
    args = parser.parse_args()

    device = resolve_device(args.device)
    train_records, val_records = load_split(args.dataset_root, args.val_fraction, args.seed)
    print(f"train samples: {len(train_records)}  val samples: {len(val_records)}")

    train_loader = DataLoader(
        RotationDataset(train_records, args.resolution),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        RotationDataset(val_records, args.resolution),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
    )

    model = RotationCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_training_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        epochs=args.epochs, patience=args.patience, checkpoint_dir=args.checkpoint_dir,
        train_step=train_step, eval_step=eval_step, extra_metrics_key="angle_error_deg",
    )


if __name__ == "__main__":
    main()
