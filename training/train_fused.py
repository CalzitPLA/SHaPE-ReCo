#!/usr/bin/env python3
"""
One network for the three questions, on the fused input: occupancy + signed distance + the vector to the nearest surface point.

The pipeline used three separate CNNs on the occupancy grid (Type, Rotation, Dimension); the SDF and the vector-field route showed what the
occupancy hides (docs/vectorfield_reconstruction.tex). `FusedNet` reads the five channels of training/field_input.py with one shared 3D backbone
and three heads:

  type      logits over the topologies (cross entropy)
  pose      a rotation matrix from the 6D representation, for BOXES: the array-frame orientation of the longest-first axes up to the box's
            symmetry group, symmetric geodesic loss (training/rotation_symmetry.py). Not defined for spheres; cylinders are left out.
  size      three log lengths in voxels of the in-scale grid: box = sorted half extents (longest first), sphere = radius, cylinder = radius and half length
            (L1 on the logs = a relative error; absolute size is observable in the in-scale data)

Every loss is masked to the samples it applies to, so one batch mixes topologies. The training augmentation rotates the OCCUPANCY by one of the 24
signed permutations of the grid axes and derives SDF and vectors from it afterwards (exactly equivariant), rotating the box label with it.
`--input occupancy` gives the same network on the baseline input, for a controlled comparison.

    python3 -m training.train_fused --dataset-root dataset_gen/local_dataset_inscale/carve --extra-root dataset_gen/local_dataset_inscale/cyl \
        --checkpoint-dir checkpoints_fused/net_fused --input fused --augment --epochs 40
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training import rotation_symmetry as rs
from training.backbone import Voxel3DBackbone
from training.field_input import CHANNELS, encode
from training.topology_schema import NUM_TOPOLOGIES, TOPOLOGY_TO_IDX
from training.train_common import check_finite, common_arg_parser, load_split, resolve_device, run_training_loop
from training.train_box_dims import REALIZATION_SCALE_MM, half_extents_vox
from training.voxel_dataset import VoxelDataset, load_voxel

SIZE_SLOTS = 3


def size_targets(record, grid_mm: float = 40.0, resolution: int = 64):
    """(log lengths in voxels (3,), mask (3,)) of a record: box sorted half extents, sphere radius, cylinder radius and half length."""
    voxel_mm = grid_mm / resolution
    k = float(record.placement["scale_factor"]) * REALIZATION_SCALE_MM / voxel_mm
    out, mask = np.zeros(SIZE_SLOTS, np.float32), np.zeros(SIZE_SLOTS, np.float32)
    if record.topology == "box":
        out[:] = np.log(half_extents_vox(record, grid_mm, resolution)); mask[:] = 1
    elif record.topology == "sphere":
        out[0] = np.log(record.parameters["radius_ratio"] * k); mask[0] = 1
    elif record.topology == "cylinder":
        out[0], out[1] = np.log(record.parameters["radius_ratio"] * k), np.log(record.parameters["half_length_ratio"] * k); mask[:2] = 1
    return out, mask


class FusedDataset(VoxelDataset):
    """(input tensor (C,R,R,R), type index, box orientation (3,3), symmetry mask (24,), has_rotation, log sizes (3,), size mask (3,))."""

    def __init__(self, records, resolution: int = 64, input_kind: str = "fused", augment: bool = False):
        super().__init__(records, resolution, input_kind=input_kind)
        self.augment = augment
        self.rng = np.random.default_rng()
        self.rot, self.sizes = [], []
        for r in records:
            has_rot = r.topology == "box" and not (r.placement or {}).get("full_field")
            if has_rot:
                M, mask = rs.record_target([r.rotation[k] for k in ("qx", "qy", "qz", "qw")], [r.parameters[k] for k in rs.DIM_KEYS])
            else:
                M, mask = np.eye(3), np.zeros(len(rs.GROUP), bool)
                mask[0] = True                                                  # a valid row so the (unused) loss stays finite
            self.rot.append((M.astype(np.float32), mask, float(has_rot)))
            self.sizes.append(size_targets(r, resolution=resolution))

    def __getitem__(self, idx):
        r = self.records[idx]
        M, mask, has_rot = self.rot[idx]
        volume = load_voxel(r.voxel_path, self.resolution)
        if self.augment:
            g = rs.GROUP[int(self.rng.integers(len(rs.GROUP)))]
            volume = rs.rotate_volume(volume, g)
            M = (g @ M).astype(np.float32)
        x = torch.from_numpy(encode(volume, self.input_kind, self.sdf_truncation))
        log_size, size_mask = self.sizes[idx]
        return (x, TOPOLOGY_TO_IDX[r.topology], torch.from_numpy(M), torch.from_numpy(mask), has_rot,
                torch.from_numpy(log_size), torch.from_numpy(size_mask))


class FusedNet(nn.Module):
    """Shared 3D backbone; heads for type, box pose (6D -> rotation matrix) and size (three log lengths)."""

    def __init__(self, in_channels: int = 5, num_types: int = NUM_TOPOLOGIES, feature_dim: int = 256):
        super().__init__()
        self.backbone = Voxel3DBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.type_head = nn.Linear(feature_dim, num_types)
        self.pose_head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 6))
        self.size_head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, SIZE_SLOTS))
        nn.init.zeros_(self.type_head.bias)
        nn.init.normal_(self.type_head.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.backbone(x)
        return {"type": self.type_head(f), "pose": rs.sixd_to_matrix_torch(self.pose_head(f)), "size": self.size_head(f)}


def losses(out, batch, weights=(1.0, 1.0, 1.0)):
    _, y_type, M, mask, has_rot, log_size, size_mask = batch
    ce = F.cross_entropy(out["type"], y_type)
    has = has_rot > 0.5
    rot = rs.symmetric_loss_torch(out["pose"][has], M[has], mask[has]) if has.any() else out["pose"].sum() * 0.0
    size = (F.l1_loss(out["size"], log_size, reduction="none") * size_mask).sum() / size_mask.sum().clamp(min=1.0)
    return ce, rot, size, weights[0] * ce + weights[1] * rot + weights[2] * size


def _to(batch, device):
    return tuple(b.to(device) if torch.is_tensor(b) else torch.as_tensor(b).to(device) for b in batch)


def train_step(model, batch, device, optimizer):
    batch = _to(batch, device)
    optimizer.zero_grad(set_to_none=True)
    ce, rot, size, loss = losses(model(batch[0]), batch, train_step.weights)
    check_finite(loss, "train_fused train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item(), "ce": ce.item(), "rot_rad": rot.item(), "size_l1": size.item()}


train_step.weights = (1.0, 1.0, 1.0)


def eval_step(model, batch, device):
    batch = _to(batch, device)
    out = model(batch[0])
    ce, rot, size, loss = losses(out, batch, train_step.weights)
    _, y_type, M, mask, has_rot, log_size, size_mask = batch
    acc = (out["type"].argmax(1) == y_type).float().mean().item()
    has = has_rot > 0.5
    metrics = {"loss": loss.item(), "ce": ce.item(), "acc": acc, "size_l1": size.item()}
    if has.any():
        err = rs.symmetric_error(out["pose"][has].double().cpu().numpy(), M[has].double().cpu().numpy(), mask[has].cpu().numpy())
        metrics.update(rot_deg=float(err.mean()), rot_median_deg=float(np.median(err)), rot_within10=float((err < 10).mean()))
    return metrics


def main() -> None:
    parser = common_arg_parser("Train the fused multi-task network (type + box pose + size)")
    parser.add_argument("--input", choices=["occupancy", "sdf", "field", "fused", "voxel_field", "dirlog"], default="dirlog")
    parser.add_argument("--augment", action="store_true", help="random grid rotations (24 signed permutations), the label rotated with it")
    parser.add_argument("--extra-root", type=Path, action="append", default=[], help="more dataset roots, each split train/validation on its own")
    parser.add_argument("--loss-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("TYPE", "POSE", "SIZE"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    train, val = load_split(args.dataset_root, args.val_fraction, args.seed)
    for extra in args.extra_root:
        t, v = load_split(extra, args.val_fraction, args.seed)
        train, val = train + t, val + v
    train, val = [r for r in train if r.topology in ("box", "sphere", "cylinder")], [r for r in val if r.topology in ("box", "sphere", "cylinder")]
    import collections
    count = lambda rs_: dict(collections.Counter(r.topology for r in rs_))                 # noqa: E731
    print(f"train {len(train)} {count(train)}  val {len(val)} {count(val)}  input={args.input} augment={args.augment}", flush=True)
    train_step.weights = tuple(args.loss_weights)
    # seeded per worker: the dataset's numpy rng was forked identically into every worker (all drew the same rotations)
    from training.train_common import loader_kwargs
    train_loader = DataLoader(FusedDataset(train, args.resolution, args.input, args.augment), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True, **loader_kwargs(device, args.num_workers, args.seed))
    val_loader = DataLoader(FusedDataset(val, args.resolution, args.input, False), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = FusedNet(CHANNELS[args.input]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(dict(
        input=args.input, augment=args.augment, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed, loss_weights=args.loss_weights,
        roots=[str(args.dataset_root)] + [str(r) for r in args.extra_root], n_train=len(train), n_val=len(val), train_per_class=count(train),
        val_per_class=count(val), heads="type (CE) + box pose (6D, symmetric geodesic) + size (3 log lengths in voxels, L1)"), indent=1))
    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler, device=device,
                      epochs=args.epochs, patience=args.patience, checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="acc")


if __name__ == "__main__":
    main()
