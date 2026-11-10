#!/usr/bin/env python3
"""
Train the voxel AXIS network for the kinds of revolution (cylinder, cone, torus): voxel grid -> the unit axis of the
primitive, in the voxel ARRAY frame. One network for the three kinds; the kind is known from the Type stage and only
decides the loss (training/train_field_rotation.axis_loss): 1 - cos^2 for cylinder and torus (+axis and -axis are one
pose), 1 - cos for the cone (the apex fixes the direction).

Label: the primitive's local Z (FreeCAD's axis of revolution; the cone's runs from the radius1 base to the radius2 end)
turned into the world by the record's quaternion and into the array frame by rotation_symmetry.A_WORLD_TO_ARRAY. Checked
against the voxels before this trainer was written (docs/data/v5_small_observability.json, scripts/check_observability_v5.py):
on every plain, uncut cylinder, cone and torus of the v5 small set the label axis and the voxel's PCA axis agree to
|cos| = 1.000; on cut and sector bodies PCA fails (cone cut: median |cos| 0.20, torus sector 0.002) - the room for this net.

Augmentation: the 24 grid rotations (exact, the label turns with the grid); vf_exact is rotated channel-wise
(field_channels.rotate_network_input, which also turns the vectors).

Baseline, printed before training and stored in config.json: the PCA axis (the eigenvector with the most distinct
eigenvalue, sign taken from the label for the cone - an optimistic prior) on the validation set.

    python3 -m training.train_axis --dataset-root dataset_gen/local_dataset_v5/small/train \\
        --val-root dataset_gen/local_dataset_v5/small/val --epochs 15 --checkpoint-dir checkpoints/axis_v5_t0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from training import rotation_symmetry as rs
from training.backbone import Voxel3DBackbone
from training.manifest import load_manifest
from training.train_common import (SPEED, add_speed_arguments, autocast, check_finite, common_arg_parser, configure_speed,
                                   load_split, loader_kwargs, prepare_input, resolve_device, run_training_loop)
from training.voxel_dataset import load_voxel

AXIS_KINDS = ("cylinder", "cone", "torus")
SIGNED = {"cylinder": 0.0, "cone": 1.0, "torus": 0.0}      # 1 = the sign of the axis is observable (1 - cos loss)


def label_axis(record) -> np.ndarray:
    """Unit axis of revolution in the array frame [z, -y, x]."""
    q = record.rotation
    world = rs.quat_to_matrix([q["qx"], q["qy"], q["qz"], q["qw"]])[:, 2]
    return (rs.A_WORLD_TO_ARRAY @ world).astype(np.float32)


def pca_axis(volume: np.ndarray) -> np.ndarray:
    pts = np.argwhere(volume > 0).astype(float)
    if len(pts) < 4:
        return np.array([1.0, 0.0, 0.0])
    lam, vec = np.linalg.eigh(np.cov(pts.T, bias=True))
    gaps = [min(abs(lam[i] - lam[j]) for j in range(3) if j != i) for i in range(3)]
    return vec[:, int(np.argmax(gaps))]


def angle_deg(pred: np.ndarray, target: np.ndarray, signed: np.ndarray) -> np.ndarray:
    """Angle between axes; sign-free (min with the flipped axis) where `signed` is 0."""
    cos = (pred * target).sum(-1) / np.maximum(np.linalg.norm(pred, axis=-1) * np.linalg.norm(target, axis=-1), 1e-12)
    cos = np.where(signed > 0, cos, np.abs(cos))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def usable(records, subclasses=None):
    out = [r for r in records if r.topology in AXIS_KINDS and not (r.placement or {}).get("degenerate")]
    if subclasses:
        out = [r for r in out if (r.meta or {}).get("subclass", "full") in subclasses]
    return out


class AxisDataset(Dataset):
    """align=True (occupancy only): each volume is first rotated into its PCA frame (rotation_symmetry.align_volume,
    as train_rotation_sym --align), and the target is the axis in that frame, E^T a - near a grid axis wherever PCA
    already found it, so the net learns the correction (measured: direct regression from the raw grid did not learn at
    276 samples, median 68 deg, against 0.3 deg for the PCA prior on the same validation set)."""

    def __init__(self, records, resolution: int = 64, augment: bool = False, input_kind: str = "occupancy",
                 align: bool = False):
        from training.field_input import FROM_NPZ
        if align and input_kind in FROM_NPZ:
            raise ValueError(f"--align resamples the occupancy grid; {input_kind} is read from the mesh field and cannot be aligned")
        self.records, self.resolution, self.augment, self.input_kind, self.align = records, resolution, augment, input_kind, align
        self.axes = [label_axis(r) for r in records]
        self.signed = [SIGNED[r.topology] for r in records]
        self.aligned, self.frames = [], []
        if align:
            for i, r in enumerate(records):
                vol = np.fromfile(r.voxel_path, dtype=np.uint8).reshape((resolution,) * 3)
                aligned, E = rs.align_volume(vol, "pca")
                self.aligned.append(aligned.astype(np.float32) * 2 - 1)
                self.frames.append(E)
                self.axes[i] = (E.T @ self.axes[i]).astype(np.float32)
        self.rng = np.random.default_rng()

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        axis = self.axes[idx]
        g = rs.GROUP[int(self.rng.integers(len(rs.GROUP)))] if self.augment else None
        if self.input_kind == "vf_exact":
            from training.field_input import load_vf_exact
            x = load_vf_exact(self.records[idx].voxel_path)
            if g is not None:
                root = str(Path(__file__).resolve().parent.parent / "dataset_gen")
                if root not in sys.path:
                    sys.path.insert(0, root)
                from src.field_channels import rotate_network_input
                x = rotate_network_input(x, g)
        else:
            from training.field_input import encode
            x = self.aligned[idx] if self.align else load_voxel(self.records[idx].voxel_path, self.resolution)
            if g is not None:
                x = rs.rotate_volume(x, g)                    # rotate the grid, THEN derive the channels (equivariant)
            x = x.reshape((self.resolution,) * 3)
            if SPEED.gpu_kind is not None:                    # --gpu-encode: ship the uint8 grid, encode on the device
                return (torch.from_numpy(np.ascontiguousarray(x > 0, dtype=np.uint8)), torch.from_numpy(axis),
                        torch.tensor(self.signed[idx], dtype=torch.float32))
            x = encode(x, self.input_kind)
        if g is not None:
            axis = (g @ axis).astype(np.float32)
        return (torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), torch.from_numpy(axis),
                torch.tensor(self.signed[idx], dtype=torch.float32))


class AxisCNN(nn.Module):
    def __init__(self, in_channels: int = 1, feature_dim: int = 256):
        super().__init__()
        self.backbone = Voxel3DBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=1)


def per_sample_loss(pred, target, signed):
    cos = F.cosine_similarity(pred, target, dim=1)
    return torch.where(signed > 0, 1.0 - cos, 1.0 - cos * cos)


def train_step(model, batch, device, optimizer):
    x, axis, signed = batch
    x, axis, signed = prepare_input(x, device), axis.to(device), signed.to(device)
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        pred = model(x)
    loss = per_sample_loss(pred.float(), axis, signed).mean()
    check_finite(loss, "train_axis train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item()}


def eval_step(model, batch, device):
    x, axis, signed = batch
    x, axis, signed = prepare_input(x, device), axis.to(device), signed.to(device)
    with autocast():
        pred = model(x)
    pred = pred.float()
    loss = per_sample_loss(pred, axis, signed).mean()
    err = angle_deg(pred.detach().cpu().numpy(), axis.cpu().numpy(), signed.cpu().numpy())
    return {"loss": loss.item(), "err_deg": float(err.mean()), "median_deg": float(np.median(err)),
            "within5": float((err < 5).mean())}


def pca_baseline(records, resolution: int = 64) -> dict:
    errs = {}
    for r in records:
        v = load_voxel(r.voxel_path, resolution)
        a, t = pca_axis(v.reshape((resolution,) * 3)), label_axis(r)
        if SIGNED[r.topology] and a @ t < 0:            # optimistic: the PCA axis has no sign, give it the right one
            a = -a
        errs.setdefault(r.topology, []).append(float(angle_deg(a[None], t[None], np.array([SIGNED[r.topology]]))[0]))
    allv = [e for v in errs.values() for e in v]
    out = {k: dict(n=len(v), median_deg=float(np.median(v)), within5=float(np.mean(np.asarray(v) < 5))) for k, v in errs.items()}
    out["all"] = dict(n=len(allv), median_deg=float(np.median(allv)), within5=float(np.mean(np.asarray(allv) < 5))) if allv else None
    return out


def main() -> None:
    parser = common_arg_parser("Train the voxel axis network (cylinder, cone, torus)")
    parser.add_argument("--val-root", type=Path, default=None, help="a separate validation tree instead of --val-fraction")
    parser.add_argument("--extra-root", type=Path, action="append", default=[],
                        help="more training roots (with --val-root: used whole; e.g. the nested increments of a learning curve)")
    parser.add_argument("--input", choices=["occupancy", "dirlog", "vf_exact"], default="dirlog",
                        help="network input (training/field_input.py); dirlog (the project standard) and occupancy come from the grid")
    parser.add_argument("--augment", action="store_true", help="random grid rotations (24 signed permutations)")
    parser.add_argument("--align", action="store_true", help="PCA-align every volume first; the net predicts the axis in that frame")
    parser.add_argument("--subclasses", nargs="+", default=None, help="only these subclasses (default: all)")
    add_speed_arguments(parser)
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.val_root:
        train = [r for root in [args.dataset_root, *args.extra_root] for r in load_manifest(root, require_voxel=True)]
        val = load_manifest(args.val_root, require_voxel=True)
    else:
        train, val = load_split(args.dataset_root, args.val_fraction, args.seed)
        for extra in args.extra_root:
            t, v = load_split(extra, args.val_fraction, args.seed)
            train, val = train + t, val + v
    train, val = usable(train, args.subclasses), usable(val, args.subclasses)
    baseline = pca_baseline(val, args.resolution)
    print(f"train: {len(train)}  val: {len(val)}  input={args.input} augment={args.augment} align={args.align}")
    print("PCA baseline on val:", json.dumps(baseline))

    from training.field_input import CHANNELS
    torch.manual_seed(args.seed)
    model = configure_speed(args, args.input, AxisCNN(in_channels=CHANNELS[args.input]).to(device))
    fast = loader_kwargs(device, args.num_workers, args.seed)
    train_loader = DataLoader(AxisDataset(train, args.resolution, args.augment, args.input, args.align), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True, **fast)
    val_loader = DataLoader(AxisDataset(val, args.resolution, False, args.input, args.align), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, **fast)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(dict(
        input=args.input, augment=args.augment, align=args.align, gpu_encode=args.gpu_encode, amp=args.amp, subclasses=args.subclasses, epochs=args.epochs, seed=args.seed,
        n_train=len(train), n_val=len(val), pca_baseline_val=baseline,
        frame="array: [z, -y, x]; label = local Z of the primitive"), indent=1))
    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      scheduler=scheduler, device=device, epochs=args.epochs, patience=args.patience,
                      checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="median_deg")


if __name__ == "__main__":
    main()
