#!/usr/bin/env python3
"""
Train a symmetry-aware Rotation CNN for BOXES on in-scale data (voxel grid -> array-frame orientation).

The old Rotation CNN regressed the raw label quaternion. For a box that label is not determined by the
voxels (a box maps onto itself under several rotations, and the sample's dimensions are independent of its
rotation, so even the axis order is arbitrary) - see training/rotation_symmetry.py. This trainer

  * canonicalises every label: body axes longest-first, expressed in the voxel array's frame (matrix M),
  * minimises the geodesic angle to the NEAREST symmetric equivalent of M (min over the admissible group
    for the sample's dimensions: 4 half-turns; 8 or 24 when two or all dimensions are nearly equal),
  * optionally predicts a 6D continuous representation (Zhou et al. 2019) instead of a quaternion,
  * optionally augments with random grid rotations of the voxel array (24 signed permutations), which
    rotate the label by the same element: exact equivariance for free.

Trained on boxes only (a sphere has no orientation, a full-field cube neither), on the in-scale data of
dataset_gen (--voxel-scale-mode fixed), where the aspect ratio of a box is visible in the grid.

    python3 -m training.train_rotation_sym --dataset-root dataset_gen/local_dataset_inscale/type \
        --checkpoint-dir checkpoints_rot_sym/rotation_cnn --rep 6d --augment --epochs 40
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training import rotation_symmetry as rs
from training.backbone import Voxel3DBackbone
from training.train_common import (
    SPEED, add_speed_arguments, autocast, check_finite, common_arg_parser, configure_speed, load_split, loader_kwargs,
    prepare_input,
    resolve_device, run_training_loop,
)
from training.voxel_dataset import VoxelDataset, load_voxel


def usable_boxes(records):
    """Boxes that have an orientation: not the completely filled grid (a cube with no visible pose)."""
    return [r for r in records if r.topology == "box" and not (r.placement or {}).get("full_field")
            and not (r.placement or {}).get("degenerate")]           # canon: degenerate samples are skipped


class SymRotationDataset(VoxelDataset):
    """align=True: every volume is first rotated so that its PCA axes lie on the grid axes (and its centroid
    at the grid centre), and the network predicts the RESIDUAL orientation E^T M; the final estimate is
    E @ residual. jitter_deg > 0 (training): the aligned volume is additionally rotated by a random angle up
    to that many degrees, standing in for the errors PCA makes on cut or noisy bodies."""

    def __init__(self, records, resolution: int = 64, augment: bool = False, group: str = "sym",
                 align: bool = False, jitter_deg: float = 0.0, align_method: str = "pca", input_kind: str = "occupancy"):
        super().__init__(records, resolution, input_kind=input_kind)
        self.augment, self.align, self.jitter_deg = augment, align, jitter_deg
        self.targets, self.aligned, self.frames = [], [], []
        for r in records:
            M, mask = rs.record_target([r.rotation[k] for k in ("qx", "qy", "qz", "qw")],
                                       rs.record_dims(r))
            if group == "none":                                  # ablation: no symmetry handling, identity only
                mask = np.zeros_like(mask); mask[0] = True
            if align:
                vol = np.fromfile(r.voxel_path, dtype=np.uint8).reshape((resolution,) * 3)
                aligned, E = rs.align_volume(vol, align_method)
                self.aligned.append(aligned)
                self.frames.append(E)
                M = E.T @ M
            self.targets.append((M.astype(np.float32), mask))
        self.rng = np.random.default_rng()

    def __getitem__(self, idx):
        M, mask = self.targets[idx]
        if self.align:
            vol = self.aligned[idx]
            if self.jitter_deg > 0:
                Rj = rs.small_rotation(self.rng, self.jitter_deg)
                vol = rs.resample_rotated(vol, Rj.T)
                M = (Rj @ M).astype(np.float32)
            return self._encode(vol.astype(np.float32) * 2 - 1), torch.from_numpy(M), torch.from_numpy(mask)
        if self.input_kind in ("vf_exact", "dirlog_exact", "dirlog_curv", "voxfield"):
            # read from the .field.npz (it cannot be re-derived from a rotated grid): rotate the channels AND turn the vectors
            from training.field_input import VECTOR_CHANNELS, load_exact_kind
            x = load_exact_kind(self.records[idx].voxel_path, self.input_kind, self.sdf_truncation)
            if self.augment:
                import sys as _sys
                from pathlib import Path as _Path
                _root = str(_Path(__file__).resolve().parent.parent / "dataset_gen")
                if _root not in _sys.path:
                    _sys.path.insert(0, _root)
                from src.field_channels import rotate_network_input
                g = rs.GROUP[int(self.rng.integers(len(rs.GROUP)))]
                x = rotate_network_input(x, g, VECTOR_CHANNELS[self.input_kind])
                M = (g @ M).astype(np.float32)
            return torch.from_numpy(x), torch.from_numpy(M), torch.from_numpy(mask)
        volume = load_voxel(self.records[idx].voxel_path, self.resolution)
        if self.augment:
            g = rs.GROUP[int(self.rng.integers(len(rs.GROUP)))]
            volume = rs.rotate_volume(volume, g)                 # rotate the OCCUPANCY, then derive SDF / field from it (equivariant)
            M = (g @ M).astype(np.float32)
        return self._encode(volume), torch.from_numpy(M), torch.from_numpy(mask)

    def _encode(self, volume_pm1: np.ndarray) -> torch.Tensor:
        if SPEED.gpu_kind is not None:                    # --gpu-encode: the uint8 grid; encoded on the device
            return torch.from_numpy(np.ascontiguousarray(np.asarray(volume_pm1) > 0, dtype=np.uint8))
        from training.field_input import encode
        return torch.from_numpy(encode(volume_pm1, self.input_kind, self.sdf_truncation))


class SymRotationCNN(nn.Module):
    """Voxel grid -> rotation matrix (B, 3, 3): the array-frame orientation of the box's longest-first axes,
    defined up to the box's symmetry group."""

    def __init__(self, rep: str = "6d", feature_dim: int = 256, in_channels: int = 1):
        super().__init__()
        assert rep in ("6d", "quat")
        self.rep = rep
        self.backbone = Voxel3DBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.head = nn.Sequential(nn.Linear(feature_dim, 128), nn.ReLU(inplace=True),
                                  nn.Linear(128, 6 if rep == "6d" else 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(self.backbone(x))
        return rs.sixd_to_matrix_torch(out) if self.rep == "6d" else rs.quat_to_matrix_torch(out)


def _unpack(batch, device):
    return tuple(t.to(device) for t in batch)


def train_step(model, batch, device, optimizer):
    voxels, target, mask = batch
    voxels, target, mask = prepare_input(voxels, device), target.to(device), mask.to(device)
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        pred = model(voxels)
    loss = rs.symmetric_loss_torch(pred.float(), target, mask)
    check_finite(loss, "train_rotation_sym train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item(), "err_deg": float(np.degrees(loss.item()))}


def eval_step(model, batch, device):
    voxels, target, mask = batch
    voxels, target, mask = prepare_input(voxels, device), target.to(device), mask.to(device)
    with autocast():
        pred = model(voxels)
    pred = pred.float()
    loss = rs.symmetric_loss_torch(pred, target, mask)
    err = rs.symmetric_error(pred.double().cpu().numpy(), target.double().cpu().numpy(), mask.cpu().numpy())
    return {"loss": loss.item(), "err_deg": float(err.mean()), "median_deg": float(np.median(err)),
            "within10": float((err < 10).mean())}


def main() -> None:
    parser = common_arg_parser("Train the symmetry-aware box Rotation CNN")
    parser.add_argument("--rep", choices=["6d", "quat"], default="6d")
    parser.add_argument("--augment", action="store_true", help="random grid rotations (24 signed permutations)")
    parser.add_argument("--extra-root", type=Path, action="append", default=[],
                        help="more dataset roots (e.g. box-only in-scale data); each is split 80/20 like --dataset-root")
    parser.add_argument("--val-root", type=Path, default=None,
                        help="a separate validation tree: --dataset-root and every --extra-root are then used whole for training")
    parser.add_argument("--subclasses", nargs="+", default=None,
                        help="only boxes of these subclasses (v5 records; default all - hollow, rhs, frame, ibeam included)")
    parser.add_argument("--align", action="store_true",
                        help="PCA-align every volume first; the network predicts the residual E^T M")
    parser.add_argument("--align-method", choices=["pca", "frame"], default="pca",
                        help="with --align: how the volume is aligned first - pca (second moments) or frame (face normals, "
                             "training/box_alignment.py; robust to cuts and bites). Recorded in config.json.")
    parser.add_argument("--jitter-deg", type=float, default=0.0,
                        help="with --align: random extra rotation of the aligned training volume, up to this many degrees")
    parser.add_argument("--input", choices=["occupancy", "sdf", "field", "fused", "voxel_field", "vf_exact", "dirlog", "dirlog_exact", "dirlog_curv", "voxfield"], default="dirlog",
                        help="network input (training/field_input.py): occupancy, sdf (computed on the fly here), field (vector to the nearest surface point) or fused (all three)")
    parser.add_argument("--group", choices=["sym", "none"], default="sym",
                        help="none = ablation: identity only, i.e. the raw canonical label without symmetry handling")
    add_speed_arguments(parser)
    args = parser.parse_args()
    if args.input in ("vf_exact", "dirlog_exact", "dirlog_curv") and args.align:
        parser.error("--align resamples the occupancy grid by an arbitrary rotation; vf_exact comes from the mesh and is only "
                     "rotated by the 24 grid rotations (--augment) - use one or the other")

    device = resolve_device(args.device)
    if args.val_root:
        from training.manifest import load_manifest
        train = [r for root in [args.dataset_root, *args.extra_root] for r in load_manifest(root, require_voxel=True)]
        val = load_manifest(args.val_root, require_voxel=True)
    else:
        train, val = load_split(args.dataset_root, args.val_fraction, args.seed)
        for extra in args.extra_root:                            # more boxes; each root is split on its own
            t, v = load_split(extra, args.val_fraction, args.seed)
            train, val = train + t, val + v
    train, val = usable_boxes(train), usable_boxes(val)
    if args.subclasses:
        train, val = ([r for r in rs_ if (r.meta or {}).get("subclass", "full") in args.subclasses] for rs_ in (train, val))
    print(f"train boxes: {len(train)}  val boxes: {len(val)}  rep={args.rep} augment={args.augment} group={args.group} align={args.align} jitter={args.jitter_deg}")

    train_loader = DataLoader(SymRotationDataset(train, args.resolution, args.augment, args.group, args.align, args.jitter_deg, args.align_method, args.input),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True,
                              **loader_kwargs(device, args.num_workers, args.seed))
    val_loader = DataLoader(SymRotationDataset(val, args.resolution, False, "sym", args.align, 0.0, args.align_method, args.input),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            **loader_kwargs(device, args.num_workers, args.seed))

    torch.manual_seed(args.seed)
    from training.field_input import CHANNELS
    model = configure_speed(args, args.input, SymRotationCNN(args.rep, in_channels=CHANNELS[args.input]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(
        dict(input=args.input, rep=args.rep, augment=args.augment, group=args.group, align=args.align, align_method=args.align_method, jitter_deg=args.jitter_deg, epochs=args.epochs, seed=args.seed,
             n_train=len(train), n_val=len(val), frame="array: [z, -y, x]; columns = longest-first box axes"), indent=1))

    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      scheduler=scheduler, device=device, epochs=args.epochs, patience=args.patience,
                      checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="err_deg")


if __name__ == "__main__":
    main()
