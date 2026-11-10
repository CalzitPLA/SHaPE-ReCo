#!/usr/bin/env python3
"""
Train ONE topology's dedicated Rotation network for the vector-field route: a random partial patch of the
shape's surface (training/field_points.sample_patch - never a whole clean shape, see field_points.py's own
docstring) -> its pose. `box` gets a full 3x3 rotation (6D representation, the SAME symmetric-geodesic loss
training/train_fused.py already uses, training/rotation_symmetry.py, unmodified); `cylinder`/`cone`/`torus`
get only their axis of revolution (a unit vector; loss `axis_loss`: 1 - cos^2 for the sign-free cylinder/torus
axis, 1 - cos for the cone - a full rotation is unobservable, and meaningless, about that axis). `sphere` has no
rotation network at all (rotation-invariant) - refused here rather than silently training something
meaningless.

One network per topology (`--topology`, run once per shape you want a checkpoint for) - no topology
embedding anywhere, unlike the original voxel train_rotation_cnn.py: each network only ever sees its own
shape.

Usage:
    python3 -m training.train_field_rotation --topology box \\
        --dataset-root dataset_gen/local_dataset_type --checkpoint-dir checkpoints_field_pertopo/box_rotation_cnn
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training import rotation_symmetry as rs
from training.field_point_nets import AXIS_KINDS, FieldAxisNet, FieldBoxPoseNet, ROTATION_KINDS, true_axis_world
from training.field_points import TopologyPatchDataset, load_field_split, records_of_topology
from training.train_common import check_finite, common_arg_parser, resolve_device, run_training_loop


def collate(batch):
    x = torch.stack([b["x"] for b in batch])
    records = [b["record"] for b in batch]
    return x, records


def box_targets(records):
    Ms, masks = [], []
    for r in records:
        M, mask = rs.record_target([r.rotation[k] for k in ("qx", "qy", "qz", "qw")],
                                   [r.parameters[k] for k in rs.DIM_KEYS])
        Ms.append(M.astype(np.float32)); masks.append(mask)
    return torch.from_numpy(np.stack(Ms)), torch.from_numpy(np.stack(masks))


def axis_targets(records):
    axes = [true_axis_world([r.rotation[k] for k in ("qx", "qy", "qz", "qw")]).astype(np.float32) for r in records]
    return torch.from_numpy(np.stack(axes))


SIGNLESS_AXIS_KINDS = ("cylinder", "torus")     # mirror-symmetric across the plane normal to the axis: +axis and -axis are one pose


def axis_loss(pred, target, topology: str):
    """Mean axis loss of a batch. cylinder / torus: 1 - cos^2 - their shape is the same for +axis and -axis, and
    `true_axis_world` does not canonicalise the sign, so the plain 1 - cos of the first version asked the network to
    tell apart two identical inputs (half the targets pointed "the other way"; the best it can do then is shrink the
    prediction towards the mean - a likely part of field_pertopo's 42-47 deg axis errors, not yet re-measured). cone: 1 - cos - the
    apex fixes the direction (LOCAL_AXIS runs from the radius1 base towards the radius2 end), so the sign is observable
    and must be learned. Both are 0 exactly at a correct axis; 1 - cos^2 is also 0 at the flipped one."""
    cos = F.cosine_similarity(pred, target, dim=1)
    if topology in SIGNLESS_AXIS_KINDS:
        return (1.0 - cos * cos).mean()
    return (1.0 - cos).mean()


def make_steps(topology: str):
    is_box = topology == "box"

    def train_step(model, batch, device, optimizer):
        x, records = batch
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        if is_box:
            target, mask = box_targets(records)
            loss = rs.symmetric_loss_torch(pred, target.to(device), mask.to(device))
        else:
            target = axis_targets(records).to(device)
            loss = axis_loss(pred, target, topology)
        check_finite(loss, "train_field_rotation train_step")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {"loss": loss.item()}

    def eval_step(model, batch, device):
        x, records = batch
        x = x.to(device)
        pred = model(x)
        if is_box:
            target, mask = box_targets(records)
            target, mask = target.to(device), mask.to(device)
            loss = rs.symmetric_loss_torch(pred, target, mask)
            err = rs.symmetric_error(pred.double().cpu().numpy(), target.double().cpu().numpy(), mask.cpu().numpy())
            return {"loss": loss.item(), "err_deg": float(err.mean())}
        target = axis_targets(records).to(device)
        loss = axis_loss(pred, target, topology)
        err_deg = np.degrees(np.arccos(np.clip((pred * target).sum(1).double().cpu().numpy(), -1.0, 1.0)))
        if topology in SIGNLESS_AXIS_KINDS:
            err_deg = np.minimum(err_deg, 180.0 - err_deg)                    # +-axis are the same pose
        # a cone keeps the signed error: pointing the apex the wrong way IS a 180 deg error there. The old metric folded it
        # for every kind (while the old loss folded it for none) - loss and metric now agree per kind.
        return {"loss": loss.item(), "err_deg": float(err_deg.mean())}

    return train_step, eval_step


def main() -> None:
    parser = common_arg_parser("Train one topology's dedicated Rotation network (vector-field route)")
    parser.add_argument("--topology", required=True, choices=ROTATION_KINDS,
                        help="sphere has no rotation network (rotation-invariant) - not a valid choice here")
    parser.add_argument("--n-points", type=int, default=512)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    train_records, val_records = load_field_split(args.dataset_root, args.val_fraction, args.seed)
    train_records, val_records = records_of_topology(train_records, args.topology), records_of_topology(val_records, args.topology)
    print(f"train {len(train_records)} {args.topology}  val {len(val_records)}", flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(dict(
        topology=args.topology, kind="box_pose" if args.topology == "box" else "axis", n_points=args.n_points,
        roots=[str(args.dataset_root)], epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        n_train=len(train_records), n_val=len(val_records)), indent=1))

    train_loader = DataLoader(TopologyPatchDataset(train_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(TopologyPatchDataset(val_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = (FieldBoxPoseNet() if args.topology == "box" else FieldAxisNet()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_step, eval_step = make_steps(args.topology)
    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      scheduler=scheduler, device=device, epochs=args.epochs, patience=args.patience,
                      checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="err_deg")


if __name__ == "__main__":
    main()
