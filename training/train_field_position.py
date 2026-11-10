#!/usr/bin/env python3
"""
Train ONE topology's dedicated Position network for the vector-field route: a random partial patch of the
shape's surface -> where the primitive's true centre is, relative to the patch's own point centroid, in
units of the patch's own extent (training/field_point_backbone.normalize_points - the same normalization the
patch tensor itself was built with, so the network reads and predicts in the same units). Meaningless on a
whole, uncropped shape (always centred at the origin - training/field_points.py's own docstring); only
meaningful because training data is partial patches, where a patch's own centroid is generally NOT the
primitive's true centre (e.g. a patch near one end of a cylinder centres far from the cylinder's own middle).
One network per topology (`--topology`, run once per shape).

Usage:
    python3 -m training.train_field_position --topology box \\
        --dataset-root dataset_gen/local_dataset_type --checkpoint-dir checkpoints_field_pertopo/box_position_cnn
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.field_point_nets import FieldPositionNet, WHOLE_KINDS
from training.field_points import TopologyPatchDataset, load_field_split, records_of_topology
from training.train_common import check_finite, common_arg_parser, resolve_device, run_training_loop


def collate(batch):
    x = torch.stack([b["x"] for b in batch])
    offsets = np.stack([(b["true_centre"] - b["patch_centroid"]) / b["patch_extent"] for b in batch]).astype(np.float32)
    return x, torch.from_numpy(offsets)


def train_step(model, batch, device, optimizer):
    x, target = batch
    x, target = x.to(device), target.to(device)
    optimizer.zero_grad(set_to_none=True)
    pred = model(x)
    loss = F.l1_loss(pred, target)
    check_finite(loss, "train_field_position train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item()}


def eval_step(model, batch, device):
    x, target = batch
    x, target = x.to(device), target.to(device)
    pred = model(x)
    loss = F.l1_loss(pred, target)
    err = (pred - target).norm(dim=1).mean()
    return {"loss": loss.item(), "err": err.item()}


def main() -> None:
    parser = common_arg_parser("Train one topology's dedicated Position network (vector-field route)")
    parser.add_argument("--topology", required=True, choices=WHOLE_KINDS)
    parser.add_argument("--n-points", type=int, default=512)
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    train_records, val_records = load_field_split(args.dataset_root, args.val_fraction, args.seed)
    train_records, val_records = records_of_topology(train_records, args.topology), records_of_topology(val_records, args.topology)
    print(f"train {len(train_records)} {args.topology}  val {len(val_records)}", flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(dict(
        topology=args.topology, n_points=args.n_points, roots=[str(args.dataset_root)], epochs=args.epochs,
        lr=args.lr, batch_size=args.batch_size, seed=args.seed, n_train=len(train_records), n_val=len(val_records)), indent=1))

    train_loader = DataLoader(TopologyPatchDataset(train_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(TopologyPatchDataset(val_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = FieldPositionNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      scheduler=scheduler, device=device, epochs=args.epochs, patience=args.patience,
                      checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="err")


if __name__ == "__main__":
    main()
