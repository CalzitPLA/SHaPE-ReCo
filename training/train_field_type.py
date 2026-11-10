#!/usr/bin/env python3
"""
Train the vector-field route's Type network: a point cloud of a whole shape's surface -> topology
classification, over reco_field.fit.WHOLE_KINDS (5-way: sphere, box, cylinder, cone, torus - what
reco_field.fit.fit_best already searches over), not the full 17 topologies the voxel Type CNN classifies.

Reads dataset_gen's plain `type`-mode STLs directly (no --voxelize needed - training/field_points.py
computes each sample's field/point cloud on the fly) so it is directly comparable, on the same 5-class
problem, to an EXISTING voxel Type CNN checkpoint restricted to WHOLE_KINDS (scripts/eval_field_nets.py) -
see the plan at ~/.claude/plans/dazzling-prancing-tower.md, "voxel side keeps its existing networks as-is".

Usage:
    python3 -m training.train_field_type \\
        --dataset-root dataset_gen/local_dataset_type \\
        --checkpoint-dir checkpoints_field_pertopo/type_cnn
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.field_point_nets import FieldTypeNet, WHOLE_KINDS
from training.field_points import TypePointDataset, load_field_split
from training.train_common import check_finite, common_arg_parser, resolve_device, run_training_loop


def train_step(model, batch, device, optimizer):
    x, y = batch
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    check_finite(loss, "train_field_type train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return {"loss": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()}


def eval_step(model, batch, device):
    x, y = batch
    x, y = x.to(device), y.to(device)
    logits = model(x)
    loss = F.cross_entropy(logits, y, label_smoothing=0.1)
    return {"loss": loss.item(), "acc": (logits.argmax(1) == y).float().mean().item()}


def main() -> None:
    parser = common_arg_parser("Train the vector-field route's Type network (point cloud, 5-way over WHOLE_KINDS)")
    parser.add_argument("--n-points", type=int, default=512, help="points per patch after resampling (training/field_point_backbone.encode_patch)")
    parser.add_argument("--extra-root", type=Path, action="append", default=[])
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    train_records, val_records = load_field_split(args.dataset_root, args.val_fraction, args.seed)
    for extra in args.extra_root:
        t, v = load_field_split(extra, args.val_fraction, args.seed)
        train_records, val_records = train_records + t, val_records + v
    train_records = [r for r in train_records if r.topology in WHOLE_KINDS]
    val_records = [r for r in val_records if r.topology in WHOLE_KINDS]
    count = lambda rs_: dict(collections.Counter(r.topology for r in rs_))    # noqa: E731
    print(f"train {len(train_records)} {count(train_records)}  val {len(val_records)} {count(val_records)}", flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(dict(
        input="field", n_points=args.n_points, roots=[str(args.dataset_root)] + list(args.extra_root),
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        n_train=len(train_records), n_val=len(val_records), train_per_class=count(train_records),
        val_per_class=count(val_records)), indent=1))

    train_loader = DataLoader(TypePointDataset(train_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(TypePointDataset(val_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    model = FieldTypeNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      scheduler=scheduler, device=device, epochs=args.epochs, patience=args.patience,
                      checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="acc")


if __name__ == "__main__":
    main()
