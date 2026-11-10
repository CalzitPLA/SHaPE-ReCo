#!/usr/bin/env python3
"""
Train ONE topology's dedicated Dimension network for the vector-field route: a random partial patch of the
shape's surface -> its size parameters, directly in real mm (training/field_point_nets.true_size_mm - the
field carries absolute scale, unlike the voxel route's grid-normalized ratios). Log-space L1 (a relative
error) on log(mm), matching training/train_box_dims.py's convention. One network per topology (`--topology`,
run once per shape) - no per-topology masking needed, since each network's own head width already matches
its own topology's parameter count (training/field_point_nets.DIMENSION_KEYS).

Usage:
    python3 -m training.train_field_dimension --topology box \\
        --dataset-root dataset_gen/local_dataset_type --checkpoint-dir checkpoints_field_pertopo/box_dimension_cnn
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from training.field_point_nets import DIMENSION_KEYS, FieldDimensionNet, WHOLE_KINDS, true_size_mm
from training.field_points import TopologyPatchDataset, load_field_split, records_of_topology
from training.train_common import check_finite, common_arg_parser, resolve_device, run_training_loop


def collate(batch):
    x = torch.stack([b["x"] for b in batch])
    records = [b["record"] for b in batch]
    return x, records


def log_size_targets(records, topology):
    return torch.from_numpy(np.stack([np.log(true_size_mm(topology, r.parameters)) for r in records]).astype(np.float32))


def make_steps(topology: str):
    def train_step(model, batch, device, optimizer):
        x, records = batch
        x = x.to(device)
        target = log_size_targets(records, topology).to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.l1_loss(pred, target)
        check_finite(loss, "train_field_dimension train_step")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return {"loss": loss.item()}

    def eval_step(model, batch, device):
        x, records = batch
        x = x.to(device)
        target = log_size_targets(records, topology).to(device)
        pred = model(x)
        loss = F.l1_loss(pred, target)
        rel_err = (torch.exp(pred) / torch.exp(target) - 1.0).abs().mean()
        return {"loss": loss.item(), "rel_err": rel_err.item()}

    return train_step, eval_step


def main() -> None:
    parser = common_arg_parser("Train one topology's dedicated Dimension network (vector-field route)")
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
        topology=args.topology, params=list(DIMENSION_KEYS[args.topology]), n_points=args.n_points,
        roots=[str(args.dataset_root)], epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
        n_train=len(train_records), n_val=len(val_records)), indent=1))

    train_loader = DataLoader(TopologyPatchDataset(train_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, drop_last=True, collate_fn=collate)
    val_loader = DataLoader(TopologyPatchDataset(val_records, args.n_points, seed=args.seed), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    model = FieldDimensionNet(len(DIMENSION_KEYS[args.topology])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_step, eval_step = make_steps(args.topology)
    run_training_loop(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      scheduler=scheduler, device=device, epochs=args.epochs, patience=args.patience,
                      checkpoint_dir=args.checkpoint_dir, train_step=train_step, eval_step=eval_step,
                      extra_metrics_key="rel_err")


if __name__ == "__main__":
    main()
