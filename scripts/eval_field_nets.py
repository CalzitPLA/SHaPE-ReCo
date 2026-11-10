#!/usr/bin/env python3
"""
Held-out evaluation of the vector-field route's point-cloud networks (training/train_field_type.py,
train_field_rotation.py, train_field_dimension.py, train_field_position.py): Type accuracy, and per-topology
rotation error (deg), dimension relative error, position error, all on patches the networks never trained
or selected a model on (training/field_points.load_field_split's own val split).

    python3 scripts/eval_field_nets.py --root checkpoints_field_pertopo --dataset-root dataset_gen/local_dataset_type_pointcnn
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import rotation_symmetry as rs  # noqa: E402
from training.field_point_backbone import encode_patch  # noqa: E402
from training.field_point_nets import (  # noqa: E402
    DIMENSION_KEYS, FieldAxisNet, FieldBoxPoseNet, FieldDimensionNet, FieldPositionNet, FieldTypeNet,
    KIND_TO_IDX, ROTATION_KINDS, WHOLE_KINDS, true_axis_world, true_size_mm,
)
from training.field_points import TopologyPatchDataset, load_field_split, records_of_topology, sample_whole  # noqa: E402


def load_net(path: Path, ctor, *ctor_args):
    net = ctor(*ctor_args)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
    return net.eval()


def eval_type(root: Path, val_records, n_points: int = 512):
    path = root / "type_cnn" / "best_model.pth"
    if not path.is_file():
        print("type_cnn: no checkpoint, skipped")
        return
    net = load_net(path, FieldTypeNet)
    records = [r for r in val_records if r.topology in WHOLE_KINDS]
    correct = 0
    confusion = {}
    with torch.no_grad():
        for r in records:
            points, normals = sample_whole(r.stl_path, n_points, seed=1)
            x = torch.from_numpy(encode_patch(points, normals, n_points, seed=1)).unsqueeze(0)
            pred = WHOLE_KINDS[int(net(x).argmax(1))]
            correct += pred == r.topology
            if pred != r.topology:
                confusion[f"{r.topology}->{pred}"] = confusion.get(f"{r.topology}->{pred}", 0) + 1
    acc = correct / max(len(records), 1)
    print(f"type_cnn: {len(records)} samples, accuracy {100 * acc:5.1f} %  confusions {confusion}")


def eval_topology(root: Path, topology: str, val_records, n_points: int = 512):
    records = records_of_topology(val_records, topology)
    if not records:
        print(f"{topology}: no held-out samples, skipped")
        return
    ds = TopologyPatchDataset(records, n_points, seed=1)

    dim_path = root / f"{topology}_dimension_cnn" / "best_model.pth"
    if dim_path.is_file():
        net = load_net(dim_path, FieldDimensionNet, len(DIMENSION_KEYS[topology]))
        errs = []
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                pred = net(item["x"].unsqueeze(0))[0].numpy()
                target = np.log(true_size_mm(topology, item["record"].parameters))
                errs.append(np.abs(np.exp(pred) / np.exp(target) - 1.0).mean())
        print(f"{topology}_dimension_cnn: {len(records)} samples, mean relative error {100 * np.mean(errs):5.1f} %")

    pos_path = root / f"{topology}_position_cnn" / "best_model.pth"
    if pos_path.is_file():
        net = load_net(pos_path, FieldPositionNet)
        errs = []
        with torch.no_grad():
            for i in range(len(ds)):
                item = ds[i]
                pred = net(item["x"].unsqueeze(0))[0].numpy()
                target = (item["true_centre"] - item["patch_centroid"]) / item["patch_extent"]
                errs.append(np.linalg.norm(pred - target))
        print(f"{topology}_position_cnn: {len(records)} samples, mean error {np.mean(errs):.3f} (patch-extent units)")

    if topology not in ROTATION_KINDS:
        return
    rot_path = root / f"{topology}_rotation_cnn" / "best_model.pth"
    if not rot_path.is_file():
        return
    is_box = topology == "box"
    net = load_net(rot_path, FieldBoxPoseNet if is_box else FieldAxisNet)
    errs = []
    with torch.no_grad():
        for i in range(len(ds)):
            item = ds[i]
            pred = net(item["x"].unsqueeze(0))[0].double().numpy()
            r = item["record"]
            if is_box:
                target, mask = rs.record_target([r.rotation[k] for k in ("qx", "qy", "qz", "qw")],
                                                [r.parameters[k] for k in rs.DIM_KEYS])
                errs.append(float(rs.symmetric_error(pred[None], target[None], mask[None])[0]))
            else:
                target = true_axis_world([r.rotation[k] for k in ("qx", "qy", "qz", "qw")])
                cos = np.clip(np.dot(pred, target), -1.0, 1.0)
                err_deg = np.degrees(np.arccos(cos))
                errs.append(min(err_deg, 180.0 - err_deg))
    print(f"{topology}_rotation_cnn: {len(records)} samples, median {np.median(errs):5.1f} deg, "
         f"within10 {100 * np.mean(np.array(errs) < 10):5.1f} %")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT / "checkpoints_field_pertopo")
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-points", type=int, default=512)
    args = ap.parse_args()

    _train, val_records = load_field_split(args.dataset_root, args.val_fraction, args.seed)
    eval_type(args.root, val_records, args.n_points)
    for topology in WHOLE_KINDS:
        eval_topology(args.root, topology, val_records, args.n_points)


if __name__ == "__main__":
    main()
