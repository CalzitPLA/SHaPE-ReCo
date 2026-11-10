#!/usr/bin/env python3
"""
Train the Type CNN: voxel grid -> topology classification (17-way), or with --hierarchical the two-head TypeNet:
5 parents (box, cylinder, cone, sphere, torus) + 24 subclasses masked to the parent (training/subclass_schema.py).

Replaces FeatureNet/ShaPeNet_type.py's ad hoc 24-class trainer (which
targeted a stale, unrelated 24-class label space left over from an
external "Machining-feature-dataset") with a trainer that classifies
exactly the 17 topologies dataset_gen actually produces.

Hierarchical mode (--hierarchical):
    one backbone, head 1 = parent logits (5), head 2 = subclass logits (24, flat order of subclass_schema).
    loss = CE(parent, label smoothing 0.1) + --sub-weight * CE(subclass logits with every subclass of the OTHER
    parents set to -inf, i.e. a softmax over the true parent's subclasses only).
    HierTypeCNN.forward returns the parent logits scattered into the 17-way TOPOLOGY_TO_IDX space (-inf for the 12
    non-parent topologies), so every existing consumer of a TypeCNN (reco_engine.predict, the cascade, the eval
    scripts) reads a hierarchical checkpoint as an ordinary type net; `heads(x)` gives both heads.

Other options: --oversample key=value[:N] (repeat matching TRAIN records N times in the sampler), --val-roots (a
separate validation tree instead of the --val-fraction split), --test-roots (after training, best_model.pth is scored
on each root and model_card.json is written next to it).

Usage:
    python3 -m training.train_type_cnn \\
        --dataset-root dataset_gen/test_output_type \\
        --checkpoint-dir checkpoints/type_cnn

    python3 -m training.train_type_cnn --hierarchical --input vf_exact \\
        --dataset-root dataset_gen/local_dataset_v5/small/train --val-roots dataset_gen/local_dataset_v5/small/val \\
        --test-roots dataset_gen/local_dataset_v5/small/test --epochs 15 --checkpoint-dir checkpoints/type_hier_vf/type_cnn
"""

from __future__ import annotations

import collections
import functools
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.backbone import Voxel3DBackbone
from training.topology_schema import NUM_TOPOLOGIES, TOPOLOGY_TO_IDX, IDX_TO_TOPOLOGY
from training.subclass_schema import (
    FLAT_SUBCLASSES, NUM_PARENTS, NUM_SUBCLASSES, PARENT_MASK, PARENTS, SUB_PARENT, SUBCLASSES,
    filter_records, labels_of, subclass_name,
)
from training.train_common import (
    SPEED, add_speed_arguments, autocast, check_finite, common_arg_parser, configure_speed, load_split, loader_kwargs,
    prepare_input, resolve_device, run_training_loop,
)
from training.voxel_dataset import VoxelDataset

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
PARENT_TOPOLOGY_IDX = [TOPOLOGY_TO_IDX[p] for p in PARENTS]


def _input(dataset: VoxelDataset, idx: int):
    """(input tensor, record): the encoded input (VoxelDataset._load), or under --gpu-encode the raw uint8 grid -
    (2,R,R,R) = [grid, inside bits] for dirlog_inside - encoded on the device by train_common.prepare_input."""
    if SPEED.gpu_kind is None:
        return dataset._load(idx)
    from training.gpu_encode import load_raw
    record = dataset.records[idx]
    grid, inside = load_raw(record.voxel_path, dataset.resolution, with_inside=SPEED.gpu_kind == "dirlog_inside")
    x = np.stack([grid, inside]) if inside is not None else grid
    return torch.from_numpy(np.ascontiguousarray(x)), record


class TypeDataset(VoxelDataset):
    def __getitem__(self, idx):
        tensor, record = _input(self, idx)
        label = TOPOLOGY_TO_IDX[record.topology]
        return tensor, torch.tensor(label, dtype=torch.long)


class HierTypeDataset(VoxelDataset):
    """(input, parent index into PARENTS, flat subclass index); records must pass subclass_schema.filter_records."""

    def __getitem__(self, idx):
        tensor, record = _input(self, idx)
        parent, sub = labels_of(record)
        return tensor, torch.tensor(parent, dtype=torch.long), torch.tensor(sub, dtype=torch.long)


class TypeCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_TOPOLOGIES, feature_dim: int = 256, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = Voxel3DBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.zeros_(self.classifier.bias)
        nn.init.normal_(self.classifier.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))


class HierTypeCNN(nn.Module):
    """Two heads on one backbone. `heads(x)` -> (parent logits (N, 5), subclass logits (N, 24)); `forward(x)` -> the
    parent logits in the 17-way topology space (-inf elsewhere), the TypeCNN interface."""

    hierarchical = True

    def __init__(self, feature_dim: int = 256, in_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.backbone = Voxel3DBackbone(in_channels=in_channels, feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, NUM_PARENTS)
        self.sub_classifier = nn.Linear(feature_dim, NUM_SUBCLASSES)
        for head in (self.classifier, self.sub_classifier):
            nn.init.zeros_(head.bias)
            nn.init.normal_(head.weight, std=0.01)
        self.register_buffer("parent_topology_idx", torch.tensor(PARENT_TOPOLOGY_IDX), persistent=False)

    def heads(self, x: torch.Tensor):
        features = self.backbone(x)
        return self.classifier(features), self.sub_classifier(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parent_logits, _ = self.heads(x)
        return expand_parent_logits(parent_logits, self.parent_topology_idx)


def expand_parent_logits(parent_logits: torch.Tensor, index=None) -> torch.Tensor:
    """(N, 5) parent logits -> (N, NUM_TOPOLOGIES) with -inf on the non-parent topologies."""
    index = torch.as_tensor(PARENT_TOPOLOGY_IDX if index is None else index, device=parent_logits.device)
    out = torch.full((parent_logits.shape[0], NUM_TOPOLOGIES), float("-inf"),
                     dtype=parent_logits.dtype, device=parent_logits.device)
    return out.index_copy(1, index, parent_logits)


def is_hierarchical_state(state_dict) -> bool:
    return "sub_classifier.weight" in state_dict


def type_cnn_for_state(state_dict, in_channels: int) -> nn.Module:
    """An empty TypeCNN or HierTypeCNN matching a checkpoint's state dict."""
    return HierTypeCNN(in_channels=in_channels) if is_hierarchical_state(state_dict) else TypeCNN(in_channels=in_channels)


def masked_subclass_logits(sub_logits: torch.Tensor, parent_idx: torch.Tensor) -> torch.Tensor:
    """Subclass logits with every subclass NOT belonging to `parent_idx` (per row) set to -inf."""
    mask = torch.as_tensor(PARENT_MASK, device=sub_logits.device)[parent_idx]          # (N, 24) bool
    return sub_logits.masked_fill(~mask, float("-inf"))


def hierarchical_loss(parent_logits, sub_logits, parent, sub, sub_weight: float = 1.0):
    """(total, parent CE, masked subclass CE): CE(parent, smoothing 0.1) + sub_weight * CE(masked to the TRUE parent)."""
    parent_loss = nn.functional.cross_entropy(parent_logits, parent, label_smoothing=0.1)
    sub_loss = nn.functional.cross_entropy(masked_subclass_logits(sub_logits, parent), sub)
    return parent_loss + sub_weight * sub_loss, parent_loss, sub_loss


def train_step(model, batch, device, optimizer):
    voxels, labels = batch
    voxels, labels = prepare_input(voxels, device), labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        logits = model(voxels)
    logits = logits.float()
    loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
    check_finite(loss, "train_type_cnn train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return {"loss": loss.item(), "acc": acc.item()}


def eval_step(model, batch, device):
    voxels, labels = batch
    voxels, labels = prepare_input(voxels, device), labels.to(device)
    with autocast():
        logits = model(voxels)
    logits = logits.float()
    loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.1)
    acc = (logits.argmax(dim=1) == labels).float().mean()
    return {"loss": loss.item(), "acc": acc.item()}


def _hier_batch_metrics(parent_logits, sub_logits, parent, sub) -> dict:
    parent_pred = parent_logits.argmax(1)
    top2 = (parent_logits.topk(min(2, parent_logits.shape[1]), dim=1).indices == parent[:, None]).any(1)
    sub_true_parent = masked_subclass_logits(sub_logits, parent).argmax(1)
    sub_pred_parent = masked_subclass_logits(sub_logits, parent_pred).argmax(1)
    return {"acc": (parent_pred == parent).float().mean().item(), "top2": top2.float().mean().item(),
            "sub_acc": (sub_true_parent == sub).float().mean().item(),
            "sub_acc_pred_parent": (sub_pred_parent == sub).float().mean().item()}


def hier_train_step(model, batch, device, optimizer, sub_weight: float = 1.0):
    voxels, parent, sub = batch
    voxels, parent, sub = prepare_input(voxels, device), parent.to(device), sub.to(device)
    optimizer.zero_grad(set_to_none=True)
    with autocast():
        parent_logits, sub_logits = model.heads(voxels)
    parent_logits, sub_logits = parent_logits.float(), sub_logits.float()
    loss, parent_loss, sub_loss = hierarchical_loss(parent_logits, sub_logits, parent, sub, sub_weight)
    check_finite(loss, "train_type_cnn hier_train_step")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    with torch.no_grad():
        return {"loss": loss.item(), "parent_loss": parent_loss.item(), "sub_loss": sub_loss.item(),
                **_hier_batch_metrics(parent_logits, sub_logits, parent, sub)}


def hier_eval_step(model, batch, device, sub_weight: float = 1.0):
    voxels, parent, sub = batch
    voxels, parent, sub = prepare_input(voxels, device), parent.to(device), sub.to(device)
    with autocast():
        parent_logits, sub_logits = model.heads(voxels)
    parent_logits, sub_logits = parent_logits.float(), sub_logits.float()
    loss, parent_loss, sub_loss = hierarchical_loss(parent_logits, sub_logits, parent, sub, sub_weight)
    return {"loss": loss.item(), "parent_loss": parent_loss.item(), "sub_loss": sub_loss.item(),
            **_hier_batch_metrics(parent_logits, sub_logits, parent, sub)}


# ------------------------------------------------------------------------------------------------ evaluation

@torch.no_grad()
def predict_records(model, records, device, batch_size: int = 16, num_workers: int = 0, **dataset_kwargs) -> dict:
    """Run a (Hier)TypeCNN over `records`: {"logits": (N, 17)} and, for a hierarchical model, "parent_logits"
    (N, 5) and "sub_logits" (N, 24). `dataset_kwargs` = VoxelDataset's (resolution, input_kind, ...)."""
    model.eval()
    hier = getattr(model, "hierarchical", False)
    loader = DataLoader(TypeDataset(records, **dataset_kwargs) if not hier else _PlainDataset(records, **dataset_kwargs),
                        batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = collections.defaultdict(list)
    for batch in loader:
        x = prepare_input(batch[0], device)            # --gpu-encode ships raw uint8 grids: encode them here as train_step does
        with autocast():
            if hier:
                p, s = model.heads(x)
                out["parent_logits"].append(p.float().cpu())
                out["sub_logits"].append(s.float().cpu())
                out["logits"].append(expand_parent_logits(p).float().cpu())
            else:
                out["logits"].append(model(x).float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


class _PlainDataset(VoxelDataset):
    def __getitem__(self, idx):
        return (_input(self, idx)[0],)          # raw uint8 grid under --gpu-encode (prepare_input encodes), the CPU encoding otherwise


def _mask_np(sub_logits: np.ndarray, parent_idx: np.ndarray) -> np.ndarray:
    return np.where(PARENT_MASK[parent_idx], sub_logits, -np.inf)


def hierarchical_metrics(parent_logits: np.ndarray, sub_logits: np.ndarray, parent_y: np.ndarray, sub_y: np.ndarray) -> dict:
    """Parent acc / top-2, subclass acc within the TRUE parent, subclass acc through the PREDICTED parent (end to end:
    right parent and right subclass), per-subclass rates and the two confusion tables the plan asks for:
    box/ibeam vs box/full, and sector/band vs full per parent. Predicted labels are end-to-end flat names."""
    n = len(parent_y)
    if n == 0:
        return {"n": 0}
    parent_pred = parent_logits.argmax(1)
    top2 = (np.argsort(-parent_logits, 1)[:, :2] == parent_y[:, None]).any(1)
    sub_true = _mask_np(sub_logits, parent_y).argmax(1)
    sub_e2e = _mask_np(sub_logits, parent_pred).argmax(1)
    ok_parent, ok_sub, ok_e2e = parent_pred == parent_y, sub_true == sub_y, sub_e2e == sub_y
    per_sub = {}
    for s in sorted(set(sub_y.tolist())):
        m = sub_y == s
        per_sub[FLAT_SUBCLASSES[s]] = dict(n=int(m.sum()), parent_acc=float(ok_parent[m].mean()),
                                           sub_acc=float(ok_sub[m].mean()), end_to_end_acc=float(ok_e2e[m].mean()))
    per_parent = {}
    for p in sorted(set(parent_y.tolist())):
        m = parent_y == p
        per_parent[PARENTS[p]] = dict(n=int(m.sum()), acc=float(ok_parent[m].mean()), sub_acc=float(ok_sub[m].mean()))

    def confusion(true_names):
        idx = [FLAT_SUBCLASSES.index(t) for t in true_names]
        table = {}
        for t in idx:
            m = sub_y == t
            if m.any():
                table[FLAT_SUBCLASSES[t]] = dict(collections.Counter(FLAT_SUBCLASSES[k] for k in sub_e2e[m].tolist()))
        return table

    segment_tables = {}
    for parent in PARENTS:
        names = [f"{parent}/{s}" for s in ("full", "sector", "band") if s in SUBCLASSES[parent]]
        if len(names) > 1:
            table = confusion(names)
            if table:
                segment_tables[parent] = table
    parent_confusion = {f"{PARENTS[a]}->{PARENTS[b]}": c
                        for (a, b), c in collections.Counter(zip(parent_y.tolist(), parent_pred.tolist())).items() if a != b}
    return dict(n=int(n), parent_acc=float(ok_parent.mean()), parent_top2=float(top2.mean()),
                sub_acc=float(ok_sub.mean()), sub_acc_pred_parent=float(ok_e2e.mean()),
                chance_parent=1.0 / NUM_PARENTS,
                chance_sub=float(np.mean([1.0 / PARENT_MASK[p].sum() for p in parent_y])),
                per_parent=per_parent, per_subclass=per_sub, parent_confusion=parent_confusion,
                confusion_ibeam_vs_box=confusion(["box/full", "box/ibeam"]),
                confusion_segment_vs_full=segment_tables)


def flat_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    """17-way topology accuracy (the non-hierarchical TypeCNN)."""
    if len(y) == 0:
        return {"n": 0}
    pred = logits.argmax(1)
    confusion = {f"{IDX_TO_TOPOLOGY[a]}->{IDX_TO_TOPOLOGY[b]}": c
                 for (a, b), c in collections.Counter(zip(y.tolist(), pred.tolist())).items() if a != b}
    return dict(n=int(len(y)), acc=float((pred == y).mean()),
                top2=float((np.argsort(-logits, 1)[:, :2] == y[:, None]).any(1).mean()), confusion=confusion)


def evaluate_records(model, records, device, batch_size: int = 16, **dataset_kwargs) -> dict:
    """Metrics of `model` on `records` (hierarchical_metrics or flat_metrics). Hierarchical: records outside the
    label space are skipped and counted under "skipped"."""
    if getattr(model, "hierarchical", False):
        records, skipped = filter_records(records)
        out = predict_records(model, records, device, batch_size, **dataset_kwargs) if records else {}
        labels = np.array([labels_of(r) for r in records], dtype=np.int64).reshape(-1, 2)
        metrics = hierarchical_metrics(out.get("parent_logits", np.zeros((0, NUM_PARENTS))),
                                       out.get("sub_logits", np.zeros((0, NUM_SUBCLASSES))), labels[:, 0], labels[:, 1])
        metrics["skipped"] = dict(skipped)
        return metrics
    records = [r for r in records if r.topology in TOPOLOGY_TO_IDX]
    out = predict_records(model, records, device, batch_size, **dataset_kwargs) if records else {"logits": np.zeros((0, NUM_TOPOLOGIES))}
    return flat_metrics(out["logits"], np.array([TOPOLOGY_TO_IDX[r.topology] for r in records], dtype=np.int64))


# ------------------------------------------------------------------------------------------------ oversampling

_TRUE, _FALSE = ("1", "true", "yes"), ("0", "false", "no")


def parse_oversample(specs, default_factor: int = 2):
    """["subclass=hollow,cut=1:3", ...] -> [(key, value, factor), ...]. `:N` sets that term's factor."""
    terms = []
    for spec in specs or []:
        for term in spec.split(","):
            term = term.strip()
            if not term:
                continue
            factor = default_factor
            if ":" in term:
                term, f = term.rsplit(":", 1)
                factor = int(f)
            if "=" not in term or factor < 1:
                raise ValueError(f"--oversample term must be key=value[:N] with N >= 1, got {term!r}")
            key, value = term.split("=", 1)
            terms.append((key.strip(), value.strip(), factor))
    return terms


def record_field(record, key):
    """A record's value for an --oversample key: subclass / parent / flat (derived), else the JSONL record's own
    top-level key (Record.meta), else its placement's (`cut` also matches placement["cuts"])."""
    if key == "subclass":
        return subclass_name(record)
    if key in ("parent", "topology"):
        return record.topology
    if key == "flat":
        return f"{record.topology}/{subclass_name(record)}"
    meta = getattr(record, "meta", None) or {}
    if key in meta:
        return meta[key]
    placement = record.placement or {}
    if key in placement:
        return placement[key]
    return placement.get(key + "s")


def _matches(field, value: str) -> bool:
    if value.lower() in _TRUE:
        return bool(field) and str(field).lower() not in _FALSE
    if value.lower() in _FALSE:
        return not field or str(field).lower() in _FALSE
    return field is not None and str(field) == value


def oversample_factors(records, terms):
    """Per-record repeat factor: the largest factor among the terms it matches, 1 if none."""
    return [max([f for k, v, f in terms if _matches(record_field(r, k), v)] or [1]) for r in records]


def oversampled_indices(records, terms):
    return [i for i, f in enumerate(oversample_factors(records, terms)) for _ in range(f)]


# ------------------------------------------------------------------------------------------------ model card

def git_commit() -> dict:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SANDBOX_ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=SANDBOX_ROOT,
                                    capture_output=True, text=True, timeout=10).stdout.strip())
        return dict(commit=head or None, dirty=dirty)
    except (OSError, subprocess.SubprocessError):
        return dict(commit=None, dirty=None)


def label_counts(records, hierarchical: bool) -> dict:
    if hierarchical:
        kept, _ = filter_records(records)
        return dict(sorted(collections.Counter(FLAT_SUBCLASSES[labels_of(r)[1]] for r in kept).items()))
    return dict(sorted(collections.Counter(r.topology for r in records).items()))


def load_checkpoint_model(checkpoint_path: Path, in_channels: int, device) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)["model_state_dict"]
    model = type_cnn_for_state(state, in_channels)
    model.load_state_dict(state)
    return model.to(device).eval()


# ------------------------------------------------------------------------------------------------ main

def main(argv=None) -> None:
    parser = common_arg_parser("Train the shape-type classification CNN")
    parser.add_argument("--input", choices=["occupancy", "sdf", "field", "fused", "voxel_field", "exact_field", "vf_exact", "dirlog", "dirlog_exact", "dirlog_curv", "dirlog_inside", "voxfield"], default="dirlog_inside",
                        help="network input: the binary occupancy grid ({-1,+1}), its signed distance field (training/sdf_input.py), the vector to the "
                             "nearest surface point derived from the grid (field, 3 channels), all three fused (5 channels; training/field_input.py), "
                             "or the MESH's own exact closest-surface-point field (exact_field, 3 channels, training/exact_field_input.py), "
                             "or vf_exact (the fused layout from the exact mesh field in each sample's .field.npz, 5 channels; "
                             "needs a dataset generated with --with-field), or voxfield (the voxel grid + the exact surface field "
                             "in one 6-channel tensor, training/inputs.py; computed once per sample into a sparse cache)")
    parser.add_argument("--voxfield-allow-fallback", action="store_true",
                        help="--input voxfield: train even when some samples have no .field.npz (they get the voxel-only "
                             "fallback, training/inputs.py). Default: refuse - a class-correlated field coverage is a shortcut")
    parser.add_argument("--sdf-truncation", type=float, default=8.0, help="voxels; the SDF is clipped there and scaled to [-1, 1]")
    parser.add_argument("--sdf-cache", type=Path, default=Path("sdf_cache"), help="where the computed SDF inputs are kept")
    parser.add_argument("--field-resolution", type=int, default=None,
                        help="--input exact_field: resolution of the exact-field grid, independent of --resolution (the occupancy "
                             "grid's). Default: Nyquist, 2x --resolution (training/exact_field_input.field_resolution_for) - the "
                             "field is continuous, so anything sampled at less than 2x the occupancy grid's rate cannot resolve "
                             "what that grid already resolves.")
    parser.add_argument("--field-cache", type=Path, default=Path("field_cache"), help="--input exact_field: where the computed exact fields are kept")
    parser.add_argument("--extra-root", type=Path, action="append", default=[],
                        help="more dataset roots (e.g. an earlier in-scale set next to a new one); each root is split "
                             "train/validation on its own with --val-fraction, then the parts are pooled")
    parser.add_argument("--topologies", nargs="+", default=None,
                        help="keep only these topologies in train/val/test (e.g. box sphere); the head keeps all classes")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="cap each topology at this many TRAIN samples (val is capped at max(1, this * val_fraction "
                             "/ (1 - val_fraction)), so the val:train ratio stays close to --val-fraction) - a bounded, "
                             "reproducible (--seed) random subsample of the pooled records, taken AFTER --extra-root "
                             "pooling. For a quick or expensive-input (--input exact_field) run on a big pooled root "
                             "without waiting for the whole thing. Default: no cap, use everything.")
    parser.add_argument("--hierarchical", action="store_true",
                        help="two heads: 5 parents + 24 subclasses masked to the true parent (training/subclass_schema.py); "
                             "records of other topologies are skipped with a count")
    parser.add_argument("--sub-weight", type=float, default=1.0, help="--hierarchical: weight of the masked subclass CE")
    parser.add_argument("--oversample", action="append", default=[], metavar="KEY=VALUE[:N][,...]",
                        help="repeat TRAIN records whose field matches N times per epoch (default N = --oversample-factor), e.g. "
                             "subclass=hollow, flat=box/ibeam:4, cut=1 (a placement stratum; 1/0 = present/absent). Keys: subclass, "
                             "parent, flat, any top-level manifest key, any placement key")
    parser.add_argument("--oversample-factor", type=int, default=2)
    parser.add_argument("--val-roots", type=Path, nargs="+", default=[],
                        help="validation roots: when given, --dataset-root/--extra-root are used WHOLE for training and these for "
                             "validation (e.g. local_dataset_v5/small/{train,val}) instead of the --val-fraction split")
    parser.add_argument("--test-roots", type=Path, nargs="+", default=[],
                        help="after training, score best_model.pth on each root and write model_card.json")
    add_speed_arguments(parser)
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    roots = [args.dataset_root] + list(args.extra_root)
    if args.val_roots:
        from training.manifest import load_manifest
        train_records = [r for root in roots for r in load_manifest(root, require_voxel=True)]
        val_records = [r for root in args.val_roots for r in load_manifest(root, require_voxel=True)]
        if not train_records or not val_records:
            raise RuntimeError(f"no voxelized samples: {len(train_records)} train under {roots}, {len(val_records)} val under {args.val_roots}")
    else:
        train_records, val_records = load_split(args.dataset_root, args.val_fraction, args.seed)
        for extra in args.extra_root:
            t, v = load_split(extra, args.val_fraction, args.seed)
            train_records, val_records = train_records + t, val_records + v
    if args.hierarchical:
        train_records, skipped_train = filter_records(train_records)
        val_records, skipped_val = filter_records(val_records)
        if skipped_train or skipped_val:
            print(f"skipped (outside the hierarchical label space): train {dict(skipped_train)}  val {dict(skipped_val)}")
        if not train_records or not val_records:
            raise RuntimeError("no train or val records left inside the hierarchical label space")
    if args.topologies:
        keep = set(args.topologies)
        train_records = [r for r in train_records if r.topology in keep]
        val_records = [r for r in val_records if r.topology in keep]
    if args.max_per_class is not None:
        import random
        rng = random.Random(args.seed)
        val_cap = max(1, round(args.max_per_class * args.val_fraction / (1 - args.val_fraction)))

        def subsample(records, cap):
            by_topology = collections.defaultdict(list)
            for r in records:
                by_topology[r.topology].append(r)
            out = []
            for topology, group in by_topology.items():
                rng.shuffle(group)
                out += group[:cap]
            return out
        train_records, val_records = subsample(train_records, args.max_per_class), subsample(val_records, val_cap)
    count = lambda records: dict(collections.Counter(r.topology for r in records))     # noqa: E731
    print(f"train samples: {len(train_records)} {count(train_records)}  val samples: {len(val_records)} {count(val_records)}")
    if args.hierarchical:
        print(f"train subclasses: {label_counts(train_records, True)}")
    from training.exact_field_input import field_resolution_for
    field_resolution = args.field_resolution or field_resolution_for(args.resolution)
    if args.input == "sdf":
        from training.sdf_input import precompute
        print("computing the SDF inputs (cached) ...")
        precompute([r.voxel_path for r in train_records + val_records], args.resolution, args.sdf_truncation, args.sdf_cache)
    elif args.input == "voxfield":
        from training.field_input import precompute_exact_kind
        from training.inputs import voxfield_source
        coverage = collections.defaultdict(collections.Counter)
        for r in train_records + val_records:
            coverage[r.topology][voxfield_source(r.voxel_path)] += 1
        print("voxfield sources per class (field = exact .field.npz, voxels = fallback):",
              {k: dict(v) for k, v in sorted(coverage.items())})
        if any(c["voxels"] for c in coverage.values()) and not args.voxfield_allow_fallback:
            raise SystemExit("--input voxfield: some samples have no .field.npz (see above); regenerate them with the field "
                             "or pass --voxfield-allow-fallback")
        print("computing the voxfield inputs (sparse cache, once per sample) ...")
        precompute_exact_kind([r.voxel_path for r in train_records + val_records], "voxfield", workers=max(1, args.num_workers))
    elif args.input == "exact_field":
        from training.exact_field_input import precompute
        print(f"computing the exact-field inputs at resolution {field_resolution} (cached, this is the slow part) ...")
        precompute([r.stl_path for r in train_records + val_records], field_resolution, args.field_cache)
    kind = dict(input_kind=args.input, sdf_truncation=args.sdf_truncation, sdf_cache=args.sdf_cache,
                field_resolution=field_resolution, field_cache=args.field_cache)
    oversample_terms = parse_oversample(args.oversample, args.oversample_factor)
    train_indices = oversampled_indices(train_records, oversample_terms) if oversample_terms else None
    if train_indices is not None:
        print(f"oversampling {oversample_terms}: {len(train_records)} -> {len(train_indices)} train draws per epoch")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = dict(input=args.input, sdf_truncation=args.sdf_truncation,
                  field_resolution=field_resolution, max_per_class=args.max_per_class,
                  roots=[str(r) for r in roots], epochs=args.epochs, lr=args.lr,
                  batch_size=args.batch_size, weight_decay=args.weight_decay, seed=args.seed, val_fraction=args.val_fraction,
                  n_train=len(train_records), n_val=len(val_records), train_per_class=count(train_records),
                  val_per_class=count(val_records), resolution=args.resolution)
    # the input contract: input kind + resolution are checked at prediction (predict.check_type_grid); other new keys
    # only when the new options are used
    if args.hierarchical:
        config.update(hierarchical=True, parents=list(PARENTS), subclasses={p: list(s) for p, s in SUBCLASSES.items()},
                      flat_subclasses=list(FLAT_SUBCLASSES), sub_weight=args.sub_weight,
                      train_per_subclass=label_counts(train_records, True), val_per_subclass=label_counts(val_records, True))
    if oversample_terms:
        config.update(oversample=[list(t) for t in oversample_terms], n_train_draws=len(train_indices))
    if args.val_roots:
        config.update(val_roots=[str(r) for r in args.val_roots])
    if args.test_roots:
        config.update(test_roots=[str(r) for r in args.test_roots])
    if args.gpu_encode or args.amp:
        config.update(gpu_encode=bool(args.gpu_encode), amp=bool(args.amp))
    (args.checkpoint_dir / "config.json").write_text(json.dumps(config, indent=1))

    dataset_cls = HierTypeDataset if args.hierarchical else TypeDataset
    # --gpu-encode: the workers only read bytes, so keep them alive across epochs and pin the batches (train_common)
    fast = ({k: v for k, v in loader_kwargs(device, args.num_workers, args.seed).items() if k != "generator"}
            if args.gpu_encode else {})
    train_set = dataset_cls(train_records, args.resolution, **kind)
    if train_indices is not None:
        generator = torch.Generator().manual_seed(args.seed)
        sampler = torch.utils.data.SubsetRandomSampler(train_indices, generator=generator)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=args.num_workers, drop_last=True, **fast)
    else:
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, drop_last=True, **fast)
    val_loader = DataLoader(
        dataset_cls(val_records, args.resolution, **kind),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, **fast,
    )

    from training.exact_field_input import CHANNELS as EXACT_FIELD_CHANNELS
    from training.field_input import CHANNELS
    in_channels = EXACT_FIELD_CHANNELS if args.input == "exact_field" else CHANNELS[args.input]
    model = (HierTypeCNN(in_channels=in_channels) if args.hierarchical else TypeCNN(in_channels=in_channels)).to(device)
    model = configure_speed(args, args.input, model)            # --gpu-encode / --amp (opt-in; defaults unchanged)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.hierarchical:
        steps = dict(train_step=functools.partial(hier_train_step, sub_weight=args.sub_weight),
                     eval_step=functools.partial(hier_eval_step, sub_weight=args.sub_weight))
    else:
        steps = dict(train_step=train_step, eval_step=eval_step)
    run_training_loop(
        model=model, train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler, device=device,
        epochs=args.epochs, patience=args.patience, checkpoint_dir=args.checkpoint_dir,
        extra_metrics_key="acc", **steps,
    )

    if args.test_roots:
        write_model_card(args, config, train_records, val_records, kind, in_channels, device)


def write_model_card(args, config, train_records, val_records, kind, in_channels, device) -> dict:
    """Score best_model.pth on the validation records and on every --test-roots root; write model_card.json."""
    from training.manifest import load_manifest
    model = load_checkpoint_model(args.checkpoint_dir / "best_model.pth", in_channels, device)
    dataset_kwargs = dict(resolution=args.resolution, num_workers=args.num_workers, **kind)   # CPU encode: parallel loaders
    history_path = args.checkpoint_dir / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else {}
    card = dict(
        model="HierTypeCNN" if args.hierarchical else "TypeCNN", hierarchical=bool(args.hierarchical),
        input_kind=args.input, in_channels=in_channels, resolution=args.resolution, seed=args.seed, git=git_commit(),
        checkpoint=str(args.checkpoint_dir / "best_model.pth"),
        dataset_roots=config["roots"], val_roots=config.get("val_roots"), test_roots=[str(r) for r in args.test_roots],
        counts=dict(train=label_counts(train_records, args.hierarchical), val=label_counts(val_records, args.hierarchical)),
        epochs_run=len(history.get("val_loss", [])),
        best_val_loss=min(history["val_loss"]) if history.get("val_loss") else None,
        train_loss_first_last=[history["train_loss"][0], history["train_loss"][-1]] if history.get("train_loss") else None,
        metrics=dict(val=evaluate_records(model, val_records, device, args.batch_size, **dataset_kwargs)),
    )
    if args.hierarchical:
        card.update(parents=list(PARENTS), subclasses={p: list(s) for p, s in SUBCLASSES.items()})
    for root in args.test_roots:
        records = load_manifest(root, require_voxel=True)
        if args.topologies:
            records = [r for r in records if r.topology in set(args.topologies)]
        card["counts"][str(root)] = label_counts(records, args.hierarchical)
        card["metrics"][str(root)] = evaluate_records(model, records, device, args.batch_size, **dataset_kwargs)
        m = card["metrics"][str(root)]
        if args.hierarchical:
            print(f"test {root}: n={m['n']} parent {m.get('parent_acc', float('nan')):.3f} top2 {m.get('parent_top2', float('nan')):.3f} "
                  f"sub {m.get('sub_acc', float('nan')):.3f} sub|pred-parent {m.get('sub_acc_pred_parent', float('nan')):.3f}")
        else:
            print(f"test {root}: n={m['n']} acc {m.get('acc', float('nan')):.3f}")
    (args.checkpoint_dir / "model_card.json").write_text(json.dumps(card, indent=1))
    print(f"wrote {args.checkpoint_dir / 'model_card.json'}")
    return card


if __name__ == "__main__":
    main()
