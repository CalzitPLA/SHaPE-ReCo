#!/usr/bin/env python3
"""
Load dataset_metadata.jsonl manifests produced by dataset_gen into a flat
list of records, each carrying enough to build one training example.

dataset_generator.py writes one dataset_metadata.jsonl per output root AND
per leaf folder (topology dir for type mode, ratio/rotation-group dir for
measure/rotation mode) - so a dataset root normally has many of these
files. We walk the tree and read them all, skipping header-only records.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Record:
    topology: str
    mode: str
    parameters: Dict[str, float]
    rotation: Dict[str, float]  # rx,ry,rz,qx,qy,qz,qw
    stl_path: Path  # absolute
    voxel_path: Optional[Path]  # absolute, or None if not voxelized
    group_key: str  # parent directory name - used to keep whole groups
    # (a fixed-ratio "measure_..." folder, or a fixed-rotation "rot_..."
    # folder) together on one side of the train/val split, so validation
    # doesn't leak the exact same ground-truth value seen in training
    # under a different sampled rotation/dimension.
    placement: Optional[Dict] = None  # in-scale samples: size, position, cuts, full_field (dataset_gen/src/inscale.py)
    # every other top-level key of the JSONL record (subclass, segment, sample_kind, source, member_kind, members, ...):
    # what training/subclass_schema.labels_of derives the hierarchical labels from
    meta: Dict = field(default_factory=dict)


def _resolve(rel: str, meta_file: Path, dataset_root: Path) -> Path:
    """Resolve a path from a metadata record. The generator writes them relative to ITS output root, which is
    `dataset_root` for an ordinary dataset but a shard directory (dataset_root/shardK) when a dataset is
    generated in parallel shards - so try the metadata file's own ancestors, nearest first, up to and
    including dataset_root, and take the first that exists."""
    for base in [*meta_file.parents]:
        if dataset_root == base or dataset_root in base.parents:
            candidate = (base / rel).resolve()
            if candidate.exists():
                return candidate
    return (dataset_root / rel).resolve()


_RECORD_KEYS = ("topology", "mode", "parameters", "rotation", "stl_path", "voxel_path", "placement")


def load_manifest(dataset_root: Path, require_voxel: bool = True) -> List[Record]:
    """Recursively load every dataset_metadata.jsonl under dataset_root."""
    dataset_root = Path(dataset_root).resolve()
    records: List[Record] = []

    for meta_file in sorted(dataset_root.rglob("dataset_metadata.jsonl")):
        with meta_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("record_type") == "dataset_header":
                    continue

                # stl_path/voxel_path in the metadata are always written
                # relative to output_root (dataset_generator.py resolves
                # against output_root, never the leaf dir a given
                # dataset_metadata.jsonl happens to live in): dataset_root
                # for an ordinary dataset, a shard directory for a sharded one
                # (see _resolve).
                stl_path = _resolve(entry["stl_path"], meta_file, dataset_root)

                voxel_rel = entry.get("voxel_path")
                voxel_path = _resolve(voxel_rel, meta_file, dataset_root) if voxel_rel else None

                if require_voxel and (voxel_path is None or not voxel_path.is_file()):
                    continue
                if not stl_path.is_file():
                    continue

                records.append(Record(
                    topology=entry["topology"],
                    mode=entry["mode"],
                    parameters=entry["parameters"],
                    rotation=entry["rotation"],
                    stl_path=stl_path,
                    voxel_path=voxel_path,
                    group_key=f"{entry['topology']}/{stl_path.parent.name}",
                    placement=entry.get("placement"),
                    meta={k: v for k, v in entry.items() if k not in _RECORD_KEYS},
                ))

    return records


def group_split(records: List[Record], val_fraction: float, seed: int) -> tuple[List[Record], List[Record]]:
    """Split by group_key (not by individual record) to avoid leakage.

    All records sharing a group_key (i.e. the same fixed-ratio or
    fixed-rotation folder from measure/rotation mode) land on the same
    side of the split. Type-mode records each get their own unique
    group_key (parent dir = topology name is shared by many, so we
    additionally key type-mode groups by index to keep them fully
    independent, matching the fact that each type-mode sample is an
    independently random draw).
    """
    import random

    keyed: Dict[str, List[Record]] = {}
    for r in records:
        key = r.group_key if r.mode != "type" else f"type/{r.stl_path.name}"
        keyed.setdefault(key, []).append(r)

    keys = list(keyed.keys())
    random.Random(seed).shuffle(keys)

    n_val_groups = max(1, int(round(len(keys) * val_fraction)))
    val_keys = set(keys[:n_val_groups])

    train, val = [], []
    for key, recs in keyed.items():
        (val if key in val_keys else train).extend(recs)

    return train, val
