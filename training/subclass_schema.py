#!/usr/bin/env python3
"""
Label space of the hierarchical Type CNN: 5 parent shapes (head 1) and 24 subclasses (head 2, masked to the parent).

    PARENTS          ("box", "cylinder", "cone", "sphere", "torus")
    SUBCLASSES       parent -> its subclasses; the FLAT subclass index is the order of appearance over PARENTS
                     (box/full = 0, box/hollow = 1, ..., torus/frame = 23)
    FLAT_SUBCLASSES  ("box/full", ..., "torus/frame"), index = flat index
    SUB_PARENT       flat index -> parent index
    parent_mask()    (5, 24) bool: row p is True on the subclasses of parent p
    labels_of(rec)   (parent_idx, flat_sub_idx) of a manifest record (training.manifest.Record or the raw JSONL dict)
    filter_records   keep the records labels_of accepts; count the rest by reason

Where the subclass comes from:
  * new records (the v5 generators) carry `subclass` next to `topology` (the parent);
  * OLD records without `subclass` are derived: a `segment` label (dataset_gen/src/solid_builder.labels: {"u": ...} ->
    sector, only {"v": ...} -> band) or `sample_kind == "segment"`; the in-scale hollow flag (`placement["hollow"]`,
    dataset_generator.py --hollow-prob; `hollow_failed` is NOT hollow) -> hollow; anything else -> full.
  * records whose topology is not one of PARENTS (the 12 other legacy topologies) are not labelled: labels_of raises
    UnknownTopology and filter_records skips them with a count.

Kept free of torch and of topology_schema's dataset_gen import, so it is cheap to import anywhere.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

PARENTS: Tuple[str, ...] = ("box", "cylinder", "cone", "sphere", "torus")
SUBCLASSES: Dict[str, Tuple[str, ...]] = {
    "box": ("full", "hollow", "ibeam", "rhs", "frame"),
    "cylinder": ("full", "sector", "hollow", "tube", "frame"),
    "cone": ("full", "sector", "hollow", "frame"),
    "sphere": ("full", "sector", "band", "hollow", "frame"),
    "torus": ("full", "sector", "band", "hollow", "frame"),
}
PARENT_TO_IDX: Dict[str, int] = {p: i for i, p in enumerate(PARENTS)}
FLAT_SUBCLASSES: Tuple[str, ...] = tuple(f"{p}/{s}" for p in PARENTS for s in SUBCLASSES[p])
FLAT_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(FLAT_SUBCLASSES)}
SUB_PARENT: Tuple[int, ...] = tuple(PARENT_TO_IDX[name.split("/")[0]] for name in FLAT_SUBCLASSES)
NUM_PARENTS = len(PARENTS)
NUM_SUBCLASSES = len(FLAT_SUBCLASSES)
assert NUM_SUBCLASSES == 24


class UnknownTopology(KeyError):
    """The record's topology is not one of PARENTS."""


def parent_mask() -> np.ndarray:
    """(NUM_PARENTS, NUM_SUBCLASSES) bool: mask[p, s] is True iff flat subclass s belongs to parent p."""
    mask = np.zeros((NUM_PARENTS, NUM_SUBCLASSES), dtype=bool)
    for s, p in enumerate(SUB_PARENT):
        mask[p, s] = True
    return mask


PARENT_MASK = parent_mask()


def _field(record, key, default=None):
    """A record field from a raw JSONL dict, or from a manifest Record (attribute, else its `meta` dict)."""
    if isinstance(record, Mapping):
        return record.get(key, default)
    if hasattr(record, key):
        return getattr(record, key)
    return (getattr(record, "meta", None) or {}).get(key, default)


def subclass_name(record) -> str:
    """The subclass name (within the parent) of a record: its `subclass`, else derived (module docstring)."""
    explicit = _field(record, "subclass")
    if explicit:
        return str(explicit)
    segment = _field(record, "segment")
    if isinstance(segment, Mapping) and segment:
        return "sector" if "u" in segment else "band"
    if segment or _field(record, "sample_kind") == "segment":
        return "sector"
    placement = _field(record, "placement") or {}
    if placement.get("hollow"):
        return "hollow"
    return "full"


def labels_of(record) -> Tuple[int, int]:
    """(parent_idx into PARENTS, flat subclass idx into FLAT_SUBCLASSES). Raises UnknownTopology for a topology
    outside PARENTS and ValueError for a subclass its parent does not have (e.g. cylinder/band)."""
    topology = _field(record, "topology")
    if topology not in PARENT_TO_IDX:
        raise UnknownTopology(topology)
    sub = subclass_name(record)
    flat = FLAT_TO_IDX.get(f"{topology}/{sub}")
    if flat is None:
        raise ValueError(f"subclass {sub!r} is not one of {topology}'s {SUBCLASSES[topology]}")
    return PARENT_TO_IDX[topology], flat


def flat_name(record) -> str:
    return FLAT_SUBCLASSES[labels_of(record)[1]]


def filter_records(records: Iterable) -> Tuple[List, Counter]:
    """(records labels_of accepts, Counter of the skipped ones: "topology:<name>" / "subclass:<parent>/<sub>")."""
    kept, skipped = [], Counter()
    for r in records:
        placement = getattr(r, "placement", None) or (r.get("placement") if isinstance(r, dict) else None) or {}
        if placement.get("degenerate"):                  # too few voxels to learn from (scripts/revoxelize_parity.py, --min-voxels)
            skipped["degenerate"] += 1
            continue
        try:
            labels_of(r)
        except UnknownTopology:
            skipped[f"topology:{_field(r, 'topology')}"] += 1
            continue
        except ValueError:
            skipped[f"subclass:{_field(r, 'topology')}/{subclass_name(r)}"] += 1
            continue
        kept.append(r)
    return kept, skipped


def subclass_counts(records: Iterable) -> Dict[str, int]:
    """{flat subclass name: count} over records labels_of accepts (the rest are ignored)."""
    return dict(sorted(Counter(flat_name(r) for r in filter_records(records)[0]).items()))
