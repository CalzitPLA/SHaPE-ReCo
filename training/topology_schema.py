#!/usr/bin/env python3
"""
Topology/parameter schema shared by all three CNNs.

Deliberately does NOT redefine TOPOLOGIES/PARAMETER_SPECS - it imports
them from dataset_gen/src/dataset_utils.py, the single source of truth
also used to *generate* the training data. Keeping one source of truth
means the label space used at training time can never silently drift
from what the dataset generator actually produces.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

SANDBOX_ROOT = Path(__file__).resolve().parent.parent
DATASET_GEN_ROOT = SANDBOX_ROOT / "dataset_gen"

if str(DATASET_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(DATASET_GEN_ROOT))

from src.dataset_utils import TOPOLOGIES, PARAMETER_SPECS  # noqa: E402

# Stable ordering: index <-> topology name, fixed by TOPOLOGIES' own order.
TOPOLOGY_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(TOPOLOGIES)}
IDX_TO_TOPOLOGY: Dict[int, str] = {i: name for name, i in TOPOLOGY_TO_IDX.items()}
NUM_TOPOLOGIES = len(TOPOLOGIES)


def _param_names(topology: str) -> List[str]:
    """Stable parameter ordering for one topology (dict insertion order)."""
    return list(PARAMETER_SPECS[topology].keys())


PARAM_NAMES: Dict[str, List[str]] = {t: _param_names(t) for t in TOPOLOGIES}
PARAM_COUNTS: Dict[str, int] = {t: len(names) for t, names in PARAM_NAMES.items()}
MAX_PARAMS = max(PARAM_COUNTS.values())


def _bounds(topology: str, name: str) -> tuple[float, float]:
    """Min/max bound for one parameter, for normalization to [-1, 1].

    Angle params (*_deg) use their own spec range. Categorical params
    (num_sides etc., {"choices": [...]}) use the choice range - they're
    regressed as a continuous value and rounded to the nearest valid
    choice at reconstruction time (see reco_engine).
    """
    spec = PARAMETER_SPECS[topology][name]
    if "min" in spec and "max" in spec:
        return float(spec["min"]), float(spec["max"])
    if "choices" in spec:
        choices = spec["choices"]
        return float(min(choices)), float(max(choices))
    raise ValueError(f"Unrecognized parameter spec for {topology}.{name}: {spec}")


PARAM_BOUNDS: Dict[str, Dict[str, tuple[float, float]]] = {
    t: {name: _bounds(t, name) for name in names}
    for t, names in PARAM_NAMES.items()
}


def normalize_params(topology: str, params: Dict[str, float]) -> List[float]:
    """Map a topology's raw parameter dict to a fixed-length [-1, 1] vector.

    Unused slots (topologies with fewer than MAX_PARAMS params) are 0.
    """
    names = PARAM_NAMES[topology]
    bounds = PARAM_BOUNDS[topology]
    vec = [0.0] * MAX_PARAMS
    for i, name in enumerate(names):
        lo, hi = bounds[name]
        value = float(params[name])
        span = hi - lo
        normed = 0.0 if span == 0 else (2.0 * (value - lo) / span) - 1.0
        vec[i] = max(-1.0, min(1.0, normed))
    return vec


def denormalize_params(topology: str, vec: List[float]) -> Dict[str, float]:
    """Inverse of normalize_params: [-1, 1] vector -> raw parameter dict.

    Categorical ({"choices": [...]}) parameters are denormalized the
    same linear way as continuous ones (they share PARAM_BOUNDS' choice
    range - see _bounds) and then snapped to the nearest valid choice,
    per this function's own contract (see _bounds's docstring): the
    network only ever regresses a single continuous scalar per slot,
    and this is the one place that scalar is turned back into one of
    the topology's actual discrete choices.
    """
    names = PARAM_NAMES[topology]
    bounds = PARAM_BOUNDS[topology]
    specs = PARAMETER_SPECS[topology]
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        lo, hi = bounds[name]
        normed = max(-1.0, min(1.0, float(vec[i])))
        value = lo + (normed + 1.0) * 0.5 * (hi - lo)
        choices = specs[name].get("choices")
        if choices is not None:
            value = min(choices, key=lambda c: abs(c - value))
        out[name] = value
    return out


def param_mask(topology: str) -> List[float]:
    """1.0 for slots that are real parameters of `topology`, else 0.0."""
    count = PARAM_COUNTS[topology]
    return [1.0] * count + [0.0] * (MAX_PARAMS - count)


# The hierarchical Type CNN's 5-parent / 24-subclass label space (training/subclass_schema.py), re-exported here so the
# whole label schema can be imported from one place.
from training.subclass_schema import (  # noqa: E402,F401
    FLAT_SUBCLASSES, NUM_PARENTS, NUM_SUBCLASSES, PARENT_MASK, PARENT_TO_IDX, PARENTS, SUB_PARENT, SUBCLASSES,
    UnknownTopology, filter_records, labels_of, parent_mask, subclass_name,
)

assert all(p in TOPOLOGY_TO_IDX for p in PARENTS), "every hierarchical parent must be a dataset_gen topology"
