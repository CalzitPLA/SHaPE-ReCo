#!/usr/bin/env python3
"""
The repair pass after the loop (the plan of 2026-10-15, step 4): every accepted operation judged once more on the
FINAL model by leaving it out. An operation whose absence does not lower the voxel IoU of the model against the input
by more than `tolerance` of the input's voxels did nothing that the later operations did not do too (a cut cut
again, a box inside a later box): it is dropped. And when the model is in more connected pieces than the input
(the ABC batch: 12.6 % multi-solid models; the gate's hard tier-E fail), the one operation whose absence restores the
input's count at the least IoU cost is dropped. Rounds until nothing changes. The operations stay in the program with
`accepted = False` and a `repair` text, so the record shows what was taken back.

    dropped = repair(state, result, tolerance=1e-4)      # [(index, reason, iou_before, iou_after)]

The model without an operation is rebuilt on the loop grid from the remaining operations' SDFs (cached per
operation), so a pass costs one SDF per operation plus one boolean fold per trial.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _fold(state, ops, sdf: Dict[int, np.ndarray], skip: Optional[int] = None) -> np.ndarray:
    s = np.full(state.occ_in.shape, -np.inf, state.dtype)
    for i, o in enumerate(ops):
        if i == skip or not o.accepted:
            continue
        if i not in sdf:
            sdf[i] = state.prim_sdf(o.primitive)
        if o.op == "add":
            s = np.maximum(s, sdf[i])
        elif o.op == "cut":
            s = np.minimum(s, -sdf[i])
        elif o.op == "intersect":
            s = np.minimum(s, sdf[i])
    return s


def _iou(state, s: np.ndarray) -> float:
    model = s >= 0
    inter = int(np.count_nonzero(model & state.occ_in))
    union = int(np.count_nonzero(model | state.occ_in))
    return inter / union if union else 0.0


def repair(state, result, tolerance: float = 1e-4, max_rounds: int = 10, pieces_fn=None,
           max_loss: float = 0.02) -> List[Tuple[int, str, float, float]]:
    """See the module docstring. `pieces_fn(mask) -> int` counts connected pieces (residual_loop._n_pieces);
    None: the solid rule is skipped. The first add (the base) is never dropped. `max_loss`: the solid rule gives up
    at most this much IoU - an add that floats a fraction of a voxel above the base makes "two pieces" on the grid
    and is a fit-precision matter, not a severing (measured 2026-10-15 on bs_box_step: the rule had dropped the step's
    add at a cost of 0.12 IoU)."""
    from reco_engine.residual_loop import _n_pieces
    pieces_fn = pieces_fn or _n_pieces
    ops = result.operations
    sdf: Dict[int, np.ndarray] = {}
    dropped: List[Tuple[int, str, float, float]] = []
    n_in = int(state.occ_in.sum())
    tol = tolerance * max(1, n_in)
    allowed = max(1, pieces_fn(state.occ_in))
    for _ in range(max_rounds):
        s_all = _fold(state, ops, sdf)
        iou_all, pieces_all = _iou(state, s_all), pieces_fn(s_all >= 0)
        active = [i for i, o in enumerate(ops) if o.accepted]
        base = next((i for i in active if ops[i].op == "add"), None)
        change = None
        # 1. the solid rule: more pieces than the input -> the one operation whose absence restores the count
        if pieces_all > allowed:
            best = None
            for i in active:
                if i == base:
                    continue
                s = _fold(state, ops, sdf, skip=i)
                if pieces_fn(s >= 0) <= allowed:
                    v = _iou(state, s)
                    if best is None or v > best[1]:
                        best = (i, v)
            if best is not None and best[1] >= iou_all - max_loss:
                change = (best[0], f"left out: the model was in {pieces_all} pieces (the input in {allowed}); without it "
                                   f"{allowed} at IoU {best[1]:.4f} instead of {iou_all:.4f}", iou_all, best[1])
        # 2. the redundancy rule: an operation whose absence changes nothing (within the tolerance)
        if change is None:
            for i in reversed(active):                     # the latest first: a later cut usually repeats an earlier one
                if i == base:
                    continue
                s = _fold(state, ops, sdf, skip=i)
                v = _iou(state, s)
                lost = int(np.count_nonzero((s_all >= 0) != (s >= 0)))
                if v >= iou_all - tolerance and lost <= tol:
                    change = (i, f"left out: without it the IoU is {v:.4f} instead of {iou_all:.4f} ({lost} voxels differ, "
                                 f"within {tol:.0f})", iou_all, v)
                    break
        if change is None:
            break
        i, reason, before, after = change
        ops[i].accepted = False
        ops[i].primitive.label = ((ops[i].primitive.label + "; ") if ops[i].primitive.label else "") + "repair: " + reason
        dropped.append(change)
        state.s_model = _fold(state, ops, sdf)
    return dropped
