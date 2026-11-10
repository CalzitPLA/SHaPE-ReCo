#!/usr/bin/env python3
"""
The largest completely filled box in a voxel mask: the box cover of the residual add-cut loop's voxel-cube fallback
(reco_engine/residual_loop.py, `voxel_boxes` / `largest_boxes`).

The module once held the guards of the greedy voxel loop (route C: inscribed-box-first proposals, sliver opening, the
assembly grade, `LoopGuards`, `propose_guarded`); route C was removed on 2026-09-30 and with it those guards. They are
in git history.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def max_rectangle(mask: np.ndarray) -> Tuple[int, Tuple[int, int, int, int]]:
    """Largest all-True rectangle of a 2D bool array: (area, (r0, r1, c0, c1)), r1 and c1 inclusive. Histogram method."""
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)
    best = (0, (0, -1, 0, -1))
    for r in range(rows):
        heights = np.where(mask[r], heights + 1, 0)
        stack = []                                             # column indices with increasing heights
        for c in range(cols + 1):
            h = heights[c] if c < cols else 0
            start = c
            while stack and stack[-1][1] >= h:
                s, height = stack.pop()
                area = height * (c - s)
                if area > best[0]:
                    best = (area, (r - height + 1, r, s, c - 1))
                start = s
            stack.append((start, h))
    return best


def largest_box_in_blocks(blocks: np.ndarray):
    """Largest all-True box of a 3D bool array: (volume, (i0, i1, r0, r1, c0, c1)), upper bounds inclusive. Exact:
    for every range of the first axis the AND of its slices is a 2D map whose largest rectangle is found."""
    n = blocks.shape[0]
    best = (0, None)
    for i0 in range(n):
        slab = blocks[i0].copy()
        for i1 in range(i0, n):
            if i1 > i0:
                slab &= blocks[i1]
            if not slab.any():
                break
            area, (r0, r1, c0, c1) = max_rectangle(slab)
            volume = area * (i1 - i0 + 1)
            if volume > best[0]:
                best = (volume, (i0, i1, r0, r1, c0, c1))
    return best
