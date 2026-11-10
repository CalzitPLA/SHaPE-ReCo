#!/usr/bin/env python3
"""
Local voxel-space position+scale refinement - the central principle of
this version of the reconstruction architecture: positioning (and, to
the same bounded degree, sizing) is a local voxel optimization
problem, not something a CNN (or placement.py's deterministic
bounding-box/center-of-gravity estimate) gets to dictate outright.
Both remain *initializers* only; this module's search is the authority
on the candidate's final position and scale.

Why this exists on top of placement.py: estimate_placement() +
place_candidate() already solve *scale and gross position* in one shot
(crop the candidate to the target's bounding-box footprint, center it
on the target's center of gravity) - a good, cheap first mark, but a
single shot, with no attempt to actually search for a better fit once
the candidate is roughly the right size and roughly in the right
place. This module takes that first mark and refines it: nudge the
candidate a few voxels in each translation axis and a small amount up
or down in scale, keep whichever nudge improves the fit, shrink the
step once nothing nearby helps, repeat until both steps are too small
to matter.

Performance requirement this exists to satisfy: geometry (freecadcmd)
and voxelization (stltovoxel) are the expensive steps in this pipeline
(confirmed: tens of seconds per candidate, mostly subprocess overhead -
see reco_engine/fit_score.py's docstring and the void-cutting roadmap).
Both happen exactly once per candidate, *before* this module ever runs
- everything below operates on plain numpy arrays already in memory,
translating/rescaling the candidate via array slicing and a
nearest-neighbor resize (the same technique
reco_engine.placement.place_candidate already uses to place a
candidate at a target position - this module just tries several
nearby positions/scales instead of committing to the first one).

Fit definition (matches the project's actual goal - a small number of
primitives that explain most of a shape's material fast enough for
large-scale Monte Carlo geometry, not exact CAD reconstruction):

    fit = correctly_explained_target_voxels / target_material_voxels

This is exactly BooleanMetrics.recall from reco_engine.compare (recall
= intersection / target_count) - reused directly, not reimplemented.
Unlike IoU, recall does not penalize a candidate for covering voxels
outside the target, so on its own it has no reason to stop growing: an
oversized candidate that fully explains the target scores as well as an
exactly-sized one, and a recall-only search drifts to the largest scale
it is allowed (measured: x1.85 on nearly every candidate, precision
0.05-0.5). That is only tolerable where oversizing can be undone later
- and in the cut phase it is destructive, since an oversized void
removes material the input really has.

So the search score is

    score = recall - spill_weight * spill_fraction
    spill_fraction = candidate voxels outside `allowed` / candidate voxels

where `allowed` is the region the candidate may occupy without penalty
- the whole input for a cover primitive (over-covering the *remaining*
residual onto already-covered material is harmless, spilling outside
the input is not), the over-coverage region for a void. With
spill_weight=0 (the default) this is exactly the recall-only score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from reco_engine.compare import BooleanMetrics, as_bool, compare, metrics_bool

# 6-neighborhood translation directions: +-X, +-Y, +-Z.
_AXIS_DIRECTIONS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
)
# 2 scale directions: grow, shrink. Evaluated alongside the 6 translation
# neighbors each iteration (8 candidate moves total), so growing/shrinking
# competes on equal footing with nudging - whichever actually improves
# the fit most wins, deterministically (fixed order breaks ties).
_SCALE_DIRECTIONS: Tuple[int, ...] = (1, -1)


@dataclass
class SearchObjective:
    """How a phase of the greedy loop wants its candidates sized.

    spill_weight: cost of a candidate's spill fraction (see module
    docstring) in the search score. min_precision: hard gate applied
    afterwards by greedy_reconstruct.greedily_explain - the fraction of
    the final candidate that must lie inside the allowed region, or it
    is rejected (0 = no gate).
    """
    spill_weight: float = 0.0
    min_precision: float = 0.0


# Cover primitives may overshoot a little - the cut phase can still carve
# it back - but not run away.
COVER_OBJECTIVE = SearchObjective(spill_weight=0.5)
# With no cut phase there is nothing to undo an overshoot.
ADDITIVE_ONLY_OBJECTIVE = SearchObjective(spill_weight=1.0)
# A void may only remove over-coverage: material the input actually has
# must survive, so this is precision-driven and gated.
CUT_OBJECTIVE = SearchObjective(spill_weight=3.0, min_precision=0.8)
# The strict cut: used with allowed = "everything that is not input material"
# (reco_engine.void_quality), so a void may protrude into empty space but must
# not touch material - a hard purity gate instead of a 20 % allowance.
STRICT_CUT_OBJECTIVE = SearchObjective(spill_weight=3.0, min_precision=0.98)


@dataclass
class LocalSearchConfig:
    """All bounds on the search, so its cost is predictable regardless
    of input. Defaults are relative to the project's standard 64^3 grid
    (reco_engine.pipeline.reconstruct's own default `resolution`):
    initial_step=4 voxels covers a meaningful fraction of a typical
    residual blob's own size in one hop; minimum_step=1 is the smallest
    a voxel-grid translation can be. initial_scale_step=0.08 (8% size
    change) / minimum_scale_step=0.02 are the analogous bounds for
    scale. step_decay=0.5 halves both on a round with no improving
    neighbor. max_iterations bounds step-levels; max_evaluations bounds
    total candidate-fit computations (8 per iteration at most, so
    max_iterations=8 * 8 neighbors = 64 == max_evaluations by default -
    both bounds are reachable, not just one dominating in practice)."""
    initial_step: int = 4
    minimum_step: int = 1
    initial_scale_step: float = 0.08
    minimum_scale_step: float = 0.02
    step_decay: float = 0.5
    max_iterations: int = 8
    max_evaluations: int = 64


@dataclass
class PlacementResult:
    """Outcome of a local search run. `position` is the final
    translation (voxel offset from the candidate's position at the
    start of search, i.e. relative to wherever placement.place_candidate
    already put it); `initial_position` is always (0, 0, 0) by
    convention (search starts from the candidate array it's handed,
    wherever that already is), so `displacement` is defined for
    clarity even though today it always equals `position`. `scale` is
    the cumulative scale factor applied relative to the starting
    candidate (1.0 = unchanged)."""
    position: Tuple[int, int, int]
    scale: float
    score: float
    initial_position: Tuple[int, int, int]
    displacement: Tuple[int, int, int]
    evaluations: int
    iterations: int
    converged: bool
    step_history: List[int] = field(default_factory=list)
    scale_history: List[float] = field(default_factory=list)
    metrics: BooleanMetrics = None
    voxel: np.ndarray = None  # the candidate translated/rescaled to its final position - ready to use


def _translate(volume: np.ndarray, offset: Tuple[int, int, int]) -> np.ndarray:
    """Shift `volume` by `offset` (voxels), zero-padding what moves off
    the grid edge."""
    if offset == (0, 0, 0):
        return volume
    shape = volume.shape
    out = np.zeros_like(volume)
    src_slices, dst_slices = [], []
    for d, size in zip(offset, shape):
        if d >= 0:
            src_slices.append(slice(0, size - d))
            dst_slices.append(slice(d, size))
        else:
            src_slices.append(slice(-d, size))
            dst_slices.append(slice(0, size + d))
    if any(s.stop <= s.start for s in src_slices + dst_slices):
        return out  # shifted entirely off-grid
    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def _resize_nearest(volume: np.ndarray, target_shape) -> np.ndarray:
    """Nearest-neighbor resize of a 3D array to `target_shape` (pure
    numpy, no scipy - same technique as reco_engine.placement's own
    resize, duplicated rather than imported since it's three lines and
    this module should stay usable even if placement.py's internals
    change shape)."""
    src_shape = np.array(volume.shape)
    tgt_shape = np.maximum(np.asarray(target_shape), 1)
    idx = [
        np.clip((np.arange(t) * (s / t)).astype(int), 0, s - 1)
        for s, t in zip(src_shape, tgt_shape)
    ]
    return volume[np.ix_(idx[0], idx[1], idx[2])]


def _rescale(volume: np.ndarray, factor: float, resolution: int) -> np.ndarray:
    """Resize the occupied region of `volume` by `factor` around its
    own current center (NOT the target's center of gravity - that was
    already handled by the one-shot initial placement; here we want to
    grow/shrink in place), re-embedded in a fresh resolution^3 grid at
    the same center. factor > 1 grows, < 1 shrinks."""
    occupied = np.argwhere(volume)
    if occupied.size == 0 or factor <= 0:
        return volume
    lo, hi = occupied.min(axis=0), occupied.max(axis=0)
    center = (lo.astype(np.float64) + hi.astype(np.float64) + 1) / 2.0
    crop = volume[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]

    new_shape = np.maximum(np.round(np.array(crop.shape) * factor).astype(int), 1)
    resized = _resize_nearest(crop, new_shape)

    out = np.zeros((resolution, resolution, resolution), dtype=volume.dtype)
    half_lo = np.array(resized.shape) // 2
    half_hi = np.array(resized.shape) - half_lo
    start = np.round(center).astype(int) - half_lo
    stop = np.round(center).astype(int) + half_hi

    src_start = np.clip(-start, 0, None)
    src_stop = np.array(resized.shape) - np.clip(stop - resolution, 0, None)
    dst_start = np.clip(start, 0, None)
    dst_stop = np.clip(stop, None, resolution)

    if np.any(src_stop <= src_start) or np.any(dst_stop <= dst_start):
        return out  # scaled entirely off-grid

    out[dst_start[0]:dst_stop[0], dst_start[1]:dst_stop[1], dst_start[2]:dst_stop[2]] = \
        resized[src_start[0]:src_stop[0], src_start[1]:src_stop[1], src_start[2]:src_stop[2]]
    return out


def spill_fraction(candidate: np.ndarray, allowed: np.ndarray) -> float:
    """Fraction of `candidate`'s voxels that lie outside `allowed`. An
    empty candidate counts as fully spilled, so a search can never
    "improve" by pushing its candidate off the grid."""
    return _spill(as_bool(candidate), as_bool(allowed))


def _spill(candidate: np.ndarray, allowed: np.ndarray) -> float:
    count = int(np.count_nonzero(candidate))
    if count == 0:
        return 1.0
    return float(count - int(np.count_nonzero(candidate & allowed))) / count


class _Scorer:
    """Scores candidates against one fixed target (and allowed region) -
    the bool grids and the target's voxel count are prepared once, not once
    per evaluation. Same numbers as compare()/spill_fraction(), fewer passes
    over the grid: the search evaluates ~64 candidates per proposal."""

    def __init__(self, target: np.ndarray, allowed: np.ndarray = None, spill_weight: float = 0.0,
                 scope: np.ndarray = None):
        self.scope = None if scope is None else as_bool(scope)
        self.target = as_bool(target)
        self.target_count = int(np.count_nonzero(self.target))
        self.allowed = self.target if allowed is None else as_bool(allowed)
        self.spill_weight = spill_weight

    def __call__(self, candidate: np.ndarray) -> Tuple[BooleanMetrics, float]:
        candidate = as_bool(candidate)
        if self.scope is not None:          # only what lies inside the scope can change anything
            candidate = candidate & self.scope
        metrics = metrics_bool(self.target, candidate, self.target_count)
        score = metrics.recall
        if self.spill_weight > 0.0:
            score -= self.spill_weight * _spill(candidate, self.allowed)
        return metrics, score


def _fit(target: np.ndarray, candidate: np.ndarray, allowed: np.ndarray = None,
         spill_weight: float = 0.0) -> Tuple[BooleanMetrics, float]:
    """(metrics, search score) of `candidate` against `target`."""
    return _Scorer(target, allowed, spill_weight)(candidate)


def refine_position(
    target: np.ndarray,
    candidate: np.ndarray,
    config: LocalSearchConfig = None,
    allowed: np.ndarray = None,
    spill_weight: float = 0.0,
    scope: np.ndarray = None,
) -> PlacementResult:
    """Bounded coarse-to-fine local translation+scale search in voxel
    space.

    `candidate` should already be at its initial position/scale (e.g.
    the output of reco_engine.placement.place_candidate against
    `target`) - this refines from there, it does not replace the
    initial placement estimate. Both arrays must be the same shape,
    {0,1} (or bool) occupancy grids.

    Each iteration evaluates 8 neighbors of the current candidate: 6
    translation nudges (+-X, +-Y, +-Z, at the current step size) and 2
    scale nudges (grow/shrink by the current scale step), all at the
    *current* position/scale - not 8 independent single-axis searches.
    Whichever neighbor improves the search score (recall minus the
    weighted spill outside `allowed` - see module docstring; `allowed`
    defaults to `target`, spill_weight to 0 = recall only) most is adopted; if none improve, both step
    sizes halve and the round repeats. Stops once both steps are below
    their configured minimums, or a bound (max_iterations /
    max_evaluations) is hit.

    Deterministic: for identical inputs and config, always produces the
    identical result (ties broken by the fixed neighbor order; "no
    improvement" is a strict >, not >=, so a tie never moves the
    candidate).
    """
    config = config or LocalSearchConfig()
    if target.shape != candidate.shape:
        raise ValueError(f"target/candidate shape mismatch: {target.shape} vs {candidate.shape}")
    resolution = target.shape[0]

    position = (0, 0, 0)
    scale = 1.0
    current = candidate
    score_of = _Scorer(target, allowed, spill_weight, scope)
    best_metrics, best_score = score_of(current)
    evaluations = 1
    iterations = 0
    step = config.initial_step
    scale_step = config.initial_scale_step
    step_history: List[int] = []
    scale_history: List[float] = []
    converged = False

    while step >= config.minimum_step or scale_step >= config.minimum_scale_step:
        if iterations >= config.max_iterations or evaluations >= config.max_evaluations:
            break
        iterations += 1
        step_history.append(step)
        scale_history.append(scale_step)

        best_move = None  # ("translate", offset) | ("scale", factor)
        best_neighbor_score = best_score
        best_neighbor_metrics = None
        best_neighbor_voxel = None

        if step >= config.minimum_step:
            for dx, dy, dz in _AXIS_DIRECTIONS:
                if evaluations >= config.max_evaluations:
                    break
                offset = (dx * step, dy * step, dz * step)
                neighbor_voxel = _translate(current, offset)
                neighbor_metrics, neighbor_score = score_of(neighbor_voxel)
                evaluations += 1
                if neighbor_score > best_neighbor_score:
                    best_neighbor_score = neighbor_score
                    best_move = ("translate", offset)
                    best_neighbor_metrics = neighbor_metrics
                    best_neighbor_voxel = neighbor_voxel

        if scale_step >= config.minimum_scale_step:
            for direction in _SCALE_DIRECTIONS:
                if evaluations >= config.max_evaluations:
                    break
                factor = 1.0 + direction * scale_step
                neighbor_voxel = _rescale(current, factor, resolution)
                neighbor_metrics, neighbor_score = score_of(neighbor_voxel)
                evaluations += 1
                if neighbor_score > best_neighbor_score:
                    best_neighbor_score = neighbor_score
                    best_move = ("scale", factor)
                    best_neighbor_metrics = neighbor_metrics
                    best_neighbor_voxel = neighbor_voxel

        if best_move is not None:
            kind, value = best_move
            if kind == "translate":
                position = tuple(p + o for p, o in zip(position, value))
            else:
                scale *= value
            current = best_neighbor_voxel
            best_score = best_neighbor_score
            best_metrics = best_neighbor_metrics
        else:
            new_step = int(step * config.step_decay)
            step = new_step if new_step < step else step - 1
            scale_step = scale_step * config.step_decay

    else:
        converged = True  # loop exited via the while condition, not a bound

    return PlacementResult(
        position=position,
        scale=scale,
        score=best_score,
        initial_position=(0, 0, 0),
        displacement=position,
        evaluations=evaluations,
        iterations=iterations,
        converged=converged,
        step_history=step_history,
        scale_history=scale_history,
        metrics=best_metrics,
        voxel=current,
    )
