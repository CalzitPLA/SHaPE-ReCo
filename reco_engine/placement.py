#!/usr/bin/env python3
"""
Positioning a candidate primitive correctly within the shared voxel
grid it's being compared/subtracted against.

State of 2026-10-01: the stretched-voxel placement below belonged to the
removed route C (reco_engine.pipeline was removed with it on 2026-09-30;
git history). What the main line uses is the axis convention:
`array_to_world_position` / `array_to_world` (reco_engine/geometry.py);
`estimate_placement` is still called by scripts/eval_stage_ensemble.py and
scripts/check_observability_v5.py. The text below describes the original
setting.

The gap this closes: reco_engine.pipeline.voxelize_stl shells out to
the external `stltovoxel` CLI (dataset_gen/src/voxelization.py), which
auto-fits *each mesh's own bounding box* to fill the resolution^3 grid
independently, snugly, regardless of the mesh's absolute physical size
- there is no shared world coordinate frame or shared voxel-to-mm scale
across separate voxelize_stl() calls. That's harmless for a single,
standalone shape (ensemble.py's top-K arbitration, or a whole-input
first classification), but it silently breaks any residual-based
comparison: a freshly built candidate always ends up filling ~the whole
grid after voxelization (that is what "auto-fit" means, independent of
what scale_mm it was built at), while the *target* it's being scored
against (a residual "missing material" blob, or an over-coverage "void"
blob) usually occupies only a small, off-center region of the *shared*
grid it lives in. Comparing the two without correcting for this
measures almost nothing.

The fix used here is deliberately the cheap, deterministic "good first
mark" rather than a learned position predictor - reshape the candidate
(in voxel-index space, not by rebuilding the mesh at a different
scale_mm, which auto-fit would immediately undo) to the target's own
bounding-box footprint, then place it at the target's center of
gravity (mean of its occupied voxel coordinates - not just the
bounding-box midpoint, which is biased by asymmetric residual shapes).
No scipy dependency: a nearest-neighbor resize is a few lines of
numpy indexing, and this needs to keep working under an isolated
Python that may not have scipy the same way model_registry.py already
works around one that may not have torch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def array_to_world_position(vec):
    """A SIGNED voxel-space quantity (a position relative to the grid centre, a displacement) in
    voxel-array order [a0, a1, a2] -> FreeCAD (x, y, z).

    Besides reversing the axis order (see array_to_world) this negates y: the voxel array is
    [z, y, x] but its y axis runs BACKWARDS relative to the world (stltovoxel writes image rows
    top to bottom, so array index a1 falls as world y rises; z and x run forwards). Verified with
    an asymmetric marker on each axis (see tests/test_array_axes.py). Sizes/extents are unsigned
    and must use array_to_world, not this."""
    a0, a1, a2 = (float(v) for v in vec)
    return [a2, -a1, a0]


def array_to_world(vec):
    """Voxel-array axis order -> FreeCAD (x, y, z) order, for UNSIGNED quantities (extents, sizes).
    A signed quantity (position, displacement) also needs y negated: use array_to_world_position.

    The voxelizer's output array is indexed [z, y, x] (verified with an
    asymmetric part: a feature on +X lands on array axis 2, one on +Z on
    axis 0), so every per-axis quantity this pipeline computes in voxel
    space (positions, displacements, extents) is in (z, y, x) order and
    has to be reversed before it is used as an XYZ vector in FreeCAD.
    Kept out of the algorithm itself, which is consistently in array
    order; call it only where voxel-space values cross into world space.
    """
    return [float(v) for v in reversed(list(vec))]


@dataclass
class Placement:
    center_of_gravity_vox: np.ndarray   # (3,) float, voxel-index space
    bbox_min_vox: np.ndarray            # (3,) int
    bbox_max_vox: np.ndarray            # (3,) int, inclusive
    extent_mm: np.ndarray               # (3,) float, physical bounding-box size
    anchor: str = "cog"                 # what center_of_gravity_vox holds: "cog" = centre of gravity, "bbox" = bounding-box centre

    @property
    def extent_vox(self) -> np.ndarray:
        """Bounding-box size in voxels - what a candidate should be
        resized to before being placed at center_of_gravity_vox."""
        return (self.bbox_max_vox - self.bbox_min_vox + 1).astype(np.int64)

    @classmethod
    def from_center_extent(cls, center_vox: np.ndarray, extent_vox: np.ndarray,
                           voxel_size_mm: float) -> "Placement":
        """Rebuild the Placement of a stored primitive from its center
        and bounding-box size in voxels, such that `extent_vox` of the
        result equals the given extent (bbox_max is inclusive)."""
        extent_vox = np.asarray(extent_vox).astype(int)
        bbox_min = (center_vox - extent_vox / 2.0).astype(int)
        return cls(
            center_of_gravity_vox=center_vox,
            bbox_min_vox=bbox_min,
            bbox_max_vox=bbox_min + extent_vox - 1,
            extent_mm=extent_vox * voxel_size_mm,
        )


def estimate_placement(target_voxel: np.ndarray, resolution: int, scale_mm: float, center: str = "cog") -> Placement:
    """Compute where and how big a primitive explaining `target_voxel`
    (a {0,1} occupancy grid - a residual or over-coverage signal) should
    be, directly from its own occupied voxels. Deterministic - no CNN
    involved - meant as the geometric "first mark" placement/scale
    correction the type/rotation/dimension CNNs don't provide.

    `scale_mm` here is the physical size the *full* resolution^3 grid
    represents (the same convention used when the grid's owning STL was
    voxelized), used only to report extent_mm for logging/bookkeeping -
    the actual placement math (extent_vox, center_of_gravity_vox) is
    voxel-index-only, which is what stays valid across separate,
    independently auto-fit voxelizations.

    `center`: "cog" (default, the historical behaviour) anchors the candidate at the target's centre of
    gravity; "bbox" at the middle of its bounding box. A candidate is resized to the bounding box, so only the
    bounding-box centre lets it fit that box exactly: for a lopsided target (a cube with a pocket, a base with an
    add-on) the centre of gravity is several voxels off, and the local search then has to move and grow the
    candidate to make up for it (measured: x1.078, 3 mm larger than the part).
    """
    if center not in ("cog", "bbox"):
        raise ValueError(f"center must be 'cog' or 'bbox', not {center!r}")
    occupied = np.argwhere(target_voxel)
    if occupied.size == 0:
        center = np.full(3, resolution / 2.0)
        return Placement(
            center_of_gravity_vox=center, bbox_min_vox=center.astype(int),
            bbox_max_vox=center.astype(int), extent_mm=np.zeros(3),
        )

    bbox_min = occupied.min(axis=0)
    bbox_max = occupied.max(axis=0)
    center_of_gravity = occupied.mean(axis=0) if center == "cog" else (bbox_min + bbox_max) / 2.0

    voxel_size_mm = scale_mm / resolution
    extent_mm = (bbox_max - bbox_min + 1).astype(np.float64) * voxel_size_mm

    return Placement(
        center_of_gravity_vox=center_of_gravity,
        bbox_min_vox=bbox_min, bbox_max_vox=bbox_max,
        extent_mm=extent_mm, anchor=center,
    )


def _crop_to_occupied(volume: np.ndarray) -> np.ndarray:
    occupied = np.argwhere(volume)
    if occupied.size == 0:
        return volume
    lo, hi = occupied.min(axis=0), occupied.max(axis=0)
    return volume[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]


def _resize_nearest(volume: np.ndarray, target_shape) -> np.ndarray:
    """Nearest-neighbor resize of a 3D array to `target_shape` (pure
    numpy - no scipy). Keeps a binary occupancy grid binary, unlike an
    interpolating resize."""
    src_shape = np.array(volume.shape)
    tgt_shape = np.maximum(np.array(target_shape), 1)
    idx = [
        np.clip((np.arange(t) * (s / t)).astype(int), 0, s - 1)
        for s, t in zip(src_shape, tgt_shape)
    ]
    return volume[np.ix_(idx[0], idx[1], idx[2])]


def place_candidate(candidate_voxel: np.ndarray, target: Placement, resolution: int) -> np.ndarray:
    """Reshape `candidate_voxel` (auto-fit to its own bounding box by
    stltovoxel, so it fills ~the whole grid regardless of the mesh's
    real build size) to `target`'s bounding-box footprint, and place it
    at `target`'s center of gravity within a fresh resolution^3 grid -
    the full placement correction, applied after the candidate is
    voxelized and before it's compared/subtracted against the target
    it's meant to explain.
    """
    crop = _crop_to_occupied(candidate_voxel)
    if crop.size == 0 or crop.sum() == 0:
        return np.zeros_like(candidate_voxel)

    resized = _resize_nearest(crop, target.extent_vox)

    out = np.zeros((resolution, resolution, resolution), dtype=candidate_voxel.dtype)
    # a bounding-box centre of an even-sized box sits on a half voxel: round it up, so the box lands exactly on the bbox
    center = (np.floor(target.center_of_gravity_vox + 0.5) if target.anchor == "bbox"
              else np.round(target.center_of_gravity_vox)).astype(int)
    half_lo = np.array(resized.shape) // 2
    half_hi = np.array(resized.shape) - half_lo
    start = center - half_lo
    stop = center + half_hi

    # Clip the placement to the grid, dropping whatever falls outside
    # (a target near the grid edge legitimately can't fit a same-sized
    # candidate fully inside it).
    src_start = np.clip(-start, 0, None)
    src_stop = np.array(resized.shape) - np.clip(stop - resolution, 0, None)
    dst_start = np.clip(start, 0, None)
    dst_stop = np.clip(stop, None, resolution)

    if np.any(src_stop <= src_start) or np.any(dst_stop <= dst_start):
        return out  # target placement is entirely off-grid

    out[dst_start[0]:dst_stop[0], dst_start[1]:dst_stop[1], dst_start[2]:dst_stop[2]] = \
        resized[src_start[0]:src_stop[0], src_start[1]:src_stop[1], src_start[2]:src_stop[2]]
    return out


def place_candidate_true_scale(candidate_voxel: np.ndarray, target: Placement, resolution: int, voxel_size_mm: float,
                               center=None):
    """Place a candidate that is ALREADY at its true size (built with measured dimensions and voxelized in the in-scale
    mode) on `target` WITHOUT resizing it: only translated so that the middle of its bounding box lies on the middle of
    the target's bounding box. Returns (placed voxel grid, the Placement of the candidate's own footprint), so that
    everything downstream that reads a footprint (the search's scale, the assembly's extents) sees the footprint the
    candidate really has.

    `place_candidate` instead stretches the candidate's bounding-box crop to the target's bounding box. That is right for
    a candidate of unknown size (the old Dimension CNN, stretched voxelizer mode) but distorts one of known size: a box
    that is a few degrees off has a bounding box that is larger than it and a fill below one, and stretching it into the
    target's box leaves the corners empty (measured on the pocket part: fill 0.85, recall 0.89), which the local search
    then repairs by scaling the candidate up by 8 %.
    """
    occupied = np.argwhere(candidate_voxel)
    if occupied.size == 0:
        return np.zeros_like(candidate_voxel), target
    lo, hi = occupied.min(axis=0), occupied.max(axis=0)
    crop = candidate_voxel[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1]
    extent = np.array(crop.shape)
    target_center = (target.bbox_min_vox + target.bbox_max_vox) / 2.0 if center is None else np.asarray(center, dtype=np.float64)
    start = np.floor(target_center - (extent - 1) / 2.0 + 0.5).astype(int)
    stop = start + extent
    out = np.zeros((resolution,) * 3, dtype=candidate_voxel.dtype)
    src_start = np.clip(-start, 0, None)
    src_stop = extent - np.clip(stop - resolution, 0, None)
    dst_start = np.clip(start, 0, None)
    dst_stop = np.clip(stop, None, resolution)
    if np.any(src_stop <= src_start) or np.any(dst_stop <= dst_start):
        return out, target
    out[dst_start[0]:dst_stop[0], dst_start[1]:dst_stop[1], dst_start[2]:dst_stop[2]] = \
        crop[src_start[0]:src_stop[0], src_start[1]:src_stop[1], src_start[2]:src_stop[2]]
    own = Placement(center_of_gravity_vox=start + (extent - 1) / 2.0, bbox_min_vox=start, bbox_max_vox=stop - 1,
                    extent_mm=extent * voxel_size_mm, anchor="bbox")
    return out, own
