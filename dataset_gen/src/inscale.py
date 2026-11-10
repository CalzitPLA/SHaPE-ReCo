#!/usr/bin/env python3
"""
In-scale training samples: one shape per file, but with a random SIZE and POSITION on top of the
parameters and rotation, in a grid that is only partly populated.

The default samples (--voxel-scale-mode fit) stretch every shape's bounding box onto the whole grid, so
a Type CNN can tell box from sphere by how much of the grid is filled (measured: below 40 % fill every
slice is "box", above 65 % ~90 % are "sphere"), a position or size can never be learned, and a partial
residual - what the reconstruction loop actually feeds the networks - looks nothing like a training grid.

Here (--voxel-scale-mode fixed) the STL a macro built is uniformly scaled so that its LARGEST EXTENT is a
random fraction of the grid, moved to a random position that keeps it inside the grid, and voxelized at a
fixed physical voxel size with the world origin at the grid centre (dataset_gen/src/voxelization.py,
anchor="origin"). Proportions, absolute size within the grid and position all survive. What was drawn
is recorded per sample, so a size or position label exists for a later network.

The rest of the module edits VOXEL ARRAYS, not meshes: `apply_cuts`, `carve`, `hollow` (and `manifold_report`,
`thickness`). dataset_generator.py without --with-field applies them to the voxelised grid, so such a .bin is not
the ray parity of any mesh and gets no <stem>.inside.npy (readers then fall back to .clean.bin or the grid itself);
with --with-field the drawn modification is then built as the modified FreeCAD solid (dataset_generator.
_exact_modified_solid, solid_builder), whose voxelisation replaces the edited array, and the inside bits and field come
from that solid. Which voxelizer that is depends on SHAPE_RECO_VOXELIZER (voxelization.py; the default is still
stltovoxel, task T-19).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


def sample_placement(rng, extents_mm: Sequence[float], grid_mm: float, size_min: float, size_max: float,
                     position: str = "random", resolution: int = 64) -> Dict:
    """Draw a size fraction and a position for a shape whose bounding box has `extents_mm` (x, y, z).

    size_fraction: the largest extent as a fraction of the grid, uniform in [size_min, size_max].
    position "random": the centre is uniform over the range that keeps the whole box inside the grid;
    "centered": the origin. `rng` is the generator's random.Random, so a seeded run is reproducible.
    The record also carries the voxel-frame centre and extent in the ARRAY's convention [z, y, x] with
    y reversed (see reco_engine.placement.array_to_world_position)."""
    if not 0 < size_min <= size_max <= 1.0:
        raise ValueError(f"size fractions must satisfy 0 < min <= max <= 1, got {size_min}, {size_max}")
    largest = float(max(extents_mm))
    if largest <= 0:
        raise ValueError("degenerate shape (zero extent)")
    fraction = rng.uniform(size_min, size_max)
    k = fraction * grid_mm / largest
    extent = [float(e) * k for e in extents_mm]
    if position == "random":
        center = [rng.uniform(-(grid_mm / 2 - e / 2), grid_mm / 2 - e / 2) for e in extent]
    elif position == "centered":
        center = [0.0, 0.0, 0.0]
    else:
        raise ValueError(f"position must be 'random' or 'centered', got {position!r}")
    v = grid_mm / resolution
    return dict(
        mode="fixed", grid_mm=float(grid_mm), size_fraction=fraction, scale_factor=k,
        center_mm=center, extent_mm=extent,
        center_vox=[resolution / 2 + center[2] / v, resolution / 2 - center[1] / v, resolution / 2 + center[0] / v],
        extent_vox=[extent[2] / v, extent[1] / v, extent[0] / v],
    )


# ---------------------------------------------------------------------------- cuts

def sample_cuts(rng, max_cuts: int = 2, keep_min: float = 0.35, keep_max: float = 0.95,
                axis_aligned_prob: float = 0.5, second_cut_prob: float = 0.35) -> List[Dict]:
    """One or two random planes to cut a shape with. Each: a unit normal in the voxel ARRAY's frame [a0, a1, a2]
    (axis-aligned with probability `axis_aligned_prob` - like the probe's caps - otherwise uniform on the
    sphere) and `keep`, the fraction of the shape's extent ALONG THAT NORMAL that is kept (from the
    low end). keep_min stays well above 0: a sphere cut to a sliver is a disc, whose label would be a
    coin flip. A second cut makes slabs and wedges."""
    cuts = []
    for i in range(max_cuts):
        if i > 0 and rng.random() >= second_cut_prob:
            break
        if rng.random() < axis_aligned_prob:
            normal = [0.0, 0.0, 0.0]
            normal[rng.randrange(3)] = rng.choice((-1.0, 1.0))
        else:
            while True:
                v = [rng.gauss(0.0, 1.0) for _ in range(3)]
                n = sum(x * x for x in v) ** 0.5
                if n > 1e-6:
                    normal = [x / n for x in v]
                    break
        cuts.append(dict(normal=normal, keep=rng.uniform(keep_min, keep_max)))
    return cuts


def thickness(volume: np.ndarray) -> float:
    """Extent, in voxels, of the occupied voxels along their thinnest principal axis (PCA of the voxel
    centres, extent = max - min + 1). A flat plate - a triangle or disc lying in one plane, however it is
    tilted - is about 1-2 voxels; a solid body is as thick as it is deep."""
    pts = np.argwhere(volume > 0).astype(np.float64)
    if len(pts) < 4:
        return 0.0
    centred = pts - pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(centred.T))
    along = centred @ vecs[:, 0]                                           # eigh: ascending, [0] = thinnest axis
    return float(along.max() - along.min() + 1.0)


def apply_cuts(volume: np.ndarray, cuts: List[Dict], min_voxels: int = 300, min_thickness: float = 4.0):
    """Remove the part of the shape beyond each cut plane: a voxel stays if it lies within the lowest `keep`
    fraction of the shape's current extent along that normal. The extent is measured between voxel EDGES
    (centre-to-centre would be one voxel short and make shallow caps too tall: +22 % volume at keep 0.2), using
    the width a voxel projects onto the normal. A cut is skipped when it would leave fewer than `min_voxels`
    voxels, or when it would flatten a solid body into a plate (thinnest principal extent below
    `min_thickness` voxels, see `thickness`): a box sliced to one layer, or a wedge shaved to a triangle in
    one plane, is a 2-D figure without the character of its base shape, and its label would be a coin flip.
    A body that is already thinner than that (a very elongated box) is left to the cuts. Returns
    (volume, the cuts that were applied)."""
    volume = volume.copy()
    applied = []
    grid = np.indices(volume.shape).reshape(3, -1).T.astype(np.float64)               # voxel indices, [a0, a1, a2]
    for cut in cuts:
        occupied = volume.reshape(-1) > 0
        if not occupied.any():
            break
        normal = np.asarray(cut["normal"], dtype=np.float64)
        width = float(np.abs(normal).sum())                                # projected width of one voxel
        along = grid @ normal
        lo, hi = along[occupied].min() - width / 2, along[occupied].max() + width / 2     # edge to edge
        beyond = along > lo + cut["keep"] * (hi - lo) + 1e-9
        if int((occupied & ~beyond).sum()) < min_voxels:
            continue
        trial = volume.copy()
        trial.reshape(-1)[occupied & beyond] = 0
        trial, whole = keep_main_piece(trial)                               # staircase crumbs of an oblique cut
        if not whole or int((trial > 0).sum()) < min_voxels:
            continue
        if min_thickness > 0 and thickness(trial) < min_thickness <= thickness(volume):
            continue
        volume = trial
        applied.append(dict(cut))
    return volume, applied


# ---------------------------------------------------------------------------- one uniform manifold

def manifold_report(volume: np.ndarray) -> Dict:
    """Is the occupied part ONE connected solid without enclosed voids? {"components": n, "cavities": m, "ok": bool}.
    components = 6-connected pieces of material; cavities = pieces of empty space that do not touch the outside
    (empty space labelled in a grid padded by one empty layer, so the outside is one piece). A part that is
    one uniform manifold has 1 component and 0 cavities: no floating islands, no closed bubbles."""
    from scipy import ndimage
    occ = volume > 0
    _, components = ndimage.label(occ)
    padded = np.pad(~occ, 1, constant_values=True)
    _, empty_pieces = ndimage.label(padded)
    cavities = max(int(empty_pieces) - 1, 0)
    return {"components": int(components), "cavities": cavities, "ok": bool(components == 1 and cavities == 0)}


def keep_main_piece(volume: np.ndarray, max_crumb_fraction: float = 0.02):
    """(volume with only its largest 6-connected piece, ok). A slicing plane that grazes a corner leaves a few
    staircase voxels that touch only along an edge: not real geometry but separate pieces to the manifold test.
    They are dropped when they hold at most `max_crumb_fraction` of the voxels (ok=True); when more material
    than that is split off the body really broke in two and ok is False."""
    from scipy import ndimage
    occ = volume > 0
    labels, n = ndimage.label(occ)
    if n <= 1:
        return volume, True
    sizes = ndimage.sum(occ, labels, index=range(1, n + 1))
    main = labels == (1 + int(np.argmax(sizes)))
    crumbs = int(occ.sum() - main.sum())
    return main.astype(volume.dtype), crumbs <= max_crumb_fraction * occ.sum()


def _random_rotation(rng) -> np.ndarray:
    q = np.array([rng.gauss(0.0, 1.0) for _ in range(4)])
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def carve(volume: np.ndarray, rng, cutter: str, size: Sequence[float] = (0.18, 0.42),
          removed: Sequence[float] = (0.04, 0.5), min_voxels: int = 300, min_thickness: float = 4.0,
          attempts: int = 60):
    """Subtract ONE cutter from the body: a sphere from a box ("box - sphere") or a box from a sphere
    ("sphere - box"). The cutter is centred near a random point of the body's surface, so it always bites into
    the surface (a pocket or notch, never a floating or enclosed cutter). `size` is the cutter's radius / half
    edge as a fraction of the body's largest extent; a box cutter has a random orientation.

    A draw is accepted only if the result is ONE uniform manifold (`manifold_report`: one connected piece, no
    enclosed void), removes a fraction within `removed` of the body, keeps at least `min_voxels` voxels and is not
    flattened into a plate (`thickness`). Returns (volume, record) or (None, None) when no draw within `attempts`
    qualifies - the caller then keeps the body uncarved. The label stays the base shape."""
    if cutter not in ("sphere", "box"):
        raise ValueError(f"cutter must be 'sphere' or 'box', not {cutter!r}")
    occ = volume > 0
    total = int(occ.sum())
    if total < min_voxels:
        return None, None
    from scipy import ndimage
    surface = np.argwhere(occ & ~ndimage.binary_erosion(occ))
    if len(surface) == 0:
        return None, None
    extent = float((np.argwhere(occ).max(axis=0) - np.argwhere(occ).min(axis=0) + 1).max())
    centroid = np.argwhere(occ).mean(axis=0)
    grid = np.indices(volume.shape).reshape(3, -1).T.astype(np.float64)
    for attempt in range(1, attempts + 1):
        point = surface[rng.randrange(len(surface))].astype(np.float64)
        outward = point - centroid
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
        radius = rng.uniform(*size) * extent
        centre = point + outward * rng.uniform(-0.35, 0.25) * radius        # a little inside .. a little outside the surface
        if cutter == "sphere":
            hit = (((grid - centre) ** 2).sum(axis=1) <= radius ** 2).reshape(volume.shape)
            record = dict(cutter="sphere", centre=[float(c) for c in centre], radius=float(radius))
        else:
            rot = _random_rotation(rng)
            hit = (np.abs((grid - centre) @ rot) <= radius).all(axis=1).reshape(volume.shape)
            record = dict(cutter="box", centre=[float(c) for c in centre], half_edge=float(radius), rotation=rot.tolist())
        result = occ & ~hit
        kept = int(result.sum())
        fraction = 1.0 - kept / total
        if kept < min_voxels or not (removed[0] <= fraction <= removed[1]):
            continue
        if not manifold_report(result)["ok"]:
            continue
        if min_thickness > 0 and thickness(result.astype(np.uint8)) < min_thickness:
            continue
        record.update(removed_fraction=float(fraction), attempts=attempt)
        return result.astype(np.uint8), record
    return None, None


def hollow(volume: np.ndarray, rng, wall_fraction: Sequence[float] = (0.08, 0.3), min_voxels: int = 300,
          min_wall_voxels: int = 2, attempts: int = 20):
    """Empty out the body's interior, leaving a shell: a box becomes a hollow box, a sphere a hollow ball (or a sphere with an
    off-centre cavity, when the erosion below happens to be asymmetric near a corner - both are valid hollow bodies). Shape-
    agnostic: eroding the occupancy by `wall` voxels approximates a uniform-thickness inner boundary for ANY solid, box or
    sphere alike, without needing to know which one it is. The base shape reconstructs this as base (the outer shell's
    silhouette) + cut (the inner cavity) - the opposite of `carve()`: a hollow body's whole point is the closed cavity, so
    `manifold_report` must show exactly one solid piece and exactly one cavity, not zero.

    `wall_fraction` is the wall thickness as a fraction of the body's half-extent. Returns (volume, record) or (None, None)
    when no draw within `attempts` leaves a valid shell (min_voxels kept, one piece, one cavity) - the caller then keeps the
    body solid."""
    occ = volume > 0
    total = int(occ.sum())
    if total < min_voxels:
        return None, None
    from scipy import ndimage
    extent = float((np.argwhere(occ).max(axis=0) - np.argwhere(occ).min(axis=0) + 1).max())
    for attempt in range(1, attempts + 1):
        wall = rng.uniform(*wall_fraction) * extent / 2.0
        wall_voxels = max(int(round(wall)), min_wall_voxels)
        eroded = ndimage.binary_erosion(occ, iterations=wall_voxels)
        if not eroded.any():
            continue
        shell = occ & ~eroded
        kept = int(shell.sum())
        if kept < min_voxels:
            continue
        report = manifold_report(shell.astype(np.uint8))
        if report["components"] != 1 or report["cavities"] != 1:
            continue
        record = dict(wall_voxels=wall_voxels, wall_fraction=float(2 * wall_voxels / extent),
                      kept_fraction=float(kept / total), attempts=attempt)
        return shell.astype(volume.dtype), record
    return None, None


def write_placed_stl(src: Path, dst: Path, placement: Dict, bbox_center: Sequence[float]) -> None:
    """`src` scaled by placement["scale_factor"] about its bounding-box centre and moved to
    placement["center_mm"]; written to `dst` (binary STL)."""
    from stl import mesh as stl_mesh
    source = stl_mesh.Mesh.from_file(str(src))
    placed = stl_mesh.Mesh(np.zeros(len(source.vectors), dtype=stl_mesh.Mesh.dtype))
    placed.vectors[:] = (source.vectors - np.asarray(bbox_center)) * placement["scale_factor"] \
        + np.asarray(placement["center_mm"])
    placed.save(str(dst))


def bounds_center_extents(stl_path: Path):
    from stl import mesh as stl_mesh
    vertices = stl_mesh.Mesh.from_file(str(stl_path)).vectors.reshape(-1, 3)
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    return (lo + hi) / 2, hi - lo
