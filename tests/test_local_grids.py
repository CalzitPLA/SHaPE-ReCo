"""Thickness-adaptive parity grids (dataset_gen/src/local_grids.py): the slab-wise Betti numbers against a dense
reference, known solids, and the thickness/pitch rule on a thin plate."""

import itertools

import numpy as np
import pytest
import trimesh
from scipy import ndimage

from dataset_gen.src.field_channels import parity_axes, parity_occupancy, grid_spec
from dataset_gen.src.closest_point_field import SurfaceField
from dataset_gen.src.local_grids import (adaptive_pitch, box_grid, grid_topology, iter_slabs, occupancy_topology,
                                          wall_thickness)


def _dense_euler(occ):
    """chi of the closed-cube complex by counting every cell (independent of the vertex-window table)."""
    o = np.pad(occ.astype(bool), 1)
    n = o.shape
    big = np.pad(o, 1)                    # voxel index i of o sits at i + 1 in big
    counts = [0, 0, 0, 0]
    for axes in itertools.product((0, 1), repeat=3):
        # a cell spans the axes flagged 1 (one voxel interval there) and sits on a lattice plane along the others,
        # where the voxels on both sides are incident; it exists if any incident voxel is set
        options = [[slice(1, n[d] + 1)] if axes[d] else [slice(0, n[d] + 1), slice(1, n[d] + 2)] for d in range(3)]
        acc = None
        for sl in itertools.product(*options):
            v = big[sl]
            acc = v.copy() if acc is None else acc | v
        counts[sum(axes)] += int(acc.sum())
    return counts[0] - counts[1] + counts[2] - counts[3]


def _dense_betti(occ):
    b0 = ndimage.label(occ, structure=np.ones((3, 3, 3)))[1]
    lab, nc = ndimage.label(~np.pad(occ, 1), structure=ndimage.generate_binary_structure(3, 1))
    b2 = nc - 1
    chi = _dense_euler(occ)
    return b0, b0 + b2 - chi, b2, chi


@pytest.mark.parametrize("seed", range(6))
def test_slab_topology_matches_dense_reference(seed):
    rng = np.random.default_rng(seed)
    occ = ndimage.binary_opening(rng.random((14, 23, 11)) < 0.55)
    t = occupancy_topology(occ, slab=5)        # several slab boundaries
    assert (t.b0, t.b1, t.b2, t.chi) == _dense_betti(occ)
    assert t.voxels == int(occ.sum())


def test_known_solids():
    # a square ring (genus 1), a hollow cube (one cavity), two separate bars
    ring = np.zeros((9, 9, 3), bool); ring[1:8, 1:8, :] = True; ring[3:6, 3:6, :] = False
    shell = np.ones((7, 7, 7), bool); shell[2:5, 2:5, 2:5] = False
    bars = np.zeros((5, 12, 5), bool); bars[1, :, 1] = True; bars[3, :, 3] = True
    for occ, want in [(ring, (1, 1, 0)), (shell, (1, 0, 1)), (bars, (2, 0, 0))]:
        t = occupancy_topology(occ, slab=4)
        assert (t.b0, t.b1, t.b2) == want


def test_parity_axes_agrees_with_parity_occupancy():
    m = trimesh.creation.box((3.0, 5.0, 2.0))
    m.apply_translation((0.7, -0.4, 0.3))
    f = SurfaceField(m.vertices, m.faces)
    spec = grid_spec(-4, 4, 20)
    c = spec.centres()
    xs, ys, zs = c[:, 0, 0, 0], c[0, :, 0, 1] + 1.2345e-4 * spec.pitch, c[0, 0, :, 2] + 2.3456e-4 * spec.pitch
    assert np.array_equal(parity_axes(m.vertices, m.faces, xs, ys, zs), parity_occupancy(f, spec))


def test_thin_plate_needs_the_adaptive_pitch():
    """A 0.3 thick, 10 x 10 plate, off-centre: thickness is measured as 0.3, and at a third of it the parity grid has the
    plate's volume and one component; at 10 x the pitch it has (almost) nothing."""
    plate = trimesh.creation.box((10.0, 10.0, 0.3))
    plate.apply_translation((1.3, -2.1, 0.77))
    _, _, tau = wall_thickness(plate.vertices, plate.faces, n=2000)
    assert np.percentile(tau[np.isfinite(tau)], 5) == pytest.approx(0.3, rel=1e-3)
    h = adaptive_pitch(tau)
    assert h == pytest.approx(0.1, rel=1e-3)
    lo, hi = plate.bounds
    fine = grid_topology(plate.vertices, plate.faces, lo, hi, h, slab=16)
    assert (fine.b0, fine.b1, fine.b2) == (1, 0, 0)
    assert fine.volume == pytest.approx(plate.volume, rel=0.1)
    coarse = grid_topology(plate.vertices, plate.faces, lo, hi, 10 * h)
    assert coarse.volume < 0.5 * plate.volume or coarse.volume > 1.5 * plate.volume


def test_crop_restricts_to_the_box():
    """Off-centre, asymmetric bar; a crop over its left half keeps only that half (voxel centres inside the crop)."""
    bar = trimesh.creation.box((6.0, 1.0, 2.0))
    bar.apply_translation((2.0, 0.5, -1.0))
    lo, hi = bar.bounds
    mid = np.array([2.0, hi[1], hi[2]])
    grid = box_grid(lo, mid, 0.25, crop=True)
    occ = np.concatenate(list(iter_slabs(bar.vertices, bar.faces, grid, slab=3)), axis=1)
    idx = np.argwhere(occ)
    centres = np.array(grid.lo) + (idx + 0.5) * grid.pitch
    assert centres[:, 0].min() > lo[0] and centres[:, 0].max() < 2.0
    assert occ.sum() * grid.pitch ** 3 == pytest.approx(bar.volume / 2, rel=0.05)


def test_wall_thickness_igl_and_trimesh_agree_and_are_exact(monkeypatch):
    """libigl's AABB rays (the default) and the trimesh fallback give the same thickness, the plate's own 0.2 on its
    big faces - on an off-centre, rotated plate (not at the origin, not axis-aligned)."""
    pytest.importorskip("igl")
    import sys as _sys
    import trimesh
    plate = trimesh.creation.box(extents=(4.0, 0.2, 2.5))
    plate.apply_transform(trimesh.transformations.rotation_matrix(0.7, (0.3, 0.2, 0.9)))
    plate.apply_translation((7.0, -3.0, 11.0))
    pts, nrm, tau_igl = wall_thickness(plate.vertices, plate.faces, n=1500)
    monkeypatch.setitem(_sys.modules, "igl", None)                       # the fallback
    pts2, _, tau_tm = wall_thickness(plate.vertices, plate.faces, n=1500)
    assert np.array_equal(pts, pts2)
    assert np.allclose(tau_igl, tau_tm, rtol=0, atol=1e-7)                # measured: <= 4.8e-8 (libigl's ray precision)
    assert abs(np.percentile(tau_igl, 5) - 0.2) < 1e-7
