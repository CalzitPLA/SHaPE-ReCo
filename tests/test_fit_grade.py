"""Fit grading (reco_engine.fit_grade): tolerance-aware error, defects, surface distances, the assembly-level Grader."""
import numpy as np
import pytest

from reco_engine.fit_grade import GradeConfig, Grader, assembly_error, grade, surface_distances, tolerant_sets

N = 48


def block(x, y, z):
    g = np.zeros((N, N, N), np.uint8)
    g[x[0]:x[1], y[0]:y[1], z[0]:z[1]] = 1
    return g


def hemisphere(cx, cy, cz, r):
    idx = np.indices((N, N, N)).astype(float)
    return (((idx[0] - cx) ** 2 + (idx[1] - cy) ** 2 + (idx[2] - cz) ** 2) <= r * r) & (idx[2] <= cz)


def test_identical_shapes_have_no_error_and_no_defects():
    a = block((8, 40), (8, 36), (8, 30))
    g = grade(a, a)
    assert g.iou == 1.0 and g.error == 0 and g.fn_voxels == 0 and g.fp_voxels == 0 and not g.defects
    assert g.chamfer == 0 and g.fscore == 1.0


def test_a_one_voxel_offset_is_within_tolerance_but_lowers_iou():
    a = block((8, 40), (8, 36), (8, 30))
    b = block((9, 41), (8, 36), (8, 30))                      # shifted by one voxel along x
    g = grade(b, a)
    assert g.iou < 0.96 and g.error == 0 and not g.defects   # IoU punishes the resolution, the tolerant error does not
    far = grade(block((12, 44), (8, 36), (8, 30)), a)        # four voxels
    assert far.error > 0.1 and {d.kind for d in far.defects} == {"missing", "spurious"}


def test_tolerance_is_a_parameter():
    a = block((8, 40), (8, 36), (8, 30))
    b = block((11, 43), (8, 36), (8, 30))                     # three voxels
    assert grade(b, a, GradeConfig(tol_voxels=1)).error > 0
    assert grade(b, a, GradeConfig(tol_voxels=3.5)).error == 0


def test_a_missing_pocket_is_a_spurious_defect_of_the_pockets_size():
    ref = block((8, 40), (8, 40), (8, 40)).astype(bool)
    pocket = hemisphere(24, 24, 39, 10) & ref
    ref_with_pocket = ref & ~pocket
    g = grade(ref.astype(np.uint8), ref_with_pocket.astype(np.uint8), GradeConfig(voxel_mm=0.625))
    assert not g.missing and len(g.spurious) == 1
    d = g.spurious[0]
    assert 0.5 * pocket.sum() < d.voxels <= pocket.sum()       # the pocket minus its tolerance skin
    assert d.volume_mm3 == pytest.approx(d.voxels * 0.625 ** 3)
    assert g.error > 0.02 and "spurious" in g.summary()


def test_a_missing_addon_is_a_missing_defect():
    base = block((6, 42), (6, 42), (6, 20))
    addon = block((16, 32), (16, 32), (20, 34))
    g = grade(base, base | addon)
    assert len(g.missing) == 1 and not g.spurious
    assert g.missing[0].voxels > 0.6 * addon.sum() and g.recall < 1.0


def test_thin_boundary_layers_are_not_defects():
    a = block((8, 40), (8, 36), (8, 30))
    fat = block((7, 41), (7, 37), (7, 31))                     # one voxel bigger all round
    g = grade(fat, a)
    assert g.error == 0 and not g.defects and g.precision < 1.0


def test_surface_distances_and_fscore_grow_with_the_offset():
    a = block((8, 40), (8, 36), (8, 30))
    near, far = grade(block((9, 41), (8, 36), (8, 30)), a), grade(block((13, 45), (8, 36), (8, 30)), a)
    assert 0 < near.chamfer < far.chamfer and near.fscore > far.fscore and far.hd95 >= 4
    d1, d2 = surface_distances(a, a)
    assert d1.max() == 0 and d2.max() == 0
    empty = grade(np.zeros_like(a), a)
    assert empty.recall == 0 and empty.error >= 1.0 and empty.fscore == 0.0


def test_voxel_mm_gives_millimetres():
    a = block((8, 40), (8, 36), (8, 30))
    g = grade(block((13, 45), (8, 36), (8, 30)), a, GradeConfig(voxel_mm=0.5))
    assert g.chamfer_mm == pytest.approx(0.5 * g.chamfer) and g.hd95_mm == pytest.approx(0.5 * g.hd95)


def test_tolerant_sets_and_assembly_error_agree():
    a = block((8, 40), (8, 36), (8, 30))
    b = block((12, 44), (8, 36), (8, 30))
    fn, fp = tolerant_sets(b, a, 1.0)
    assert assembly_error(b, a) == pytest.approx((fn.sum() + fp.sum()) / a.sum())


def test_grader_in_cover_mode_prefers_the_clean_candidate_and_charges_for_redundancy():
    ref = block((6, 42), (6, 42), (6, 20)) | block((16, 32), (16, 32), (20, 34))
    grader = Grader(ref, np.zeros_like(ref), mode="add")
    base = block((6, 42), (6, 42), (6, 20))
    bbox = block((6, 42), (6, 42), (6, 34))                    # the over-covering bounding box
    err_base, gain_base = grader.evaluate(base)
    err_bbox, gain_bbox = grader.evaluate(bbox)
    assert gain_base > gain_bbox and gain_base > 0
    grader.accept(base)
    assert grader.error == pytest.approx(err_base)
    _, gain_dup = grader.evaluate(base)                        # the same candidate again: nothing gained, one shape paid
    assert gain_dup == pytest.approx(-grader.shape_cost)
    _, gain_top = grader.evaluate(block((16, 32), (16, 32), (20, 34)))
    assert gain_top > 0.03


def test_grader_in_cut_mode_rewards_removing_spurious_material():
    ref = block((6, 42), (6, 42), (6, 30)).astype(bool) & ~hemisphere(24, 24, 29, 9)
    assembly = block((6, 42), (6, 42), (6, 30))
    grader = Grader(ref.astype(np.uint8), assembly, mode="subtract")
    void = hemisphere(24, 24, 29, 9).astype(np.uint8)
    wrong = block((6, 42), (6, 42), (6, 30))                   # a cutter that removes everything
    assert grader.evaluate(void)[1] > 0.02 > grader.evaluate(wrong)[1]
    with pytest.raises(ValueError):
        Grader(ref, assembly, mode="multiply")
