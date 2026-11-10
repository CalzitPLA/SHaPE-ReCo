"""
Tests for reco_engine.local_search: the bounded, deterministic
voxel-space position+scale refinement that is the actual positioning
authority in this version of the reconstruction architecture (CNNs and
placement.py only ever propose an initial position/scale - see
reco_engine/greedy_reconstruct.py's module docstring).

All synthetic - no trained CNN weights, no FreeCAD, no stltovoxel
anywhere in this file. Run with: pytest tests/test_local_search.py -v
"""

import numpy as np
import pytest

from reco_engine.local_search import LocalSearchConfig, refine_position

R = 32  # small grid keeps these tests fast


def _box(lo, hi, resolution=R):
    """An axis-aligned box occupancy grid, lo/hi as (x, y, z) tuples,
    hi exclusive - the simplest possible synthetic target/candidate."""
    v = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    v[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    return v


TARGET = _box((10, 10, 10), (20, 18, 16))  # 10 x 8 x 6


# 1. Candidate already correctly positioned.
def test_already_correctly_positioned():
    candidate = _box((10, 10, 10), (20, 18, 16))
    result = refine_position(TARGET, candidate)
    assert result.score == pytest.approx(1.0)
    assert result.position == (0, 0, 0)
    assert result.scale == pytest.approx(1.0)


# 2-7. Candidate displaced along each axis, search recovers full fit.
# Position+scale search may reach recall=1.0 via a different combined
# path than a translation-only search would (e.g. a partial shift plus
# a small grow), so this checks the outcome that actually matters -
# full target coverage - rather than pinning one specific translation
# vector out of several valid ones.
@pytest.mark.parametrize("axis_offset", [
    (6, 0, 0), (-6, 0, 0),
    (0, 6, 0), (0, -6, 0),
    (0, 0, 6), (0, 0, -6),
])
def test_displaced_candidate_recovers(axis_offset):
    dx, dy, dz = axis_offset
    candidate = _box((10 + dx, 10 + dy, 10 + dz), (20 + dx, 18 + dy, 16 + dz))
    result = refine_position(TARGET, candidate)
    assert result.score == pytest.approx(1.0), "a same-size candidate displaced by a reachable amount should fully recover"
    assert result.metrics.recall == pytest.approx(1.0)


# 8. Candidate oversized but correctly covering the target - not penalized.
def test_oversized_candidate_not_penalized():
    candidate = _box((8, 8, 8), (22, 20, 18))  # strictly contains TARGET
    result = refine_position(TARGET, candidate)
    assert result.metrics.recall == pytest.approx(1.0), "fit = recall must not penalize extra coverage"
    assert result.metrics.iou < 1.0, "IoU *would* penalize it - confirms recall and IoU genuinely differ here"


# 9. Candidate with a poor (but not hopeless - partial initial overlap)
# initial position that local search can recover from.
def test_poor_initial_position_with_partial_overlap_recovers():
    # Overlaps TARGET by a small corner rather than not at all - local
    # search needs *some* gradient signal to follow; zero-overlap start
    # positions are an inherent limit of any local (non-global) search,
    # not something this test claims to solve (see module docstring).
    candidate = _box((16, 14, 12), (26, 22, 18))
    result = refine_position(TARGET, candidate, LocalSearchConfig(initial_step=4, max_iterations=8))
    initial_recall = 0  # sanity: compute what recall was before any search, for comparison
    from reco_engine.compare import compare
    initial_recall = compare(TARGET, candidate).recall
    assert result.score > initial_recall, "search must improve on a recoverable poor start"


# 10. CUT / over-coverage target - local_search doesn't care what the
# target represents, but this exercises that explicitly since the
# CUT phase in greedy_reconstruct treats over-coverage exactly the
# same way as missing material.
def test_overcoverage_style_target():
    overcoverage_target = _box((5, 5, 5), (9, 9, 9))  # a small blob, as over-coverage typically is
    candidate = _box((5, 5, 5), (9, 9, 9))
    result = refine_position(overcoverage_target, candidate)
    assert result.score == pytest.approx(1.0)


# 11. Search stops at minimum step (translation) - step_history never
# goes below minimum_step once no scale search is left either.
def test_stops_at_minimum_step():
    candidate = _box((10, 10, 10), (20, 18, 16))
    # Scale search disabled by giving it a minimum above its (0.0) start,
    # so this isolates translation-step behavior specifically. (A
    # minimum_scale_step of exactly 0.0 would never disable it - 0.0 is
    # never "below" a minimum of 0.0 - which is itself worth knowing.)
    config = LocalSearchConfig(initial_step=4, minimum_step=1, initial_scale_step=0.0, minimum_scale_step=0.01)
    result = refine_position(TARGET, candidate, config)
    assert all(s >= config.minimum_step for s in result.step_history)
    assert result.converged


# 12. Search respects maximum evaluation/iteration bounds.
def test_respects_evaluation_bound():
    candidate = _box((0, 0, 0), (10, 8, 6))  # far off, will not converge
    config = LocalSearchConfig(max_evaluations=12, max_iterations=100)
    result = refine_position(TARGET, candidate, config)
    assert result.evaluations <= 12
    assert not result.converged


def test_respects_iteration_bound():
    candidate = _box((0, 0, 0), (10, 8, 6))
    config = LocalSearchConfig(max_iterations=2, max_evaluations=10_000)
    result = refine_position(TARGET, candidate, config)
    assert result.iterations <= 2


# 13. Same input produces the same result (determinism).
def test_deterministic():
    candidate = _box((16, 10, 10), (26, 18, 16))
    r1 = refine_position(TARGET, candidate)
    r2 = refine_position(TARGET, candidate)
    assert r1.position == r2.position
    assert r1.scale == r2.scale
    assert r1.score == r2.score
    assert r1.evaluations == r2.evaluations


# 14. Candidate voxelization happens once, not once per position
# hypothesis: refine_position never touches FreeCAD/stltovoxel at all -
# it only ever receives and returns numpy arrays. This test asserts
# that contract directly: no subprocess module is imported or used by
# local_search, and the same *object* passed in is never mutated (a
# real voxelization would produce a brand-new array each call).
def test_never_calls_subprocess():
    import reco_engine.local_search as local_search_module
    import inspect
    source = inspect.getsource(local_search_module)
    # Check actual usage, not just the word appearing in prose (this
    # module's own docstring explains *why* it avoids subprocess calls,
    # which would otherwise false-positive a naive "not in source" check).
    assert "import subprocess" not in source
    assert "subprocess.run" not in source
    assert "subprocess.Popen" not in source
    assert "voxelize_stl_to_bin" not in source  # the actual stltovoxel entry point elsewhere in this package

    candidate = _box((10, 10, 10), (20, 18, 16))
    original = candidate.copy()
    refine_position(TARGET, candidate)
    assert np.array_equal(candidate, original), "refine_position must not mutate its input in place"


def test_scale_search_shrinks_an_oversized_candidate_when_it_helps_recall():
    # A candidate much larger than the target: recall is already 1.0
    # (oversized is never penalized), so scale search should find no
    # reason to shrink - this documents that behavior explicitly, since
    # it's easy to assume "smaller is always better" incorrectly.
    candidate = _box((0, 0, 0), (32, 32, 32))
    result = refine_position(TARGET, candidate)
    assert result.metrics.recall == pytest.approx(1.0)
    assert result.scale == pytest.approx(1.0), "recall is already perfect; nothing should change"


# ---------------------------------------------------------------------
# Spill penalty: recall alone never stops a candidate from growing.
# ---------------------------------------------------------------------

def _sphere(radius, center, resolution):
    g = np.indices((resolution,) * 3)
    c = np.array(center)[:, None, None, None]
    return (((g - c) ** 2).sum(0) <= radius ** 2).astype(np.uint8)


def test_recall_only_search_runs_away_and_spill_penalty_stops_it():
    from reco_engine.local_search import spill_fraction

    r = 64
    target = _box((20, 20, 20), (44, 44, 44), r)
    sphere = _sphere(10, (32, 32, 32), r)  # cannot cover a cube's corners without growing a lot

    plain = refine_position(target, sphere)                      # recall only (the default)
    penalized = refine_position(target, sphere, spill_weight=1.0)

    assert spill_fraction(plain.voxel, target) > 0.25, "recall-only search should overshoot the target"
    assert spill_fraction(penalized.voxel, target) < 0.10
    assert penalized.metrics.precision > plain.metrics.precision
    assert penalized.scale < plain.scale


def test_spill_is_measured_against_allowed_not_target():
    """A cover primitive chasing a small residual may sit on material the
    input already has: spill is judged against the whole input."""
    r = 64
    whole_input = _box((20, 20, 20), (44, 44, 44), r)
    residual = _box((32, 20, 20), (44, 44, 44), r)          # only half still to explain
    candidate = _box((20, 20, 20), (44, 44, 44), r)         # overlaps the covered half

    against_target = refine_position(residual, candidate, spill_weight=1.0)
    against_input = refine_position(residual, candidate, spill_weight=1.0, allowed=whole_input)
    assert against_input.score == pytest.approx(1.0), "covering the covered half is free"
    assert against_target.score < 1.0, "judged against the residual alone it is half spill"


def test_empty_candidate_counts_as_fully_spilled():
    from reco_engine.local_search import spill_fraction

    empty = np.zeros_like(TARGET)
    assert spill_fraction(empty, TARGET) == 1.0
    assert spill_fraction(TARGET, TARGET) == 0.0
    # pushing the candidate off the grid must never look like an improvement
    candidate = _box((26, 10, 10), (32, 18, 16))  # touching the grid edge
    result = refine_position(TARGET, candidate, spill_weight=3.0)
    assert result.voxel.sum() > 0


def test_default_is_still_recall_only():
    candidate = _box((12, 10, 10), (20, 18, 16))
    plain = refine_position(TARGET, candidate)
    explicit = refine_position(TARGET, candidate, allowed=TARGET, spill_weight=0.0)
    assert plain.score == explicit.score and plain.position == explicit.position
    assert plain.scale == explicit.scale


# ------------------------------------------------ the fast scorer equals the plain maths

def test_fast_scoring_is_identical_to_the_naive_formulas():
    """compare()/spill_fraction()/the search's _Scorer avoid redundant passes over the
    grid; they must not change a single number."""
    from reco_engine.compare import compare as fast_compare
    from reco_engine.local_search import _Scorer, spill_fraction

    rng = np.random.default_rng(0)
    for density_a, density_b in ((0.5, 0.5), (0.05, 0.9), (0.9, 0.05), (0.0, 0.5), (0.5, 0.0), (0.0, 0.0)):
        a = (rng.random((16, 16, 16)) < density_a).astype(np.uint8)
        b = (rng.random((16, 16, 16)) < density_b).astype(np.uint8)
        allowed = (rng.random((16, 16, 16)) < 0.6).astype(np.uint8)
        A, B, L = a.astype(bool), b.astype(bool), allowed.astype(bool)

        inter, union = np.logical_and(A, B).sum(), np.logical_or(A, B).sum()
        want = (float(inter / union) if union else 0.0,
                float(inter / B.sum()) if B.sum() else 0.0,
                float(inter / A.sum()) if A.sum() else 0.0)
        m = fast_compare(a, b)
        assert (m.iou, m.precision, m.recall) == want
        assert (fast_compare(A, B).iou, fast_compare(a, B).recall) == (want[0], want[2])   # dtype-independent

        want_spill = 1.0 if B.sum() == 0 else float(np.logical_and(B, ~L).sum()) / B.sum()
        assert spill_fraction(b, allowed) == want_spill
        metrics, score = _Scorer(a, allowed, 0.7)(b)
        assert (metrics.iou, metrics.precision, metrics.recall) == want
        assert score == want[2] - 0.7 * want_spill
