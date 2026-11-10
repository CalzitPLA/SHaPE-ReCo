"""reco_field/fit.py's fit_box_bounding and reco_field/peel.py's bounding_box/invented_slack parameters
(docs/FRAME_FAILURE.md, "rethinking the acceptance gate"): a box that COVERS every point of a cluster
instead of the tightest fit to most of them, opt-in and off by default - the default peel()/peel_steps()
behaviour must not change at all when bounding_box is left False (checked directly here and, more broadly,
against all 41 example parts in docs/data/peel_examples.json, unaffected by this feature)."""
import sys
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.fit import fit_box, fit_box_bounding, make_samples  # noqa: E402
from reco_field.peel import peel  # noqa: E402
from src.closest_point_field import SurfaceField, load_field  # noqa: E402


def posed_box(extents=(12.0, 7.0, 4.0), seed=0):
    mesh = trimesh.creation.box(extents=extents)
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    x, y, z, w = q / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    t = rng.uniform(-5, 5, 3)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    mesh = mesh.copy()
    mesh.apply_transform(T)
    return SurfaceField(mesh.vertices, mesh.faces), R, t


def test_fit_box_bounding_recovers_a_clean_box_like_fit_box_does():
    field, R, t = posed_box(seed=1)
    s = make_samples(field)
    r = fit_box_bounding(s)
    assert r.kind == "box" and r.explained > 0.99
    assert np.allclose(sorted(r.half), [2.0, 3.5, 6.0], atol=0.1)
    assert np.allclose(r.centre, t, atol=0.1)


def test_fit_box_bounding_never_excludes_any_point_of_its_own_cluster():
    """The defining property: every surface point must lie within the box (up to floating point), since a
    bounding box is defined as containing all of them - unlike a converged (Tukey-IRLS) fit, which may treat
    some as outliers and shrink away from them."""
    field, R, t = posed_box(seed=2)
    s = make_samples(field)
    r = fit_box_bounding(s)
    local = (s.surface - r.centre) @ r.frame
    assert np.all(np.abs(local) <= np.asarray(r.half) + 1e-6)


def test_fit_box_bounding_with_an_orientation_hint_uses_it_directly():
    field, R, t = posed_box(seed=3)
    s = make_samples(field)
    identity = np.eye(3)
    r = fit_box_bounding(s, orientation_hint=identity)
    assert np.allclose(r.frame, identity)


def test_bounding_box_off_by_default_does_not_change_peel_result():
    """The exact property the 41-example regression check (docs/data/peel_examples.json) verifies at scale:
    leaving `bounding_box` at its default False must be identical to not having the parameter at all."""
    field, R, t = posed_box(extents=(10.0, 6.0, 3.0), seed=4)
    default = peel(field, max_ops=6)
    explicit_off = peel(field, max_ops=6, bounding_box=False)
    assert default.explained == explicit_off.explained
    assert [o.fit.kind for o in default.ops] == [o.fit.kind for o in explicit_off.ops]


def test_bounding_box_mode_only_substitutes_geometry_not_kind_selection():
    """A regression this feature could easily cause: if the bounding box's own (structurally lower) explained
    score were allowed to influence which KIND wins fit_best's ranking, a worse kind could win outright (this
    was caught during development - see the plan's write-up). The base kind must stay "box" here regardless."""
    field, R, t = posed_box(extents=(10.0, 6.0, 3.0), seed=5)
    result = peel(field, max_ops=1, bounding_box=True)
    assert result.ops[0].fit.kind == "box"


def test_bounding_box_mode_with_a_wide_invented_slack_still_explains_the_box_well():
    field, R, t = posed_box(extents=(10.0, 6.0, 3.0), seed=6)
    result = peel(field, max_ops=6, bounding_box=True, invented_slack=0.3)
    assert result.explained > 0.95
