"""reco_field/peel.py's `verify_freecad` option and reco_field/export.py's `build_split_in_freecad` (docs/
FRAME_FAILURE.md, "Materializing every step in real FreeCAD"): every accepted step is actually built in FreeCAD
(real Part booleans via freecadcmd), not just the symbolic SDF - opt-in and off by default, so the ordinary
peel()/peel_steps() behaviour (and build_in_freecad's own, unsplit, biggest-only export) must not change at all
when it is left False."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_field.export import build_in_freecad, build_split_in_freecad, find_freecadcmd  # noqa: E402
from reco_field.peel import peel, peel_steps  # noqa: E402
from src import sdf_primitives as sp  # noqa: E402
from src.closest_point_field import SurfaceField  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found")


def posed_box(extents=(10.0, 6.0, 3.0), seed=0):
    mesh = trimesh.creation.box(extents=extents)
    rng = np.random.default_rng(seed)
    t = rng.uniform(-5, 5, 3)
    mesh = mesh.copy()
    mesh.apply_translation(t)
    return SurfaceField(mesh.vertices, mesh.faces)


@needs_freecad
def test_build_in_freecad_a_single_solid_is_unaffected_by_the_split_helper_existing():
    model = sp.Box((5.0, 4.0, 3.0))
    with tempfile.TemporaryDirectory() as d:
        ex = build_in_freecad(model, d, "onebox")
    assert ex.n_solids == 1 and ex.valid and ex.solids is None
    assert abs(ex.volume - 480.0) < 1e-6


@needs_freecad
def test_build_split_in_freecad_exports_every_solid_when_the_csg_actually_splits():
    model = sp.Union(sp.Placed(sp.Box((2, 2, 2)), translation=(-10, 0, 0)), sp.Placed(sp.Box((3, 3, 3)), translation=(10, 0, 0)))
    with tempfile.TemporaryDirectory() as d:
        ex = build_split_in_freecad(model, d, "twoboxes")
        assert ex.n_solids == 2 and ex.valid
        assert ex.solids is not None and len(ex.solids) == 2
        assert abs(ex.solids[0]["volume"] - 216.0) < 1e-6           # biggest first
        assert abs(ex.solids[1]["volume"] - 64.0) < 1e-6
        assert ex.solids[0]["stl"].is_file() and ex.solids[1]["step"].is_file()


@needs_freecad
def test_verify_freecad_off_by_default_does_not_change_peel_result():
    field = posed_box(seed=4)
    default = peel(field, max_ops=3)
    explicit_off = peel(field, max_ops=3, verify_freecad=False)
    assert default.explained == explicit_off.explained
    assert [o.fit.kind for o in default.ops] == [o.fit.kind for o in explicit_off.ops]


@needs_freecad
def test_verify_freecad_reports_the_base_as_one_valid_solid():
    field = posed_box(seed=5)
    steps = list(peel_steps(field, max_ops=2, verify_freecad=True))
    base = next(s for s in steps if s.stage == "base")
    assert base.freecad is not None and "error" not in base.freecad
    assert base.freecad["valid"] is True and base.freecad["n_solids"] == 1
    assert base.freecad["leftover_volumes"] == []


def test_verify_freecad_off_leaves_the_freecad_field_none():
    """No `needs_freecad` mark: this must hold even without FreeCAD installed - the whole point of the default."""
    field = posed_box(seed=6)
    steps = list(peel_steps(field, max_ops=2))
    base = next(s for s in steps if s.stage == "base")
    assert base.freecad is None


@needs_freecad
def test_freecad_check_reports_the_built_genus_against_the_input(tmp_path):
    """The genus gate (report only): a box with a through bore is genus 1 and matches a genus-1 input; the same box
    without the bore is genus 0 and is flagged; an unknown input genus is None, never a pass. (A first version of this
    test used a cylinder from z = 0, i.e. a BLIND hole, and the gate correctly reported genus 0.)"""
    from reco_field.peel import _freecad_check
    through = sp.Placed(sp.Cylinder(1.0, 10.0), translation=np.array([0.0, 0.0, -5.0]))   # Cylinder starts at z = 0
    bored = sp.Placed(sp.Box((5.0, 4.0, 2.0)) - through, translation=np.array([3.0, -2.0, 1.0]))
    solid = sp.Placed(sp.Box((5.0, 4.0, 2.0)), translation=np.array([3.0, -2.0, 1.0]))
    ok = _freecad_check(bored, tmp_path, "bored", input_genus=1.0)
    assert "error" not in ok and ok["genus"] == 1 and ok["genus_match"] is True
    bad = _freecad_check(solid, tmp_path, "solid", input_genus=1.0)
    assert bad["genus"] == 0 and bad["genus_match"] is False
    unknown = _freecad_check(solid, tmp_path, "solid2", input_genus=float("nan"))
    assert unknown["genus_match"] is None
