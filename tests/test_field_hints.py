"""reco_field/field_hints.py: turning the vector-field route's trained point-cloud networks into an
ADDITIONAL starting hypothesis for reco_field.fit.fit_box - never a replacement for its existing PCA/
Manhattan-frame geometric starts. Uses freshly constructed (untrained) networks saved to a temp checkpoint
tree, so this runs without any real training data or checkpoint."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from training.field_point_nets import (  # noqa: E402
    DIMENSION_KEYS, FieldAxisNet, FieldBoxPoseNet, FieldDimensionNet, FieldPositionNet, FieldTypeNet,
)

from reco_field.field_hints import FieldNetHints  # noqa: E402
from reco_field.fit import fit_box, make_samples  # noqa: E402


def _save(path: Path, model):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path / "best_model.pth")


def _full_checkpoint_root(tmp_path: Path) -> Path:
    root = tmp_path / "checkpoints_field_pertopo"
    _save(root / "type_cnn", FieldTypeNet())
    for topology in ("box", "sphere", "cylinder", "cone", "torus"):
        _save(root / f"{topology}_dimension_cnn", FieldDimensionNet(len(DIMENSION_KEYS[topology])))
        _save(root / f"{topology}_position_cnn", FieldPositionNet())
    _save(root / "box_rotation_cnn", FieldBoxPoseNet())
    for topology in ("cylinder", "cone", "torus"):
        _save(root / f"{topology}_rotation_cnn", FieldAxisNet())
    return root


def test_load_finds_every_network_when_all_are_present(tmp_path):
    hints = FieldNetHints.load(_full_checkpoint_root(tmp_path))
    assert hints.type_net is not None
    assert set(hints.rotation) == {"box", "cylinder", "cone", "torus"}       # sphere has no rotation network
    assert set(hints.dimension) == {"box", "sphere", "cylinder", "cone", "torus"}
    assert set(hints.position) == {"box", "sphere", "cylinder", "cone", "torus"}


def test_load_on_an_empty_directory_finds_nothing(tmp_path):
    hints = FieldNetHints.load(tmp_path / "empty")
    assert hints.type_net is None and hints.rotation == {} and hints.dimension == {}
    from reco_field.fit import make_samples
    from src.closest_point_field import load_field
    s = make_samples(load_field(ROOT / "dataset_gen/examples/bs_box.stl"), n_surface=200, n_near=1)
    assert hints.box_hint(s) is None
    assert hints.type_ranking(s) is None


def test_box_hint_is_a_valid_starting_hypothesis(tmp_path):
    from src.closest_point_field import load_field
    hints = FieldNetHints.load(_full_checkpoint_root(tmp_path))
    s = make_samples(load_field(ROOT / "dataset_gen/examples/bs_box.stl"), n_surface=500, n_near=1)
    hint = hints.box_hint(s)
    assert hint is not None
    centre, R, half = hint
    assert centre.shape == (3,) and R.shape == (3, 3) and half.shape == (3,)
    assert np.all(half > 0)
    assert np.isclose(np.linalg.det(R), 1.0, atol=0.2)                       # a (possibly improper) rotation-ish matrix


def test_type_ranking_is_a_permutation_of_whole_kinds(tmp_path):
    from src.closest_point_field import load_field
    from training.field_point_nets import WHOLE_KINDS
    hints = FieldNetHints.load(_full_checkpoint_root(tmp_path))
    s = make_samples(load_field(ROOT / "dataset_gen/examples/bs_box.stl"), n_surface=200, n_near=1)
    ranking = hints.type_ranking(s)
    assert sorted(ranking) == sorted(WHOLE_KINDS)


def test_fit_box_with_a_hint_never_does_worse_than_without_one(tmp_path):
    """The hint is an ADDITIONAL candidate; _optimise keeps whichever of it and the existing geometric
    starts has the lower cost - so passing one can only match or improve the no-hint result, never regress
    it. An untrained network's hint is a poor guess (large, arbitrary half-extents), the clean case this
    checks: fit_box must be robust to a bad hint, not just a good one."""
    from src.closest_point_field import load_field
    hints = FieldNetHints.load(_full_checkpoint_root(tmp_path))
    s = make_samples(load_field(ROOT / "dataset_gen/examples/bs_box.stl"), n_surface=500, n_near=1)
    without = fit_box(s)
    with_hint = fit_box(s, hint=hints.box_hint(s))
    assert with_hint.cost <= without.cost + 1e-9
    assert with_hint.explained >= without.explained - 1e-9


def test_fit_box_default_signature_is_unchanged_for_existing_callers():
    """No `hint` argument at all - the exact call every existing caller (reco_field.peel, reco_field.export,
    the whole 41-example regression suite) already makes - must behave identically to before this feature."""
    from src.closest_point_field import load_field
    s = make_samples(load_field(ROOT / "dataset_gen/examples/bs_box.stl"), n_surface=500, n_near=1)
    result = fit_box(s)
    assert result.kind == "box" and result.explained > 0.99
