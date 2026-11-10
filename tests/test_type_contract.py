#!/usr/bin/env python3
"""The type stage's input contract (R-05c, 2026-10-01): a grid of another resolution than the one a type source was
trained on is refused instead of answered silently (a 32^3 grid into the 64^3 Type CNN cost 0.23 parent accuracy), and
the cascade builds its target at the source's own resolution. No checkpoints, no torch: fake models only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))

from reco_engine import cascade, model_registry as mr  # noqa: E402
from reco_engine.cascade import _Entry  # noqa: E402
from reco_engine.model_registry import Models  # noqa: E402
from reco_engine.predict import check_type_grid, predict_type  # noqa: E402
from reco_engine.type_descriptors import DescriptorModel  # noqa: E402

POSE = [e for e in cascade.DEFAULT_REGISTRY if e.stage != "type"]


def _models(resolution):
    return Models(type_cnn=object(), rotation_cnn=None, dimension_cnn=None, device=None,
                  checkpoint_root=Path("ckpt"), type_resolution=resolution)


def test_a_grid_of_another_resolution_is_refused():
    with pytest.raises(ValueError, match="64\\^3"):
        predict_type(_models(64), -np.ones((32, 32, 32), np.float32))   # raises before the net is touched
    check_type_grid(_models(64), -np.ones((64, 64, 64), np.float32))


def test_shape_checks_without_a_recorded_resolution():
    check_type_grid(_models(None), -np.ones((32, 32, 32), np.float32))   # legacy checkpoint: any cubic grid
    with pytest.raises(ValueError, match="cubic"):
        check_type_grid(_models(None), -np.ones((64, 64, 32), np.float32))
    with pytest.raises(ValueError, match="inside"):
        check_type_grid(_models(64), -np.ones((64, 64, 64), np.float32), inside=np.zeros((32, 32, 32), bool))


def test_descriptor_model_refuses_a_grid_of_another_resolution():
    model = DescriptorModel(parent_clf=None, sub_clf=None, meta={})        # a fit without "resolution": 64
    with pytest.raises(ValueError, match="64\\^3"):
        cascade.check_descriptor_grid(model, np.zeros((32, 32, 32), bool))
    with pytest.raises(ValueError, match="32\\^3"):
        cascade.check_descriptor_grid(DescriptorModel(parent_clf=None, sub_clf=None, meta={"resolution": 32}),
                                      np.zeros((64,) * 3, bool))
    cascade.check_descriptor_grid(model, np.zeros((64,) * 3, bool))


def test_the_descriptor_source_is_the_fitted_models_fingerprint():
    """type_descriptors.py's bytes are the fingerprint a fitted model is checked against: the live model must match."""
    path = ROOT / "models" / "type_descriptors_v5_medium.joblib"
    if not path.is_file():
        pytest.skip("live descriptor model not present")
    import joblib
    from reco_engine.type_descriptors import source_fingerprint
    m = joblib.load(path)
    assert (m["meta"] if isinstance(m, dict) else m.meta).get("fingerprint") == source_fingerprint()


def test_resolution_is_read_from_config_then_model_card(tmp_path):
    type_dir = tmp_path / "type_cnn"
    type_dir.mkdir()
    assert mr._type_cnn_resolution(tmp_path) is None
    (type_dir / "model_card.json").write_text(json.dumps({"resolution": 48}))
    assert mr._type_cnn_resolution(tmp_path) == 48
    (type_dir / "config.json").write_text(json.dumps({"input": "occupancy", "resolution": 64}))
    assert mr._type_cnn_resolution(tmp_path) == 64


def test_live_checkpoint_records_its_resolution():
    root = ROOT / "checkpoints" / "type_hier_medium_parity_dirlog_inside"
    if not (root / "type_cnn" / "config.json").is_file():
        pytest.skip("live type checkpoint not present")
    assert mr._type_cnn_resolution(root) == 64


class _Net:
    """A type source that answers on whatever grid it gets and records its shape."""

    def __init__(self, resolution=None):
        self.type_resolution, self.shapes = resolution, []

    def __call__(self, x):
        self.shapes.append(x.shape)
        return [("box", 0.9), ("sphere", 0.1)]


def _box_target():
    from src.field_channels import voxel_channels
    spec = cascade.grid_spec(-12.0, 12.0, 32)
    c = spec.centres()
    occ = (np.abs(c[..., 0]) <= 6) & (np.abs(c[..., 1]) <= 4) & (np.abs(c[..., 2]) <= 3)
    return voxel_channels(occ, spec), spec


@pytest.mark.parametrize("resolution, expected", [(None, cascade.INSCALE_RESOLUTION), (48, 48)])
def test_cascade_builds_the_target_at_the_source_resolution(resolution, expected):
    ch, spec = _box_target()
    net = _Net(resolution)
    reg = [_Entry("net", "type", ("box", "sphere"), "cnn", cost=1)] + POSE
    cascade.type_stage(ch, spec, reg, models={"net": net})
    assert net.shapes == [(expected,) * 3]
