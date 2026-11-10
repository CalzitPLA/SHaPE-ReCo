"""
Restricting the Type CNN to the topologies it was actually trained on
(box, sphere) - for troubleshooting with bodies that are exactly representable.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reco_engine import model_registry
from reco_engine.cache import Cache
from reco_engine.predict import _predict_type_in_process
from training.topology_schema import IDX_TO_TOPOLOGY, NUM_TOPOLOGIES, TOPOLOGY_TO_IDX


def fake_models(favourite: str, allowed=None):
    """A type network whose logits strongly prefer `favourite`."""
    logits = torch.zeros(1, NUM_TOPOLOGIES)
    logits[0, TOPOLOGY_TO_IDX[favourite]] = 8.0
    logits[0, TOPOLOGY_TO_IDX["box"]] = 2.0
    logits[0, TOPOLOGY_TO_IDX["sphere"]] = 1.0
    return SimpleNamespace(type_cnn=lambda tensor: logits, device=torch.device("cpu"), allowed_topologies=allowed)


VOXEL = np.zeros((4, 4, 4), dtype=np.float32)


def test_unrestricted_prediction_can_be_an_untrained_class():
    prediction = _predict_type_in_process(fake_models("parallelepiped"), VOXEL)
    assert prediction.topology == "parallelepiped"


def test_restriction_never_returns_a_masked_topology():
    prediction = _predict_type_in_process(fake_models("parallelepiped", ("box", "sphere")), VOXEL)
    assert prediction.topology == "box"                       # best of the allowed, despite the favourite
    assert [name for name, _ in prediction.top5] == ["box", "sphere"]      # no padding with p = 0 entries
    assert sum(p for _, p in prediction.top5) == pytest.approx(1.0)
    assert prediction.confidence == pytest.approx(prediction.top5[0][1])


def test_restriction_to_one_topology():
    prediction = _predict_type_in_process(fake_models("box", ("sphere",)), VOXEL)
    assert prediction.topology == "sphere" and prediction.confidence == pytest.approx(1.0)


def test_parse_allowed(monkeypatch):
    monkeypatch.delenv("SHAPE_RECO_TOPOLOGIES", raising=False)
    assert model_registry._parse_allowed(None) is None
    assert model_registry._parse_allowed("box,sphere") == ("box", "sphere")
    assert model_registry._parse_allowed(["box"]) == ("box",)
    monkeypatch.setenv("SHAPE_RECO_TOPOLOGIES", "box sphere")
    assert model_registry._parse_allowed(None) == ("box", "sphere")
    with pytest.raises(ValueError, match="unknown topologies"):
        model_registry._parse_allowed("box,dodecahedron")


def test_cache_key_depends_on_the_restriction(tmp_path):
    root = tmp_path / "ckpt"
    for sub in ("type_cnn", "rotation_cnn", "dimension_cnn"):
        (root / sub).mkdir(parents=True)
        (root / sub / "best_model.pth").write_bytes(b"w")
    base = SimpleNamespace(checkpoint_root=root, checkpoint_name="best_model.pth", allowed_topologies=None)
    restricted = SimpleNamespace(checkpoint_root=root, checkpoint_name="best_model.pth",
                                 allowed_topologies=("sphere", "box"))
    swapped = SimpleNamespace(checkpoint_root=root, checkpoint_name="best_model.pth",
                              allowed_topologies=("box", "sphere"))
    c = Cache(tmp_path / "c")
    grid = np.zeros((4, 4, 4), dtype=np.float32)
    assert c.predict_key(base, grid) != c.predict_key(restricted, grid)
    assert c.predict_key(restricted, grid) == c.predict_key(swapped, grid)     # order-insensitive
