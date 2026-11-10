"""Type CNN input kind (training/field_input.py) end to end through reco_engine: model_registry reads the
checkpoint's own config.json to pick in_channels, predict.py encodes the grid the same way before every forward
pass, and a checkpoint trained on the mesh's exact field (needs the STL, not just the grid) is refused rather
than silently mis-encoded."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reco_engine import model_registry
from reco_engine.predict import _predict_type_in_process, _type_tensor
from training.topology_schema import NUM_TOPOLOGIES


def _fake_checkpoint_dir(tmp_path, config=None):
    d = tmp_path / "type_cnn"
    d.mkdir()
    if config is not None:
        (d / "config.json").write_text(json.dumps(config))
    (d / "best_model.pth").write_bytes(b"")  # never actually loaded by _type_cnn_input_kind
    return tmp_path


def test_no_config_json_defaults_to_occupancy(tmp_path):
    root = _fake_checkpoint_dir(tmp_path, config=None)
    kind, truncation, channels = model_registry._type_cnn_input_kind(root, "best_model.pth")
    assert (kind, truncation, channels) == ("occupancy", 8.0, 1)


def test_old_config_without_input_key_defaults_to_occupancy(tmp_path):
    """checkpoints_type_v4's config.json predates --input: no "input" key at all."""
    root = _fake_checkpoint_dir(tmp_path, config={"epochs": 40, "n_train": 3840})
    kind, truncation, channels = model_registry._type_cnn_input_kind(root, "best_model.pth")
    assert (kind, truncation, channels) == ("occupancy", 8.0, 1)


@pytest.mark.parametrize("kind,channels", [("occupancy", 1), ("sdf", 1), ("field", 3), ("fused", 5), ("voxel_field", 4)])
def test_channel_count_matches_the_input_kind(tmp_path, kind, channels):
    root = _fake_checkpoint_dir(tmp_path, config={"input": kind, "sdf_truncation": 8.0})
    got_kind, truncation, got_channels = model_registry._type_cnn_input_kind(root, "best_model.pth")
    assert got_kind == kind and got_channels == channels


def test_exact_field_is_refused_with_a_clear_reason(tmp_path):
    """exact_field needs the STL mesh; the reconstruction loop only ever has re-voxelized sub-volumes with no
    mesh of their own (reco_engine/greedy_reconstruct.py), so this must fail loudly at load time, not silently
    fall back to occupancy or crash deep inside a forward pass."""
    root = _fake_checkpoint_dir(tmp_path, config={"input": "exact_field", "field_resolution": 128})
    with pytest.raises(ValueError, match="exact_field"):
        model_registry._type_cnn_input_kind(root, "best_model.pth")


def test_unknown_input_kind_is_refused(tmp_path):
    root = _fake_checkpoint_dir(tmp_path, config={"input": "nonsense"})
    with pytest.raises(ValueError, match="nonsense"):
        model_registry._type_cnn_input_kind(root, "best_model.pth")


def test_type_tensor_encodes_occupancy_as_is():
    vol = (np.random.default_rng(0).random((8, 8, 8)) > 0.5).astype(np.float32) * 2 - 1
    models = SimpleNamespace(type_input_kind="occupancy", type_sdf_truncation=8.0, device=torch.device("cpu"))
    tensor = _type_tensor(models, vol)
    assert tuple(tensor.shape) == (1, 1, 8, 8, 8)
    np.testing.assert_array_equal(tensor[0, 0].numpy(), vol)


def test_type_tensor_encodes_fused_with_five_channels():
    vol = (np.random.default_rng(0).random((8, 8, 8)) > 0.5).astype(np.float32) * 2 - 1
    models = SimpleNamespace(type_input_kind="fused", type_sdf_truncation=8.0, device=torch.device("cpu"))
    tensor = _type_tensor(models, vol)
    assert tuple(tensor.shape) == (1, 5, 8, 8, 8)


def test_type_tensor_defaults_to_occupancy_when_the_model_object_has_no_input_kind():
    """A lightweight test double (e.g. tests/test_topology_restriction.py's fake_models) has no type_input_kind
    attribute at all - must behave like an occupancy-trained model, not crash."""
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    models = SimpleNamespace(type_cnn=lambda t: torch.zeros(1, NUM_TOPOLOGIES), device=torch.device("cpu"))
    tensor = _type_tensor(models, vol)
    assert tuple(tensor.shape) == (1, 1, 4, 4, 4)


def test_predict_type_in_process_routes_through_the_right_encoding():
    """A fake fused-input network that only produces sane logits when it actually receives 5 channels (not the
    bare occupancy grid) - proves _predict_type_in_process really encodes before calling the network, not just
    that _type_tensor does in isolation."""
    def fake_net(tensor):
        assert tensor.shape[1] == 5
        return torch.zeros(1, NUM_TOPOLOGIES)
    vol = (np.random.default_rng(1).random((8, 8, 8)) > 0.5).astype(np.float32) * 2 - 1
    models = SimpleNamespace(type_cnn=fake_net, device=torch.device("cpu"), allowed_topologies=None,
                             type_input_kind="fused", type_sdf_truncation=8.0)
    prediction = _predict_type_in_process(models, vol)
    assert prediction.topology
