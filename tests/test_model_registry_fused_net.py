"""reco_engine/model_registry.py: $SHAPE_RECO_FUSED_NET loads a FusedNet checkpoint into Models.fused_net, the same
env-var convention SHAPE_RECO_BOX_ORIENTER/SHAPE_RECO_SPHERE_FIT/SHAPE_RECO_CYLINDER_FIT already use - what the
FreeCAD node layer (_bootstrap.get_models -> node_support.model_environment) relies on. Fake type/rotation/dimension
checkpoints (no FreeCAD needed - load_models only loads torch state dicts) plus a fake FusedNet checkpoint, so this
runs without any real training."""
import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.field_input import CHANNELS  # noqa: E402
from training.train_dimension_cnn import DimensionCNN  # noqa: E402
from training.train_fused import FusedNet  # noqa: E402
from training.train_rotation_cnn import RotationCNN  # noqa: E402
from training.train_type_cnn import TypeCNN  # noqa: E402

from reco_engine.fused_predict import FusedNetHandle  # noqa: E402
from reco_engine.model_registry import load_models  # noqa: E402


def _fake_checkpoints(root: Path):
    for name, model in (("type_cnn", TypeCNN()), ("rotation_cnn", RotationCNN()), ("dimension_cnn", DimensionCNN())):
        d = root / name
        d.mkdir(parents=True)
        torch.save({"model_state_dict": model.state_dict()}, d / "best_model.pth")


def _fake_fused_checkpoint(root: Path):
    root.mkdir(parents=True)
    torch.save({"model_state_dict": FusedNet(in_channels=CHANNELS["fused"]).state_dict()}, root / "best_model.pth")
    (root / "config.json").write_text(json.dumps({"input": "fused", "heads": "type+pose+size"}))


def test_load_models_reads_shape_reco_fused_net_env_var(tmp_path, monkeypatch):
    _fake_checkpoints(tmp_path / "ckpt")
    _fake_fused_checkpoint(tmp_path / "fused")
    monkeypatch.setenv("SHAPE_RECO_FUSED_NET", str(tmp_path / "fused"))
    models = load_models(tmp_path / "ckpt", device=torch.device("cpu"))
    assert isinstance(models.fused_net, FusedNetHandle)
    assert models.fused_net.input_kind == "fused"


def test_load_models_without_the_env_var_leaves_fused_net_none(tmp_path, monkeypatch):
    _fake_checkpoints(tmp_path / "ckpt")
    monkeypatch.delenv("SHAPE_RECO_FUSED_NET", raising=False)
    models = load_models(tmp_path / "ckpt", device=torch.device("cpu"))
    assert models.fused_net is None
