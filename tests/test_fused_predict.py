"""reco_engine/fused_predict.py: loading a train_fused.py checkpoint and turning FusedNet's raw output into what
build_primitive_stl expects. Uses a freshly constructed (untrained) FusedNet with a biased type head to force each
topology branch deterministically - no real checkpoint needed, so this runs without training/train_fused.py ever
having been invoked (see tests/test_notebook_pipeline_fused.py for the trained-checkpoint integration test)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.field_input import CHANNELS  # noqa: E402
from training.topology_schema import TOPOLOGY_TO_IDX  # noqa: E402
from training.train_fused import FusedNet  # noqa: E402

from reco_engine.fused_predict import (  # noqa: E402
    FUSED_TOPOLOGIES, FusedNetHandle, fused_estimate_for_topology, load_fused_model, predict_fused,
)

N = 64


def box_voxel(half=16):
    v = np.zeros((N, N, N), dtype=np.uint8)
    lo, hi = N // 2 - half, N // 2 + half
    v[lo:hi, lo:hi, lo:hi] = 1
    return v


def sphere_voxel(radius=20):
    zz, yy, xx = np.indices((N, N, N))
    c = N // 2
    return (((zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2) < radius ** 2).astype(np.uint8)


def biased_handle(topology, kind="fused"):
    """A FusedNetHandle whose type head always answers `topology`, so the pose/size branches run deterministically."""
    model = FusedNet(in_channels=CHANNELS[kind])
    with torch.no_grad():
        model.type_head.bias.zero_()
        model.type_head.bias[TOPOLOGY_TO_IDX[topology]] = 50.0
    model.eval()
    return FusedNetHandle(model=model, input_kind=kind, device=torch.device("cpu"))


def test_load_fused_model_round_trips_a_checkpoint(tmp_path):
    model = FusedNet(in_channels=CHANNELS["fused"])
    torch.save({"model_state_dict": model.state_dict()}, tmp_path / "best_model.pth")
    (tmp_path / "config.json").write_text(json.dumps({"input": "fused", "heads": "type+pose+size"}))
    handle = load_fused_model(tmp_path)
    assert handle.input_kind == "fused"
    assert handle.device is not None


def test_load_fused_model_needs_a_config_json(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_fused_model(tmp_path)


def test_predict_fused_box_gives_a_refined_size_close_to_the_true_extent():
    handle = biased_handle("box")
    pred = predict_fused(handle, box_voxel(half=16))
    assert pred.topology == "box"
    assert pred.quaternion is not None and len(pred.quaternion) == 4
    # refine_box optimizes against the actual voxels regardless of the (untrained) network's own size guess
    for key in ("x_half_ratio", "y_half_ratio", "z_half_ratio"):
        assert 0.85 < pred.parameters[key] < 1.15   # true half extent: 16 vox * 0.625 mm/vox / 10 mm realization = 1.0


def test_predict_fused_sphere_has_identity_quaternion_and_no_pose_head_use():
    handle = biased_handle("sphere")
    pred = predict_fused(handle, sphere_voxel())
    assert pred.topology == "sphere"
    assert pred.quaternion == [0.0, 0.0, 0.0, 1.0]
    assert set(pred.parameters) == {"radius_ratio"}


def test_predict_fused_outside_fused_topologies_reports_type_only():
    handle = biased_handle("torus")
    pred = predict_fused(handle, box_voxel())
    assert pred.topology == "torus"
    assert pred.topology not in FUSED_TOPOLOGIES
    assert pred.quaternion is None
    assert pred.parameters == {}


def test_fused_estimate_for_topology_matches_predict_fused_on_a_box():
    handle = biased_handle("box")
    voxel = box_voxel(half=16)
    estimate = fused_estimate_for_topology(handle, voxel, "box")
    pred = predict_fused(handle, voxel)
    assert estimate.parameters == pytest.approx(pred.parameters)
    assert list(estimate.quaternion) == pytest.approx(list(pred.quaternion))


def test_fused_estimate_for_topology_rejects_an_uncovered_topology():
    handle = biased_handle("box")
    assert fused_estimate_for_topology(handle, box_voxel(), "torus") is None


def test_fused_estimate_for_topology_needs_enough_material():
    handle = biased_handle("box")
    assert fused_estimate_for_topology(handle, np.zeros((N, N, N), dtype=np.uint8), "box") is None
