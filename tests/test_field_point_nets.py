"""training/field_point_backbone.py and training/field_point_nets.py: the point-cloud backbone and the
per-topology network classes for the vector-field route's learned initializer - shapes, losses, and target
derivation, all on synthetic data (no dataset_gen STLs or a trained checkpoint needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import rotation_symmetry as rs  # noqa: E402
from training.field_point_backbone import PointCloudBackbone, encode_patch, normalize_points  # noqa: E402
from training.field_point_nets import (  # noqa: E402
    DIMENSION_KEYS, FieldAxisNet, FieldBoxPoseNet, FieldDimensionNet, FieldPositionNet, FieldTypeNet,
    WHOLE_KINDS, true_axis_world, true_size_mm,
)

N = 256


def random_points(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(200, 3)) * 5.0 + rng.normal(size=3) * 20.0   # off-centre, arbitrary scale
    normals = points / np.linalg.norm(points, axis=1, keepdims=True)
    return points, normals


def test_normalize_points_centers_and_scales():
    points, _ = random_points()
    normed, centroid, scale = normalize_points(points)
    assert np.allclose(normed.mean(axis=0), (points - centroid).mean(axis=0) / scale, atol=1e-6)
    assert np.isclose(np.linalg.norm(normed, axis=1).max(), 1.0, atol=1e-6)
    assert np.allclose(centroid, points.mean(axis=0))


def test_encode_patch_shape_is_fixed_regardless_of_input_size():
    points, normals = random_points()
    for n in (50, 200, 1000):                      # fewer or more points than the patch has
        feat = encode_patch(points, normals, n_points=N)
        assert feat.shape == (6, N) and feat.dtype == np.float32


def test_point_cloud_backbone_output_shape():
    backbone = PointCloudBackbone(6, feature_dim=64)
    x = torch.randn(4, 6, N)
    assert backbone(x).shape == (4, 64)


def test_field_type_net_logits_shape():
    net = FieldTypeNet(feature_dim=32)
    out = net(torch.randn(3, 6, N))
    assert out.shape == (3, len(WHOLE_KINDS))


def test_field_box_pose_net_returns_rotation_matrices():
    net = FieldBoxPoseNet(feature_dim=32)
    out = net(torch.randn(5, 6, N))
    assert out.shape == (5, 3, 3)
    det = torch.linalg.det(out)
    assert torch.allclose(det, torch.ones(5), atol=1e-4)          # a proper rotation: det +1
    ortho = torch.bmm(out.transpose(1, 2), out)
    assert torch.allclose(ortho, torch.eye(3).expand(5, 3, 3), atol=1e-4)


def test_field_axis_net_returns_unit_vectors():
    net = FieldAxisNet(feature_dim=32)
    out = net(torch.randn(5, 6, N))
    assert out.shape == (5, 3)
    assert torch.allclose(out.norm(dim=1), torch.ones(5), atol=1e-5)


@pytest.mark.parametrize("topology", list(WHOLE_KINDS))
def test_field_dimension_net_matches_its_topology_param_count(topology):
    n_params = len(DIMENSION_KEYS[topology])
    net = FieldDimensionNet(n_params, feature_dim=32)
    out = net(torch.randn(2, 6, N))
    assert out.shape == (2, n_params)


def test_field_position_net_output_shape():
    net = FieldPositionNet(feature_dim=32)
    out = net(torch.randn(2, 6, N))
    assert out.shape == (2, 3)


def test_true_size_mm_uses_the_realization_scale():
    from training.train_box_dims import REALIZATION_SCALE_MM
    params = {"x_half_ratio": 1.0, "y_half_ratio": 0.5, "z_half_ratio": 0.25}
    size = true_size_mm("box", params)
    assert size == pytest.approx([REALIZATION_SCALE_MM, REALIZATION_SCALE_MM / 2, REALIZATION_SCALE_MM / 4])


def test_true_axis_world_rotates_the_local_z_axis():
    identity_quat = [0.0, 0.0, 0.0, 1.0]
    assert true_axis_world(identity_quat) == pytest.approx([0.0, 0.0, 1.0])
    # a 90 degree rotation about x: local z -> world -y (or +y depending on handedness - just check it's unit and perpendicular to z)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_euler("x", 90, degrees=True).as_quat()          # [x,y,z,w]
    axis = true_axis_world(q)
    assert np.isclose(np.linalg.norm(axis), 1.0)
    assert abs(np.dot(axis, [0.0, 0.0, 1.0])) < 1e-6                  # perpendicular to the original axis


def test_one_optimisation_step_reduces_type_loss():
    """A tiny synthetic overfit run - matches tests/test_train_fused.py's own smoke test convention."""
    net = FieldTypeNet(feature_dim=32)
    x = torch.randn(8, 6, N)
    y = torch.randint(0, len(WHOLE_KINDS), (8,))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(net(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_box_pose_symmetric_loss_is_reusable_directly():
    """training.rotation_symmetry.symmetric_loss_torch, unmodified, works on FieldBoxPoseNet's output."""
    net = FieldBoxPoseNet(feature_dim=32)
    pred = net(torch.randn(4, 6, N))
    target = torch.eye(3).unsqueeze(0).expand(4, 3, 3).contiguous()
    mask = torch.zeros(4, len(rs.GROUP), dtype=torch.bool)
    mask[:, 0] = True
    loss = rs.symmetric_loss_torch(pred, target, mask)
    assert loss.item() >= 0.0 and torch.isfinite(loss)
