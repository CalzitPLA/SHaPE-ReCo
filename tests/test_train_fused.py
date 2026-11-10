"""training/train_fused.py: the multi-task network on the fused input - targets, masking across topologies, one optimisation step, a short overfit."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import rotation_symmetry as rs  # noqa: E402
from training.manifest import Record  # noqa: E402
from training.train_fused import FusedDataset, FusedNet, eval_step, size_targets, train_step  # noqa: E402

N = 32


def solid(kind):
    idx = np.indices((N, N, N)).astype(float) - (N - 1) / 2
    if kind == "sphere":
        return (idx ** 2).sum(0) <= 8 ** 2
    if kind == "cylinder":
        return ((idx[1] ** 2 + idx[2] ** 2) <= 5 ** 2) & (np.abs(idx[0]) <= 9)
    return (np.abs(idx[0]) <= 10) & (np.abs(idx[1]) <= 6) & (np.abs(idx[2]) <= 4)


def record(tmp_path, kind, i=0):
    path = tmp_path / f"{kind}{i}.bin"
    solid(kind).astype(np.uint8).tofile(path)
    params = {"box": dict(x_half_ratio=2.5, y_half_ratio=1.5, z_half_ratio=1.0), "sphere": dict(radius_ratio=1.6),
              "cylinder": dict(radius_ratio=1.25, half_length_ratio=2.25)}[kind]
    return Record(topology=kind, mode="type", parameters=params, rotation=dict(qx=0.0, qy=0.0, qz=0.0, qw=1.0), stl_path="", voxel_path=path,
                  group_key=f"{kind}{i}", placement=dict(scale_factor=0.4, full_field=False))


def test_size_targets_per_topology(tmp_path):
    k = 0.4 * 10.0 / (40.0 / 64)                                                  # voxels per unit ratio
    out, mask = size_targets(record(tmp_path, "box"))
    np.testing.assert_allclose(np.exp(out), np.array([2.5, 1.5, 1.0]) * k, rtol=1e-5)
    assert mask.tolist() == [1, 1, 1]                                            # longest first
    out, mask = size_targets(record(tmp_path, "sphere"))
    assert mask.tolist() == [1, 0, 0] and np.exp(out[0]) == pytest.approx(1.6 * k, rel=1e-5)
    out, mask = size_targets(record(tmp_path, "cylinder"))
    assert mask.tolist() == [1, 1, 0] and np.exp(out[:2]) == pytest.approx([1.25 * k, 2.25 * k], rel=1e-5)


def test_dataset_yields_channels_and_masks(tmp_path):
    recs = [record(tmp_path, k) for k in ("box", "sphere", "cylinder")]
    for kind, c in (("occupancy", 1), ("fused", 5)):
        ds = FusedDataset(recs, N, kind)
        x, t, M, mask, has_rot, log_size, size_mask = ds[0]
        assert x.shape == (c, N, N, N) and t == 0 and has_rot == 1.0 and mask.dtype == torch.bool and M.shape == (3, 3)
        assert ds[1][4] == 0.0 and ds[2][4] == 0.0 and ds[1][3].sum() == 1          # no rotation label for sphere / cylinder


def test_augmentation_rotates_the_box_label_with_the_volume(tmp_path):
    rec = record(tmp_path, "box")
    ds = FusedDataset([rec], N, "occupancy", augment=True)
    ds.rng = np.random.default_rng(3)
    x, _, M, mask, *_ = ds[0]
    g_used = None
    M0 = ds.rot[0][0]
    for g in rs.GROUP:                                                            # the sampled group element: M = g @ M0
        if np.allclose(g @ M0, M.numpy(), atol=1e-6):
            g_used = g
            break
    assert g_used is not None
    np.testing.assert_array_equal(x[0].numpy(), rs.rotate_volume(solid("box").astype(np.float32) * 2 - 1, g_used))


def test_step_is_finite_and_the_network_can_overfit_a_tiny_batch(tmp_path):
    recs = [record(tmp_path, k, i) for i in range(2) for k in ("box", "sphere", "cylinder")]
    ds = FusedDataset(recs, N, "fused")
    batch = torch.utils.data.default_collate([ds[i] for i in range(len(ds))])
    torch.manual_seed(0)
    model = FusedNet(5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    first = None
    # 180 steps, not 60: after 60 the size error depends on the number of threads (0.51 / 0.42 / 0.20 with 1 / 4 / 8
    # threads, the same seed; the summation order differs), after 180 it is 0.05 / 0.05 / 0.16 (measured 2026-09-27)
    for step in range(180):
        m = train_step(model, batch, torch.device("cpu"), opt)
        first = first or m
        assert np.isfinite(m["loss"])
    assert m["loss"] < 0.4 * first["loss"] and m["ce"] < 0.3
    model.eval()
    with torch.no_grad():
        e = eval_step(model, batch, torch.device("cpu"))
    assert e["acc"] == 1.0 and "rot_median_deg" in e and e["size_l1"] < 0.3
