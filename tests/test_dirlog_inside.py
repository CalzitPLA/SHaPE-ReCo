"""training/field_input "dirlog_inside": the five dirlog channels of the (possibly noisy) grid plus the base-point bit
(voxel centre inside the exact mesh = the CLEAN voxelisation). The bit comes from `inside` when given (training: the
sample's .clean.bin), else from the grid itself (inference: the loop's grid from the input mesh is clean)."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import field_input as fi  # noqa: E402
from training.manifest import Record  # noqa: E402
from training.voxel_dataset import VoxelDataset  # noqa: E402


def _cube(r=16):
    v = -np.ones((r, r, r), np.float32)
    v[4:12, 4:12, 4:12] = 1.0
    return v


def test_channels_and_default_bit():
    vol = _cube()
    out = fi.encode(vol, "dirlog_inside")
    assert out.shape == (6, 16, 16, 16) and fi.CHANNELS["dirlog_inside"] == 6
    assert np.array_equal(out[:5], fi.encode(vol, "dirlog"))                 # the first five ARE dirlog
    assert np.array_equal(out[5] > 0.5, vol > 0)                              # no `inside`: the grid's own occupancy


def test_explicit_inside_is_the_clean_grid_under_noise():
    clean = _cube()
    noisy = clean.copy()
    noisy[0, 0, 0] = 1.0                                                      # a salt voxel
    noisy[8, 8, 8] = -1.0                                                     # a pepper voxel
    out = fi.encode(noisy, "dirlog_inside", inside=clean > 0)
    assert out[0, 0, 0, 0] == 1 and out[5, 0, 0, 0] == 0                     # material says yes, the base point says no
    assert out[0, 8, 8, 8] == 0 and out[5, 8, 8, 8] == 1


def test_dataset_uses_the_clean_copy(tmp_path):
    clean = _cube(64) > 0
    noisy = clean.copy()
    noisy[0, 0, 0] = True
    noisy.astype(np.uint8).tofile(tmp_path / "a.bin")
    clean.astype(np.uint8).tofile(tmp_path / "a.clean.bin")
    rec = Record(topology="box", mode="type", parameters={}, rotation={}, stl_path=tmp_path / "a.stl",
                 voxel_path=tmp_path / "a.bin", group_key="g", placement={"noise": True})
    t, _ = VoxelDataset([rec], 64, input_kind="dirlog_inside")._load(0)
    assert isinstance(t, torch.Tensor) and tuple(t.shape) == (6, 64, 64, 64)
    assert t[5].numpy().sum() == clean.sum() and t[0].numpy().sum() == clean.sum() + 1


def test_dataset_prefers_the_stored_inside_bits(tmp_path):
    """The standard <stem>.inside.npy (parity of the placed mesh, dataset_gen/src/inside_bits.py) wins over .clean.bin and
    over the grid; a record flagged degenerate is dropped by filter_records."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "dataset_gen"))
    from dataset_gen.src.inside_bits import inside_path, read_inside
    from training.subclass_schema import filter_records
    grid = _cube(64) > 0
    grid.astype(np.uint8).tofile(tmp_path / "a.bin")
    grid.astype(np.uint8).tofile(tmp_path / "a.clean.bin")
    truth = np.zeros_like(grid)
    truth[10:20, 30:40, 5:9] = True                                          # deliberately different from both
    np.save(inside_path(tmp_path / "a.bin"), np.packbits(truth.ravel()))
    assert np.array_equal(read_inside(tmp_path / "a.bin"), truth)
    rec = Record(topology="box", mode="type", parameters={}, rotation={}, stl_path=tmp_path / "a.stl",
                 voxel_path=tmp_path / "a.bin", group_key="g", placement={"noise": True})
    t, _ = VoxelDataset([rec], 64, input_kind="dirlog_inside")._load(0)
    assert np.array_equal(t[5].numpy() > 0.5, truth)
    bad = Record(topology="box", mode="type", parameters={}, rotation={}, stl_path=tmp_path / "b.stl",
                 voxel_path=tmp_path / "a.bin", group_key="g", placement={"degenerate": True})
    kept, skipped = filter_records([rec, bad])
    assert kept == [rec] and skipped["degenerate"] == 1
