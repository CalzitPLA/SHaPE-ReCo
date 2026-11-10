"""training/gpu_encode.py against the CPU definitions of training/field_input.py: same channels (material, inside and
d_log to float rounding, the normal within 2e-3), the 24 grid rotations exactly as rotation_symmetry.rotate_volume, and
the host-side loader's inside-bit fallback order."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training import field_input as fi  # noqa: E402
from training import gpu_encode as ge  # noqa: E402
from training.rotation_symmetry import GROUP, rotate_volume  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
NORMAL_TOL = 2e-3


def asymmetric_body(r=32):
    """An off-centre, asymmetric L-shaped body with a hole - not a cube, not centred (the project's voxel-frame rule)."""
    v = np.zeros((r, r, r), bool)
    v[3:20, 5:14, 9:30] = True
    v[3:9, 14:27, 9:16] = True
    v[10:14, 7:11, 12:25] = False
    return v


def grids_under_test(r=32):
    real = sorted(p for p in (ROOT / "dataset_gen" / "local_dataset_v5" / "small" / "train").rglob("*.bin")
                  if not p.name.endswith(".clean.bin"))[:3]
    out = [asymmetric_body(r)]
    if real:
        out = [np.fromfile(p, np.uint8).reshape(64, 64, 64) > 0 for p in real]
    return out


def cpu(kind, occ, inside=None):
    return fi.encode(np.where(occ, 1.0, -1.0).astype(np.float32), kind, inside=inside)


def compare(a, b):
    """material/inside/d_log exact (to 1e-5), normals within NORMAL_TOL."""
    assert a.shape == b.shape
    scalar = [0, 4] + ([5] if a.shape[0] == 6 else [])
    assert np.abs(a[scalar] - b[scalar]).max() < 1e-5
    assert np.abs(a[1:4] - b[1:4]).max() < NORMAL_TOL


@pytest.mark.parametrize("kind", ["dirlog", "dirlog_inside"])
def test_gpu_dirlog_matches_the_cpu_encoding(kind):
    for occ in grids_under_test() + [asymmetric_body()]:
        inside = np.roll(occ, 1, axis=0) if kind == "dirlog_inside" else None        # a bit grid different from the grid
        g = torch.as_tensor(occ[None].astype(np.uint8), device=DEV)
        ins = torch.as_tensor(inside[None].astype(np.uint8), device=DEV) if inside is not None else None
        got = ge.encode_batch(g, kind, inside=ins)[0].cpu().numpy()
        compare(got, cpu(kind, occ, inside))


def test_occupancy_and_degenerate_grids():
    occ = asymmetric_body()
    got = ge.encode_batch(torch.as_tensor(occ[None].astype(np.uint8), device=DEV), "occupancy")[0].cpu().numpy()
    assert np.array_equal(got, cpu("occupancy", occ))
    for special in (np.zeros((16, 16, 16), bool), np.ones((16, 16, 16), bool)):      # empty and full-field grids
        got = ge.encode_batch(torch.as_tensor(special[None].astype(np.uint8), device=DEV), "dirlog")[0].cpu().numpy()
        compare(got, cpu("dirlog", special))


def test_a_batch_equals_its_samples_one_by_one():
    grids = np.stack([asymmetric_body(), np.roll(asymmetric_body(), 5, axis=2)]).astype(np.uint8)
    both = ge.encode_batch(torch.as_tensor(grids, device=DEV), "dirlog").cpu().numpy()
    for b in range(2):
        one = ge.encode_batch(torch.as_tensor(grids[b:b + 1], device=DEV), "dirlog")[0].cpu().numpy()
        d = np.abs(both[b] - one)
        assert d[[0, 4]].max() < 1e-5                            # material and log distance: identical
        # normals: where the smoothed gradient nearly vanishes (medial points) float rounding - which depends on the batch
        # composition on the CPU - decides between "no direction" and a unit vector; measured <= 11 of 32768 voxels
        assert (d[1:4].max(axis=0) > 1e-3).mean() < 1e-3


def test_rotation_is_rotate_volume_for_all_24_and_commutes_with_encoding():
    occ = asymmetric_body()
    t = torch.as_tensor(occ.astype(np.uint8), device=DEV)
    for g in GROUP:
        assert np.array_equal(ge.rotate_grid(t, g).cpu().numpy() > 0, rotate_volume(occ, g))
    idx = list(range(24))
    rot, _ = ge.rotate_batch(t[None].expand(24, -1, -1, -1), None, idx)
    enc = ge.encode_batch(rot, "dirlog").cpu().numpy()
    for i in idx:                                 # rotate-then-encode on the GPU == the CPU encoding of the rotated grid
        compare(enc[i], cpu("dirlog", rotate_volume(occ, GROUP[i])))


def test_load_raw_prefers_stored_bits_then_clean_grid(tmp_path):
    grid = asymmetric_body(16).astype(np.uint8)
    b = tmp_path / "s.bin"
    grid.tofile(b)
    g, ins = ge.load_raw(b, 16, with_inside=True)
    assert np.array_equal(ins, grid)                                        # nothing stored: the grid itself
    clean = np.roll(grid, 1, axis=1)
    clean.tofile(tmp_path / "s.clean.bin")
    assert np.array_equal(ge.load_raw(b, 16, with_inside=True)[1], clean)   # then the clean copy
    bits = np.roll(grid, 2, axis=2) > 0
    np.save(tmp_path / "s.inside.npy", np.packbits(bits.ravel()))
    assert np.array_equal(ge.load_raw(b, 16, with_inside=True)[1], bits)    # the stored standard first


def test_seed_worker_gives_each_loader_worker_its_own_augmentation_stream():
    """Before: every forked worker copied the dataset's numpy generator and drew the same rotations."""
    from torch.utils.data import DataLoader, Dataset
    from training.train_common import loader_kwargs

    class D(Dataset):
        def __init__(self):
            self.rng = np.random.default_rng()

        def __len__(self):
            return 8

        def __getitem__(self, i):
            return int(self.rng.integers(1_000_000))

    draws = [b.tolist() for b in DataLoader(D(), batch_size=2, num_workers=4, **loader_kwargs("cpu", 4, 0))]
    assert len({tuple(d) for d in draws}) == len(draws)                          # the workers differ
    again = [b.tolist() for b in DataLoader(D(), batch_size=2, num_workers=4, **loader_kwargs("cpu", 4, 0))]
    assert draws == again                                                        # and the run is reproducible
