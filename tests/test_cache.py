"""
reco_engine.cache: content-addressed reuse of predictions, built primitives
and voxelizations. CNN / FreeCAD / stltovoxel are stubbed.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from reco_engine.cache import Cache, cached_predictions, default_cache, disable_cache, enable_cache
from reco_engine.predict import TypePrediction

RES = 16


@pytest.fixture
def macro(tmp_path):
    path = tmp_path / "macros" / "box_gdml_dataset_learning_CNN.FCMacro"
    path.parent.mkdir()
    path.write_text("# macro v1")
    return path


@pytest.fixture
def models(tmp_path):
    root = tmp_path / "ckpt"
    for sub in ("type_cnn", "rotation_cnn", "dimension_cnn"):
        (root / sub).mkdir(parents=True)
        (root / sub / "best_model.pth").write_bytes(b"weights")
    return SimpleNamespace(checkpoint_root=root, checkpoint_name="best_model.pth")


# ------------------------------------------------------------------- keys

def test_build_key_depends_on_every_input(tmp_path, macro):
    c = Cache(tmp_path / "c")
    args = dict(topology="box", parameters={"a": 1.0, "b": 2.0}, quaternion=[0, 0, 0, 1.0],
                scale_mm=10.0, resolution=64, macro_path=macro)
    base = c.build_key(**args)
    assert c.build_key(**args) == base
    assert c.build_key(**{**args, "parameters": {"a": 1.0, "b": 2.0000001}}) != base
    assert c.build_key(**{**args, "parameters": {"b": 2.0, "a": 1.0}}) == base       # order-insensitive
    assert c.build_key(**{**args, "quaternion": [0, 0, 1e-9, 1.0]}) != base
    assert c.build_key(**{**args, "scale_mm": 20.0}) != base
    assert c.build_key(**{**args, "resolution": 32}) != base
    assert c.build_key(**{**args, "topology": "sphere"}) != base
    macro.write_text("# macro v2")                                                    # macro edited...
    assert Cache(tmp_path / "c").build_key(**args) != base                            # ...no stale hit


def test_prediction_key_depends_on_input_and_checkpoints(tmp_path, models):
    c = Cache(tmp_path / "c")
    grid = np.zeros((RES,) * 3, dtype=np.float32)
    key = c.predict_key(models, grid)
    other = grid.copy()
    other[0, 0, 0] = 1
    assert c.predict_key(models, other) != key
    ckpt = models.checkpoint_root / "type_cnn" / "best_model.pth"
    ckpt.write_bytes(b"retrained weights")                                            # checkpoint changed
    assert Cache(tmp_path / "c").predict_key(models, grid) != key


# ---------------------------------------------------------------- entries

def test_prediction_round_trip_and_persistence(tmp_path, models):
    grid = np.ones((RES,) * 3, dtype=np.float32)
    calls = []

    def predict_all():
        calls.append(1)
        return (TypePrediction("box", 0.9, [("box", 0.9), ("sphere", 0.1)]), [0.0, 0.0, 0.0, 1.0], {"x": 2.5})

    first = cached_predictions(Cache(tmp_path / "c"), models, grid, predict_all)
    second_cache = Cache(tmp_path / "c")                                              # a new process, same dir
    second = cached_predictions(second_cache, models, grid, predict_all)
    assert len(calls) == 1 and second_cache.stats.predict_hits == 1
    assert first[0].top5 == second[0].top5 and first[1] == second[1] and first[2] == second[2]
    assert second[0].top5[0] == ("box", 0.9)                                          # tuples, not lists


def test_no_cache_just_computes(models):
    grid = np.zeros((RES,) * 3, dtype=np.float32)
    assert cached_predictions(None, models, grid, lambda: ("t", "q", "p")) == ("t", "q", "p")


def test_corrupt_entries_are_misses(tmp_path):
    c = Cache(tmp_path / "c")
    (c.root / "predict").mkdir()
    (c.root / "predict" / "k.json").write_text("{not json")
    (c.root / "voxel").mkdir()
    (c.root / "voxel" / "k.npz").write_bytes(b"garbage")
    (c.root / "voxel" / "empty.npz").write_bytes(b"")
    assert c.get_prediction("k") is None and c.get_voxel("k") is None and c.get_voxel("empty") is None
    assert c.get_build("absent", tmp_path / "out.stl") is False


# ------------------------------------------- shared key recipe and the voxel key (T-19)

def _legacy_voxel_key(stl, resolution, scale_mode="fit", scale_mm=None, anchor="center"):
    """The voxel key formula before T-19 (STL bytes + resolution [+ mode]): the new keys must never equal it."""
    from reco_engine.cache import _digest
    parts = [Path(stl).read_bytes(), str(int(resolution)).encode()]
    if scale_mode != "fit":
        parts += [scale_mode.encode(), repr(scale_mm).encode()]
        if anchor != "center":
            parts.append(anchor.encode())
    return _digest(*parts)


def test_voxel_key_names_the_voxelizer_that_runs(tmp_path, monkeypatch):
    stl = tmp_path / "a.stl"
    stl.write_bytes(b"solid x")
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
    inscale = Cache.voxel_key(stl, 64, "fixed", 40.0, "origin")
    assert inscale == Cache.voxel_key(stl, 64, "fixed", 40.0, "origin", voxelizer="parity")      # parity is the default
    legacy = Cache.voxel_key(stl, 64, "fixed", 40.0, "origin", voxelizer="stltovoxel")
    assert legacy != inscale
    monkeypatch.setenv("SHAPE_RECO_VOXELIZER", "stltovoxel")                                    # the env picks it too
    assert Cache.voxel_key(stl, 64, "fixed", 40.0, "origin") == legacy
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER")
    for args in ((64,), (64, "uniform"), (64, "fixed", 40.0), (64, "fixed", 40.0, "origin")):
        assert Cache.voxel_key(stl, *args) != _legacy_voxel_key(stl, *args)                     # old entries never hit


def test_switching_the_voxelizer_misses_the_cached_grid(tmp_path, monkeypatch):
    """A grid cached under one voxelizer is not served to a run that uses the other."""
    stl = tmp_path / "a.stl"
    stl.write_bytes(b"solid x")
    c = Cache(tmp_path / "c")
    monkeypatch.setenv("SHAPE_RECO_VOXELIZER", "stltovoxel")
    c.put_voxel(Cache.voxel_key(stl, 16, "fixed", 40.0, "origin"), np.ones((16,) * 3, np.uint8))
    assert c.get_voxel(Cache.voxel_key(stl, 16, "fixed", 40.0, "origin")) is not None
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER")
    assert c.get_voxel(Cache.voxel_key(stl, 16, "fixed", 40.0, "origin")) is None


def test_voxel_key_changes_with_the_voxelizer_code(tmp_path, monkeypatch):
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parent.parent / "dataset_gen"))
    from src import field_channels
    stl = tmp_path / "a.stl"
    stl.write_bytes(b"solid x")
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
    before = Cache.voxel_key(stl, 64, "fixed", 40.0, "origin")
    monkeypatch.setattr(field_channels, "DEFAULT_TRUNCATION", 7.0)          # a constant of the parity voxelizer's modules
    assert Cache.voxel_key(stl, 64, "fixed", 40.0, "origin") != before
    monkeypatch.undo()
    monkeypatch.delenv("SHAPE_RECO_VOXELIZER", raising=False)
    assert Cache.voxel_key(stl, 64, "fixed", 40.0, "origin") == before


def test_cache_key_follows_the_file_identity_and_the_parameters(tmp_path):
    import os
    from dataset_gen.src.cache_keys import cache_key
    f = tmp_path / "s.bin"
    f.write_bytes(b"\0" * 8)
    k = cache_key("t", files=[f], params=dict(a=1.0))
    assert k == cache_key("t", files=[f], params=dict(a=1.0))
    assert k != cache_key("t", files=[f], params=dict(a=2.0)) and k != cache_key("u", files=[f], params=dict(a=1.0))
    assert k != cache_key("t", files=[f], voxelizer="parity", params=dict(a=1.0))
    st = f.stat()
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))          # rewritten in place (revoxelize_parity.py)
    assert cache_key("t", files=[f], params=dict(a=1.0)) != k


def test_code_fingerprint_sees_code_and_constants_not_comments(tmp_path, monkeypatch):
    import sys
    from dataset_gen.src import cache_keys
    mod = tmp_path / "enc_mod_t19.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    versions = {
        "base": '"""Encoder."""\nSCALE = 16.0\n\ndef enc(x):\n    """Doc."""\n    return x / SCALE\n',
        "docs": '"""Encoder, documented."""\n# a comment\nSCALE = 16.0  # voxels\n\ndef enc(x):\n    """Other doc."""\n'
                '    return x / SCALE\n',
        "const": '"""Encoder."""\nSCALE = 8.0\n\ndef enc(x):\n    """Doc."""\n    return x / SCALE\n',
        "code": '"""Encoder."""\nSCALE = 16.0\n\ndef enc(x):\n    """Doc."""\n    return x * SCALE\n',
    }
    prints = {}
    for name, text in versions.items():
        mod.write_text(text)
        sys.modules.pop("enc_mod_t19", None)
        cache_keys.source_digest.cache_clear()
        prints[name] = cache_keys.code_fingerprint("enc_mod_t19")
    sys.modules.pop("enc_mod_t19", None)
    cache_keys.source_digest.cache_clear()
    assert prints["docs"] == prints["base"]                                  # comments and docstrings do not count
    assert prints["const"] != prints["base"] and prints["code"] != prints["base"]


def test_default_cache_switch(tmp_path, monkeypatch):
    try:
        disable_cache()
        assert default_cache() is None
        assert enable_cache(tmp_path / "x").root == tmp_path / "x" and default_cache() is not None
    finally:
        disable_cache()


# ------------------------------------------------------- in the loop
