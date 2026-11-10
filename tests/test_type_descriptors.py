"""reco_engine/type_descriptors.py: the 27 descriptors, fit / save / load / predict_proba on a slice of v5 small (CPU, a
temporary joblib file), the CLI, and the two baseline scripts importing the feature code instead of copying it."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine import type_descriptors as td  # noqa: E402
from training.subclass_schema import FLAT_SUBCLASSES, PARENTS  # noqa: E402

SMALL = ROOT / "dataset_gen" / "local_dataset_v5" / "small"
pytestmark = pytest.mark.skipif(not (SMALL / "train").is_dir(), reason="v5 small dataset not there")


def _grid(fn):
    c = np.arange(64) - 31.5
    z, y, x = np.meshgrid(c, c, c, indexing="ij")
    return fn(x, y, z)


def test_descriptors_are_invariant_and_topological():
    cube = _grid(lambda x, y, z: (np.abs(x) < 12) & (np.abs(y) < 12) & (np.abs(z) < 12))
    f = td.descriptors(cube)
    assert f.shape == (len(td.FEATURES),) == (27,) and np.all(np.isfinite(f))
    names = dict(zip(td.FEATURES, f))
    assert names["chi"] == 1 and names["b0"] == 1 and names["b2_cavities"] == 0 and names["b1_tunnels"] == 0
    assert names["convex_fill"] == pytest.approx(1.0, abs=0.05)
    assert np.array_equal(td.descriptors(np.transpose(cube, (2, 0, 1))), f)                 # a rotation by 90 degrees
    assert np.array_equal(td.descriptors(cube.astype(np.float32) * 2 - 1), f)                # the nets' {-1, +1} too
    hollow = cube & ~_grid(lambda x, y, z: (np.abs(x) < 6) & (np.abs(y) < 6) & (np.abs(z) < 6))
    h = dict(zip(td.FEATURES, td.descriptors(hollow)))
    assert h["b2_cavities"] == 1 and h["chi"] == 2 and h["cavity_frac"] > 0
    tube = _grid(lambda x, y, z: (np.hypot(x, y) < 12) & (np.hypot(x, y) > 6) & (np.abs(z) < 12))
    assert dict(zip(td.FEATURES, td.descriptors(tube)))["b1_tunnels"] == 1


def test_fit_save_load_predict(tmp_path):
    model = td.fit(SMALL / "train", limit=48, verbose=False)
    assert model.features == td.FEATURES and model.parents == PARENTS and model.subclasses == FLAT_SUBCLASSES
    assert model.meta["n_train"] == 48 and model.meta["fingerprint"] == td.source_fingerprint()
    out = td.save(model, tmp_path / "desc.joblib")
    loaded = td.load(out)
    X, parent, sub, recs, _ = td.dataset_features(SMALL / "test", limit=6)
    for r, p_true in zip(recs, parent):
        occ = np.fromfile(r.voxel_path, dtype=np.uint8).reshape(64, 64, 64) > 0
        parents, subs = loaded.predict_proba(occ)
        assert set(parents) == set(PARENTS) and set(subs) == set(FLAT_SUBCLASSES)
        assert sum(parents.values()) == pytest.approx(1.0) and sum(subs.values()) == pytest.approx(1.0)
        a, b = model.predict_proba(occ)
        assert a == parents and b == subs
        within = loaded.subclasses_of("box", subs)
        assert set(within) == {"full", "hollow", "ibeam", "rhs", "frame"} and sum(within.values()) == pytest.approx(1.0)
    ev = td.evaluate(loaded, SMALL / "test", limit=20)
    assert ev["n"] == 20 and 0.0 <= ev["parent_acc"] <= 1.0 and ev["parent_top2"] >= ev["parent_acc"]
    with pytest.raises(FileNotFoundError):
        td.load(tmp_path / "none.joblib")


def test_cli_fits_and_writes_the_model(tmp_path, capsys):
    out = tmp_path / "m.joblib"
    td.main(["--fit", str(SMALL / "train"), "--out", str(out), "--eval", str(SMALL / "test"), "--limit", "40"])
    text = capsys.readouterr().out
    assert out.is_file() and "wrote" in text and "parent_acc" in text
    m = td.load(out)
    assert m.meta["eval"]["n"] == 40 and m.meta["train_roots"] == [str(SMALL / "train")]


def test_descriptor_model_through_the_cascade(tmp_path):
    """A fitted model as a registry "descriptor" entry (models={id: model}) answers the type stage with a subclass."""
    from reco_engine import cascade
    from reco_engine.cascade import _Entry
    from src.field_channels import voxel_channels
    model = td.fit(SMALL / "train", limit=48, verbose=False)
    spec = cascade.grid_spec(-12, 12, 32)
    c = spec.centres()
    occ = (np.abs(c[..., 0]) <= 6) & (np.abs(c[..., 1]) <= 4) & (np.abs(c[..., 2]) <= 3)
    reg = [_Entry("desc", "type", cascade.KINDS, "descriptor", str(tmp_path / "x.joblib"), cost=1)]
    ts = cascade.type_stage(voxel_channels(occ, spec), spec, reg, models={"desc": model})
    assert ts.mode == "descriptor:desc" and set(ts.probs) == set(cascade.KINDS) and ts.subclass is not None
    assert ts.as_dict()["entries"][0]["weight"] == cascade.DEFAULT_TYPE_WEIGHT


def test_the_baseline_scripts_import_the_library():
    for name, attrs in (("eval_minkowski_baseline", {"descriptors": td.minkowski_descriptors, "cell_counts": td.cell_counts}),
                        ("eval_descriptor_baseline_v5", {"features": td.features, "extra_descriptors": td.extra_descriptors})):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr, fn in attrs.items():
            assert getattr(mod, attr) is fn
        assert tuple(mod.FEATURES) == (td.MINKOWSKI_FEATURES if name == "eval_minkowski_baseline" else td.FEATURES)
