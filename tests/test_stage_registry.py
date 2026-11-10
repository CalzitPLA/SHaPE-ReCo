"""reco_engine/stage_registry.py: load, filter, ordering, errors - and the $SHAPE_RECO_REGISTRY type override in
reco_engine/model_registry.py (default behaviour unchanged when the variable is unset)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine import model_registry as mr  # noqa: E402
from reco_engine.stage_registry import (DEFAULT_REGISTRY, StageEntry, entries_for, load_registry,  # noqa: E402
                                        load_registry_meta)


def _write(tmp_path, entries, **meta):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(dict(entries=entries, **meta)))
    return path


def _e(id_, stage, kinds, mode="analytic", cost=1, path=None, input_kind="occupancy", gate=None):
    return dict(id=id_, stage=stage, kinds=kinds, mode=mode, path=path, input_kind=input_kind, cost=cost, gate=gate, metrics={})


def test_repo_registry_loads_and_points_at_existing_dirs():
    reg = load_registry()
    assert reg and all(isinstance(e, StageEntry) for e in reg)
    assert {e.stage for e in reg} == {"type", "rotation", "dimensions", "position"}
    for e in reg:
        assert isinstance(e.kinds, tuple)
        if e.mode == "cnn":
            assert e.resolved_path().is_dir(), e.path
    # the choices the held-out numbers support: network-free sources first for box rotation and dimensions
    assert entries_for(reg, "rotation", "box")[0].id == "box_rot_frame"
    assert entries_for(reg, "dimensions", "box")[0].id == "box_dims_measured"
    assert load_registry_meta()["default_type_entry"] in {e.id for e in entries_for(reg, "type")}


def test_filter_and_cost_order(tmp_path):
    path = _write(tmp_path, [
        _e("b", "rotation", ["box"], cost=5),
        _e("a", "rotation", ["box", "cylinder"], cost=1),
        _e("c", "rotation", ["cylinder"], cost=3),
        _e("t", "type", ["box"], mode="cnn", path="x", cost=0),
    ])
    reg = load_registry(path)
    assert [e.id for e in entries_for(reg, "rotation")] == ["a", "c", "b"]
    assert [e.id for e in entries_for(reg, "rotation", "box")] == ["a", "b"]
    assert [e.id for e in entries_for(reg, "rotation", "cylinder")] == ["a", "c"]
    assert entries_for(reg, "rotation", "torus") == []
    assert [e.id for e in entries_for(reg, "type")] == ["t"]
    with pytest.raises(ValueError):
        entries_for(reg, "colour")


def test_plain_list_file_is_accepted(tmp_path):
    path = tmp_path / "list.json"
    path.write_text(json.dumps([_e("a", "position", ["box"])]))
    assert [e.id for e in load_registry(path)] == ["a"]


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_registry(tmp_path / "nope.json")


@pytest.mark.parametrize("bad", [
    _e("x", "colour", ["box"]),                                   # unknown stage
    _e("x", "type", ["box"], mode="guess"),                       # unknown mode
    _e("x", "type", ["box"], mode="cnn", path=None),              # cnn without a checkpoint
    _e("x", "type", ["box"], input_kind="pixels"),                # unknown input
    {"id": "x", "stage": "type", "kinds": ["box"], "mode": "prior"},   # no cost
])
def test_malformed_entries(tmp_path, bad):
    with pytest.raises(ValueError):
        load_registry(_write(tmp_path, [bad]))


def test_duplicate_id(tmp_path):
    with pytest.raises(ValueError):
        load_registry(_write(tmp_path, [_e("a", "type", ["box"]), _e("a", "type", ["sphere"])]))


def test_model_registry_override_default_off(monkeypatch):
    monkeypatch.delenv("SHAPE_RECO_REGISTRY", raising=False)
    assert mr.registry_type_dir() is None
    assert mr._type_dir(Path("/ckpt")) == Path("/ckpt/type_cnn")


def test_model_registry_override_picks_entry(monkeypatch):
    monkeypatch.setenv("SHAPE_RECO_REGISTRY", str(DEFAULT_REGISTRY))
    monkeypatch.delenv("SHAPE_RECO_REGISTRY_TYPE", raising=False)
    default = load_registry_meta()["default_type_entry"]
    entry = next(e for e in load_registry() if e.id == default and e.stage == "type")
    assert mr.registry_type_dir() == entry.resolved_path()
    monkeypatch.setenv("SHAPE_RECO_REGISTRY_TYPE", "type_hollow")
    assert mr._type_dir(Path("/ignored")) == ROOT / "checkpoints_hollow" / "type_cnn"
    kind, _trunc, channels = mr._type_cnn_input_kind(Path("/ignored"), "best_model.pth")
    assert (kind, channels) == ("occupancy", 1)


@pytest.mark.parametrize("entry", ["type_fused_net", "type_field_pertopo", "no_such_entry"])
def test_model_registry_override_rejects_unloadable(monkeypatch, entry):
    monkeypatch.setenv("SHAPE_RECO_REGISTRY", str(DEFAULT_REGISTRY))
    monkeypatch.setenv("SHAPE_RECO_REGISTRY_TYPE", entry)
    with pytest.raises(ValueError):
        mr.registry_type_dir()
