"""Registry "legacy": true (decision D-007): legacy type entries stay loadable (tests, $SHAPE_RECO_REGISTRY) but the
cascade's type stage does not run them, so production fuses exactly the sources gate T measured
(docs/data/gate_t_medium.md: descriptors + type_hier_dirlog_inside_medium)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reco_engine import cascade  # noqa: E402
from reco_engine.stage_registry import StageEntry, load_registry  # noqa: E402


def _entry(id_, legacy=False, mode="analytic"):
    return StageEntry(id=id_, stage="type", kinds=("box", "sphere"), mode=mode, path=None, input_kind="occupancy",
                      cost=1, gate=None, metrics={}, legacy=legacy)


def test_type_stage_skips_legacy_entries(monkeypatch):
    ran = []

    def fake(entry, ctx, kinds):
        ran.append(entry.id)
        return cascade.TypeAnswer({k: 1.0 / len(kinds) for k in kinds})

    monkeypatch.setattr(cascade, "_run_type_entry", fake)
    ctx = type("Ctx", (), {"channels": None, "fits": {}})()
    cascade._type_stage(ctx, [_entry("old", legacy=True), _entry("new")], ["box", "sphere"])
    assert ran == ["new"]


def test_default_registry_votes_descriptors_and_dirlog_inside_only():
    active = sorted(e.id for e in load_registry() if e.stage == "type" and not e.legacy)
    assert active == ["type_descriptors_v5", "type_hier_dirlog_inside_medium"]
    new = next(e for e in load_registry() if e.id == "type_hier_dirlog_inside_medium")
    assert new.input_kind == "dirlog_inside" and new.mode == "cnn" and 0 < new.metrics["parent_acc"] < 1
