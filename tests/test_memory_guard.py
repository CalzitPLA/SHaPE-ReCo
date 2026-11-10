"""reco_engine/memory_guard.py: the watcher reads, releases once, and names the reason; the loop stops a run on it."""

import numpy as np

from reco_engine import memory_guard as mg


def test_guard_releases_once_then_reports():
    readings = iter([5000.0, 900.0, 950.0, 3000.0])
    freed = []
    g = mg.MemoryGuard(min_free_mb=1200, min_interval_s=0.0, reader=lambda: next(readings))
    g.releases.append(lambda: freed.append(1))
    assert g.check() is None and freed == []                       # 5000 MB: fine
    r = g.check()                                                  # 900 -> release -> 950: still low
    assert r is not None and "memory low" in r and "950 MB" in r and freed == [1] and g.low_count == 1
    assert g.check() is None                                       # 3000: fine again, the count resets
    assert g.low_count == 0
    assert mg.MemoryGuard(min_free_mb=0, reader=lambda: 1.0).check() is None       # off
    assert mg.MemoryGuard(min_free_mb=1200, reader=lambda: None).check() is None   # unknown never stops anything
    assert mg.available_mb() is None or mg.available_mb() > 0


def test_loop_stops_cleanly_when_memory_is_low(monkeypatch):
    from reco_engine import residual_loop as rl
    monkeypatch.setattr(mg, "available_mb", lambda: 100.0)         # every reading: 100 MB left
    c = rl.ResidualConfig(resolution=32, max_resolution=32, grid="cube", freecad=False, feature_phase=False,
                          voxel_fallback=False, field_backend="igl", polish=False, memory_min_free_mb=1200)
    steps = list(rl.residual_steps("dataset_gen/examples/bs_box.stl", c))
    assert steps[-1].kind == "done" and "memory low" in steps[-1].info.get("stopped", "")
    assert sum(s.kind == "accepted" for s in steps) == 0           # stopped before any round
