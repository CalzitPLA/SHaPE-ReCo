"""reco_engine/path_policy.py: the variants, their measured ranking, and the IoU a finished run is judged by."""

import pytest

from reco_engine import path_policy as pp


def test_variant_of_and_apply_round_trip():
    bare = dict(cascade_start_shape="legacy", cascade_type_source="registry", cascade_subparts=False, other=1)
    assert pp.variant_of(bare) == "bare"
    for name in pp.VARIANTS:
        s = pp.apply(bare, name)
        assert pp.variant_of(s) == name and s["other"] == 1
    assert pp.variant_of(dict(cascade_start_shape="cubes", cascade_subparts=True)) == "custom"
    assert pp.variant_of(dict(cascade_registry="/somewhere/else.json")) == "custom"
    assert pp.run_part_args("cubes") == ["--start", "cubes"] and pp.run_part_args("bare") == []
    assert (pp.REPO_ROOT / "models" / "registry_analytic.json").is_file()
    assert pp.run_part_args("analytic") == ["--registry", "models/registry_analytic.json"]


def test_rank_measured_first_then_unmeasured_in_order_and_skips_tried():
    weights = dict(variants={"sub-parts": dict(p_good=0.60, n=300, source="s", p_rescue=0.20, n_fail=150, p_break=0.05),
                             "auto": dict(p_good=0.90, n=12, source="c", p_rescue=0.5, n_fail=2),
                             "cubes": dict(p_good=0.99, n=60, source="d"),            # retired: never ranked
                             "cnn": dict(p_good=1.0, n=1, source="liner")})
    rows = pp.rank(tried=["bare"], weights=weights)
    names = [r["variant"] for r in rows]
    assert "bare" not in names and not set(names) & set(pp.RETIRED)
    # a paired rescue rate ranks first; auto has only 2 paired failures, so its share of good runs counts (12 runs)
    assert names == ["sub-parts", "auto", "leaves"]                       # leaves: unmeasured, in the variants' order
    assert rows[0]["by_rescue"] and rows[1]["measured"] and not rows[1]["by_rescue"] and not rows[-1]["measured"]
    assert "rescued 0.20 of 150" in pp.describe(rows[0]) and "lost 0.05" in pp.describe(rows[0])
    assert "0.90 of 12" in pp.describe(rows[1]) and "not measured" in pp.describe(rows[-1])


def test_iou_of_done_prefers_the_exact_grade():
    assert pp.iou_of_done(dict(stats=dict(iou=0.81))) == 0.81
    assert pp.iou_of_done(dict(stats=dict(iou=0.81), grade=dict(iou=0.97))) == 0.97
    assert pp.iou_of_done(dict(grade="failed", stats={})) is None and pp.iou_of_done(None) is None


def test_settings_panel_applies_a_path_and_collects_the_retry_fields():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from reco_gui.settings_panel import SettingsPanel
    panel = SettingsPanel()
    panel.method.setCurrentIndex(max(0, panel.method.findData("cascade")))
    panel.set_input(str(pp.REPO_ROOT / "dataset_gen" / "examples" / "l_bracket.stl"))
    s = panel.collect()
    assert pp.variant_of(s) == "bare" and s.retry_iou == 0.95 and s.retry_max == 3
    for name in ("sub-parts", "auto", "leaves", "bare"):
        panel.apply_path(pp.VARIANTS[name]["overrides"])
        assert pp.variant_of(panel.collect()) == name
    # the retired paths (method grading of 2026-10-30) are not offered; their runs keep their name
    assert panel.c_start.findData("cubes") == -1 and panel.c_start.findData("compartments") == -1
    assert panel.c_type_source.findData("cnn") == -1 and panel.c_type_source.findData("descriptors") == -1
    assert pp.variant_of(dict(cascade_start_shape="cubes")) == "cubes" and pp.run_part_args("cubes") == ["--start", "cubes"]
    del app


def test_every_measured_weight_names_a_variant():
    """docs/data/path_weights.json is written from the run directories (runs_<variant>); every key must be a VARIANTS
    key, else rank() silently ranks that variant as unmeasured (the "auto start" vs "auto" mismatch of 2026-10-10)."""
    w = pp.load_weights()
    if not w:
        pytest.skip("no docs/data/path_weights.json")
    unknown = sorted(set(w.get("variants", {})) - set(pp.KNOWN))           # retired variants are known
    assert unknown == [], f"weights for unknown variants: {unknown}"
    assert "auto" in w.get("variants", {}) and pp.rank(weights=w)[0]["measured"]
