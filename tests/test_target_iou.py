"""
The level of detail: which congruence groups the feature phase takes for a target IoU (residual_loop.select_for_target),
and the settings' slider that sets it.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["RECO_GUI_NO_3D"] = "1"
SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from reco_engine.residual_loop import ResidualConfig, select_for_target  # noqa: E402

# the liner's first difference, rounded: input 45.95e6, invented 7.48e6; groups of invented pieces
V_IN, INVENTED = 45.95e6, 7.48e6
GROUPS = [(0.0, 3.38e6), (0.0, 1.22e6), (0.0, 9 * 86.5e3), (0.0, 0.35e6), (0.0, 0.46e6), (0.0, 36 * 2909.0), (0.0, 1.19e6)]


def test_no_target_takes_everything():
    keep, est = select_for_target(GROUPS, 0.0, INVENTED, V_IN, None)
    assert keep == list(range(len(GROUPS)))
    assert est == pytest.approx(V_IN / (V_IN + max(0.0, INVENTED - sum(g[1] for g in GROUPS))))


def test_a_lower_target_takes_the_largest_groups_only():
    keep, est = select_for_target(GROUPS, 0.0, INVENTED, V_IN, 0.93)
    assert keep == [0, 1] and est == pytest.approx(V_IN / (V_IN + INVENTED - 4.60e6))      # 0.941
    keep, est = select_for_target(GROUPS, 0.0, INVENTED, V_IN, 0.95)
    assert keep == [0, 1, 6] and est >= 0.95
    more, est_more = select_for_target(GROUPS, 0.0, INVENTED, V_IN, 0.97)
    assert more == [0, 1, 2, 6] and est_more >= 0.97           # the bolt holes (5) are detail at this level


def test_a_target_out_of_reach_takes_everything():
    keep, est = select_for_target(GROUPS[:6], 0.0, INVENTED, V_IN, 0.99)
    assert keep == list(range(6)) and est == pytest.approx(V_IN / (V_IN + 1.19e6), rel=1e-3)


def test_missing_material_counts_too():
    keep, est = select_for_target([(2.0, 0.0), (0.0, 1.0)], 2.0, 1.0, 10.0, 0.95)
    assert keep == [0, 1] and est == pytest.approx(1.0)
    keep, est = select_for_target([(2.0, 0.0), (0.0, 1.0)], 2.0, 1.0, 10.0, 0.9)
    assert keep == [0] and est == pytest.approx(10.0 / 11.0)


def test_the_config_has_the_field():
    assert ResidualConfig().feature_target_iou is None
    assert ResidualConfig(feature_target_iou=0.95).feature_target_iou == 0.95


def test_slider():
    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets
    from reco_gui.cascade_session import make_cascade_session
    from reco_gui.settings_panel import SettingsPanel
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])  # noqa: F841
    panel = SettingsPanel()
    assert panel.target_iou() is None and panel.c_target_label.text() == "all details"
    panel.set_target_iou(0.95)
    assert panel.target_iou() == 0.95 and panel.c_target_label.text() == "0.950"
    panel.c_target.setValue(panel.c_target.maximum())
    assert panel.target_iou() is None
    panel.set_target_iou(0.5)                                   # below the slider: its left end
    assert panel.target_iou() == 0.8
    seen = []
    make_cascade_session("part.stl", steps_fn=lambda path, config: seen.append(config) or iter(()), feature_target_iou=0.95)
    assert seen[0]["feature_target_iou"] == 0.95
