"""The dataset-generation window driven headlessly (offscreen Qt): a real, tiny src.dataset_generator run through
the actual QProcess wiring (not mocked - dataset_gen/tests already run real FreeCAD for small datasets, same
convention here), checking the settings panel collects sane args, shards run and finish, the log fills in, and the
summary reflects what was actually generated."""
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")
SANDBOX_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SANDBOX_ROOT))
sys.path.insert(0, str(SANDBOX_ROOT / "dataset_gen"))

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from reco_gui.dataset_gen_window import DatasetGenWindow  # noqa: E402
from src.freecad_path import find_freecadcmd  # noqa: E402

needs_freecad = pytest.mark.skipif(find_freecadcmd() is None, reason="freecadcmd not found ($SHAPE_RECO_FREECADCMD, "
                                   "PATH, /usr/bin/freecadcmd)")


@pytest.fixture(scope="module")
def qapp():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_until_idle(app, window, timeout=120.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if not window._processes or all(p.state() == QtCore.QProcess.NotRunning for p, _ in window._processes):
            return True
        time.sleep(0.02)
    return False


def test_settings_panel_defaults_are_a_valid_run(qapp):
    window = DatasetGenWindow()
    settings = window.settings_panel.collect()
    assert settings.mode == "type" and settings.topologies == ("box", "sphere")
    assert settings.output_dir == Path(SANDBOX_ROOT / "dataset_gen" / "local_dataset")


@needs_freecad
def test_generate_runs_a_tiny_real_dataset_and_summarizes_it(qapp, tmp_path):
    window = DatasetGenWindow()
    window.settings_panel._select_topologies(("box",))
    window.settings_panel.count_type.setValue(2)
    window.settings_panel.output_dir.setText(str(tmp_path / "out"))
    window.settings_panel.hollow_prob.setValue(1.0)
    window.generate()
    assert window.act_go.isEnabled() is False and window.act_cancel.isEnabled() is True
    assert wait_until_idle(qapp, window), "dataset generation did not finish in time"
    assert "Done: 1 shard" in window.progress.text()
    assert window.act_go.isEnabled() is True and window.act_cancel.isEnabled() is False
    assert "Generated: 2" in window.log.toPlainText()
    summary = window.summary.toPlainText()
    assert "total samples so far: 2" in summary
    assert "'box': 2" in summary
    assert "hollow: 2" in summary


@needs_freecad
def test_shards_write_to_separate_subdirs_with_offset_seeds(qapp, tmp_path):
    window = DatasetGenWindow()
    window.settings_panel._select_topologies(("box",))
    window.settings_panel.count_type.setValue(1)
    window.settings_panel.output_dir.setText(str(tmp_path / "out"))
    window.settings_panel.shards.setValue(2)
    window.settings_panel.seed.setValue(1000)
    window.generate()
    assert wait_until_idle(qapp, window)
    assert "Done: 2 shard" in window.progress.text()
    assert (tmp_path / "out" / "shard0" / "box").is_dir()
    assert (tmp_path / "out" / "shard1" / "box").is_dir()
    assert "total samples so far: 2" in window.summary.toPlainText()


def test_cancel_stops_running_processes(qapp, tmp_path):
    window = DatasetGenWindow()
    window.settings_panel._select_topologies(("box",))
    window.settings_panel.count_type.setValue(50)                # enough to still be running when we cancel
    window.settings_panel.output_dir.setText(str(tmp_path / "out"))
    window.generate()
    qapp.processEvents()
    window.cancel()
    assert wait_until_idle(qapp, window, timeout=15.0)
    assert window.progress.text() == "Cancelled."
