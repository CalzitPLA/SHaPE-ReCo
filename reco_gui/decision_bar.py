#!/usr/bin/env python3
"""
The decision bar: the question "keep this result or try another path?" as a bar above the viewport instead of a modal
box (the user's rule of 2026-10-15: the decision window blocked the viewport). The viewport, the docks and the toolbar
stay usable while the question is open; the bar offers the paths (reco_engine/path_policy.py), the comparison export
(reco_gui/nephogram.py: the guess and the input as STL and as distance-coloured PLY) and opens CloudCompare on it.

    bar = DecisionBar(); bar.ask(text, ranked)         # ranked: path_policy.rank(...) rows
    bar.tryRequested[str]  bar.chooseRequested  bar.keepRequested  bar.exportRequested
    bar.exported(info)                                  # after the export: Open in CloudCompare / Open folder
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from PyQt5 import QtCore, QtGui, QtWidgets


class ExportThread(QtCore.QThread):
    done = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, input_mesh, model_mesh, out_dir, name: str, input_path=None, result=None):
        super().__init__()
        self.args = (input_mesh, model_mesh, out_dir, name)
        self.input_path, self.result = input_path, result

    def run(self):
        try:
            from reco_gui.nephogram import write_nephogram
            model_step = None
            if self.result is not None and self.input_path is not None:          # the program as STEP: exact distances
                try:
                    from reco_engine.result_to_program import program_model
                    from reco_field.export import build_in_freecad
                    ex = build_in_freecad(program_model(self.result), self.args[2], "model", timeout=900)
                    model_step = str(ex.step)
                except Exception as exc:                                        # the mesh field stands in
                    model_step = None
                    self.note = f"model STEP not built: {exc}"
            self.done.emit(write_nephogram(*self.args, input_path=self.input_path, model_step=model_step))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class DecisionBar(QtWidgets.QFrame):
    tryRequested = QtCore.pyqtSignal(str)
    chooseRequested = QtCore.pyqtSignal()
    keepRequested = QtCore.pyqtSignal()
    exportRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("DecisionBar { background: #fff6d5; border: 1px solid #d8b84a; }")
        self.ranked: List[Dict] = []
        self.export_info: Optional[Dict] = None
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        self.text = QtWidgets.QLabel()
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.text, 1)
        self.b_try = QtWidgets.QPushButton("Try")
        self.b_choose = QtWidgets.QPushButton("Choose...")
        self.b_keep = QtWidgets.QPushButton("Keep this result")
        self.b_export = QtWidgets.QPushButton("Export for comparison")
        self.b_export.setToolTip("The guess and the input as STL and as PLY coloured by the surface distance (HSV nephogram, "
                                 "scalar field 'distance'), with the statistics - for FreeCAD or CloudCompare")
        self.b_freecad = QtWidgets.QPushButton("Open in FreeCAD")
        self.b_freecad.setToolTip("FreeCAD with nephogram.FCMacro: the input and the guess as Mesh objects coloured by their surface distance")
        self.b_cloud = QtWidgets.QPushButton("Open in CloudCompare")
        self.b_folder = QtWidgets.QPushButton("Open folder")
        for b in (self.b_try, self.b_choose, self.b_keep, self.b_export, self.b_freecad, self.b_cloud, self.b_folder):
            lay.addWidget(b)
        self.b_freecad.hide()
        self.b_cloud.hide()
        self.b_folder.hide()
        self.b_try.clicked.connect(lambda: self.tryRequested.emit(self.ranked[0]["variant"] if self.ranked else ""))
        self.b_choose.clicked.connect(self.chooseRequested)
        self.b_keep.clicked.connect(self.keepRequested)
        self.b_export.clicked.connect(self.exportRequested)
        self.b_freecad.clicked.connect(self.open_freecad)
        self.b_cloud.clicked.connect(self.open_cloudcompare)
        self.b_folder.clicked.connect(self.open_folder)
        self.hide()

    def ask(self, text: str, ranked: List[Dict]) -> None:
        from reco_engine import path_policy
        self.ranked = list(ranked)
        self.export_info = None
        self.text.setText(text)
        self.text.setToolTip("\n".join(path_policy.describe(r) for r in ranked))
        self.b_try.setText(f"Try {ranked[0]['variant']}" if ranked else "Try")
        self.b_try.setEnabled(bool(ranked))
        self.b_choose.setVisible(len(ranked) > 1)
        self.b_export.setEnabled(True)
        self.b_export.setText("Export for comparison")
        self.b_freecad.hide()
        self.b_cloud.hide()
        self.b_folder.hide()
        self.show()

    def exporting(self) -> None:
        self.b_export.setEnabled(False)
        self.b_export.setText("Exporting...")

    def exported(self, info: Dict) -> None:
        from reco_gui.nephogram import cloudcompare_command, freecad_command
        self.export_info = info
        s = info["stats"]
        self.b_export.setText(f"Exported (chamfer {info['chamfer']:.3g}, max {max(s['input_to_model']['max'], s['model_to_input']['max']):.3g})")
        self.b_export.setToolTip(info["dir"] + "\n" + (Path(info["dir"]) / "README.txt").read_text())
        if info.get("legend"):
            self.text.setToolTip("distance bands: " + "; ".join(b["text"] for b in info["legend"]))
        self.b_folder.show()
        self.b_freecad.setVisible(freecad_command() is not None and bool(info["files"].get("macro")))
        self.b_cloud.setVisible(cloudcompare_command() is not None)

    def open_freecad(self) -> None:
        from reco_gui.nephogram import open_in_freecad
        if self.export_info and self.export_info["files"].get("macro"):
            open_in_freecad(self.export_info["files"]["macro"])

    def export_failed(self, text: str) -> None:
        self.b_export.setEnabled(True)
        self.b_export.setText("Export failed")
        self.b_export.setToolTip(text)

    def open_cloudcompare(self) -> None:
        from reco_gui.nephogram import open_in_cloudcompare
        if self.export_info:
            open_in_cloudcompare(self.export_info["files"]["ply"])

    def open_folder(self) -> None:
        if self.export_info:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.export_info["dir"]))
