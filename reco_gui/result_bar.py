#!/usr/bin/env python3
"""
The bar above the 3D view when a part is reconstructed (the user, 2026-10-29: "when the part is reconstructed enable a
method to get into the compare mode"): what came out, and the way into the comparison - Compare (the window's compare
mode: the Benchmark tab with the run checked and its histograms, the input coloured by its distance to the model) or
the Benchmark tab alone. Never a modal box: the model can still be turned, the docks opened.
"""

from __future__ import annotations

from typing import List, Optional

from PyQt5 import QtCore, QtWidgets


class ResultBar(QtWidgets.QFrame):
    compareRequested = QtCore.pyqtSignal()
    benchmarkRequested = QtCore.pyqtSignal()
    postprocessRequested = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("resultBar")
        self.setStyleSheet("#resultBar { background: #e6f2e8; border: 1px solid #7fb48a; border-radius: 4px; }")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        self.label = QtWidgets.QLabel("")
        self.label.setWordWrap(True)
        lay.addWidget(self.label, 1)
        self.compare = QtWidgets.QPushButton("Compare ▸")
        self.compare.setToolTip("the compare mode: the input coloured by its distance to the model (the nephogram), the "
                                "Benchmark tab with this run checked and its histograms - step 4 of the flow")
        self.compare.setDefault(True)
        self.compare.clicked.connect(self.compareRequested.emit)
        lay.addWidget(self.compare)
        self.bench = QtWidgets.QPushButton("Benchmark")
        self.bench.setToolTip("the Benchmark tab of this part: every stored run, local and server")
        self.bench.clicked.connect(self.benchmarkRequested.emit)
        lay.addWidget(self.bench)
        self.post = QtWidgets.QPushButton("Post-process ▸")
        self.post.setToolTip("the reconstruction's tree with the mass ratio of every part: select, benchmark, minimize the "
                             "number of parts, decide and export")
        self.post.clicked.connect(self.postprocessRequested.emit)
        lay.addWidget(self.post)
        close = QtWidgets.QToolButton()
        close.setText("×")
        close.setToolTip("hide the bar (Compare stays in the flow, step 4)")
        close.clicked.connect(self.hide)
        lay.addWidget(close)
        self.run_dirs: List[str] = []
        self.hide()

    def show_result(self, text: str, run_dirs: Optional[List[str]] = None) -> None:
        """Show the bar: `text` what came out, `run_dirs` the stored runs Compare starts with (the first is coloured)."""
        self.run_dirs = [str(d) for d in (run_dirs or []) if d]
        self.label.setText(text)
        self.show()
