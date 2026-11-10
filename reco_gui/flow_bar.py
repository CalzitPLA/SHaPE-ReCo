#!/usr/bin/env python3
"""
The window's flow as one line of steps (the user, 2026-10-29: "the flow through the application is not that clear
anymore for the user: make it linear"):

    1 Part  >  2 Starting part  >  3 Reconstruct  >  4 Compare  >  5 Choose & export          [Next: ... >]

Every step is a button: done ones carry a check mark, the current one is highlighted, the next ones are plain; a short
note under each says what it holds (the part's name, "automatic", "3 reconstructions", the chosen run). "Next" does the
current step. The window (reco_gui/app.py) decides what a step does and when it is done; this widget only shows it.
"""

from __future__ import annotations

from typing import List

from PyQt5 import QtCore, QtWidgets

STEPS = ("Part", "Starting part", "Reconstruct", "Compare", "Choose && export")
TIPS = ("Choose the part: an example, Browse ... or Random (the Settings panel)",
        "Define the starting part (optional): automatic, a basic shape around the part or a sub-part, or an exact shape",
        "Run the reconstruction; it goes on with further paths until the set number of different reconstructions exists",
        "Compare the reconstructions: nephogram and histograms, the event time in aegir, ranked in the Benchmark tab",
        "Choose one result and take it on: FreeCAD (rebuild, groups), GDML for Geant4, the run store")
NUMBERS = "①②③④⑤"                       # (1) .. (5)
STYLE = {"current": "QToolButton{background:#dbe8fb;border:1px solid #2a78d6;border-radius:4px;font-weight:bold;padding:2px 8px}",
         "done": "QToolButton{color:#1f6b2a;border:1px solid #c9d8c9;border-radius:4px;padding:2px 8px}",
         "open": "QToolButton{color:#52514e;border:1px solid transparent;border-radius:4px;padding:2px 8px}"}


class FlowBar(QtWidgets.QWidget):
    stepClicked = QtCore.pyqtSignal(int)
    nextClicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current, self.done = 0, set()
        self.notes: List[str] = [""] * len(STEPS)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(2)
        self.buttons: List[QtWidgets.QToolButton] = []
        for i, (name, tip) in enumerate(zip(STEPS, TIPS)):
            if i:
                arrow = QtWidgets.QLabel("▸")
                arrow.setStyleSheet("color:#8a8984")
                lay.addWidget(arrow)
            b = QtWidgets.QToolButton()
            b.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            b.setToolTip(tip.replace("&&", "&"))
            b.clicked.connect(lambda _=False, k=i: self.stepClicked.emit(k))
            self.buttons.append(b)
            lay.addWidget(b)
        lay.addStretch(1)
        self.next = QtWidgets.QPushButton()
        self.next.setDefault(True)
        self.next.clicked.connect(self.nextClicked.emit)
        lay.addWidget(self.next)
        self._paint()

    def set_current(self, i: int) -> None:
        self.current = max(0, min(len(STEPS) - 1, int(i)))
        self._paint()

    def set_done(self, i: int, done: bool = True) -> None:
        (self.done.add if done else self.done.discard)(int(i))
        self._paint()

    def set_note(self, i: int, text: str) -> None:
        self.notes[int(i)] = text or ""
        self._paint()

    def state(self, i: int) -> str:
        return "current" if i == self.current else "done" if i in self.done else "open"

    def _paint(self) -> None:
        for i, b in enumerate(self.buttons):
            mark = "✓ " if i in self.done and i != self.current else ""
            note = self.notes[i]
            if len(note) > 20:                                   # the bar must fit a 1024 px screen
                note = note[:19] + "…"
            b.setText(f"{mark}{NUMBERS[i]} {STEPS[i]}" + (f"\n{note}" if note else ""))
            b.setToolTip(TIPS[i].replace("&&", "&") + (f"\n\n{self.notes[i]}" if self.notes[i] else ""))
            b.setStyleSheet(STYLE[self.state(i)])
        self.next.setText(f"Next {NUMBERS[self.current]} ▸")
        self.next.setToolTip(TIPS[self.current].replace("&&", "&"))
