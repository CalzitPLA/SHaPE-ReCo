#!/usr/bin/env python3
"""
The starting part, defined by the user (the user, 2026-10-29: "enable a method to define a starting part for the user:
by a basic shape type and the bounding part or sub-part, or by giving exact location and size; define it via a
dialog"). Step 2 of the window's flow (reco_gui/flow_bar.py).

Three ways:
  automatic        the loop chooses its starting shape (the path's own start)
  around the part  a basic shape type that bounds the whole part or a sub-part (a box region in mm); the axis x, y, z
                   or automatic (reco_engine.user_guess.bounding_shape)
  exact            the shape type at an exact centre, turned by angles about x, y, z, with exact sizes
                   (reco_engine.user_guess.exact_shape)

The result is the loop's first guess (ResidualConfig.first_guess, the contract in reco_engine/user_guess.py): the shape
is the first add, judged by the loop's own checks. A shape around the part is a cover (user_guess role "cover"): kept
when it holds the part's material, its excess cut away later, never refitted; an exact shape is a cover when ticked,
else it must gain material like any add, and "let the loop fit it to the surface" refines it first (a rough guess),
unticked it is used exactly as given ("refined": true). The hint says how much of the part the shape holds and
how much of the shape lies in the part.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from reco_engine import user_guess as ug

MODES = (("automatic", "Automatic: the reconstruction chooses its starting shape"),
         ("around", "Around the part: a basic shape that bounds the part or a sub-part"),
         ("exact", "Exact: a shape at a given location with given sizes"))


def _spin(lo=-1e7, hi=1e7, value=0.0, decimals=3, suffix=" mm") -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setValue(value)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class StartPartDialog(QtWidgets.QDialog):
    """The dialog; `entry()` is the first-guess entry (a dict) or None for automatic, `summary()` a short text."""

    def __init__(self, vertices, faces, parent=None, part_name: str = "the part"):
        super().__init__(parent)
        self.setWindowTitle("Starting part")
        self.V, self.F = np.asarray(vertices, float), np.asarray(faces)
        self.lo, self.hi = self.V.min(axis=0), self.V.max(axis=0)
        self._shape: Optional[ug.UserShape] = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(QtWidgets.QLabel(f"<b>Starting part of {part_name}</b>: the first shape the reconstruction "
                                       f"starts from; it then cuts and adds until the part is explained."))
        self.mode_buttons = {}
        group = QtWidgets.QButtonGroup(self)
        for key, text in MODES:
            b = QtWidgets.QRadioButton(text)
            group.addButton(b)
            self.mode_buttons[key] = b
            lay.addWidget(b)
            b.toggled.connect(self._mode_changed)
        kind_row = QtWidgets.QHBoxLayout()
        kind_row.addWidget(QtWidgets.QLabel("shape type"))
        self.kind = QtWidgets.QComboBox()
        for k in ug.KINDS:
            self.kind.addItem(k, k)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        kind_row.addWidget(self.kind, 1)
        lay.addLayout(kind_row)

        # ---- around the part / a sub-part
        self.around = QtWidgets.QGroupBox("Around")
        af = QtWidgets.QFormLayout(self.around)
        self.whole = QtWidgets.QRadioButton("the whole part")
        self.sub = QtWidgets.QRadioButton("a sub-part: the part inside this box (mm)")
        self.whole.setChecked(True)
        sub_group = QtWidgets.QButtonGroup(self)
        sub_group.addButton(self.whole)
        sub_group.addButton(self.sub)
        af.addRow(self.whole)
        af.addRow(self.sub)
        self.region = {}
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("from"), 1, 0)
        grid.addWidget(QtWidgets.QLabel("to"), 2, 0)
        for j, ax in enumerate("xyz"):
            grid.addWidget(QtWidgets.QLabel(ax), 0, j + 1, alignment=QtCore.Qt.AlignCenter)
            for i, end in enumerate(("lo", "hi")):
                s = _spin(value=float((self.lo if end == "lo" else self.hi)[j]))
                s.valueChanged.connect(self._update)
                self.region[(end, j)] = s
                grid.addWidget(s, i + 1, j + 1)
        af.addRow(grid)
        self.axis = QtWidgets.QComboBox()
        for key, text in (("auto", "automatic (the tightest)"), ("x", "x"), ("y", "y"), ("z", "z")):
            self.axis.addItem(text, key)
        self.axis.setToolTip("a box: automatic = the oriented box, x / y / z = aligned with the world axes; a cylinder, "
                             "cone or torus: its axis")
        af.addRow("axis", self.axis)
        for w in (self.whole, self.sub):
            w.toggled.connect(self._update)
        self.axis.currentIndexChanged.connect(self._update)
        lay.addWidget(self.around)

        # ---- exact location and size
        self.exact = QtWidgets.QGroupBox("Exact")
        ef = QtWidgets.QFormLayout(self.exact)
        self.centre = [_spin(value=float((self.lo[j] + self.hi[j]) / 2)) for j in range(3)]
        self.angles = [_spin(-360.0, 360.0, 0.0, 2, " deg") for _ in range(3)]
        self.sizes = [_spin(1e-6, 1e7, 1.0) for _ in range(3)]
        self.size_labels = [QtWidgets.QLabel() for _ in range(3)]
        crow, arow = QtWidgets.QHBoxLayout(), QtWidgets.QHBoxLayout()
        for s in self.centre:
            crow.addWidget(s)
        for s in self.angles:
            arow.addWidget(s)
        ef.addRow("centre x, y, z", crow)
        ef.addRow("turned about x, y, z", arow)
        for lab, s in zip(self.size_labels, self.sizes):
            ef.addRow(lab, s)
        self.fill = QtWidgets.QPushButton("Fill from 'around the part'")
        self.fill.setToolTip("the values of the shape that bounds the part (or the sub-part) as a start to edit")
        self.fill.clicked.connect(self._fill_exact)
        ef.addRow(self.fill)
        self.cover = QtWidgets.QCheckBox("it covers the part (or the sub-part): its excess is cut away later")
        self.cover.setToolTip("a cover is kept when it holds 99 % of the part's material (of its region) on the loop's "
                              "grid, whatever it holds in excess; without the tick the shape must gain material as any add")
        self.cover.toggled.connect(self._update)
        ef.addRow(self.cover)
        self._fill_region = None
        for s in self.centre + self.angles + self.sizes:
            s.valueChanged.connect(self._update)
        lay.addWidget(self.exact)

        self.refit = QtWidgets.QCheckBox("let the reconstruction fit it to the part's surface (a rough guess)")
        self.refit.setChecked(True)
        self.refit.setToolTip("ticked: the shape is refitted to the surface near it before it is used; unticked: it is "
                              "used exactly as given. Either way the loop's own checks judge it, and a shape that fails "
                              "is reported and dropped, never forced")
        lay.addWidget(self.refit)
        self.hint = QtWidgets.QLabel("")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        lay.addWidget(self.buttons)
        self.mode_buttons["automatic"].setChecked(True)
        self._kind_changed()

    # ---------------------------------------------------------------- state
    def mode(self) -> str:
        return next(k for k, b in self.mode_buttons.items() if b.isChecked())

    def _mode_changed(self, *_):
        m = self.mode()
        self.kind.setEnabled(m != "automatic")
        self.around.setVisible(m == "around")
        self.exact.setVisible(m == "exact")
        self.refit.setVisible(m == "exact")                    # a cover around the part is never refitted
        if m == "exact" and self._shape is None:
            self._fill_exact()
        self._update()
        self.adjustSize()

    def _kind_changed(self, *_):
        names = ug.SIZES[self.kind.currentData()]
        for i, (lab, s) in enumerate(zip(self.size_labels, self.sizes)):
            on = i < len(names)
            lab.setText(names[i] if on else "")
            lab.setVisible(on)
            s.setVisible(on)
        if self.mode() == "exact":
            self._fill_exact()
        self._update()

    def _region(self):
        if not self.sub.isChecked():
            return None, None
        lo = np.array([self.region[("lo", j)].value() for j in range(3)])
        hi = np.array([self.region[("hi", j)].value() for j in range(3)])
        return lo, hi

    def _around_shape(self) -> ug.UserShape:
        lo, hi = self._region()
        return ug.bounding_shape(self.kind.currentData(), self.V, self.F, lo, hi, axis=self.axis.currentData())

    def _fill_exact(self):
        try:
            s = self._around_shape()
        except ValueError as exc:
            self.hint.setText(f"<span style='color:#b3261e'>{exc}</span>")
            return
        widgets = self.centre + self.angles + self.sizes
        for w in widgets:
            w.blockSignals(True)
        for w, v in zip(self.centre, s.center_mm):
            w.setValue(float(v))
        for w, v in zip(self.angles, ug.quaternion_euler(s.quaternion)):
            w.setValue(float(v))
        for w, v in zip(self.sizes, ug.shape_sizes(s)):
            w.setValue(float(v))
        self._fill_region = s.region_mm
        self.cover.blockSignals(True)
        self.cover.setChecked(True)
        widgets.append(self.cover)
        for w in widgets:
            w.blockSignals(False)
        self._update()

    def shape(self) -> Optional[ug.UserShape]:
        """The starting part as given now (None: automatic). Raises ValueError for an impossible shape."""
        m = self.mode()
        if m == "automatic":
            return None
        if m == "around":
            return self._around_shape()
        kind = self.kind.currentData()
        n = len(ug.SIZES[kind])
        shape = ug.exact_shape(kind, [w.value() for w in self.centre], [w.value() for w in self.angles],
                               [w.value() for w in self.sizes[:n]])
        if self.cover.isChecked():
            shape.role, shape.region_mm = "cover", self._fill_region
        return shape

    def _update(self, *_):
        ok = self.buttons.button(QtWidgets.QDialogButtonBox.Ok)
        try:
            s = self.shape()
        except ValueError as exc:
            self._shape = None
            self.hint.setText(f"<span style='color:#b3261e'>{exc}</span>")
            ok.setEnabled(False)
            return
        self._shape = s
        ok.setEnabled(True)
        if s is None:
            self.hint.setText("The reconstruction chooses its starting shape (the path's own start).")
            return
        held = ug.held_share(s, self.V)
        inside = ug.inside_share(s, self.V, self.F)
        rule = (" As a cover it is kept when it holds 99 % of the part's material on the loop's grid." if s.role == "cover"
                else " It is kept when it gains material, as any add.")
        self.hint.setText(f"The shape holds <b>{held:.1%}</b> of the part's vertices; "
                          + (f"<b>{inside:.1%}</b> of the shape lies inside the part - the rest is cut away later."
                             if inside is not None else "it has no volume.") + rule)

    # ---------------------------------------------------------------- result
    def entry(self) -> Optional[Dict]:
        """The first-guess entry of the loop (UserShape.as_dict + "refined": used as given) or None (automatic)."""
        s = self.shape()
        if s is None:
            return None
        d = s.as_dict()
        d["refined"] = s.role == "cover" or not self.refit.isChecked()     # a cover is taken as it is
        return d

    def summary(self) -> str:
        s = self._shape
        if self.mode() == "automatic" or s is None:
            return "automatic"
        if self.mode() == "around":
            return f"{s.kind} around the {'sub-part' if self.sub.isChecked() else 'part'}"
        return f"{s.kind} exact" + (", a cover" if s.role == "cover" else "") + ("" if self.refit.isChecked() else ", as given")
