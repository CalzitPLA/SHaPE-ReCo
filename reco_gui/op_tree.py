#!/usr/bin/env python3
"""
The operations tree of the supervision window: which operations the run has performed, as a tree.

- "Model": the accepted program as the boolean tree FreeCAD builds (Fusion / Cut nodes, the primitives as leaves, in the
  order they were applied; consecutive adds share one Fusion, consecutive cuts one Cut with the model as its base).
- "History": every step, grouped by the component (iteration) it belongs to, with the verdict of each candidate. The
  feature phase has one group per piece ("feature piece 7 (iteration 2)"): its guesses, the rejected ones with their reason;
  the segments of a piece that was split are groups inside their piece's group.
  An operation the feature phase applied is a leaf of the Model like any other, labelled with the source of its guess.

Clicking an entry shows that step (`stateActivated(index)`). `csg_tree` is the Qt-free core.
A benchmark run (the Benchmark tab) is shown by `show_benchmark`: its program as the Model, the known tree it was rated
against (`truth_nodes`, Qt-free) in place of the History.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from PyQt5 import QtCore, QtGui, QtWidgets

from reco_gui import report

ACCEPTED_STAGES = ("accepted", "add", "fallback", "feature_accepted")
REJECTED_STAGES = ("rejected", "feature_rejected")
GREEN, RED = "#127a52", "#b3261e"


def csg_tree(program) -> Optional[tuple]:
    """The accepted operations as nested ("fuse", [children]) / ("cut", [base, tool, ...]) / ("prim", i, operation) nodes;
    a leading cut (nothing to cut from yet) gets ("empty",) as its base. None for an empty program."""
    model = None
    for i, o in enumerate(program):
        leaf = ("prim", i, o)
        if o.op == "intersect":                             # model & P: a Common node, like FreeCAD's Part::Common
            if model is None:
                continue                                    # nothing to intersect yet
            model = ("common", model[1] + [leaf]) if model[0] == "common" else ("common", [model, leaf])
            continue
        if o.op != "cut":                                   # add, and the voxel-cube fallbacks (they are fused too)
            if model is None:
                model = leaf
            elif model[0] == "fuse":
                model = ("fuse", model[1] + [leaf])
            else:
                model = ("fuse", [model, leaf])
        elif model is not None and model[0] == "cut":
            model = ("cut", model[1] + [leaf])
        else:
            model = ("cut", [model if model is not None else ("empty",), leaf])
    return model


def truth_nodes(d) -> Optional[tuple]:
    """A known tree (reco_engine.truth's JSON: Union / Intersection / Subtraction nodes over Placed primitives, as
    aegir and the v5 truths write them) as nested ("op", title, [children]) / ("leaf", name, dims, tip) nodes."""
    if not isinstance(d, dict):
        return None
    if "program" in d and "type" not in d:
        return truth_nodes(d["program"])
    t = d.get("type")
    if t in ("Union", "Intersection", "Subtraction"):
        kids = [truth_nodes(d.get("a")), truth_nodes(d.get("b"))]
        kids = [k for k in kids if k is not None]
        # consecutive nodes of the same kind are shown flat, like the Model's Fusion / Cut
        flat = []
        for k in kids:
            if k[0] == "op" and k[1] == t and t != "Subtraction":
                flat.extend(k[2])
            else:
                flat.append(k)
        return ("op", t, flat)
    if t == "Placed":
        leaf = truth_nodes(d.get("primitive"))
        if leaf is None:
            return None
        tr = d.get("translation") or (0.0, 0.0, 0.0)
        tip = leaf[3] + f"\ncentre ({', '.join(f'{float(v):+.2f}' for v in tr)})"
        return ("leaf", leaf[1], leaf[2], tip)
    if t is None:
        return None
    nums = {k: v for k, v in d.items() if k != "type" and isinstance(v, (int, float))}
    if "half" in d:
        nums.update({f"half_{a}": v for a, v in zip("xyz", d["half"])})
    dims = "  ".join(f"{k}={float(v):.4g}" for k, v in nums.items() if k != "inner_radius" or v)
    sweep = ""
    for key in ("u", "v"):
        rng = d.get(key)
        if rng and abs(float(rng[1]) - float(rng[0])) < 6.283:
            sweep += f"  {key} {float(rng[0]):.2f}..{float(rng[1]):.2f} rad"
    return ("leaf", t.lower(), dims + sweep, f"{t}: {dims}{sweep}")


def _prim_key(p) -> tuple:
    from reco_gui.cascade_session import CascadeFolder
    return CascadeFolder._prim_key(p)


def _leaf_text(i: int, o, feature: Optional[dict] = None) -> Tuple[str, str, str]:
    """(name, tooltip, dimensions) of a model leaf; `feature`: dict(piece, source, score) of a feature-phase operation."""
    p = o.primitive
    kind = {"composite": f"composite ({getattr(p, 'combine', '')})", "halfspace": "half-space"}.get(p.kind, p.kind)
    name = f"{i + 1}. {o.op}: {kind}" + (f"  — {p.label}" if getattr(p, "label", None) else "")
    if feature is not None and "feature" in (getattr(p, "label", None) or ""):     # the loop's label names the source
        name += f", piece {feature.get('piece')}"
    elif feature is not None:
        name += f"  — feature piece {feature.get('piece')}" + (f", {feature['source']}" if feature.get("source") else "")
    dims = "  ".join(f"{k}={v:.4g}" for k, v in (p.dims_mm or {}).items())
    centre = ", ".join(f"{v:+.2f}" for v in p.center_mm)
    angle, axis = report.quaternion_angle_axis(p.quaternion)
    tip = (f"{dims}\ncentre ({centre}) mm, {angle:.1f}° about ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f})\n"
           f"confidence {p.confidence:.2f}, net gain {o.net_gain:.0f} voxels")
    if o.freecad_valid is not None:
        tip += ", FreeCAD " + ("valid" if o.freecad_valid else "INVALID")
    if feature is not None:
        tip += f"\nfeature phase: piece {feature.get('piece')}, source {feature.get('source') or 'not given'}"
        if feature.get("score") is not None:
            tip += f", score {float(feature['score']):.3f}"
    return name, tip, dims


class OperationTree(QtWidgets.QTreeWidget):
    stateActivated = QtCore.pyqtSignal(int)            # the step to show

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(2)
        self.setHeaderLabels(["operation", "details"])
        self.header().setStretchLastSection(True)
        self.setUniformRowHeights(True)
        self.itemClicked.connect(self._on_clicked)
        self.clear_all()

    def clear_all(self):
        self.clear()
        self.model_root = QtWidgets.QTreeWidgetItem(self, ["Model", "no operation yet"])
        self.history_root = QtWidgets.QTreeWidgetItem(self, ["History", ""])
        for item in (self.model_root, self.history_root):
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
        self.model_root.setExpanded(True)
        self.history_root.setExpanded(True)
        self._groups: Dict[int, QtWidgets.QTreeWidgetItem] = {}
        self._accepted_at: Dict[tuple, int] = {}          # primitive -> the step that accepted it
        self._feature_ops: Dict[tuple, dict] = {}         # primitive -> piece, source, score (feature-phase operations)
        self._feature_groups: Dict[tuple, QtWidgets.QTreeWidgetItem] = {}   # (feature iteration, piece id) -> its group
        self._shown_program = None

    # ---------------------------------------------------------------------------------------- history
    def add_step(self, state, follow: bool = True):
        """Append one step of the run (called for every new state, in order)."""
        step = state.step
        it = getattr(step, "iteration", None)
        if it is None:
            it = getattr(state, "iteration", None) if not hasattr(state, "program") else None
        parent = self.history_root
        if it is not None:
            parent = self._groups.get(it)
            if parent is None:
                title, under = f"component {it + 1}", self.history_root
                if step.stage.startswith("feature_"):
                    info = getattr(state, "info", None) or {}
                    piece = info.get("piece_id", (info.get("piece") or {}).get("id") if isinstance(info.get("piece"), dict) else None)
                    title = f"feature piece {piece}" + (f" (iteration {info['iteration']})" if info.get("iteration") else "")
                    feature = getattr(state, "feature", None)
                    known = feature.piece(piece) if feature is not None and piece is not None else None
                    if known is not None and known.parent is not None:       # a segment: inside its piece's group
                        title = f"segment {piece}"
                        under = self._feature_groups.get((info.get("iteration"), known.parent), self.history_root)
                parent = QtWidgets.QTreeWidgetItem(under, [title, step.message])
                if step.stage.startswith("feature_"):
                    self._feature_groups[(info.get("iteration"), piece)] = parent
                    parent.setExpanded(True)
                parent.setData(0, QtCore.Qt.UserRole, state.index)
                self._groups[it] = parent
        item = QtWidgets.QTreeWidgetItem(parent, [f"{state.index + 1}  {step.stage}", step.message])
        item.setData(0, QtCore.Qt.UserRole, state.index)
        item.setToolTip(1, step.message)
        colour = report.STAGE_COLOURS.get(step.stage)
        if colour:
            for col in (0, 1):
                item.setBackground(col, QtGui.QColor(colour))
        candidate = getattr(state, "candidate", None) if hasattr(state, "program") else None     # cascade primitives only
        if candidate is not None and getattr(state, "verdict", None) == "accepted" and step.stage in ACCEPTED_STAGES:
            self._accepted_at[_prim_key(candidate)] = state.index
            source = ""
            if step.stage == "feature_accepted":
                info = getattr(state, "info", None) or {}
                self._feature_ops[_prim_key(candidate)] = dict(piece=info.get("piece_id"), source=info.get("source"),
                                                               score=info.get("score"))
                source = f" ({info['source']})" if info.get("source") else ""
            if parent is not self.history_root:
                parent.setText(1, f"✓ {getattr(state, 'candidate_op', '') or ''}: {candidate.kind}{source}"
                                  + (f" — {candidate.label}" if getattr(candidate, "label", None) else ""))
                parent.setForeground(1, QtGui.QBrush(QtGui.QColor(GREEN)))
        elif step.stage == "feature_accepted" and candidate is None and parent is not self.history_root:
            done = (getattr(state, "info", None) or {}).get("segments_accepted") or ()      # a split piece: no primitive
            parent.setText(1, f"✓ split: {len(done)} segment(s) accepted")
            parent.setForeground(1, QtGui.QBrush(QtGui.QColor(GREEN)))
        elif step.stage in REJECTED_STAGES and parent is not self.history_root \
                and not parent.text(1).startswith("✓"):
            reason = (getattr(state, "info", None) or {}).get("reason") if step.stage == "feature_rejected" else None
            parent.setText(1, f"✗ rejected: {reason or step.message}")
            parent.setForeground(1, QtGui.QBrush(QtGui.QColor(RED)))
        if follow:
            self.scrollToItem(item)

    # ---------------------------------------------------------------------------------------- model
    def show_model(self, state):
        """Rebuild the "Model" branch from the accepted program of `state` (the step being looked at)."""
        program = tuple(getattr(state, "program", ()) or ())
        result = getattr(state, "result", None)
        key = (tuple(id(o) for o in program), getattr(state, "finished", False))
        if key == self._shown_program:
            return
        self._shown_program = key
        self.model_root.takeChildren()
        n_cut = sum(1 for o in program if o.op == "cut")
        self.model_root.setText(1, f"{len(program)} operation(s): {len(program) - n_cut} add, {n_cut} cut"
                                if program else "no operation yet")
        tree = csg_tree(program)
        if tree is not None:
            self._add_node(self.model_root, tree)
        if getattr(state, "finished", False) and result is not None:
            for label, path in (("STEP", getattr(result, "step_path", None)), ("STL", getattr(result, "stl_path", None))):
                if path:
                    QtWidgets.QTreeWidgetItem(self.model_root, [f"output {label}", str(path)]).setToolTip(1, str(path))
        self.model_root.setExpanded(True)
        self._expand(self.model_root)

    def show_benchmark(self, operations, truth=None, label: str = ""):
        """A benchmark run's program as the Model branch (its accepted operations, result_schema.Operation) and the
        known tree it is rated against (a truth JSON dict) as the second branch, in place of the History."""
        self.clear_all()
        self._shown_program = None
        program = tuple(o for o in (operations or ()) if getattr(o, "accepted", True))
        n_cut = sum(1 for o in program if o.op == "cut")
        self.model_root.setText(1, (f"{label}: " if label else "") + (f"{len(program)} operation(s): {len(program) - n_cut} add, {n_cut} cut"
                                                                      if program else "no operation"))
        tree = csg_tree(program)
        if tree is not None:
            self._add_node(self.model_root, tree)
        self.history_root.setText(0, "Known tree")
        nodes = truth_nodes(truth)
        if nodes is None:
            self.history_root.setText(1, "none for this part")
        else:
            leaves = (truth.get("meta") or {}).get("leaves") if isinstance(truth, dict) else None
            self.history_root.setText(1, (truth.get("name") if isinstance(truth, dict) else "") + (f": {leaves} leaves" if leaves else ""))
            self._add_truth(self.history_root, nodes)
        self._expand(self.model_root)
        self._expand(self.history_root)

    def _add_truth(self, parent, node):
        if node[0] == "leaf":
            _, name, dims, tip = node
            item = QtWidgets.QTreeWidgetItem(parent, [name, dims])
            item.setToolTip(0, tip)
            item.setToolTip(1, tip)
            cut = parent.text(0) == "Subtraction" and parent.indexOfChild(item) > 0
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(RED if cut else GREEN)))
            return
        _, title, children = node
        what = {"Union": f"{len(children)} parts", "Subtraction": f"base minus {len(children) - 1} tool(s)",
                "Intersection": f"{len(children)} parts in common"}[title]
        item = QtWidgets.QTreeWidgetItem(parent, [title, what])
        font = item.font(0)
        font.setItalic(True)
        item.setFont(0, font)
        for child in children:
            self._add_truth(item, child)

    def _add_node(self, parent, node):
        if node[0] == "prim":
            _, i, o = node
            name, tip, dims = _leaf_text(i, o, self._feature_ops.get(_prim_key(o.primitive)))
            item = QtWidgets.QTreeWidgetItem(parent, [name, dims])
            item.setToolTip(0, tip)
            item.setToolTip(1, tip)
            item.setForeground(0, QtGui.QBrush(QtGui.QColor(RED if o.op == "cut" else GREEN)))
            index = self._accepted_at.get(_prim_key(o.primitive))
            if index is not None:
                item.setData(0, QtCore.Qt.UserRole, index)
            return
        if node[0] == "empty":
            QtWidgets.QTreeWidgetItem(parent, ["(empty)", "nothing to cut from yet"])
            return
        kind, children = node
        title = {"fuse": "Fusion", "cut": "Cut", "common": "Common"}[kind]
        what = {"fuse": f"{len(children)} parts", "cut": f"base minus {len(children) - 1} tool(s)",
                "common": f"base trimmed to {len(children) - 1} tool(s)"}[kind]
        item = QtWidgets.QTreeWidgetItem(parent, [title, what])
        font = item.font(0)
        font.setItalic(True)
        item.setFont(0, font)
        for child in children:
            self._add_node(item, child)

    def _expand(self, item):
        item.setExpanded(True)
        for k in range(item.childCount()):
            self._expand(item.child(k))

    def _on_clicked(self, item, _column):
        index = item.data(0, QtCore.Qt.UserRole)
        if index is not None:
            self.stateActivated.emit(int(index))
