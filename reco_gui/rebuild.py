#!/usr/bin/env python3
"""
The FreeCAD rebuild on demand (the user's rule 2026-09-29: "the rebuild in freecad takes quite long. give it a
button"): the run ends without building its program in FreeCAD (106 s of 962 s on the liner, with the 39 features of
2026-09-29 more than 13 minutes), the toolbar's button does it for the result on screen - the program as a solid, its
chamfer distance to the input and the exact grade - in a thread of its own, and the window shows the solid when it is
there.

The fast way (the user's rule 2026-09-29: "just loading and grouping primitives and leaving out the final boolean"):
`groups` writes every operation's tool as an object of a FreeCAD document, grouped by operation, and folds nothing.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path

from PyQt5 import QtCore


def rebuild(result, config, grade_fn=None, exact: bool = False):
    """(result with chamfer_mm, step_path and stl_path filled in, the grade's dict). `exact`: also the exact grade
    (OCCT difference against the input). Measured on the liner with 41 operations, 2026-09-29: the build itself takes
    23.5 s; the exact grade gave no answer in 300 s and failed after 1268 s - it was what made the rebuild long, so
    it is off unless asked for. `grade_fn(result, field, config, name)` replaces reco_engine.residual_loop._grade
    (tests)."""
    import copy
    import dataclasses
    result = copy.deepcopy(result)                       # the states of the run stay as they were
    if dataclasses.is_dataclass(config) and hasattr(config, "exact_grade"):
        config = dataclasses.replace(config, exact_grade=bool(exact))
    if grade_fn is None:
        from reco_engine.residual_loop import _grade as grade_fn, _load
        field, _ = _load(result.input_path)
    else:
        field = None
    grade = grade_fn(result, field, config, Path(str(result.input_path)).stem)
    return result, grade


def groups(result, out_dir=None, tools: str = "primitives", build_fn=None) -> dict:
    """The result's accepted operations as grouped tools in a FreeCAD document, without the final booleans
    (reco_field.export.build_groups: 12.6 s for the liner's 41 operations with tools "solid", 0.5 s with "leaves",
    measured 2026-09-29). `out_dir`: default ~/.cache/shape_reco/groups/<part>_<time>."""
    from reco_engine.result_to_program import primitive_to_sp
    name = Path(str(result.input_path)).stem
    if out_dir is None:
        out_dir = Path.home() / ".cache" / "shape_reco" / "groups" / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    ops = [(o.op, o.primitive.label or o.primitive.kind, primitive_to_sp(o.primitive))
           for o in result.operations if o.accepted]
    if build_fn is None:
        from reco_field.export import build_groups as build_fn
    return build_fn(ops, out_dir, name, tools=tools)


def groups_text(r: dict, seconds: float) -> str:
    ops = r.get("operations") or []
    failed = [o for o in ops if o.get("error")]
    invalid = [o for o in ops if o.get("valid") is False]
    return (f"FreeCAD groups ({seconds:.1f} s): {len(ops)} tool(s) in "
            + ", ".join(f"{n} {k}" for k, n in sorted((r.get("groups") or {}).items()))
            + (f", {len(failed)} not built" if failed else "") + (f", {len(invalid)} not valid" if invalid else "")
            + f", no final boolean: {r.get('fcstd')}")


def grade_text(grade: dict, seconds: float) -> str:
    if grade.get("error"):
        return f"FreeCAD rebuild failed after {seconds:.0f} s: {grade['error'][-200:]}"
    parts = [f"FreeCAD rebuild ({seconds:.0f} s): " + ("one valid solid" if grade.get("valid") else
                                                        f"NOT one valid solid ({grade.get('n_solids')} solid(s))")]
    if grade.get("volume") is not None:
        parts.append(f"volume {grade['volume']:.6g}")
    if grade.get("chamfer") is not None:
        parts.append(f"chamfer {grade['chamfer']:.4g} mm")
    if grade.get("exact_iou") is not None:
        parts.append(f"exact IoU {grade['exact_iou']:.4f} ({grade.get('exact_method') or 'occt'})")
    elif grade.get("step"):
        parts.append(str(grade["step"]))
    if grade.get("exact_error"):
        parts.append(f"no exact grade: {str(grade['exact_error'])[-120:]}")
    return ", ".join(parts)


class FreecadRebuild(QtCore.QObject):
    done = QtCore.pyqtSignal(object, object, float)      # result, grade, seconds
    grouped = QtCore.pyqtSignal(object, float)           # build_groups' dict, seconds
    failed = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, grade_fn=None, groups_fn=None):
        super().__init__(parent)
        self._grade_fn = grade_fn
        self._groups_fn = groups_fn
        self.thread = None

    def request_groups(self, result, tools: str = "primitives", out_dir=None) -> bool:
        """Start the grouped build; False when a build is running already."""
        if self.running:
            return False
        self.thread = threading.Thread(target=self._run_groups, args=(result, tools, out_dir), daemon=True)
        self.thread.start()
        return True

    def _run_groups(self, result, tools, out_dir):
        t = time.time()
        try:
            r = groups(result, out_dir, tools, self._groups_fn)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {str(exc)[-300:]}")
            return
        self.grouped.emit(r, time.time() - t)

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def request(self, result, config) -> bool:
        """Start the rebuild; False when one is running already."""
        if self.running:
            return False
        self.thread = threading.Thread(target=self._run, args=(result, config), daemon=True)
        self.thread.start()
        return True

    def _run(self, result, config):
        t = time.time()
        try:
            built, grade = rebuild(result, config, self._grade_fn)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {str(exc)[-300:]}")
            return
        self.done.emit(built, grade, time.time() - t)
