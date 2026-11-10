#!/usr/bin/env python3
"""The settings dock: which part, which networks, which topologies, which thresholds - collected into a RunSettings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5 import QtCore, QtWidgets

from reco_gui import setup


def _spin(lo, hi, value, step=0.01, decimals=3, tip=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    w.setToolTip(tip)
    return w


def _int(lo, hi, value, tip=""):
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setValue(value)
    w.setToolTip(tip)
    return w


class SettingsPanel(QtWidgets.QScrollArea):
    inputChosen = QtCore.pyqtSignal(str)         # a part was picked (list, Browse, Random): the window previews it

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.setWidget(body)
        root = QtWidgets.QVBoxLayout(body)
        defaults_s = setup.RunSettings(input_stl=Path())

        # ---- method
        box = QtWidgets.QGroupBox("Reconstruction")
        form = QtWidgets.QFormLayout(box)
        self.method = QtWidgets.QComboBox()
        self.method.addItem("Cascade (add-cut)", "cascade")
        self.method.addItem("Vector field (geometry: STL / STEP)", "field")
        self.method.setToolTip("Cascade (add-cut), the main line: type -> rotation -> dimensions -> position per leftover "
                               "component, from the stage registry, added or cut until the residual is gone "
                               "(reco_engine/residual_loop.py).\nVector field, the comparison route: fit primitives to the "
                               "closest-surface-point field of the model itself, peel off what they explain (no voxels, no "
                               "networks).")
        form.addRow(self.method)
        root.addWidget(box)

        # ---- part
        box = QtWidgets.QGroupBox("Part")
        form = QtWidgets.QFormLayout(box)
        self.stl = QtWidgets.QComboBox()
        self.stl.setEditable(True)
        self.stl.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)   # a long name must not
        self.stl.setMinimumContentsLength(10)                                                     # widen the dock
        for path in setup.discover_examples():
            self.stl.addItem(path.name, str(path))
        for group, paths in setup.discover_library():                 # the part library (2026-10-29): benchmark parts,
            self.stl.insertSeparator(self.stl.count())                  # the detail study, the known-tree corpora
            self.stl.addItem(f"── {group} ──")
            self.stl.model().item(self.stl.count() - 1).setEnabled(False)
            for path in paths:
                self.stl.addItem(path.name, str(path))
                self.stl.setItemData(self.stl.count() - 1, str(path), QtCore.Qt.ToolTipRole)
        browse = QtWidgets.QPushButton("Browse ...")
        browse.clicked.connect(self._browse)
        self.random = QtWidgets.QPushButton("Random")
        self.random.setToolTip(f"A random STEP file of the downloaded test parts ({setup.TEST_PARTS}); "
                               "every part once before one comes again")
        self.random.clicked.connect(self.pick_random)
        self._random_left = []
        form.addRow(self.stl)                                   # the buttons on a row of their own: one row of all
        row = QtWidgets.QHBoxLayout()                           # three made the dock 402 px wide at least
        row.addWidget(browse)
        row.addWidget(self.random)
        row.addStretch(1)
        form.addRow(row)
        self.stl.activated.connect(lambda _=0: self.inputChosen.emit(str(self._stl_path())))
        root.addWidget(box)

        # ---- results
        box = QtWidgets.QGroupBox("Results")
        form = QtWidgets.QFormLayout(box)
        self.save_runs = QtWidgets.QCheckBox("save every finished run")
        self.save_runs.setChecked(defaults_s.save_finished_run)
        from reco_engine.run_store import store_root
        self.save_runs.setToolTip(f"Store every finished run (settings, code commit, program, metrics, timings, the built "
                                  f"model, a screenshot) under {store_root()} ($SHAPE_RECO_RESULTS), to compare runs "
                                  f"later: toolbar 'Compare', or scripts/compare_runs.py. 'Save' stores a run by hand.")
        form.addRow(self.save_runs)
        root.addWidget(box)

        # ---- vector-field loop
        self.field_box = QtWidgets.QGroupBox("Vector-field loop")
        form = QtWidgets.QFormLayout(self.field_box)
        self.f_max_ops = _int(1, 100, defaults_s.field_max_ops, "operations at most, the base included")
        self.f_target = _spin(0.5, 1.0, defaults_s.field_target, 0.005, 3, "stop when this share of the input surface is explained")
        self.f_surface = _int(500, 50000, defaults_s.field_surface_points, "surface points the fits work with")
        self.f_near = _int(0, 50000, defaults_s.field_near_points, "off-surface points (the field's distances pin the size of the base)")
        self.f_vectors = _int(0, 3000, defaults_s.field_vectors, "vectors V(x) drawn in the 3D view")
        self.f_deflection = _spin(0.001, 5.0, defaults_s.field_deflection, 0.01, 3, "STEP only: tessellation deflection, in the model's units")
        self.f_grade = QtWidgets.QCheckBox("chamfer against the input at the end")
        self.f_grade.setChecked(defaults_s.field_grade)
        self.f_grade.setToolTip("Distance of the model's surface to the input and back (a few seconds)")
        self.f_bounding_box = QtWidgets.QCheckBox("bounding-box base/cut (rethought acceptance gate)")
        self.f_bounding_box.setChecked(defaults_s.field_bounding_box)
        self.f_bounding_box.setToolTip("reco_field.peel's bounding_box option: a box base/cut covers every point of its cluster "
                                       "instead of the tightest (Tukey-IRLS) fit to most of them - docs/FRAME_FAILURE.md, "
                                       "'Bounding-box fits for the acceptance-gated cover'. Off by default.")
        self.f_invented_slack = QtWidgets.QDoubleSpinBox()
        self.f_invented_slack.setRange(0.0, 1.0)
        self.f_invented_slack.setDecimals(3)
        self.f_invented_slack.setSingleStep(0.01)
        self.f_invented_slack.setSpecialValueText("default (2%)")
        self.f_invented_slack.setValue(0.0)
        self.f_invented_slack.setToolTip("reco_field.peel.INVENTED_SLACK override - bounding-box mode typically needs this opened "
                                         "well past its default (0.2+) before it accepts what it proposes")
        self.f_cuts_first = QtWidgets.QComboBox()
        self.f_cuts_first.addItem("follow bounding-box", None)
        self.f_cuts_first.addItem("cuts before additions", "on")
        self.f_cuts_first.addItem("mixed (original order)", "off")
        self.f_cuts_first.setToolTip("Try every cutter cluster before any addition one, so an oversized bounding-box base gets "
                                     "carved down before anything is added to it - can be decoupled from the checkbox above.")
        self.f_max_candidate_invented = QtWidgets.QDoubleSpinBox()
        self.f_max_candidate_invented.setRange(0.0, 1.0)
        self.f_max_candidate_invented.setDecimals(3)
        self.f_max_candidate_invented.setSingleStep(0.05)
        self.f_max_candidate_invented.setSpecialValueText("off")
        self.f_max_candidate_invented.setValue(0.0)
        self.f_max_candidate_invented.setToolTip("Reject an ADDITION outright if its own standalone invented share exceeds this, "
                                                 "regardless of the invented-slack above - a per-candidate purity check a global "
                                                 "slack alone cannot provide (docs/FRAME_FAILURE.md).")
        self.f_mesh_clusters = QtWidgets.QCheckBox("full-mesh clustering (known-CAD-pipeline components)")
        self.f_mesh_clusters.setChecked(defaults_s.field_mesh_clusters)
        self.f_mesh_clusters.setToolTip("reco_field.peel's mesh_clusters option: cluster by the input mesh's own triangle "
                                        "connectivity (trimesh.Trimesh.split) instead of Euclidean-distance-clustering the "
                                        "sparse point sample - finds real small features the sparse sample never landed on "
                                        "(docs/FRAME_FAILURE.md, \"Full-mesh component clustering\").")
        self.f_verify_freecad = QtWidgets.QCheckBox("verify each step in FreeCAD (real CSG, slow)")
        self.f_verify_freecad.setChecked(defaults_s.field_verify_freecad)
        self.f_verify_freecad.setToolTip("reco_field.peel's verify_freecad option: actually build every accepted step's model "
                                         "in FreeCAD (real Part booleans via freecadcmd), not just the symbolic SDF - reports "
                                         "whether it stayed one valid solid or the CSG actually split it apart (the symbolic "
                                         "SDF never notices). A freecadcmd subprocess per step: much slower "
                                         "(docs/FRAME_FAILURE.md, \"Materializing every step in real FreeCAD\").")
        self.f_octree = QtWidgets.QCheckBox("occupancy octree over the missing volume")
        self.f_octree.setChecked(defaults_s.field_octree)
        self.f_octree.setToolTip("reco_field.peel's octree option: cluster by an occupancy octree over the "
                                 "missing VOLUME (input solid XOR model), coarse and axis-aligned, instead of "
                                 "surface-point or mesh-triangle clustering - avoids fragmenting a simple "
                                 "missing region into many slivers, but axis-aligned leaf cropping can bias "
                                 "curved features toward being read as boxes; measured mixed, not a clean win "
                                 "(docs/FRAME_FAILURE.md, \"Debugging the identification schema with the frame\").")
        self.f_predict = QtWidgets.QPushButton("Predict from genus")
        self.f_predict.setToolTip("Fill 'max operations' in from the part's own topology (reco_field.peel.predict_ops_budget) "
                                  "instead of guessing a fixed number - see docs/FRAME_FAILURE.md")
        self.f_predict.clicked.connect(self._predict_max_ops)
        self.f_predict_label = QtWidgets.QLabel("")
        self.f_predict_label.setWordWrap(True)
        max_ops_row = QtWidgets.QHBoxLayout()
        max_ops_row.addWidget(self.f_max_ops)
        max_ops_row.addWidget(self.f_predict)
        for label, w in (("max operations", max_ops_row), ("stop at explained", self.f_target), ("surface points", self.f_surface),
                         ("field points", self.f_near), ("vectors drawn", self.f_vectors), ("STEP deflection", self.f_deflection)):
            form.addRow(label, w)
        form.addRow(self.f_grade)
        form.addRow(self.f_bounding_box)
        form.addRow("invented slack", self.f_invented_slack)
        form.addRow("cluster order", self.f_cuts_first)
        form.addRow("max candidate invented", self.f_max_candidate_invented)
        form.addRow(self.f_mesh_clusters)
        form.addRow(self.f_verify_freecad)
        form.addRow(self.f_octree)
        form.addRow(self.f_predict_label)
        root.addWidget(self.field_box)

        # ---- cascade (add-cut) loop
        self.cascade_box = QtWidgets.QGroupBox("Cascade (add-cut) loop")
        form = QtWidgets.QFormLayout(self.cascade_box)
        from reco_gui.cascade_session import DEFAULT_REGISTRY
        self.c_registry = QtWidgets.QLineEdit(str(DEFAULT_REGISTRY))
        self.c_registry.setToolTip("reco_engine.stage_registry: which network / analytic fit answers which stage for which kind")
        registry_browse = QtWidgets.QPushButton("Browse ...")
        registry_browse.clicked.connect(self._browse_registry)
        registry_row = QtWidgets.QHBoxLayout()
        registry_row.addWidget(self.c_registry, 1)
        registry_row.addWidget(registry_browse)
        self.c_top_k = _int(1, 10, defaults_s.cascade_top_k, "branches kept per stage (the rest are shown as alternatives)")
        self.c_max_ops = _int(0, 200, defaults_s.cascade_max_ops or 0, "operations at most (0: the loop's own budget)")
        self.c_max_ops.setSpecialValueText("loop default")
        self.c_resolution = _int(16, 256, defaults_s.cascade_resolution,
                                 "voxels per side of the cubic grid the loop starts with (more: finer residual, slower)")
        self.c_resolution.setSingleStep(16)
        self.c_max_resolution = _int(16, 384, defaults_s.cascade_max_resolution,
                                     "the loop refines the grid up to this many voxels per side while thin walls still "
                                     "fall between the voxel centres (the frame: 0 input voxels at 48^3)")
        self.c_max_resolution.setSingleStep(16)
        self.c_op_guard = QtWidgets.QCheckBox("operation guard (bypassed)")
        self.c_op_guard.setChecked(defaults_s.cascade_op_guard)
        self.c_op_guard.setToolTip("Off (the default): the guard is bypassed and every candidate with a net gain is accepted. "
                                   "On: an operation must also pass the loop's guard.")
        self.c_op_guard.toggled.connect(lambda on: self.c_op_guard.setText("operation guard" + ("" if on else " (bypassed)")))
        self.c_freecad = QtWidgets.QCheckBox("build every step in FreeCAD (slow)")
        self.c_freecad.setChecked(defaults_s.cascade_freecad_per_step)
        self.c_freecad.setToolTip("A freecadcmd build after every accepted operation: is the model still one valid solid?")
        self.c_type_source = QtWidgets.QComboBox()
        # descriptors only and CNN only are no longer offered: graded C on 2026-10-30 (no measured effect against the
        # fused registry; docs/data/method_grades.md); run_part --type-source still takes them
        for label, data in (("registry default (fused)", "registry"),):
            self.c_type_source.addItem(label, data)
        self.c_type_source.setCurrentIndex(max(0, self.c_type_source.findData(defaults_s.cascade_type_source)))
        self.c_type_source.setToolTip("Which type entries of the registry answer the type stage: all of them, fused by their "
                                      "measured parent accuracy (default); the rotation-invariant descriptors alone "
                                      "(reco_engine/type_descriptors.py, no network); the Type CNNs alone. Analytic and prior "
                                      "type entries stay in every choice.")
        form.addRow("registry", registry_row)
        form.addRow("type source", self.c_type_source)
        self.c_field_backend = QtWidgets.QComboBox()
        for label, data in (("igl (exact, fast)", "igl"), ("environment default (cuda one-pass)", "")):
            self.c_field_backend.addItem(label, data)
        self.c_field_backend.setCurrentIndex(max(0, self.c_field_backend.findData(defaults_s.cascade_field_backend)))
        self.c_field_backend.setToolTip("The closest-point query behind the input's field channels and the final chamfer. "
                                        "igl: libigl's exact AABB tree (the frame at 96^3: 0.6 s). The environment default "
                                        "($SHAPE_RECO_FIELD_BACKEND, auto = a GPU brute-force search over the refined mesh) "
                                        "took 313 s there and gave the same program.")
        form.addRow("field backend", self.c_field_backend)
        self.c_grid = QtWidgets.QComboBox()
        for label, data in (("finest: wall thickness or cube (more cells)", "tau"), ("cube (voxels per side)", "cube")):
            self.c_grid.addItem(label, data)
        self.c_grid.setCurrentIndex(max(0, self.c_grid.findData(defaults_s.cascade_grid)))
        self.c_grid.setToolTip("The grid the loop computes on. finest (default): a box over the part at the finer of two "
                               "pitches - the thinnest walls (5th-percentile wall thickness tau) and the cube's pitch at "
                               "'voxels per side' - so thin walls are never lost between voxel centres (the frame: 88 x 562 "
                               "x 419 cells at 0.12 instead of 96^3 at 0.81, 504458 instead of 1376 input voxels). "
                               "cube: the old refined cube (walls thinner than a voxel alias).")
        form.addRow("grid", self.c_grid)
        from reco_engine.start_shape import DROPPED_STARTS, START_TYPES
        self.c_start = QtWidgets.QComboBox()
        for t in (t for t in START_TYPES if t not in DROPPED_STARTS):
            self.c_start.addItem({"legacy": "legacy (sub-loops first)", "auto": "auto (most surface, least excess)"}.get(t, t), t)
        self.c_start.setCurrentIndex(max(0, self.c_start.findData(defaults_s.cascade_start_shape)))
        self.c_start.setToolTip("The first add. legacy: member / hollow sub-loops first, then the graded first add. auto: "
                                "the starting-shape stage first - every candidate that holds the part, the one with the most "
                                "of the part's surface on it and the least excess. Or force one type (the dominating cover = "
                                "the intersection of every surface the whole part lies behind).")
        form.addRow("starting shape", self.c_start)
        self.c_mode = QtWidgets.QComboBox()
        self.c_mode.addItem("loop (the rounds as before)", "loop")
        self.c_mode.addItem("tree (decisions on the difference, children)", "tree")
        self.c_mode.addItem("search (every base shape, the best surface coverage)", "search")
        self.c_mode.setCurrentIndex(max(0, self.c_mode.findData(getattr(defaults_s, "cascade_mode", "loop"))))
        self.c_mode.setToolTip("The rounds after the start. tree (2026-10-15): every round an explicit decision on the "
                               "difference between model and input (accept / carve the excess / add the missing / leave "
                               "the rest to the feature phase / stop), and a component no primitive explains becomes a "
                               "child: its inside boxes as one composite operation, else its bounding box with the material "
                               "taken wrongly given back by a grandchild. Meant with the auto starting shape.")
        form.addRow("rounds", self.c_mode)
        form.addRow("top-k", self.c_top_k)
        form.addRow("max operations", self.c_max_ops)
        form.addRow("voxels per side", self.c_resolution)
        form.addRow("max voxels per side", self.c_max_resolution)
        form.addRow(self.c_op_guard)
        form.addRow(self.c_freecad)
        self.c_freecad_final = QtWidgets.QCheckBox("FreeCAD rebuild at the end of the run (slow)")
        self.c_freecad_final.setChecked(defaults_s.cascade_freecad_final)
        self.c_freecad_final.setToolTip("Build the finished program in FreeCAD and grade it (chamfer, exact IoU) as the run's "
                                        "last step: minutes on a part with many features. Off: the toolbar's button "
                                        "'FreeCAD rebuild' does it when asked")
        form.addRow(self.c_freecad_final)
        self.c_subparts = QtWidgets.QCheckBox("sub-parts (rough model and pieces without a tool)")
        self.c_subparts.setChecked(defaults_s.cascade_subparts)
        self.c_subparts.setToolTip("The rough model as several primitives that are added before any cut, and a feature "
                                   "piece that no tool explains as several primitives before the voxel fallback "
                                   "(reco_engine/subpart_cover.py)")
        form.addRow(self.c_subparts)
        self.c_at_once = QtWidgets.QCheckBox("apply an accepted feature at once")
        self.c_at_once.setChecked(defaults_s.cascade_apply_at_once)
        self.c_at_once.setToolTip("A feature whose exact check in FreeCAD passes joins the model at once and is shown; "
                                  "the build of the whole model and its difference against the input follow once per "
                                  "batch as the confirmation. Off: a batch is guessed, then validated together")
        form.addRow(self.c_at_once)
        root.addWidget(self.cascade_box)

        # ---- cascade: the feature phase
        from reco_gui.cascade_session import config_defaults
        loop = config_defaults()                                # the loop's own limits, where its config has them
        self.feature_box = QtWidgets.QGroupBox("Feature phase")
        form = QtWidgets.QFormLayout(self.feature_box)
        self.c_features = QtWidgets.QCheckBox("feature phase after the rough model")
        self.c_features.setChecked(defaults_s.cascade_feature_phase)
        self.c_features.setToolTip("After the rough model: FreeCAD computes the exact difference between the model and the "
                                   "part, the difference is split into pieces (possible features), per piece a primitive is "
                                   "guessed and scored, a good guess is applied as a cut / an add and validated by a real "
                                   "FreeCAD build; repeated within the limits below. The Cascade view draws the pieces by "
                                   "state and the guess as a wireframe.")
        self.c_feature_pieces = _int(0, 5000, int(loop.get("max_pieces") or 0), "pieces of the difference looked at per "
                                     "iteration, the largest first (0: the loop's own limit)")
        self.c_feature_total = _int(0, 100000, int(loop.get("max_total_pieces") or 0), "pieces looked at in the whole "
                                    "phase (0: the loop's own limit)")
        self.c_feature_applied = _int(0, 100000, int(loop.get("max_applied") or 0), "features applied at most in the "
                                      "whole phase (0: the loop's own limit)")
        self.c_feature_iterations = _int(0, 100, int(loop.get("max_iterations") or 0),
                                         "iterations (difference, pieces, guesses) at most (0: the loop's own limit)")
        self.c_feature_seconds = _spin(0, 86400, float(loop.get("time_budget_s") or 0.0), 30.0, 0,
                                       "the phase stops when it has run this long (0: the loop's own limit)")
        self.c_feature_seconds.setSuffix(" s")
        self.c_feature_score = _spin(0, 1, float(loop.get("min_score") or 0.0), 0.05, 3,
                                     "a guess needs at least this score to be applied (0: the loop's own limit)")
        # the level of detail: the IoU the feature phase works up to, in steps of 0.005; the right end is "all details"
        self.c_target = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.c_target.setRange(self.TARGET_LO, self.TARGET_HI)
        self.c_target.setSingleStep(1)
        self.c_target.setPageStep(4)
        self.c_target.setTickInterval(4)
        self.c_target.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.c_target.setToolTip("Level of detail. The feature phase takes the groups of pieces with the most volume first "
                                 "and stops when the IoU estimated from the volumes reaches this target: a lower target "
                                 "gives a model with fewer, larger features. At the right end every piece is taken")
        self.c_target_label = QtWidgets.QLabel()
        self.c_target.valueChanged.connect(lambda v: self.c_target_label.setText(self.target_text()))
        self.set_target_iou(defaults_s.cascade_feature_target_iou)
        self.c_target_label.setText(self.target_text())
        target_row = QtWidgets.QWidget()
        target_layout = QtWidgets.QHBoxLayout(target_row)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.addWidget(self.c_target, 1)
        target_layout.addWidget(self.c_target_label)
        self._feature_limits = (self.c_feature_pieces, self.c_feature_total, self.c_feature_iterations, self.c_feature_seconds, self.c_feature_score)
        self.c_features.toggled.connect(lambda on: [w.setEnabled(on) for w in (self.c_target, self.c_target_label)])
        for w in self._feature_limits:
            w.setSpecialValueText("loop default")
        form.addRow(self.c_features)
        form.addRow("detail: target IoU", target_row)
        # the user's rule of 2026-10-10: a finished run below this IoU asks whether to try the next path variant
        # (reco_engine/path_policy.py: the variants ranked by their measured share of good runs)
        self.c_retry_iou = QtWidgets.QDoubleSpinBox()
        self.c_retry_iou.setRange(0.0, 1.0)
        self.c_retry_iou.setSingleStep(0.01)
        self.c_retry_iou.setDecimals(2)
        self.c_retry_iou.setValue(defaults_s.retry_iou)
        self.c_retry_iou.setSpecialValueText("never")
        self.c_retry_iou.setToolTip("When a run ends with an IoU below this, the window asks whether to try another path: the "
                                    "sub-part cover, the voxel-cube cover, another start or type source - the untried variants "
                                    "ranked by the share of good runs they gave on the measured parts (docs/data/path_weights.json). "
                                    "0 = never ask")
        self.c_retry_max = QtWidgets.QSpinBox()
        self.c_retry_max.setRange(1, 6)
        self.c_retry_max.setValue(defaults_s.retry_max)
        self.c_retry_max.setToolTip("further paths tried at most, per part")
        retry_row = QtWidgets.QWidget()
        retry_layout = QtWidgets.QHBoxLayout(retry_row)
        retry_layout.setContentsMargins(0, 0, 0, 0)
        retry_layout.addWidget(self.c_retry_iou, 1)
        retry_layout.addWidget(QtWidgets.QLabel("paths at most"))
        retry_layout.addWidget(self.c_retry_max)
        form.addRow("ask for another path below IoU", retry_row)
        self.c_alternatives = QtWidgets.QSpinBox()
        self.c_alternatives.setRange(0, 6)
        self.c_alternatives.setValue(int(getattr(defaults_s, "alternatives", 3)))
        self.c_alternatives.setSpecialValueText("off (ask below the IoU)")
        self.c_alternatives.setToolTip("After a run, the window goes on with the next paths by itself - the untried path variants "
                                       "ranked by their measured rescue rate (docs/data/path_weights.json), one after the other "
                                       "- until this many different reconstructions of the part exist; then the Benchmark tab "
                                       "shows them checked: their nephograms and histograms (made after every run) and their "
                                       "event time in aegir (press 'Event time'), to choose one. 0: off, the decision bar asks "
                                       "below the IoU above instead.")
        form.addRow("reconstructions per part", self.c_alternatives)
        self.c_compare_tool = QtWidgets.QComboBox()
        for key, label in (("freecad", "FreeCAD (macro: coloured meshes)"), ("cloudcompare", "CloudCompare (scalar field)"),
                           ("inside", "inside the framework (3D view layer 'Distance')"), ("none", "nothing, only the files")):
            self.c_compare_tool.addItem(label, key)
        self.c_compare_tool.setCurrentIndex(max(0, self.c_compare_tool.findData(getattr(defaults_s, "compare_tool", "freecad"))))
        self.c_compare_tool.setToolTip("After 'Export for comparison' on the decision bar, the tool that opens on the export: "
                                       "FreeCAD runs nephogram.FCMacro (the input and the guess as Mesh objects coloured by "
                                       "their surface distance), CloudCompare loads the PLYs with the scalar field 'distance'; inside: the "
                                       "input coloured by its distance as the 3D view's layer 'Distance' (shown in every case).")
        form.addRow("comparison opens in", self.c_compare_tool)
        self.c_show_voxels = QtWidgets.QCheckBox("build the voxel layers (cost time)")
        self.c_show_voxels.setChecked(bool(getattr(defaults_s, "show_voxels", False)))
        self.c_show_voxels.setToolTip("Off (the user's rule 2026-10-15): no voxel cubes on screen - the input is the surface "
                                      "distance field with its inside / outside colouring and the part itself. On: the cube "
                                      "meshes of the input's grid, the residual, the cut history, the component and the tau "
                                      "grid are built every step for the debug layers (a marching pass each).")
        form.addRow("voxel layers", self.c_show_voxels)
        self.c_memory = QtWidgets.QSpinBox()
        self.c_memory.setRange(0, 64000)
        self.c_memory.setSingleStep(100)
        self.c_memory.setValue(int(getattr(defaults_s, "memory_min_free_mb", 1200)))
        self.c_memory.setSuffix(" MB")
        self.c_memory.setSpecialValueText("off")
        self.c_memory.setToolTip("The memory watcher (reco_engine/memory_guard.py): when the machine's available memory falls "
                                 "below this, the window pauses the run and drops its mesh caches, and the loop stops a headless "
                                 "run with the model it has - instead of the system killing the framework (2026-10-15).")
        form.addRow("pause below free memory", self.c_memory)
        form.addRow("max pieces per iteration", self.c_feature_pieces)
        form.addRow("max pieces in all", self.c_feature_total)
        form.addRow("max features applied", self.c_feature_applied)
        form.addRow("max iterations", self.c_feature_iterations)
        form.addRow("time budget", self.c_feature_seconds)
        form.addRow("min score", self.c_feature_score)
        self.c_features.toggled.connect(lambda on: [w.setEnabled(on) for w in self._feature_limits])
        root.addWidget(self.feature_box)

        # ---- minimal by default: the method's own settings and "Advanced" behind one toggle
        self.more = QtWidgets.QToolButton()
        self.more.setText("More settings")
        self.more.setCheckable(True)
        self.more.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.more.setArrowType(QtCore.Qt.RightArrow)
        self.more.setToolTip("show the method's loop settings, the networks and the advanced options")
        self.more.toggled.connect(lambda on: (self.more.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow),
                                              self._method_changed()))
        root.addWidget(self.more)

        # ---- advanced
        box = QtWidgets.QGroupBox("Advanced")
        self.advanced_box = box
        form = QtWidgets.QFormLayout(box)
        self.freecad = QtWidgets.QLineEdit(defaults_s.freecad)
        form.addRow("freecadcmd", self.freecad)
        root.addWidget(box)
        root.addStretch(1)
        self.method.currentIndexChanged.connect(lambda _=0: self._method_changed())
        self._method_changed()
        self._compact()

    # ------------------------------------------------------------------------------------------ helpers
    def _method_changed(self):
        method = self.method.currentData()
        more = self.more.isChecked()
        self.field_box.setVisible(more and method == "field")
        self.cascade_box.setVisible(more and method == "cascade")
        self.feature_box.setVisible(more and method == "cascade")
        self.advanced_box.setVisible(more)

    def _compact(self, max_chars: int = 34) -> None:
        """A dock that fits a laptop screen (the user, 2026-10-29): every combo box may shrink below its longest entry,
        and a check box whose label is a sentence shows its first clause - the whole sentence leads its tooltip (Qt
        does not wrap check boxes; the longest made the panel 396 px wide at least)."""
        for cb in self.findChildren(QtWidgets.QComboBox):
            cb.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            cb.setMinimumContentsLength(12)
        for box in self.findChildren(QtWidgets.QCheckBox):
            text = box.text()
            if len(text) <= max_chars:
                continue
            short = text.split(" (")[0].split(": ")[0]
            if len(short) > max_chars:
                short = short[:max_chars - 1].rstrip() + "\u2026"
            box.setToolTip(text + ("\n\n" + box.toolTip() if box.toolTip() else ""))
            box.setText(short)

    def _browse_registry(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Stage registry", str(setup.ROOT / "models"), "JSON (*.json)")
        if path:
            self.c_registry.setText(path)

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Input model", str(setup.ROOT),
                                                        "STL or STEP (*.stl *.step *.stp);;STL (*.stl);;STEP (*.step *.stp)")
        if path:
            self.stl.setCurrentText(path)
            self.inputChosen.emit(path)

    def pick_random(self, _checked: bool = False, rng=None) -> Optional[Path]:
        """Put a random test part (setup.discover_test_parts) into the part field and announce it; None when there
        is none. The parts are drawn without replacement, and the list is read again when it is used up."""
        import random
        if not self._random_left:
            self._random_left = setup.discover_test_parts()
            (rng or random).shuffle(self._random_left)
        if not self._random_left:
            QtWidgets.QMessageBox.information(self, "Random part", f"No STEP files below {setup.TEST_PARTS} "
                                              "(set SHAPE_RECO_TEST_PARTS to the directory of the test parts).")
            return None
        path = self._random_left.pop()
        self.stl.setCurrentText(str(path))
        self.inputChosen.emit(str(path))
        return path

    def _predict_max_ops(self):
        path = self._stl_path()
        if not path.is_file():
            self.f_predict_label.setText(f"Input model not found: {path}")
            return
        from reco_field.peel import predict_ops_budget
        from src.closest_point_field import load_field
        self.f_predict_label.setText("Predicting ...")
        QtWidgets.QApplication.processEvents()
        try:
            budget = predict_ops_budget(load_field(path))
        except Exception as exc:                                                             # e.g. STEP needs freecadcmd
            self.f_predict_label.setText(f"Could not predict a budget: {exc}")
            return
        self.f_max_ops.setValue(budget.max_ops)
        self.f_predict_label.setText(f"{budget.max_ops} ops ({budget.reason}; genus {budget.plateau_genus:.0f} at resolution "
                                     f"{budget.plateau_resolution}, exact genus {budget.exact_genus:.0f})")

    def set_input(self, path: str):
        i = self.stl.findData(str(path))
        if i >= 0:
            self.stl.setCurrentIndex(i)
        else:
            self.stl.setCurrentText(str(path))

    def current_path(self) -> Path:
        """The chosen input file (an STL or a STEP), as typed or as the listed example's full path."""
        return self._stl_path()

    def _stl_path(self) -> Path:
        text = self.stl.currentText().strip()
        index = self.stl.findText(text)                       # a listed example: its full path is the item's data
        return Path(self.stl.itemData(index) if index >= 0 else text)

    TARGET_LO, TARGET_HI, TARGET_STEP = 160, 200, 0.005      # slider positions: 0.800 ... 0.995, 200 = all details

    def target_iou(self):
        """The slider's target IoU, None at its right end (every piece)."""
        v = self.c_target.value()
        return None if v >= self.TARGET_HI else round(v * self.TARGET_STEP, 3)

    def set_target_iou(self, value):
        self.c_target.setValue(self.TARGET_HI if value is None
                               else max(self.TARGET_LO, min(self.TARGET_HI - 1, int(round(float(value) / self.TARGET_STEP)))))

    def target_text(self) -> str:
        t = self.target_iou()
        return "all details" if t is None else f"{t:.3f}"

    def apply_path(self, overrides: dict) -> None:
        """The path fields of a variant (reco_engine.path_policy.VARIANTS[...]["overrides"]: cascade_start_shape,
        cascade_type_source, cascade_subparts) set on the widgets, the bare values for what the dict leaves out - so the
        panel shows the path the next run takes."""
        start = overrides.get("cascade_start_shape", "legacy")
        self.c_start.setCurrentIndex(max(0, self.c_start.findData(start)))
        self.c_mode.setCurrentIndex(max(0, self.c_mode.findData(overrides.get("cascade_mode", "loop"))))
        source = overrides.get("cascade_type_source", "registry")
        self.c_type_source.setCurrentIndex(max(0, self.c_type_source.findData(source)))
        self.c_subparts.setChecked(bool(overrides.get("cascade_subparts", False)))
        from reco_engine.path_policy import DEFAULT_REGISTRY as default_registry
        self.c_registry.setText(str(overrides.get("cascade_registry") or default_registry))

    def collect(self) -> Optional[setup.RunSettings]:
        path = self._stl_path()
        if not path.is_file():
            raise ValueError(f"Input model not found: {path}")
        if self.method.currentData() == "field":
            return setup.RunSettings(
                save_finished_run=self.save_runs.isChecked(),
                input_stl=path, method="field", field_max_ops=self.f_max_ops.value(), field_target=self.f_target.value(),
                field_surface_points=self.f_surface.value(), field_near_points=self.f_near.value(),
                field_vectors=self.f_vectors.value(), field_deflection=self.f_deflection.value(), field_grade=self.f_grade.isChecked(),
                field_bounding_box=self.f_bounding_box.isChecked(),
                field_invented_slack=self.f_invented_slack.value() if self.f_invented_slack.value() > 0 else None,
                field_cuts_first=self.f_cuts_first.currentData(),
                field_max_candidate_invented=self.f_max_candidate_invented.value() if self.f_max_candidate_invented.value() > 0 else None,
                field_mesh_clusters=self.f_mesh_clusters.isChecked(),
                field_verify_freecad=self.f_verify_freecad.isChecked(),
                field_octree=self.f_octree.isChecked(),
                freecad=self.freecad.text().strip() or setup.default_freecadcmd())
        if self.method.currentData() == "cascade":
            registry = self.c_registry.text().strip()
            return setup.RunSettings(
                save_finished_run=self.save_runs.isChecked(),
                input_stl=path, method="cascade", cascade_registry=Path(registry) if registry else None,
                cascade_top_k=self.c_top_k.value(), cascade_op_guard=self.c_op_guard.isChecked(),
                cascade_max_ops=self.c_max_ops.value() or None, cascade_freecad_per_step=self.c_freecad.isChecked(),
                cascade_freecad_final=self.c_freecad_final.isChecked(),
                cascade_resolution=self.c_resolution.value(),
                cascade_max_resolution=max(self.c_max_resolution.value(), self.c_resolution.value()),
                cascade_type_source=self.c_type_source.currentData(),
                cascade_field_backend=self.c_field_backend.currentData(),
                cascade_grid=self.c_grid.currentData(),
                cascade_start_shape=self.c_start.currentData(),
                cascade_mode=self.c_mode.currentData(),
                cascade_feature_phase=self.c_features.isChecked(),
                cascade_feature_max_pieces=self.c_feature_pieces.value() or None,
                cascade_feature_max_total_pieces=self.c_feature_total.value() or None,
                cascade_feature_max_iterations=self.c_feature_iterations.value() or None,
                cascade_feature_time_budget_s=self.c_feature_seconds.value() or None,
                cascade_feature_min_score=self.c_feature_score.value() or None,
                cascade_feature_target_iou=self.target_iou(),
                cascade_subparts=self.c_subparts.isChecked(),
                cascade_apply_at_once=self.c_at_once.isChecked(),
                cascade_feature_max_applied=self.c_feature_applied.value() or None,
                retry_iou=float(self.c_retry_iou.value()), retry_max=int(self.c_retry_max.value()),
                alternatives=int(self.c_alternatives.value()),
                compare_tool=self.c_compare_tool.currentData(),
                show_voxels=self.c_show_voxels.isChecked(),
                memory_min_free_mb=float(self.c_memory.value()),
                freecad=self.freecad.text().strip() or setup.default_freecadcmd())
        raise ValueError(f"Unknown method: {self.method.currentData()!r}")
