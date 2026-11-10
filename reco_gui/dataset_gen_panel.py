#!/usr/bin/env python3
"""The dataset-generation settings dock: everything src.dataset_generator takes, grouped the way RUNNING.md
describes the workflow (mode -> output -> voxelization -> in-scale extras -> noise), collected into a
DatasetGenSettings (reco_gui/dataset_gen_setup.py)."""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from reco_gui.dataset_gen_setup import DATASET_GEN, ROOT, DatasetGenSettings
from reco_gui.settings_panel import _int, _spin


def _range(lo, hi, value, step=0.01, decimals=3):
    a = _spin(lo, hi, value[0], step, decimals)
    b = _spin(lo, hi, value[1], step, decimals)
    row = QtWidgets.QHBoxLayout()
    row.addWidget(a)
    row.addWidget(QtWidgets.QLabel("to"))
    row.addWidget(b)
    return a, b, row


class DatasetGenSettingsPanel(QtWidgets.QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.setWidget(body)
        root = QtWidgets.QVBoxLayout(body)
        d = DatasetGenSettings()

        # ---- mode / topologies / counts
        box = QtWidgets.QGroupBox("What to generate")
        form = QtWidgets.QFormLayout(box)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItem("type - topology classification", "type")
        self.mode.addItem("measure - dimensions/proportions", "measure")
        self.mode.addItem("rotation - orientation", "rotation")
        self.mode.setToolTip("Which CNN this data trains: TYPE varies dimensions and rotations, MEASURE fixes ratios and varies "
                             "rotations, ROTATION fixes a rotation and varies ratios.")
        form.addRow("mode", self.mode)

        import sys
        sys.path.insert(0, str(DATASET_GEN))
        from src.dataset_utils import TOPOLOGIES
        self.topo = QtWidgets.QListWidget()
        self.topo.setMaximumHeight(160)
        for name in TOPOLOGIES:
            item = QtWidgets.QListWidgetItem(name)
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.Checked if name in d.topologies else QtCore.Qt.Unchecked)
            self.topo.addItem(item)
        form.addRow(self.topo)
        row = QtWidgets.QHBoxLayout()
        for text, names in (("box + sphere", ("box", "sphere")), ("all", None)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, n=names: self._select_topologies(n))
            row.addWidget(b)
        form.addRow(row)

        self.count_type = _int(1, 100000, d.count_per_type, "samples per topology (mode: type)")
        self.measure_ratios = _int(1, 10000, d.measure_ratios_per_type, "ratio sets per topology (mode: measure)")
        self.count_measure = _int(1, 10000, d.count_per_measure, "rotations per ratio set (mode: measure)")
        self.rotation_groups = _int(1, 10000, d.rotation_groups_per_type, "fixed rotation groups per topology (mode: rotation)")
        self.count_rotation = _int(1, 10000, d.count_per_rotation, "ratio sets per rotation group (mode: rotation)")
        for label, w in (("count per type", self.count_type), ("ratio sets per type", self.measure_ratios),
                         ("rotations per ratio set", self.count_measure), ("rotation groups per type", self.rotation_groups),
                         ("ratio sets per group", self.count_rotation)):
            form.addRow(label, w)
        self.mode.currentIndexChanged.connect(lambda _=0: self._mode_changed())
        root.addWidget(box)

        # ---- output / sharding
        box = QtWidgets.QGroupBox("Output")
        form = QtWidgets.QFormLayout(box)
        self.output_dir = QtWidgets.QLineEdit(str(ROOT / "dataset_gen" / "local_dataset"))
        browse = QtWidgets.QPushButton("Browse ...")
        browse.clicked.connect(self._browse_output)
        r = QtWidgets.QHBoxLayout()
        r.addWidget(self.output_dir, 1)
        r.addWidget(browse)
        form.addRow("output dir", r)
        self.macro_dir = QtWidgets.QLineEdit(str(d.macro_dir))
        form.addRow("macro dir", self.macro_dir)
        self.freecad = QtWidgets.QLineEdit(d.freecad)
        form.addRow("freecadcmd", self.freecad)
        self.scale_mm = _spin(0.01, 10000, d.scale_mm, 1.0, 2)
        form.addRow("scale (mm)", self.scale_mm)
        self.seed = _int(0, 2_000_000_000, d.seed, "base seed; each shard adds its own index to it")
        form.addRow("seed", self.seed)
        self.shards = _int(1, 64, d.shards, "run this many parallel shards, output_dir/shard0.. shard{N-1}, "
                                            "seed+0.. seed+{N-1} (dataset_gen/docs/RUNNING.md's own workflow: "
                                            "FreeCAD is single-threaded per sample, so shards are how this "
                                            "project uses many cores)")
        form.addRow("shards", self.shards)
        self.resume = QtWidgets.QCheckBox("resume (skip STLs that already exist)")
        form.addRow(self.resume)
        root.addWidget(box)

        # ---- voxelization
        box = QtWidgets.QGroupBox("Voxelization")
        form = QtWidgets.QFormLayout(box)
        self.voxelize = QtWidgets.QCheckBox("voxelize each STL to a .bin next to it")
        self.voxelize.setChecked(d.voxelize)
        form.addRow(self.voxelize)
        self.voxel_resolution = _int(4, 512, d.voxel_resolution)
        form.addRow("resolution", self.voxel_resolution)
        self.voxel_scale_mode = QtWidgets.QComboBox()
        self.voxel_scale_mode.addItem("fixed - in-scale (size + position drawn in a real-mm grid)", "fixed")
        self.voxel_scale_mode.addItem("fit - bounding box stretched onto the whole grid (older datasets)", "fit")
        form.addRow("scale mode", self.voxel_scale_mode)
        self.grid_mm = _spin(1, 1000, d.grid_mm, 1.0, 1)
        form.addRow("grid (mm), fixed", self.grid_mm)
        a, b, r = _range(0.01, 1.0, d.size_fraction, 0.01, 3)
        self.size_frac_min, self.size_frac_max = a, b
        form.addRow("size fraction", r)
        self.position = QtWidgets.QComboBox()
        self.position.addItems(["random", "centered"])
        form.addRow("position", self.position)
        self.full_field_prob = _spin(0, 1, d.full_field_prob, tip="share of BOX samples replaced by the fully filled grid (type-CNN only)")
        form.addRow("full-field prob", self.full_field_prob)
        self.voxel_scale_mode.currentIndexChanged.connect(lambda _=0: self._scale_mode_changed())
        root.addWidget(box)
        self.fixed_only_rows = [self.grid_mm, self.size_frac_min, self.size_frac_max, self.position, self.full_field_prob]

        # ---- hollow / carve / cut (in-scale only)
        box = QtWidgets.QGroupBox("Hollow / carve / cut (--voxel-scale-mode fixed only)")
        self.inscale_box = box
        form = QtWidgets.QFormLayout(box)
        self.hollow_prob = _spin(0, 1, d.hollow_prob, tip="probability a box/sphere sample becomes a shell (hollow body, "
                                                          "closed cavity ON PURPOSE); drawn before carve/cut")
        form.addRow("hollow prob", self.hollow_prob)
        a, b, r = _range(0.0, 1.0, d.hollow_wall, 0.01, 3)
        self.hollow_wall_min, self.hollow_wall_max = a, b
        form.addRow("hollow wall fraction", r)
        self.carve_prob = _spin(0, 1, d.carve_prob, tip="probability a box/sphere sample is carved by one sphere/box "
                                                        "(the OTHER shape), accepted only if still one uniform manifold")
        form.addRow("carve prob", self.carve_prob)
        a, b, r = _range(0.0, 1.0, d.carve_size, 0.01, 3)
        self.carve_size_min, self.carve_size_max = a, b
        form.addRow("carve cutter size", r)
        a, b, r = _range(0.0, 1.0, d.carve_removed, 0.01, 3)
        self.carve_removed_min, self.carve_removed_max = a, b
        form.addRow("carve removed fraction", r)
        self.cut_prob = _spin(0, 1, d.cut_prob, tip="probability a sample is sliced by 1-2 planes (label stays the base shape)")
        form.addRow("cut prob", self.cut_prob)
        a, b, r = _range(0.0, 1.0, d.cut_keep, 0.01, 3)
        self.cut_keep_min, self.cut_keep_max = a, b
        form.addRow("cut keep fraction", r)
        self.cut_min_thickness = _int(0, 64, d.cut_min_thickness, "skip a cut leaving a plate thinner than this many voxels; 0 = off")
        form.addRow("cut min thickness (vox)", self.cut_min_thickness)
        self.max_cuts = _int(0, 10, d.max_cuts)
        form.addRow("max cuts", self.max_cuts)
        self.cut_axis_aligned_prob = _spin(0, 1, d.cut_axis_aligned_prob)
        form.addRow("cut axis-aligned prob", self.cut_axis_aligned_prob)
        self.require_manifold = QtWidgets.QCheckBox("require manifold (redraw up to 6x until the CLEAN part is one "
                                                     "piece with no enclosed void; hollow is exempt on purpose)")
        form.addRow(self.require_manifold)
        self.noise_prob = _spin(0, 1, d.noise_prob, tip="probability a sample gets noise at all, independent of class/kind")
        form.addRow("noise prob (gate)", self.noise_prob)
        root.addWidget(box)
        self.fixed_only_rows.append(box)

        # ---- voxel-grid noise
        box = QtWidgets.QGroupBox("Voxel-grid noise (needs --voxelize)")
        form = QtWidgets.QFormLayout(box)
        self.noise_salt = _spin(0, 1, d.noise_salt_prob, 0.001, 4, "fraction of EMPTY voxels flipped to occupied")
        self.noise_pepper = _spin(0, 1, d.noise_pepper_prob, 0.001, 4, "fraction of OCCUPIED voxels flipped to empty")
        self.noise_blob_count = _int(0, 50, d.noise_blob_count, "foreign-object cubes spliced in")
        self.noise_blob_min = _int(1, 64, d.noise_blob_min_size)
        self.noise_blob_max = _int(1, 64, d.noise_blob_max_size)
        self.noise_dilate = _spin(0, 1, d.noise_dilate_prob, tip="probability of one dilation pass (thickens surfaces)")
        self.noise_erode = _spin(0, 1, d.noise_erode_prob, tip="probability of one erosion pass (thins surfaces)")
        for label, w in (("salt prob", self.noise_salt), ("pepper prob", self.noise_pepper),
                         ("blob count", self.noise_blob_count), ("blob min size", self.noise_blob_min),
                         ("blob max size", self.noise_blob_max), ("dilate prob", self.noise_dilate),
                         ("erode prob", self.noise_erode)):
            form.addRow(label, w)
        root.addWidget(box)
        root.addStretch(1)

        self._mode_changed()
        self._scale_mode_changed()

    # ------------------------------------------------------------------------------------------ helpers
    def _select_topologies(self, names):
        for i in range(self.topo.count()):
            item = self.topo.item(i)
            on = names is None or item.text() in names
            item.setCheckState(QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def _selected_topologies(self):
        return tuple(self.topo.item(i).text() for i in range(self.topo.count())
                    if self.topo.item(i).checkState() == QtCore.Qt.Checked)

    def _mode_changed(self):
        mode = self.mode.currentData()
        self.count_type.setEnabled(mode == "type")
        for w in (self.measure_ratios, self.count_measure):
            w.setEnabled(mode == "measure")
        for w in (self.rotation_groups, self.count_rotation):
            w.setEnabled(mode == "rotation")

    def _scale_mode_changed(self):
        fixed = self.voxel_scale_mode.currentData() == "fixed"
        for w in self.fixed_only_rows:
            w.setEnabled(fixed)

    def _browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Output directory", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def collect(self) -> DatasetGenSettings:
        topologies = self._selected_topologies()
        if not topologies:
            raise ValueError("Select at least one topology.")
        out = Path(self.output_dir.text().strip())
        if not out.parts:
            raise ValueError("Set an output directory.")
        return DatasetGenSettings(
            mode=self.mode.currentData(), topologies=topologies,
            count_per_type=self.count_type.value(), measure_ratios_per_type=self.measure_ratios.value(),
            count_per_measure=self.count_measure.value(), rotation_groups_per_type=self.rotation_groups.value(),
            count_per_rotation=self.count_rotation.value(),
            output_dir=out, macro_dir=Path(self.macro_dir.text().strip()), freecad=self.freecad.text().strip(),
            scale_mm=self.scale_mm.value(), seed=self.seed.value(), shards=self.shards.value(),
            resume=self.resume.isChecked(),
            voxelize=self.voxelize.isChecked(), voxel_resolution=self.voxel_resolution.value(),
            voxel_scale_mode=self.voxel_scale_mode.currentData(), grid_mm=self.grid_mm.value(),
            size_fraction=(self.size_frac_min.value(), self.size_frac_max.value()), position=self.position.currentText(),
            full_field_prob=self.full_field_prob.value(),
            hollow_prob=self.hollow_prob.value(), hollow_wall=(self.hollow_wall_min.value(), self.hollow_wall_max.value()),
            carve_prob=self.carve_prob.value(), carve_size=(self.carve_size_min.value(), self.carve_size_max.value()),
            carve_removed=(self.carve_removed_min.value(), self.carve_removed_max.value()),
            cut_prob=self.cut_prob.value(), cut_keep=(self.cut_keep_min.value(), self.cut_keep_max.value()),
            cut_min_thickness=self.cut_min_thickness.value(), max_cuts=self.max_cuts.value(),
            cut_axis_aligned_prob=self.cut_axis_aligned_prob.value(), require_manifold=self.require_manifold.isChecked(),
            noise_prob=self.noise_prob.value(),
            noise_salt_prob=self.noise_salt.value(), noise_pepper_prob=self.noise_pepper.value(),
            noise_blob_count=self.noise_blob_count.value(), noise_blob_min_size=self.noise_blob_min.value(),
            noise_blob_max_size=self.noise_blob_max.value(), noise_dilate_prob=self.noise_dilate.value(),
            noise_erode_prob=self.noise_erode.value(),
        )
