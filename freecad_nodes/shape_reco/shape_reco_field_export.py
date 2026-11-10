# -*- coding: utf-8 -*-
"""FieldExport node: the program of Peel (Vector Field) -> a FreeCAD solid, as STEP (exact B-rep) and STL.

The primitive tree is rebuilt inside freecadcmd with the Part module's own shapes and booleans (reco_field/export.py, freecad_builder.py), applying the
Part conventions measured in docs/freecad_part_shapes.tex (box corner, sector start, plane-limited sphere latitude, the mirrored torus v).

Program (JSON from Peel (Vector Field)), Workdir (a directory, created) and an optional Name (default "reconstruction") ->
STEP Path, STL Path (feed it to Load Mesh to see the solid) and Summary (volume, validity, faces). Runs freecadcmd, so it has its own Run button.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import _bootstrap  # noqa: E402

from core.nodes_conf import register_node
from core.nodes_default_node import FCNNodeModel
from nodes_locator import icon


@register_node
class ShapeRecoFieldExport(FCNNodeModel):

    icon: str = icon("nodes_python.svg")
    op_title: str = "Export Program"
    op_category: str = "Shape Reco"
    content_label_objname: str = "fcn_node_bg"
    NodeContent_class = _bootstrap.RunButtonContent          # shells out to freecadcmd
    _armed: bool = False

    def __init__(self, scene):
        super().__init__(scene=scene,
                          inputs_init_list=[("Program", False), ("Workdir", False), ("Name", False)],
                          outputs_init_list=[("STEP Path", False), ("STL Path", False), ("Summary", False)])

        self.grNode.resize(180, 150)
        for socket in self.inputs + self.outputs:
            socket.setSocketPosition()

    def eval_operation(self, sockets_input_data: list) -> list:
        if not self._armed:
            return _bootstrap.not_armed_result(self)
        self._armed = False

        program_input, workdir_input, name_input = sockets_input_data[0], sockets_input_data[1], sockets_input_data[2]
        if not program_input or not workdir_input:
            raise ValueError("Export Program: connect Program (from Peel (Vector Field)) and Workdir.")
        workdir = str(_bootstrap.resolve_path(workdir_input[0]))
        name = str(name_input[0]) if name_input else "reconstruction"

        from reco_field.node_support import export_program
        step, stl, summary = export_program(program_input[0], workdir, name)
        return [[step], [stl], [summary]]
