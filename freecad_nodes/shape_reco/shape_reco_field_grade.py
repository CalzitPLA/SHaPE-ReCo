# -*- coding: utf-8 -*-
"""FieldGrade node: the geometric grade of a reconstruction, in millimetres - the two-way chamfer distance between two meshes.

Exact closest-point queries (reco_field, dataset_gen/src/closest_point_field.py): the surface of the reconstruction to the input and back. Large
input->result means the model misses geometry; large result->input means it invents geometry. Unlike voxel IoU it is in the geometry's own units.

Result STL (e.g. Export Program's STL Path), Input STL and an optional Points (samples per direction, default 3000) ->
Chamfer (mean, mm) and Summary (mean, maximum and each direction). Cheap: no Run button.
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
class ShapeRecoFieldGrade(FCNNodeModel):

    icon: str = icon("nodes_python.svg")
    op_title: str = "Field Grade"
    op_category: str = "Shape Reco"
    content_label_objname: str = "fcn_node_bg"

    def __init__(self, scene):
        super().__init__(scene=scene,
                          inputs_init_list=[("Result STL", False), ("Input STL", False), ("Points", False)],
                          outputs_init_list=[("Chamfer", False), ("Summary", False)])

        self.grNode.resize(170, 120)
        for socket in self.inputs + self.outputs:
            socket.setSocketPosition()

    def eval_operation(self, sockets_input_data: list) -> list:
        result_input, input_input, points_input = sockets_input_data[0], sockets_input_data[1], sockets_input_data[2]
        if not result_input or not input_input:
            raise ValueError("Field Grade: connect Result STL (Export Program's STL Path) and Input STL.")
        result_stl = str(_bootstrap.resolve_path(result_input[0]))
        input_stl = str(_bootstrap.resolve_path(input_input[0]))
        n = int(points_input[0]) if points_input else 3000

        from reco_field.node_support import field_grade
        chamfer, summary = field_grade(result_stl, input_stl, n)
        return [[chamfer], [summary]]
