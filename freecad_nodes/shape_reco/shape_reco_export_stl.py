# -*- coding: utf-8 -*-
"""ExportShapeSTL node: Part.Shape, output path (str) -> stl_path (str).

Reverse of shape_reco_load_shape.py: bridges a stock Nodes workbench
Part.Shape-producing node (e.g. Transforms > Rotate) back into
shape_reco's STL-path convention, so its result can feed Voxelize STL /
Load Mesh / any of the reco_engine-backed nodes the same way a plain
Text-node path does.
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
class ShapeRecoExportSTL(FCNNodeModel):

    icon: str = icon("nodes_python.svg")
    op_title: str = "Export Shape to STL"
    op_category: str = "Shape Reco"
    content_label_objname: str = "fcn_node_bg"

    def __init__(self, scene):
        super().__init__(scene=scene,
                          inputs_init_list=[("Shape", False), ("Output STL", False)],
                          outputs_init_list=[("STL Path", False)])

        self.grNode.resize(150, 90)
        for socket in self.inputs + self.outputs:
            socket.setSocketPosition()

    def eval_operation(self, sockets_input_data: list) -> list:
        shape_input = sockets_input_data[0]
        output_input = sockets_input_data[1]
        if not shape_input:
            raise ValueError("Export Shape to STL: connect a Shape input (e.g. Transforms > Rotate).")
        if not output_input:
            raise ValueError("Export Shape to STL: connect an Output STL path (Text node).")

        shape = shape_input[0]
        output_stl = _bootstrap.resolve_path(output_input[0])
        output_stl.parent.mkdir(parents=True, exist_ok=True)

        shape.exportStl(str(output_stl))

        return [[str(output_stl)]]
