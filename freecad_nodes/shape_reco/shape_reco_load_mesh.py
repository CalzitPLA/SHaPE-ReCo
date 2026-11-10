# -*- coding: utf-8 -*-
"""LoadMesh node: stl_path (str) -> Mesh.Mesh object.

Bridges shape_reco's STL outputs (Build Primitive's rebuilt primitive,
or any input STL) into the Nodes workbench's own built-in "Viz"
category viewer nodes (MViewer expects a Mesh.Mesh on its "In" socket;
see nodes/viz/viz_mesh_viewer.py), so reconstructed/input shapes show
up in the FreeCAD 3D view instead of only existing as files on disk.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import _bootstrap  # noqa: E402

import Mesh

from core.nodes_conf import register_node
from core.nodes_default_node import FCNNodeModel
from nodes_locator import icon


@register_node
class ShapeRecoLoadMesh(FCNNodeModel):

    icon: str = icon("nodes_python.svg")
    op_title: str = "Load Mesh"
    op_category: str = "Shape Reco"
    content_label_objname: str = "fcn_node_bg"

    def __init__(self, scene):
        super().__init__(scene=scene,
                          inputs_init_list=[("STL Path", False)],
                          outputs_init_list=[("Mesh", True)])

        self.grNode.resize(140, 70)
        for socket in self.inputs + self.outputs:
            socket.setSocketPosition()

    def eval_operation(self, sockets_input_data: list) -> list:
        stl_input = sockets_input_data[0]
        if not stl_input:
            raise ValueError("Load Mesh: connect an STL Path (Text node, or Build Primitive's output).")

        stl_path = str(_bootstrap.resolve_path(stl_input[0]))
        mesh = Mesh.Mesh(stl_path)

        return [[mesh]]
