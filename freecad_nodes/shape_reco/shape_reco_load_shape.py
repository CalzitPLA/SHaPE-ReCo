# -*- coding: utf-8 -*-
"""LoadShape node: stl_path (str) -> Part.Shape.

Bridges shape_reco's STL-path convention into the stock Nodes
workbench's Part.Shape-based nodes (Transforms category etc., e.g.
Rotate) - those operate on an in-memory Part.Shape, not a file path,
and Part.Shape's own read() doesn't understand the STL (mesh) format
at all ("Unknown extension" - it's built for BREP formats like STEP).
The bridge is via Mesh: load the STL as a Mesh.Mesh, then
Part.Shape.makeShapeFromMesh() turns its triangles into a (non-BREP,
but valid) Part.Shape shell - confirmed round-trips correctly through
rotate()/exportStl() (see shape_reco_export_stl.py, the reverse
bridge).
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
class ShapeRecoLoadShape(FCNNodeModel):

    icon: str = icon("nodes_python.svg")
    op_title: str = "Load Shape"
    op_category: str = "Shape Reco"
    content_label_objname: str = "fcn_node_bg"

    def __init__(self, scene):
        super().__init__(scene=scene,
                          inputs_init_list=[("STL Path", False), ("Tolerance", False)],
                          outputs_init_list=[("Shape", False)])

        self.grNode.resize(140, 90)
        for socket in self.inputs + self.outputs:
            socket.setSocketPosition()

    def eval_operation(self, sockets_input_data: list) -> list:
        stl_input = sockets_input_data[0]
        tolerance_input = sockets_input_data[1]
        if not stl_input:
            raise ValueError("Load Shape: connect an STL Path (Text node, or an upstream STL-producing node).")

        stl_path = str(_bootstrap.resolve_path(stl_input[0]))
        tolerance = float(tolerance_input[0]) if tolerance_input else 0.05

        import Mesh
        import Part

        mesh = Mesh.Mesh(stl_path)
        shape = Part.Shape()
        shape.makeShapeFromMesh(mesh.Topology, tolerance)

        return [[shape]]
