# -*- coding: utf-8 -*-
"""FieldPeel node: an STL/STEP file -> the CSG program of analytic primitives (the vector-field route, branch vector-field).

The closest-surface-point field of the mesh is fitted with sphere / box / cylinder / cone / torus / plane (the round ones with their sweep angle
u, sphere and torus also v) and a peeling loop explains the surface piece by piece: a base primitive, then cuts, clips and additions
(reco_field/peel.py, docs/vectorfield_reconstruction.tex). No voxels, no networks, no checkpoints.

Input STL (path; a .step is tessellated by freecadcmd), optional Max Ops (default 8) and Target (share of the surface to explain, default 0.985) ->
Program (the primitive tree as JSON text, for Export Program), Explained (0..1), Summary (one line per operation). A few seconds per part, so the node
has its own Run button. The logic is in reco_field/node_support.py.
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
class ShapeRecoFieldPeel(FCNNodeModel):

    icon: str = icon("nodes_python.svg")
    op_title: str = "Peel (Vector Field)"
    op_category: str = "Shape Reco"
    content_label_objname: str = "fcn_node_bg"
    NodeContent_class = _bootstrap.RunButtonContent          # seconds of fitting: only when its own button is clicked
    _armed: bool = False

    def __init__(self, scene):
        super().__init__(scene=scene,
                          inputs_init_list=[("Input STL", False), ("Max Ops", False), ("Target", False)],
                          outputs_init_list=[("Program", False), ("Explained", False), ("Summary", False)])

        self.grNode.resize(190, 150)
        for socket in self.inputs + self.outputs:
            socket.setSocketPosition()

    def eval_operation(self, sockets_input_data: list) -> list:
        if not self._armed:
            return _bootstrap.not_armed_result(self)
        self._armed = False

        stl_input, ops_input, target_input = sockets_input_data[0], sockets_input_data[1], sockets_input_data[2]
        if not stl_input:
            raise ValueError("Peel (Vector Field): connect an Input STL (Text node with a path to an STL or STEP file).")
        input_stl = str(_bootstrap.resolve_path(stl_input[0]))
        max_ops = int(ops_input[0]) if ops_input else 8
        target = float(target_input[0]) if target_input else 0.985

        from reco_field.node_support import peel_program
        program, explained, summary = peel_program(input_stl, max_ops, target)
        return [[program], [explained], [summary]]
