# -*- coding: utf-8 -*-
"""Cascade Reconstruct node: an STL/STEP file -> the residual add-cut loop (reco_engine/residual_loop.py).

Per leftover component of the residual, the TYPE -> ROTATION -> DIMENSIONS -> POSITION cascade (networks and analytic fits from the
stage registry, models/registry.json by default) proposes a primitive that is added or cut until nothing worth fitting is left.

Input STL (path; a .step is tessellated by freecadcmd), optional Top K (default 2), Max Ops (default: the loop's own budget),
Registry (path of the stage registry) and Op Guard (1 = on; default 0 = bypassed) -> Program (ReconstructionResult as JSON,
reco_engine/result_schema.py), Residual Pos (voxels still missing), Residual Neg (voxels in excess), Summary (one line per operation).
Seconds to minutes per part, so the node has its own Run button. The logic is in reco_engine/cascade_node_support.py.
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
class ShapeRecoCascade(FCNNodeModel):

    icon: str = icon("nodes_python.svg")
    op_title: str = "Cascade Reconstruct"
    op_category: str = "Shape Reco"
    content_label_objname: str = "fcn_node_bg"
    NodeContent_class = _bootstrap.RunButtonContent          # networks + fitting: only when its own button is clicked
    _armed: bool = False

    def __init__(self, scene):
        super().__init__(scene=scene,
                          inputs_init_list=[("Input STL", False), ("Top K", False), ("Max Ops", False), ("Registry", False),
                                            ("Op Guard", False)],
                          outputs_init_list=[("Program", False), ("Residual Pos", False), ("Residual Neg", False),
                                             ("Summary", False)])

        self.grNode.resize(210, 190)
        for socket in self.inputs + self.outputs:
            socket.setSocketPosition()

    def eval_operation(self, sockets_input_data: list) -> list:
        if not self._armed:
            return _bootstrap.not_armed_result(self)
        self._armed = False

        stl_input, k_input, ops_input, registry_input, guard_input = sockets_input_data[:5]
        if not stl_input:
            raise ValueError("Cascade Reconstruct: connect an Input STL (Text node with a path to an STL or STEP file).")
        input_stl = str(_bootstrap.resolve_path(stl_input[0]))
        top_k = int(k_input[0]) if k_input and str(k_input[0]).strip() else 2
        max_ops = int(ops_input[0]) if ops_input and str(ops_input[0]).strip() else None
        registry = str(_bootstrap.resolve_path(registry_input[0])) if registry_input and str(registry_input[0]).strip() else None
        op_guard = bool(int(guard_input[0])) if guard_input and str(guard_input[0]).strip() else False

        from reco_engine.cascade_node_support import cascade_program
        program, pos, neg, summary = cascade_program(input_stl, top_k=top_k, max_ops=max_ops, registry_path=registry,
                                                     op_guard=op_guard)
        return [[program], [pos], [neg], [summary]]
