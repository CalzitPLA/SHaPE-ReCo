# -*- coding: utf-8 -*-
"""
Is FreeCAD's installed copy of the shape_reco nodes current, and do the example graphs load in the REAL Nodes workbench? Run with freecadcmd:

    QT_QPA_PLATFORM=offscreen freecadcmd scripts/check_nodes_install.py            # from anywhere; prints INSTALL / REGISTERED / LOADED lines

Symptoms of an outdated install (re-run `bash freecad_nodes/install.sh`, then restart FreeCAD):
  "TypeError: %d format: a real number is required, not str"  in get_class_from_opcode   a node in the graph is not registered (the workbench's own error
                                                                                        message is broken; the real error is OpCodeNotRegistered)
  "FCNSocketModel.__init__() got an unexpected keyword argument 'socket_type'"           the graph has more input sockets than the installed node class
  "KeyError: <socket id>" in Edge.deserialize                                            the wires of the nodes that failed to load
"""
import filecmp
import json
import os
import sys
import traceback

REPO = os.environ.get("SHAPE_RECO_SANDBOX_ROOT") or os.getcwd()
if not os.path.isdir(os.path.join(REPO, "freecad_nodes")):
    REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else "~/Documents/Sandbox"
MOD = os.environ.get("FREECAD_NODES_MOD", os.path.expanduser("~/.local/share/FreeCAD/Mod/Nodes"))
SRC, DST = os.path.join(REPO, "freecad_nodes", "shape_reco"), os.path.join(MOD, "nodes", "shape_reco")

bad = []
for f in sorted(os.listdir(SRC)):
    if f.endswith(".py"):
        if not os.path.isfile(os.path.join(DST, f)):
            bad.append("missing " + f)
        elif not filecmp.cmp(os.path.join(SRC, f), os.path.join(DST, f), shallow=False):
            bad.append("outdated " + f)
print("INSTALL", "current" if not bad else "OUTDATED: " + ", ".join(bad) + "  -> run: bash freecad_nodes/install.sh")

for p in (MOD, MOD + "/lib", MOD + "/nodes"):
    sys.path.insert(0, p)
os.environ.setdefault("SHAPE_RECO_SANDBOX_ROOT", REPO)
from qtpy.QtWidgets import QApplication  # noqa: E402
app = QApplication.instance() or QApplication(sys.argv)
from core.nodes_conf import NodesStore  # noqa: E402
NodesStore.refresh_nodes_list()
print("REGISTERED", sum("shape_reco" in k for k in NodesStore.nodes), "shape_reco nodes")
from core.nodes_sub_window import FCNSubWindow  # noqa: E402
examples = os.path.join(REPO, "freecad_nodes", "examples")
for name in sorted(f for f in os.listdir(examples) if f.endswith(".json")):
    d = json.load(open(os.path.join(examples, name)))
    missing = sorted({n["op_code"] for n in d["nodes"] if n["op_code"] not in NodesStore.nodes})
    w = FCNSubWindow()
    try:
        w.fileLoad(os.path.join(examples, name))
    except Exception:
        traceback.print_exc()
    print("LOADED", name, f"nodes {len(w.scene.nodes)}/{len(d['nodes'])} edges {len(w.scene.edges)}/{len(d['edges'])}",
          "OK" if not missing and len(w.scene.nodes) == len(d["nodes"]) and len(w.scene.edges) == len(d["edges"]) else f"PROBLEM missing={missing}", flush=True)
