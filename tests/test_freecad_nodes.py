"""The FreeCAD Nodes-workbench nodes, run WITHOUT FreeCAD: the workbench's base classes are replaced by minimal stand-ins, so the
node classes can be imported and their eval_operation exercised; plus a structural check of the example graphs (sockets vs the
node classes, edges, ids)."""
import ast
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
NODES = ROOT / "freecad_nodes" / "shape_reco"


@pytest.fixture(scope="module")
def registry():
    """Stub the Nodes workbench modules, import every shape_reco node, return {module_name: class}."""
    names = ("core", "core.nodes_conf", "core.nodes_default_node", "nodes_locator", "qtpy", "qtpy.QtWidgets", "nodeeditor",
             "nodeeditor.node_content_widget")
    saved = {k: sys.modules.get(k) for k in names}
    registered = {}

    class Base:
        def __init__(self, scene=None, inputs_init_list=(), outputs_init_list=()):
            self.inputs = [types.SimpleNamespace(name=n, setSocketPosition=lambda: None) for n, _ in inputs_init_list]
            self.outputs = [types.SimpleNamespace(name=n, setSocketPosition=lambda: None) for n, _ in outputs_init_list]
            self.grNode = types.SimpleNamespace(resize=lambda *a: None)
            self.inputs_init = [n for n, _ in inputs_init_list]
            self.outputs_init = [n for n, _ in outputs_init_list]

    def register_node(cls):
        registered[cls.__module__] = cls
        return cls

    core = types.ModuleType("core")
    conf = types.ModuleType("core.nodes_conf"); conf.register_node = register_node
    default = types.ModuleType("core.nodes_default_node"); default.FCNNodeModel = Base
    loc = types.ModuleType("nodes_locator"); loc.icon = lambda name: name
    qtpy = types.ModuleType("qtpy")
    widgets = types.ModuleType("qtpy.QtWidgets"); widgets.QPushButton = type("QPushButton", (), {}); widgets.QVBoxLayout = type("QVBoxLayout", (), {})
    editor = types.ModuleType("nodeeditor")
    content = types.ModuleType("nodeeditor.node_content_widget")
    content.QDMNodeContentWidget = type("QDMNodeContentWidget", (), {"__init__": lambda self, *a, **k: None})
    sys.modules.update({"core": core, "core.nodes_conf": conf, "core.nodes_default_node": default, "nodes_locator": loc,
                        "qtpy": qtpy, "qtpy.QtWidgets": widgets, "nodeeditor": editor, "nodeeditor.node_content_widget": content})
    sys.path.insert(0, str(NODES))
    sys.path.insert(0, str(ROOT))
    freecad_stubs = {name: types.ModuleType(name) for name in ("Mesh", "FreeCAD", "FreeCADGui", "Part", "MeshPart")}
    sys.modules.update(freecad_stubs)                       # some nodes import FreeCAD modules at the top; only during the imports
    try:
        for f in sorted(NODES.glob("shape_reco_*.py")):
            importlib.import_module(f.stem)
        for name in freecad_stubs:                          # the real code paths must not see a fake FreeCAD (node_support falls back to trimesh)
            sys.modules.pop(name, None)
        yield registered
    finally:
        sys.path.remove(str(NODES))
        for f in NODES.glob("shape_reco_*.py"):
            sys.modules.pop(f.stem, None)
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def make(registry, module):
    return registry[module](scene=None)


def ball(c, r, n=64):
    idx = np.indices((n, n, n)).astype(float)
    return (((idx[0] - c[0]) ** 2 + (idx[1] - c[1]) ** 2 + (idx[2] - c[2]) ** 2) <= r * r).astype(np.uint8)


# ------------------------------------------------------------------------------ the example graphs

def node_input_names(module):
    """Input names of a node class, read from the source (the class needs the workbench to be instantiated)."""
    tree = ast.parse((NODES / f"{module}.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "inputs_init_list":
            return [e.elts[0].value for e in node.value.elts]
    return None


@pytest.mark.parametrize("path", sorted((ROOT / "freecad_nodes" / "examples").glob("*.json")))
def test_example_graphs_are_structurally_consistent(path):
    d = json.loads(path.read_text())
    ids = [n["id"] for n in d["nodes"]] + [s["id"] for n in d["nodes"] for s in n["inputs"] + n["outputs"]] + [e["id"] for e in d["edges"]]
    assert len(ids) == len(set(ids)), "duplicate ids"
    sockets = {}
    for n in d["nodes"]:
        module = n["op_code"].split("'")[1].split(".")[0]
        if module.startswith("shape_reco_"):
            names = node_input_names(module)
            assert names is not None
            if path.stem.endswith("_legacy"):                     # an archived graph saved before optional inputs were appended
                assert len(n["inputs"]) <= len(names), f"{n['title']}: more inputs in the graph than in the node"
            else:
                assert len(n["inputs"]) == len(names), f"{n['title']}: {len(n['inputs'])} inputs in the graph, {names} in the node"
        for s in n["inputs"]:
            sockets[s["id"]] = ("in", n["title"], s["index"])
        for s in n["outputs"]:
            sockets[s["id"]] = ("out", n["title"], s["index"])
        assert [s["index"] for s in n["inputs"]] == list(range(len(n["inputs"])))
    used_inputs = set()
    for e in d["edges"]:
        assert sockets[e["start"]][0] == "out" and sockets[e["end"]][0] == "in", e
        assert e["end"] not in used_inputs, f"input socket wired twice: {sockets[e['end']]}"
        used_inputs.add(e["end"])


# ------------------------------------------------------------------------------------ the vector-field nodes

def box_stl(path, extents=(20.0, 14.0, 10.0)):
    import trimesh
    trimesh.creation.box(extents=extents).export(str(path))
    return str(path)


def test_vector_field_nodes_are_registered_with_their_sockets(registry):
    assert {"shape_reco_field_peel", "shape_reco_field_export", "shape_reco_field_grade"} <= set(registry)
    assert make(registry, "shape_reco_field_peel").inputs_init == ["Input STL", "Max Ops", "Target"]
    assert make(registry, "shape_reco_field_peel").outputs_init == ["Program", "Explained", "Summary"]
    assert make(registry, "shape_reco_field_export").inputs_init == ["Program", "Workdir", "Name"]
    assert make(registry, "shape_reco_field_export").outputs_init == ["STEP Path", "STL Path", "Summary"]
    assert make(registry, "shape_reco_field_grade").inputs_init == ["Result STL", "Input STL", "Points"]


def test_peel_node_waits_for_its_run_button_then_returns_the_program(registry, tmp_path):
    node = make(registry, "shape_reco_field_peel")
    stl = box_stl(tmp_path / "box.stl")
    assert node.eval_operation([[stl], [], []]) == [[], [], []]                     # not armed: nothing runs on file open
    with pytest.raises(ValueError, match="connect an Input STL"):
        node._armed = True
        node.eval_operation([[], [], []])
    node._armed = True
    program, explained, summary = node.eval_operation([[stl], ["4"], ["0.98"]])
    tree = json.loads(program[0])
    assert tree["type"] == "Placed" and tree["primitive"]["type"] == "Box" and sorted(tree["primitive"]["half"]) == pytest.approx([5, 7, 10], abs=0.1)
    assert explained[0] > 0.99 and "box" in summary[0] and "explained" in summary[0]
    assert node._armed is False                                                     # one click, one run


def test_export_and_grade_nodes_build_a_freecad_solid_and_grade_it(registry, tmp_path):
    from reco_field.export import find_freecadcmd
    if find_freecadcmd() is None:
        pytest.skip("freecadcmd not found")
    stl = box_stl(tmp_path / "box.stl")
    peel_node = make(registry, "shape_reco_field_peel")
    peel_node._armed = True
    program = peel_node.eval_operation([[stl], [], []])[0]
    export = make(registry, "shape_reco_field_export")
    assert export.eval_operation([program, [str(tmp_path)], []]) == [[], [], []]     # not armed
    export._armed = True
    with pytest.raises(ValueError, match="connect Program"):
        export.eval_operation([[], [], []])
    export._armed = True
    step, out_stl, summary = export.eval_operation([program, [str(tmp_path / "work")], ["part"]])
    assert Path(step[0]).name == "part.step" and Path(out_stl[0]).is_file() and "valid" in summary[0] and "2800.0" in summary[0]     # 20 x 14 x 10
    grade = make(registry, "shape_reco_field_grade")
    chamfer, text = grade.eval_operation([out_stl, [stl], ["1500"]])
    assert chamfer[0] < 0.01 and "chamfer" in text[0]
    other = box_stl(tmp_path / "big.stl", (24.0, 14.0, 10.0))
    assert 0.3 < grade.eval_operation([[other], [stl], []])[0][0] < 0.6              # the two x faces are 2 mm off (~20 % of the area, both ways)
    with pytest.raises(ValueError, match="connect Result STL"):
        grade.eval_operation([[], [stl], []])



def test_every_node_imports_and_the_kept_ones_are_registered(registry):
    """Only the nodes of the main line and the comparison route remain (route C's voxel nodes were removed 2026-09-30)."""
    assert {"shape_reco_cascade", "shape_reco_field_peel", "shape_reco_field_export", "shape_reco_field_grade",
            "shape_reco_load_mesh"} <= set(registry)


def test_the_vector_field_graph_carries_the_peel_pipeline():
    d = json.loads((ROOT / "freecad_nodes" / "examples" / "example_vector_field.json").read_text())
    titles = [n["title"] for n in d["nodes"]]
    assert titles.count("Peel (Vector Field)") == 1 and titles.count("Export Program") == 1 and titles.count("Field Grade") == 1
    sockets = {s["id"]: (n["title"], s["index"]) for n in d["nodes"] for s in n["inputs"] + n["outputs"]}
    wired = {(sockets[e["start"]], sockets[e["end"]]) for e in d["edges"]}
    assert (("Peel (Vector Field)", 0), ("Export Program", 0)) in wired              # the program
    assert (("Export Program", 1), ("Field Grade", 0)) in wired                      # the FreeCAD solid's STL is graded against the input
