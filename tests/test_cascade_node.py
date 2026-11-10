"""The Cascade Reconstruct node (freecad_nodes/shape_reco/shape_reco_cascade.py) and its FreeCAD-free logic
(reco_engine/cascade_node_support.py), with a fake residual loop; plus its place in the example graphs."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from reco_engine import cascade_node_support as support  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult, StageSource  # noqa: E402
from test_freecad_nodes import make, registry  # noqa: E402,F401  (the stubbed Nodes workbench fixture)


@dataclass
class FakeStep:
    kind: str
    result: ReconstructionResult
    info: dict


BOX = Primitive("box", dims_mm={"x_half": 10, "y_half": 10, "z_half": 10}, confidence=0.97,
                sources=[StageSource("type", "cnn:type_v3", 0.97)])
BALL = Primitive("sphere", center_mm=(0, 0, 10), dims_mm={"radius": 6}, confidence=0.9,
                 sources=[StageSource("dimensions", "analytic:sphere_fit")])


def fake_steps(path, config):
    fake_steps.config = config
    r = ReconstructionResult(input_path=path)
    yield FakeStep("input", r, {})
    r = ReconstructionResult(input_path=path, operations=[Operation("add", BOX, net_gain=4000)], residual_voxels_neg=120)
    yield FakeStep("add", r, {"candidate": BOX})
    r = ReconstructionResult(input_path=path, operations=[Operation("add", BOX, net_gain=4000),
                                                          Operation("cut", BALL, net_gain=110, freecad_valid=True),
                                                          Operation("add", BALL, net_gain=-5, accepted=False)],
                             residual_voxels_pos=3, residual_voxels_neg=10, chamfer_mm=0.03)
    yield FakeStep("done", r, {})


def test_cascade_program_returns_the_schema_json_and_a_summary():
    program, pos, neg, summary = support.cascade_program("part.stl", top_k=3, max_ops=5, steps_fn=fake_steps)
    assert fake_steps.config["top_k"] == 3 and fake_steps.config["max_ops"] == 5 and fake_steps.config["op_guard"] is False
    back = ReconstructionResult.from_dict(json.loads(program))
    assert [(o.op, o.primitive.kind, o.accepted) for o in back.operations] == [("add", "box", True), ("cut", "sphere", True),
                                                                              ("add", "sphere", False)]
    assert (pos, neg) == (3, 10)
    assert summary.startswith("2 operation(s) accepted, 1 rejected; residual +3 / -10 voxels, chamfer 0.03")
    assert "add:box" in summary and "cut:sphere" in summary and "analytic:sphere_fit" in summary and "FreeCAD valid" in summary


def test_node_is_registered_waits_for_run_and_passes_its_inputs(registry, monkeypatch):  # noqa: F811
    assert "shape_reco_cascade" in registry
    node = make(registry, "shape_reco_cascade")
    assert node.inputs_init == ["Input STL", "Top K", "Max Ops", "Registry", "Op Guard"]
    assert node.outputs_init == ["Program", "Residual Pos", "Residual Neg", "Summary"]
    assert node.eval_operation([["x.stl"], [], [], [], []]) == [[], [], [], []]           # not armed: nothing on file open
    node._armed = True
    with pytest.raises(ValueError, match="connect an Input STL"):
        node.eval_operation([[], [], [], [], []])
    seen = {}

    def fake(path, **kw):
        seen.update(kw, path=path)
        return "{}", 1, 2, "summary"
    monkeypatch.setattr(support, "cascade_program", fake)
    node._armed = True
    out = node.eval_operation([["dataset_gen/examples/bs_box_sphere_pocket.stl"], ["3"], [""], ["models/registry.json"], ["1"]])
    assert out == [["{}"], [1], [2], ["summary"]] and node._armed is False
    assert Path(seen["path"]).is_absolute() and seen["top_k"] == 3 and seen["max_ops"] is None and seen["op_guard"] is True
    assert seen["registry_path"].endswith("models/registry.json")



def test_the_cascade_graph_carries_the_cascade_node():
    d = json.loads((ROOT / "freecad_nodes" / "examples" / "example_cascade.json").read_text())
    assert [n["title"] for n in d["nodes"]].count("Cascade Reconstruct") == 1
    sockets = {s["id"]: (n["title"], s["index"]) for n in d["nodes"] for s in n["inputs"] + n["outputs"]}
    wired = {sockets[e["end"]] for e in d["edges"]}
    assert {("Cascade Reconstruct", i) for i in range(5)} <= wired
