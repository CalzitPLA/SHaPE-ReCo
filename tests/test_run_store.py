"""The run store (reco_engine/run_store.py) and its command line (scripts/compare_runs.py), on a temporary store
($SHAPE_RECO_RESULTS) and a temporary git repository: save -> list -> load, null metrics, the dirty flag, no overwrite,
atomic writing, the diff of two runs. No Qt, no FreeCAD, no networks."""
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reco_engine import run_store  # noqa: E402
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult, StageSource  # noqa: E402


@dataclass
class Settings:
    input_stl: Path
    method: str = "cascade"
    cascade_top_k: int = 2
    cascade_grid: str = "tau"


@dataclass
class Config:
    top_k: int = 2
    feature_phase: bool = True
    feature_min_score: float = 0.85
    registry: list = field(default_factory=list)
    feature_build: object = None


def prim(kind, centre, dims, mode):
    return Primitive(kind, center_mm=centre, dims_mm=dims, confidence=0.9, sources=[StageSource("type", mode, 0.9)])


BOX = prim("box", (0, 0, 0), {"x_half": 10, "y_half": 10, "z_half": 10}, "fused")
HOLE = prim("cylinder", (6, -5, 0), {"radius": 1.5, "half_length": 11}, "feature:fit")
BOSS = prim("box", (7, 6, 7), {"x_half": 2, "y_half": 1, "z_half": 1}, "feature:step faces")


def make_session(part, ops, graded=True, finished=True, top_k=2, radius=None, files=None):
    """A stand-in for a finished cascade StepSession: states with step / info / result, a config, no recorder."""
    ops = [Operation(o.op, o.primitive, freecad_valid=o.freecad_valid) for o in ops]
    if radius is not None:
        ops[1].primitive = prim("cylinder", (6, -5, 0), {"radius": radius, "half_length": 11}, "feature:fit")
    files = files or {}
    result = ReconstructionResult(input_path=str(part), operations=ops + [Operation("add", BOSS, accepted=False)],
                                  residual_voxels_pos=3, residual_voxels_neg=0,
                                  chamfer_mm=0.031 if graded else None, exact_iou=0.97 if graded else None,
                                  step_path=files.get("step"), stl_path=files.get("stl"))

    def state(i, stage, message, info=None, phase=""):
        return SimpleNamespace(index=i, step=SimpleNamespace(stage=stage, phase=phase, message=message), info=info or {},
                               result=result, program=tuple(ops), timings=())
    states = [state(0, "input", "input part"), state(1, "add", "first addition add:box", phase="add"),
              state(2, "feature_done", "feature phase: max iterations", phase="feature",
                    info=dict(iterations=2, applied=1, rejected=3, skipped=0, stopped="max iterations", seconds=31.4,
                              exact_iou=0.9731, missing_volume=0.0, invented_volume=198.5))]
    if finished:
        grade = dict(valid=True, n_solids=1, chamfer=0.031, chamfer_max=0.4) if graded else None
        states.append(state(3, "done", "done", info=dict(
            stopped="residual within tolerance", fallback_boxes=0, input_voxels=4096, grade=grade, seconds=40.0,
            stats=dict(iou=np.float32(0.95), coverage=0.9, covered=1.0, score=0.9), cut_history=np.zeros((40, 40, 40)))))
    return SimpleNamespace(states=states, finished=finished, config=Config(top_k=top_k), recorder=None)


@pytest.fixture
def part(tmp_path):
    p = tmp_path / "in" / "frame.stl"
    p.parent.mkdir()
    p.write_bytes(b"solid frame\nendsolid frame\n")
    return p


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = tmp_path / "store"
    monkeypatch.setenv("SHAPE_RECO_RESULTS", str(s))
    return s


@pytest.fixture
def repo(tmp_path):
    """A temporary git repository with one commit (the real one is not touched)."""
    r = tmp_path / "repo"
    r.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(r), "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: E731
                                    check=True, capture_output=True)
    git("init", "-q", "-b", "work")
    (r / "code.py").write_text("x = 1\n")
    git("add", "code.py")
    git("commit", "-q", "-m", "one")
    return r


OPS = [Operation("add", BOX, freecad_valid=True), Operation("cut", HOLE, freecad_valid=True)]


def test_store_is_outside_the_repository(monkeypatch, tmp_path):
    monkeypatch.delenv("SHAPE_RECO_RESULTS", raising=False)
    assert run_store.store_root() == Path.home() / "shape_reco_results"
    assert ROOT not in run_store.store_root().parents
    monkeypatch.setenv("SHAPE_RECO_RESULTS", str(tmp_path / "x"))
    assert run_store.store_root() == tmp_path / "x"


def test_save_list_load_round_trip(store, part, repo, tmp_path):
    step, stl = tmp_path / "built.step", tmp_path / "built.stl"
    step.write_text("ISO-10303-21;")
    stl.write_text("solid x")
    session = make_session(part, OPS, files=dict(step=str(step), stl=str(stl)))
    path = run_store.save_run(session, Settings(part), part, label="first try", repo=repo, extra=dict(note="n"))
    assert path.parent == store / "frame" and path.name.endswith("_first-try") and not list(store.glob("*/*.tmp"))
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert f"_{commit[:8]}_" in path.name
    assert sorted(p.name for p in path.iterdir()) == ["model.step", "model.stl", "run.json", "steps.jsonl"]
    assert (path / "model.step").read_text() == "ISO-10303-21;"
    run = run_store.load_run(path)
    assert run["schema"] == run_store.SCHEMA and run["part"] == "frame" and run["label"] == "first try"
    assert run["method"] == "cascade" and run["finished"] is True and run["steps"] == 4 and run["host"]
    import hashlib
    assert run["input"] == dict(path=str(part), name="frame.stl", size=part.stat().st_size,
                                sha256=hashlib.sha256(part.read_bytes()).hexdigest())
    assert run["code"]["commit"] == commit and run["code"]["branch"] == "work" and run["code"]["dirty"] is False
    assert run["settings"]["gui"] == dict(input_stl=str(part), method="cascade", cascade_top_k=2, cascade_grid="tau")
    assert run["settings"]["config"]["feature_min_score"] == 0.85 and run["settings"]["config"]["feature_build"] is None
    assert run["model"] == dict(step="model.step", stl="model.stl", note=None, step_source=str(step), stl_source=str(stl),
                                fcstd=None)
    assert run["view"] is None and run["extra"] == {"note": "n"}
    # the program: the accepted operations, by the schema's own serialisation
    assert [(o["op"], o["primitive"]["kind"]) for o in run["program"]] == [("add", "box"), ("cut", "cylinder")]
    back = ReconstructionResult.from_dict(dict(input_path=str(part), operations=run["program"]))
    assert back.operations[1].primitive.dims_mm == {"radius": 1.5, "half_length": 11}
    assert back.operations[1].primitive.sources[0].mode == "feature:fit"
    m = run["metrics"]
    assert m["exact_iou"] == 0.97 and m["chamfer_mm"] == 0.031 and m["chamfer_mean"] == 0.031 and m["chamfer_max"] == 0.4
    assert m["voxel_iou"] == pytest.approx(0.95) and m["solution_stats"]["coverage"] == 0.9
    assert m["freecad_valid"] is True and m["n_solids"] == 1 and m["fallback_boxes"] == 0 and m["input_voxels"] == 4096
    assert m["stopped"] == "residual within tolerance" and m["n_operations"] == 2
    assert m["operations_by_kind"] == {"add:box": 1, "cut:cylinder": 1}
    assert m["operations_by_source"] == {"feature:fit": 1, "fused": 1}
    assert m["feature_phase"]["stopped"] == "max iterations" and m["feature_phase"]["applied"] == 1
    assert m["invented_volume"] == 198.5 and m["missing_volume"] == 0.0            # the phase's exact volumes
    assert run["timings"]["wall"] is None and run["timings"]["peak_rss_mb"] > 0
    steps = [json.loads(line) for line in (path / "steps.jsonl").read_text().splitlines()]
    assert [s["stage"] for s in steps] == ["input", "add", "feature_done", "done"] and steps[2]["seconds"] == 31.4
    assert steps[1] == dict(index=1, stage="add", phase="add", message="first addition add:box", seconds=None)
    listed = run_store.list_runs("frame")
    assert [r["path"] for r in listed] == [str(path)] and run_store.list_runs("liner") == []
    assert run_store.list_runs(sha256=run["input"]["sha256"])[0]["name"] == path.name


def test_one_part_has_one_directory_whatever_its_spelling(store, part, repo, tmp_path):
    a = tmp_path / "Liner-LINER-001.stl"
    b = tmp_path / "Liner_LINER_001.step"
    a.write_bytes(b"solid a\nendsolid a\n")
    b.write_bytes(b"ISO-10303-21;")
    first = run_store.save_run(make_session(a, OPS), Settings(a), a, repo=repo)
    second = run_store.save_run(make_session(b, OPS), Settings(b), b, repo=repo)
    other = run_store.save_run(make_session(part, OPS), Settings(part), part, repo=repo)
    assert first.parent == second.parent == store / "Liner-LINER-001" and other.parent == store / "frame"
    assert run_store.part_key("Liner-LINER-001") == run_store.part_key("liner_st1a07599_001") == "linerst1a07599001"
    for name in ("liner", "Liner_LINER_001", "LINER-LINER-001", "st1a07599"):       # start, spelling, a part of it
        assert {r["path"] for r in run_store.list_runs(name)} == {str(first), str(second)}, name
    assert run_store.list_runs("housing") == []
    assert run_store.resolve("liner:latest") in (first, second) and run_store.resolve("frame:1") == other


def test_a_saved_run_gets_a_label_instead_of_a_copy(store, part, repo):
    path = run_store.save_run(make_session(part, OPS), Settings(part), part, repo=repo)
    named = run_store.relabel(path, "coarse rough")
    assert named.name == path.name + "_coarse-rough" and not path.exists() and len(list((store / "frame").iterdir())) == 1
    run = run_store.load_run(named)
    assert run["label"] == "coarse rough" and run["name"] == named.name
    again = run_store.relabel(named, "kept")                                      # the old label is replaced
    assert again.name == path.name + "_kept" and run_store.load_run(again)["label"] == "kept"
    assert run_store.relabel(again, "  ") == again


def test_the_feature_document_is_stored_with_the_run(store, part, repo, tmp_path):
    document = tmp_path / "features.FCStd"
    document.write_bytes(b"PK document")
    session = make_session(part, OPS)
    session.states[2].info["fcstd"] = str(document)
    path = run_store.save_run(session, Settings(part), part, repo=repo)
    assert (path / "features.FCStd").read_bytes() == b"PK document"
    model = run_store.load_run(path)["model"]
    assert model["fcstd"] == "features.FCStd" and model["fcstd_source"] == str(document)
    session.states[2].info["fcstd"] = str(tmp_path / "gone.FCStd")                # a document that is not there
    assert run_store.load_run(run_store.save_run(session, Settings(part), part, repo=repo))["model"]["fcstd"] is None


def test_a_metric_that_was_not_computed_is_null(store, part, repo):
    path = run_store.save_run(make_session(part, OPS, graded=False, finished=False), Settings(part), part, repo=repo)
    run = json.loads((path / "run.json").read_text())
    m = run["metrics"]
    for name in ("chamfer_mm", "chamfer_mean", "chamfer_max", "voxel_iou", "solution_stats", "n_solids", "fallback_boxes",
                 "stopped", "genus", "exact_r_plus", "grade"):
        assert m[name] is None, name
    assert m["exact_iou"] == 0.9731                                   # known from the feature phase only
    assert m["residual_voxels_neg"] == 0                              # a computed zero stays a zero
    assert run["finished"] is False and run["model"]["step"] is None and "the run built none" in run["model"]["note"]
    text = (path / "run.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    assert run_store.plain(float("nan")) is None and run_store.plain(np.arange(1000)) == {"array": [1000], "dtype": "int64"}


def test_a_result_alone_can_be_saved(store, part, repo):
    result = ReconstructionResult(input_path=str(part), operations=list(OPS), chamfer_mm=0.02)
    run = run_store.load_run(run_store.save_run(result, None, part, repo=repo))
    assert run["method"] == "cascade" and run["finished"] is True and run["metrics"]["chamfer_mm"] == 0.02
    assert run["metrics"]["n_operations"] == 2 and run["settings"] == dict(gui=None, config=None) and run["steps"] is None


def test_dirty_working_tree_is_recorded(store, part, repo):
    (repo / "code.py").write_text("x = 2\n")
    (repo / "untracked.txt").write_text("not code that ran")
    run = run_store.load_run(run_store.save_run(make_session(part, OPS), Settings(part), part, repo=repo))
    assert run["code"]["dirty"] is True and run["code"]["modified"] == ["code.py"]
    assert run_store.git_state(repo.parent / "nowhere")["commit"] is None


def test_no_overwrite_and_no_half_run(store, part, repo, monkeypatch):
    session = make_session(part, OPS)
    paths = [run_store.save_run(session, Settings(part), part, repo=repo) for _ in range(3)]
    assert len(set(paths)) == 3 and all(p.is_dir() for p in paths)               # the same second: three directories
    assert len(run_store.list_runs("frame")) == 3

    def fail(*a, **k):
        raise RuntimeError("disk full")
    monkeypatch.setattr(run_store, "collect_steps", fail)
    with pytest.raises(RuntimeError):
        run_store.save_run(session, Settings(part), part, repo=repo)
    assert len(list((store / "frame").iterdir())) == 3                            # nothing half-written is left
    (store / "frame" / "20260101-000000_abc.tmp").mkdir()                        # a crashed writer's directory
    assert len(run_store.list_runs("frame")) == 3


def test_screenshot_is_kept_only_when_written(store, part, repo):
    session = make_session(part, OPS)
    a = run_store.save_run(session, Settings(part), part, repo=repo, screenshot=lambda p: bool(Path(p).write_bytes(b"png")))
    b = run_store.save_run(session, Settings(part), part, repo=repo, screenshot=lambda p: False)
    c = run_store.save_run(session, Settings(part), part, repo=repo, screenshot=lambda p: 1 / 0)
    assert (a / "view.png").is_file() and run_store.load_run(a)["view"] == "view.png"
    assert not (b / "view.png").exists() and run_store.load_run(b)["view"] is None and run_store.load_run(c)["view"] is None


def saved_pair(part, repo):
    a = run_store.save_run(make_session(part, OPS), Settings(part), part, label="a", repo=repo)
    later = make_session(part, OPS + [Operation("add", BOSS, freecad_valid=True)], top_k=3, radius=1.75)
    later.states[-1].info["stats"]["iou"] = 0.975
    b = run_store.save_run(later, Settings(part, cascade_grid="cube"), part, label="b", repo=repo)
    return a, b


def test_diff_finds_setting_metric_parameter_and_operation(store, part, repo):
    a, b = saved_pair(part, repo)
    d = run_store.diff_runs(a, b)
    assert d["same_input"] and d["same_code"] and d["a"]["label"] == "a" and d["b"]["code"]["branch"] == "work"
    assert {(s["key"], s["a"], s["b"]) for s in d["settings"]} == {("gui.cascade_grid", "tau", "cube"), ("config.top_k", 2, 3)}
    metrics = {m["key"]: m for m in d["metrics"]}
    iou = metrics["voxel_iou"]
    assert iou["changed"] and iou["a"] == pytest.approx(0.95) and iou["difference"] == pytest.approx(0.025)
    assert metrics["n_operations"]["difference"] == 1 and not metrics["exact_iou"]["changed"]
    assert metrics["exact_iou"]["difference"] == 0 and metrics["genus"]["a"] is None and metrics["genus"]["difference"] is None
    assert metrics["operations_by_source.feature:step faces"]["a"] is None
    ops = d["operations"]
    assert [o["status"] for o in ops] == ["same", "changed", "only in b"]
    assert ops[1]["parameters"] == [dict(key="dims_mm.radius", a=1.5, b=1.75, difference=0.25)]
    assert ops[2]["a"] is None and ops[2]["b"] == "add:box"
    assert d["summary"] == dict(settings_changed=2, metrics_changed=d["summary"]["metrics_changed"], operations_a=2,
                                operations_b=3, operations_changed=2)
    same = run_store.diff_runs(a, a)
    assert same["settings"] == [] and same["summary"]["metrics_changed"] == 0 and same["summary"]["operations_changed"] == 0
    assert run_store.diff_runs(a, b, tolerance=0.5)["operations"][1]["status"] == "same"      # below the tolerance
    assert [o["status"] for o in run_store.diff_runs(b, a)["operations"]] == ["same", "changed", "only in a"]
    json.dumps(d)                                                                       # pure data


def test_previous_run_is_the_newest_earlier_one_of_the_same_input(store, part, repo, tmp_path):
    a, b = saved_pair(part, repo)
    other = tmp_path / "in" / "liner.stl"
    other.write_bytes(b"another part")
    c = run_store.save_run(make_session(other, OPS), Settings(other), other, repo=repo)
    assert run_store.previous_run(run_store.load_run(b))["path"] == str(a)
    assert run_store.previous_run(run_store.load_run(a)) is None and run_store.previous_run(run_store.load_run(c)) is None
    assert run_store.resolve("frame:latest") == b and run_store.resolve("frame:previous") == a
    assert run_store.resolve("frame:2") == a and run_store.resolve(str(a)) == a
    with pytest.raises(ValueError):
        run_store.resolve("frame:3")
    with pytest.raises(ValueError):
        run_store.resolve("frame:newest")


# The test of the old `scripts/compare_runs.py list|diff` CLI was retired on 2026-10-15: that script became the
# run-directory comparer (several run_part.py directories side by side) on 2026-10-10 and has no list/diff.


# ------------------------------------------------------------------------------------------------ deep composites
# T-32 (2026-10-01): the store cut a program's composite tools below a fixed depth down to their text (12 levels
# until 2026-09-29: the liner's rounded pockets and hole stacks; 400 until 2026-10-01), so the run could not be built
# again. A program is now written whole at any depth.

def deep_tool(depth, width, seed=0):
    """A composite tool `depth` levels deep - a left-deep chain of booleans, as a rounded pocket or a hole stack is -
    and wide: every level's second operand is a fuse of `width` leaves. Every node has its own placement, dimensions,
    label and sources (numpy numbers among them), some leaves a sweep, some levels an alternative."""
    rng = np.random.default_rng(seed)

    def quaternion():
        q = rng.normal(size=4)
        return tuple(q / np.linalg.norm(q))                        # numpy float64 components

    def centre():
        return tuple(float(x) for x in rng.uniform(-500.0, 500.0, 3))

    def leaf(i, j):
        kind = ("box", "cylinder", "cone", "torus", "sphere")[(i + j) % 5]
        r = lambda: float(rng.uniform(0.5, 40.0))                       # noqa: E731
        dims = dict(box=dict(x_half=np.float64(r()), y_half=r(), z_half=np.float32(r())),
                    cylinder=dict(radius=r(), half_length=np.float64(r())),
                    cone=dict(radius1=r(), radius2=r(), half_length=r()),
                    torus=dict(major_radius=r() + 40.0, minor_radius=r()), sphere=dict(radius=r()))[kind]
        sweep = (0.0, float(rng.uniform(1.0, 6.2))) if kind in ("cylinder", "torus") else None
        return Primitive(kind, quaternion=quaternion(), center_mm=centre(), dims_mm=dims, u=sweep,
                         v=(-0.5, 0.5) if kind == "sphere" else None, confidence=float(rng.uniform()),
                         label=f"leaf {i}.{j}", residual=0.25 if j == 1 else None,
                         sources=[StageSource("dimensions", "analytic:step_faces", 1.0, 0.0, detail=dict(piece=i, faces=[j, j + 1]))])

    node = leaf(0, 0)
    for i in range(1, depth + 1):
        side = leaf(i, 0)
        for j in range(1, width):
            side = Primitive("composite", center_mm=centre(), parts=[side, leaf(i, j)], combine="fuse")
        node = Primitive("composite", quaternion=quaternion(), center_mm=centre(), parts=[node, side],
                         combine=("cut", "common", "fuse")[i % 3], label=f"level {i}", confidence=1.0,
                         sources=[StageSource("type", "feature:tool (contact faces)", 0.99, 0.0, detail=dict(piece=i))],
                         alternatives=[leaf(i, 99)] if i % 10 == 0 else [])
    return node


def nodes(prim):
    """Every primitive of a tree (a Primitive or its stored dict), depth first, as (path, the stored fields), without
    recursion: the tree is compared node for node, parameters and placements."""
    out, stack = [], [(prim, "")]
    while stack:
        p, path = stack.pop()
        d = run_store.plain(p, limit=None) if isinstance(p, Primitive) else p
        assert isinstance(d, dict), f"{path}: {str(d)[:60]!r} where a primitive is expected"
        out.append((path, {k: v for k, v in d.items() if k not in ("parts", "alternatives")}))
        stack += [(q, f"{path}.alternatives[{i}]") for i, q in reversed(list(enumerate(d.get("alternatives") or [])))]
        stack += [(q, f"{path}.parts[{i}]") for i, q in reversed(list(enumerate(d.get("parts") or [])))]
    return out


def json_depth(x):
    depth, stack = 0, [(x, 1)]
    while stack:
        y, d = stack.pop()
        depth = max(depth, d)
        stack += [(v, d + 1) for v in (y.values() if isinstance(y, dict) else y if isinstance(y, list) else ())]
    return depth


def test_a_deeply_nested_composite_round_trips_node_for_node(store, part, repo):
    tool = deep_tool(depth=64, width=4)
    ops = [Operation("add", BOX, freecad_valid=True), Operation("cut", tool, freecad_valid=True), Operation("cut", HOLE)]
    result_ops = make_session(part, ops).states[-1].result.operations
    assert run_store.truncated(run_store.plain(ReconstructionResult("x", result_ops).to_dict())["operations"])  # as cut before
    path = run_store.save_run(make_session(part, ops), Settings(part), part, repo=repo)
    run = run_store.load_run(path)
    assert run["program_source"] == "run.json" and run["program_truncated"] == 0 and run_store.truncated(run["program"]) == []
    stored = json.loads((path / "run.json").read_text())["program"]
    assert json_depth(stored[1]["primitive"]) >= 2 * 64 and len(nodes(tool)) == 1 + 64 * (2 * 4) + 6
    assert [nodes(o["primitive"]) for o in stored] == [nodes(o.primitive) for o in ops]     # node for node
    back = ReconstructionResult.from_dict(dict(input_path=str(part), operations=run["program"]))
    assert [o.primitive for o in back.operations] == [o.primitive for o in ops]             # the dataclasses, equal
    assert back.operations[1].primitive.parts[0].parts[0].parts[1].parts[0].parts[0].dims_mm == \
        tool.parts[0].parts[0].parts[1].parts[0].parts[0].dims_mm


def test_a_composite_deeper_than_any_former_limit_is_written_whole(store, part, repo):
    tool = deep_tool(depth=300, width=1, seed=1)                   # 600 JSON levels: beyond the limit of 400 till 10-01
    result = ReconstructionResult(input_path=str(part), operations=[Operation("add", BOX), Operation("cut", tool)])
    run = run_store.load_run(run_store.save_run(result, None, part, repo=repo))
    assert run_store.truncated(run["program"]) == [] and json_depth(run["program"][1]["primitive"]) > 600
    assert nodes(run["program"][1]["primitive"]) == nodes(tool)


def write_cut_run(part, repo, ops):
    """A run as the store wrote it before 2026-10-01 for a program this deep: the program cut to text at 12 levels."""
    path = run_store.save_run(ReconstructionResult(input_path=str(part), operations=ops), None, part, repo=repo)
    run = json.loads((path / "run.json").read_text())
    whole = run["program"]
    run["program"] = run_store.plain(ReconstructionResult(str(part), ops).to_dict())["operations"]
    (path / "run.json").write_text(json.dumps(run, indent=1))
    return path, whole


def test_a_program_cut_by_an_older_store_is_reported_and_its_recovery_read(store, part, repo):
    ops = [Operation("add", BOX), Operation("cut", deep_tool(depth=8, width=2))]
    path, whole = write_cut_run(part, repo, ops)
    run = run_store.load_run(path)
    cut = run_store.truncated(run["program"])
    assert cut and run["program_truncated"] == len(cut) and run["program_source"] == "run.json"
    assert cut[0].startswith("[1].primitive.parts[0].parts[0].parts[0].parts[0].parts[") and not any(c.startswith("[0]") for c in cut)
    assert run_store.truncated(whole) == [] and run_store.truncated(None) == []
    text = (path / "run.json").read_text()
    (path / run_store.RECOVERED_PROGRAM).write_text(json.dumps(dict(recovery={}, program=whole)))
    run = run_store.load_run(path)
    assert run["program_source"] == run_store.RECOVERED_PROGRAM and run["program"] == whole and run["program_truncated"] == len(cut)
    assert (path / "run.json").read_text() == text                                          # never rewritten
    assert [r["path"] for r in run_store.list_runs("frame")] == [str(path)]


def job_prim(p: Primitive) -> dict:
    """A primitive as the feature phase writes it into features_job.json (residual_loop's prim_job)."""
    d = dict(kind=p.kind, dims_mm=dict(p.dims_mm), center_mm=list(p.center_mm), quaternion=list(p.quaternion), u=p.u,
             v=p.v, label=p.label)
    if p.kind == "composite":
        d.update(combine=p.combine, parts=[job_prim(q) for q in p.parts])
    return d


def geometry(prim):
    keys = ("kind", "combine", "label", "quaternion", "center_mm", "dims_mm", "u", "v")
    return [(path, {k: d.get(k) for k in keys}) for path, d in nodes(prim)]


def test_recover_run_program_rebuilds_the_cut_tools_from_the_feature_job(store, part, repo, tmp_path, monkeypatch, capsys):
    import time as _time
    import recover_run_program as rr
    tools = [deep_tool(depth=9, width=2, seed=s) for s in (2, 3)]
    ops = [Operation("add", BOX), Operation("cut", tools[0]), Operation("cut", tools[1])]
    for o in ops[1:]:                     # the job writes JSON numbers (no float32); the liner's had plain floats
        stack = [o.primitive]
        while stack:
            p = stack.pop()
            p.dims_mm = {k: float(v) for k, v in p.dims_mm.items()}
            p.quaternion = tuple(float(x) for x in p.quaternion)
            stack += p.parts + p.alternatives
    path, whole = write_cut_run(part, repo, ops)
    jobs = tmp_path / "jobs"
    monkeypatch.setattr(rr, "JOBS", jobs)
    started = _time.strftime("%Y%m%d-%H%M%S", _time.localtime(_time.time() - 60))
    job = jobs / f"frame_{started}" / "features" / "features_job.json"
    job.parent.mkdir(parents=True)
    features = [dict(id=i, primitive=job_prim(o.primitive)) for i, o in enumerate(ops[1:])][::-1]   # in another order
    job.write_text(json.dumps(dict(input=str(part), final="", features=features), default=str))
    r = rr.recover(path)
    assert r["status"] == "recovered", r.get("tried")
    assert r["first_feature"] == 1 and r["counts"]["from_job"] > 0
    assert run_store.truncated(r["program"]) == []
    assert [geometry(o["primitive"]) for o in r["program"]] == [geometry(o["primitive"]) for o in whole]
    assert r["program"][1]["primitive"]["sources"] == whole[1]["primitive"]["sources"]      # kept where not cut
    assert rr.main([str(path), "--write"]) == 0 and "recovered" in capsys.readouterr().out
    run = run_store.load_run(path)
    assert run["program_source"] == run_store.RECOVERED_PROGRAM and run["program"] == r["program"]
    # a job that is not the run's: one placement differs where the store kept it whole -> nothing recovered
    other = json.loads(job.read_text())
    other["features"][0]["primitive"]["parts"][0]["center_mm"][0] += 1e-9
    job.write_text(json.dumps(other))
    assert rr.recover(path)["status"] == "not recovered"
