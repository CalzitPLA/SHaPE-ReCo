"""The dumb shape search (reco_engine/shape_search.py, mode "search", 2026-10-16): every base shape tried on the nearby
surface samples, graded by the share of them on the shape's surface."""
import numpy as np
import pytest

from reco_engine import shape_search as ss
from reco_engine.result_to_program import primitive_to_sp


def _ring(rng, n, radius, z0, z1, c=(0.0, 0.0, 0.0), axis=None):
    """Points on a cylinder surface of `radius` between z0 and z1 about `axis` (default z) through `c`."""
    t = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(z0, z1, n)
    pts = np.stack([radius * np.cos(t), radius * np.sin(t), z], axis=1)
    if axis is not None:
        pts = pts @ ss.frame_with_z(axis).T
    return pts + np.asarray(c, float)


def _box_surface(rng, n, half, c=(0.0, 0.0, 0.0)):
    half = np.asarray(half, float)
    face = rng.integers(0, 6, n)
    p = rng.uniform(-1, 1, (n, 3)) * half
    for i in range(n):
        ax, side = face[i] % 3, 1 if face[i] >= 3 else -1
        p[i, ax] = side * half[ax]
    return p + np.asarray(c, float)


def test_search_finds_the_cylinder_and_its_radius():
    rng = np.random.default_rng(1)
    pts = _ring(rng, 2500, 12.0, -30.0, 30.0, c=(5.0, -3.0, 8.0), axis=(1.0, 0.3, 0.2))
    found = ss.search(pts, extent=100.0, evals_per_kind=150, seed=0)
    best = found[0]
    assert best.primitive.kind == "cylinder", [(f.primitive.kind, round(f.coverage, 2)) for f in found]
    assert best.coverage >= 0.9
    assert abs(best.primitive.dims_mm["radius"] - 12.0) <= 0.3
    assert best.primitive.sources[0].mode == ss.SOURCE and "search cylinder" in best.primitive.label


def test_search_finds_the_box_and_the_sphere():
    rng = np.random.default_rng(2)
    box = _box_surface(rng, 2500, (20.0, 8.0, 5.0), c=(1.0, 2.0, 3.0))
    found = ss.search(box, extent=100.0, evals_per_kind=120, seed=0)
    assert found[0].primitive.kind == "box" and found[0].coverage >= 0.9
    half = sorted(found[0].primitive.dims_mm.values())
    assert np.allclose(half, [5.0, 8.0, 20.0], atol=0.4)
    v = rng.normal(size=(2000, 3))
    sphere = 9.0 * v / np.linalg.norm(v, axis=1, keepdims=True) + np.array([4.0, 4.0, -2.0])
    found = ss.search(sphere, extent=100.0, evals_per_kind=60, seed=0)
    assert found[0].primitive.kind == "sphere" and found[0].coverage >= 0.95
    assert abs(found[0].primitive.dims_mm["radius"] - 9.0) <= 0.2


def test_coverage_is_the_share_of_points_on_the_surface():
    rng = np.random.default_rng(3)
    on = _ring(rng, 500, 10.0, -5.0, 5.0)
    off = on * 1.5                                                        # scaled away from the surface
    cyl = ss._Shape("cylinder", np.zeros(3), np.eye(3), np.array([10.0, 5.0])).primitive(1e-3)
    assert ss.coverage(cyl, on, 0.05) == 1.0
    assert ss.coverage(cyl, np.vstack([on, off]), 0.05) == pytest.approx(0.5)
    assert primitive_to_sp(cyl).sdf(np.array([[10.0, 0.0, 0.0]]))[0] == pytest.approx(0.0, abs=1e-9)


class _State:
    def __init__(self, n=10, pitch=2.0):
        g = (np.arange(n) + 0.5) * pitch
        self.centres = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1)
        self.pitch = pitch


def test_nearby_points_take_the_samples_in_the_dilated_component():
    st = _State()
    comp = np.zeros((10, 10, 10), bool)
    comp[4:6, 4:6, 4:6] = True                                            # a 2-voxel cube at 8..12 mm
    pts = np.array([[9.0, 9.0, 9.0], [13.0, 9.0, 9.0], [17.0, 9.0, 9.0], [-5.0, 0.0, 0.0]])
    near = ss.nearby_points(st, comp, pts, dilate=1)
    assert len(near) == 2 and near[0, 0] == 9.0 and near[1, 0] == 13.0   # inside, one voxel out; not two, not outside


def test_best_candidate_judges_every_kind_and_takes_the_covered_passing_one():
    rng = np.random.default_rng(4)
    st = _State(n=40, pitch=2.0)                                          # 0..80 mm
    comp = np.zeros((40, 40, 40), bool)
    comp[10:30, 10:30, 10:30] = True
    pts = _ring(rng, 3000, 15.0, 22.0, 58.0, c=(40.0, 40.0, 0.0))        # a cylinder inside the component

    class Field:                                                          # the samples come from the stub below
        pass

    ss._SAMPLES[id(Field)] = (Field, pts, 100.0)
    calls = []

    def evaluate(state, prim, op, purity_min):
        calls.append(prim.kind)
        ok = prim.kind != "cylinder"                                      # the loop refuses the best-covered one
        return dict(gain=100 if prim.kind == "box" else 50, ok=ok, reason="" if ok else "void check", s_new=None, s_prim=None)

    cfg = type("C", (), dict(min_gain=1, search_evals=40, search_min_coverage=0.0))()
    chosen = ss.best_candidate(st, comp, "cut", Field, cfg, 0.9, evaluate)
    kinds = [p.kind for p, _ in chosen["tried"]]
    assert set(kinds) == {"box", "sphere", "cylinder", "cone", "torus"} and kinds[0] == "cylinder"
    assert chosen["best"] is not None and chosen["best"][0].kind != "cylinder"
    assert chosen["best"][1]["surface"] == max(ev["surface"] for p, ev in chosen["tried"] if ev["ok"])
    assert all(ev["op"] == "cut" for _, ev in chosen["tried"])
    empty = ss.best_candidate(st, np.zeros((40, 40, 40), bool), "cut", Field, cfg, 0.9, evaluate)
    assert empty["best"] is None and "surface samples near" in empty["why"]
    strict = type("C", (), dict(min_gain=1, search_evals=40, search_min_coverage=1.01))()   # the coverage floor: nothing reaches it
    chosen = ss.best_candidate(st, comp, "cut", Field, strict, 0.9, evaluate)
    assert chosen["best"] is None and "101%" in chosen["why"]
    assert all("of the nearby surface on it" in ev["reason"] for p, ev in chosen["tried"] if p.kind != "cylinder")


def test_a_shape_that_swallows_surface_points_scores_lower():
    """The score is coverage minus the share of the part's surface samples strictly inside the shape (a cutter found
    by coverage alone reached into the material on every example part, 2026-10-16)."""
    rng = np.random.default_rng(5)
    on = _ring(rng, 600, 10.0, -5.0, 5.0)
    cyl = ss._Shape("cylinder", np.zeros(3), np.eye(3), np.array([10.0, 5.0])).primitive(1e-3)
    inner, outer = _ring(rng, 300, 8.0, -4.0, 4.0), _ring(rng, 300, 12.0, -4.0, 4.0)
    assert ss.swallowed(cyl, inner, 0.05) == 1.0 and ss.swallowed(cyl, outer, 0.05) == 0.0
    assert ss.score(cyl, on, 0.05, outer) == pytest.approx(1.0) and ss.score(cyl, on, 0.05, inner) == pytest.approx(0.0)
    # a plate of surface points just beyond the cylinder's end: a longer cylinder would swallow them
    t, r = rng.uniform(0, 2 * np.pi, 400), 9.0 * np.sqrt(rng.uniform(0, 1, 400))
    plate = np.stack([r * np.cos(t), r * np.sin(t), np.full(400, 6.0)], axis=1)
    found = ss.search(on, extent=100.0, kinds=("cylinder",), evals_per_kind=80, forbid=plate, seed=0)
    assert found[0].swallowed == 0.0 and abs(found[0].primitive.dims_mm["half_length"] - 5.0) <= 0.6
    assert found[0].score == pytest.approx(found[0].coverage - found[0].swallowed)


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "dataset_gen" / "examples" / "bs_box_two_pockets.stl").is_file(), reason="example part missing")
def test_search_mode_cuts_the_pockets_of_the_example_part():
    """bs_box_two_pockets in mode "search": the legacy start's box, then every cut from the search alone (no cascade,
    no fit), to IoU >= 0.95 (measured 2026-10-16: 0.983 in 4 operations, 20 s at 48^3)."""
    import os
    from reco_engine import residual_loop as rl
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    c = rl.ResidualConfig(mode="search", resolution=48, max_resolution=48, grid="cube", freecad=False, feature_phase=False,
                          voxel_fallback=False, field_backend="igl", polish=False, start_vary=False, exact_solids_gate=False)
    steps = list(rl.residual_steps(str(ROOT / "dataset_gen" / "examples" / "bs_box_two_pockets.stl"), c))
    done = steps[-1]
    assert done.kind == "done" and done.info["stats"]["iou"] >= 0.95
    ops = [o for o in done.result.operations if o.accepted]
    assert len(ops) >= 2 and ops[0].op == "add"
    for o in ops[1:]:
        assert o.op == "cut" and all(s.mode == ss.SOURCE for s in o.primitive.sources), o.primitive.sources
    cands = [s for s in steps if s.kind == "candidate"]
    assert cands and all(s.info.get("surface") is not None for s in cands)      # every candidate carries its coverage
