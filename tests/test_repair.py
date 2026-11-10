"""reco_engine/repair.py: a redundant cut is taken back, a cut that severs the model is taken back, the base add never."""

import numpy as np

from reco_engine import cascade
from reco_engine import residual_loop as rl
from reco_engine.repair import repair
from reco_engine.result_schema import Operation, Primitive, ReconstructionResult


def box(c, h):
    return Primitive(kind="box", center_mm=tuple(float(x) for x in c), dims_mm=dict(x_half=h[0], y_half=h[1], z_half=h[2]))


def state_of(occ, spec):
    t_in = np.stack([occ.astype(np.float32), np.where(occ, 1.0, -1.0).astype(np.float32)] + [np.zeros(occ.shape, np.float32)] * 3)
    return rl._State(t_in, spec)


def test_redundant_cut_is_taken_back_and_the_base_never():
    spec = cascade.box_grid_spec((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0), 0.5)
    c = spec.centres()
    occ = (np.abs(c[..., 0]) <= 8) & (np.abs(c[..., 1]) <= 8) & (np.abs(c[..., 2]) <= 8) & ~((c[..., 0] > 4) & (np.abs(c[..., 1]) <= 3))
    st = state_of(occ, spec)
    ops = [Operation("add", box((0, 0, 0), (8, 8, 8))), Operation("cut", box((6, 0, 0), (2, 3, 9))),
           Operation("cut", box((7, 0, 0), (1, 3, 9)))]                   # the third lies inside the second
    res = ReconstructionResult(input_path="x", operations=ops)
    dropped = repair(st, res, tolerance=1e-4)
    assert [d[0] for d in dropped] == [2] and "within" in dropped[0][1]
    assert ops[2].accepted is False and ops[0].accepted and ops[1].accepted and "repair" in ops[2].primitive.label
    assert repair(st, res) == []                                           # nothing more to take back


def test_severing_cut_is_taken_back_at_the_least_cost():
    spec = cascade.box_grid_spec((-10.0, -10.0, -4.0), (10.0, 10.0, 4.0), 0.5)
    c = spec.centres()
    occ = (np.abs(c[..., 0]) <= 8) & (np.abs(c[..., 1]) <= 8) & (np.abs(c[..., 2]) <= 3)
    st = state_of(occ, spec)
    ops = [Operation("add", box((0, 0, 0), (8, 8, 3))), Operation("cut", box((0, 6, 0), (9, 1, 4))),   # a slot: two pieces
           Operation("cut", box((0, -7, 0), (2, 1, 4)))]                                                # a small pocket
    res = ReconstructionResult(input_path="x", operations=ops)
    dropped = repair(st, res)
    assert [d[0] for d in dropped] == [1] and "pieces" in dropped[0][1]
    assert ops[1].accepted is False and ops[2].accepted
