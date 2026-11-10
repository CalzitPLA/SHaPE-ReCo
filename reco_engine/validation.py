#!/usr/bin/env python3
"""
The validation cycle: a reconstruction of a shape whose CSG tree is KNOWN (reco_engine/truth.py) is rated against that
tree with a multivariate metric, graded in tiers, and its defects are traced back along the run's path (path.jsonl,
reco_engine/run_path.py) to the step that caused them - the separation point.

    rating = rate_run(run_dir, truth_json)       # writes <run_dir>/rating.json

Everything is relative to D, the diagonal of the truth's box, and computed on the analytic trees (the truth's and the
program's, dataset_gen/src/sdf_primitives.py), whose sign is exact:

  volume     one jittered point per cell of a cells^3 grid over the joint box (truth + every add's primitive). Each
             operation is folded once, in the order of program_model: an add writes 1 inside its primitive, a cut writes
             0 inside, an intersect writes 0 outside; a point's state is its last writer's value. IoU, missing and
             invented share, containment - for the final program and for every prefix (the error trajectory).
  surface    area-uniform boundary samples of both trees (csg_surface.boundary_samples), bracketed distances both ways
             (|sdf| below, nearest sample or Newton projection above; graded on the upper bound): mean, signed mean, std,
             RMS, median, p95, p99, max (Hausdorff H = the larger max); coverage over a ladder of tolerances tau x D -
             truth area within tau with normals within 2 deg (recall) and model area within tau of the truth (precision);
             per prefix, which truth surface is covered (first covered, last lost).
  topology   the build's validity and solid count and the model's genus (run_part's record: FreeCAD and OCCT) against the
             truth's genus (its mesh).
  tree       diagnostic only (equivalent trees exist): leaf matching by type, centre, size and orientation.
  validity   the analytic model against the built one (volume, and IoU against OCCT's exact IoU): a disagreement makes the
             rating "unmeasured" (a convention mismatch, not a reconstruction error).

Grade (GRADE_V1, to be calibrated on a calibration split and then frozen): hard conditions (a program, one valid solid,
the truth's genus) or tier E; then the worst metric decides among A..D (IoU, H/D, RMS/D, coverage both ways at a tau
coupled to the tier).

Attribution (per volume point, from the write history): missing material that some add covered was removed by the first
0-writer after the last covering add (over-cut / over-clip); missing material no add ever covered is an omitted add;
invented material was brought by the first add after the last 0-writer - an omitted cut when the point lies inside a cut
leaf of the truth, uncut excess when that add was the cover (the first add), an over-add otherwise. Model surface owned by
an operation that lies farther than 2 tau from the truth without a volume defect is a surface deviation of that operation.
A defect that lies between two nearby surfaces (the truth's and the model's, both within 2e-3 D) is a fit deviation of the
operation that owns the model surface there, collected per operation. Other defects form connected regions
(26-connectivity, per class and culprit); a region is significant above 1e-3 of the truth's volume or 5e-3 D deep. The time of a region is its culprit's operation; for an omission, the earliest candidate of the
path that would have repaired it (covers half of it, gains against the truth) or the stop. The separation point is the
earliest DECISIVE region (significant and at least a tenth of the run's significant damage); first_defect is the
earliest significant one, the dominant defect the largest one.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco_engine.csg_surface import (CompiledTree, boundary_samples, compile_program, compile_tree, finite_bounds,
                                     gradient, surface_distances)
from reco_engine.truth import Truth, sp, truth_from_json

TAUS = (2.5e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2)          # coverage tolerances, x D
ANGLE_DEG = 2.0
GRADE_V1 = dict(version="v1", tiers=[
    # tier, IoU >=, H/D <=, RMS/D <=, coverage both ways >=, at tau x D
    ("A", 0.999, 2e-3, 3e-4, 0.99, 1e-3),
    ("B", 0.99, 1e-2, 2e-3, 0.95, 2e-3),
    ("C", 0.95, 3e-2, 6e-3, 0.85, 5e-3),
    ("D", 0.85, 0.10, 2e-2, 0.60, 1e-2),
])
SIGNIFICANT_VOLUME = 1e-3                             # x V*
SIGNIFICANT_DEPTH = 5e-3                              # x D
DECISIVE_SHARE = 0.1                                  # of the run's significant damage: may be the separation point


# ----------------------------------------------------------------------------------------------- the program

def program_ops(result) -> List[Tuple[str, sp.Primitive]]:
    """(op, sp primitive) of the accepted operations, in order (the left fold of result_to_program.program_model)."""
    from reco_engine.result_to_program import primitive_to_sp
    return [(o.op, primitive_to_sp(o.primitive)) for o in result.operations if o.accepted]


def load_program(run_dir) -> Tuple[Optional[object], Optional[object]]:
    """(ReconstructionResult, PathLog) of a run directory: program.json, else rebuilt from path.jsonl."""
    from reco_engine.result_schema import ReconstructionResult
    from reco_engine.run_path import load_path, program_from_path
    run_dir = Path(run_dir)
    log = load_path(run_dir / "path.jsonl") if (run_dir / "path.jsonl").is_file() else None
    if (run_dir / "program.json").is_file():
        return ReconstructionResult.from_dict(json.loads((run_dir / "program.json").read_text())), log
    if log is not None and log.final:
        return program_from_path(log), log
    return None, log


# ------------------------------------------------------------------------------------------------------ volume

@dataclass
class VolumeHistory:
    lo: np.ndarray
    cell: np.ndarray                 # cell size (3)
    shape: Tuple[int, int, int]
    points: np.ndarray               # (N, 3), one per cell, C order over shape
    truth: np.ndarray                # (N,) bool
    truth_sdf: np.ndarray            # (N,)
    ops: List[str]
    writes: List[np.ndarray]         # per op: packed bits of the points it writes
    final: np.ndarray                # (N,) bool
    trajectory: List[dict]           # per prefix k (1..n): iou, missing, invented (shares of V*)
    changed: List[int]               # per op: points whose state it changed when applied

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.cell))

    def write_mask(self, k: int, idx: Optional[np.ndarray] = None) -> np.ndarray:
        m = np.unpackbits(self.writes[k], count=len(self.points)).astype(bool)
        return m if idx is None else m[idx]

    def state_after(self, n_ops: int, idx: np.ndarray) -> np.ndarray:
        """The model's inside/outside at points idx after the first n_ops operations."""
        state = np.zeros(len(idx), bool)
        for k in range(min(n_ops, len(self.ops))):
            w = self.write_mask(k, idx)
            state[w] = self.ops[k] == "add"
        return state


def joint_box(truth: Truth, ops) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = truth.lo.copy(), truth.hi.copy()
    for op, prim in ops:
        if op != "add":
            continue
        try:
            a, b = finite_bounds(compile_tree(prim).leaves, roles=("add",))
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        except ValueError:
            pass
    pad = 0.02 * float(np.linalg.norm(hi - lo))
    return lo - pad, hi + pad


def volume_history(truth: Truth, ops, cells: int = 128, seed: int = 0) -> VolumeHistory:
    lo, hi = joint_box(truth, ops)
    span = hi - lo
    shape = tuple(int(max(8, round(cells * s / span.max()))) for s in span)
    cell = span / np.array(shape)
    rng = np.random.default_rng(seed)
    idx = np.indices(shape).reshape(3, -1).T
    P = lo + (idx + rng.random(idx.shape)) * cell
    ts = truth.compiled.sdf(P)
    tin = ts <= 0
    state = np.zeros(len(P), bool)
    writes, traj, changed = [], [], []
    nt = max(int(tin.sum()), 1)
    for op, prim in ops:
        inside = compile_tree(prim).sdf(P) <= 0
        w = inside if op in ("add", "cut") else ~inside
        before = state.copy()
        state[w] = op == "add"
        changed.append(int((state != before).sum()))
        writes.append(np.packbits(w))
        inter, union = int((state & tin).sum()), int((state | tin).sum())
        traj.append(dict(iou=inter / max(union, 1), missing=int((tin & ~state).sum()) / nt,
                         invented=int((state & ~tin).sum()) / nt))
    return VolumeHistory(lo, cell, shape, P, tin, ts, [o for o, _ in ops], writes, state, traj, changed)


def volume_metrics(h: VolumeHistory) -> dict:
    t, m = h.truth, h.final
    inter, union = int((t & m).sum()), int((t | m).sum())
    nt, cv = max(int(t.sum()), 1), h.cell_volume
    iou = inter / max(union, 1)
    return dict(iou=iou, iou_se=math.sqrt(max(iou * (1 - iou), 1e-12) / max(union, 1)),
                missing=int((t & ~m).sum()) / nt, invented=int((m & ~t).sum()) / nt, containment=inter / nt,
                volume_truth=nt * cv, volume_model=int(m.sum()) * cv, volume_se=math.sqrt(max(int(m.sum()), 1)) * cv,
                cells=list(h.shape), cell=h.cell.tolist())


# ----------------------------------------------------------------------------------------------------- surface

def _stats(x: np.ndarray, D: float) -> dict:
    if not len(x):
        return {}
    a = np.abs(x)
    return dict(mean=float(a.mean() / D), signed_mean=float(x.mean() / D), std=float(a.std() / D),
                rms=float(np.sqrt((x ** 2).mean()) / D), median=float(np.median(a) / D),
                p95=float(np.percentile(a, 95) / D), p99=float(np.percentile(a, 99) / D), max=float(a.max() / D), n=len(x))


def surface_metrics(truth: Truth, model: CompiledTree, n: int = 200_000, seed: int = 0) -> dict:
    D = truth.diag
    lo, hi = finite_bounds(model.leaves, roles=("add", "clip"))
    clip = (np.minimum(lo, truth.lo) - 0.05 * D, np.maximum(hi, truth.hi) + 0.05 * D)
    sT = boundary_samples(truth.compiled, n_target=n, seed=seed, clip=clip)
    sM = boundary_samples(model, n_target=n, seed=seed + 1, clip=clip)
    kw = dict(open_tol=1e-7 * D, near=0.05 * D)                  # Newton where the bracket is open and the distance matters
    dT = surface_distances(sT.points, sT.normals, model, sM, **kw)
    dM = surface_distances(sM.points, sM.normals, truth.compiled, sT, **kw)
    out = dict(truth_to_model=_stats(dT["signed"], D), model_to_truth=_stats(dM["signed"], D),
               spacing=float(max(sT.spacing, sM.spacing) / D), area_truth=sT.area, area_model=sM.area,
               unresolved=dict(truth=float(np.mean(dT["hi"] - dT["lo"] > 0.5 * 1e-3 * D)),
                               model=float(np.mean(dM["hi"] - dM["lo"] > 0.5 * 1e-3 * D))))
    out["hausdorff"] = max(out["truth_to_model"].get("max", 0.0), out["model_to_truth"].get("max", 0.0))
    out["rms"] = math.sqrt(0.5 * (out["truth_to_model"].get("rms", 0.0) ** 2 + out["model_to_truth"].get("rms", 0.0) ** 2))
    cov = {}
    for tau in TAUS:
        rec = (dT["hi"] <= tau * D) & (dT["angle_deg"] <= ANGLE_DEG)
        pre = (dM["hi"] <= tau * D) & (dM["angle_deg"] <= ANGLE_DEG)
        cov[f"{tau:g}"] = dict(recall=float(rec.mean()) if len(rec) else 0.0, precision=float(pre.mean()) if len(pre) else 0.0)
    out["coverage"] = cov
    # per operation: the model surface it owns that deviates (> 2 tau at tau = 1e-3 D) - area share of the model
    far = dM["hi"] > 2e-3 * D
    by_op: Dict[int, dict] = {}
    for k in np.unique(sM.owner):
        mine = sM.owner == k
        by_op[int(k)] = dict(area_share=float(mine.mean()), deviating_share=float((mine & far).mean()),
                             max=float(dM["hi"][mine].max() / D) if mine.any() else 0.0)
    out["by_op"] = by_op
    out["_samples"] = dict(truth=sT, model=sM, dT=dT, dM=dM)
    return out


def prefix_surface_coverage(truth: Truth, ops, n: int = 20_000, tau: float = 1e-3, seed: int = 3) -> dict:
    """Which truth surface points each prefix covers (|s_k| <= tau D, the field locally a distance, normals within 2 deg):
    the share per prefix, and per point the first prefix that covers it and the last one before it is lost for good."""
    D = truth.diag
    sT = boundary_samples(truth.compiled, n_target=n, seed=seed)
    P, N = sT.points, sT.normals
    h = 1e-4 * D
    covered = np.zeros((len(ops), len(P)), bool)
    for k in range(len(ops)):
        tree = compile_program(ops[:k + 1])
        s = tree.sdf(P)
        near = np.abs(s) <= tau * D
        if near.any():
            g = gradient(tree.sdf, P[near], h)
            gn = np.linalg.norm(g, axis=1)
            ang = np.degrees(np.arccos(np.clip((g * N[near]).sum(1) / np.maximum(gn, 1e-300), -1, 1)))
            ok = (np.abs(gn - 1.0) <= 0.01) & (ang <= ANGLE_DEG)
            covered[k, np.flatnonzero(near)[ok]] = True
    share = covered.mean(axis=1).tolist() if len(ops) else []
    first = np.where(covered.any(0), covered.argmax(0), -1)
    final = covered[-1] if len(ops) else np.zeros(len(P), bool)
    last = np.full(len(P), -1)
    lost = ~final & covered.any(0)
    if lost.any():
        rev = covered[::-1, lost]
        last[lost] = len(ops) - 1 - rev.argmax(0)                          # the last prefix that still covered it
    return dict(share=share, first_covered=first, last_covered=last, final=final, points=P, tau=tau)


# --------------------------------------------------------------------------------------------- leaf matching

def _leaf_descriptor(leaf) -> Optional[dict]:
    p = leaf.prim
    if isinstance(p, sp.HalfSpace):
        return None
    a, b = p.bounds()
    centre = leaf.R @ (0.5 * (np.asarray(a) + np.asarray(b))) + leaf.t
    kind = type(p).__name__
    if isinstance(p, sp.Box):
        order = np.argsort(p.half)[::-1]
        dims = np.asarray(p.half, float)[order]
        axes = leaf.R[:, order]
    elif isinstance(p, sp.Cylinder):
        dims, axes = np.array([p.radius, p.height, p.inner_radius]), leaf.R[:, 2:3]
    elif isinstance(p, sp.Cone):                                   # canonical: big end first, axis big -> small end
        big_first = p.radius1 >= p.radius2
        dims = np.array([max(p.radius1, p.radius2), min(p.radius1, p.radius2), p.height])
        axes = leaf.R[:, 2:3] if big_first else -leaf.R[:, 2:3]
    elif isinstance(p, sp.Sphere):
        dims, axes = np.array([p.radius, p.inner_radius]), np.zeros((3, 0))
    elif isinstance(p, sp.Torus):
        dims, axes = np.array([p.major, p.minor, p.inner_minor]), leaf.R[:, 2:3]
    else:
        return None
    sweep = float(getattr(p, "u", sp.FULL)[1] - getattr(p, "u", sp.FULL)[0])
    return dict(kind=kind, centre=centre, dims=dims, axes=axes, role="cut" if leaf.role == "cut" else "add",
                owner=leaf.owner, sweep=sweep, signed=isinstance(p, sp.Cone))


def _pair_cost(a: dict, b: dict, D: float) -> float:
    if a["kind"] != b["kind"]:
        return np.inf
    if len(a["dims"]) != len(b["dims"]):
        return np.inf
    c = float(np.linalg.norm(a["centre"] - b["centre"])) / (0.02 * D)
    d = float(np.max(np.abs(a["dims"] - b["dims"]))) / (0.02 * D)
    r = 0.0
    if a["axes"].shape[1]:
        angs = []
        for i in range(a["axes"].shape[1]):
            cosv = float(a["axes"][:, i] @ b["axes"][:, i])
            angs.append(math.degrees(math.acos(min(1.0, cosv if a["signed"] else abs(cosv)))))
        r = max(angs) / 5.0
    s = abs(a["sweep"] - b["sweep"]) / 0.05
    return max(c, d, r, s)


def match_leaves(truth_tree: CompiledTree, model_tree: CompiledTree, D: float, visible_truth=None, visible_model=None) -> dict:
    from scipy.optimize import linear_sum_assignment
    T = [(i, _leaf_descriptor(l)) for i, l in enumerate(truth_tree.leaves)]
    M = [(i, _leaf_descriptor(l)) for i, l in enumerate(model_tree.leaves)]
    T = [(i, d) for i, d in T if d is not None and (visible_truth is None or i in visible_truth)]
    M = [(i, d) for i, d in M if d is not None and (visible_model is None or i in visible_model)]
    if not T or not M:
        return dict(precision=0.0, recall=0.0, f1=0.0, matched=0, truth_leaves=len(T), model_leaves=len(M),
                    first_unmatched_op=min((d["owner"] for _, d in M), default=None), pairs=[])
    C = np.array([[_pair_cost(a, b, D) for _, b in M] for _, a in T])
    big = 1e6
    rows, cols = linear_sum_assignment(np.where(np.isfinite(C), C, big))
    pairs = [(T[r][0], M[c][0], float(C[r, c])) for r, c in zip(rows, cols) if C[r, c] <= 1.0]
    matched_model = {m for _, m, _ in pairs}
    tp = len(pairs)
    roles = [T[[i for i, _ in T].index(t)][1]["role"] == M[[i for i, _ in M].index(m)][1]["role"] for t, m, _ in pairs]
    unmatched_ops = sorted(d["owner"] for i, d in M if i not in matched_model)
    return dict(precision=tp / len(M), recall=tp / len(T), f1=2 * tp / (len(M) + len(T)), matched=tp,
                truth_leaves=len(T), model_leaves=len(M), role_agreement=float(np.mean(roles)) if roles else None,
                first_unmatched_op=unmatched_ops[0] if unmatched_ops else None,
                pairs=[dict(truth=t, model=m, cost=c) for t, m, c in pairs])


# --------------------------------------------------------------------------------------------------------- grade

def hard_conditions(record: Optional[dict], truth_genus: Optional[int], n_ops: int,
                    mesh_genus_fallback: Optional[int] = None) -> dict:
    """A program, one valid solid (FreeCAD), the truth's genus (OCCT's model genus, else the built mesh's)."""
    model = (record or {}).get("model") or {}
    exact = ((record or {}).get("comparison") or {}).get("exact") or {}
    genus = exact.get("model_genus")
    if genus is None:
        genus = mesh_genus_fallback
    out = dict(program=n_ops > 0, one_valid_solid=model.get("one_valid_solid"), model_genus=genus, truth_genus=truth_genus)
    failed = []
    if not out["program"]:
        failed.append("no program")
    if out["one_valid_solid"] is False:
        failed.append(f"build: valid={model.get('valid')} solids={model.get('solids')}")
    if genus is not None and truth_genus is not None and int(round(float(genus))) != int(truth_genus):
        failed.append(f"genus {genus} != {truth_genus}")
    out["failed"] = failed
    return out


def grade(vector: dict, hard: dict, spec: dict = GRADE_V1) -> dict:
    if hard.get("failed"):
        return dict(tier="E", failed=list(hard["failed"]), version=spec["version"], per_tier={})
    v, s = vector.get("volume", {}), vector.get("surface", {})
    per_tier, tier, fail_first, limiting = {}, "E", None, []
    for name, iou, hd, rms, covmin, tau in spec["tiers"]:
        cov = (s.get("coverage") or {}).get(f"{tau:g}", {})
        measured = dict(iou=(v.get("iou", 0.0), ">=", iou), hausdorff=(s.get("hausdorff", np.inf), "<=", hd),
                        rms=(s.get("rms", np.inf), "<=", rms), recall=(cov.get("recall", 0.0), ">=", covmin),
                        precision=(cov.get("precision", 0.0), ">=", covmin))
        checks = {k: (val >= thr if op == ">=" else val <= thr) for k, (val, op, thr) in measured.items()}
        per_tier[name] = checks
        if all(checks.values()):
            tier = name
            break
        if fail_first is None:
            fail_first = [k for k, ok in checks.items() if not ok]
        # what kept the run out of this tier (the last one tried before the achieved tier is the next better one)
        limiting = [dict(tier=name, check=k, value=float(val), op=op, threshold=thr)
                    for k, (val, op, thr) in measured.items() if not checks[k]]
    return dict(tier=tier, failed=[] if tier == "A" else [f"tier A: {', '.join(fail_first or [])}"], version=spec["version"],
                per_tier=per_tier, limiting=[] if tier == "A" else limiting)


def regrade(rating: dict, spec: dict = GRADE_V1) -> dict:
    return grade(rating["vector"], rating["hard"], spec)


# --------------------------------------------------------------------------------------------------- attribution

def _regions(mask_grid: np.ndarray):
    from scipy import ndimage
    lab, n = ndimage.label(mask_grid, structure=np.ones((3, 3, 3), bool))
    return lab, n


def attribute(truth: Truth, hist: VolumeHistory, ops, log=None, surface: Optional[dict] = None,
              prefix_cov: Optional[dict] = None, model: Optional[CompiledTree] = None, band_rel: float = 2e-3) -> dict:
    D, N = truth.diag, len(hist.points)
    t, m = hist.truth, hist.final
    n_ops = len(hist.ops)
    nt = max(int(t.sum()), 1)
    culprit = np.full(N, -1, np.int64)
    klass = np.full(N, "", object)
    missing = np.flatnonzero(t & ~m)
    invented = np.flatnonzero(m & ~t)
    # missing: the last add covering the point, then the first 0-writer after it
    if len(missing):
        last_add = np.full(len(missing), -1)
        for k, op in enumerate(hist.ops):
            if op == "add":
                last_add[hist.write_mask(k, missing)] = k
        first_zero = np.full(len(missing), -1)
        for k, op in enumerate(hist.ops):
            if op == "add":
                continue
            w = hist.write_mask(k, missing) & (k > last_add) & (first_zero < 0) & (last_add >= 0)
            first_zero[w] = k
        culprit[missing] = np.where(last_add >= 0, first_zero, -1)
        kinds = np.array(hist.ops + [""], object)
        klass[missing] = np.where(last_add < 0, "omitted add",
                                  np.where(kinds[np.maximum(first_zero, 0)] == "intersect", "over-clip", "over-cut"))
    if len(invented):
        last_zero = np.full(len(invented), -1)
        for k, op in enumerate(hist.ops):
            if op != "add":
                last_zero[hist.write_mask(k, invented)] = k
        first_add = np.full(len(invented), -1)
        for k, op in enumerate(hist.ops):
            if op == "add":
                w = hist.write_mask(k, invented) & (k > last_zero) & (first_add < 0)
                first_add[w] = k
        culprit[invented] = first_add
        # the truth's deciding leaf: inside a cut leaf of the truth -> an omitted cut
        _, active = truth.compiled.evaluate(hist.points[invented])
        roles = np.array([l.role for l in truth.compiled.leaves], object)[active]
        in_cut = roles == "cut"
        klass[invented] = np.where(in_cut, "omitted cut", np.where(first_add == 0, "uncut excess", "over-add"))
    defect = np.zeros(N, bool)
    defect[missing] = True
    defect[invented] = True
    # a fit deviation: the defect lies between two nearby surfaces - the truth's and the model's (both within band x D):
    # the operation that owns the model surface there placed or sized its primitive slightly wrong. Such a skin falls
    # apart into many one-cell pieces on the grid, so it is collected per operation, not by connectivity.
    if model is not None and defect.any():
        d_idx = np.flatnonzero(defect)
        ms, mact = model.evaluate(hist.points[d_idx])
        near = (np.abs(ms) <= band_rel * D) & (np.abs(hist.truth_sdf[d_idx]) <= band_rel * D)
        klass[d_idx[near]] = "fit deviation"
        culprit[d_idx[near]] = model.owners()[mact[near]]
    # regions per (class, culprit)
    cv = hist.cell_volume
    regions = []
    keys = sorted({(klass[i], int(culprit[i])) for i in np.flatnonzero(defect)})
    for cls, cul in keys:
        mask = defect & (klass == cls) & (culprit == cul)
        if cls == "fit deviation":
            idx = np.flatnonzero(mask)
            depth = float(np.abs(hist.truth_sdf[idx]).max())
            regions.append(dict(cls=cls, culprit=cul, volume=len(idx) * cv, volume_share=len(idx) / nt, depth=depth / D,
                                centre=hist.points[idx].mean(0).tolist(), lo=hist.points[idx].min(0).tolist(),
                                hi=hist.points[idx].max(0).tolist(), cells=len(idx),
                                significant=bool(len(idx) / nt >= SIGNIFICANT_VOLUME), _idx=idx))
            continue
        lab, n = _regions(mask.reshape(hist.shape))
        lab = lab.reshape(-1)
        for r in range(1, n + 1):
            idx = np.flatnonzero(lab == r)
            vol = len(idx) * cv
            depth = float(np.abs(hist.truth_sdf[idx]).max())
            regions.append(dict(cls=cls, culprit=cul, volume=vol, volume_share=len(idx) / nt, depth=depth / D,
                                centre=hist.points[idx].mean(0).tolist(), lo=hist.points[idx].min(0).tolist(),
                                hi=hist.points[idx].max(0).tolist(), cells=len(idx),
                                significant=bool(len(idx) / nt >= SIGNIFICANT_VOLUME or depth >= SIGNIFICANT_DEPTH * D),
                                _idx=idx))
    # surface deviations of an operation without a volume region of its own
    if surface is not None:
        vol_culprits = {r["culprit"] for r in regions if r["significant"]}
        for k, d in (surface.get("by_op") or {}).items():
            if d["deviating_share"] >= 1e-2 and int(k) not in vol_culprits:
                regions.append(dict(cls="surface deviation", culprit=int(k), volume=0.0, volume_share=0.0,
                                    depth=d["max"], significant=True, area_share=d["deviating_share"], _idx=np.zeros(0, int)))
    # when did it happen, and why
    final_events = log.final if log is not None else []
    for r in regions:
        if r["significant"]:
            r["time"], r["cause"], r["event"] = _time_and_cause(r, hist, ops, log, final_events, n_ops)
        else:
            r["time"], r["cause"], r["event"] = None, None, None
    sig = [r for r in regions if r["significant"]]
    damage = lambda r: r["volume_share"] + r.get("area_share", 0.0)  # noqa: E731
    total = sum(damage(r) for r in sig)
    # the separation point is the earliest DECISIVE defect (at least DECISIVE_SHARE of the run's significant damage):
    # a small fit deviation of the base at step 0 must not hide the omitted cut that failed the run later
    for r in regions:
        r["decisive"] = bool(r["significant"] and damage(r) >= DECISIVE_SHARE * total)
    dec = [r for r in sig if r["decisive"]]
    sep = min(dec, key=lambda r: (r["time"], -damage(r)), default=None)
    first = min(sig, key=lambda r: (r["time"], -damage(r)), default=None)
    dom = max(sig, key=damage, default=None)
    # per operation
    by_op = []
    for k in range(n_ops):
        wrong = hist.write_mask(k) & ((t & (hist.ops[k] != "add")) | (~t & (hist.ops[k] == "add")))
        persisting = int((culprit == k).sum())
        by_op.append(dict(op_index=k, op=hist.ops[k], changed=hist.changed[k] * cv / (nt * cv),
                          wrong_writes=int(wrong.sum()) / nt, persisting=persisting / nt,
                          transient=max(int(wrong.sum()) - persisting, 0) / nt, redundant=hist.changed[k] == 0,
                          delta_iou=hist.trajectory[k]["iou"] - (hist.trajectory[k - 1]["iou"] if k else 0.0),
                          net_gain=(final_events[k].get("net_gain") if k < len(final_events) else None)))
    clean = lambda r: None if r is None else {k: v for k, v in r.items() if not k.startswith("_")}  # noqa: E731
    return dict(regions=[clean(r) for r in sorted(regions, key=lambda r: -r["volume_share"])[:200]],
                separation=clean(sep), first_defect=clean(first), dominant=clean(dom), by_op=by_op,
                classes={c: sum(r["volume_share"] for r in regions if r["cls"] == c) for c in {r["cls"] for r in regions}})


def _time_and_cause(region, hist, ops, log, final_events, n_ops):
    cls, cul = region["cls"], region["culprit"]
    if cul >= 0 and cls not in ("omitted add", "omitted cut"):
        ev = final_events[cul] if cul < len(final_events) else {}
        step = hist.trajectory[cul]
        prev = hist.trajectory[cul - 1]["iou"] if cul else 0.0
        gain = ev.get("net_gain")
        harmed = step["iou"] - prev < -1e-3
        cause = "judge accepted harm" if (gain is not None and gain > 0 and harmed) else \
            ("fit precision of an accepted operation" if cls == "fit deviation" else
             "persisting damage of an accepted operation" if cls != "uncut excess" else "the cover's excess was never cut")
        return cul, cause, _event(ev)
    # an omission: the earliest candidate that would have repaired it
    idx = region.get("_idx")
    if log is not None and idx is not None and len(idx):
        want_add = cls == "omitted add"
        best = None
        for e in sorted(log.candidates + [dict(r, kind="withdrawn") for r in log.removed if "primitive" in r],
                        key=lambda e: e.get("index", 0)):
            prim = e.get("primitive")
            op = e.get("op") or ("add" if e.get("kind") == "alternative" else None)
            if prim is None or op is None or (op == "add") != want_add:
                continue
            try:
                from reco_engine.result_to_program import primitive_to_sp
                from reco_engine.result_schema import ReconstructionResult
                p = ReconstructionResult.from_dict(dict(input_path="", operations=[dict(op=op, primitive=dict(
                    prim, parts=prim.get("parts") or []))])).operations[0].primitive
                tree = compile_tree(primitive_to_sp(p))
            except Exception:
                continue
            inside = tree.sdf(hist.points[idx]) <= 0
            if inside.mean() < 0.5:
                continue
            n_at = int(e.get("n_ops", n_ops))
            # gain against the truth where the candidate writes (only its own box is looked at)
            lo, hi = finite_bounds(tree.leaves, roles=("add",))             # the candidate's own box
            near = np.flatnonzero(np.all((hist.points >= lo) & (hist.points <= hi), axis=1))
            if not len(near):
                continue
            w = tree.sdf(hist.points[near]) <= 0
            st = hist.state_after(n_at, near)
            if want_add:
                gain = int((w & ~st & hist.truth[near]).sum()) - int((w & ~st & ~hist.truth[near]).sum())
            else:
                gain = int((w & st & ~hist.truth[near]).sum()) - int((w & st & hist.truth[near]).sum())
            if gain > 0:
                best = e
                break
        if best is not None:
            kind = best.get("kind")
            cause = "withdrawn by confirmation" if kind == "withdrawn" else \
                ("first-add alternative not taken" if kind == "alternative" else "judge rejected a good candidate")
            return int(best.get("n_ops", n_ops)), cause, _event(best)
    stopped = (log.stopped if log is not None else None) or ""
    cause = "budget or time" if any(w in stopped for w in ("budget", "time", "max_ops", "operations")) else "never proposed"
    return n_ops, cause, dict(kind="stop", stopped=stopped[:200])


def _event(e: dict) -> dict:
    keep = ("index", "kind", "phase", "branch", "n_ops", "op", "op_index", "source", "reason", "gain", "net_gain", "score",
            "piece_id", "iteration", "primitive")
    return {k: e[k] for k in keep if k in e}


# ------------------------------------------------------------------------------------------------------ rating

def rate(truth: Truth, result, log=None, record: Optional[dict] = None, cells: int = 128, n_surface: int = 200_000,
         n_prefix: int = 20_000, seed: int = 0, model_mesh=None) -> dict:
    from reco_engine.truth import load_mesh, mesh_genus
    ops = program_ops(result) if result is not None else []
    truth_genus = truth.meta.get("genus")
    if truth_genus is None and truth.input_path:
        try:
            truth_genus = mesh_genus(*load_mesh(truth.input_path))
        except Exception:
            truth_genus = None
    fallback = None
    if model_mesh is not None and Path(model_mesh).is_file():
        try:
            fallback = mesh_genus(*load_mesh(model_mesh))
        except Exception:
            fallback = None
    hard = hard_conditions(record, truth_genus, len(ops), fallback)
    out = dict(truth=dict(name=truth.name, source=truth.source, diag=truth.diag, meta=truth.meta, genus=truth_genus),
               n_ops=len(ops), hard=hard)
    if not ops:
        out["vector"] = dict(volume=dict(iou=0.0, missing=1.0, invented=0.0, containment=0.0))
        out["grade"] = grade(out["vector"], hard)
        out["attribution"] = dict(separation=dict(cls="omitted add", culprit=-1, time=0, cause="no program",
                                                  volume_share=1.0, significant=True), regions=[], by_op=[])
        return out
    model = compile_program(ops)
    hist = volume_history(truth, ops, cells=cells, seed=seed)
    vol = volume_metrics(hist)
    surf = surface_metrics(truth, model, n=n_surface, seed=seed)
    samples = surf.pop("_samples")
    pc = prefix_surface_coverage(truth, ops, n=n_prefix, seed=seed + 7)
    vis_t = {int(i) for i, c in zip(*np.unique(samples["truth"].leaf, return_counts=True)) if c >= 1e-3 * len(samples["truth"].leaf)}
    vis_m = {int(i) for i, c in zip(*np.unique(samples["model"].leaf, return_counts=True)) if c >= 1e-3 * len(samples["model"].leaf)}
    tree = match_leaves(truth.compiled, model, truth.diag, vis_t, vis_m)
    vector = dict(volume=vol, surface=surf, trajectory=hist.trajectory, prefix_coverage=pc["share"], tree=tree,
                  ops=len(ops), ops_per_truth_leaf=len(ops) / max(len(truth.compiled.leaves), 1))
    # measurement validity against the built model (run_part's record)
    validity = {}
    model_rec = (record or {}).get("model") or {}
    if model_rec.get("volume") is not None:
        dv = abs(vol["volume_model"] - float(model_rec["volume"]))
        validity["volume"] = dict(analytic=vol["volume_model"], built=float(model_rec["volume"]),
                                  ok=bool(dv <= 3 * vol["volume_se"] + 1e-3 * vol["volume_truth"]))
    ex = ((record or {}).get("comparison") or {}).get("exact") or {}
    if ex.get("iou") is not None:
        validity["iou"] = dict(analytic=vol["iou"], exact=float(ex["iou"]),
                               ok=bool(abs(vol["iou"] - float(ex["iou"])) <= 3 * vol["iou_se"] + 2e-3))
    vector["validity"] = validity
    out["unmeasured"] = any(not v["ok"] for v in validity.values())
    out["vector"] = vector
    out["grade"] = grade(vector, hard)
    out["attribution"] = attribute(truth, hist, ops, log, surf, pc, model=model)
    return out


def rate_run(run_dir, truth_json, **kw) -> dict:
    run_dir = Path(run_dir)
    truth = truth_from_json(truth_json)
    if truth.input_path is None:
        rec = json.loads((run_dir / "record.json").read_text()) if (run_dir / "record.json").is_file() else {}
        truth.input_path = (rec.get("comparison") or {}).get("input_stl") or None
    result, log = load_program(run_dir)
    record = json.loads((run_dir / "record.json").read_text()) if (run_dir / "record.json").is_file() else None
    rating = rate(truth, result, log, record, model_mesh=run_dir / "model.stl", **kw)
    (run_dir / "rating.json").write_text(json.dumps(rating, indent=1, default=_json_default))
    return rating


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
