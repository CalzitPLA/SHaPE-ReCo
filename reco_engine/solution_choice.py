#!/usr/bin/env python3
"""
Other solutions before the fallback (the user's rule 2026-09-29: "before going into fall[back] after member phase
check other solution, maybe classic cut-add").

The loop's order is: first-add sub-loop (members, hollow body) -> add/cut loop on the leftovers -> voxel-cube fallback
or feature phase. When a sub-loop gave the run's first solution and that solution is not good enough - or the member
phase was asked for and found nothing usable - the CLASSIC solution is computed on a fresh copy of the state before
any fallback: no sub-loop, the plain first add (residual_loop._first_add: the graded bounding primitive) and the
add/cut loop with cuts first (D-003). Both solutions are judged by the same measure and the better one is kept.

The measure: the SAMPLED EXACT IoU on the input's mesh (`sampled_measure`), by the judge the member phase accepts its
operations with (ibeam.ExactJudge: generalized winding number of the input, the program's own CSG signed distance).
Not the voxel residual |R| on the run's grid: on a thin-walled part the grid is blind - all of the frame review's
runs ended "residual within tolerance" with |R| = 0, r08 at an exact IoU of 0.025 (2026-09-29,
~/shape_reco_results/frame_review_fix1). Not OCCT (solution_stats / _exact_grade): it needs a FreeCAD build and an
exact difference per solution; the kept solution gets that grade at the end of the run, as before.

When the classic solution is tried (ResidualConfig.solution_alternatives):
  "never"   not at all;
  "always"  whenever a sub-loop gave the first solution or the member phase found nothing;
  "auto"    (default) only when
            - the member phase was asked for (a type hint, or members detected that pass the analytic gate) and no
              operation of it was accepted, or
            - the voxel residual after the add/cut loop is above the tolerance (the fallback would run), or
            - the exact account of the accepted operations (`open_share`: 1 - the summed exact gains / the input's
              volume, free - the judge measured every gain already) is above solution_exact_tolerance.
            A run that ends within both tolerances costs nothing more (the frame).

Only on the legacy path: residual_steps records the context (`state.solution_ctx`) there; after a starting shape
(start_shape != "legacy") or a user's first guess there is no alternative (the guard in residual_loop._residual_phase).

The steps: the classic solution's own steps with the loop's kinds and `solution="classic"` in their info (their
result is the classic program so far), then ONE step of kind "solution" that states both measures, which solution
was kept and why (info: solutions, kept, measure, reasons, seconds).
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from reco_engine.result_schema import Primitive, ReconstructionResult
from reco_engine.result_to_program import primitive_to_sp, program_model

MEASURE = "sampled exact IoU (the input's mesh, ibeam.ExactJudge)"
MODES = ("auto", "always", "never")


# ------------------------------------------------------------------------------------------------ the measure

def mesh_volume(field) -> float:
    tri = np.asarray(field.vertices, float)[np.asarray(field.faces)]
    return abs(float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0)


def _samples(prim: Primitive, n: int, seed: int, lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, float]:
    """(points, volume): ~n uniform points inside the primitive and its volume. Box, cylinder, sphere: exact
    (ibeam.primitive_samples / primitive_volume); any other kind (composite, cone, torus, sector, half-space): by
    rejection in its bounding box clipped to [lo, hi], the volume from the accepted share."""
    from reco_engine import ibeam as ib
    if prim.kind in ("box", "cylinder", "sphere") and not prim.u and not prim.v:
        return ib.primitive_samples(prim, n, seed), ib.primitive_volume(prim)
    sp = primitive_to_sp(prim)
    try:
        b_lo, b_hi = (np.asarray(b, float) for b in sp.bounds())
        b_lo, b_hi = np.maximum(np.where(np.isfinite(b_lo), b_lo, lo), lo), np.minimum(np.where(np.isfinite(b_hi), b_hi, hi), hi)
    except Exception:
        b_lo, b_hi = lo, hi
    if np.any(b_hi <= b_lo):
        return np.zeros((0, 3)), 0.0
    pts = np.random.default_rng(seed).uniform(b_lo, b_hi, (4 * n, 3))
    keep = np.asarray(sp.sdf(pts)) <= 0
    return pts[keep], float(np.prod(b_hi - b_lo)) * float(keep.mean())


def sampled_measure(result: ReconstructionResult, contains: Callable, v_in: float, bounds, n: int = 1500) -> Dict:
    """The exact volumes of a program against the input, sampled: every accepted ADD is sampled uniformly (n points),
    a point counts for the first add it lies in (no volume twice) and only when the whole program keeps it (cuts,
    intersections); inside the input by `contains` (ExactJudge.contains). common = V(model & input), invented =
    V(model - input), missing = V_in - common, IoU = common / (V_in + invented) - the definition of the exact judge
    (scripts/s6_frame_eval.py). Every point of the model lies in one of its adds, so nothing is left out.
    dict(iou, common, invented, missing, missing_share, invented_share, open_share, model_volume, points, operations)."""
    ops = [o for o in result.operations if o.accepted]
    out = dict(iou=0.0, common=0.0, invented=0.0, missing=float(v_in), missing_share=1.0, invented_share=0.0, open_share=1.0,
               model_volume=0.0, points=0, operations=len(ops), adds=sum(o.op == "add" for o in ops),
               cuts=sum(o.op != "add" for o in ops))
    model = program_model(result) if ops else None
    if model is None or v_in <= 0:
        return out
    lo_in, hi_in = (np.asarray(b, float) for b in bounds)
    size = hi_in - lo_in
    lo, hi = lo_in - size, hi_in + size                           # a model far outside is clipped to 3 x the input
    common = invented = 0.0
    earlier: List = []
    points = 0
    for k, o in enumerate(ops):
        if o.op != "add":
            continue
        pts, vol = _samples(o.primitive, n, k, lo, hi)
        sp = primitive_to_sp(o.primitive)
        if len(pts) and vol > 0:
            first = np.ones(len(pts), bool)
            for e in earlier:
                first &= np.asarray(e.sdf(pts)) > 0
            kept = first & (np.asarray(model.sdf(pts)) <= 0)
            inside = np.zeros(len(pts), bool)
            if kept.any():
                inside[kept] = np.asarray(contains(pts[kept]), bool)
            common += vol * float((kept & inside).sum()) / len(pts)
            invented += vol * float((kept & ~inside).sum()) / len(pts)
            points += len(pts)
        earlier.append(sp)
    common = min(common, float(v_in))
    missing = float(v_in) - common
    out.update(iou=common / (float(v_in) + invented), common=common, invented=invented, missing=missing,
               missing_share=missing / float(v_in), invented_share=invented / float(v_in),
               open_share=(missing + invented) / float(v_in), model_volume=common + invented, points=points)
    return out


def open_share(result: ReconstructionResult, pitch: float, v_in: float) -> Optional[float]:
    """The share of the input's volume the accepted operations leave open, from the gains they were accepted with:
    1 - sum(net_gain) x pitch^3 / V_in. The member and hollow sub-loops' gains are exact (residual_loop._exact_eval:
    the volume in voxel volumes), so this costs nothing; the add/cut loop's gains are voxel counts. A member's own
    cutter (a tube's bore, an rhs's hole) is left out: its add was judged without the cutter's volume
    (`exclude`), so the cutter's gain is no material of the input - counted, the account of r06 (17 rhs members,
    exact IoU 0.9378) stayed below the tolerance (2026-09-29). None without a volume."""
    from reco_engine import ibeam as ib
    if v_in <= 0:
        return None
    paired = (ib.MODE_MEMBER, ib.MODE_IBEAM)
    gained = sum(float(o.net_gain) for o in result.operations if o.accepted
                 and not (o.op == "cut" and any(x.mode in paired for x in o.primitive.sources)))
    return 1.0 - gained * float(pitch) ** 3 / float(v_in)


def better(first: Dict, other: Dict, margin: float) -> bool:
    """The other solution replaces the first only when its IoU is higher by more than `margin` (meant as the sampling's
    noise: a draw within it keeps the first, the solution the run found by its own rules). The variance of the sampled
    IoU at `solution_samples` points per add was not measured (review R-01, 2026-09-30); 0.01 is a setting."""
    return float(other["iou"]) > float(first["iou"]) + float(margin)


# ------------------------------------------------------------------------------------------------ when

def reasons_to_try(cfg, ctx: Optional[Dict], voxel_size: int, n_in: int, exact_open: Optional[float], blind=None) -> List[str]:
    """Why the classic solution is tried now ([]: it is not). `ctx`: what residual_steps recorded about the first
    solution - dict(first="members" | "hollow" | "classic", members=dict(asked, detected, accepted) | None)."""
    mode = getattr(cfg, "solution_alternatives", "auto")
    if mode not in MODES:
        raise ValueError(f"ResidualConfig.solution_alternatives must be one of {MODES}, got {mode!r}")
    if ctx is None or mode == "never":
        return []
    rep = ctx.get("members") or {}
    missed = bool(rep.get("asked")) and not rep.get("accepted")
    if ctx.get("first") == "classic" and not missed:
        return []                                                     # no sub-loop was used or asked for: one solution
    out = []
    if missed:
        out.append(f"the member phase found nothing usable ({rep.get('detected', 0)} member(s) detected, "
                   f"0 operations accepted)")
    if voxel_size > cfg.tolerance * n_in:
        out.append(f"the voxel residual {voxel_size} is above the tolerance ({cfg.tolerance * n_in:.1f})")
    tol = getattr(cfg, "solution_exact_tolerance", None)
    if tol is not None and exact_open is not None and exact_open > tol:
        out.append(f"the exact account leaves {exact_open:.1%} of the input's volume open (tolerance {tol:.1%}"
                   + (f"; the grid is blind, pitch {blind:.1f} x the thinnest wall" if blind is not None and blind > 1.0 else "")
                   + ")")
    if mode == "always" and not out:
        out.append("solution_alternatives is 'always'")
    return out


# ------------------------------------------------------------------------------------------------ the classic solution

def _tagged(gen, tag: str, holder: Dict):
    """The generator's steps with info["solution"] = tag; its "done" step (a first add that found nothing) becomes a
    "rejected" one - the run is not over; its return value goes to holder["value"]."""
    from reco_engine.residual_loop import ResidualStep
    while True:
        try:
            s = next(gen)
        except StopIteration as stop:
            holder["value"] = stop.value
            return
        info = dict(s.info or {})
        info["solution"] = tag
        kind = s.kind
        if kind == "done":
            kind = "rejected"
            info = dict(solution=tag, reason=info.get("stopped"), message=f"{tag} solution: {info.get('stopped')}")
        elif info.get("message"):
            info["message"] = f"[{tag}] {info['message']}"
        yield ResidualStep(kind, s.result, info)


def before_fallback(state, field, result, cfg, registry, purity_min, n_in, name, t_start, stopped: str):
    """The hook of residual_loop._residual_phase, after its add/cut loop and before the fallback / the feature
    phase (a generator of ResidualSteps; returns the `stopped` text of the solution that was kept). When the classic
    solution wins, `state` and `result` are replaced IN PLACE by the classic ones - the caller goes on with them."""
    from reco_engine import ibeam as ib
    from reco_engine import residual_loop as rl
    ctx = getattr(state, "solution_ctx", None)
    v_in = mesh_volume(field) if field is not None and getattr(field, "vertices", None) is not None else 0.0
    blind = getattr(state, "blind", None)
    account = open_share(result, state.pitch, v_in) if ctx is not None and ctx.get("first") != "classic" else None
    reasons = reasons_to_try(cfg, ctx, state.size(), n_in, account, blind)
    if ctx is not None:
        ctx["account"] = account                                      # for the "done" step: what the choice rested on
    if not reasons or v_in <= 0:
        return stopped
    t0 = time.time()
    judge = ib.ExactJudge(field, n=cfg.member_samples)
    n = int(getattr(cfg, "solution_samples", cfg.member_samples))
    first_name = ctx["first"]

    def row(which, res, m, stop, seconds=None):
        return dict(solution=which, iou=m["iou"], missing_share=m["missing_share"], invented_share=m["invented_share"],
                    operations=m["operations"], adds=m["adds"], cuts=m["cuts"], points=m["points"], stopped=stop,
                    voxel_residual=[int(res.residual_voxels_pos), int(res.residual_voxels_neg)], seconds=seconds)

    with rl.timed("solution measure"):
        m_first = sampled_measure(result, judge.contains, v_in, field.bounds, n)
    if first_name == "classic":                                       # the member phase found nothing: the run's
        rows = [row("classic", result, m_first, stopped)]             # solution already is the classic one
        yield rl.ResidualStep("solution", rl._snapshot(result), dict(
            solutions=rows, kept="classic", measure=MEASURE, reasons=reasons, tried=False, seconds=time.time() - t0,
            message=f"solution: classic (first add + add/cut loop), the only one - {reasons[0]}; "
                    f"exact IoU {m_first['iou']:.4f} (sampled), {m_first['operations']} operation(s)"))
        return stopped

    alt_state = rl._State(state.t_in, state.spec)
    alt_state.mesh, alt_state.blind, alt_state.trial = getattr(state, "mesh", None), blind, True
    alt = ReconstructionResult(input_path=result.input_path)
    alt.grid_pitch_over_tau = getattr(result, "grid_pitch_over_tau", None)
    cfg_alt = replace(cfg, cuts_first=True, grade_each_step=False)   # "classic cut-add": cuts first (D-003)
    budget = int(ctx.get("budget") or 12)
    holder: Dict = {}
    yield from _tagged(rl._first_add(alt_state, field, alt, cfg_alt, registry, state.t_in, state.spec, n_in, name, t_start,
                                     ctx.get("type_result")), "classic", holder)
    alt_stopped = "no first add"
    if any(o.accepted for o in alt.operations):
        yield from _tagged(rl._residual_phase(alt_state, field, alt, cfg_alt, registry, purity_min, budget, n_in, name,
                                              t_start), "classic", holder)
        alt_stopped = holder.get("value") or ""
    pos, neg = alt_state.residual()
    alt.residual_voxels_pos, alt.residual_voxels_neg = int(pos.sum()), int(neg.sum())
    with rl.timed("solution measure"):
        m_alt = sampled_measure(alt, judge.contains, v_in, field.bounds, n)
    margin = float(getattr(cfg, "solution_margin", 0.01))
    swap = better(m_first, m_alt, margin)
    rows = [row(first_name, result, m_first, stopped), row("classic", alt, m_alt, alt_stopped, time.time() - t0)]
    if swap:
        why = f"its exact IoU {m_alt['iou']:.4f} is above the {first_name} solution's {m_first['iou']:.4f} by more than {margin:g}"
        keep = {k: v for k, v in state.__dict__.items() if k in ("solution_ctx", "member_report")}
        state.__dict__.clear()
        state.__dict__.update(alt_state.__dict__)
        state.__dict__.update(keep)
        state.trial = False
        state.solution_ctx = dict(ctx, first="classic", replaced=first_name)
        result.operations = list(alt.operations)
        result.residual_voxels_pos, result.residual_voxels_neg = alt.residual_voxels_pos, alt.residual_voxels_neg
        result.recut_voxels = int(np.count_nonzero(state.cut_history >= 2))
        stopped = f"{alt_stopped}; the classic solution replaced the {first_name} solution"
    else:
        why = (f"the classic solution's exact IoU {m_alt['iou']:.4f} is not above its {m_first['iou']:.4f} by more "
               f"than {margin:g}")
    kept = "classic" if swap else first_name
    seconds = time.time() - t0
    yield rl.ResidualStep("solution", rl._snapshot(result), dict(
        solutions=rows, kept=kept, measure=MEASURE, reasons=reasons, tried=True, margin=margin, seconds=seconds,
        cut_history=state.cut_history.copy(),
        message=f"solution: {kept} kept - {why}; {first_name}: IoU {m_first['iou']:.4f}, {m_first['operations']} operation(s), "
                f"missing {m_first['missing_share']:.1%}, invented {m_first['invented_share']:.1%}; classic: IoU "
                f"{m_alt['iou']:.4f}, {m_alt['operations']} operation(s), missing {m_alt['missing_share']:.1%}, invented "
                f"{m_alt['invented_share']:.1%}; tried because {'; '.join(reasons)}; {seconds:.1f} s more"))
    return stopped
