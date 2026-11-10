#!/usr/bin/env python3
"""
The difference-driven decision tree (the user's rule of 2026-10-15): after the rough guess, every round of the loop
is one explicit decision on the difference between the model and the input, and a component of that difference the
primitives cannot explain becomes a CHILD - a sub-problem of its own, solved by the same procedure, that comes back as
one composite operation. The same node procedure runs at every depth; the loop repeats it until the difference is
within tolerance or nothing makes progress.

    node(T, sign, depth):
      G <- the rough guess of T (the starting-shape stage of residual_loop: start_shape "auto"; at a child: a cover)
      loop:
        r+ <- T \\ G (missing), r- <- G \\ T (excess): the loop grid's banded residual (_State.residual), as components
        D0  |r+| + |r-| <= tolerance x |T|                 accept, stop
        D6  the operation budget is used                   stop
        D1  an excess component above the floor (cuts first, D-003)   CARVE it: a primitive the judge passes, else a
                                                            child: the largest boxes inside it as one composite cut
                                                            (inside the excess by construction), else the cover child
        D2  a missing component above the floor            ADD it: a primitive, else a child (boxes inside it as one
                                                            composite add), else the cover child
        D3  only components below the rough floor remain   left for the feature phase (the leaf of the tree)
        D6  no component made progress                     stop with the reason
      a child's operation counts ONCE against the budget, however many boxes it holds; it is judged like every other
      operation (net gain >= the floor; for a cut the sign bound: no input material removed)

The cover child (tree_cover_child): a component that neither a primitive nor its inside boxes explain gets its
bounding box as the operation (a cut box over an excess component takes input material with it), and the material
it took wrongly is the GRANDCHILD, re-added at depth + 1 by the same procedure - the alternating sum of volumes
(literature note L04 section 1), depth-limited (tree_max_depth). The pair is accepted only when its net gain passes
the floor and the grandchild's leftover is small (the progress rule: a cover whose wrong material is not smaller than
the component itself makes no progress).

Why (the review of 2026-10-15, plan file "check-the-servers-review"): on the gate of 60 known trees every type source
gave 28 good runs; the aegir attribution named "the cover's excess was never cut" as the dominant defect (1127
regions, capped volume share 191 against 14 for omitted cuts), 94 good cutters were rejected by the material purity
check, 30 of 60 bare runs used up the operation budget. A single primitive seldom fits an excess region; its inside
boxes always do, and as one operation they do not use up the budget.

Every decision is a step of kind "decision" (ResidualStep, info: branch, reason, pos, neg, components) so that the path
record, the atlas and the aegir attribution can be read per branch.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field as dc_field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from reco_engine.result_schema import Operation, Primitive, StageSource

BRANCHES = {"D0": "accept", "D1": "carve", "D2": "add", "D3": "features", "D4": "ladder", "D5": "solids", "D6": "stop"}
SOURCE_PREFIX = "tree:"                    # StageSource.mode of a child's operation: "tree:carve" / "tree:add" / "tree:cover"


@dataclass
class Decision:
    branch: str                            # "D0" ... "D6"
    reason: str
    size: int = 0
    pos: int = 0
    neg: int = 0
    components: List[Tuple[str, np.ndarray]] = dc_field(default_factory=list)   # (op, mask), the order to try

    @property
    def message(self) -> str:
        return f"{self.branch} {BRANCHES[self.branch]}: {self.reason}"


# ------------------------------------------------------------------------------------------------ the composite

def fuse_chain(parts: Sequence[Primitive], label: str, mode: str) -> Primitive:
    """parts[0] | parts[1] | ... as nested composites (combine "fuse"); one part stays itself. Sources mode `mode`."""
    out = parts[0]
    for p in parts[1:]:
        out = Primitive(kind="composite", center_mm=out.center_mm, dims_mm={}, parts=[out, p], combine="fuse")
    out = copy.copy(out)
    out.label = label
    out.sources = [StageSource(s, mode, 0.0) for s in ("type", "rotation", "dimensions", "position")]
    return out


def is_child(p: Primitive) -> bool:
    return any(s.mode.startswith(SOURCE_PREFIX) for s in (p.sources or []))


# ------------------------------------------------------------------------------------------------ the classification

def classify(state, cfg, n_in: int, n_ops: int, max_ops: int, floor_of: Callable[[str], float],
             components: Callable, min_piece: int, tried: set, signature: Callable) -> Decision:
    """The branch for the current difference (the docstring's rules), with the components to try in order: cuts first
    (D-003) unless cfg.cuts_first is False, the largest first within a kind."""
    pos, neg = state.residual()
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    size = n_pos + n_neg
    d = Decision("D0", "", size=size, pos=n_pos, neg=n_neg)
    if size <= cfg.tolerance * n_in:
        # the banded residual is within tolerance; the plain voxel IoU beside it, because on a thin part the
        # half-voxel band hides real volume (a stepped shaft of five-voxel walls: residual 0, IoU 0.946)
        model = state.occ_model
        inter, union = int(np.count_nonzero(model & state.occ_in)), int(np.count_nonzero(model | state.occ_in))
        iou = inter / union if union else 0.0
        d.branch, d.reason = "D0", (f"residual within tolerance ({size} <= {cfg.tolerance:g} x {n_in}); plain voxel IoU "
                                    f"{iou:.4f}" + (" - the band hides volume on thin walls" if iou < 0.98 else ""))
        return d
    if n_ops >= max_ops:
        d.branch, d.reason = "D6", "operation budget used"
        return d
    cuts = [("cut", c) for c in components(neg, max(min_piece, int(np.ceil(floor_of("cut")))))]
    adds = [("add", c) for c in components(pos, max(min_piece, int(np.ceil(floor_of("add")))))]
    cuts = [(op, c) for op, c in cuts if signature(op, c) not in tried]
    adds = [(op, c) for op, c in adds if signature(op, c) not in tried]
    cuts_first = cfg.cuts_first is not False
    first, second = (cuts, adds) if cuts_first else (adds, cuts)
    if first:
        d.components = first + second
    elif second:
        d.components = second
    if d.components:
        op = d.components[0][0]
        d.branch = "D1" if op == "cut" else "D2"
        d.reason = (f"{len(cuts)} excess and {len(adds)} missing component(s) above the floor; "
                    f"the {'largest excess' if op == 'cut' else 'largest missing'} one first ({int(d.components[0][1].sum())} voxels)")
        return d
    # nothing above the floor: is there anything at all?
    small = [c for m in (neg, pos) for c in components(m, min_piece)[:1]]
    if small:
        d.branch, d.reason = "D3", (f"only pieces below the rough floor remain (largest {max(int(c.sum()) for c in small)} "
                                    f"voxels): left for the feature phase")
    else:
        d.branch, d.reason = "D6", ("no component left that a primitive explains" if min_piece <= cfg.min_component
                                    else f"no piece of >= {min_piece} voxels left (the rest is for the feature phase)")
    return d


# ------------------------------------------------------------------------------------------------ the children

def _box_cover(state, comp: np.ndarray, cfg, field, largest_boxes, cube_primitive, snap) -> Tuple[List[Primitive], List[Primitive]]:
    """The largest completely filled boxes inside `comp` (the voxel-cube cover of the component): (each snapped onto the
    input's surface where its samples support it, the plain voxel boxes). Inside the component by construction."""
    boxes = largest_boxes(comp, count=int(cfg.tree_child_boxes), max_side=48, min_voxels=max(1, int(cfg.min_component)))
    snapped, plain, first_n = [], [], None
    for lo, hi in boxes:
        n = int(np.prod(hi - lo + 1))
        if first_n is None:
            first_n = n
        elif n < cfg.cubes_min_share * first_n:
            break
        p = cube_primitive(state, lo, hi)
        q = snap(state, p, field) if cfg.cubes_snap and field is not None else None
        plain.append(p)
        snapped.append(q if q is not None else p)
    return snapped, plain


def _raw_apply(state, prim: Primitive, op: str) -> Dict:
    """The new model SDF and the net gain of `op` with `prim`, WITHOUT the judge's material checks (the cover child is
    allowed to take material: its grandchild gives it back)."""
    bad = prim.defect()
    if bad is not None:
        return dict(ok=False, reason=f"invalid primitive: {bad}", gain=0, s_new=state.s_model, s_prim=None)
    s_c = state.prim_sdf(prim)
    new = state.apply(state.s_model, s_c, op)
    return dict(ok=True, reason="", gain=state.size() - state.size(new), s_new=new, s_prim=s_c)


class Node:
    """One target and the procedure on it. The root is the loop itself (its state holds the input and the model); a
    child is a component of the root's difference, solved on the same grid and state (the model is shared: a child's
    operation changes the root's model directly, and the root's next difference judges the result)."""

    def __init__(self, loop, state, field, result, cfg, registry, purity_min: float, floor_of: Callable):
        self.rl, self.state, self.field, self.result, self.cfg = loop, state, field, result, cfg
        self.registry, self.purity_min, self.floor_of = registry, purity_min, floor_of
        self.work = 0                                            # components solved at this node (the work budget)
        self.candidates = 0                                      # candidates evaluated (reported, not a limit)

    # -- one component

    def solve(self, op: str, comp: np.ndarray, depth: int, snapshot, what):
        """Generator: the steps of one component; returns (primitive, evaluation, how) or (None, None, why)."""
        rl, cfg, state = self.rl, self.cfg, self.state
        floor = max(float(cfg.min_gain), self.floor_of(op))
        n_comp = int(comp.sum())
        # 1. a primitive the judge passes (the loop's own candidates: the cascade, the void fit, the largest boxes,
        #    strict surfaces, booleans and tools)
        self.work += 1
        with rl._field_backend(cfg.field_backend):
            chosen = rl._best_candidate(state, comp, op, self.field, self.registry, cfg, self.purity_min)
        self.candidates += len(chosen["tried"])
        # D5 at the candidate: a cut that severs the model into more solids than the input has is no candidate; the
        # next passing one by gain is taken instead (the frame review and the ABC batch: 12.6 % multi-solid models)
        severed = []
        if True:                                                      # every op: D5 on the grid for cuts, D5 exact for all
            passing = sorted([(p, e) for p, e in chosen["tried"] if e["ok"] and e.get("s_new") is not None],
                             key=lambda t: -t[1]["gain"])
            if op == "add" and chosen["best"] is not None:            # an add keeps the loop's own ranking (the surface rule)
                passing = [chosen["best"]] + [t for t in passing if t[0] is not chosen["best"][0]]
            best, checks = None, 0
            for p, e in passing:
                if op in ("cut", "intersect") and rl._severs(state, e["s_new"]):
                    e["ok"], e["reason"] = False, "severs the model (D5)"
                    severed.append(p)
                    continue
                if checks < cfg.exact_solids_max_checks:               # D5 exact: the FreeCAD build of the program + p
                    checks += 1
                    why = rl._exact_severs(state, self.result, cfg, p, e.get("op", op), name=f"d{depth}")
                    if why:
                        e["ok"], e["reason"] = False, why
                        severed.append(p)
                        continue
                best = (p, e)
                break
            chosen["best"] = best
        for prim, ev in chosen["tried"]:
            c_op = ev.get("op", op)
            yield rl.ResidualStep("candidate", snapshot(self.result), dict(
                op=c_op, primitive=prim, gain=ev["gain"], ok=ev["ok"], reason=ev["reason"], surface=ev.get("surface"),
                depth=depth, message=f"{c_op} {what(prim)}: gain {ev['gain']}" + (f", {ev['reason']}" if ev["reason"] else "")))
        best_prim = chosen["best"] if chosen["best"] is not None and chosen["best"][1]["gain"] >= floor else None
        if best_prim is not None and best_prim[1]["gain"] >= cfg.tree_primitive_share * n_comp:
            prim, ev = best_prim                                   # the primitive explains the component: taken
            return prim, ev, "primitive"
        why = chosen.get("why") or "no primitive passes"
        if chosen["best"] is not None and best_prim is None:
            why = f"best primitive gains {chosen['best'][1]['gain']} < the floor {floor:.0f}"
        elif best_prim is not None:
            why = f"best primitive gains {best_prim[1]['gain']} < {cfg.tree_primitive_share:g} of the component's {n_comp}"
        # 2. the child: the boxes inside the component as one composite operation; it competes with a primitive that
        #    explains less than tree_primitive_share of the component by gain
        boxes, plain = _box_cover(state, comp, cfg, self.field, rl.largest_boxes, rl._cube_primitive, rl._snap_box_faces)
        if boxes:
            mode = SOURCE_PREFIX + ("carve" if op == "cut" else "add")
            # each box's open faces pushed where that costs nothing (as the loop does for every box: a cutter ending on
            # the part's face leaves a skin), so the child is not one voxel short on its air side
            grown = []
            for b in boxes:
                e = rl._evaluate(state, b, op, self.purity_min, connectivity=False)
                self.candidates += 1
                if e["ok"]:
                    b, e = rl._extend_open_faces(state, b, op, self.purity_min, e)
                grown.append(b)
            boxes = grown
            child = fuse_chain(boxes, f"{op} child: {len(boxes)} box(es) inside the {'excess' if op == 'cut' else 'missing'} "
                                      f"component of {n_comp} voxels", mode)
            ev = rl._evaluate(state, child, op, self.purity_min)
            self.candidates += 1
            if ev["ok"] and op == "cut" and rl._severs(state, ev["s_new"]):
                ev["ok"], ev["reason"] = False, "severs the model (D5)"
            if ev["ok"]:                                              # D5 exact: a cut that severs, an add that floats
                severed_by = rl._exact_severs(state, self.result, cfg, child, op, name=f"child{depth}")
                if severed_by:                                        # (not `why`: that is the explanation so far,
                    ev["ok"], ev["reason"] = False, severed_by        # appended to below - 3 of 60 tree_lg runs crashed)
            if not ev["ok"] and any(a is not b for a, b in zip(boxes, plain)):   # a snapped face moved into material
                child2 = fuse_chain(plain, child.label + ", not snapped", mode)
                ev2 = rl._evaluate(state, child2, op, self.purity_min)
                self.candidates += 1
                if ev2["ok"] and op == "cut" and rl._severs(state, ev2["s_new"]):
                    ev2["ok"], ev2["reason"] = False, "severs the model (D5)"
                if ev2["ok"]:
                    severed_by = rl._exact_severs(state, self.result, cfg, child2, op, name=f"child{depth}")
                    if severed_by:
                        ev2["ok"], ev2["reason"] = False, severed_by
                if ev2["ok"]:
                    child, ev = child2, ev2
            yield rl.ResidualStep("candidate", snapshot(self.result), dict(
                op=op, primitive=child, gain=ev["gain"], ok=ev["ok"], reason=ev["reason"], depth=depth, child="boxes",
                message=f"{op} {child.label}: gain {ev['gain']}" + (f", {ev['reason']}" if ev["reason"] else "")))
            if ev["ok"] and ev["gain"] >= floor:
                if best_prim is not None:
                    prim, pev = best_prim
                    if pev["gain"] >= ev["gain"]:
                        return prim, pev, "primitive (over the box child)"
                    # the loop's surface rule (cfg.surface_choice): a primitive that gains at least
                    # surface_min_gain_share of the child's gain and lies on more of the nearby surface wins - the voxel
                    # gain cannot see the half-voxel skins a staircase of boxes leaves on a curved face
                    if cfg.surface_choice and self.field is not None and pev["gain"] >= cfg.surface_min_gain_share * ev["gain"]:
                        ev["surface"] = self.surface_of(child, comp, op)
                        if pev.get("surface") is None:
                            pev["surface"] = self.surface_of(prim, comp, op)
                        if (pev["surface"] or 0.0) > (ev["surface"] or 0.0):
                            return prim, pev, f"primitive (surface {pev['surface']:.0%} over the box child's {ev['surface'] or 0:.0%})"
                return child, ev, "child boxes"
            why += f"; the box child gains {ev['gain']}" + (f" ({ev['reason']})" if ev["reason"] else "") + f" < {floor:.0f}"
        else:
            why += "; no filled box inside the component"
        if best_prim is not None:                                    # a small primitive is still better than nothing
            prim, ev = best_prim
            return prim, ev, "primitive (small)"
        # 3. the cover child: the component's bounding box, its wrong material re-added by the grandchild
        if cfg.tree_cover_child and depth < cfg.tree_max_depth:
            got = yield from self.cover_child(op, comp, depth, floor, snapshot, what)
            if got is not None:
                return got
            why += "; no cover child"
        return None, None, why

    def surface_of(self, prim: Primitive, comp: np.ndarray, op: str) -> Optional[float]:
        """The share of the nearby input surface that lies on `prim` (cascade.near_surface_agreement in the component's
        window), None when it cannot be measured."""
        rl, state = self.rl, self.state
        try:
            sub, sl = rl._window(state, comp)
            channels = rl._voxel_channels(comp[sl], sub)
            scope = (~state.occ_model if op == "add" else state.occ_model)[sl]
            return float(rl.cascade.near_surface_agreement(prim, channels, sub, self.field, scope=scope))
        except Exception:
            return None

    def cover_child(self, op: str, comp: np.ndarray, depth: int, floor: float, snapshot, what):
        """The component's bounding box as the operation, and the material it takes wrongly re-added at depth + 1 (for
        a cut: input material inside the box; for an add: non-input material the box adds, cut again). Returns
        (composite, evaluation, "cover child") or None; the state is left unchanged unless accepted."""
        rl, cfg, state = self.rl, self.cfg, self.state
        lo, hi = rl._bbox(comp)
        box = rl._cube_primitive(state, np.asarray(lo), np.asarray(hi))
        snapped = rl._snap_box_faces(state, box, self.field) if cfg.cubes_snap and self.field is not None else None
        if snapped is not None:
            box = snapped
        box.label = f"{op} cover child: the bounding box of a {int(comp.sum())}-voxel component"
        ev = _raw_apply(state, box, op)
        self.candidates += 1
        if not ev["ok"] or ev["s_prim"] is None:
            return None
        s_before = state.s_model
        inside_box = ev["s_prim"] >= 0
        if op == "cut":
            wrong = inside_box & state.occ_in & (s_before >= 0)        # input material the cut takes
            back_op = "add"
        else:
            wrong = inside_box & ~state.occ_in & ~(s_before >= 0)     # non-input material the add puts in
            back_op = "cut"
        n_wrong, n_comp = int(wrong.sum()), int(comp.sum())
        if n_wrong >= n_comp:                                          # the progress rule
            yield rl.ResidualStep("rejected", snapshot(self.result), dict(
                op=op, primitive=box, depth=depth, child="cover",
                message=f"{box.label}: takes {n_wrong} voxels wrongly, not fewer than the component's {n_comp}: no progress"))
            return None
        state.s_model = ev["s_new"]                                     # the grandchild works on the cut/added model
        try:
            parts, grand_gain = [box], 0
            if n_wrong >= cfg.min_component:
                g_prim, g_ev, g_how = yield from self.solve(back_op, wrong, depth + 1, snapshot, what)
                if g_prim is None:
                    yield rl.ResidualStep("rejected", snapshot(self.result), dict(
                        op=op, primitive=box, depth=depth, child="cover",
                        message=f"{box.label}: the grandchild ({back_op} of {n_wrong} voxels) found nothing ({g_how})"))
                    return None
                state.s_model = g_ev["s_new"]
                grand_gain = int(g_ev["gain"])
                parts.append((back_op, g_prim))
            total_gain = state.size(s_before) - state.size(state.s_model)
            if rl._severs(state, state.s_model, before=s_before):
                yield rl.ResidualStep("rejected", snapshot(self.result), dict(
                    op=op, primitive=box, depth=depth, child="cover", message=f"{box.label}: severs the model (D5)"))
                return None
            left = int(((state.s_model < 0) & state.occ_in & wrong).sum()) if op == "cut" else \
                int(((state.s_model >= 0) & ~state.occ_in & wrong).sum())
            if total_gain < floor or left > cfg.tree_cover_leftover * max(1, n_wrong):
                yield rl.ResidualStep("rejected", snapshot(self.result), dict(
                    op=op, primitive=box, depth=depth, child="cover",
                    message=f"{box.label}: net gain {total_gain} (grandchild {grand_gain}), {left} of {n_wrong} wrong voxels "
                            f"left: below the floor {floor:.0f} or above the leftover share {cfg.tree_cover_leftover:g}"))
                return None
        finally:
            s_after = state.s_model
            state.s_model = s_before
        # one composite: for a cut, box minus the re-added material (cut (box - M)); for an add, box minus the cut-away
        # material (add (box - M)); so the operation is a single boolean on the parent
        if len(parts) == 1:
            prim = box
        else:
            prim = Primitive(kind="composite", center_mm=box.center_mm, dims_mm={}, parts=[box, parts[1][1]], combine="cut")
        prim.label = f"{op} cover child (box of a {n_comp}-voxel component" + (f", {n_wrong} voxels given back)" if len(parts) > 1 else ")")
        prim.sources = [StageSource(s, SOURCE_PREFIX + "cover", 0.0) for s in ("type", "rotation", "dimensions", "position")]
        why = rl._exact_severs(state, self.result, cfg, prim, op, name=f"cover{depth}")   # D5 exact on the composite
        if why:
            yield rl.ResidualStep("rejected", snapshot(self.result), dict(
                op=op, primitive=prim, depth=depth, child="cover", message=f"{prim.label}: {why}"))
            return None
        s_prim = state.prim_sdf(prim)
        ev2 = dict(ok=True, reason="", gain=int(state.size() - state.size(s_after)), s_new=s_after, s_prim=s_prim,
                   depth=depth)
        yield rl.ResidualStep("candidate", snapshot(self.result), dict(
            op=op, primitive=prim, gain=ev2["gain"], ok=True, reason="", depth=depth, child="cover",
            message=f"{op} {prim.label}: gain {ev2['gain']}"))
        return prim, ev2, "cover child"


# ------------------------------------------------------------------------------------------------ the cover check

def contained_start(rl, state, field, result, cfg, registry, snapshot, what):
    """The tree's first decision, D1c (2026-10-15, cy_t: a T of two cylinders got the bounding cylinder as its cover,
    +48 % excess, and the carving of a curved excess around two bars cannot be done with boxes): when the cover's
    excess is more than tree_max_cover_excess of the input's voxels and the cascade's best CONTAINED primitive (its
    material at least tree_contained_min_precision inside the input) explains at least tree_contained_min_explained
    of the input, the cover is replaced by that primitive - a union of bars is built by adds, not carved out of a
    cover. Returns the step info dict when it replaced the cover, else None; the result keeps the cover with
    accepted = False and a label saying why."""
    ops = [o for o in result.operations if o.accepted]
    if len(ops) != 1 or ops[0].op != "add":
        return None
    pos, neg = state.residual()
    n_in = int(state.occ_in.sum())
    excess = int(neg.sum())
    if excess <= cfg.tree_max_cover_excess * n_in:
        return None
    n_in = max(1, n_in)
    # the candidates: the cascade's primitives at ANY pass mark (with the first add's 0.9 a union of sixteen leaves
    # offers nothing: measured 2026-10-15 on aegir_n16_U_000130), and the largest filled boxes inside the part - a leaf
    # by construction, snapped onto the part's faces
    prim = getattr(state, "root_cascade", None)                   # the start stage's answer, when it had one
    if prim is None or prim.is_unknown:
        prim = rl.cascade.run_cascade(state.t_in, state.spec, registry, field=field, top_k=cfg.top_k, models=cfg.models,
                                      pass_iou=0.0, part_diagonal=state.part_diagonal)
    cands = ([] if prim.is_unknown else [prim]) + [a for a in prim.alternatives if not a.is_unknown]
    for lo, hi in rl.largest_boxes(state.occ_in, count=3, max_side=48, min_voxels=max(8, int(cfg.min_component))):
        b = rl._cube_primitive(state, lo, hi)
        b.label = "a largest box inside the part"
        q = rl._snap_box_faces(state, b, field) if field is not None else None
        cands.append(q if q is not None else b)
    best = None
    cover_in = state.occ_model
    for c in cands:
        if c.defect() is not None:
            continue
        s_c = state.prim_sdf(c)
        inside = s_c >= 0
        n_prim = int(inside.sum())
        if n_prim == 0:
            continue
        hit = int(np.count_nonzero(inside & state.occ_in))
        precision, explained = hit / n_prim, hit / max(1, n_in)
        missing_after = n_in - hit
        same_as_cover = int(np.count_nonzero(inside & cover_in)) >= 0.99 * max(n_prim, int(cover_in.sum()))
        # the contained route leaves `missing_after` to add, the cover route leaves `excess` to carve: swap when the
        # former is clearly the smaller task (an n16 tree: the largest leaf explains 15 % of the input, the cover's
        # excess is 367 % of it - the adds win by far; a box with a hole: the cover IS the box - no swap)
        if precision >= cfg.tree_contained_min_precision and explained >= cfg.tree_contained_min_explained \
                and missing_after < cfg.tree_contained_margin * excess and not same_as_cover \
                and (best is None or explained > best[2]):
            best = (c, s_c, explained, precision)
    if best is None:
        return None
    c, s_c, explained, precision = best
    cover = ops[0]
    cover.accepted = False
    cover.primitive.label = ((cover.primitive.label + "; ") if cover.primitive.label else "") + \
        f"replaced by the tree: the cover's excess was {excess / n_in:.0%} of the input (> {cfg.tree_max_cover_excess:.0%})"
    c = copy.copy(c)
    c.alternatives = []
    c.label = f"contained start: {c.kind} explaining {explained:.0%} of the input ({precision:.0%} inside it)"
    c.sources = list(c.sources) + [StageSource("type", SOURCE_PREFIX + "contained", float(explained))]
    state.s_model = s_c
    state.cut_history[:] = 0
    state.tools = [c]
    result.operations.append(Operation("add", c, component_voxels=n_in, net_gain=float(hit_gain(state))))
    return dict(branch="D1c", reason=f"the cover's excess was {excess / n_in:.0%} of the input: replaced by a contained "
                                      f"{c.kind} explaining {explained:.0%} ({precision:.0%} inside the input)",
                primitive=c, cover=cover.primitive)


def hit_gain(state) -> int:
    """|R| removed by the current model against an empty one (the net gain of a start)."""
    empty = np.full(state.occ_in.shape, -np.inf, state.dtype)
    return state.size(empty) - state.size()


# ------------------------------------------------------------------------------------------------ the loop

def grade_of(first_gain: Optional[Dict], k_op: str, gain: int, child: bool) -> Dict:
    """The rough grade of an accepted operation (its gain over the first operation's of its kind), and the first
    gain's bookkeeping: a CHILD never sets the scale - a composite of eight boxes gains the whole excess at once, and
    every later component would then fall "below the floor" (measured 2026-10-15 on aegir_n16_U_000130: a 62677-voxel
    component left to the feature phase after a child's first cut). Returns {} when there is no rough rule or no scale
    yet."""
    if first_gain is None:
        return {}
    if first_gain.get(k_op) is None:
        if child:
            return {}
        first_gain[k_op] = float(gain)
    grade = float(gain) / first_gain[k_op] if first_gain[k_op] else 1.0
    return dict(grade=grade, first_cut_gain=first_gain[k_op])


def tree_loop(rl, state, field, result, cfg, registry, purity_min: float, max_ops: int, n_in: int,
              floor_of: Callable, kind_of: Callable, first_gain: Dict, name: str):
    """The rounds of the tree at the root (replaces residual_loop._residual_phase's while loop when cfg.mode == "tree"):
    classify -> act on the first component that gives an operation -> repeat. Returns the `stopped` text. `rl` is the
    residual_loop module (its helpers, steps and judge)."""
    node = Node(rl, state, field, result, cfg, registry, purity_min, floor_of)
    tried: set = set()
    stopped, stalls = "", 0
    snapshot, what = rl._snapshot, rl._what
    first_round = True
    while True:
        low = state.memory_guard.check() if getattr(state, "memory_guard", None) is not None else None
        if low:
            stopped = f"{low}: the tree stops with the model it has"
            yield rl.ResidualStep("decision", snapshot(result), dict(branch="D6", reason=stopped, pos=0, neg=0, size=state.size(),
                                                                     components=0, work=node.work, candidates=node.candidates,
                                                                     message=f"D6 stop: {stopped}"))
            break
        n_ops = len([o for o in result.operations if o.accepted])
        min_piece = max(cfg.min_component, int(round(cfg.rough_min_piece_share * n_in)))
        d = classify(state, cfg, n_in, n_ops, max_ops, floor_of, rl._components, min_piece, tried, rl._signature)
        pos, neg = state.residual()
        result.residual_voxels_pos, result.residual_voxels_neg = d.pos, d.neg
        result.recut_voxels = int(np.count_nonzero(state.cut_history >= 2))
        yield rl.ResidualStep("residual", snapshot(result), dict(pos=pos, neg=neg, size=d.size, cut_history=state.cut_history.copy(),
                                                                 message=f"|R| = {d.size} (+{d.pos} / -{d.neg})"))
        yield rl.ResidualStep("decision", snapshot(result), dict(branch=d.branch, reason=d.reason, pos=d.pos, neg=d.neg,
                                                                 size=d.size, components=len(d.components), work=node.work,
                                                                 candidates=node.candidates, message=d.message))
        if d.branch in ("D0", "D3", "D6"):
            stopped = d.reason if d.branch != "D0" else "residual within tolerance"
            break
        if node.work >= cfg.tree_work_budget:
            stopped = f"work budget used ({node.work} components solved, {node.candidates} candidates evaluated)"
            break
        n_children = sum(1 for o in result.operations if o.accepted and is_child(o.primitive))
        if n_children >= cfg.tree_max_children:
            # aegir_n16_U_000130 (2026-10-15): 24 box children of 8 boxes each crashed freecadcmd in the final build
            # (OCCT on hundreds of coincident box faces); such a program is a sign of the wrong cover, not a model
            stopped = f"child budget used ({n_children} children of boxes; the cover is probably wrong for this part)"
            break
        taken = None
        for op, comp in d.components:
            lo, hi = rl._bbox(comp)
            yield rl.ResidualStep("component", snapshot(result), dict(op=op, mask=comp, bbox=(lo, hi), voxels=int(comp.sum()),
                                                                      depth=0, message=f"{op} component: {int(comp.sum())} voxels"))
            prim, ev, how = yield from node.solve(op, comp, 0, snapshot, what)
            if prim is None:
                tried.add(rl._signature(op, comp))
                yield rl.ResidualStep("rejected", snapshot(result), dict(op=op, bbox=(lo, hi), depth=0, message=f"{op}: {how}"))
                continue
            taken = (op, comp, prim, ev, how)
            break
        # D1c, the cover check, once: the first round found no PRIMITIVE for the largest excess component (a child or
        # nothing) and the cover's excess is large - a union of bars under a cover; the cover is replaced by the
        # cascade's best contained primitive and the round starts again (the missing bars come as adds)
        if first_round and cfg.tree_contained_start and d.branch == "D1" and \
                (taken is None or not taken[4].startswith("primitive")):
            first_round = False
            swapped = contained_start(rl, state, field, result, cfg, registry, snapshot, what)
            if swapped is not None:
                tried.clear()
                yield rl.ResidualStep("decision", snapshot(result), dict(swapped, pos=0, neg=0, size=state.size(), components=0,
                                                                         work=node.work, candidates=node.candidates,
                                                                         message=f"D1c cover replaced: {swapped['reason']}"))
                yield rl.ResidualStep("accepted", snapshot(result), dict(op="add", primitive=swapped["primitive"], gain=hit_gain(state),
                                                                         recut=0, cut_history=state.cut_history.copy(), how="contained start",
                                                                         message=f"add {what(swapped['primitive'])} (contained start): |R| -{hit_gain(state)}"))
                continue
        first_round = False
        if taken is None:
            stalls += 1
            if stalls >= 2 or not d.components:
                stopped = "no component made progress (D6)"
                break
            continue
        stalls = 0
        op, comp, prim, ev, how = taken
        gain = int(ev["gain"])
        k_op = kind_of(op)
        grade_info = grade_of(first_gain, k_op, gain, is_child(prim))
        if grade_info:
            prim.sources = list(prim.sources) + [StageSource("dimensions", "rough_grade", float(min(grade_info["grade"], 1.0)),
                                                             detail=dict(grade=grade_info["grade"], first_gain=grade_info["first_cut_gain"], op=k_op))]
        recut = state.commit(ev["s_new"], ev["s_prim"], op)
        if op in ("add", "cut") and prim.kind != "halfspace" and not is_child(prim):
            state.tools.append(prim)
        result.operations.append(Operation(op, prim, component_voxels=int(comp.sum()), net_gain=float(gain), recut_voxels=recut))
        yield rl.ResidualStep("accepted", snapshot(result), dict(
            op=op, primitive=prim, gain=gain, recut=recut, cut_history=state.cut_history.copy(), how=how, **grade_info,
            message=f"{op} {what(prim)} ({how}): |R| -{gain}" + (f", grade {grade_info['grade']:.3f}" if grade_info else "")
                    + (f", {recut} voxels cut again" if recut else "")))
    return stopped
