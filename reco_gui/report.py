#!/usr/bin/env python3
"""
Qt-free rendering of a ViewState into the HTML of the detail panel, and small helpers for the timeline rows.
Everything a supervisor needs to judge one step is here: what the Type CNN thought (with the runners-up), the pose the
Rotation CNN proposed, the parameters the Dimension CNN proposed, what the local search did to the candidate, the fit
against the acceptance threshold, and the reason for the decision.
"""

from __future__ import annotations

import html
import math
import re
from pathlib import Path

STAGE_COLOURS = {          # timeline row background per stage
    "accepted": "#d8f2e8", "rejected": "#fbdcd8", "phase_done": "#e9e9e4", "done": "#dce9f9",
    "feature_residual": "#efece0", "feature_accepted": "#d8f2e8", "feature_rejected": "#fde6cf", "feature_done": "#e9e9e4",
}
PIECE_COLOURS = {"pending": "#6b6a66", "current": "#b38f00", "guessed": "#2a78d6", "split": "#5a4fcf", "accepted": "#127a52", "rejected": "#c2600a",
                 "skipped": "#3d3c39"}
MAX_PIECE_ROWS = 60                     # pieces listed in the detail panel (the largest first, as the loop lists them)


def quaternion_angle_axis(q):
    """[qx, qy, qz, qw] -> (angle in degrees, unit axis)."""
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    angle = 2.0 * math.acos(max(-1.0, min(1.0, w)))
    s = math.sqrt(max(1.0 - w * w, 0.0))
    axis = (x / s, y / s, z / s) if s > 1e-9 else (0.0, 0.0, 1.0)
    return math.degrees(angle), axis


def bar(fraction: float, colour: str = "#2a78d6", width: int = 120) -> str:
    fraction = max(0.0, min(1.0, fraction))
    return (f'<span style="display:inline-block;width:{width}px;background:#e6e5e0;">'
            f'<span style="display:inline-block;width:{int(width * fraction)}px;background:{colour};">&nbsp;</span></span>')


def timeline_row(state) -> tuple:
    """(#, phase, stage, iteration, summary) for the timeline table."""
    step = state.step
    if getattr(step, "iteration", None) is None:                     # the cascade mode's steps outside an iteration
        it = ""
    else:
        it = "" if step.stage in ("seed", "phase_done", "done", "field", "base", "grade") else str(step.iteration + 1)
    return (str(state.index + 1), step.phase, step.stage, it, step.message)


def _section(title: str, body: str) -> str:
    return f'<h4 style="margin:10px 0 3px 0;">{html.escape(title)}</h4>{body}'


def _kv(rows) -> str:
    cells = "".join(f'<tr><td style="color:#6b6a66;padding-right:10px;">{k}</td><td>{v}</td></tr>' for k, v in rows)
    return f'<table cellspacing="0" cellpadding="1">{cells}</table>'


def render_field_state(state) -> str:
    """HTML for the detail panel in the vector-field mode (a reco_gui.field_session.FieldState)."""
    if state is None:
        return "<p style='color:#6b6a66'>No step yet. Choose a part and press <b>Start</b>.</p>"
    step = state.step
    parts = [f"<p style='margin:0 0 4px 0;'>Step {state.index + 1}: <b>{html.escape(step.stage)}</b><br>"
             f"<span style='color:#3d3c39;'>{html.escape(step.message)}</span></p>"]
    if state.surface is not None:
        rows = [("surface points", str(len(state.surface))),
                ("tolerance", f"{state.tol:.4g} mm (0.4 % of the extent)"),
                ("explained by the model", f"{bar(state.explained, '#1baf7a')} {state.explained:.1%}")]
        if state.unexplained is not None:
            rows.append(("unexplained points", str(len(state.unexplained))))
        if state.invented is not None:
            rows.append(("model surface off the input", f"{state.invented:.1%} (farther than 2 tol: invented geometry)"))
        parts.append(_section("Field and model", _kv(rows)))
    if state.ops:
        rows = []
        for i, o in enumerate(state.ops):
            params = "  ".join(f"{k}={v:.4g}" for k, v in o.params.items())
            rows.append((f"{i + 1}. <b>{o.op}:{o.kind}</b>", f"{o.explained_after:.1%} after"
                         f"<br><span style='color:#6b6a66'>{html.escape(params)}</span>"))
        parts.append(_section("Program", _kv(rows)))
    fc = state.freecad
    if fc is not None:
        if "error" in fc:
            rows = [("<span style='color:#b3261e'>error</span>", html.escape(str(fc["error"])))]
        else:
            colour = "#127a52" if fc["valid"] and fc["n_solids"] == 1 else "#b3261e" if not fc["valid"] else "#b36a1e"
            rows = [("valid", f"<b style='color:{colour}'>{fc['valid']}</b>"), ("volume", f"{fc['volume']:.4g} mm&sup3;"),
                   ("solids", str(fc["n_solids"]))]
            if fc["leftover_volumes"]:
                rows.append(("<span style='color:#b36a1e'>the CSG split off</span>",
                             ", ".join(f"{v:.4g} mm&sup3;" for v in fc["leftover_volumes"])))
        parts.append(_section("FreeCAD (real CSG)", _kv(rows)))
    if state.cluster is not None:
        where = {True: "in the material: a cutter", False: "outside: an addition"}.get(state.inside, "")
        parts.append(_section("Cluster", _kv([("points", str(len(state.cluster))), ("lies", where)])))
    c = state.candidate
    if c is not None and state.verdict != "accepted" and c.op != "base":
        rows = [("operation", f"<b>{c.op}:{c.kind}</b>"), ("explains of the cluster", f"{c.explained_fit:.0%}"),
                ("model would explain", f"{state.trial_explained:.1%}" if state.trial_explained is not None else "")]
        if state.trial_invented is not None:
            rows.append(("model surface off the input", f"{state.trial_invented:.1%} with it (now {state.invented:.1%})"))
        rows += [(html.escape(k), f"{v:.4g}") for k, v in c.params.items()]
        if state.verdict:
            colour = "#127a52" if state.verdict == "accepted" else "#b3261e"
            rows.insert(1, ("decision", f"<b style='color:{colour}'>{state.verdict}</b>"))
        parts.append(_section("Candidate", _kv(rows)))
    g = state.grade
    if g is not None:
        rows = [("input &rarr; model", f"mean {g['input_to_model']['mean']:.4g}, max {g['input_to_model']['max']:.4g} mm"),
                ("model &rarr; input", f"mean {g['model_to_input']['mean']:.4g}, max {g['model_to_input']['max']:.4g} mm"),
                ("Chamfer (both ways)", f"{g['symmetric_mean']:.4g} mm")]
        parts.append(_section("Grade against the input", _kv(rows)))
    if state.finished:
        parts.append(_section("Result", _kv([("operations", str(len(state.ops))), ("explained", f"{state.explained:.1%}"),
                                             ("stopped because", html.escape(state.stopped))])))
    return "".join(parts)


def _pose(p) -> str:
    angle, axis = quaternion_angle_axis(p.quaternion)
    c = ", ".join(f"{v:+.2f}" for v in p.center_mm)
    return f"centre ({c}) mm, {angle:.1f}&deg; about ({axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f})"


def _dims(p) -> str:
    text = "  ".join(f"{k}={v:.4g}" for k, v in (p.dims_mm or {}).items())
    if p.u:
        text += f"  u=[{math.degrees(p.u[0]):.0f}, {math.degrees(p.u[1]):.0f}]&deg;"
    if p.v:
        text += f"  v=[{math.degrees(p.v[0]):.0f}, {math.degrees(p.v[1]):.0f}]&deg;"
    return text


def _mode_colour(mode: str) -> str:
    if mode == "voxel_fallback":
        return "#7b3fb3"
    if mode.startswith("cnn:"):
        return "#2a78d6"
    if mode.startswith("descriptor"):
        return "#0e7c86"
    if mode == "fused":
        return "#5a4fcf"
    return "#127a52"


def _type_votes(entries, parent=None) -> list:
    """Rows per type entry (cascade.TypeVote.as_dict): source, weight, its probability of the winner (or its own top kind)
    and its subclass - what the fused type stage was made of."""
    rows = []
    for e in entries or ():
        probs = e.get("probs") or {}
        kind = parent if parent in probs else (max(probs, key=probs.get) if probs else None)
        colour = _mode_colour(str(e.get("mode", "")))
        text = f"<span style='color:{colour}'>{html.escape(str(e.get('mode', e.get('id', ''))))}</span> w={float(e.get('weight', 0)):.2f}"
        if kind is not None:
            text += f" &rarr; {html.escape(kind)} {bar(float(probs[kind]), colour, 40)} {float(probs[kind]):.2f}"
        if e.get("subclass"):
            text += f" <span style='color:#6b6a66'>/{html.escape(str(e['subclass']))}</span>"
        rows.append(("&nbsp;&nbsp;vote", text))
    return rows


def _provenance(p) -> str:
    """One line per cascade stage: which mode answered it and how sure it was; a fused type stage lists its votes."""
    if not p.sources:
        return "<span style='color:#6b6a66'>no provenance recorded</span>"
    rows = []
    for s in p.sources:
        colour = _mode_colour(s.mode)
        text = (f"<span style='color:{colour}'>{html.escape(s.mode)}</span> {bar(s.confidence, colour, 60)} {s.confidence:.2f}"
                + (f" <span style='color:#6b6a66'>({s.seconds:.2f} s)</span>" if s.seconds else ""))
        detail = getattr(s, "detail", None) or {}
        if s.stage == "type" and detail.get("subclass"):
            text += (f"<br><span style='color:#6b6a66'>subclass <b>{html.escape(str(detail['subclass']))}</b> "
                     f"from {html.escape(str(detail.get('subclass_source') or ''))}</span>")
        rows.append((html.escape(s.stage), text))
        if s.stage == "type" and detail.get("entries") and (len(detail["entries"]) > 1 or s.mode == "fused"):
            rows += _type_votes(detail["entries"], p.kind)
    return _kv(rows)


def _type_stage_section(ts) -> str:
    """The type stage on the whole input (cascade.TypeStageResult.as_dict() on the input step): parent, subclass and the
    source it came from, then every entry's vote with its fusion weight."""
    if not ts or not ts.get("parent"):
        return ""
    probs = ts.get("probs") or {}
    mode = str(ts.get("mode", ""))
    rows = [("parent", f"<b>{html.escape(str(ts['parent']))}</b> {bar(float(probs.get(ts['parent'], 0.0)), _mode_colour(mode), 60)} "
                       f"{float(probs.get(ts['parent'], 0.0)):.2f} <span style='color:{_mode_colour(mode)}'>{html.escape(mode)}</span>")]
    if ts.get("subclass"):
        rows.append(("subclass", f"<b>{html.escape(str(ts['subclass']))}</b> <span style='color:#6b6a66'>from "
                                 f"{html.escape(str(ts.get('subclass_source') or ''))} &rarr; sub-loop hint</span>"))
    else:
        rows.append(("subclass", "<span style='color:#6b6a66'>none (analytic sub-loop gates)</span>"))
    ranked = sorted(probs.items(), key=lambda kv: -kv[1])
    rows.append(("kinds", "  ".join(f"{html.escape(k)} {v:.2f}" for k, v in ranked)))
    rows += _type_votes(ts.get("entries"), ts["parent"])
    return _section("Type stage (whole input)", _kv(rows))





def _alternatives(p) -> str:
    if not p.alternatives:
        return ""
    rows = [(f"{i + 2}. {html.escape(a.kind)}", f"conf {a.confidence:.2f}"
             + (f", residual {a.residual:.1%}" if a.residual is not None else "")
             + f"<br><span style='color:#6b6a66'>{_dims(a)}</span>") for i, a in enumerate(p.alternatives)]
    return "<div style='margin-left:8px'><i>alternatives (pruned top-k)</i>" + _kv(rows) + "</div>"


def _member_section(state) -> str:
    """The member of the member sub-loop (reco_engine/ibeam.py): kind, profile (standard name when snapped, else the
    measured section and the nearest profile's envelope), axis, extent, the fitted section."""
    info = getattr(state, "member_info", None) or {}
    m = getattr(state, "member", None)
    if m is None and not info:
        return ""
    sec = info.get("section") if info.get("section") is not None else getattr(m, "section", None)
    kind = info.get("kind") or getattr(m, "kind", "")
    rows = [("kind", f"<b>{html.escape(str(kind))}</b>"),
            ("profile", f"<b style='color:#b38f00'>{html.escape(str(info.get('profile') or getattr(sec, 'label', '')))}</b>")]
    ibi = info.get("ibeam")
    if ibi:
        if ibi.get("snapped"):
            rows.append(("standard", html.escape(str(ibi.get("standard") or ""))
                         + f" (max. deviation {100 * (ibi.get('rel_distance') or 0):.1f} %)"))
        elif ibi.get("profile"):
            rows.append(("nearest standard", f"{html.escape(str(ibi['profile']))} envelope, not snapped "
                                             f"(max. deviation {100 * (ibi.get('rel_distance') or 0):.1f} %)"))
        if ibi.get("unit_scale", 1.0) != 1.0:
            rows.append(("unit scale", f"model units x {ibi['unit_scale']:g} = mm of the profile table"))
        meas = ibi.get("measured") or {}
        rows.append(("section (used)", "  ".join(f"{k}={ibi[k]:.4g}" for k in ("h", "b", "tw", "tf", "r") if k in ibi)))
        if meas:
            rows.append(("section (measured)", "  ".join(f"{k}={float(v):.4g}" for k, v in meas.items())))
        rows.append(("fit confidence", f"{bar(float(ibi.get('confidence', 0.0)), '#b38f00', 60)} {float(ibi.get('confidence', 0.0)):.3f}"))
    elif sec is not None and getattr(sec, "params", None):
        rows.append(("section", "  ".join(f"{k}={float(v):.4g}" for k, v in sec.params.items())))
    axis = info.get("axis") if info.get("axis") is not None else (tuple(m.axis) if m is not None else None)
    if axis is not None:
        rows.append(("axis", "(" + ", ".join(f"{float(v):+.3f}" for v in axis) + ")"))
    ext = info.get("extent") if info.get("extent") is not None else ((m.start, m.end) if m is not None else None)
    if ext is not None:
        rows.append(("extent", f"{float(ext[0]):.4g} .. {float(ext[1]):.4g} (length {float(ext[1]) - float(ext[0]):.4g})"))
    if m is not None and any(getattr(m, "joined", ())):
        rows.append(("joined ends", ", ".join(n for n, j in zip(("start", "end"), m.joined) if j)))
    if info.get("index") is not None and info.get("members"):
        rows.append(("member", f"{int(info['index']) + 1} of {int(info['members'])}"))
    if info.get("group"):                                            # stored, grouped by congruency (member_store.py)
        others = [int(i) + 1 for i in (info.get("group_members") or ()) if i != info.get("index")]
        rows.append(("congruency group", f"<b>{html.escape(str(info['group']))}</b>, {int(info.get('group_size') or 1)} member(s)"
                     + (f" <span style='color:#6b6a66'>(with member {', '.join(str(i) for i in others)})</span>" if others else "")))
    return _section("Member (yellow)", _kv(rows))


SOLUTION_KEPT, SOLUTION_DROPPED = "#127a52", "#6b6a66"              # the kept solution green, the other grey


def _solution_section(state) -> str:
    """The choice between the solutions before the fallback (reco_engine/solution_choice.py): one row per solution
    with its measure, the kept one marked; on the classic solution's own steps a note that they are a trial."""
    step, info = state.step, state.info or {}
    if step.stage != "solution":
        if info.get("solution"):
            return _section("Solution", _kv([("trial", f"<b style='color:#b36a1e'>{html.escape(str(info['solution']))}</b> solution, "
                                                       "computed for the comparison before the fallback (not the run's program yet)")]))
        return ""
    rows = []
    for r in info.get("solutions") or ():
        kept = r.get("solution") == info.get("kept")
        colour = SOLUTION_KEPT if kept else SOLUTION_DROPPED
        rows.append((f"<b style='color:{colour}'>{html.escape(str(r.get('solution')))}</b>" + (" (kept)" if kept else ""),
                     f"{bar(float(r.get('iou') or 0.0), colour, 60)} IoU {float(r.get('iou') or 0.0):.4f}, "
                     f"{int(r.get('operations') or 0)} operation(s), missing {float(r.get('missing_share') or 0.0):.1%}, "
                     f"invented {float(r.get('invented_share') or 0.0):.1%}"))
    rows.append(("measure", html.escape(str(info.get("measure") or ""))))
    if info.get("reasons"):
        rows.append(("tried because", "<br>".join(html.escape(str(x)) for x in info["reasons"])))
    if info.get("seconds") is not None:
        rows.append(("extra time", f"{float(info['seconds']):.1f} s"))
    return _section("Solutions before the fallback", _kv(rows))


def _num(value, fmt: str = ".4g", none: str = "-") -> str:
    try:
        return format(float(value), fmt) if value is not None else none
    except (TypeError, ValueError):
        return html.escape(str(value))


def _validation_rows(v) -> list:
    """The FreeCAD build that validated (or refused) a feature operation."""
    if not v:
        return []
    valid = v.get("valid")
    rows = [("FreeCAD build", ("<b style='color:#127a52'>valid</b>" if valid else "<b style='color:#b3261e'>INVALID</b>"
                               if valid is not None else "not built")
             + (f", {v['n_solids']} solid(s)" if v.get("n_solids") is not None else "")
             + (f", {_num(v.get('seconds'), '.1f')} s" if v.get("seconds") is not None else ""))]
    if v.get("residual_before") is not None or v.get("residual_after") is not None:
        before, after = v.get("residual_before"), v.get("residual_after")
        change = ""
        try:
            change = f" ({float(after) - float(before):+.4g})"
        except (TypeError, ValueError):
            pass
        rows.append(("exact residual", f"{_num(before)} &rarr; {_num(after)} mm&sup3;{change}"))
    if v.get("exact_iou") is not None:
        rows.append(("exact IoU after", _num(v["exact_iou"], ".4f")))
    known = ("valid", "n_solids", "seconds", "residual_before", "residual_after", "exact_iou")
    rows += [(html.escape(str(k).replace("_", " ")), html.escape(str(x))) for k, x in v.items()
             if k not in known and isinstance(x, (str, int, float, bool))]
    return rows


def _limits_text(limits) -> str:
    return "  ".join(f"{html.escape(str(k).replace('_', ' '))} {html.escape(_num(v) if isinstance(v, float) else str(v))}"
                     for k, v in (limits or {}).items())


def _feature_section(state) -> str:
    """The feature phase (reco_gui.cascade_session.FeatureView): the iteration's difference and limits, the piece in work
    with its guess, score and verdict (the validation numbers, the reason of a rejection), the pieces by state, the
    iterations so far and, once it ended, the summary and which limit ended it."""
    f = getattr(state, "feature", None)
    if f is None:
        return ""
    stage, info = state.step.stage, state.info or {}
    in_phase = stage.startswith("feature_")
    parts = []
    counts = f.counts()
    rows = [("iteration", str(f.iteration)),
            ("<span style='color:#1baf7a'>missing</span>", f"{_num(f.missing_volume)} mm&sup3;"),
            ("<span style='color:#b3261e'>invented</span>", f"{_num(f.invented_volume)} mm&sup3;")]
    if f.input_volume:
        rows.append(("input volume", f"{_num(f.input_volume)} mm&sup3;"))
    if f.exact_iou is not None:
        rows.append(("exact IoU", f"{bar(float(f.exact_iou), '#2a78d6', 60)} <b>{f.exact_iou:.4f}</b>"))
    rows.append(("pieces", f"{len(f.pieces)}: " + ", ".join(
        f"<span style='color:{PIECE_COLOURS[k]}'>{n} {k}</span>" for k, n in counts.items() if n)
        + (f"; dropped: {f.dropped_small} too small, {f.dropped_limit} over the limit"
           if f.dropped_small or f.dropped_limit else "")))
    if f.limits:
        rows.append(("limits in force", _limits_text(f.limits)))
    parts.append(_section("Feature phase", _kv(rows)))

    pid = info.get("piece_id")
    if pid is None and isinstance(info.get("piece"), dict):
        pid = info["piece"].get("id")
    piece = f.piece(pid) if pid is not None else None
    if in_phase and piece is not None:
        rows = [("piece" if piece.parent is None else "segment",
                 f"<b>{piece.id}</b>" + (f" ({f.index} of {f.of})" if f.index and f.of and piece.split is None else "")
                 + (f" of piece <b>{piece.parent}</b>" if piece.parent is not None else "")),
                ("state", f"<b style='color:{PIECE_COLOURS.get(piece.state, '#000')}'>{piece.state}</b>"),
                ("side", f"{html.escape(piece.side)} &rarr; {html.escape(piece.op)}"),
                ("volume", f"{_num(piece.volume)} mm&sup3; ({piece.share:.1%} of the difference)")]
        if piece.bbox is not None:
            lo, hi = piece.bbox
            rows.append(("bounding box", "(" + ", ".join(f"{v:.4g}" for v in lo) + ") .. ("
                         + ", ".join(f"{v:.4g}" for v in hi) + ") mm"))
        if piece.voxels is not None:
            rows.append(("voxels", str(piece.voxels)))
        if piece.mesh is not None:
            rows.append(("mesh", f"{len(piece.mesh.triangles)} triangles" + (" (truncated)" if piece.mesh_truncated else "")))
        split = piece.split or (f.piece(piece.parent).split if piece.parent is not None and f.piece(piece.parent) else None)
        if split:
            segments = f.segments(piece.id if piece.split else piece.parent)
            done = {k: sum(1 for x in segments if x.state == k) for k in ("accepted", "rejected")}
            rows.append(("split", f"by <b>{html.escape(str(split.get('method')))}</b> into {len(split.get('segments') or ())} "
                                  f"segments ({done['accepted']} accepted, {done['rejected']} rejected)"
                                  f"<br><span style='color:#6b6a66'>pitch {_num(split.get('pitch'))} mm, grid "
                                  + (" x ".join(str(n) for n in split["grid"]) if split.get("grid") else "-")
                                  + (f" cells, {_num(split.get('seconds'), '.2f')} s" if split.get("seconds") is not None
                                     else " cells") + "</span>"))
        if piece.segments_accepted:
            rows.append(("segments accepted", ", ".join(str(x) for x in piece.segments_accepted)))
        if stage in ("feature_candidate", "feature_accepted", "feature_rejected"):
            p = info.get("candidate")
            if p is not None:
                rows.append(("guess", f"<b>{html.escape(str(info.get('op') or piece.op))}:{html.escape(str(p.kind))}</b>"
                                      f"<br><span style='color:#6b6a66'>{_dims(p)}<br>{_pose(p)}</span>"))
            else:
                rows.append(("guess", "<span style='color:#6b6a66'>none</span>"))
            if info.get("source"):
                rows.append(("source", f"<span style='color:{_mode_colour(str(info['source']))}'>"
                                       f"{html.escape(str(info['source']))}</span>"))
            if info.get("score") is not None:
                rows.append(("score", f"{bar(float(info['score']), '#eb6834', 60)} {_num(info['score'], '.3f')}"
                             + (f" (needs &ge; {_num(f.limits['min_score'])})"
                                if (f.limits or {}).get("min_score") is not None else "")))
            for name in ("precision", "recall"):
                if info.get(name) is not None:
                    rows.append((name, f"{bar(float(info[name]), '#127a52', 60)} {_num(info[name], '.3f')}"))
            if info.get("seconds") is not None:
                rows.append(("took", f"{_num(info['seconds'], '.2f')} s"))
            if piece.candidates:
                rows.append(("guesses tried", str(piece.candidates)))
        if stage == "feature_rejected":
            rows.append(("decision", "<b style='color:#b3261e'>rejected</b>"))
            rows.append(("reason", html.escape(str(info.get("reason") or "not given"))))
        elif stage == "feature_accepted":
            rows.append(("decision", "<b style='color:#127a52'>accepted</b>"))
        if stage in ("feature_accepted", "feature_rejected"):
            rows += _validation_rows(info.get("validation"))
        parts.append(_section("Piece in work", _kv(rows)))

    if in_phase and f.pieces:
        rows = []
        for p in f.pieces[:MAX_PIECE_ROWS]:                    # a split piece's segments follow it, indented
            text = f"{html.escape(p.side)} &rarr; {html.escape(p.op)}, {_num(p.volume)} mm&sup3; ({p.share:.1%})"
            if p.split:
                text += f", split by {html.escape(str(p.split.get('method')))} into {len(p.split.get('segments') or ())}"
            if p.precision is not None and p.recall is not None:
                text += f", precision {_num(p.precision, '.3f')}, recall {_num(p.recall, '.3f')}"
            if p.primitive is not None or p.score is not None:
                text += (f"<br><span style='color:#6b6a66'>{html.escape(str(getattr(p.primitive, 'kind', '') or ''))}"
                         + (f" from {html.escape(p.source)}" if p.source else "")
                         + (f", score {_num(p.score, '.3f')}" if p.score is not None else "")
                         + (f": {html.escape(p.reason)}" if p.reason else "") + "</span>")
            nest = "&nbsp;&nbsp;&nbsp;&nbsp;&#8627; " if p.parent is not None else ""
            rows.append((f"{nest}<span style='color:{PIECE_COLOURS.get(p.state, '#000')}'>{p.id} {p.state}</span>", text))
        if len(f.pieces) > MAX_PIECE_ROWS:
            rows.append(("...", f"{len(f.pieces) - MAX_PIECE_ROWS} more"))
        parts.append(_section("Pieces of this iteration", _kv(rows)))

    if f.history:
        rows = [(f"iteration {h.get('iteration')}",
                 f"missing {_num(h.get('missing_volume'))}, invented {_num(h.get('invented_volume'))} mm&sup3;, exact IoU "
                 f"{_num(h.get('exact_iou'), '.4f')}; {h.get('pieces', 0)} pieces: {h.get('applied', 0)} applied, "
                 f"{h.get('rejected', 0)} rejected, {h.get('skipped', 0)} skipped") for h in f.history]
        parts.append(_section("Feature iterations", _kv(rows)))
    d = f.done
    if d is not None:
        rows = [("iterations", str(d.get("iterations", f.iteration))),
                ("operations", f"<b>{d.get('applied', f.applied)} applied</b>, {d.get('rejected', f.rejected)} rejected, "
                               f"{d.get('skipped', 0)} skipped"),
                ("stopped because", f"<b>{html.escape(str(d.get('stopped') or 'not given'))}</b>"),
                ("took", f"{_num(d.get('seconds'), '.1f')} s"),
                ("exact IoU", _num(d.get("exact_iou"), ".4f")),
                ("left", f"missing {_num(d.get('missing_volume'))}, invented {_num(d.get('invented_volume'))} mm&sup3;")]
        if d.get("split_pieces") is not None or d.get("segments") is not None:
            rows.append(("split", f"{d.get('split_pieces', 0)} piece(s) into {d.get('segments', 0)} segments, "
                                  f"{d.get('segments_accepted', 0)} accepted"))
        if f.limits:
            rows.append(("limits in force", _limits_text(f.limits)))
        parts.append(_section("Feature phase: summary", _kv(rows)))
    return "".join(parts)


def feature_log_lines(state) -> list:
    """The verbose lines of a feature step for the Output panel, after its one-line message: the pieces and the limits of
    an iteration, the numbers of a verdict's FreeCAD build, the summary."""
    f = getattr(state, "feature", None)
    stage = state.step.stage
    if f is None or not stage.startswith("feature_"):
        return []
    info = state.info or {}
    lines = []
    if stage == "feature_residual":
        if f.limits:
            lines.append("limits: " + ", ".join(f"{k} {v}" for k, v in f.limits.items()))
        if f.dropped_small or f.dropped_limit:
            lines.append(f"dropped: {f.dropped_small} too small, {f.dropped_limit} over the piece limit")
        for p in f.pieces[:12]:
            lines.append(f"piece {p.id:3d}  {p.side:8s} -> {p.op:3s}  {p.volume:12.4g} mm^3  {p.share:6.1%}")
        if len(f.pieces) > 12:
            lines.append(f"... and {len(f.pieces) - 12} more piece(s)")
    elif stage == "feature_split":
        for p in f.segments(info.get("piece_id"))[:12]:
            lines.append(f"segment {p.id:3d}  {p.voxels if p.voxels is not None else '?':>8} voxels  "
                         f"{p.volume:12.4g} mm^3  {p.share:6.1%}")
    elif stage in ("feature_accepted", "feature_rejected"):
        parent = f.piece(info.get("piece_id"))
        if parent is not None and parent.split:                 # the split piece's own verdict: its segments summed up
            segments = f.segments(parent.id)
            counts = {k: sum(1 for x in segments if x.state == k) for k in ("accepted", "rejected")}
            rest = len(segments) - counts["accepted"] - counts["rejected"]
            lines.append(f"piece #{parent.id} split by {parent.split.get('method')} into {len(segments)} segments: "
                         f"{counts['accepted']} accepted, {counts['rejected']} rejected"
                         + (f", {rest} without a verdict" if rest else ""))
        v = info.get("validation")
        if isinstance(v, dict):
            lines.append("FreeCAD build: " + ", ".join(f"{k} {_num(x) if isinstance(x, float) else x}" for k, x in v.items()))
        if stage == "feature_rejected" and info.get("reason"):
            lines.append(f"reason: {info['reason']}")
    elif stage == "feature_done":
        for h in f.history:
            lines.append(f"iteration {h.get('iteration')}: exact IoU {_num(h.get('exact_iou'), '.4f')}, missing "
                         f"{_num(h.get('missing_volume'))}, invented {_num(h.get('invented_volume'))} mm^3, "
                         f"{h.get('applied', 0)} applied / {h.get('rejected', 0)} rejected / {h.get('skipped', 0)} skipped")
    return lines


def _wall_grid_section(wall) -> str:
    """The loop's voxel pitch against the input's wall thickness (reco_gui.cascade_session.wall_grid_voxels)."""
    if not wall:
        return ""
    if wall.get("error"):
        return _section("Voxel grid vs walls", f"<span style='color:#6b6a66'>not computed: {html.escape(wall['error'])}</span>")
    loop, tau = wall.get("loop_pitch"), wall["tau_p5"]
    rows = [("wall thickness tau", f"{tau:.4g} (5th percentile), median {wall['tau_median']:.4g}")]
    if loop is not None:
        ok = loop <= tau
        colour = "#127a52" if ok else "#b3261e"
        rows.append(("loop's pitch", f"<b style='color:{colour}'>{loop:.4g} = {loop / tau:.1f} x tau</b>"
                     + ("" if ok else "<br><span style='color:#b3261e'>coarser than the thin walls: its voxels alias them "
                                      "(walls vanish or become whole voxel slabs; D-005)</span>")))
    shape = " x ".join(str(n) for n in wall["shape"])
    rows.append(("'Voxels at \u03c4' layer", f"pitch {wall['pitch']:.4g}" + (" (coarsened: too many voxels)" if wall["capped"] else "")
                 + f", {shape} grid, {wall['voxels']} voxels, {wall['volume_ratio']:.2f} of the mesh volume, "
                 f"{wall['seconds']:.1f} s"))
    return _section("Voxel grid vs walls", _kv(rows))


def render_cascade_state(state) -> str:
    """HTML for the detail panel in the cascade (add-cut) mode (a reco_gui.cascade_session.CascadeState)."""
    step = state.step
    parts = [f"<p style='margin:0 0 4px 0;'>Step {state.index + 1}: <b>{html.escape(step.stage)}</b><br>"
             f"<span style='color:#3d3c39;'>{html.escape(step.message)}</span></p>"]
    rows = [("input", html.escape(Path(state.input_path).name) if state.input_path else ""),
            ("<span style='color:#1baf7a'>missing material</span>", f"{state.residual_pos} voxels (green)"),
            ("<span style='color:#b3261e'>excess material</span>", f"{state.residual_neg} voxels (red)")]
    if state.component_voxels:
        rows.append(("<span style='color:#b38f00'>current component</span>", f"{state.component_voxels} voxels (yellow)"))
    parts.append(_section("Residual", _kv(rows)))
    if step.stage == "input" or not state.program:
        parts.append(_type_stage_section(getattr(state, "type_stage", None)))
        parts.append(_wall_grid_section(getattr(state, "wall_grid", None)))
    member = _member_section(state)
    if member:
        parts.append(member)
    parts.append(_solution_section(state))
    parts.append(_feature_section(state))
    if state.program:
        rows = []
        for i, o in enumerate(state.program):
            p = o.primitive
            extra = f", net gain {o.net_gain:.0f} vox" if o.net_gain else ""
            if getattr(o, "recut_voxels", 0):
                extra += f", {o.recut_voxels} vox cut again"
            if o.freecad_valid is not None:
                extra += ", FreeCAD " + ("<b style='color:#127a52'>valid</b>" if o.freecad_valid else "<b style='color:#b3261e'>INVALID</b>")
            if getattr(p, "label", None):
                extra += f"<br><span style='color:#b38f00'>{html.escape(p.label)}</span>"
            rows.append((f"{i + 1}. <b>{o.op}:{html.escape(p.kind)}</b>",
                         f"conf {p.confidence:.2f}{extra}<br><span style='color:#6b6a66'>{_dims(p)}<br>{_pose(p)}</span>"
                         + _provenance(p) + _alternatives(p)))
        parts.append(_section("Program", _kv(rows)))
    c = state.candidate
    if c is not None and not (state.verdict == "accepted" and step.stage in ("accepted", "add", "feature_accepted")):
        rows = [("operation", f"<b>{html.escape(str(state.candidate_op))}:{html.escape(c.kind)}</b>"
                              + (" <span style='color:#7b3fb3'>(voxel-cube fallback)</span>" if state.candidate_fallback else "")),
                ("dimensions", _dims(c)), ("pose", _pose(c)), ("confidence", f"{c.confidence:.2f}")]
        info = state.info or {}
        if getattr(c, "label", None):
            rows.insert(1, ("name", f"<span style='color:#b38f00'>{html.escape(c.label)}</span>"))
        if state.net_gain is not None:
            exact = info.get("precision") is not None
            rows.append(("net gain", f"{state.net_gain:.1f} voxel volumes (exact, from the mesh)" if exact
                         else f"{state.net_gain:.0f} voxels"))
        if info.get("precision") is not None:
            what = "in the input" if state.candidate_op == "add" else "of the removal outside the input"
            rows.append(("precision", f"{bar(float(info['precision']), '#127a52', 60)} {float(info['precision']):.1%} {what}"))
        if info.get("grid_gain") is not None:
            rows.append(("grid gain", f"{info['grid_gain']} voxels (walls thinner than the pitch gain ~0 here)"))
        if info.get("by_congruency"):
            rows.append(("by congruency", f"member {int(info.get('source_member', 0)) + 1}'s primitive moved here "
                                          f"(group {html.escape(str(info.get('group')))})"))
        if step.stage in ("candidate", "ibeam_box", "member_op") and info.get("ok") is not None:
            rows.append(("passes", ("yes" if info["ok"] else "no") + (f" ({html.escape(str(info['reason']))})" if info.get("reason") else "")))
        if state.verdict:
            colour = "#127a52" if state.verdict == "accepted" else "#b3261e"
            rows.insert(1, ("decision", f"<b style='color:{colour}'>{state.verdict}</b>"))
        parts.append(_section("Candidate", _kv(rows) + _section("Provenance", _provenance(c)) + _alternatives(c)))
    r = state.result
    if r is not None and r.chamfer_mm is not None:
        parts.append(_section("Grade against the input", _kv([("Chamfer (both ways)", f"{r.chamfer_mm:.4g} mm")])))
    if state.finished and r is not None:
        rows = [("operations", f"{len(state.program)} accepted, {len(r.operations) - len(state.program)} rejected"),
                ("residual", f"+{r.residual_voxels_pos} / -{r.residual_voxels_neg} voxels"),
                ("cut again", f"{getattr(r, 'recut_voxels', 0)} voxels (cut by two or more cuts)")]
        if r.step_path:
            rows.append(("STEP", html.escape(str(r.step_path))))
        parts.append(_section("Result", _kv(rows)))
    return "".join(parts)


def render_state(state, settings=None) -> str:
    """HTML for the detail panel: a vector-field state, a cascade state, or nothing yet. `settings` is accepted for the
    callers' sake (the voxel mode that read it was removed on 2026-09-30)."""
    if state is not None and hasattr(state, "vectors"):
        return render_field_state(state)
    if state is not None and hasattr(state, "program"):
        return render_cascade_state(state)
    if state is None:
        return "<p style='color:#6b6a66'>No step yet. Choose a part and press <b>Start</b>.</p>"
    step = state.step
    return (f"<p style='margin:0 0 4px 0;'>Step {state.index + 1}: <b>{html.escape(step.stage)}</b><br>"
            f"<span style='color:#3d3c39;'>{html.escape(step.message)}</span></p>")


def _cell(value) -> str:
    if value is None:
        return "<span style='color:#6b6a66'>-</span>"
    if isinstance(value, float):
        return f"{value:.6g}"
    text = str(value)
    return html.escape(text if len(text) <= 60 else text[:57] + "...")


def _table(header, rows) -> str:
    head = "".join(f"<th align='left' style='padding-right:12px;color:#6b6a66'>{html.escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td style='padding-right:12px'>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table cellspacing='0' cellpadding='1'><tr>{head}</tr>{body}</table>"


def render_run_diff(diff, run=None) -> str:
    """HTML of reco_engine.run_store.diff_runs(a = the earlier run, b = the run just saved): the settings that differ,
    the changed metrics side by side, the programs operation by operation. `diff` None: there is no earlier run."""
    if diff is None:
        name = html.escape(str((run or {}).get("name", "")))
        return (f"<p><b>Compare with previous</b><br>{name}: the store has no earlier run of this input "
                f"(the same file by content).</p>")

    def code(c):
        c = c or {}
        return (f"{html.escape(str(c.get('short')))} ({html.escape(str(c.get('branch')))})"
                + (f" <b style='color:#b3261e'>dirty: {len(c.get('modified') or [])} modified file(s)</b>" if c.get("dirty") else ""))
    parts = ["<p style='margin:0 0 4px 0;'><b>Compare with previous</b></p>"]
    rows = [(f"{side} ({what})", f"{html.escape(str(r.get('name')))}<br><span style='color:#6b6a66'>"
                                 f"{html.escape(str(r.get('timestamp')))}, code {code(r.get('code'))}"
                                 + ("" if r.get("finished") else ", <b>not finished</b>") + "</span>")
            for side, what, r in (("a", "previous", diff["a"]), ("b", "this run", diff["b"]))]
    rows.append(("input", "the same file" if diff["same_input"] else "<b style='color:#b3261e'>different files</b>"))
    parts.append(_section("Runs", _kv(rows)))
    parts.append(_section(f"Settings that differ ({len(diff['settings'])})",
                          _table(["setting", "a", "b"], [[html.escape(x["key"]), _cell(x["a"]), _cell(x["b"])]
                                                         for x in diff["settings"]]) if diff["settings"] else "none"))
    changed = [m for m in diff["metrics"] if m["changed"]]
    parts.append(_section(f"Metrics that changed ({len(changed)} of {len(diff['metrics'])})",
                          _table(["metric", "a", "b", "b - a"],
                                 [[html.escape(m["key"]), _cell(m["a"]), _cell(m["b"]),
                                   f"{m['difference']:+.6g}" if m["difference"] is not None else ""] for m in changed])
                          if changed else "none"))
    s = diff["summary"]
    rows = []
    for o in diff["operations"]:
        colour = {"same": "#6b6a66", "changed": "#c2600a"}.get(o["status"], "#b3261e")
        rows.append([str(o["index"] + 1), f"<span style='color:{colour}'>{html.escape(o['status'])}</span>",
                     _cell(o["a"]), _cell(o["b"]), ""])
        rows += [["", "", f"&nbsp;&nbsp;{html.escape(p['key'])}", f"{_cell(p['a'])} &rarr; {_cell(p['b'])}",
                  f"{p['difference']:+.6g}" if p["difference"] is not None else ""] for p in o["parameters"]]
    parts.append(_section(f"Program: {s['operations_a']} operation(s) before, {s['operations_b']} now, "
                          f"{s['operations_changed']} differ",
                          _table(["#", "status", "a", "b", "b - a"], rows) if rows else "no operations"))
    return "".join(parts)


COLLAPSED_STAGES = ("candidate", "feature_candidate")        # steps the Output panel sums up in one line per component


def _reason_key(reason: str) -> str:
    """A reason without its numbers: 'only 4% of the nearby surface ...' and 'only 7% of ...' are the same reason."""
    return re.sub(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "#", str(reason)).strip()


def candidates_summary(states) -> str:
    """One line for the candidates of a component / a feature piece (the states of their steps, in order): how many, the
    best one (a passing one with the largest gain, or the highest score in the feature phase), and the reasons of the
    others counted - '47 candidates: best cut cylinder gain 39321 (47 % of the nearby surface); 21 x failed the void
    check, 14 x only #% of the nearby surface ...'. Reasons that differ only in their numbers are counted together and
    shown with '#' for the numbers."""
    rows = []
    for st in states:
        info = getattr(st, "info", None) or {}
        p = info.get("candidate")
        rows.append(dict(op=info.get("op") or "", kind=getattr(p, "label", None) or getattr(p, "kind", "?"),
                         gain=info.get("net_gain"), score=info.get("score"), surface=info.get("surface"),
                         precision=info.get("precision"), recall=info.get("recall"),
                         ok=info.get("ok"), reason=str(info.get("reason") or ""), source=info.get("source")))
    if not rows:
        return ""
    n = len(rows)
    text = f"{n} candidate{'s' if n != 1 else ''}"
    scored = [r for r in rows if r["score"] is not None]
    passing = [r for r in rows if r["ok"] and r["gain"] is not None]
    if scored:
        b = max(scored, key=lambda r: float(r["score"]))
        text += f": best {b['op']} {b['kind']}" + (f" from {b['source']}" if b["source"] else "") + f" score {float(b['score']):.3f}"
        if b["precision"] is not None and b["recall"] is not None:
            text += f" (precision {float(b['precision']):.3f}, recall {float(b['recall']):.3f})"
    elif passing:
        b = max(passing, key=lambda r: float(r["gain"]))
        text += (f": best {b['op']} {b['kind']} gain {float(b['gain']):.0f}"
                 + (f" ({float(b['surface']):.0%} of the nearby surface)" if b["surface"] is not None else ""))
    else:
        text += ": none passes"
    counts = {}
    for r in rows:
        if r["ok"] or (r["ok"] is None and not r["reason"]):
            continue
        key = _reason_key(r["reason"]) or "no reason given"
        counts[key] = counts.get(key, 0) + 1
    if passing and not scored and len(passing) > 1:
        text += f"; {len(passing)} pass"
    if counts:
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        text += "; " + ", ".join(f"{c} x {k if len(k) <= 70 else k[:67] + '...'}" for k, c in ranked[:6])
        if len(ranked) > 6:
            text += f", {sum(c for _, c in ranked[6:])} x other reasons"
    return text
