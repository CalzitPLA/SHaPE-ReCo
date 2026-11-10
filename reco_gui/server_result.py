#!/usr/bin/env python3
"""
The results of a server run in the window (the user, 2026-10-29: "the results need to be accessible inside the gui"):
after a followed run has ended its files are fetched (scripts/remote_run.py fetch: record.json, program.json, rating.json,
model.stl into <mirror>/run) and shown - the exact model over the part in the 3D view, and in the detail panel the
record (exact and sampled IoU, one valid solid, operations by kind and source, why the run stopped, seconds), the
program, the rating against the known tree (tier, the IoU after every operation, the separation point and the defect
classes) and, when scripts/detail_study.py compared this run, the truth leaves against the nearest reconstructed
primitive and the steps where the software lacked.

    html, info = result_html(mirror)          # info: dict(model_stl, caption, lines) for the 3D view and the log
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DETAIL = ROOT / "docs" / "data" / "detail"


def _read(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _f(x, nd: int = 4) -> str:
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else html.escape(str(x)))


def fetch(mirror) -> Path:
    """The finished run's files into <mirror>/run (remote_run.cmd_fetch); returns that directory."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("remote_run", ROOT / "scripts" / "remote_run.py")
    rr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rr)
    rr.cmd_fetch(type("A", (), {"mirror": str(mirror)})())
    return Path(mirror).expanduser() / "run"


def detail_for(run_dir: Path) -> Tuple[Optional[dict], Optional[dict]]:
    """(compare.json, lacks.json) of the detail study that analysed this run directory, if one did."""
    for cmp_path in sorted(DETAIL.glob("*/compare*.json")):
        cmp = _read(cmp_path)
        if cmp and Path(cmp.get("run", "")).resolve() == Path(run_dir).resolve():
            return cmp, _read(cmp_path.parent / cmp_path.name.replace("compare", "lacks"))
    return None, None


def result_html(mirror) -> Tuple[str, Dict]:
    mirror = Path(mirror).expanduser()
    run = mirror / "run"
    link = _read(mirror / "link.json") or {}
    rec = _read(run / "record.json") or {}
    prog = _read(run / "program.json") or {}
    rating = _read(run / "rating.json")
    cmp, lacks = detail_for(run)
    comp = rec.get("comparison") or {}
    ex, sa = comp.get("exact") or {}, comp.get("sampled") or {}
    model = rec.get("model") or {}
    secs = rec.get("seconds") or {}
    tier = ((rating or {}).get("grade") or {}).get("tier")
    caption = (f"server run {link.get('id', mirror.name)} on {link.get('host', rec.get('host', '?'))}: exact IoU "
               f"{_f(ex.get('iou'))}" + (f", tier {tier}" if tier else "")
               + (", one valid solid" if model.get("one_valid_solid") else f", {model.get('solids')} solids"))
    H = [f"<h3>{html.escape(caption)}</h3>",
         "<table border='0' cellspacing='2'>",
         f"<tr><td>input</td><td>{html.escape(str(link.get('step') or rec.get('input')))}</td></tr>",
         f"<tr><td>host / code</td><td>{html.escape(str(rec.get('host')))} / {html.escape(str(rec.get('commit')))}</td></tr>",
         f"<tr><td>exact IoU / sampled</td><td>{_f(ex.get('iou'))} (missing {_f(ex.get('missing'), 1)}, invented "
         f"{_f(ex.get('invented'), 1)} mm&sup3;) / {_f(sa.get('iou'))}</td></tr>",
         f"<tr><td>model</td><td>{model.get('operations')} operations, by kind {html.escape(str(model.get('by_kind')))}, "
         f"by source {html.escape(str(model.get('by_source')))}, valid {model.get('valid')}, solids {model.get('solids')}</td></tr>",
         f"<tr><td>stopped</td><td>{html.escape(str((rec.get('run') or {}).get('stopped')))}</td></tr>",
         f"<tr><td>seconds</td><td>" + ", ".join(f"{html.escape(k)} {v:.0f}" for k, v in secs.items() if isinstance(v, (int, float)))
         + "</td></tr></table>"]
    lines = [f"--- server run result: {caption}", f"    stopped: {(rec.get('run') or {}).get('stopped')}"]
    if rating:
        att = rating.get("attribution") or {}
        sep = att.get("separation") or {}
        traj = ((rating.get("vector") or {}).get("trajectory")) or []
        H += [f"<h4>Rating against the known tree: tier {html.escape(str(tier))}</h4>",
              f"<p>separation point: <b>{html.escape(str(sep.get('cls')))}</b>, {100 * (sep.get('volume_share') or 0):.2f} % of "
              f"the volume, caused by operation {sep.get('culprit')} at step {(sep.get('event') or {}).get('index')}: "
              f"{html.escape(str(sep.get('cause')))}</p>",
              "<p>defect classes (share of the volume): " + ", ".join(
                  f"{html.escape(k)} {100 * v:.3f} %" for k, v in (att.get("classes") or {}).items()) + "</p>"]
        lines.append(f"    separation point: {sep.get('cls')} ({100 * (sep.get('volume_share') or 0):.2f} % of the volume), "
                     f"operation {sep.get('culprit')}: {sep.get('cause')}")
        ops = prog.get("operations") or []
        H.append("<table border='1' cellspacing='0' cellpadding='2'><tr><th>#</th><th>op</th><th>kind</th><th>label</th>"
                 "<th>IoU after</th><th>change</th></tr>")
        prev = 0.0
        for i, o in enumerate(ops):
            p = o.get("primitive") or {}
            iou = traj[i]["iou"] if i < len(traj) else None
            H.append(f"<tr><td>{i}</td><td>{html.escape(str(o.get('op')))}</td><td>{html.escape(str(p.get('kind')))}</td>"
                     f"<td>{html.escape(str(p.get('label') or ''))[:70]}</td><td>{_f(iou)}</td>"
                     f"<td>{'' if iou is None else f'{iou - prev:+.4f}'}</td></tr>")
            prev = iou if iou is not None else prev
        H.append("</table>")
    if cmp:
        H.append("<h4>Against the three trees of the detail study</h4><table border='1' cellspacing='0' cellpadding='2'>"
                 "<tr><th>tree</th><th>tier</th><th>volume IoU</th><th>Hausdorff/D</th><th>leaves matched</th></tr>")
        for k, v in (cmp.get("trees") or {}).items():
            t = v.get("tree") or {}
            H.append(f"<tr><td>{k}</td><td>{v.get('tier')}</td><td>{_f((v.get('volume') or {}).get('iou'))}</td>"
                     f"<td>{_f(v.get('hausdorff'))}</td><td>{t.get('matched')}/{t.get('truth_leaves')} "
                     f"(model {t.get('model_leaves')})</td></tr>")
        H.append("</table><p>T1's leaves against the nearest reconstructed primitive of their kind:</p>"
                 "<table border='1' cellspacing='0' cellpadding='2'><tr><th>leaf</th><th>kind</th><th>op</th>"
                 "<th>centre mm</th><th>size mm</th><th>axis deg</th><th>matched</th></tr>")
        for x in ((cmp.get("trees") or {}).get("T1") or {}).get("leaves") or []:
            H.append(f"<tr><td>{html.escape(str(x.get('label')))}</td><td>{x.get('kind')} ({x.get('role')})</td>"
                     f"<td>{x.get('nearest_op')}</td><td>{_f(x.get('centre_mm'), 2)}</td><td>{_f(x.get('size_mm'), 2)}</td>"
                     f"<td>{_f(x.get('axis_deg'), 1)}</td><td>{'yes' if (x.get('cost') or 9) <= 1 else 'no'}</td></tr>")
        H.append("</table>")
    if lacks:
        H.append("<h4>Where the software lacked</h4><table border='1' cellspacing='0' cellpadding='2'><tr><th>defect</th>"
                 "<th>share</th><th>op</th><th>step</th><th>stage</th><th>cause</th></tr>")
        for r in (lacks.get("regions") or [])[:12]:
            H.append(f"<tr><td>{html.escape(str(r.get('cls')))}</td><td>{100 * (r.get('volume_share') or 0):.3f} %</td>"
                     f"<td>{r.get('culprit')}</td><td>{r.get('step')}</td><td>{html.escape(str(r.get('stage')))}</td>"
                     f"<td>{html.escape(str(r.get('cause')))}{' [decisive]' if r.get('decisive') else ''}</td></tr>")
        H.append("</table>")
    info = dict(model_stl=str(run / "model.stl") if (run / "model.stl").is_file() else None, caption=caption, lines=lines)
    return "\n".join(H), info
