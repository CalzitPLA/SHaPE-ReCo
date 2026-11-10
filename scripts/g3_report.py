#!/usr/bin/env python3
"""
Gate G3 report: two runs of scripts/g3_run.py (or of the 2026-09-24 ad-hoc night runner, docs/data/night_g3/<run>)
compared part by part, written as JSON + Markdown + LaTeX macros (the thesis \\input's the macros; no number is
typed by hand).

    python3 scripts/g3_report.py --base docs/data/night_g3/new_fix --new docs/data/g3/s6full \\
        --json docs/data/g3_report.json --md docs/data/g3_report.md --tex docs/data/g3_macros.tex

Verdict per part (the rule of scripts/s6_g3_compare.py and docs/data/night_g3_eval.md): chamfer new > 1.5 x base
+ 0.05 -> worse, base > 1.5 x new + 0.05 -> better, else same; "n/a" when either run has no chamfer (failed, timed
out, nothing built), "missing" when a run has no record of the part. The chamfer is in mm (D-008: the frame STL's
units x 100) unless --chamfer-unit raw, which reproduces the night's numbers exactly (the +0.05 is then in each
file's own unit). Exact IoU: OCCT and sampled values are counted separately, never pooled.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3_run import UNIT_TABLE, mm_per_unit  # noqa: E402

FRAME_PATTERN = "Frame_4_x_6"
VERDICT_ORDER = {"worse": 0, "better": 1, "missing": 2, "n/a": 3, "same": 4}


def _num(x):
    return x if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def verdict(a: Optional[float], b: Optional[float], rel: float = 1.5, abs_: float = 0.05) -> str:
    """base chamfer `a`, new chamfer `b` (lower is better)."""
    if a is None or b is None:
        return "n/a"
    if b > rel * a + abs_:
        return "worse"
    if a > rel * b + abs_:
        return "better"
    return "same"


def iou_verdict(a: Optional[float], b: Optional[float], method_a, method_b, tol: float = 0.01) -> str:
    """Secondary verdict on the exact IoU (higher is better), only when both runs measured it by OCCT: the chamfer
    alone is blind to wrong volume (s6: s04_tube chamfer 6.1 mm "same", IoU 0.21). Sampled values are accurate to a
    few percent only (reco_field.exact_check.sampled_iou), so they get no verdict."""
    if a is None or b is None or method_a != "occt" or method_b != "occt":
        return "n/a"
    return "worse" if b < a - tol else "better" if b > a + tol else "same"


def normalize(rec: Dict) -> Dict:
    """One part record of either format as the same flat dict (lengths in mm, volumes in mm^3)."""
    res = rec.get("result") or {}
    ok = bool(rec.get("ok"))
    if rec.get("schema", "").startswith("g3.part/"):
        n = {k: rec.get(k) for k in ("n_ops", "freecad_valid", "chamfer_units", "chamfer_mm", "exact_iou",
                                     "exact_method", "exact_error", "sampled_iou", "r_plus_mm3", "r_minus_mm3",
                                     "genus", "input_genus", "genus_match", "grid_pitch_over_tau", "cpu_seconds",
                                     "wall_seconds", "commit", "dirty", "commit_drift", "mm_per_unit", "timeout", "error")}
        n["unit_source"] = "measured extent"
    else:                                                  # the ad-hoc night runner: chamfer_mm is in file units
        part = rec.get("part") or res.get("input_path") or ""
        scale = mm_per_unit(part, None)
        ch = _num(rec.get("chamfer_mm"))
        rp, rm = _num(res.get("exact_r_plus")), _num(res.get("exact_r_minus"))
        g, ig = _num(res.get("genus")), _num(res.get("input_genus"))
        n = dict(n_ops=rec.get("n_ops"), freecad_valid=rec.get("freecad_valid"), chamfer_units=ch,
                 chamfer_mm=None if ch is None else ch * scale, exact_iou=_num(res.get("exact_iou")),
                 exact_method=res.get("exact_method"), exact_error=None, sampled_iou=None,
                 r_plus_mm3=None if rp is None else rp * scale ** 3, r_minus_mm3=None if rm is None else rm * scale ** 3,
                 genus=g, input_genus=ig, genus_match=None if g is None or ig is None else round(g) == round(ig),
                 grid_pitch_over_tau=_num(res.get("grid_pitch_over_tau")), cpu_seconds=None,
                 wall_seconds=rec.get("seconds"), commit=None, dirty=None, commit_drift=None, mm_per_unit=scale,
                 timeout=None, error=rec.get("error"))
        n["unit_source"] = "UNIT_TABLE by name" if Path(str(part)).name in UNIT_TABLE else "assumed mm"
    n["ok"] = ok
    if not ok:
        n["chamfer_mm"] = n["chamfer_units"] = None
    return n


def load_run(d: Path) -> Dict[str, Dict]:
    out = {}
    for p in sorted(Path(d).glob("*__*.json")):
        out[p.stem] = normalize(json.loads(p.read_text()))
    return out


def iou_kind(method) -> Optional[str]:
    if not method:
        return None
    return "occt" if method == "occt" else "sampled"


def summary(run: Dict[str, Dict]) -> Dict:
    rows = list(run.values())
    occt = [r["exact_iou"] for r in rows if r["ok"] and iou_kind(r["exact_method"]) == "occt" and r["exact_iou"] is not None]
    samp = [r["exact_iou"] for r in rows if r["ok"] and iou_kind(r["exact_method"]) == "sampled" and r["exact_iou"] is not None]
    gen = [r for r in rows if r["genus_match"] is not None]
    cpu = [r["cpu_seconds"] for r in rows if r["cpu_seconds"] is not None]
    wall = [r["wall_seconds"] for r in rows if r["wall_seconds"] is not None]
    commits = sorted({r["commit"] for r in rows if r["commit"]})
    return dict(parts=len(rows), ok=sum(r["ok"] for r in rows), failed=sum(not r["ok"] and not r["timeout"] for r in rows),
                timeout=sum(bool(r["timeout"]) for r in rows), valid=sum(bool(r["freecad_valid"]) for r in rows),
                genus_match=sum(bool(r["genus_match"]) for r in gen), genus_known=len(gen),
                iou_occt_n=len(occt), iou_occt_median=statistics.median(occt) if occt else None,
                iou_occt_min=min(occt) if occt else None, iou_sampled_n=len(samp),
                iou_sampled_median=statistics.median(samp) if samp else None,
                cpu_seconds=sum(cpu) if cpu else None, cpu_known=len(cpu), wall_seconds=sum(wall) if wall else None,
                commits=commits, dirty=any(bool(r["dirty"]) for r in rows),
                commit_drift=sum(bool(r["commit_drift"]) for r in rows),
                mixed_commits=len(commits) > 1)


def compare(base: Dict[str, Dict], new: Dict[str, Dict], chamfer_unit: str = "mm", rel: float = 1.5,
            abs_: float = 0.05, iou_tol: float = 0.01) -> Dict:
    key = "chamfer_mm" if chamfer_unit == "mm" else "chamfer_units"
    rows = []
    for k in sorted(set(base) | set(new)):
        a, b = base.get(k), new.get(k)
        v = "missing" if a is None or b is None else verdict(a[key], b[key], rel, abs_)
        a, b = a or {}, b or {}
        rows.append(dict(key=k, part=k.split("__", 1)[-1], verdict=v,
                         ops_base=a.get("n_ops"), ops_new=b.get("n_ops"),
                         chamfer_base=a.get(key), chamfer_new=b.get(key),
                         valid_base=a.get("freecad_valid"), valid_new=b.get("freecad_valid"),
                         iou_base=a.get("exact_iou"), iou_new=b.get("exact_iou"),
                         iou_method_base=a.get("exact_method"), iou_method_new=b.get("exact_method"),
                         genus_new=b.get("genus"), input_genus=b.get("input_genus"), genus_match_new=b.get("genus_match"),
                         pitch_over_tau_new=b.get("grid_pitch_over_tau"), unit_source=b.get("unit_source") or a.get("unit_source"),
                         mm_per_unit=b.get("mm_per_unit") or a.get("mm_per_unit"),
                         cpu_base=a.get("cpu_seconds"), cpu_new=b.get("cpu_seconds"),
                         error_new=b.get("error"), exact_error_new=b.get("exact_error")))
        rows[-1]["iou_verdict"] = iou_verdict(a.get("exact_iou"), b.get("exact_iou"), a.get("exact_method"),
                                              b.get("exact_method"), iou_tol)
    return dict(chamfer_unit=chamfer_unit, rule=f"worse if new > {rel} x base + {abs_}, better if base > {rel} x new + {abs_}",
                counts=dict(Counter(r["verdict"] for r in rows)),
                iou_rule=f"OCCT on both sides only: worse if new < base - {iou_tol}, better if new > base + {iou_tol}",
                iou_counts=dict(Counter(r["iou_verdict"] for r in rows)), rows=rows)


def fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def markdown(cmp: Dict, sb: Dict, sn: Dict, base_label: str, new_label: str) -> str:
    u = "mm" if cmp["chamfer_unit"] == "mm" else "file units"
    L = [f"# Gate G3: {new_label} against {base_label}", "",
         f"Verdicts ({cmp['rule']}, chamfer in {u}): {cmp['counts']}.",
         f"IoU verdicts ({cmp['iou_rule']}): {cmp['iou_counts']}.", ""]
    for lab, s in ((base_label, sb), (new_label, sn)):
        L.append(f"- **{lab}**: {s['ok']}/{s['parts']} ok ({s['failed']} failed, {s['timeout']} timed out), FreeCAD-valid "
                 f"{s['valid']}/{s['parts']}, genus of the result = input's on {s['genus_match']}/{s['genus_known']} with "
                 f"both known, exact IoU by OCCT on {s['iou_occt_n']} parts (median {fmt(s['iou_occt_median'], 4)}, min "
                 f"{fmt(s['iou_occt_min'], 4)}), sampled on {s['iou_sampled_n']}; CPU "
                 f"{fmt(s['cpu_seconds'], 5)} s over {s['cpu_known']} parts, wall {fmt(s['wall_seconds'], 5)} s; commits "
                 f"{[c[:8] for c in s['commits']] or 'not recorded'}{' (DIRTY)' if s['dirty'] else ''}"
                 f"{' (MIXED COMMITS)' if s['mixed_commits'] else ''}"
                 + (f" (HEAD moved during {s['commit_drift']} parts)" if s["commit_drift"] else "") + ".")
    L += ["", f"| part | verdict | IoU verdict | ops base / new | chamfer base / new ({u}) | exact IoU base / new (method) | genus (input) | pitch / tau | valid base / new |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(cmp["rows"], key=lambda r: (VERDICT_ORDER[r["verdict"]], r["part"])):
        g = "–" if r["genus_new"] is None and r["input_genus"] is None else f"{fmt(r['genus_new'])} ({fmt(r['input_genus'])})"
        m = r["iou_method_new"] or "–"
        L.append(f"| {r['part']} | {r['verdict']} | {r['iou_verdict']} | {fmt(r['ops_base'])} / {fmt(r['ops_new'])} | {fmt(r['chamfer_base'])} / "
                 f"{fmt(r['chamfer_new'])} | {fmt(r['iou_base'], 4)} / {fmt(r['iou_new'], 4)} ({m}) | {g} | "
                 f"{fmt(r['pitch_over_tau_new'])} | {fmt(r['valid_base'])} / {fmt(r['valid_new'])} |")
    notes = [r for r in cmp["rows"] if r["error_new"] or r["exact_error_new"]]
    if notes:
        L += ["", "Errors in the new run:", ""] + [f"- {r['part']}: {r['error_new'] or r['exact_error_new']}" for r in notes]
    return "\n".join(L) + "\n"


def _tex_escape(s: str) -> str:
    return re.sub(r"([_%&#$])", r"\\\1", s)


def _tex_num(x, nd=4, kind="g") -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{nd}f}" if kind == "f" else f"{x:.{nd}g}"


def macros(cmp: Dict, sb: Dict, sn: Dict, base_label: str, new_label: str, prefix: str = "Gthree") -> str:
    """\\newcommand lines; names are letters only (TeX control words)."""
    m: Dict[str, str] = {}
    m["Run"], m["BaseRun"] = _tex_escape(new_label), _tex_escape(base_label)
    m["Commit"] = sn["commits"][0][:8] if len(sn["commits"]) == 1 else ("mixed" if sn["commits"] else "n/a")
    m["Dirty"] = _tex_num(sn["dirty"])
    for k, name in (("parts", "Parts"), ("ok", "PartsOk"), ("failed", "PartsFailed"), ("timeout", "PartsTimeout"),
                    ("valid", "PartsValid"), ("genus_match", "GenusMatch"), ("genus_known", "GenusKnown"),
                    ("iou_occt_n", "IoUOcctN"), ("iou_sampled_n", "IoUSampledN")):
        m[name], m["Base" + name] = str(sn[k]), str(sb[k])
    m["IoUOcctMedian"], m["IoUOcctMin"] = _tex_num(sn["iou_occt_median"], 4, "f"), _tex_num(sn["iou_occt_min"], 4, "f")
    m["CPUSeconds"] = _tex_num(None if sn["cpu_seconds"] is None else round(sn["cpu_seconds"]))
    m["BaseCPUSeconds"] = _tex_num(None if sb["cpu_seconds"] is None else round(sb["cpu_seconds"]))
    m["WallSeconds"] = _tex_num(None if sn["wall_seconds"] is None else round(sn["wall_seconds"]))
    m["BaseWallSeconds"] = _tex_num(None if sb["wall_seconds"] is None else round(sb["wall_seconds"]))
    c = cmp["counts"]
    for v, name in (("better", "Better"), ("same", "Same"), ("worse", "Worse"), ("n/a", "NA"), ("missing", "Missing")):
        m[name] = str(c.get(v, 0))
        m["IoU" + name] = str(cmp["iou_counts"].get(v, 0))
    frame = [r for r in cmp["rows"] if FRAME_PATTERN in r["key"]]
    if frame:
        f = frame[0]
        m["FrameIoU"] = _tex_num(f["iou_new"], 4, "f")
        m["FrameIoUMethod"] = _tex_escape(f["iou_method_new"] or "n/a")
        m["FrameBaseIoU"] = _tex_num(f["iou_base"], 4, "f")
        m["FrameChamfer" + ("Mm" if cmp["chamfer_unit"] == "mm" else "Units")] = _tex_num(f["chamfer_new"], 3)
        m["FrameBaseChamfer" + ("Mm" if cmp["chamfer_unit"] == "mm" else "Units")] = _tex_num(f["chamfer_base"], 3)
        m["FrameOps"], m["FrameBaseOps"] = _tex_num(f["ops_new"]), _tex_num(f["ops_base"])
        m["FrameGenus"] = _tex_num(None if f["genus_new"] is None else int(round(f["genus_new"])))
        m["FrameInputGenus"] = _tex_num(None if f["input_genus"] is None else int(round(f["input_genus"])))
        m["FrameValid"] = _tex_num(f["valid_new"])
        m["FrameVerdict"] = _tex_escape(f["verdict"])
        m["FramePitchOverTau"] = _tex_num(f["pitch_over_tau_new"], 3)
    lines = [f"% Gate G3 macros: {new_label} against {base_label}. Written by scripts/g3_report.py - do not edit by hand.",
             f"% Chamfer in {cmp['chamfer_unit']}; verdict rule: {cmp['rule']}."]
    for k, v in m.items():
        assert re.fullmatch(r"[A-Za-z]+", prefix + k), k
        lines.append(f"\\newcommand{{\\{prefix}{k}}}{{{v}}}")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--base-label", default=None)
    ap.add_argument("--new-label", default=None)
    ap.add_argument("--chamfer-unit", choices=("mm", "raw"), default="mm")
    ap.add_argument("--rel", type=float, default=1.5)
    ap.add_argument("--abs", dest="abs_", type=float, default=0.05)
    ap.add_argument("--iou-tol", type=float, default=0.01)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--md", type=Path, default=None)
    ap.add_argument("--tex", type=Path, default=None)
    ap.add_argument("--prefix", default="Gthree")
    a = ap.parse_args(argv)
    bl, nl = a.base_label or a.base.name, a.new_label or a.new.name
    base, new = load_run(a.base), load_run(a.new)
    if not base or not new:
        print(f"no part records in {a.base if not base else a.new}", file=sys.stderr)
        return 2
    cmp = compare(base, new, a.chamfer_unit, a.rel, a.abs_, a.iou_tol)
    sb, sn = summary(base), summary(new)
    md = markdown(cmp, sb, sn, bl, nl)
    print(md)
    if a.md:
        a.md.write_text(md)
    if a.json:
        a.json.write_text(json.dumps(dict(base=dict(label=bl, dir=str(a.base), summary=sb),
                                          new=dict(label=nl, dir=str(a.new), summary=sn), **cmp), indent=1, default=str))
    if a.tex:
        a.tex.write_text(macros(cmp, sb, sn, bl, nl, a.prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
