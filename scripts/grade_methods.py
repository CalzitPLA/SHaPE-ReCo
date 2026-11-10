#!/usr/bin/env python3
"""
Every method of the framework graded by its MEASURED effect on the reconstruction results (the user, 2026-10-30:
"grade all methods based on their impact on the reconstruction results, then decide to keep or drop"). A method is a
path the window offers (path_policy.VARIANTS), a loop setting with an on/off switch, or a stage. Each is compared with
the bare loop of ITS OWN batch - the same parts, the same checkout, the same day (and host where named) - through the
protocol's row reader (scripts/sci_test.run_row: good = one valid solid and IoU >= 0.95, exact before sampled).

Grades (written before the script's first run):
    A  confirmed     accepted by a pre-registered protocol on TEST (docs/data/protocol/<name>/result_test.json)
    B  better        paired: rescued - broken >= 2 and broken <= rescued / 2, or the paired IoU higher (Wilcoxon p < 0.05)
    C  neutral       neither better nor worse by these rules
    D  worse         paired: broken - rescued >= 2, or the paired IoU lower (Wilcoxon p < 0.05), or a protocol on TEST
                     rejected it with a failed harm check (one valid solid lower, an anchor below its band)
    E  unmeasured    no paired runs
Added after the first readout (2026-10-30, named here because it was not fixed before): the protocol's own harm check
(R7, "one valid solid not lower") applies on DEV too - a method with 3 or more fewer one-valid-solid results than its
bare arm (5 % of the gate) is D, whatever its IoU (tree_l had been B by its IoU while losing 14 valid solids); and a
loop SETTING graded C that costs no time is kept, not dropped (the C rule below was written for paths, an alternative
that changes nothing costs a whole run; a setting costs nothing).
A class where the method's paired median IoU change is >= +0.10 over >= 4 parts is named (a targeted strength).
Decisions: A -> keep (default); B -> keep (alternative, candidate for a protocol); C -> drop from what the window
offers (an alternative that changes nothing costs a full run) unless it has a targeted strength; a stage or setting
graded C stays (it takes no slot of the window), with its evidence from outside the gate where there is some
(KEEP_NOTES); D -> drop (off, not offered); E -> measure.

    python3 scripts/grade_methods.py              # -> docs/data/method_grades.{json,md}
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import sci_test  # noqa: E402
from reco_engine import stats_tests as st  # noqa: E402

H = Path.home() / "shape_reco_results"
GATE = H / "validation" / "gate_paths"
PROTO = ROOT / "docs" / "data" / "protocol"

# (method, kind, what it is, A glob, B glob, evidence note); A is the bare loop of the same batch
PAIRED = [
    ("sub-parts", "path", "the rough model as several sub-parts", GATE / "runs_bare", GATE / "runs_sub-parts",
     "gate, 2026-10-10, 73eae617, both server_a"),
    ("cubes", "path", "the voxel-cube cover first", GATE / "runs_bare", GATE / "runs_cubes",
     "gate, 2026-10-10, 73eae617, both server_a"),
    ("auto", "path", "the graded starting shape", GATE / "runs_bare", GATE / "runs_auto",
     "gate, 2026-10-10, 73eae617, server_a / server_b"),
    ("descriptors", "path", "the type from the descriptor model alone", GATE / "runs_bare", GATE / "runs_descriptors",
     "gate, 2026-10-10, 73eae617, server_a / server_b"),
    ("analytic", "path", "the AI-free type stage", GATE / "runs_bare", GATE / "runs_analytic",
     "gate, 2026-10-10, server_a / server_d+server_a: 36 runs on a 3x slower host (time budgets confounded)"),
    ("cnn", "path", "the type from the network alone", GATE / "runs_bare", GATE / "runs_cnn",
     "gate, 2026-10-10, server_a / server_c: 51 runs on a 3x slower host (time budgets confounded)"),
    ("k5t", "type net", "the k5t type network in the registry", GATE / "runs_bare_b", GATE / "runs_k5t",
     "gate, 2026-10-15, daa4e9c4/eb152a8c, both server_b"),
    ("tree (auto start)", "path", "the decision tree after the graded start", GATE / "runs_bare_b", GATE / "runs_tree",
     "gate, 2026-10-15, daa4e9c4/eb152a8c, server_b / server_a"),
    ("tree_c", "path", "the decision tree, variant c", GATE / "runs_bare_b", GATE / "runs_tree_c",
     "gate, 2026-10-15, daa4e9c4, server_b / server_a"),
    ("tree_l (legacy start)", "path", "the decision tree after the legacy start", GATE / "runs_bare_g", GATE / "runs_tree_l",
     "gate, 2026-10-16, 26d0d758 / e371cbf5, server_a+server_b"),
    ("tree_lg (legacy start, exact gate)", "path", "the decision tree after the legacy start, exact one-solid gate",
     GATE / "runs_bare_g", GATE / "runs_tree_lg", "gate, 2026-10-16, 26d0d758, server_a+server_b"),
    ("rough guess + merged cuts + tidy", "setting", "feature_rough, cut_merge, cut_tidy",
     H / "protocol" / "rough_tidy_v2" / "runs_rt2_dev_off", H / "protocol" / "rough_tidy_v2" / "runs_rt2_dev_on",
     "protocol rough_tidy_v2 DEV, adc2d6c3, server_b, both at once"),
    ("feature_min_share 1e-3", "setting", "pieces under 0.1 % of the input dropped unless known",
     H / "protocol" / "focus_min_share_v1" / "runs_focus_dev_A", H / "protocol" / "focus_min_share_v1" / "runs_focus_dev_B",
     "protocol focus_min_share_v1 DEV, e6aa19ed, server_a, both at once"),
    ("hollow_cut_first", "setting", "the cavity cut first, its intruders added back",
     H / "protocol" / "hollow_cf" / "hcf_off_*", H / "protocol" / "hollow_cf" / "hcf_on_*",
     "5 gate parts that reach the hollow sub-loop, d8367ee6, server_c/server_d, both at once (exploratory)"),
    ("cut_grow", "setting", "every face of a box cut grows until material",
     H / "remote" / "bracket_T1_rough_*", H / "remote" / "bracket_T1_grow_*", "the bracket only (one part, server_b)"),
    ("cut_open_faces", "setting", "box-cut faces that see open space end past the grid",
     H / "remote" / "bracket_T1_rough_*", H / "remote" / "bracket_T1_open_*", "the bracket only (one part, server_a)"),
]
# the current batch (2026-10-30, checkout gui_18468486 with the descriptor model; scripts: server queue methods_20261030.sh):
# every path the window offers and the stage ablations on the gate, each against the bare control that ran at the same
# time on the same host. A stage is graded present against absent: A = the run without it, B = the bare loop.
MG = H / "methods_20261030"
CURRENT = [(n, "path", w, MG / f"runs_mg_bare_{h}", MG / f"runs_mg_{a}_{h}", f"gate, 2026-10-30, gui_18468486, {h}, at once")
           for n, a, w, h in (("sub-parts", "sub-parts", "the rough model as several sub-parts", "server_a"),
                              ("cubes", "cubes", "the voxel-cube cover first", "server_a"),
                              ("auto", "auto", "the graded starting shape", "server_a"),
                              ("descriptors", "descriptors", "the type from the descriptor model alone", "server_a2"),
                              ("cnn", "cnn", "the type from the network alone", "server_a2"),
                              ("tree (auto start)", "tree", "the decision tree after the graded start", "server_a2"),
                              ("analytic", "analytic", "the AI-free type stage", "server_a2"),
                              ("compartments", "compartments", "the mass compartments first", "server_b"))]
CURRENT += [(n, "stage", w, MG / f"runs_mg_{a}_server_b", MG / "runs_mg_bare_server_b",
             "gate, 2026-10-30, gui_18468486, server_b, at once (A = without the stage)")
            for n, a, w in (("feature phase", "no-features", "the pieces of the exact difference explained by tools"),
                            ("polish", "no-polish", "curved operations refitted to the input's vertices"),
                            ("member sub-loop", "no-members", "straight members as their primitives (SHAPE_RECO_IBEAM=0 off)"),
                            ("hollow sub-loop", "no-hollow", "a closed cavity: outer add + cavity cut (SHAPE_RECO_HOLLOW=0 off)"))]
# server_a ran in two rounds (2026-10-30: 8 arms at once drove its load to 198 on 32 cores; round 2 = server_a2 with its own
# bare control): the bare loop twice on one host and once on another - the noise between times and between hosts
REPLICATE = (MG / "runs_mg_bare_server_b", MG / "runs_mg_bare_server_a")
REPLICATE_TIME = (MG / "runs_mg_bare_server_a", MG / "runs_mg_bare_server_a2")

# protocols judged on TEST: (method as in PAIRED, protocol name)
PROTOCOLS = {"tree_lg (legacy start, exact gate)": "tree_vs_bare_v1", "rough guess + merged cuts + tidy": "rough_tidy_v2",
             "feature_min_share 1e-3": "focus_min_share_v1"}
# offered or switchable but never run against the bare loop
UNMEASURED = [
    ("compartments", "path", "the mass compartments first", "one ABC plate in the window (0.869 against bare 0.9966)"),
    ("feature phase", "stage", "the pieces of the exact difference explained by tools", "no ablation on a corpus"),
    ("member sub-loop", "stage", "straight members (I-beam, rod, bar, tube, rhs) as their primitives", "the frame (S6) only"),
    ("hollow sub-loop", "stage", "a closed cavity: outer shell add + cavity cut", "example parts only"),
    ("polish", "stage", "curved operations refitted to the input's vertices", "no ablation on a corpus"),
]


def group_of(part: str) -> str:
    m = re.match(r"aegir_n(\d+)_([A-Z])_", part)
    if m:
        return {"U": "union", "I": "intersection", "S": "subtraction", "P": "one primitive"}[m.group(2)]
    return "v5 single" if part.startswith("v5_") else "other"


def _arm(path: Path) -> dict:
    """part -> row; a bracket-style run directory (<run>/run/record.json) counts as the part 'run'."""
    rows = sci_test.load_arm(str(path) + "/*")
    if not rows:
        rows = sci_test.load_arm(str(path) + "/run")
    return rows


def paired(a_rows: dict, b_rows: dict) -> dict:
    parts = sorted(set(a_rows) & set(b_rows))
    ga = [a_rows[p]["good"] for p in parts]
    gb = [b_rows[p]["good"] for p in parts]
    d = st.discordant(ga, gb)
    ia = [a_rows[p].get("iou") or 0.0 for p in parts]
    ib = [b_rows[p].get("iou") or 0.0 for p in parts]
    diffs = [y - x for x, y in zip(ia, ib)]
    wil = st.wilcoxon_paired(ia, ib) if len(parts) >= 6 and any(abs(x) > 1e-9 for x in diffs) else dict(p=None, median_diff=None)
    sa = [a_rows[p].get("seconds") for p in parts]
    sb = [b_rows[p].get("seconds") for p in parts]
    ds = [y - x for x, y in zip(sa, sb) if x is not None and y is not None]
    groups = {}
    for p, x in zip(parts, diffs):
        groups.setdefault(group_of(p), []).append(x)
    strengths = {g: round(statistics.median(v), 3) for g, v in groups.items() if len(v) >= 4 and statistics.median(v) >= 0.10}
    weak = {g: round(statistics.median(v), 3) for g, v in groups.items() if len(v) >= 4 and statistics.median(v) <= -0.10}
    return dict(n=len(parts), good_a=sum(ga), good_b=sum(gb), rescued=d["rescues"], broken=d["breaks"],
                mcnemar_p=st.mcnemar_exact(d["breaks"], d["rescues"]),
                iou_median_diff=round(statistics.median(diffs), 4) if diffs else None,
                iou_mean_diff=round(sum(diffs) / len(diffs), 4) if diffs else None, wilcoxon_p=wil.get("p"),
                one_solid_a=sum(bool(a_rows[p].get("one_valid_solid")) for p in parts),
                one_solid_b=sum(bool(b_rows[p].get("one_valid_solid")) for p in parts),
                seconds_median_diff=round(statistics.median(ds), 1) if ds else None,
                strengths=strengths, weaknesses=weak)


def grade(m: dict, protocol: dict | None) -> str:
    if protocol is not None:
        v = protocol["verdict"]
        if v["decision"].startswith("accept"):
            return "A"
        failed = set(v.get("failed") or [])
        if failed & {"one_solid_not_lower", "anchors", "no_cell_worse"}:
            return "D"
    r, b, p, md = m["rescued"], m["broken"], m["wilcoxon_p"], m["iou_median_diff"] or 0.0
    mean = m["iou_mean_diff"] or 0.0
    if m["one_solid_a"] - m["one_solid_b"] >= 3:                     # the harm check (added after the first readout)
        return "D"
    if b - r >= 2 or (p is not None and p < 0.05 and mean < 0):
        return "D"
    if (r - b >= 2 and b <= r / 2) or (p is not None and p < 0.05 and mean > 0):
        return "B"
    if m["n"] < 6 and mean < -0.01:                                  # a few parts, all of the same sign: worse
        return "D"
    return "C"


DECISION = {"A": "keep (default)", "B": "keep (alternative; candidate for a protocol)", "D": "drop (off, not offered)",
            "E": "measure"}


KEEP_NOTES = {   # evidence outside the gate that a C on the gate does not overturn
    "member sub-loop": "keep: the gate has no frame; the frame's 0.9655 rests on it (S6, CLAUDE.md)",
    "polish": "keep: no time cost; it raised the tiers of the calib and eval runs (A 24 -> 56 of 74, RUNNING)",
}


def decide(g: str, m: dict | None, kind: str = "path", name: str = "") -> str:
    if g == "C" and name in KEEP_NOTES:
        return KEEP_NOTES[name]
    if g == "C" and kind in ("setting", "stage") and m and (m["seconds_median_diff"] or 0.0) <= 0:
        return "keep (no time cost; its effect is not shown)"
    if g == "C" and kind == "stage":
        return "keep (a stage, not an alternative: no slot of the window to free; its effect is not shown)"
    if g == "C":
        return "keep as a targeted alternative: " + ", ".join(m["strengths"]) if m and m["strengths"] else \
            "drop from what the window offers (no measured effect)"
    return DECISION[g]


def main():
    rows = []
    cur = [c for c in CURRENT if any(Path(x).is_dir() for x in [str(c[3]), str(c[4])]) and Path(c[4]).is_dir()]
    current_names = {c[0] for c in cur}
    for name, kind, what, a, b, note in cur + PAIRED:
        if (name, kind, what, a, b, note) in PAIRED and name in current_names:
            note = "earlier evidence (superseded by the current batch): " + note
        import glob
        a_dirs = [Path(x) for x in sorted(glob.glob(str(a)))]
        b_dirs = [Path(x) for x in sorted(glob.glob(str(b)))]
        ar, br = {}, {}
        for x in a_dirs:
            ar.update(_arm(x))
        for x in b_dirs:
            br.update(_arm(x))
        if not ar or not br:
            rows.append(dict(method=name, kind=kind, what=what, evidence=note + " (runs not found)", grade="E",
                             decision=DECISION["E"]))
            continue
        m = paired(ar, br)
        prot = None
        if name in PROTOCOLS and (PROTO / PROTOCOLS[name] / "result_test.json").is_file():
            prot = json.loads((PROTO / PROTOCOLS[name] / "result_test.json").read_text())
        g = grade(m, prot)
        dec = decide(g, m, kind, name) if not note.startswith("earlier evidence") else "see the current batch"
        row = dict(method=name, kind=kind, what=what, evidence=note, grade=g, decision=dec, dev=m)
        if prot is not None:
            c = prot["compare"]
            row["test"] = dict(protocol=PROTOCOLS[name], decision=prot["verdict"]["decision"], good_a=c["a"]["good"]["k"],
                               good_b=c["b"]["good"]["k"], n=c["n"], rescued=c["rescued"], broken=c["broken"],
                               mcnemar_p=c["mcnemar_p"], failed=prot["verdict"].get("failed"))
        rows.append(row)
    for name, kind, what, note in UNMEASURED:
        if name not in current_names:
            rows.append(dict(method=name, kind=kind, what=what, evidence=note, grade="E", decision=DECISION["E"]))
    rep = None
    if all(Path(x).is_dir() for x in REPLICATE):
        ra, rb = _arm(REPLICATE[0]), _arm(REPLICATE[1])
        rep = paired(ra, rb)
        rep = dict(n=rep["n"], flips=rep["rescued"] + rep["broken"], good_server_b=rep["good_a"], good_server_a=rep["good_b"],
                   iou_mean_diff=rep["iou_mean_diff"], one_solid=(rep["one_solid_a"], rep["one_solid_b"]))
    out = ROOT / "docs" / "data" / "method_grades.json"
    out.write_text(json.dumps(dict(schema="method_grades/1", rules=__doc__.split("Grades")[1].split("python3")[0].strip(),
                                   host_replicate=rep, methods=rows), indent=1) + "\n")
    L = ["# Methods graded by their effect on the reconstruction results", "",
         "Written by `scripts/grade_methods.py` (rules in its docstring). good = one valid solid and IoU >= 0.95.", "",
         "| method | kind | grade | DEV: good A -> B (n) | rescued / broken | IoU median / mean change (Wilcoxon p) | "
         "one solid A -> B | s change | TEST | decision | evidence |", "|" + "---|" * 11]
    for r in rows:
        m, t = r.get("dev"), r.get("test")
        if m:
            dev = f"{m['good_a']} -> {m['good_b']} ({m['n']})"
            rb = f"{m['rescued']} / {m['broken']}"
            wp = "-" if m["wilcoxon_p"] is None else f"{m['wilcoxon_p']:.3f}"
            iou = f"{m['iou_median_diff']:+.3f} / {m['iou_mean_diff']:+.3f} ({wp})"
            solid = f"{m['one_solid_a']} -> {m['one_solid_b']}"
            sec = "-" if m["seconds_median_diff"] is None else f"{m['seconds_median_diff']:+.0f}"
            if m["strengths"]:
                iou += " strong: " + ", ".join(f"{k} {v:+.2f}" for k, v in m["strengths"].items())
            if m["weaknesses"]:
                iou += " weak: " + ", ".join(f"{k} {v:+.2f}" for k, v in m["weaknesses"].items())
        else:
            dev = rb = iou = solid = sec = "-"
        test = "-" if not t else f"{t['protocol']}: {t['good_a']} -> {t['good_b']} of {t['n']}, {t['decision']}"
        L.append(f"| {r['method']} | {r['kind']} | **{r['grade']}** | {dev} | {rb} | {iou} | {solid} | {sec} | {test} | "
                 f"{r['decision']} | {r['evidence']} |")
    if rep:
        L += ["", f"Host noise (the bare loop on server_b and on server_a, same checkout, same time): {rep['flips']} of "
                  f"{rep['n']} parts flip between good and not good (good {rep['good_server_b']} / {rep['good_server_a']}), "
                  f"mean IoU change {rep['iou_mean_diff']:+.4f}, one valid solid {rep['one_solid'][0]} / {rep['one_solid'][1]}."]
    (ROOT / "docs" / "data" / "method_grades.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
