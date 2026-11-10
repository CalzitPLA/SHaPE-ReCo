#!/usr/bin/env python3
"""
The measured weight of every path variant (reco_engine/path_policy.py): from the recorded runs below
$SHAPE_RECO_RESULTS (default ~/shape_reco_results), the share of GOOD runs a variant gave - one valid solid and an IoU
(exact where the exact difference answered, else sampled) of at least 0.95 - with the number of runs and the source,
and, where a paired bare run of the same parts exists, the bare share, the RESCUE rate (the share of the parts the
bare run failed on that the variant got right - the weight of "try this path next") and the break rate (the share of
the parts the bare run got right that the variant lost). Written to docs/data/path_weights.json.

    python3 scripts/path_weights.py [--out docs/data/path_weights.json] [--line 0.95]

Sources (a missing one is skipped and said):
  bare         abc/abc_known/<part>/record.json (the known ABC subset, 2026-09-30), aifree_bench/ab_20261002/orig/<part>
  sub-parts    abc/abc_known_subparts/<part> (the same parts with --subparts)
  cubes        cubes_examples/<part>/cubes_k4 (2026-10-10; bare on the same parts: cubes_examples/<part>/fused_k4)
  descriptors, cnn, analytic   liner_compare/<variant> (one part: evidence, not a share)
  every variant  validation/gate_paths/runs_<variant>/<stem> (the gate corpus of 60 known trees, paired with runs_bare)
Also written: the ABC batch's share of good runs by the path TAKEN (abc/abc_paths/paths.jsonl: start of the rough
model), for the report - not a variant anyone can choose.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from pathlib import Path

RES = Path(os.environ.get("SHAPE_RECO_RESULTS", Path.home() / "shape_reco_results"))
OUT = Path(__file__).resolve().parents[1] / "docs" / "data" / "path_weights.json"


def record(d: Path):
    p = d / "record.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def iou(rec) -> float | None:
    c = (rec or {}).get("comparison") or {}
    e = (c.get("exact") or {}).get("iou")
    if e is not None:
        return float(e)
    s = (c.get("sampled") or {}).get("iou")
    return float(s) if s is not None else None


def good(rec, line: float) -> bool | None:
    """True / False for a run that ended, None for a run without a result (killed from outside, still running, or
    without an IoU): such a run is no evidence either way."""
    if not rec or rec.get("state") != "ended":
        return False if rec and rec.get("state") in ("crashed",) else None
    m = rec.get("model") or {}
    if not m.get("one_valid_solid"):
        return False
    x = iou(rec)
    return None if x is None else x >= line


def share(records, line):
    g = [good(r, line) for r in records]
    g = [x for x in g if x is not None]
    return (sum(g) / len(g) if g else None), len(g)


def parts_of(root: Path) -> dict:
    return {d.name: record(d) for d in sorted(root.iterdir()) if d.is_dir() and (d / "record.json").is_file()} if root.is_dir() else {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--line", type=float, default=0.95)
    a = ap.parse_args(argv)
    line = a.line
    out = dict(created=time.strftime("%Y-%m-%dT%H:%M:%S"), line=line, results_root=str(RES), variants={}, evidence={}, paths_taken={})

    def add(name, recs, source, bare_recs=None, evidence_only=False):
        """`evidence_only`: a measured set that is no path variant of reco_engine/path_policy.py (a type-net or a
        code-version comparison on the gate, e.g. runs_k5t, runs_bare_b): its paired shares go under "evidence" only."""
        p, n = share(recs, line)
        if n == 0:
            print(f"{name}: no runs under {source}")
            return
        entry = dict(p_good=round(p, 4), n=n, source=source)
        if bare_recs is not None:                               # paired with the bare run on the same parts, same order
            pb, nb = share(bare_recs, line)
            if nb:
                entry.update(bare_on_same_set=round(pb, 4), n_bare=nb)
            pairs = [(good(b, line), good(v, line)) for b, v in zip(bare_recs, recs)]
            fails = [gv for gb, gv in pairs if gb is False and gv is not None]       # the parts the bare run failed on
            oks = [gv for gb, gv in pairs if gb is True and gv is not None]           # ... and the ones it got right
            if fails:
                entry.update(p_rescue=round(sum(fails) / len(fails), 4), n_fail=len(fails))
            if oks:
                entry.update(p_break=round(1 - sum(oks) / len(oks), 4), n_ok=len(oks))
        prev = out["variants"].get(name)
        if evidence_only:
            out["evidence"].setdefault(name, []).append(entry)
        elif prev is None or n > prev["n"]:                    # the largest measured set names the variant's weight
            out["variants"][name] = entry
            if prev is not None:
                out["evidence"].setdefault(name, []).append(prev)
        else:
            out["evidence"].setdefault(name, []).append(entry)
        print(f"{name:12s} p_good {p:.3f} on {n} runs ({source})" + (f", bare {entry['bare_on_same_set']:.3f} on {entry['n_bare']}" if "bare_on_same_set" in entry else "")
              + (f"; rescued {entry['p_rescue']:.3f} of {entry['n_fail']} bare failures, broke {entry.get('p_break', 0):.3f} of {entry.get('n_ok', 0)}" if "p_rescue" in entry else ""))

    # the known ABC subset: baseline and --subparts, paired
    known, known_sub = parts_of(RES / "abc" / "abc_known"), parts_of(RES / "abc" / "abc_known_subparts")
    paired = [k for k in known if k in known_sub]
    if paired:
        add("bare", [known[k] for k in paired], "abc/abc_known (332 parts of ABC chunk 0000, 2026-09-30, commit ccd502a0)")
        add("sub-parts", [known_sub[k] for k in paired], "abc/abc_known_subparts (the same parts with --subparts)",
            bare_recs=[known[k] for k in paired])
    # the A/B bench: the original (bare) and the AI-free copy (analytic type stage)
    ab = RES / "aifree_bench" / "ab_20261002"
    orig, aifree = parts_of(ab / "orig"), parts_of(ab / "aifree")
    paired = [k for k in orig if k in aifree]
    if paired:
        add("bare", [orig[k] for k in paired], "aifree_bench/ab_20261002/orig (99 parts: thesis, examples, beams, frames, ABC pilot; 2026-10-02)")
        add("analytic", [aifree[k] for k in paired], "aifree_bench/ab_20261002/aifree (the same parts, the AI-free type stage)",
            bare_recs=[orig[k] for k in paired])
    # the voxel-cube cover on the example parts, paired with the bare run
    ce = RES / "cubes_examples"
    if ce.is_dir():
        pairs = [(record(d / "cubes_k4"), record(d / "fused_k4")) for d in sorted(ce.iterdir())
                 if (d / "cubes_k4" / "record.json").is_file() and (d / "fused_k4" / "record.json").is_file()]
        if pairs:
            add("cubes", [c for c, f in pairs], f"cubes_examples ({len(pairs)} example parts, 2026-10-10, --start cubes)",
                bare_recs=[f for c, f in pairs])
    # the gate corpus: every path variant on the same 60 known trees (validation/gate_paths/runs_<variant>/<stem>),
    # paired with the bare run; "good" stays one valid solid + IoU >= line, the tier (A/B) is kept beside it
    gate = RES / "validation" / "gate_paths"
    if gate.is_dir():
        import sys
        if str(Path(__file__).resolve().parents[1]) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from reco_engine.path_policy import VARIANTS
        bare_gate = parts_of(gate / "runs_bare")

        def dated(recs):
            days = sorted({str(r.get("started", ""))[:10] for r in recs if r and r.get("started")})
            return days[0] if len(days) == 1 else (f"{days[0]}..{days[-1]}" if days else "undated")
        for d in sorted(gate.glob("runs_*")):
            name = d.name[len("runs_"):]
            if name == "bare":
                continue
            recs = parts_of(d)
            paired = [k for k in recs if k in bare_gate]
            if paired:
                add(name, [recs[k] for k in paired], f"validation/gate_paths/runs_{name} ({len(paired)} known trees, {dated(recs.values())})",
                    bare_recs=[bare_gate[k] for k in paired], evidence_only=name not in VARIANTS)
                tiers = [((json.loads((d / k / "rating.json").read_text()).get("grade") or {}).get("tier")) for k in paired
                         if (d / k / "rating.json").is_file()]
                if tiers:
                    out["evidence"].setdefault(name, []).append(dict(gate_tiers=dict(collections.Counter(tiers)), n=len(tiers)))
        if bare_gate:
            add("bare", list(bare_gate.values()), f"validation/gate_paths/runs_bare ({len(bare_gate)} known trees, {dated(bare_gate.values())})")
    # the liner comparison: one part, every variant - evidence lines, not shares
    lc = RES / "liner_compare"
    if lc.is_dir():
        for d in sorted(lc.iterdir()):
            r = record(d)
            if r:
                out["evidence"].setdefault("liner", {})[d.name] = dict(iou=iou(r), good=good(r, line), state=r.get("state"),
                                                                        operations=(r.get("model") or {}).get("operations"))
    # the ABC batch: the share of good runs by the path taken (the start of the rough model)
    pj = RES / "abc" / "abc_paths" / "paths.jsonl"
    if pj.is_file():
        by = collections.defaultdict(list)
        for line_ in pj.read_text().splitlines():
            r = json.loads(line_)
            sig = dict(x.strip().split("=", 1) for x in r["signature"].split("|")) if r.get("signature") else {}
            g = r["state"] == "ended" and bool(r.get("one")) and r.get("iou") is not None and r["iou"] >= line
            by[sig.get("start") or "none"].append(g)
        out["paths_taken"] = {k: dict(p_good=round(sum(v) / len(v), 4), n=len(v)) for k, v in by.items()}
        out["paths_taken_source"] = "abc/abc_paths (6054 parts of ABC chunk 0000, 2026-09-30 to 10-06, commit 2c8cde5a, --subparts)"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
