#!/usr/bin/env python3
"""
D-003: the cascade add-cut loop in both orders (ResidualConfig.cuts_first True / False), everything else the same.

Each run is one child process (clean wall time and peak memory, no state shared between runs), one after the other.
Per run: accepted operations by kind (add / cut / ...), where they came from (loop, member, voxel fallback), the exact
grade (OCCT IoU against the input, missed r+ and invented r-, pieces of r+, genus of the result and of the input),
the chamfer, the voxel residual, recut voxels, which solution was kept (solution_choice: the classic alternative
always runs cuts first), the stop reason, wall and CPU seconds and peak memory.

    python3 scripts/d003_orders.py --parts frame_step box_with_bore ... --starts legacy --orders adds-first cuts-first \
        --repeats 1 --out DIR [--json docs/data/d003_orders_<date>.json]
    python3 scripts/d003_orders.py --one --part box_with_bore --start legacy --order cuts-first --row DIR/rows/x.json
    python3 scripts/d003_orders.py --collect DIR --json out.json      # merge the rows written so far

Settings: ResidualConfig defaults (cube grid 48 / 128, start "legacy" unless --starts says otherwise, feature phase off,
voxel fallback on, member_min_length_ratio 1.5, solution_alternatives "auto"), field_backend "igl", FreeCAD build and
exact grade on. A part name is a key of PARTS or a path; volumes are reported in mm^3 (`scale`: input unit -> mm).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

PARTS_DIR = Path(os.environ.get("D003_PARTS", "~/shape_reco_parts/benchmark_parts/1.0"))
EXAMPLES = ROOT / "dataset_gen" / "examples"
# name -> (path, input unit in mm)
PARTS = {
    "frame_step": (PARTS_DIR / "Frame_4_x_6.stp", 1.0),
    "frame_stl": (PARTS_DIR / "Unbenannt-Frame_4_x_6.stl", 100.0),    # the STL is 1/100 of the STEP (D-008)
    "liner_step": (Path(os.environ.get("D003_LINER", PARTS_DIR / "Liner_LINER_001-LINER_001.step")), 1.0),
}
ORDERS = {"adds-first": False, "cuts-first": True, "default": None}


def _part(name: str):
    if name in PARTS:
        return PARTS[name]
    p = EXAMPLES / f"{name}.stl"
    if p.is_file():
        return p, 1.0
    return Path(name), 1.0


def run_one(part: str, start: str, order: str) -> dict:
    from reco_engine.residual_loop import ResidualConfig, residual_steps
    path, scale = _part(part)
    t0, c0 = time.time(), time.process_time()
    last, solution, done = None, None, None
    with tempfile.TemporaryDirectory() as tmp:
        cfg = ResidualConfig(start_shape=start, cuts_first=ORDERS[order], field_backend="igl", freecad=True,
                             out_dir=Path(tmp) / "fc")
        for st in residual_steps(str(path), cfg):
            last = st
            if st.kind == "solution":
                solution = {k: st.info.get(k) for k in ("kept", "tried", "reasons", "solutions", "margin", "seconds")}
            if st.kind == "done":
                done = st.info
    wall, cpu = time.time() - t0, time.process_time() - c0
    res = last.result
    acc = [o for o in res.operations if o.accepted]
    kinds, sources = {}, {}
    for o in acc:
        kinds[o.op] = kinds.get(o.op, 0) + 1
        modes = [s.mode for s in o.primitive.sources]
        src = ("fallback" if "voxel_fallback" in modes else "member" if any("member" in m or "ibeam" in m for m in modes)
               else "loop")
        sources[f"{src} {o.op}"] = sources.get(f"{src} {o.op}", 0) + 1
    u3 = scale ** 3
    grade = (done or {}).get("grade") or {}
    stats = (done or {}).get("stats") or {}
    children = resource.getrusage(resource.RUSAGE_CHILDREN)            # FreeCAD subprocesses
    return dict(
        part=part, input=str(path), start=start, order=order, cuts_first=ORDERS[order],
        operations=len(acc), adds=kinds.get("add", 0), cuts=kinds.get("cut", 0), kinds=kinds, sources=sources,
        program=[f"{o.op}:{o.primitive.kind}" for o in acc],
        exact_iou=res.exact_iou, exact_method=res.exact_method,
        r_plus_mm3=None if res.exact_r_plus is None else res.exact_r_plus * u3,
        r_minus_mm3=None if res.exact_r_minus is None else res.exact_r_minus * u3,
        r_plus_solids=res.exact_r_plus_solids, genus=res.genus, input_genus=res.input_genus,
        chamfer_mm=None if res.chamfer_mm is None else res.chamfer_mm * scale,
        valid=grade.get("valid"), n_solids=grade.get("n_solids"),
        residual_voxels=[res.residual_voxels_pos, res.residual_voxels_neg], recut_voxels=res.recut_voxels,
        input_voxels=(done or {}).get("input_voxels"), fallback_boxes=(done or {}).get("fallback_boxes"),
        grid_pitch_over_tau=res.grid_pitch_over_tau, voxel_iou=stats.get("iou"),
        solution_first=(done or {}).get("solution"), solution=solution,
        stopped=(done or {}).get("stopped"),
        wall_s=round(wall, 2), cpu_s=round(cpu, 2), cpu_children_s=round(children.ru_utime + children.ru_stime, 2),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 2),
        host=os.uname().nodename, date=time.strftime("%Y-%m-%dT%H:%M:%S"))


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return None
    return x


def collect(out: Path) -> list:
    rows = []
    for f in sorted((out / "rows").glob("*.json")):
        rows.append(json.loads(f.read_text()))
    return rows


def _f(x, fmt="{:.4f}"):
    return "—" if x is None else fmt.format(x)


def table(rows: list) -> str:
    """Markdown: one line per part and start, both orders side by side; then the summary. The timing noise is read off
    the pairs whose two orders gave the SAME program (their wall-time difference can only be noise)."""
    by = {}
    for r in rows:
        by.setdefault((r["part"], r.get("start", "legacy")), {})[r.get("order")] = r
    out = ["| part | start | adds-first: IoU, add/cut, s | cuts-first: IoU, add/cut, s | same program | Δ IoU (cf − af) |",
           "|---|---|---|---|---|---|"]
    same_dt, diff, err = [], [], []
    for (part, start), d in by.items():
        a, c = d.get("adds-first"), d.get("cuts-first")
        if not a or not c or "error" in a or "error" in c:
            err.append(f"{part}/{start}: " + "; ".join(f"{k}: {v.get('error')}" for k, v in d.items() if "error" in v))
            continue
        same = a["program"] == c["program"]
        cell = lambda r: (f"{_f(r['exact_iou'])} ({'occt' if str(r['exact_method']).startswith('occt') else 'smp'}), "
                          f"{r['adds']}/{r['cuts']}, {r['wall_s']:.0f}")
        dio = None if a["exact_iou"] is None or c["exact_iou"] is None else c["exact_iou"] - a["exact_iou"]
        out.append(f"| {part} | {start} | {cell(a)} | {cell(c)} | {'yes' if same else 'NO'} | {_f(dio, '{:+.4f}')} |")
        if same:
            same_dt.append(abs(c["wall_s"] - a["wall_s"]) / max(1e-9, 0.5 * (a["wall_s"] + c["wall_s"])))
        else:
            diff.append((part, start, dio, c["wall_s"] - a["wall_s"]))
    import statistics as st
    out.append("")
    out.append(f"pairs: {len(by)}; same program: {len(same_dt)}; different: {len(diff)}; failed: {len(err)}")
    if same_dt:
        out.append(f"timing noise (same program, |Δt| / mean): median {st.median(same_dt):.1%}, max {max(same_dt):.1%}")
    for p, s, dio, dt in diff:
        out.append(f"different: {p}/{s}: Δ IoU {_f(dio, '{:+.4f}')}, Δ wall {dt:+.0f} s")
    out += [f"failed: {e}" for e in err]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--one", action="store_true")
    ap.add_argument("--part")
    ap.add_argument("--start", default="legacy")
    ap.add_argument("--order", choices=tuple(ORDERS), default="default")
    ap.add_argument("--row", type=Path)
    ap.add_argument("--parts", nargs="*", default=[])
    ap.add_argument("--starts", nargs="*", default=["legacy"])
    ap.add_argument("--orders", nargs="*", default=["adds-first", "cuts-first"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=4 * 3600, help="s per run")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--collect", type=Path, nargs="*", help="run directories: merge their rows (and print the table)")
    ap.add_argument("--table", type=Path, nargs="*", help="merged JSON files: print the table")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)

    if args.one:
        try:
            row = run_one(args.part, args.start, args.order)
        except Exception as e:                                          # recorded, the batch goes on
            import traceback
            row = dict(part=args.part, start=args.start, order=args.order, error=repr(e), traceback=traceback.format_exc())
        args.row.parent.mkdir(parents=True, exist_ok=True)
        args.row.write_text(json.dumps(_jsonable(row), indent=1))
        print(json.dumps({k: v for k, v in row.items() if k not in ("solution", "program", "traceback")}), flush=True)
        return

    if args.collect or args.table:
        rows = []
        for d in (args.collect or []):
            rows += collect(d)
        for f in (args.table or []):
            rows += json.loads(f.read_text())["rows"]
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(dict(rows=rows), indent=1))
        print(table(rows))
        return

    out = args.out
    (out / "rows").mkdir(parents=True, exist_ok=True)
    for rep in range(args.repeats):
        for part in args.parts:
            for start in args.starts:
                for order in args.orders:                               # orders interleaved, one run at a time
                    tag = f"{part}__{start}__{order}__r{rep}"
                    row = out / "rows" / f"{tag}.json"
                    if row.is_file():
                        continue                                        # resumable
                    t = time.time()
                    print(f"{time.strftime('%H:%M:%S')} start {tag}", flush=True)
                    try:
                        subprocess.run([sys.executable, str(Path(__file__).resolve()), "--one", "--part", part, "--start",
                                        start, "--order", order, "--row", str(row)],
                                       stdout=open(out / f"{tag}.log", "w"), stderr=subprocess.STDOUT,
                                       timeout=args.timeout, check=False)
                    except subprocess.TimeoutExpired:
                        row.write_text(json.dumps(dict(part=part, start=start, order=order, error=f"timeout {args.timeout} s")))
                    r = json.loads(row.read_text()) if row.is_file() else dict(error="no row")
                    r["repeat"] = rep
                    row.write_text(json.dumps(r, indent=1))
                    print(f"{time.strftime('%H:%M:%S')} done  {tag} {time.time() - t:.0f} s: "
                          f"{r.get('adds')} add {r.get('cuts')} cut, IoU {r.get('exact_iou')} ({r.get('exact_method')}), "
                          f"genus {r.get('genus')}/{r.get('input_genus')} {r.get('error', '')}", flush=True)
    if args.json:
        args.json.write_text(json.dumps(dict(rows=collect(out)), indent=1))


if __name__ == "__main__":
    main()
