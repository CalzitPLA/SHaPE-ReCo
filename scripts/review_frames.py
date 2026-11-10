#!/usr/bin/env python3
"""
Reconstruct the frame review set (dataset_gen/examples/frame_review, scripts/make_frame_review_set.py) and judge every
result: first each assembly as ONE body (its union), then as a MULTI-BODY compound.

    python3 scripts/review_frames.py union      [--only r02_rect_ipe ...] [--out ~/shape_reco_results/frame_review]
    python3 scripts/review_frames.py multibody
    python3 scripts/review_frames.py table      # the table of what has been run

union:      the loop (reco_engine.residual_loop, member_min_length_ratio 1.5, FreeCAD build, exact judge) on
            <name>_union.step.
multibody:  the compound <name>_multibody.step is exploded into its solids by FreeCAD; the loop runs on every body by
            itself (a single body is what the loop and the Type CNN are made for), the programs are concatenated, the
            whole program is built and judged against the UNION (the assembly is what has to be explained).
Every part runs in its own process under a timeout, one after the other, and waits while another heavy job of the
project runs (time_cascade, peel_case, pytest): the timings stay comparable. Per run: run.json (operations, grade,
exact IoU, genus, members found against the ground truth, seconds, peak RSS), model.step / model.stl, and a nephogram
of the surface difference (scripts/frame_report_nephogram.py).
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))
from src.freecad_path import freecadcmd_or_default  # noqa: E402  ($SHAPE_RECO_FREECADCMD, PATH, /usr/bin)
SET = ROOT / "dataset_gen" / "examples" / "frame_review"
HEAVY = "time_cascade|peel_case|pytest|s6_frame_eval"

EXPLODE = """
import json, os, Part
s = Part.Shape(); s.read(os.environ["STEP_IN"])
out = []
for i, b in enumerate(s.Solids):
    p = os.path.join(os.environ["OUT_DIR"], "body_%03d.step" % i)
    b.exportStep(p)
    out.append(dict(index=i, step=p, volume=b.Volume, valid=b.isValid(), faces=len(b.Faces)))
json.dump(out, open(os.path.join(os.environ["OUT_DIR"], "bodies.json"), "w"))
"""


def wait_for_quiet():
    while subprocess.run(["pgrep", "-f", HEAVY], capture_output=True).returncode == 0:
        time.sleep(20)


def member_lines(ops):
    """The members a program names: one line per group of boxes / cylinders that share an axis line."""
    return [dict(op=o["op"], kind=o["kind"], label=o.get("label"), size=sorted(round(2 * v, 1) for v in o["dims_mm"].values())) for o in ops]


def match_members(truth, ops):
    """How many members of the ground truth the program names: a member counts as found when the program has the
    operations its kind needs (I-beam: 3 boxes whose longest side is the member's length within 2 %, ...), each
    operation used once."""
    need = {"ibeam": [("add", "box")] * 3, "bar": [("add", "box")], "rod": [("add", "cylinder")], "tube": [("add", "cylinder"), ("cut", "cylinder")],
            "rhs": [("add", "box"), ("cut", "box")]}
    free = [dict(o, length=max(2 * v for v in o["dims_mm"].values())) for o in ops]
    found = []
    for i, m in enumerate(truth["members"]):
        L = float(np.linalg.norm(np.asarray(m["end"]) - np.asarray(m["start"])))
        mid = 0.5 * (np.asarray(m["end"]) + np.asarray(m["start"]))
        axis = (np.asarray(m["end"]) - np.asarray(m["start"])) / max(L, 1e-9)
        took = []
        for op, kind in need[m["kind"]]:
            best = None
            for k, o in enumerate(free):
                if k in took or o["op"] != op or o["kind"] != kind or abs(o["length"] - L) > max(0.03 * L, 45.0):
                    continue
                d = np.asarray(o["center_mm"]) - mid
                off = float(np.linalg.norm(d - axis * (d @ axis)))          # distance of the primitive's centre from the member's line
                if off <= 0.75 * max(m["section"].get("h", 0), m["section"].get("b", 0), m["section"].get("d", 0), m["section"].get("w", 0), 1.0) \
                        and (best is None or off < best[1]):
                    best = (k, off)
            if best is None:
                break
            took.append(best[0])
        if len(took) == len(need[m["kind"]]):
            found.append(i)
            free = [o for k, o in enumerate(free) if k not in took]
    return dict(members=len(truth["members"]), found=len(found), missing=[i for i in range(len(truth["members"])) if i not in found],
                operations=len(ops), unmatched_operations=len(free), expected_operations=truth["expected_program"]["total"])


def run_one(name, mode, out):
    """In a child process: reconstruct, judge, save."""
    from reco_engine.residual_loop import ResidualConfig, residual_steps
    from reco_engine.result_schema import ReconstructionResult
    truth = json.loads((SET / f"{name}.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    inputs = [SET / f"{name}_union.step"]
    if mode == "multibody":
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        (out / "bodies").mkdir(exist_ok=True)
        env.update(STEP_IN=str(SET / f"{name}_multibody.step"), OUT_DIR=str(out / "bodies"))
        (out / "explode.py").write_text(EXPLODE)
        subprocess.run([freecadcmd_or_default(), str(out / "explode.py")], env=env, capture_output=True, timeout=600)
        inputs = [Path(b["step"]) for b in json.loads((out / "bodies/bodies.json").read_text())]
    ops, steps, bodies, result = [], [], [], None
    for k, path in enumerate(inputs):
        cfg = ResidualConfig(member_min_length_ratio=1.5, out_dir=out / ("fc" if mode == "union" else f"fc_body_{k:03d}"),
                             freecad=mode == "union")
        tb = time.time()
        last = None
        for s in residual_steps(path, cfg):
            steps.append(dict(body=k, t=round(time.time() - t0, 2), step=s.kind, message=str(s.info.get("message", ""))[:240]))
            last = s
        accepted = [o for o in last.result.operations if o.accepted]
        bodies.append(dict(body=k, input=str(path), operations=len(accepted), seconds=time.time() - tb, stopped=last.info.get("stopped"),
                           grid_pitch_over_tau=getattr(last.result, "grid_pitch_over_tau", None)))
        if result is None:
            result = last.result
            grade = last.info.get("grade") or {}
        else:
            result.operations.extend(accepted)
        for o in accepted:
            p = o.primitive
            ops.append(dict(body=k, op=o.op, kind=p.kind, center_mm=[float(x) for x in p.center_mm], quaternion=[float(x) for x in p.quaternion],
                            dims_mm={a: float(b) for a, b in p.dims_mm.items()}, label=getattr(p, "label", None),
                            source=(p.sources[0].mode if p.sources else None)))
    union_in = SET / f"{name}_union.step"
    if mode == "multibody":                                        # build the whole program, judge it against the assembly
        from reco_engine.residual_loop import _load, freecad_grade
        from reco_engine.result_to_program import program_model
        from reco_field.exact_check import exact_residual
        from src.closest_point_field import SurfaceField
        result.operations = [o for o in result.operations if o.accepted]
        fld, _ = _load(union_in)
        grade = freecad_grade(program_model(result), out / "fc", name, fld)
        ex = exact_residual(union_in, grade["step"]) if grade.get("step") else dict(error="no build")
        grade.update(exact_iou=ex.get("iou"), exact_r_plus=ex.get("r_plus"), exact_r_minus=ex.get("r_minus"), exact_r_plus_solids=ex.get("r_plus_solids"),
                     exact_r_minus_solids=ex.get("r_minus_solids"), genus=ex.get("model_genus"), exact_error=ex.get("error"))
    seconds = time.time() - t0
    run = dict(name=name, mode=mode, description=truth["description"], input_volume=truth["union"]["volume_mm3"], input_faces=truth["union"]["faces"],
               diagonal_mm=truth["diagonal_mm"], bodies=bodies, operations=ops, match=match_members(truth, ops),
               grade={k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in grade.items()},
               seconds=seconds, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, steps=steps,
               code=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
               branch=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip(),
               time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "run.json").write_text(json.dumps(run, indent=1, default=str))
    if grade.get("stl"):
        subprocess.run([sys.executable, str(ROOT / "scripts/frame_report_nephogram.py"), str(union_in), grade["stl"], str(out / "nephogram.png"),
                        "--title", f"{name} ({mode}): model against the input"], capture_output=True, timeout=900)
    print(json.dumps({k: run[k] for k in ("name", "mode", "match", "seconds", "peak_rss_gb")}), flush=True)


def table(out):
    rows = []
    for mode in ("union", "multibody"):
        for d in sorted((out / mode).glob("*/run.json")):
            r = json.loads(d.read_text())
            g, m = r["grade"], r["match"]
            rows.append((r["name"], mode, m["members"], m["found"], m["operations"], m["expected_operations"], g.get("valid"), g.get("n_solids"),
                         g.get("exact_iou"), g.get("genus"), g.get("input_genus"), g.get("chamfer"), g.get("exact_r_plus"), g.get("exact_r_minus"),
                         r["seconds"], r["peak_rss_gb"], (r["bodies"][0].get("stopped") or "")[:40]))
    head = ("assembly", "mode", "members", "found", "ops", "expected", "valid", "solids", "exact IoU", "genus", "input genus", "chamfer mm",
            "missing mm3", "invented mm3", "seconds", "peak GB", "stopped")
    def f(x):
        return "-" if x is None else (f"{x:.4g}" if isinstance(x, float) else str(x))
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(f(x) for x in r) + " |" for r in rows]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    (out / "table.json").write_text(json.dumps([dict(zip(head, r)) for r in rows], indent=1))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["union", "multibody", "table", "_one"])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("SHAPE_RECO_RESULTS", Path.home() / "shape_reco_results")) / "frame_review")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--name")
    ap.add_argument("--one-mode")
    a = ap.parse_args()
    if a.mode == "_one":
        run_one(a.name, a.one_mode, a.out)
        return
    if a.mode == "table":
        table(a.out)
        return
    names = sorted(p.name[:-5] for p in SET.glob("r*.json"))
    for name in names:
        if a.only and name not in a.only:
            continue
        out = a.out / a.mode / name
        if (out / "run.json").exists():
            print(f"{name}: done, skipped", flush=True)
            continue
        wait_for_quiet()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        t = time.time()
        try:
            p = subprocess.run([sys.executable, __file__, "_one", "--name", name, "--one-mode", a.mode, "--out", str(out)], env=env,
                               capture_output=True, text=True, timeout=a.timeout)
            ok = p.returncode == 0 and (out / "run.json").exists()
            tail = (p.stdout.strip().splitlines() or [""])[-1] if ok else (p.stderr or p.stdout)[-700:]
        except subprocess.TimeoutExpired:
            ok, tail = False, f"timeout after {a.timeout} s"
        if not ok:
            out.mkdir(parents=True, exist_ok=True)
            (out / "failed.txt").write_text(tail)
        print(f"{time.strftime('%H:%M:%S')} {name} {a.mode}: {'ok' if ok else 'FAILED'} in {time.time() - t:.0f} s  {tail[-300:]}", flush=True)
    table(a.out)


if __name__ == "__main__":
    main()
