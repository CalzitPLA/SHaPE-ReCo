#!/usr/bin/env python3
"""
The vector-field peeling loop (reco_field.peel, route A) on one part, graded like scripts/run_part.py grades the cascade:
the program is built in FreeCAD, the exact difference and the sampled comparison are run, and the same record.json is
written - so the two routes can be compared on one part with one reader (e.g. the AI-free A/B report).

    python3 scripts/peel_part.py part.step --out <dir> [--name NAME] [--max-ops N] [--grade] [--stl mesh.stl]
                                [--points 2000000] [--exact-timeout 600] [--build-timeout 900]

Writes <dir>/record.json (settings.route = "vector_field"; run.explained = the share of the surface samples the model
explains, run.grade = the loop's own two-way chamfer with --grade), steps.jsonl (the stage log), program.json
(reco_field.export.program_to_dict), model.step / model.stl, exact.json, compare.json and compare.log.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "dataset_gen"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="STL or STEP")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--stl", default=None, help="the input's mesh for the sampled comparison (default: tessellated here)")
    ap.add_argument("--max-ops", type=int, default=None, help="default: predicted from the part's genus (reco_field.peel.predict_ops_budget)")
    ap.add_argument("--grade", action="store_true", help="the loop's own two-way chamfer at the end")
    ap.add_argument("--points", type=int, default=2_000_000)
    ap.add_argument("--exact-timeout", type=int, default=600)
    ap.add_argument("--build-timeout", type=int, default=900)
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.path).stem
    rec = dict(name=name, input=str(args.path), state="running", stage="start", error=None, seconds={},
               commit=(ROOT / ".commit").read_text().strip() if (ROOT / ".commit").is_file() else "unknown",
               host=os.uname().nodename, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               settings=dict(route="vector_field", max_ops=args.max_ops, grade=bool(args.grade), points=args.points))

    def save():
        (out / "record.json").write_text(json.dumps(rec, indent=1, default=str))

    def stage(label):
        rec["stage"] = label
        save()
        return time.time()

    t_all = time.time()
    try:
        from reco_field.peel import INVENTED_SLACK, peel_steps, predict_ops_budget
        from src.closest_point_field import load_field

        t = stage("field")
        field = load_field(args.path)
        genus = field.genus()
        budget = None if args.max_ops is not None else predict_ops_budget(field)
        rec["input_facts"] = dict(triangles=int(len(field.faces)), genus=None if genus != genus else float(genus),
                                  predicted_max_ops=None if budget is None else int(budget.max_ops),
                                  budget_reason=None if budget is None else str(budget.reason))
        rec["seconds"]["field"] = time.time() - t

        t = stage("reconstruction")
        last, kinds = None, {}
        with open(out / "steps.jsonl", "w") as f:
            for i, st in enumerate(peel_steps(field, max_ops=args.max_ops, grade=args.grade, invented_slack=INVENTED_SLACK)):
                kinds[st.stage] = kinds.get(st.stage, 0) + 1
                f.write(json.dumps(dict(index=i, kind=st.stage, phase=st.phase, iteration=st.iteration,
                                        seconds=round(time.time() - t, 2), explained=st.explained,
                                        message=str(st.message)[:600])) + "\n")
                f.flush()
                last = st
        rec["seconds"]["reconstruction"] = time.time() - t
        ops = list(last.ops) if last is not None else []
        rec["run"] = dict(ended=bool(last is not None and last.stage == "done"), last_step=getattr(last, "stage", None),
                          stopped=str(getattr(last, "message", ""))[:400], steps=kinds,
                          explained=float(last.explained) if last is not None else None,
                          grade=(last.grade if last is not None and last.grade else None),
                          program=[dict(op=o.op, kind=o.fit.kind, params={k: float(v) for k, v in o.fit.params.items()},
                                        explained_after=float(o.explained_after)) for o in ops])
        if last is None or last.model is None or not ops:
            rec["model"] = dict(operations=0)
            rec.update(state="ended", stage="no model")
            rec["seconds"]["all"] = time.time() - t_all
            save()
            return 0
        by_kind: dict = {}
        for o in ops:
            key = f"{o.op} {o.fit.kind}"
            by_kind[key] = by_kind.get(key, 0) + 1
        rec["model"] = dict(operations=len(ops), by_source={"peel": len(ops)}, by_kind=by_kind)
        from reco_field.export import program_to_dict
        (out / "program.json").write_text(json.dumps(program_to_dict(last.model), default=str))

        t = stage("build")
        from reco_field.export import build_in_freecad
        ex = build_in_freecad(last.model, out, "model", timeout=args.build_timeout)
        rec["model"].update(valid=bool(ex.valid), solids=int(ex.n_solids), faces=int(ex.n_faces), volume=float(ex.volume),
                            one_valid_solid=bool(ex.valid and ex.n_solids == 1))
        rec["seconds"]["build"] = time.time() - t

        t = stage("exact difference")
        from reco_field.exact_check import exact_residual
        e = exact_residual(args.path, str(ex.step), timeout=args.exact_timeout)
        (out / "exact.json").write_text(json.dumps({k: v for k, v in e.items() if k != "pieces"}, indent=1, default=str))
        rec["comparison"] = dict(exact=dict(iou=e.get("iou"), method=e.get("method"), missing=e.get("r_plus"),
                                            invented=e.get("r_minus"), error=(e.get("error") or "")[-300:] or None,
                                            model_genus=e.get("model_genus"), input_volume=e.get("input_volume")))
        rec["seconds"]["exact difference"] = time.time() - t

        t = stage("sampled comparison")
        stl = args.stl
        if not stl or not Path(stl).is_file():
            from src.closest_point_field import tessellate_step
            stl = str(tessellate_step(args.path)) if Path(args.path).suffix.lower() in (".step", ".stp") else str(args.path)
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_pair.py"), name, stl, str(ex.stl), str(out),
                               "--n", str(args.points), "--surface", "200000", "--normalise", "1000", "--budget", "4e7",
                               "--top", "20"], capture_output=True, text=True)
        (out / "compare.log").write_text(proc.stdout[-20000:] + proc.stderr[-5000:])
        if (out / "compare.json").is_file():
            c = json.loads((out / "compare.json").read_text())
            s = float(c.get("scale") or 1.0)
            sm = c["sampled"]
            surf = c.get("surface") or {}
            rec["comparison"]["sampled"] = dict(
                iou=sm["iou"], iou_se=sm["iou_se"], missing=sm["missing"]["volume"] / s ** 3,
                invented=sm["invented"]["volume"] / s ** 3,
                missing_share=sm["missing"]["points"] / max(1, sm["common"]["points"] + sm["missing"]["points"]),
                invented_share=sm["invented"]["points"] / max(1, sm["common"]["points"] + sm["missing"]["points"]),
                chamfer=(surf.get("chamfer") / s) if surf.get("chamfer") is not None else None,
                chamfer_over_diagonal=(surf.get("chamfer") / 1000.0) if surf.get("chamfer") is not None else None, scale=s)
        else:
            rec["comparison"]["sampled"] = dict(error=(proc.stderr or proc.stdout)[-300:])
        rec["seconds"]["sampled comparison"] = time.time() - t
        rec.update(state="ended", stage="done")
    except Exception as exc:
        tb = traceback.format_exc()
        rec.update(state="crashed", error=f"{type(exc).__name__}: {str(exc)[-400:]}",
                   traceback=tb if len(tb) <= 6000 else tb[:3000] + "\n...\n" + tb[-3000:])
    rec["seconds"]["all"] = time.time() - t_all
    try:
        import resource
        rec["peak_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 2)
    except Exception:
        pass
    save()
    print(json.dumps({k: rec.get(k) for k in ("name", "state", "stage", "error")}), flush=True)
    return 0 if rec["state"] == "ended" else 1


if __name__ == "__main__":
    sys.exit(main())
