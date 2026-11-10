#!/usr/bin/env python3
"""
One part of a test set, from its STEP file to its record (the ABC runs, 2026-09-29): the input's facts, the
reconstruction by the cascade add-cut loop with the feature phase, the model built in FreeCAD, and the comparison of
model and input - exact and sampled.

    python3 scripts/run_part.py <part.step> --out <directory> [--stl <part.stl>] [--feature-time-budget 900]
                                [--points 2000000] [--no-features] [--name NAME] [--truth <truth.json>]

Written into <directory>:
  record.json    the record, written again after every stage: a run that is ended from outside (the batch's timeout)
                 leaves the record of what it had, with `state` "running" and the stage it was in
  program.json   the operations in full (ReconstructionResult.to_dict)
  steps.jsonl    the loop's steps: kind, seconds, message
  path.jsonl     the run's path: every operation that entered or left the program at which step, and every candidate
                 with its primitive, gain and reason (reco_engine/run_path.py)
  rating.json    with --truth (a shape whose CSG tree is known, reco_engine/truth.py): the validation cycle's rating -
                 metric vector, tier, defects and the separation point (reco_engine/validation.py)
  model.step, model.stl, compare.json, regions.json, exact.json

record.json: dict(name, input, state "ended" | "crashed" | "running" | "killed", stage, error, seconds (per stage),
commit, host, settings, input_facts, run, model, comparison). "killed" (since 2026-10-15): SIGTERM from outside (the
batch's timeout) - the record names the stage it was in; before, such a run stayed "running" and could not be told
from one still going. The gate of 2026-10-10 lost 25 of its 36 timeouts in the sampled comparison, not in the loop:
hence --reconstruction-timeout (the loop ends with what it has, run.timed_out, and the partial model is still built
and measured) and --compare-timeout (the comparison answers an error, the run still ends). Lengths and volumes are the file's units (not checked: D-008); the
sampled comparison scales the part to a box diagonal of 1000 and gives its numbers back in the file's units.

The settings are the GUI's cascade defaults (reco_gui/setup.py): 96 voxels per side up to 192, the tau grid, the
starting shape "legacy", top-k 2, the feature phase on; the run itself ends without the FreeCAD build, which comes here.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

PRIMITIVE_SURFACES = ("plane", "cylinder", "cone", "sphere", "torus")


def commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if out:
            return out
    except OSError:
        pass
    for name in (".commit", "COMMIT"):                  # a git-archive checkout on /data (server/shape_atlas: COMMIT)
        f = ROOT / name
        if f.is_file():
            return f.read_text().strip()
    return "unknown"


def input_facts(step: str) -> dict:
    """Faces by surface type, the share of free-form faces, volume - from the exact faces (freecadcmd)."""
    from reco_field.exact_check import exact_faces
    r = exact_faces(step, timeout=600)
    if "error" in r:
        return dict(error=r["error"][-300:])
    faces = r.get("faces") or []
    kinds: dict = {}
    area: dict = {}
    for f in faces:
        kinds[f["type"]] = kinds.get(f["type"], 0) + 1
        area[f["type"]] = area.get(f["type"], 0.0) + float(f["area"])
    total = sum(area.values()) or 1.0
    free = sum(n for k, n in kinds.items() if k not in PRIMITIVE_SURFACES)
    return dict(faces=len(faces), faces_by_type=kinds, area_share_by_type={k: v / total for k, v in area.items()},
                free_form_faces=free, free_form_area_share=sum(v for k, v in area.items() if k not in PRIMITIVE_SURFACES) / total,
                in_scope=free == 0, volume=r.get("volume"))


def source_of(op) -> str:
    modes = [s.mode for s in op.primitive.sources]
    if any(m.startswith("feature:") for m in modes):
        return "fallback" if any("fallback" in m for m in modes) else "feature"
    if any(m.startswith("tree:") for m in modes):                    # a child of the decision tree (residual_tree.py)
        return "tree"
    if any(m.startswith("search:") for m in modes):                  # the dumb shape search (shape_search.py)
        return "search"
    return "voxel_fallback" if "voxel_fallback" in modes else "loop"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("step")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stl", default=None, help="the input's mesh for the sampled comparison (default: tessellated here)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resolution", type=int, default=96)
    ap.add_argument("--max-resolution", type=int, default=192)
    ap.add_argument("--no-features", action="store_true")
    ap.add_argument("--subparts", action="store_true", help="the rough model as several sub-parts (ResidualConfig.rough_subparts)")
    ap.add_argument("--feature-subparts", action="store_true",
                    help="pieces without a tool as several sub-parts before the voxels (ResidualConfig.feature_subparts)")
    ap.add_argument("--feature-time-budget", type=float, default=900.0)
    ap.add_argument("--no-rough-guess", action="store_true", help="ResidualConfig.feature_rough off (protocol rough_tidy_v1)")
    ap.add_argument("--no-cut-merge", action="store_true", help="ResidualConfig.cut_merge off")
    ap.add_argument("--no-cut-tidy", action="store_true", help="ResidualConfig.cut_tidy off")
    ap.add_argument("--no-hollow-cut-first", action="store_true", help="ResidualConfig.hollow_cut_first off (the default "
                                                                       "since 2026-10-29: measured worse)")
    ap.add_argument("--hollow-cut-first", action="store_true", help="ResidualConfig.hollow_cut_first on")
    ap.add_argument("--prune", action="store_true", help="ResidualConfig.prune_parts on: the parts the shape does not need "
                                                         "dropped at the end (one valid solid checked in FreeCAD)")
    ap.add_argument("--first-guess", default=None,
                    help="the user's starting part(s): a first_guess.json (reco_engine/user_guess.py; the window's step 2)")
    ap.add_argument("--feature-min-share", type=float, default=None,
                    help="pieces under this share of the input volume are dropped unless known (ResidualConfig."
                         "feature_min_share, default 1e-3 since 2026-10-30 by protocol focus_min_share_v1; before 1e-5)")
    ap.add_argument("--start", default="legacy",
                    help="the starting shape (reco_engine.start_shape.START_TYPES): legacy (sub-loops first), auto (graded), "
                         "cubes (the voxel-cube cover), or a named type")
    ap.add_argument("--mode", default="loop", choices=["loop", "tree", "search"],
                    help="the rounds after the start: loop (as before) or tree (the difference-driven decision tree with "
                         "children, reco_engine/residual_tree.py; meant with --start auto)")
    ap.add_argument("--registry", type=Path, default=None,
                    help="a registry file instead of models/registry.json (reco_engine.cascade_node_support.residual_config)")
    ap.add_argument("--type-source", default="registry", choices=["registry", "descriptors", "cnn"],
                    help="the type entries that vote: all (fused), the descriptor model only, or the networks only")
    ap.add_argument("--no-polish", action="store_true",
                    help="skip the refit of the curved primitives to the input's vertices (ResidualConfig.polish)")
    ap.add_argument("--points", type=int, default=2_000_000, help="points of the sampled comparison")
    ap.add_argument("--exact-timeout", type=int, default=600)
    ap.add_argument("--build-timeout", type=int, default=900)
    ap.add_argument("--reconstruction-timeout", type=int, default=0,
                    help="seconds for the loop (0: none); when it runs out the loop ends with the model it has (run.timed_out)")
    ap.add_argument("--compare-timeout", type=int, default=900, help="seconds for the sampled comparison (an error, not a kill)")
    ap.add_argument("--no-mesh-residual", action="store_true",
                    help="no mesh difference (manifold3d) when OCCT's exact difference fails (the record then keeps the OCCT error)")
    ap.add_argument("--truth", default=None, help="the shape's known CSG tree (reco_engine/truth.program_json): rate the run")
    args = ap.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or Path(args.step).stem
    rec = dict(name=name, input=str(args.step), state="running", stage="start", error=None, seconds={},
               commit=commit(), host=os.uname().nodename, started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               settings=dict(resolution=args.resolution, max_resolution=args.max_resolution, grid="tau", start=args.start,
                             top_k=2, features=not args.no_features, subparts=bool(args.subparts),
                             feature_subparts=bool(args.feature_subparts), feature_time_budget=args.feature_time_budget,
                             feature_min_share=args.feature_min_share, first_guess=args.first_guess,
                             rough_guess=not args.no_rough_guess, cut_merge=not args.no_cut_merge, cut_tidy=not args.no_cut_tidy,
                             hollow_cut_first=bool(args.hollow_cut_first and not args.no_hollow_cut_first),
                             prune=bool(args.prune),
                             polish=not args.no_polish, points=args.points, gpu=os.environ.get("SHAPE_RECO_TEST_GPU") == "1",
                             registry=str(args.registry) if args.registry else None, type_source=args.type_source,
                             mode=args.mode))

    def save():
        (out / "record.json").write_text(json.dumps(rec, indent=1, default=str))

    def stage(label):
        rec["stage"] = label
        save()
        return time.time()

    t_all = time.time()

    def on_term(signum, frame):
        """Ended from outside (the batch's timeout sends TERM): the record says so and names the stage, then the
        process leaves at once (os._exit: an exception raised here can be lost inside a C extension that mishandles
        pending signals - measured 2026-10-15, a run went on to its end after the record said "killed")."""
        rec.update(state="killed", error=f"killed by signal {signum} in stage {rec['stage']}")
        rec["seconds"]["all"] = time.time() - t_all
        save()
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)

    class StageTimeout(BaseException):       # not Exception: the loop's advisory `except Exception` blocks must not eat it
        pass

    def on_alarm(signum, frame):
        raise StageTimeout()

    def rate():
        """The validation cycle's rating against the known tree (--truth); a failure is recorded, never raised."""
        if not args.truth:
            return
        t0 = stage("rating")
        try:
            from reco_engine.validation import rate_run
            r = rate_run(out, args.truth)
            rec["rating"] = dict(tier=r["grade"]["tier"], failed=r["grade"]["failed"],
                                 separation=(r.get("attribution") or {}).get("separation"))
        except Exception as exc:
            rec["rating"] = dict(error=f"{type(exc).__name__}: {str(exc)[-300:]}")
        rec["seconds"]["rating"] = time.time() - t0

    try:
        t = stage("input facts")
        rec["input_facts"] = input_facts(args.step)
        rec["seconds"]["input facts"] = time.time() - t

        t = stage("reconstruction")
        from reco_engine.cascade_node_support import residual_config
        from reco_engine.residual_loop import residual_steps
        if args.registry and not args.registry.is_file():
            raise FileNotFoundError(f"--registry {args.registry}")
        cfg = residual_config(top_k=2, resolution=args.resolution, max_resolution=args.max_resolution,
                              registry_path=args.registry, type_source=args.type_source,
                              field_backend="igl", grid="tau", start_shape=args.start, mode=args.mode, freecad=False,
                              feature_phase=not args.no_features, feature_time_budget=args.feature_time_budget,
                              rough_subparts=bool(args.subparts), feature_subparts=bool(args.feature_subparts),
                              polish=not args.no_polish, out_dir=str(out / "loop"),          # the feature phase's files: not below the home directory
                              **({} if args.feature_min_share is None else dict(feature_min_share=args.feature_min_share)),
                              **({} if not args.first_guess else dict(first_guess=str(args.first_guess))),
                              **({"feature_rough": False} if args.no_rough_guess else {}),
                              **({"cut_merge": False} if args.no_cut_merge else {}),
                              **({"cut_tidy": False} if args.no_cut_tidy else {}),
                              **({"hollow_cut_first": False} if args.no_hollow_cut_first else {}),
                              **({"hollow_cut_first": True} if args.hollow_cut_first and not args.no_hollow_cut_first else {}),
                              **({"prune_parts": True} if args.prune else {}))
        last, kinds = None, {}
        from reco_engine.run_path import PathRecorder
        path = PathRecorder(out / "path.jsonl")
        timed_out = False
        if args.reconstruction_timeout > 0:
            signal.signal(signal.SIGALRM, on_alarm)
            signal.alarm(int(args.reconstruction_timeout))
        try:
            with open(out / "steps.jsonl", "w") as f:
                try:
                    for i, step in enumerate(residual_steps(args.step, cfg)):
                        kinds[step.kind] = kinds.get(step.kind, 0) + 1
                        f.write(json.dumps(dict(index=i, kind=step.kind, seconds=round(time.time() - t, 2),
                                                message=str(step.info.get("message", ""))[:600])) + "\n")
                        f.flush()
                        try:
                            path.feed(i, step, time.time() - t)
                        except Exception as exc:                              # the path is advisory
                            rec.setdefault("path_errors", []).append(f"{type(exc).__name__}: {exc}"[:200])
                        last = step
                except StageTimeout:
                    timed_out = True
                    f.write(json.dumps(dict(index=len(kinds), kind="timeout", seconds=round(time.time() - t, 2),
                                            message=f"reconstruction timeout {args.reconstruction_timeout} s")) + "\n")
        finally:
            signal.alarm(0)
            path.close()
        result = last.result if last is not None else None
        rec["seconds"]["reconstruction"] = time.time() - t
        info = dict(last.info) if last is not None else {}
        stats = info.get("stats") or {}
        rec["run"] = dict(ended=bool(last is not None and last.kind == "done"), last_step=getattr(last, "kind", None),
                          stopped=str(info.get("stopped") or info.get("message") or "")[:400], steps=kinds,
                          voxel_iou=stats.get("iou"), input_voxels=info.get("input_voxels"),
                          exact_gate=stats.get("exact_gate"),
                          grid_pitch_over_tau=getattr(result, "grid_pitch_over_tau", None), timed_out=timed_out)
        if timed_out:
            where = f"after step {last.kind}" if last is not None else "before the first step"
            rec["run"]["stopped"] = (f"reconstruction timeout ({args.reconstruction_timeout} s) {where}; "
                                     + rec["run"]["stopped"])[:400]
        if result is None or not [o for o in result.operations if o.accepted]:
            rec["model"] = dict(operations=0)
            rate()
            rec.update(state="ended", stage="no model")
            rec["seconds"]["all"] = time.time() - t_all
            save()
            return 0
        ops = [o for o in result.operations if o.accepted]
        by_source: dict = {}
        by_kind: dict = {}
        for o in ops:
            by_source[source_of(o)] = by_source.get(source_of(o), 0) + 1
            key = f"{o.op} {o.primitive.kind}"
            by_kind[key] = by_kind.get(key, 0) + 1
        rec["model"] = dict(operations=len(ops), by_source=by_source, by_kind=by_kind)
        (out / "program.json").write_text(json.dumps(result.to_dict(), default=str))

        t = stage("build")
        from reco_engine.result_to_program import program_model
        from reco_field.export import build_in_freecad
        ex = build_in_freecad(program_model(result), out, "model", timeout=args.build_timeout)
        rec["model"].update(valid=bool(ex.valid), solids=int(ex.n_solids), faces=int(ex.n_faces), volume=float(ex.volume),
                            one_valid_solid=bool(ex.valid and ex.n_solids == 1))
        rec["seconds"]["build"] = time.time() - t

        t = stage("exact difference")
        from reco_field.exact_check import exact_residual
        e = exact_residual(args.step, str(ex.step), timeout=args.exact_timeout)
        occt_error = None
        if "error" in e and not args.no_mesh_residual:               # the ladder's middle rung (2026-10-15)
            from reco_field.exact_check import mesh_residual
            occt_error = e["error"]
            mesh_in = args.stl if args.stl and Path(args.stl).is_file() else args.step
            e2 = mesh_residual(mesh_in, str(ex.stl))
            e = dict(e2, occt_error=occt_error) if "error" not in e2 else dict(e, error=f"{occt_error}; {e2['error']}")
        (out / "exact.json").write_text(json.dumps({k: v for k, v in e.items() if k != "pieces"}, indent=1, default=str))
        rec["comparison"] = dict(exact=dict(iou=e.get("iou"), method=e.get("method"), missing=e.get("r_plus"),
                                            invented=e.get("r_minus"), error=(e.get("error") or "")[-300:] or None,
                                            occt_error=(occt_error or "")[-300:] or None,
                                            model_genus=e.get("model_genus"), input_volume=e.get("input_volume")))
        rec["seconds"]["exact difference"] = time.time() - t

        t = stage("sampled comparison")
        stl = args.stl
        if not stl or not Path(stl).is_file():
            from src.closest_point_field import tessellate_step
            stl = str(tessellate_step(args.step))
        for f in ("compare.json", "regions.json"):
            (out / f).unlink(missing_ok=True)                      # a stale pair from an earlier run must not be read
        try:
            proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_pair.py"), name, stl, str(ex.stl), str(out),
                                   "--n", str(args.points), "--surface", "200000", "--normalise", "1000", "--budget", "4e7",
                                   "--top", "20"], capture_output=True, text=True, timeout=args.compare_timeout)
            (out / "compare.log").write_text(proc.stdout[-20000:] + proc.stderr[-5000:])
        except subprocess.TimeoutExpired as exc:
            proc = None
            (out / "compare.log").write_text(f"sampled comparison: no answer within {args.compare_timeout} s\n"
                                             + str(exc.stdout or "")[-5000:])
        # compare_pair.py writes compare.json after the IoU and the chamfer (seconds) and regions.json after the
        # region labelling (minutes on a big difference): a timeout loses the regions, not the IoU
        if proc is None and not (out / "compare.json").is_file():
            rec["comparison"]["sampled"] = dict(error=f"no answer within {args.compare_timeout} s")
        elif (out / "compare.json").is_file():
            c = json.loads((out / "compare.json").read_text())
            s = float(c.get("scale") or 1.0)
            sm = c["sampled"]
            surf = c.get("surface") or {}
            rec["comparison"]["sampled"] = dict(
                **({"regions_error": f"no answer within {args.compare_timeout} s"} if proc is None else {}),
                iou=sm["iou"], iou_se=sm["iou_se"], missing=sm["missing"]["volume"] / s ** 3,
                invented=sm["invented"]["volume"] / s ** 3, missing_share=sm["missing"]["points"] / max(1, sm["common"]["points"] + sm["missing"]["points"]),
                invented_share=sm["invented"]["points"] / max(1, sm["common"]["points"] + sm["missing"]["points"]),
                chamfer=(surf.get("chamfer") / s) if surf.get("chamfer") is not None else None,
                chamfer_over_diagonal=(surf.get("chamfer") / 1000.0) if surf.get("chamfer") is not None else None, scale=s)
        else:
            rec["comparison"]["sampled"] = dict(error=(proc.stderr or proc.stdout)[-300:])
        if (out / "regions.json").is_file():
            g = json.loads((out / "regions.json").read_text())
            s = float(rec["comparison"]["sampled"].get("scale") or 1.0)
            regions = {}
            for key in ("missing", "invented"):
                r = (g.get(key) or {}).get("opened") or g.get(key) or {}
                regions[key] = dict(regions=r.get("components"), thin_volume=(r.get("thin_volume") or 0.0) / s ** 3,
                                    largest=[dict(volume=x["volume"] / s ** 3, bbox_lo=[v / s for v in x["bbox_lo"]],
                                                  bbox_hi=[v / s for v in x["bbox_hi"]]) for x in (r.get("regions") or [])[:5]])
            rec["comparison"]["regions"] = regions
        rec["seconds"]["sampled comparison"] = time.time() - t
        rate()
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
    raise SystemExit(main())
