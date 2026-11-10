#!/usr/bin/env python3
"""
Where the cascade (residual add-cut) loop spends its time, on one part.

    python3 scripts/time_cascade.py <stl|stp> [--resolution 96 --max-resolution 192] [--field-backend igl]
                                    [--features [--feature-time-budget 900 ...]] [--gui-fold] [--json out.json]

Runs reco_engine.residual_loop.residual_steps with the GUI's cascade settings under a reco_engine.timing.Recorder and
prints one line per step (its wall time and its biggest sections), then the run's sections aggregated (calls, wall,
share, self time, CPU, cores). --gui-fold also folds every step as the GUI does (reco_gui.cascade_session.CascadeFolder:
meshing the model, the residual, the tau-pitch voxels), timed as "gui fold". The JSON has the per-step and the total
sections, and the program's summary (operations, residual), so two runs can be compared.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "dataset_gen")]

from reco_engine.timing import Recorder, Section, cpu_seconds, format_sections, format_summary, timed  # noqa: E402



def _peak_rss_gb() -> float:
    """The process's peak resident memory (Linux: VmHWM), for the memory column of a run's summary."""
    import resource
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0, 2)

def _source(op) -> str:
    """Where an operation comes from: "feature" (the feature phase), "voxel_fallback", else "loop"."""
    modes = [s.mode for s in op.primitive.sources]
    if any(m.startswith("feature:") for m in modes):
        return "feature"
    return "voxel_fallback" if "voxel_fallback" in modes else "loop"


def _by_source(r) -> dict:
    """{source: {"add cylinder": n, ...}} of the accepted operations."""
    out = {}
    for o in (r.operations if r else []):
        if o.accepted:
            d = out.setdefault(_source(o), {})
            d[f"{o.op} {o.primitive.kind}"] = d.get(f"{o.op} {o.primitive.kind}", 0) + 1
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--resolution", type=int, default=96, help="the GUI's default start resolution")
    ap.add_argument("--max-resolution", type=int, default=192)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--field-backend", default="igl", help="igl (default, as the GUI) | cuda | torch | numpy | '' (the environment's)")
    ap.add_argument("--grid", default="cube", choices=("cube", "tau"), help="the loop's grid (ResidualConfig.grid)")
    ap.add_argument("--start", default="legacy",
                    help="starting shape: legacy (sub-loops first), auto (graded) or a type of reco_engine.start_shape.START_TYPES")
    ap.add_argument("--mode", default="loop", choices=("loop", "tree", "search"),
                    help="the rounds after the start: loop, or tree (the decision tree with children, residual_tree.py)")
    ap.add_argument("--order", default="default", choices=("default", "cuts-first", "adds-first"),
                    help="op order per round (ResidualConfig.cuts_first; default: cuts first after a starting shape)")
    ap.add_argument("--excess-boxes", type=int, default=None, help="largest filled boxes per cut piece (ResidualConfig.excess_boxes; 0 = off)")
    ap.add_argument("--no-vary-start", action="store_true", help="no coordinate search on the chosen starting shape")
    ap.add_argument("--variants", type=int, default=0,
                    help="after the run: k solutions from the k best starting shapes (reco_engine.residual_loop.reconstruct_variants), their statistics in the JSON")
    ap.add_argument("--max-ops", type=int, default=None, help="stop after this many operations (short runs)")
    ap.add_argument("--no-tools", action="store_true", help="no boolean results as tools")
    ap.add_argument("--no-vary", action="store_true", help="no varied views for a component that fails")
    ap.add_argument("--no-more-booleans", action="store_true", help="add and cut of single primitives only")
    ap.add_argument("--no-surface-choice", action="store_true", help="cuts ranked by voxel gain only (the old choice)")
    ap.add_argument("--features", action="store_true",
                    help="the feature phase after the voxel loop (ResidualConfig.feature_phase: exact difference, a "
                         "primitive per piece, validated in FreeCAD; replaces the voxel-cube fallback)")
    ap.add_argument("--feature-max-pieces", type=int, default=None, help="pieces per iteration")
    ap.add_argument("--feature-max-total-pieces", type=int, default=None, help="pieces in the whole phase")
    ap.add_argument("--feature-max-iterations", type=int, default=None)
    ap.add_argument("--feature-time-budget", type=float, default=None, help="s, wall clock of the phase")
    ap.add_argument("--feature-call-timeout", type=int, default=None, help="s, each FreeCAD call of the phase")
    ap.add_argument("--feature-min-share", type=float, default=None, help="smallest piece, share of the input volume")
    ap.add_argument("--feature-min-score", type=float, default=None, help="sampled volume IoU a guess needs")
    ap.add_argument("--feature-min-gain", type=float, default=None, help="residual drop needed, share of the piece")
    ap.add_argument("--feature-sources", default=None, help='comma separated, e.g. "step faces,fit" (default by input)')
    ap.add_argument("--no-feature-batch", action="store_true", help="one FreeCAD validation per guess")
    ap.add_argument("--tool-store", default=None,
                    help="with --features: the session's tool store (JSON), loaded before the phase and saved after it")
    ap.add_argument("--no-feature-types", action="store_true", help="no tool types (catalogue, session store, grouping)")
    ap.add_argument("--no-feature-split", action="store_true", help="no splitting of pieces no primitive explains")
    ap.add_argument("--no-feature-patterns", action="store_true", help="no reuse of a guess on congruent pieces")
    ap.add_argument("--no-feature-check", action="store_true", help="no exact FreeCAD check of the features")
    ap.add_argument("--no-feature-document", action="store_true", help="no features.FCStd")
    ap.add_argument("--feature-scoring", default=None, choices=("part", "iou"), help="how a whole piece's guesses are scored")
    ap.add_argument("--rough-cut-grade", type=float, default=None,
                    help="with --features: a later rough cut must gain this share of the first cut's gain (an add: of the first add's)")
    ap.add_argument("--rough-min-gain-share", type=float, default=None,
                    help="with --features: an operation of the voxel loop must gain this share of the input voxels")
    ap.add_argument("--rough-max-ops", type=int, default=None, help="with --features: at most this many loop operations")
    ap.add_argument("--gui-fold", action="store_true", help="also fold the steps as the GUI does (its meshing)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    from reco_engine.cascade_node_support import residual_config
    from reco_engine.residual_loop import residual_steps
    cfg = residual_config(top_k=args.top_k, resolution=args.resolution, max_resolution=args.max_resolution,
                          field_backend=args.field_backend or None, grid=args.grid,
                          surface_choice=not args.no_surface_choice, more_booleans=not args.no_more_booleans,
                          vary_on_failure=not args.no_vary, start_shape=args.start, mode=args.mode,
                          boolean_tools=not args.no_tools,
                          max_ops=args.max_ops,
                          **({"excess_boxes": args.excess_boxes} if args.excess_boxes is not None else {}),
                          start_vary=not args.no_vary_start,
                          cuts_first={"default": None, "cuts-first": True, "adds-first": False}[args.order],
                          feature_phase=args.features, feature_batch=not args.no_feature_batch,
                          feature_split=not args.no_feature_split, feature_patterns=not args.no_feature_patterns,
                          feature_types=not args.no_feature_types,
                          **({"feature_tool_store": args.tool_store} if args.tool_store else {}),
                          feature_check=not args.no_feature_check, feature_document=not args.no_feature_document,
                          **({"feature_scoring": args.feature_scoring} if args.feature_scoring else {}),
                          **{f"feature_{k}": v for k, v in (
                              ("max_pieces", args.feature_max_pieces), ("max_total_pieces", args.feature_max_total_pieces),
                              ("max_iterations", args.feature_max_iterations), ("time_budget", args.feature_time_budget),
                              ("call_timeout", args.feature_call_timeout), ("min_share", args.feature_min_share),
                              ("min_score", args.feature_min_score), ("min_gain", args.feature_min_gain),
                              ("sources", tuple(x.strip() for x in args.feature_sources.split(",") if x.strip())
                               if args.feature_sources else None)) if v is not None},
                          **{k: v for k, v in (("rough_min_gain_share", args.rough_min_gain_share),
                                               ("rough_cut_grade", args.rough_cut_grade),
                                               ("rough_max_ops", args.rough_max_ops)) if v is not None})
    fold = prev = None
    if args.gui_fold:
        from reco_gui.cascade_session import CascadeFolder
        fold = CascadeFolder()

    steps, last, feature_done = [], None, None
    with Recorder() as rec:
        it = iter(residual_steps(args.input, cfg))
        i = 0
        while True:
            w0, c0 = time.time(), cpu_seconds()
            with timed("loop step"):
                try:
                    st = next(it)
                except StopIteration:
                    break
            if fold is not None:
                with timed("gui fold"):
                    prev = fold(prev, st, i)
            sections = rec.take()
            wall, cpu = time.time() - w0, cpu_seconds() - c0
            inner = [Section(s.path[len("loop step/"):], s.wall, s.cpu, s.depth - 1) for s in sections
                     if s.path.startswith("loop step/")] + [s for s in sections if s.path == "gui fold"]
            print(f"{i + 1:3d} {st.kind:10s} {wall:7.2f} s (CPU {cpu:6.1f})  {format_sections(inner) if inner else ''}",
                  flush=True)
            if st.info.get("phase") == "feature":                          # the phase's verbose output
                print(f"      {st.info.get('message', '')}", flush=True)
                if st.kind == "feature_done":
                    feature_done = {k: v for k, v in st.info.items() if k not in ("phase",)}
            steps.append(dict(index=i, kind=st.kind, wall=wall, cpu=cpu, message=str(st.info.get("message", ""))[:300],
                              sections=[dict(path=s.path, wall=s.wall, cpu=s.cpu) for s in sections]))
            last = st
            i += 1
        elapsed = rec.elapsed()
        totals = rec.summary()

    # "loop step/x" -> "x": the wrapper is only there to separate the loop from the fold
    for t in totals:
        t.path = t.path.split("/", 1)[1] if t.path.startswith("loop step/") else t.path
    print()
    print(format_summary(totals, elapsed))
    r = last.result if last is not None else None
    summary = dict(input=str(args.input), start=args.start, order=args.order, grid=args.grid, surface_choice=not args.no_surface_choice, resolution=args.resolution, max_resolution=args.max_resolution,
                   field_backend=args.field_backend, gui_fold=args.gui_fold, elapsed=elapsed, peak_rss_gb=_peak_rss_gb(), fallback_boxes=(last.info.get('fallback_boxes') if last is not None else None), stats=(last.info.get('stats') if last is not None else None),
                   operations=len([o for o in r.operations if o.accepted]) if r else None,
                   residual=[r.residual_voxels_pos, r.residual_voxels_neg] if r else None,
                   chamfer=r.chamfer_mm if r else None,
                   grade=({k: v for k, v in (last.info.get("grade") or {}).items() if isinstance(v, (int, float, str, bool))}
                          if last is not None else None),
                   step_path=getattr(r, "step_path", None) if r else None,
                   features=feature_done,
                   exact_iou=getattr(r, "exact_iou", None) if r else None,
                   exact_method=getattr(r, "exact_method", None) if r else None,
                   operations_by_source=_by_source(r))
    program = [dict(op=o.op, kind=o.primitive.kind, label=o.primitive.label, source=_source(o), center=list(o.primitive.center_mm),
                    quaternion=list(o.primitive.quaternion), dims=dict(o.primitive.dims_mm or {}), net_gain=o.net_gain,
                    component_voxels=o.component_voxels)
               for o in (r.operations if r else []) if o.accepted]
    print("\n" + json.dumps(summary))
    if args.variants:
        from reco_engine.residual_loop import reconstruct_variants
        with timed("variants"):
            vs = reconstruct_variants(args.input, cfg, k=args.variants)
        summary["variants"] = [dict(start=v["start"], chosen=v["chosen"], stopped=v["stopped"], stats=v["stats"],
                                    program=[f"{o.op}:{o.primitive.kind}" for o in v["result"].operations if o.accepted]) for v in vs]
        for v in summary["variants"]:
            st = v["stats"] or {}
            print(f"variant {v['start']:18s} ops {len(v['program']):2d} IoU {st.get('iou', float('nan')):.4f} coverage {st.get('coverage', float('nan')):.3f} "
                  f"score {st.get('score', float('nan')):.3f} chamfer {st.get('chamfer')} valid {st.get('valid')}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(summary=summary, totals=[t.as_dict() for t in totals], steps=steps,
                                             program=program), indent=1))
        print(f"written {args.json}")


if __name__ == "__main__":
    main()
