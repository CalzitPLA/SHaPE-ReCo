#!/usr/bin/env python3
"""
The benchmark of the framework's own results (the user, 2026-10-11: "the software should not be linked to the server
results; benchmarking should be run with aegir on the results of the shape reco after the run of the framework; it
should enable the user to choose the most suitable result for his application").

The runs of one part are the stored runs of its input file, by content (reco_engine.run_store; the GUI saves every
finished run). Each is judged by what it measured against the input - one valid solid, exact IoU, voxel IoU, chamfer,
missing and invented volume, operations, seconds - and, where a known tree of the part exists, rated with the aegir
validation (reco_engine.validation.rate: tier A..E, failed criteria, separation, per-operation attribution), the rating
cached as rating.json in the run's directory. A known tree is <stem>.json beside the input, or truths/<stem>.json
beside it or one level up (the layout of the validation corpora). The rows are ranked for a stated preference
(PREFERENCES) and the user chooses: `choose` labels the run "chosen" in the store.

    from reco_engine import benchmark as bm
    b = bm.benchmark(input_path)                    # dict(truth=<path or None>, rows=[...]) - no rating computed
    b = bm.benchmark(input_path, rate=True)         # rates every run with a program against the known tree (cached)
    for row in bm.rank(b["rows"], "accuracy"): ...  # best first for the preference (also "event time")
    bm.choose(row["dir"])                           # -> the run's new directory, labelled "chosen"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

from reco_engine import run_store

RATING_JSON = "rating.json"

# preference -> (description, the sort key: larger is better, None last)
PREFERENCES: Dict[str, str] = {
    "accuracy": "the model closest to the input: one valid solid, then the exact IoU (else the voxel IoU), then fewer operations",
    "design": "the model closest to the known tree: the aegir tier, then its IoU against the tree, then fewer operations",
    "simplicity": "the fewest operations among the valid models with an IoU within 0.02 of the best, then the IoU",
    "speed": "the fastest run among the valid models with an IoU within 0.02 of the best, then the IoU",
    "event time": "the cheapest model to simulate among the valid ones with an IoU within 0.02 of the best: the navigation "
                  "time per event of the aegir scan (reco_engine/event_time.py; runs without one last), then the IoU",
}
NEPHOGRAM_STATS = ("nephogram", "stats.json")                  # reco_gui/nephogram.py stored_nephogram / the GUI's export


def read_nephogram(run_dir) -> Optional[dict]:
    """The stats of a run's nephogram (<run dir>/nephogram/stats.json), or None."""
    p = Path(run_dir).joinpath(*NEPHOGRAM_STATS)
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def truth_for(input_path) -> Optional[Path]:
    """The known tree of a part: <stem>.json beside the input, or truths/<stem>.json beside it or one level up."""
    p = Path(str(input_path))
    for cand in (p.with_suffix(".json"), p.parent / "truths" / f"{p.stem}.json", p.parent.parent / "truths" / f"{p.stem}.json"):
        if cand.is_file():
            try:
                d = json.loads(cand.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(d, dict) and ("program" in d or "type" in d):
                return cand
    return None


def result_of(run: dict):
    """The ReconstructionResult of a stored run (its program as run.json keeps it: the accepted operations in
    ReconstructionResult.to_dict's format); None when the program is not readable (cut to text, another method)."""
    from reco_engine.result_schema import ReconstructionResult
    ops = run.get("program") or []
    if not ops or run.get("program_truncated"):
        return None
    try:
        return ReconstructionResult.from_dict(dict(input_path=str((run.get("input") or {}).get("path") or ""), operations=ops))
    except (KeyError, TypeError, ValueError):
        return None


def read_rating(run_dir) -> Optional[dict]:
    p = Path(run_dir) / RATING_JSON
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def rate_stored(run: dict, truth_json, force: bool = False, **kw) -> Optional[dict]:
    """The aegir rating of a stored run against the known tree, cached as rating.json next to run.json; None when the
    run has no readable program. `kw` goes to reco_engine.validation.rate (cells, n_surface, seed)."""
    from reco_engine import validation
    from reco_engine.truth import truth_from_json
    run_dir = Path(run["path"])
    if not force:
        cached = read_rating(run_dir)
        if cached is not None:
            return cached
    result = result_of(run)
    if result is None:
        return None
    truth = truth_from_json(truth_json)
    if truth.input_path is None:
        truth.input_path = (run.get("input") or {}).get("path") or None
    m = run.get("metrics") or {}
    record = dict(model=dict(one_valid_solid=m.get("freecad_valid")),
                  comparison=dict(exact=dict(iou=m.get("exact_iou"))) if m.get("exact_iou") is not None else {})
    stl = (run.get("model") or {}).get("stl")
    rating = validation.rate(truth, result, None, record, model_mesh=run_dir / stl if stl else None, **kw)
    (run_dir / RATING_JSON).write_text(json.dumps(rating, indent=1, default=validation._json_default))
    return rating


def _text(x) -> str:
    """A grade's failed / limiting entry as text: a string, or validation.grade's dict(tier, check, value, op, threshold)."""
    if isinstance(x, dict):
        if "check" in x:
            return f"{x.get('tier')}: {x.get('check')} {x.get('value', float('nan')):.3g} {x.get('op', '')} {x.get('threshold', float('nan')):.3g}"
        return json.dumps(x, default=str)
    return str(x)


def row_of(run: dict, rating: Optional[dict] = None) -> dict:
    """One line of the benchmark: what the run measured, its path variant, and the rating where there is one."""
    from reco_engine import path_policy
    m, t = run.get("metrics") or {}, run.get("timings") or {}
    gui = (run.get("settings") or {}).get("gui") or {}
    g = (rating or {}).get("grade") or {}
    att = (rating or {}).get("attribution") or {}
    sep = att.get("separation") if isinstance(att.get("separation"), dict) else None
    from reco_engine.event_time import read_event_time
    neph = read_nephogram(run.get("path")) if run.get("path") else None
    ev = read_event_time(run.get("path")) if run.get("path") else None
    ns = (neph or {}).get("stats") or {}
    hist = Path(str(run.get("path"))).joinpath(NEPHOGRAM_STATS[0], "histogram.png") if neph else None
    return dict(
        dir=str(run.get("path")), name=run.get("name"), label=run.get("label"), saved=str(run.get("timestamp") or "")[:19].replace("T", " "),
        method=run.get("method"), variant=path_policy.variant_of(gui) if gui else "?", finished=bool(run.get("finished")),
        ops=int(m.get("n_operations") or len(run.get("program") or [])), one_valid_solid=m.get("freecad_valid"),
        exact_iou=m.get("exact_iou"), voxel_iou=m.get("voxel_iou"), chamfer_mm=m.get("chamfer_mean"),
        missing_volume=m.get("missing_volume"), invented_volume=m.get("invented_volume"), seconds=t.get("wall"),
        has_model=bool((run.get("model") or {}).get("stl")),
        rated=rating is not None, tier=g.get("tier"), failed=[_text(x) for x in (g.get("failed") or [])],
        limiting=[_text(x) for x in (g.get("limiting") or [])],
        separation=(sep or {}).get("cls"), rating_iou=(((rating or {}).get("vector") or {}).get("volume") or {}).get("iou"),
        commit=(run.get("code") or {}).get("short"),
        p95_missing=(ns.get("input_to_model") or {}).get("p95"), p95_invented=(ns.get("model_to_input") or {}).get("p95"),
        histogram=str(hist) if hist is not None and hist.is_file() else None, nephogram=bool(neph),
        event_us=(ev or {}).get("navigation_us"), event_min_us=(ev or {}).get("min_us"), event_max_us=(ev or {}).get("max_us"),
        event_gun=((ev or {}).get("placement") or {}).get("gun"))


def benchmark(input_path, store=None, rate: bool = False, progress: Optional[Callable[[str], None]] = None, **kw) -> dict:
    """dict(input, truth, rows): the stored runs of this input (by content, newest first), each as row_of; with `rate`
    every run with a program is rated against the known tree (cached ratings are read, not recomputed)."""
    ident = run_store.file_identity(input_path)
    runs = run_store.list_runs(store=store, sha256=ident["sha256"]) if ident["sha256"] else []
    truth = truth_for(input_path)
    rows = []
    for run in runs:
        rating = read_rating(run["path"])
        if rating is None and rate and truth is not None:
            if progress:
                progress(f"rating {run.get('name')} against {truth.name} ...")
            try:
                rating = rate_stored(run, truth, **kw)
            except Exception as exc:                                        # one run's rating must not end the benchmark
                if progress:
                    progress(f"rating of {run.get('name')} failed: {exc}")
                rating = None
        rows.append(row_of(run, rating))
    return dict(input=str(input_path), truth=str(truth) if truth else None, rows=rows)


def _iou(r: dict) -> Optional[float]:
    return r["exact_iou"] if r.get("exact_iou") is not None else r.get("voxel_iou")


def _valid(r: dict) -> int:
    return 1 if r.get("one_valid_solid") else 0


TIERS = ("A", "B", "C", "D", "E")


def rank(rows: List[dict], preference: str = "accuracy") -> List[dict]:
    """The rows best first for the preference (PREFERENCES); unfinished runs and runs without a number last."""
    if preference not in PREFERENCES:
        raise ValueError(f"unknown preference {preference!r}; one of {', '.join(PREFERENCES)}")
    finished = [r for r in rows if r.get("finished") and _iou(r) is not None]
    rest = [r for r in rows if r not in finished]
    best = max((_iou(r) for r in finished if _valid(r)), default=max((_iou(r) for r in finished), default=0.0))
    near = lambda r: 1 if _iou(r) >= best - 0.02 else 0                          # noqa: E731 - within 0.02 of the best

    def key(r):
        if preference == "accuracy":
            return (_valid(r), _iou(r), -r["ops"])
        if preference == "design":
            t = -TIERS.index(r["tier"]) if r.get("tier") in TIERS else -len(TIERS)
            return (t, r.get("rating_iou") or 0.0, _valid(r), -r["ops"])
        if preference == "simplicity":
            return (_valid(r), near(r), -r["ops"], _iou(r))
        if preference == "event time":
            ev = r.get("event_us")
            return (_valid(r), near(r), 0 if ev is None else 1, -(ev if ev is not None else 0.0), _iou(r))
        return (_valid(r), near(r), -(r.get("seconds") or float("inf")), _iou(r))
    return sorted(finished, key=key, reverse=True) + rest


def choose(run_dir, label: str = "chosen") -> Path:
    """Mark the run the user chose: its label in the store (the directory is renamed accordingly)."""
    return run_store.relabel(run_dir, label)
