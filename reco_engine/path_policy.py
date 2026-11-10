#!/usr/bin/env python3
"""
Which path to try next (the user's rule of 2026-10-10): the first run takes the framework as it is ("bare"); when its
IoU is too low, the GUI asks whether to try another approach; the approaches are the loop's path variants - the start of
the rough model, the type source, the sub-part cover - and they are ranked by their MEASURED outcome, the share of good
runs each gave (docs/data/path_weights.json, written by scripts/path_weights.py from the recorded runs), never by
opinion. A variant without enough measured runs ranks last and says so.

    from reco_engine import path_policy as pp
    name = pp.variant_of(settings)                        # which variant a reco_gui.setup.Settings is ("custom" if none)
    iou = pp.iou_of_done(info)                            # the IoU a finished run is judged by (exact grade, else voxel IoU)
    for s in pp.rank(tried=["bare"]): ...                 # the untried variants, best measured first
    pp.apply(settings_dict, "cubes")                      # the variant's overrides (reco_gui.setup.Settings field names)
    pp.run_part_args("cubes")                             # the same as scripts/run_part.py options
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "docs" / "data" / "path_weights.json"
MIN_RUNS = 5                                    # a variant needs this many measured runs to be ranked by its share
EXACT_IOU = 0.999                               # a run at least this good leaves no alternative anything to improve
# the settings a path changes that act before the feature phase only: a path that changes nothing else and reaches the
# rough model of a tried path builds the same program from there (the alternatives stop it, reco_gui/app.py)
ROUGH_ONLY_FIELDS = frozenset({"cascade_start_shape", "cascade_mode"})

# The path variants: the user-facing label, the reco_gui.setup.Settings overrides and the scripts/run_part.py options.
# "bare" is the framework as it is: the legacy start (sub-loops first, graded first add), every live type entry fused,
# the rough model as one primitive.
VARIANTS: Dict[str, dict] = {
    "bare": dict(label="the framework as it is", overrides={}, args=[]),
    "sub-parts": dict(label="the rough model as several sub-parts", overrides=dict(cascade_subparts=True), args=["--subparts"]),
    "auto": dict(label="the graded starting shape (most surface on it, least excess)",
                       overrides=dict(cascade_start_shape="auto"), args=["--start", "auto"]),
    # 2026-10-30: the primitives the STEP part's own surfaces name (reco_engine/leaf_recognition.py); judged by protocol
    # leaves_v1
    "leaves": dict(label="the primitives the part's own surfaces name first (recognized leaves), then the loop",
                   overrides=dict(cascade_start_shape="leaves"), args=["--start", "leaves"]),
}
# Retired on 2026-10-30 by the method grading (scripts/grade_methods.py, docs/data/method_grades.md: every path against
# the bare loop of its own batch on the gate, current code): cubes (D: 1 rescued / 8 broken, 17 fewer valid solids),
# compartments (D: good 29 -> 1), tree (D: 3 fewer valid solids; rejected on TEST by tree_vs_bare_v1), descriptors, cnn
# and analytic (C: no measured effect; analytic +182 s per part). Known - stored runs keep their name, run_part takes
# their options - but neither ranked nor offered as an alternative.
RETIRED: Dict[str, dict] = {
    "cubes": dict(label="the voxel-cube cover first (inside the material, the rest added)",
                  overrides=dict(cascade_start_shape="cubes"), args=["--start", "cubes"]),
    "descriptors": dict(label="the type decided by the descriptor model alone",
                        overrides=dict(cascade_type_source="descriptors"), args=["--type-source", "descriptors"]),
    "cnn": dict(label="the type decided by the network alone", overrides=dict(cascade_type_source="cnn"),
                args=["--type-source", "cnn"]),
    "tree": dict(label="the decision tree: the graded starting shape, then every round a decision on the difference, "
                       "components no primitive explains as children (reco_engine/residual_tree.py)",
                 overrides=dict(cascade_start_shape="auto", cascade_mode="tree"), args=["--start", "auto", "--mode", "tree"]),
    "analytic": dict(label="the AI-free type stage (every kind scored by the voxel IoU of its own fit)",
                     overrides=dict(cascade_registry=str(REPO_ROOT / "models" / "registry_analytic.json")),
                     args=["--registry", "models/registry_analytic.json"]),
    "compartments": dict(label="the mass compartments first (the part's mass split into well-filled boxes, added), then "
                               "cut and added until the fits are good",
                         overrides=dict(cascade_start_shape="compartments"), args=["--start", "compartments"]),
}
KNOWN: Dict[str, dict] = {**VARIANTS, **RETIRED}
# the bare values of the path fields; cascade_registry None = models/registry.json
_FIELDS = dict(cascade_start_shape="legacy", cascade_type_source="registry", cascade_subparts=False, cascade_registry=None,
               cascade_mode="loop")


DEFAULT_REGISTRY = REPO_ROOT / "models" / "registry.json"


def _norm(k, v):
    """The registry field compared by path, the default registry (or no path) as None."""
    if k != "cascade_registry":
        return v
    if not v:
        return None
    try:
        p = Path(v).expanduser().resolve()
    except (OSError, ValueError):
        return str(v)
    return None if p == DEFAULT_REGISTRY.resolve() else str(p)


def variant_of(settings) -> str:
    """The variant name of a Settings object or dict (the three path fields), "custom" when it is none of them."""
    get = (lambda k: settings.get(k, _FIELDS[k])) if isinstance(settings, dict) else (lambda k: getattr(settings, k, _FIELDS[k]))
    values = {k: _norm(k, get(k)) for k in _FIELDS}
    for name, v in KNOWN.items():                               # a retired path's runs keep their name
        if values == {k: _norm(k, x) for k, x in {**_FIELDS, **v["overrides"]}.items()}:
            return name
    return "custom"


def rough_only(variant: str) -> bool:
    """True when the path changes only settings that act before the feature phase (ROUGH_ONLY_FIELDS); the bare path
    changes nothing. Type sources and registries also steer the feature phase's guesses, sub-parts its sub-part guesses."""
    return set(KNOWN[variant]["overrides"]) <= ROUGH_ONLY_FIELDS


def apply(settings: dict, variant: str) -> dict:
    """`settings` (a dict of Settings fields) with the variant's path fields: the bare values, then the overrides."""
    out = dict(settings)
    out.update(_FIELDS)
    out.update(KNOWN[variant]["overrides"])
    return out


def run_part_args(variant: str) -> List[str]:
    return list(KNOWN[variant]["args"])


def iou_of_done(info: Optional[dict]) -> Optional[float]:
    """The IoU a finished run is judged by, from the done step's info: the FreeCAD grade's exact IoU when the run was
    graded, else the loop's voxel IoU (stats["iou"]). None when neither is there."""
    if not info:
        return None
    grade = info.get("grade")
    if isinstance(grade, dict) and grade.get("iou") is not None:
        return float(grade["iou"])
    stats = info.get("stats")
    if isinstance(stats, dict) and stats.get("iou") is not None:
        return float(stats["iou"])
    return None


def load_weights(path: Optional[Path] = None) -> dict:
    """docs/data/path_weights.json: {"variants": {name: {"p_good", "n", "source", ...}}, ...}; {} without the file."""
    p = Path(path) if path else WEIGHTS_PATH
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def rank(tried: Iterable[str] = (), weights: Optional[dict] = None, min_runs: int = MIN_RUNS) -> List[dict]:
    """The variants not yet tried, the best measured one first: by the rescue rate (the share of the parts the bare run
    failed on that the variant got right, measured on the same parts) where it is known, else by the share of good
    runs; a variant with fewer than `min_runs` measured runs (rescue: failures) comes after every measured one, in the
    order of VARIANTS, marked measured=False."""
    w = (load_weights() if weights is None else weights).get("variants", {})
    tried = set(tried)
    rows = []
    for name, v in VARIANTS.items():
        if name in tried:
            continue
        m = w.get(name) or {}
        n = int(m.get("n") or 0)
        p = m.get("p_good")
        rescue, n_fail = m.get("p_rescue"), int(m.get("n_fail") or 0)
        by_rescue = rescue is not None and n_fail >= min_runs
        measured = by_rescue or (p is not None and n >= min_runs)
        rows.append(dict(variant=name, label=v["label"], p_good=float(p) if p is not None else None, n=n,
                         p_rescue=float(rescue) if rescue is not None else None, n_fail=n_fail, by_rescue=by_rescue,
                         p_break=m.get("p_break"), measured=measured, source=m.get("source"),
                         bare_on_same_set=m.get("bare_on_same_set")))
    order = list(VARIANTS)

    def weight(r):
        if not r["measured"]:
            return 0.0
        return r["p_rescue"] if r["by_rescue"] else r["p_good"]
    rows.sort(key=lambda r: (0 if r["measured"] else 1, 0 if r["by_rescue"] else 1, -weight(r), order.index(r["variant"])))
    return rows


def describe(row: dict) -> str:
    """One line for the dialog / the log: the variant, its label and its measured share."""
    if row.get("by_rescue"):
        risk = f", lost {row['p_break']:.2f} of the parts bare got right" if row.get("p_break") is not None else ""
        return (f"{row['variant']}: {row['label']} - rescued {row['p_rescue']:.2f} of {row['n_fail']} parts the bare run "
                f"failed on{risk}")
    if row["measured"]:
        same = f", the bare framework {row['bare_on_same_set']:.2f} on the same parts" if row.get("bare_on_same_set") is not None else ""
        return f"{row['variant']}: {row['label']} - good in {row['p_good']:.2f} of {row['n']} measured runs{same}"
    return f"{row['variant']}: {row['label']} - not measured enough ({row['n']} runs, {row['n_fail']} paired failures)"
