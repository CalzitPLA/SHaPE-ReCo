#!/usr/bin/env python3
"""
Everything the GUI needs to turn its settings into a running reconstruction: the RunSettings record, discovery of the
example and test parts on disk, and `make_session`, which returns a StepSession over the chosen mode's step generator:
the cascade add-cut loop (the main line, the default) or the vector-field peeling loop (the comparison route). The
voxel mode (route C: three networks, a voxelizer and the greedy loop) was removed on 2026-09-30.

Heavy imports (the engine) are deferred to the functions that need them, so the settings dialog and the tests do not
pay for them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional

ROOT = Path(__file__).resolve().parent.parent


def default_freecadcmd() -> str:
    """dataset_gen/src/freecad_path.py's rule ($SHAPE_RECO_FREECADCMD, then freecadcmd on PATH, then /usr/bin/freecadcmd)."""
    from reco_gui.dataset_gen_setup import default_freecadcmd as rule
    return rule()


@dataclass
class RunSettings:
    input_stl: Path                                           # an STL or a STEP file
    method: str = "cascade"                                   # "cascade" (residual add-cut loop, reco_engine.residual_loop)
                                                              # | "field" (closest-point vector field, reco_field)
    field_max_ops: int = 24                                   # field mode: operations at most (base included)
    field_target: float = 0.985                               # field mode: stop when this share of the surface is explained
    field_surface_points: int = 4000
    field_near_points: int = 2500
    field_vectors: int = 300                                  # field mode: vectors drawn
    field_deflection: float = 0.02                            # field mode: STEP tessellation deflection (the model's units)
    field_grade: bool = True                                  # field mode: chamfer of the result against the input as the last step
    field_bounding_box: bool = False                          # field mode: reco_field.peel's "rethought acceptance gate" (docs/FRAME_FAILURE.md)
    field_invented_slack: Optional[float] = None               # None: reco_field.peel.INVENTED_SLACK's own default
    field_cuts_first: Optional[str] = None                     # "on" | "off" | None (follow field_bounding_box) - tri-state for the UI
    field_max_candidate_invented: Optional[float] = None        # None: no per-candidate purity gate
    field_mesh_clusters: bool = False                          # field mode: cluster by full-mesh triangle connectivity, not point distance
    field_verify_freecad: bool = False                         # field mode: build every accepted step for real in FreeCAD (slow, diagnostic)
    field_octree: bool = False                                 # field mode: cluster by an occupancy octree over the missing volume
    cascade_registry: Optional[Path] = None                    # cascade mode: the stage registry (None: models/registry.json)
    cascade_top_k: int = 2                                     # cascade mode: branches kept per stage
    cascade_op_guard: bool = False                             # cascade mode: the per-operation guard (off = bypassed)
    cascade_max_ops: Optional[int] = None                      # cascade mode: None = the loop's own budget
    cascade_freecad_per_step: bool = False                     # cascade mode: build every step in FreeCAD (slow)
    cascade_freecad_final: bool = False                        # cascade mode: build + grade in FreeCAD at the end (slow; else the button)
    cascade_resolution: int = 96                               # cascade mode: voxels per side to start with
    cascade_max_resolution: int = 192                          # cascade mode: the loop refines up to this when thin walls vanish
    cascade_start_shape: str = "legacy"                        # cascade mode: the starting shape (start_shape.START_TYPES)
    cascade_first_guess: Optional[list] = None                 # cascade mode: the user's starting part (the window's step 2,
                                                               # reco_gui/start_part_dialog.py): ResidualConfig.first_guess
    cascade_mode: str = "loop"                                 # cascade mode: the rounds - "loop" | "tree" | "search" (ResidualConfig.mode)
    cascade_grid: str = "tau"                                  # cascade mode: "tau" = the finer of the wall pitch and the
                                                               # cube's (ResidualConfig.grid, B-090) | "cube"
    cascade_field_backend: str = "igl"                         # cascade mode: closest-point backend of T_in and the grade
                                                               # (igl: exact; 0.6 s instead of 313 s for the frame at 96^3)
    cascade_type_source: str = "registry"                      # cascade mode: type entries used - "registry" (all, fused) |
                                                               # "descriptors" (reco_engine/type_descriptors.py only) | "cnn" (the nets only)
    cascade_feature_phase: bool = True                         # cascade mode: the feature phase after the rough model (the exact
                                                               # difference in pieces, a guess per piece, FreeCAD-validated)
    cascade_feature_max_pieces: Optional[int] = None           # ... its limits; None = the loop's own default
    cascade_feature_max_total_pieces: Optional[int] = None
    cascade_feature_max_iterations: Optional[int] = None
    cascade_feature_time_budget_s: Optional[float] = None
    cascade_feature_min_score: Optional[float] = None
    cascade_feature_target_iou: Optional[float] = None         # the level of detail: the IoU the feature phase works up to (None: all)
    cascade_subparts: bool = False                             # sub-parts: the rough model as several added primitives, and
                                                               # pieces without a tool as sub-parts before the voxels
    cascade_feature_max_applied: Optional[int] = None          # features applied at most (None: the loop's 60)
    cascade_apply_at_once: bool = True                         # an accepted feature joins the model at once (the user, 2026-09-30)
    retry_iou: float = 0.95                                    # cascade mode: a finished run below this IoU asks whether to try the
                                                               # next path variant (reco_engine/path_policy.py); 0 = never ask
    retry_max: int = 3                                         # ... at most this many further paths per part
    alternatives: int = 3                                      # cascade mode: reconstructions per part, made one after the other
                                                               # after a run by the untried paths of the best measured rescue rate,
                                                               # then compared in the Benchmark tab (nephogram, histograms, aegir
                                                               # event time; the user, 2026-10-29); 0 = off: the decision bar asks
    memory_min_free_mb: float = 1200.0                         # the memory watcher: pause the run / stop the loop below
                                                               # this much available memory (MB; 0 = off)
    show_voxels: bool = False                                  # build the cube meshes of the voxel layers (input, residual,
                                                               # cut history, component, tau grid): off - no voxels on screen
                                                               # (the user, 2026-10-15), the field shows the input
    compare_tool: str = "freecad"                              # the comparison export (decision bar) opens in: freecad |
                                                               # cloudcompare | none (the user prefers FreeCAD, 2026-10-15)
    save_finished_run: bool = True                            # store every finished run (reco_engine/run_store.py)
    freecad: str = field(default_factory=default_freecadcmd)


# ------------------------------------------------------------------------------------------------- discovery

def discover_examples(root: Path = ROOT) -> List[Path]:
    return sorted((root / "dataset_gen" / "examples").glob("*.stl"))


# The part library of the window's part field (the user, 2026-10-29: "add the parts and benchmarks for later local or
# server use"): groups of parts, each part with its known tree where it has one (benchmark.truth_for finds
# truths/<stem>.json one level up, so the Benchmark tab rates them and a server run passes --truth). Outside the
# repository; a group whose directory is missing is left out.
RESULTS = Path(os.environ.get("SHAPE_RECO_RESULTS") or Path.home() / "shape_reco_results")
BENCH_PARTS = Path.home() / "shape_reco_parts/benchmark_parts"
LIBRARY = (("benchmark parts", [BENCH_PARTS / "1.0/Frame_4_x_6.stp",
                                BENCH_PARTS / "2.0/Liner_LINER_001-LINER_001.step"]),
           ("detail study", RESULTS / "detail" / "*" / "step" / "*.step"),
           ("known trees: gate (60, DEV)", [RESULTS / "validation" / "gate_paths" / "step" / "*.step",
                                            RESULTS / "validation" / "gate_paths" / "stl" / "*.stl"]),
           ("known trees: class_known (130, TEST)", RESULTS / "protocol" / "corpora" / "class_known_20261029" / "step" / "*.step"),
           ("known trees: machined (72, TEST)", RESULTS / "protocol" / "corpora" / "machined_20261029" / "step" / "*.step"))


def _glob(spec) -> List[Path]:
    parts = Path(spec).parts
    first = next((i for i, x in enumerate(parts) if "*" in x), len(parts))
    base = Path(*parts[:first])
    return sorted(base.glob(str(Path(*parts[first:])))) if base.is_dir() else []


def discover_library(library=LIBRARY) -> List[Tuple[str, List[Path]]]:
    """[(group, [paths])] of the part library that exist here, in the library's order. A group is a list of files or one
    or more globs; a part found as STEP and as STL is listed once, as STEP (the user's rule of 2026-10-17: STEP first)."""
    out = []
    for group, spec in library:
        specs = spec if isinstance(spec, (list, tuple)) else [spec]
        found = []
        for s in specs:
            found += [Path(s)] if "*" not in str(s) else _glob(s)
        found = [p for p in found if p.is_file()]
        steps = {p.stem for p in found if p.suffix.lower() in (".step", ".stp")}
        paths = [p for p in found if p.suffix.lower() in (".step", ".stp") or p.stem not in steps]
        if paths:
            out.append((group, paths))
    return out


# The downloaded test parts (outside the repo; their README.md names the sources): ABC, NIST PMI, Farnsworth House,
# FreeCAD's examples, the stability set. $SHAPE_RECO_TEST_PARTS names another directory.
TEST_PARTS = Path(os.environ.get("SHAPE_RECO_TEST_PARTS") or Path.home() / "shape_reco_parts/external_parts")


def discover_test_parts(root: Optional[Path] = None) -> List[Path]:
    """The STEP files below `root` (default TEST_PARTS), sorted. Where a directory has a manifest.csv with the columns
    `step` and `status` (the ABC subset), only its rows of status "ok" count: the single solids that tessellated."""
    import csv
    root = Path(TEST_PARTS if root is None else root)
    if not root.is_dir():
        return []
    listed, found = [], []
    for manifest in sorted(root.rglob("manifest.csv")):
        with open(manifest, newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows and {"step", "status"} <= set(rows[0]):
            listed.append(manifest.parent)
            found += [manifest.parent / r["step"] for r in rows if r["status"] == "ok" and r["step"]]
    for path in root.rglob("*"):
        if path.suffix.lower() in (".step", ".stp") and path.is_file() and not any(d in path.parents for d in listed):
            found.append(path)
    return sorted(p for p in set(found) if p.is_file())


# ------------------------------------------------------------------------------------------------- building

def make_session(s: RunSettings, models=None):
    """(StepSession, None, None) for the settings: the modes load what they need themselves (the cascade its networks from
    the stage registry). `models` is accepted for the worker's sake and not used."""
    if s.method == "field":
        from reco_gui.field_session import make_field_session
        cuts_first = {"on": True, "off": False}.get(s.field_cuts_first)
        return (make_field_session(s.input_stl, max_ops=s.field_max_ops, target=s.field_target, n_surface=s.field_surface_points,
                                   n_near=s.field_near_points, n_vectors=s.field_vectors, grade=s.field_grade,
                                   deflection=s.field_deflection, bounding_box=s.field_bounding_box,
                                   invented_slack=s.field_invented_slack, cuts_first=cuts_first,
                                   max_candidate_invented=s.field_max_candidate_invented,
                                   mesh_clusters=s.field_mesh_clusters, verify_freecad=s.field_verify_freecad,
                                   octree=s.field_octree), None, None)
    if s.method == "cascade":
        from reco_gui.cascade_session import make_cascade_session
        return (make_cascade_session(s.input_stl, top_k=s.cascade_top_k, op_guard=s.cascade_op_guard,
                                     max_ops=s.cascade_max_ops, freecad_per_step=s.cascade_freecad_per_step, freecad_final=s.cascade_freecad_final,
                                     registry_path=s.cascade_registry, resolution=s.cascade_resolution,
                                     max_resolution=s.cascade_max_resolution, type_source=s.cascade_type_source,
                                     field_backend=s.cascade_field_backend or None, grid=s.cascade_grid,
                                     start_shape=s.cascade_start_shape, mode=s.cascade_mode, feature_phase=s.cascade_feature_phase,
                                     first_guess=getattr(s, "cascade_first_guess", None),
                                     voxel_meshes=bool(getattr(s, "show_voxels", False)),
                                     memory_min_free_mb=float(getattr(s, "memory_min_free_mb", 1200.0)),
                                     feature_max_pieces=s.cascade_feature_max_pieces,
                                     feature_max_total_pieces=s.cascade_feature_max_total_pieces,
                                     feature_max_iterations=s.cascade_feature_max_iterations,
                                     feature_time_budget_s=s.cascade_feature_time_budget_s,
                                     feature_min_score=s.cascade_feature_min_score,
                                     feature_target_iou=s.cascade_feature_target_iou,
                                     rough_subparts=s.cascade_subparts or None,
                                     feature_subparts=s.cascade_subparts or None,
                                     feature_apply_at_once=s.cascade_apply_at_once,
                                     feature_max_applied=s.cascade_feature_max_applied), None, None)
    raise ValueError(f"method must be 'cascade' or 'field', got {s.method!r}")
